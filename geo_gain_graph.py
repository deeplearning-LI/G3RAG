# Geometric Gain Graph: Zero-Token Graph Construction for Multi-Hop RAG

import os
import hashlib
import time
import numpy as np
import igraph as ig
from dataclasses import dataclass, field
from typing import Dict, Any
from openai import OpenAI

from tools.embedding_function import RemoteEmbeddingFunction
from tools.rerank import Reranker


def min_max_normalize(scores):
    """Min-max normalize scores to [0, 1] range"""
    min_score = np.min(scores)
    max_score = np.max(scores)
    if max_score == min_score:
        return np.ones_like(scores)
    return (scores - min_score) / (max_score - min_score)


@dataclass
class DocumentNode:
    """Document node"""
    doc_id: str
    content: str
    embedding: np.ndarray = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class GeoGainGraph:
    """
    Graph Construction Algorithm (Asymmetric IG):
        1. Compute pairwise cosine similarity matrix among all document embeddings.
        2. Compute Local Density for each node: the mean similarity to all other nodes.
           Higher density means the node contains very general/common information.
        3. Normalize density to [0, 1], then apply sparsity excitation: exp(-beta * norm_density).
           Nodes with lower density (rare facts) receive stronger excitation.
        4. Compute base orthogonal gain: x * sqrt(1 - x^2) for each similarity value x.
        5. Build directed gain matrix: directed_gain[i][j] = base_gain[i][j] * sparsity_excitation[j].
           This means: the gain of flowing from node i to node j is modulated by j's rarity.
        6. Apply similarity threshold filtering (sim_min <= sim <= sim_max) to prune noisy edges.
        7. Build a directed igraph with the remaining edges and their gain weights.
        8. Row-normalize the directed gain matrix into a transition matrix for diffusion.

    Retrieval Algorithm (retrieve_exact_diffusion):
        1. Compute cosine similarity between query and all document embeddings.
        2. Select top-k candidates, optionally rerank via LLM to obtain pure seeds.
        3. Initialize a sparse seed vector p (uniform weight over seed nodes).
        4. Perform exactly 1-hop diffusion: y = p @ T, where T is the pre-computed transition matrix.
        5. Strict seed masking: set diffused scores of seed nodes to -inf to force retrieval of NEW evidence.
        6. Combine original seeds with top diffused nodes as final results.
    """

    def __init__(self,
                 emb_api_key: str,
                 embedding_model: str = "qwen-max",
                 embedding_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings",
                 emb_batch_size: int = 16,
                 workdir: str = "./DB/sim_ig_graph",
                 api_key: str = None,
                 base_url: str = "http://localhost:33333/v1",
                 model_name: str = "Llama-3.3-70B",
                 sim_min: float = 0.3,
                 sim_max: float = 0.9,
                 beta: float = 1.0,
                 logger=None):

        self.emb_api_key = emb_api_key
        self.embedding_model = embedding_model
        self.embedding_url = embedding_url
        self.emb_batch_size = emb_batch_size
        self.workdir = workdir
        self.api_key = api_key or emb_api_key
        self.base_url = base_url
        self.model_name = model_name
        self.sim_min = sim_min
        self.sim_max = sim_max
        self.beta = beta
        self.logger = logger

        os.makedirs(self.workdir, exist_ok=True)

        self.embed = RemoteEmbeddingFunction(
            api_url=embedding_url,
            api_key=emb_api_key,
            model_name=embedding_model
        )

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        self.reranker = Reranker(
            api_key=self.api_key,
            base_url=self.base_url,
            model_name=self.model_name,
            temperature=0.0,
            prompt_type="rerank_chunk"
        )

        # Storage
        self.documents: Dict[str, DocumentNode] = {}
        self.graph: ig.Graph = None

    def add_document(self, content: str, doc_id: str = None, metadata: Dict[str, Any] = None) -> str:
        """Add a document to the graph builder"""
        if doc_id is None:
            doc_id = hashlib.md5(content.encode("utf-8")).hexdigest()

        doc_node = DocumentNode(
            doc_id=doc_id,
            content=content,
            metadata=metadata or {}
        )
        self.documents[doc_id] = doc_node

        return doc_id

    def compute_embeddings(self):
        """Compute embeddings for all documents in batch"""
        if not self.documents:
            return

        doc_ids = list(self.documents.keys())
        doc_contents = [self.documents[did].content for did in doc_ids]
        self.logger.info(f"Computing embeddings for {len(doc_ids)} documents...")

        if hasattr(self.embed, 'batch_encode'):
            doc_embs = self.embed.batch_encode(doc_contents, batch_size=self.emb_batch_size)
        else:
            doc_embs = [self.embed.embed_query(c) for c in doc_contents]
            doc_embs = np.array(doc_embs)

        for did, emb in zip(doc_ids, doc_embs):
            self.documents[did].embedding = np.array(emb)

    def _compute_info_gain(self, similarity: float) -> float:
        """Compute information gain: x * sqrt(1 - x^2)"""
        if similarity < 0 or similarity > 1:
            return 0.0
        return similarity * np.sqrt(1 - similarity ** 2)

    def build_graph(self):
        """
        Build the document-document information gain graph using Asymmetric IG method.

        Steps:
            1. Compute pairwise cosine similarity matrix among all document embeddings.
            2. Compute Local Density (average similarity to all other nodes).
               Higher density means the node contains very general/common information.
            3. Normalize density to [0, 1], apply sparsity excitation: exp(-beta * norm_density).
            4. Compute base orthogonal gain: x * sqrt(1 - x^2).
            5. Build directed gain matrix: gain[i][j] = base_gain[i][j] * sparsity_excitation[j].
            6. Apply similarity threshold filtering (sim_min <= sim <= sim_max).
            7. Build directed igraph with remaining edges.
            8. Row-normalize the gain matrix into a transition matrix.
        """
        self.logger.info("Building asymmetric information gain graph...")
        doc_ids = list(self.documents.keys())
        doc_embs = np.array([self.documents[did].embedding for did in doc_ids])
        n = len(doc_ids)

        # Step 1: Compute similarity matrix
        norms = np.linalg.norm(doc_embs, axis=1, keepdims=True)
        norm_doc_embs = doc_embs / np.where(norms == 0, 1, norms)
        sim_matrix = np.dot(norm_doc_embs, norm_doc_embs.T)
        np.fill_diagonal(sim_matrix, 0.0)

        # Step 2: Compute Local Density (average similarity to all other nodes)
        # Higher density means the node contains very general/common information
        local_density = np.mean(np.maximum(0, sim_matrix), axis=1)

        # Normalize density to [0, 1]
        d_min, d_max = local_density.min(), local_density.max()
        norm_density = (local_density - d_min) / (d_max - d_min + 1e-9)

        # Step 3: Compute Asymmetric Gain Matrix
        # Base orthogonal gain: x * sqrt(1 - x^2)
        sim_clipped = np.clip(sim_matrix, 0.0, 1.0)
        base_gain = sim_clipped * np.sqrt(1.0 - sim_clipped**2)

        # Apply asymmetric sparsity excitation
        # Broadcasting: penalty applies to target node j
        sparsity_excitation = np.exp(-self.beta * norm_density)
        directed_gain_matrix = base_gain * sparsity_excitation

        # Step 4: Build Directed Graph
        self.graph = ig.Graph(n=n, directed=True)
        self.graph.vs["name"] = doc_ids

        edges = []
        weights = []
        for i in range(n):
            for j in range(n):
                if i != j and self.sim_min <= sim_matrix[i, j] <= self.sim_max:
                    gain = directed_gain_matrix[i, j]
                    if gain > 0:
                        edges.append((i, j))
                        weights.append(float(gain))

        if edges:
            self.graph.add_edges(edges)
            self.graph.es["weight"] = weights

        # Step 5: Row-normalize to transition matrix
        row_sums = directed_gain_matrix.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        self.transition_matrix = directed_gain_matrix / row_sums

        self.logger.info(f"Directed graph built: {n} nodes, {len(edges)} edges")

        # Pre-compute document embeddings matrix for efficient query
        self._precompute_doc_embeddings()

    def _precompute_transition_matrix(self):
        """Pre-compute row-normalized transition matrix for efficient diffusion (supports both directed and undirected graphs)"""
        n = self.graph.vcount()

        # Build transition matrix using original IG weights
        transition_matrix = np.zeros((n, n))
        for edge in self.graph.es:
            source_idx = edge.source
            target_idx = edge.target
            weight = edge["weight"]
            transition_matrix[source_idx, target_idx] = weight
            # Only add reverse if graph is undirected
            if not self.graph.is_directed():
                transition_matrix[target_idx, source_idx] = weight

        # Row-normalize to create probability transition matrix
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        self.transition_matrix = transition_matrix / row_sums

    def _precompute_doc_embeddings(self):
        """Pre-compute document embeddings matrix for efficient query processing"""
        doc_ids = self.graph.vs["name"]
        doc_embs = np.array([self.documents[doc_id].embedding for doc_id in doc_ids])

        # Store raw embeddings
        self.doc_embeddings = doc_embs

        # Compute and store normalized embeddings for cosine similarity
        norms = np.linalg.norm(doc_embs, axis=1, keepdims=True)
        self.doc_embeddings_normalized = doc_embs / np.where(norms == 0, 1, norms)

    def retrieve_exact_diffusion(self, query: str, seed_k: int = 3, qa_top_k: int = 5,
                                 retrieve_topk: int = 15, node_rerank: bool = True) -> Dict[str, Any]:
        """
        One-step Exact Diffusion with strict seed masking.

        Algorithm:
            1. Calculate cosine similarity between query and all documents.
            2. Retrieve top-k candidates and (optionally) rerank them via LLM to get highly pure seeds.
            3. Build 1st-order row-normalized transition matrix T.
            4. Perform exactly 1-hop diffusion (y = p @ T) from the pure seeds.
            5. Mask original seeds to force the retrieval of NEW indirect evidence.
            6. Combine the original pure seeds with the top diffused nodes.
        """
        if self.graph is None:
            raise ValueError("Graph not built. Call build_graph() first.")

        start_time = time.time()

        # Get document IDs and count
        doc_ids = self.graph.vs["name"]
        n = len(doc_ids)

        # Compute query embedding and similarities
        query_emb = np.array(self.embed.embed_query(query))
        # Use pre-computed normalized embeddings for cosine similarity
        norm_doc_embs = self.doc_embeddings_normalized
        query_norm = np.linalg.norm(query_emb)
        norm_query_emb = query_emb / query_norm if query_norm > 0 else query_emb

        # Query-document cosine similarities
        sim_scores = np.dot(norm_doc_embs, norm_query_emb)
        sim_scores = np.maximum(0, sim_scores)

        # Select top candidates for potential LLM reranking
        candidate_indices = np.argsort(sim_scores)[::-1][:retrieve_topk]
        candidate_docs = [self.documents[doc_ids[idx]] for idx in candidate_indices]

        # Determine actual seed count: at most half of qa_top_k, or seed_k if smaller
        actual_seed_count = seed_k

        # Acquire pure seeds
        seed_indices = []
        if node_rerank and len(candidate_docs) > actual_seed_count:
            # Use LLM reranker as a hard gate for pure seeds
            reranked_docs = self.reranker.rerank(query, candidate_docs, actual_seed_count)
            reranked_ids = [doc.doc_id for doc in reranked_docs]
            seed_indices = [doc_ids.index(doc_id) for doc_id in reranked_ids if doc_id in doc_ids]
            rerank_suffix = "_nr"
        else:
            # Fallback: strict similarity top-k
            seed_indices = candidate_indices[:actual_seed_count].tolist()
            rerank_suffix = ""

        # Use pre-computed transition matrix (cached during build_graph)
        transition_matrix = self.transition_matrix

        # Initialize sparse pure seed vector 'p'
        p_vector = np.zeros(n)
        for idx in seed_indices:
            p_vector[idx] = 1.0 / len(seed_indices)

        # Perform exact diffusion (information flowing outward from seeds)
        diffused_scores = np.dot(p_vector, transition_matrix)

        # Strict Seed Masking: Force the model to select NEW context, not the seeds again
        diffused_scores[seed_indices] = -np.inf

        # Find the top highly-gained indirect nodes to fill the remaining qa_top_k slots
        needed_count = qa_top_k - len(seed_indices)

        # Get valid (non -inf) diffused scores and sort them
        valid_mask = diffused_scores > -np.inf
        valid_indices = np.where(valid_mask)[0]
        valid_scores = diffused_scores[valid_mask]

        # Sort by score (descending) and take top needed_count
        if len(valid_scores) >= needed_count:
            top_indices = np.argsort(valid_scores)[-needed_count:][::-1]
            top_diffused_indices = valid_indices[top_indices].tolist()
        else:
            # Not enough valid diffused nodes: use all valid, then fill with top similar docs
            top_diffused_indices = valid_indices.tolist()
            remaining = needed_count - len(top_diffused_indices)
            if remaining > 0:
                # Fill with top similarity docs (excluding seeds)
                all_sim_indices = np.argsort(sim_scores)[::-1]
                for idx in all_sim_indices:
                    if idx not in seed_indices and idx not in top_diffused_indices:
                        top_diffused_indices.append(idx)
                        if len(top_diffused_indices) >= needed_count:
                            break

        final_indices = seed_indices + top_diffused_indices[:needed_count]

        # Build final result array
        result_docs = []
        for idx in final_indices:
            doc_id = doc_ids[idx]
            doc_node = self.documents[doc_id]
            # Score logic: seed gets priority (1.0), diffused node gets its structural gain score
            score = 1.0 if idx in seed_indices else diffused_scores[idx]
            result_docs.append({
                "doc_id": doc_id,
                "content": doc_node.content,
                "score": float(score),
                "metadata": doc_node.metadata
            })

        elapsed = time.time() - start_time
        algorithm_name = f"exact_diffusion_k{seed_k}{rerank_suffix}"

        return {
            "query": query,
            "results": result_docs,
            "total": len(result_docs),
            "algorithm": algorithm_name,
            "diffusion_time": elapsed
        }


if __name__ == "__main__":
    from tools.setup_logger import setup_logger
    import datetime

    # ============================================================
    # 1. Sample corpus (single entry for demonstration)
    # ============================================================
    # Each entry is a dict with "title" and "text" fields.
    corpus = [
        {
            "title": "Albert Einstein",
            "text": "Albert Einstein (14 March 1879 - 18 April 1955) was a German-born "
                    "theoretical physicist who is widely held to be one of the greatest "
                    "and most influential scientists of all time."
        },
        {
            "title": "Theory of Relativity",
            "text": "The theory of relativity usually encompasses two interrelated theories "
                    "by Albert Einstein: special relativity and general relativity. Special "
                    "relativity applies to all physical phenomena in the absence of gravity. "
                    "General relativity explains the law of gravitation and its relation to "
                    "other forces of nature."
        },
        {
            "title": "Nobel Prize in Physics",
            "text": "The Nobel Prize in Physics is awarded once a year by the Royal Swedish "
                    "Academy of Sciences. Albert Einstein received the 1921 Nobel Prize in "
                    "Physics for his discovery of the law of the photoelectric effect."
        },
        {
            "title": "E=mc2",
            "text": "Mass-energy equivalence is the relationship between mass and energy in "
                    "a system's rest frame, where the equivalence is expressed by the equation "
                    "E = mc2. The equation is attributed to Albert Einstein."
        },
        {
            "title": "Max Planck",
            "text": "Max Planck (23 April 1858 - 4 October 1947) was a German theoretical "
                    "physicist whose discovery of energy quanta won him the Nobel Prize in "
                    "Physics in 1918."
        }
    ]

    # ============================================================
    # 2. Build graph
    # ============================================================
    WORKDIR = "./DB/geo_gain_graph_demo"
    os.makedirs(WORKDIR, exist_ok=True)

    day = datetime.datetime.now().strftime("%Y-%m-%d")
    timestamp = datetime.datetime.now().strftime("%H-%M")
    logger = setup_logger(WORKDIR, day, timestamp)

    graph = GeoGainGraph(
        emb_api_key="your-api-key",       # replace with your API key
        embedding_model="qwen-max",        # replace with your embedding model
        embedding_url="http://localhost:2222/v1",  # replace with your embedding API URL
        workdir=WORKDIR,
        emb_batch_size=128,
        api_key="your-api-key",            # replace with your LLM API key
        base_url="http://localhost:33333/v1",       # replace with your LLM API URL
        model_name="Llama-3.3-70B",       # replace with your LLM model name
        sim_min=0.1,
        sim_max=0.7,
        beta=0.1,
        logger=logger
    )

    for doc in corpus:
        content = f"{doc['title']}\n{doc['text']}"
        doc_md5 = hashlib.md5(content.encode('utf-8')).hexdigest()[:12]
        doc_id = f"doc_{doc_md5}"
        graph.add_document(content=content, doc_id=doc_id, metadata={"title": doc["title"]})

    graph.compute_embeddings()
    graph.build_graph()
    graph.save()

    # ============================================================
    # 3. Single-sample inference
    # ============================================================
    query = "What did Einstein win the Nobel Prize for?"

    result = graph.retrieve_exact_diffusion(
        query=query,
        seed_k=2,
        qa_top_k=3,
        retrieve_topk=5,
        node_rerank=False
    )

    print(f"\nQuery: {query}")
    print(f"Algorithm: {result['algorithm']}")
    print(f"Diffusion time: {result['diffusion_time']:.4f}s")
    print(f"Total results: {result['total']}\n")

    for i, doc in enumerate(result["results"]):
        print(f"[{i + 1}] Score: {doc['score']:.4f} | Title: {doc['metadata'].get('title', 'N/A')}")
        print(f"    Content: {doc['content'][:120]}...")
        print()
