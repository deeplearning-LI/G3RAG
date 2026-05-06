import numpy as np
from openai import OpenAI
from tqdm import tqdm


class RemoteEmbeddingFunction:
    """Remote embedding function using OpenAI-compatible API"""

    def __init__(self, api_url: str, api_key: str = None,
                 model_name: str = "models/BAAI/bge-m3",
                 batch_size: int = 16, norm: bool = False):
        self.client = OpenAI(api_key=api_key, base_url=api_url)
        self.model_name = model_name
        self.batch_size = batch_size
        self.norm = norm

    def encode(self, texts):
        """
        Basic encoding method for a list of texts.

        Args:
            texts: List of strings to encode

        Returns:
            numpy array of embeddings
        """
        texts = [t.replace("\n", " ") for t in texts]
        texts = [t if t != '' else ' ' for t in texts]

        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts,
            encoding_format="float"
        )
        results = np.array([v.embedding for v in response.data])
        return results

    def batch_encode(self, texts, batch_size: int = None):
        """
        Batch encoding method with progress bar and normalization support.

        Args:
            texts: List of strings to encode
            batch_size: Batch size for processing (overrides instance default)
            **kwargs: Additional parameters

        Returns:
            numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        effective_batch_size = batch_size if batch_size is not None else self.batch_size

        if len(texts) <= effective_batch_size:
            results = self.encode(texts)
        else:
            pbar = tqdm(total=len(texts), desc="Batch Encoding")
            results = []
            for i in range(0, len(texts), effective_batch_size):
                batch = texts[i:i + effective_batch_size]
                try:
                    batch_results = self.encode(batch)
                    results.append(batch_results)
                except Exception as e:
                    print(f"Error processing batch {i // effective_batch_size + 1}: {e}")
                    raise e
                pbar.update(len(batch))
            pbar.close()
            results = np.concatenate(results)

        if self.norm:
            results = (results.T / np.linalg.norm(results, axis=1)).T

        return results

    def __call__(self, texts):
        """Legacy call method - delegates to batch_encode"""
        if isinstance(texts, str):
            texts = [texts]
        return self.batch_encode(texts)

    def embed_query(self, query):
        """Embed a single query string"""
        return self.batch_encode([query])[0] if isinstance(query, str) else self.batch_encode(query)

    def embed_documents(self, documents):
        """Embed multiple documents using batch processing"""
        return self.batch_encode(documents)
