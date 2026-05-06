import json
import re
from typing import List, Dict, Any, Optional
from openai import OpenAI

from tools.prompts import PROMPTS


class Reranker:
    """Reranker class for reranking chunks using LLM API with configurable prompts"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        temperature: float = 0.1,
        prompt_type: str = "rerank_chunk"
    ):
        """
        Initialize the reranker.

        Args:
            api_key: OpenAI API key
            base_url: Base URL for the API
            model_name: Model name to use for reranking
            temperature: Temperature for generation
            prompt_type: Type of prompt to use from prompts.py (default: "rerank_chunk")
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.temperature = temperature
        self.prompt_type = prompt_type
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _get_prompt(self, prompt_type: Optional[str] = None) -> Dict[str, str]:
        """Get prompt template from prompts.py"""
        prompt_key = prompt_type if prompt_type else self.prompt_type
        if prompt_key not in PROMPTS:
            prompt_key = "rerank_chunk"
        return PROMPTS.get(prompt_key, PROMPTS["rerank_chunk"])

    def rerank(
        self,
        query: str,
        chunks: List[Any],
        top_k: int = 10,
        prompt_type: Optional[str] = None
    ) -> List[Any]:
        """
        Rerank chunks based on their relevance to the query.

        Args:
            query: The search query
            chunks: List of chunk objects to rerank
            top_k: Number of top chunks to return after reranking
            prompt_type: Override the default prompt type for this call

        Returns:
            List of reranked chunks (top_k most relevant)
        """
        if not chunks:
            return []

        if len(chunks) <= top_k:
            return chunks

        prompt_template = self._get_prompt(prompt_type)

        chunk_texts = []
        for i, chunk in enumerate(chunks):
            if hasattr(chunk, 'content'):
                content = chunk.content
            elif isinstance(chunk, dict) and 'content' in chunk:
                content = chunk['content']
            else:
                content = str(chunk)

            chunk_texts.append(f"=== Document {i + 1} ===\n{content}")

        documents_text = "\n\n".join(chunk_texts)

        user_prompt = prompt_template["user"].format(
            query=query,
            documents_text=documents_text,
            num_documents=len(chunks)
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": prompt_template["system"]},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"} if "json" in self.model_name.lower() else None
            )

            result_text = response.choices[0].message.content.strip()
            ranking = self._parse_ranking_result(result_text, chunks)

            reranked_chunks = [chunks[i] for i in ranking[:top_k]]
            return reranked_chunks

        except Exception as e:
            print(f"Reranking failed: {e}")
            return chunks[:top_k]

    def _parse_ranking_result(self, result_text: str, chunks: List[Any]) -> List[int]:
        """Parse the ranking result from LLM response"""
        try:
            if result_text.startswith('{'):
                result_json = json.loads(result_text)
                if 'ranking' in result_json:
                    ranking = result_json['ranking']
                elif 'documents' in result_json:
                    ranking = result_json['documents']
                elif 'order' in result_json:
                    ranking = result_json['order']
                elif 'selected' in result_json:
                    ranking = result_json['selected']
                elif isinstance(result_json, dict) and len(result_json) == 1:
                    ranking = list(result_json.values())[0]
                else:
                    ranking = list(range(1, len(chunks) + 1))
            elif result_text.startswith('['):
                ranking = json.loads(result_text)
            else:
                obj_match = re.search(r'\{[^}]*"ranking"\s*:\s*\[[^\]]*\][^}]*\}', result_text)
                if obj_match:
                    result_json = json.loads(obj_match.group())
                    ranking = result_json['ranking']
                elif '"selected"' in result_text:
                    obj_match = re.search(r'\{[^}]*"selected"\s*:\s*\[[^\]]*\][^}]*\}', result_text)
                    if obj_match:
                        result_json = json.loads(obj_match.group())
                        ranking = result_json['selected']
                    else:
                        ranking = list(range(1, len(chunks) + 1))
                else:
                    array_match = re.search(r'\[[\d,\s]+\]', result_text)
                    if array_match:
                        ranking = json.loads(array_match.group())
                    else:
                        ranking = list(range(1, len(chunks) + 1))

            valid_ranking = []
            for doc_num in ranking:
                if isinstance(doc_num, int) and 1 <= doc_num <= len(chunks):
                    valid_ranking.append(doc_num - 1)

            missing_indices = set(range(len(chunks))) - set(valid_ranking)
            valid_ranking.extend(list(missing_indices))

            return valid_ranking

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Failed to parse ranking result: {e}")
            return list(range(len(chunks)))


class PathReranker:
    """Path Reranker for selecting relevant paths from graph exploration"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        temperature: float = 0.0,
        prompt_type: str = "select_path"
    ):
        """
        Initialize the path reranker.

        Args:
            api_key: OpenAI API key
            base_url: Base URL for the API
            model_name: Model name to use for reranking
            temperature: Temperature for generation
            prompt_type: Type of prompt to use from prompts.py (default: "select_path")
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.temperature = temperature
        self.prompt_type = prompt_type
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _get_prompt(self, prompt_type: Optional[str] = None) -> Dict[str, str]:
        """Get prompt template from prompts.py"""
        prompt_key = prompt_type if prompt_type else self.prompt_type
        if prompt_key not in PROMPTS:
            prompt_key = "select_path"
        return PROMPTS.get(prompt_key, PROMPTS["select_path"])

    def rerank_paths(
        self,
        query: str,
        paths: List[List[str]],
        chunks: Dict[str, Any],
        top_k: int = 10,
        prompt_type: Optional[str] = None
    ) -> List[List[str]]:
        """
        Rerank/select paths based on their relevance to the query.

        Args:
            query: The search query
            paths: List of paths, each path is a list of chunk_ids
            chunks: Dict mapping chunk_id to chunk object (with content)
            top_k: Number of top paths to return after selection
            prompt_type: Override the default prompt type for this call

        Returns:
            List of selected paths (top_k most relevant), each as a list of chunk_ids
        """
        if not paths:
            return []

        if len(paths) <= top_k:
            return paths

        prompt_template = self._get_prompt(prompt_type)

        path_texts = []
        for i, path in enumerate(paths):
            path_content = []
            for chunk_id in path:
                if chunk_id in chunks:
                    chunk = chunks[chunk_id]
                    if hasattr(chunk, 'content'):
                        path_content.append(chunk.content)
                    elif isinstance(chunk, dict) and 'content' in chunk:
                        path_content.append(chunk['content'])
                    else:
                        path_content.append(str(chunk))

            path_str = " -> ".join(path_content)
            path_texts.append(f"=== Path {i + 1} ===\n{path_str}")

        paths_text = "\n\n".join(path_texts)

        user_prompt = prompt_template["user"].format(
            query=query,
            paths_text=paths_text,
            num_paths=len(paths)
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": prompt_template["system"]},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"} if "json" in self.model_name.lower() else None
            )

            result_text = response.choices[0].message.content.strip()
            ranking = self._parse_ranking_result(result_text, paths)

            selected_paths = [paths[i] for i in ranking[:top_k]]
            print(f"Selected {len(selected_paths)} paths from {len(paths)} candidates")
            return selected_paths

        except Exception as e:
            print(f"Path selection failed: {e}")
            return paths[:top_k]

    def _parse_ranking_result(self, result_text: str, paths: List[List[str]]) -> List[int]:
        """Parse the ranking result from LLM response"""
        try:
            if result_text.startswith('{'):
                result_json = json.loads(result_text)
                if 'ranking' in result_json:
                    ranking = result_json['ranking']
                elif 'selected' in result_json:
                    ranking = result_json['selected']
                elif 'documents' in result_json:
                    ranking = result_json['documents']
                elif 'order' in result_json:
                    ranking = result_json['order']
                elif isinstance(result_json, dict) and len(result_json) == 1:
                    ranking = list(result_json.values())[0]
                else:
                    ranking = list(range(1, len(paths) + 1))
            elif result_text.startswith('['):
                ranking = json.loads(result_text)
            else:
                obj_match = re.search(r'\{[^}]*"ranking"\s*:\s*\[[^\]]*\][^}]*\}', result_text)
                if obj_match:
                    result_json = json.loads(obj_match.group())
                    ranking = result_json['ranking']
                else:
                    array_match = re.search(r'\[[\d,\s]+\]', result_text)
                    if array_match:
                        ranking = json.loads(array_match.group())
                    else:
                        ranking = list(range(1, len(paths) + 1))

            valid_ranking = []
            for path_num in ranking:
                if isinstance(path_num, int) and 1 <= path_num <= len(paths):
                    valid_ranking.append(path_num - 1)

            missing_indices = set(range(len(paths))) - set(valid_ranking)
            valid_ranking.extend(list(missing_indices))

            return valid_ranking

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Failed to parse ranking result: {e}")
            return list(range(len(paths)))
