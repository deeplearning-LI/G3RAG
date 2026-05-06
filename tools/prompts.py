"""
Prompts for Knowledge Graph Builder

This module contains all the prompts used in the GraphBuilder class,
organized in a structured dictionary format for easy management and maintenance.
Each prompt has a "system" part (rules and guidelines) and a "user" part (input placeholders).
"""

PROMPTS = {}


# -------------------------------------------------------------------------
# Prompts for Chunk/Document Reranking
# -------------------------------------------------------------------------

PROMPTS["rerank_chunk"] = {
    "system": (
        "You are a precise document ranking assistant. "
        "Return only a JSON object with a 'ranking' field containing "
        "an array of document numbers ranked by relevance."
    ),

    "user": (
        "You are a document reranking system. Given a query and a list of documents, "
        "you need to rank the documents by their relevance to the query.\n\n"
        "Query: {query}\n\n"
        "Documents:\n{documents_text}\n\n"
        "Please rank these documents from most relevant to least relevant for answering the query.\n\n"
        'You must return a valid JSON object with a "ranking" field containing an array of '
        "document numbers (1-{num_documents}) in order of relevance, from most relevant to least relevant.\n\n"
        'Example format:\n{{"ranking": [3, 1, 5, 2, 4]}}\n\n'
        "This means Document 3 is most relevant, followed by Document 1, then Document 5, etc.\n\n"
        "Return only the JSON object, no other text."
    )
}


# -------------------------------------------------------------------------
# Prompts for Path Selection
# -------------------------------------------------------------------------

PROMPTS["select_path"] = {
    "system": (
        "You are a path selection assistant for graph-based retrieval. "
        "Given a query and a list of explored paths (each containing connected documents), "
        'select the most relevant paths for answering the query. '
        'Return only a JSON object with a \'ranking\' field.'
    ),

    "user": (
        "You are a path selection system. Given a query and a list of explored paths "
        "through a knowledge graph, select the paths that are most relevant "
        "for answering the query.\n\n"
        "Query: {query}\n\n"
        "Paths:\n{paths_text}\n\n"
        "Each path is a sequence of connected documents that form a reasoning chain.\n\n"
        "Please select and rank the paths from most relevant to least relevant "
        "for answering the query.\n\n"
        'You must return a valid JSON object with a "ranking" field containing an array of '
        "path numbers (1-{num_paths}) in order of relevance, from most relevant to least relevant.\n\n"
        'Example format:\n{{"ranking": [3, 1, 5, 2, 4]}}\n\n'
        "This means Path 3 is most relevant, followed by Path 1, then Path 5, etc.\n\n"
        "Return only the JSON object, no other text."
    )
}


# -------------------------------------------------------------------------
# Prompts for Chunk Filtering
# -------------------------------------------------------------------------

PROMPTS["filter_chunk"] = {
    "system": (
        "You are a document filtering assistant. Given a query and a list of documents, "
        "select the top-k most relevant documents. Return only a JSON object with a "
        "'selected' field containing an array of document numbers."
    ),

    "user": (
        "You are a document filtering system. Given a query and a list of documents, "
        "select the top {top_k} most relevant documents for answering the query.\n\n"
        "Query: {query}\n\n"
        "Documents:\n{documents_text}\n\n"
        "Please select the {top_k} most relevant documents.\n\n"
        'You must return a valid JSON object with a "selected" field containing an array of '
        "document numbers (1-{num_documents}) of the selected documents, ranked by relevance.\n\n"
        'Example format:\n{{"selected": [3, 1, 5]}}\n\n'
        "This means Documents 3, 1, and 5 are selected, in that order of relevance.\n\n"
        "Return only the JSON object, no other text."
    )
}
