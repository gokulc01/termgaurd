import os
import json
import numpy as np

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss


class HybridDocRetriever:
    def __init__(
        self,
        json_path: str = "~/Documents/iiith/sem3/lma/grp_prj/termgaurd_agent/tldr_corpus.json",
    ):
        with open(json_path, "r", encoding="utf-8") as f:
            corpus_dict = json.load(f)

        self.utilities = list(corpus_dict.keys())
        self.docs = list(corpus_dict.values())

        # Stage 1: Lexical Index (BM25)
        tokenized_docs = [doc.lower().split() for doc in self.docs]
        self.bm25 = BM25Okapi(tokenized_docs)

        # Stage 2: Dense Index (FAISS)
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")

        index_path = os.path.expanduser(
            "~/Documents/iiith/sem3/lma/grp_prj/termgaurd_agent/faiss_index.bin"
        )
        if os.path.exists(index_path):
            # Instant load
            self.index = faiss.read_index(index_path)
        else:
            # First-time build
            self.index = faiss.IndexFlatL2(384)
            embeddings = self.encoder.encode(self.docs, convert_to_numpy=True)
            self.index.add(embeddings)
            faiss.write_index(self.index, index_path)

    def retrieve(self, prompt: str, top_k: int = 2) -> str:
        prompt_lower = prompt.lower()

        # 1. Lexical Ranking
        tokenized_query = prompt_lower.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_ranks = np.argsort(bm25_scores)[::-1]

        # 2. Dense Ranking
        query_emb = self.encoder.encode([prompt], convert_to_numpy=True)
        _, dense_ranks = self.index.search(query_emb, len(self.utilities))
        dense_ranks = dense_ranks[0]

        # 3. Reciprocal Rank Fusion (Combines exact keywords + semantic meaning)
        rrf_scores = np.zeros(len(self.utilities))
        k = 60
        for rank, doc_idx in enumerate(bm25_ranks):
            rrf_scores[doc_idx] += 1.0 / (k + rank + 1)
        for rank, doc_idx in enumerate(dense_ranks):
            rrf_scores[doc_idx] += 1.0 / (k + rank + 1)

        # 4. Extract Top K results
        best_indices = np.argsort(rrf_scores)[::-1][:top_k]

        results = []
        for idx in best_indices:
            util = self.utilities[idx]
            doc = self.docs[idx]
            results.append(f"Utility: {util}\nDocumentation: {doc}")

        return "\n\n".join(results)
