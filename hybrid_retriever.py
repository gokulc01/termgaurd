import hashlib
import json
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Offline by default for reproducible benchmarking; override by exporting the vars.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import faiss  # noqa: E402
from rank_bm25 import BM25Okapi  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

from schemas import RetrievalLog  # noqa: E402
from static_gate import SUPPORTED_UTILITIES  # noqa: E402

DEFAULT_CORPUS = os.environ.get("TERMGUARD_CORPUS", "tldr_corpus.json")
_TOKEN_RE = re.compile(r"[a-z0-9_\-\.]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class HybridDocRetriever:
    """BM25 + dense (MiniLM/FAISS) retrieval fused with Reciprocal Rank Fusion."""

    def __init__(
        self,
        json_path: str = DEFAULT_CORPUS,
        index_path: Optional[str] = None,
        model_name: str = "all-MiniLM-L6-v2",
        rrf_k: int = 60,
        supported_only: bool = False,
    ):
        """
        supported_only=True indexes only utilities the static gate can allow, so RAG
        stops steering the model toward e.g. zgrep/ack that are always denied.
        """
        corpus_file = Path(os.path.expanduser(json_path)).resolve()
        raw = corpus_file.read_bytes()
        corpus_dict = json.loads(raw.decode("utf-8"))
        if supported_only:
            corpus_dict = {u: d for u, d in corpus_dict.items() if u in SUPPORTED_UTILITIES}
            raw += b"|supported_only"

        self.utilities: List[str] = list(corpus_dict.keys())
        self.docs: List[str] = list(corpus_dict.values())
        self.rrf_k = rrf_k
        # Fingerprint corpus + encoder so a stale cached index is never reused.
        self.corpus_sha = hashlib.sha256(raw + model_name.encode()).hexdigest()[:12]

        # Stage 1: lexical index
        self.bm25 = BM25Okapi([_tokenize(d) for d in self.docs])

        # Stage 2: dense index, cached next to the corpus and keyed by its hash
        self.encoder = SentenceTransformer(model_name)
        dim = self.encoder.get_sentence_embedding_dimension()
        if index_path is None:
            index_path = str(corpus_file.with_suffix(f".{self.corpus_sha}.faiss"))
        index_path = os.path.expanduser(index_path)

        self.index = None
        if os.path.exists(index_path):
            idx = faiss.read_index(index_path)
            if idx.ntotal == len(self.docs) and idx.d == dim:
                self.index = idx
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
            emb = self.encoder.encode(self.docs, convert_to_numpy=True).astype("float32")
            self.index.add(emb)
            faiss.write_index(self.index, index_path)

    def _rank(self, prompt: str) -> Tuple[np.ndarray, np.ndarray]:
        bm25_ranks = np.argsort(self.bm25.get_scores(_tokenize(prompt)))[::-1]
        query_emb = self.encoder.encode([prompt], convert_to_numpy=True).astype("float32")
        _, dense = self.index.search(query_emb, len(self.utilities))

        rrf = np.zeros(len(self.utilities))
        for rank, idx in enumerate(bm25_ranks):
            rrf[idx] += 1.0 / (self.rrf_k + rank + 1)
        for rank, idx in enumerate(dense[0]):
            if idx >= 0:
                rrf[idx] += 1.0 / (self.rrf_k + rank + 1)
        return np.argsort(rrf)[::-1], rrf

    def retrieve_with_log(self, prompt: str, top_k: int = 2) -> Tuple[RetrievalLog, str]:
        order, rrf = self._rank(prompt)
        best = list(order[:top_k])
        text = "\n\n".join(
            f"Utility: {self.utilities[i]}\nDocumentation: {self.docs[i]}" for i in best
        )
        top_util = self.utilities[best[0]] if best else ""
        max_possible = 2.0 / (self.rrf_k + 1)  # rank 1 in both lists
        log = RetrievalLog(
            utility=top_util,
            # Whether the top retrieved utility is one the static gate can ever allow.
            supported_utility=top_util in SUPPORTED_UTILITIES,
            matched_passages=[self.utilities[i] for i in best],
            coverage_score=round(float(rrf[best[0]]) / max_possible, 4) if best else 0.0,
        )
        return log, text

    def retrieve(self, prompt: str, top_k: int = 2) -> str:
        return self.retrieve_with_log(prompt, top_k)[1]
