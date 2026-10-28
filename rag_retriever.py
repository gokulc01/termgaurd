from __future__ import annotations
import math
import re
from collections import Counter
from typing import Dict, List, Tuple

from schemas import RetrievalLog


CURATED_DOCS: Dict[str, List[Dict[str, str]]] = {
    "find": [
        {
            "id": "find_man_1",
            "title": "find - search for files in a directory hierarchy",
            "content": (
                "find [path...] [expression]\n"
                "Safe flags: -name pattern, -iname pattern, -type [f|d|l], -mtime [-|+n], "
                "-size [-|+n][cwbkMG], -maxdepth levels, -mindepth levels, -empty, -print.\n"
                "Avoid: -exec, -delete (unless explicitly confirmed)."
            ),
        },
        {
            "id": "find_tldr_1",
            "title": "tldr: find common examples",
            "content": (
                "Find files by extension: find /workspace/logs -type f -name '*.log'\n"
                "Find files modified in last 7 days: find /workspace -type f -mtime -7\n"
                "Find empty files: find /workspace -type f -empty"
            ),
        },
    ],
    "grep": [
        {
            "id": "grep_man_1",
            "title": "grep - print lines that match patterns",
            "content": (
                "grep [OPTION...] PATTERNS [FILE...]\n"
                "Flags: -i (ignore case), -v (invert match), -n (line number), -c (count matches), "
                "-r (recursive), -E (extended regex), -F (fixed strings), -l (files with matches)."
            ),
        }
    ],
    "tar": [
        {
            "id": "tar_man_1",
            "title": "tar - archiving utility",
            "content": (
                "tar -czf archive.tar.gz -C /workspace/src .\n"
                "Flags: -c (create), -x (extract), -t (list), -z (gzip), -f (archive file), -C (change directory).\n"
                "Security: Never pass -P or --absolute-names."
            ),
        }
    ],
    "df": [
        {
            "id": "df_man_1",
            "title": "df - report file system disk space usage",
            "content": "df -h (human-readable), df -i (inodes), df -T (print filesystem type).",
        }
    ],
    "du": [
        {
            "id": "du_man_1",
            "title": "du - estimate file space usage",
            "content": "du -sh /workspace/dir (summary, human-readable), du -h --max-depth=1 /workspace.",
        }
    ],
    "chmod": [
        {
            "id": "chmod_man_1",
            "title": "chmod - change file mode bits",
            "content": (
                "chmod [OCTAL-MODE] FILE... (e.g., chmod 644 file.txt, chmod 750 script.sh, chmod 600 secret.key).\n"
                "Security: Never grant world-writable (777) or SUID (+s) permissions."
            ),
        }
    ],
}


class DocumentationRetriever:
    """
    Lightweight keyword + TF-IDF retriever over curated Linux man and tldr snippets.
    Logs retrieval coverage and unsupported-utility requests.
    """

    def __init__(self, docs_by_utility: Dict[str, List[Dict[str, str]]] = CURATED_DOCS):
        self.docs_by_utility = docs_by_utility
        self.retrieval_history: List[RetrievalLog] = []

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9_\-]+", text.lower())

    def retrieve(
        self, utility: str, operation_query: str, top_k: int = 2
    ) -> Tuple[RetrievalLog, str]:
        util_clean = utility.strip().lower()
        supported = util_clean in self.docs_by_utility

        if not supported:
            log = RetrievalLog(
                utility=util_clean,
                supported_utility=False,
                matched_passages=[],
                coverage_score=0.0,
            )
            self.retrieval_history.append(log)
            return (
                log,
                f"[RAG Notice] Utility '{util_clean}' has no curated man/tldr entry. "
                f"Supported documented utilities: {sorted(self.docs_by_utility.keys())}.",
            )

        query_tokens = Counter(self._tokenize(f"{util_clean} {operation_query}"))
        candidates = self.docs_by_utility[util_clean]

        scored: List[Tuple[float, Dict[str, str]]] = []
        for entry in candidates:
            doc_tokens = Counter(
                self._tokenize(entry["title"] + " " + entry["content"])
            )
            overlap = sum(min(query_tokens[t], doc_tokens[t]) for t in query_tokens)
            norm = math.sqrt(sum(v * v for v in query_tokens.values())) * math.sqrt(
                sum(v * v for v in doc_tokens.values())
            )
            score = round(overlap / norm, 4) if norm > 0 else 0.0
            scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        selected = scored[:top_k]

        passages_text = "\n\n".join(
            f"### {item['title']} (id={item['id']})\n{item['content']}"
            for _, item in selected
        )
        best_score = selected[0][0] if selected else 0.0

        log = RetrievalLog(
            utility=util_clean,
            supported_utility=True,
            matched_passages=[item["id"] for _, item in selected],
            coverage_score=best_score,
        )
        self.retrieval_history.append(log)
        return log, passages_text
