#!/usr/bin/env python3
import os
import json
import urllib.request
import zipfile
from pathlib import Path

# Config
TLDR_ZIP_URL = "https://github.com/tldr-pages/tldr/archive/refs/heads/main.zip"
ZIP_PATH = "/tmp/tldr.zip"
EXTRACT_DIR = "/tmp/tldr-main"
OUTPUT_JSON = "tldr_corpus.json"


def download_and_extract():
    print(f"Downloading tldr pages from {TLDR_ZIP_URL}...")
    urllib.request.urlretrieve(TLDR_ZIP_URL, ZIP_PATH)

    print("Extracting archive to /tmp...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall("/tmp")


def parse_tldr_md(filepath: Path) -> str:
    """Parses tldr markdown to extract the command description."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return ""

    description_lines = []
    for line in lines:
        line = line.strip()
        # In tldr syntax, summaries always start with '>'
        if line.startswith(">"):
            description_lines.append(line.lstrip(">").strip())
        # We can stop reading as soon as we hit the examples section (starts with '-')
        elif line.startswith("-"):
            break

    return " ".join(description_lines)


def build_corpus():
    corpus = {}

    # We only want 'linux' and 'common' commands, ignoring windows/osx specific ones
    target_dirs = [
        Path("/tmp/tldr-main/pages/common"),
        Path("/tmp/tldr-main/pages/linux"),
    ]

    print("Parsing Markdown files into JSON corpus...")
    for t_dir in target_dirs:
        if not t_dir.exists():
            continue

        for md_file in t_dir.glob("*.md"):
            cmd_name = md_file.stem
            description = parse_tldr_md(md_file)

            if description:
                # If a command exists in both 'common' and 'linux', 'linux' will overwrite
                # (which is preferred for a Linux shell copilot)
                corpus[cmd_name] = description

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2)

    print(f"Success! Extracted {len(corpus)} commands into {OUTPUT_JSON}")


if __name__ == "__main__":
    download_and_extract()
    build_corpus()

    # Cleanup temporary files
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
