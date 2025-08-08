"""Minimal utilities to print DOIs for local dataset or newly fetched important papers.

Usage:
  # Print DOIs for LANDMARK_PAPERS
  uv run python -m src.data.get_dois

  # Fetch and print DOIs for top-cited papers (optionally filter by concept and year)
  uv run python -m src.data.get_dois --top 50 --since 2015 --concept C41008148
"""

from __future__ import annotations

from src.data.dataset import LANDMARK_PAPERS
from src.services.openalex_client import OpenAlexClient
import argparse
import json
from pathlib import Path


def print_local_landmark_dois(client: OpenAlexClient) -> None:
    results = client.get_many_papers(LANDMARK_PAPERS)
    print("openalex_id\tdoi")
    for paper_id in LANDMARK_PAPERS:
        paper = results.get(paper_id) or {}
        doi = (paper.get("ids") or {}).get("doi") if paper else None
        print(f"{paper_id}\t{doi or ''}")


def print_top_papers_dois(client: OpenAlexClient, top: int, since: int | None, concept: str | None) -> None:
    works = client.get_top_papers(limit=top, since_year=since, concept_id=concept)
    print("openalex_id\ttitle\tdoi")
    for w in works:
        openalex_id = (w.get("id") or "").split("/")[-1]
        title = w.get("title") or ""
        doi = (w.get("ids") or {}).get("doi") or ""
        print(f"{openalex_id}\t{title}\t{doi}")


def save_works_to_json(filepath: str, works: list[dict]) -> None:
    data = []
    for w in works:
        data.append(
            {
                "openalex_id": (w.get("id") or "").split("/")[-1],
                "title": w.get("title") or "",
                "doi": (w.get("ids") or {}).get("doi") or "",
            }
        )
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print or save DOIs from dataset or OpenAlex top papers")
    parser.add_argument("--top", type=int, default=0, help="Fetch top-cited papers (1-200). If 0, use local dataset")
    parser.add_argument("--since", type=int, default=None, help="Lower bound publication year for top papers")
    parser.add_argument("--concept", type=str, default=None, help="OpenAlex concept ID filter (e.g., C41008148)")
    parser.add_argument("--out", type=str, default=None, help="If set, save results to this JSON file")
    args = parser.parse_args()

    client = OpenAlexClient()
    if args.top and args.top > 0:
        works = client.get_top_papers(limit=args.top, since_year=args.since, concept_id=args.concept)
        if args.out:
            save_works_to_json(args.out, works)
        else:
            print("openalex_id\ttitle\tdoi")
            for w in works:
                openalex_id = (w.get("id") or "").split("/")[-1]
                title = w.get("title") or ""
                doi = (w.get("ids") or {}).get("doi") or ""
                print(f"{openalex_id}\t{title}\t{doi}")
    else:
        if args.out:
            # Load local dataset and save in unified JSON schema
            results = client.get_many_papers(LANDMARK_PAPERS)
            works = []
            for pid in LANDMARK_PAPERS:
                w = results.get(pid) or {}
                works.append(
                    {
                        "id": f"https://openalex.org/{pid}",
                        "title": w.get("title") or "",
                        "ids": w.get("ids") or {},
                    }
                )
            save_works_to_json(args.out, works)
        else:
            print_local_landmark_dois(client)


if __name__ == "__main__":
    main()


