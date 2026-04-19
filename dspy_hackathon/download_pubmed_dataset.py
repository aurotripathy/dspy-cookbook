"""Download the PubMed text-classification dataset CSVs into ./dataset/.

By default, downloads BOTH the train and test splits.

Usage:
    python download_pubmed_dataset.py                    # downloads train.csv and test.csv
    python download_pubmed_dataset.py --split train      # only train.csv
    python download_pubmed_dataset.py --split test       # only test.csv
    python download_pubmed_dataset.py --url <full_url> --out-dir some_folder

Uses certifi's CA bundle to avoid macOS-style `SSL: CERTIFICATE_VERIFY_FAILED` errors.
"""

from __future__ import annotations

import argparse
import ssl
import sys
import urllib.request
from pathlib import Path

import certifi

BASE_URL = (
    "https://huggingface.co/datasets/ml4pubmed/"
    "pubmed-text-classification-cased/resolve/main"
)
DEFAULT_URLS = [
    f"{BASE_URL}/train.csv",
    f"{BASE_URL}/test.csv",
]


def download(url: str, out_dir: Path, chunk_size: int = 1 << 20) -> Path:
    """Download `url` into `out_dir` using the filename from the URL; returns the local path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / Path(url).name

    ctx = ssl.create_default_context(cafile=certifi.where())
    print(f"Downloading {url}")
    print(f"         -> {dest}")

    with urllib.request.urlopen(url, context=ctx, timeout=120) as resp:
        total = resp.headers.get("Content-Length")
        total_int = int(total) if total and total.isdigit() else None
        written = 0
        with dest.open("wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                written += len(chunk)
                if total_int:
                    pct = written / total_int * 100.0
                    print(f"  {written / 1e6:7.2f} MB / {total_int / 1e6:.2f} MB  ({pct:5.1f}%)", end="\r")
                else:
                    print(f"  {written / 1e6:7.2f} MB", end="\r")
        print()
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", help="Single CSV URL to download (overrides --split and defaults).")
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        help="Download only the named split (default: both train and test).",
    )
    parser.add_argument(
        "--out-dir",
        default="dataset",
        type=Path,
        help="Destination directory (default: ./dataset).",
    )
    args = parser.parse_args(argv)

    if args.url:
        urls = [args.url]
    elif args.split:
        urls = [f"{BASE_URL}/{args.split}.csv"]
    else:
        urls = list(DEFAULT_URLS)

    saved: list[Path] = []
    for url in urls:
        dest = download(url, args.out_dir)
        saved.append(dest)

    print()
    print("Saved files:")
    for p in saved:
        print(f"  {p}  ({p.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
