#!/usr/bin/env python3
"""
HuggingFace Dataset Downloader CLI

A fast, multithreaded CLI tool for downloading datasets from HuggingFace.
Supports images, labels (Parquet, JSON, CSV), and preserves folder structure.
"""

import argparse
import os
import sys
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import requests
from huggingface_hub import HfApi, hf_hub_url, login
from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="hf_downloader",
        description="Download datasets from HuggingFace with multithreading support.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s edc505/pokemon
  %(prog)s jise/simworld-20k-balanced --threads 16
  %(prog)s edc505/pokemon --output ./my_data --progress
  %(prog)s private/dataset --token YOUR_HF_TOKEN
  %(prog)s edc505/pokemon --filter .png,.jpg --limit 100
        """
    )

    parser.add_argument(
        "dataset",
        help="Dataset path (e.g., 'edc505/pokemon' or 'jise/simworld-20k-balanced')"
    )

    parser.add_argument(
        "--base-url",
        default="https://huggingface.co/datasets",
        help="Base URL for HuggingFace datasets (default: https://huggingface.co/datasets)"
    )

    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output directory (default: dataset name with '_' instead of '/')"
    )

    parser.add_argument(
        "-t", "--threads",
        type=int,
        default=8,
        help="Number of parallel download threads (default: 8)"
    )

    parser.add_argument(
        "-f", "--filter",
        default=None,
        help="Comma-separated file extensions to download (e.g., '.png,.jpg,.json')"
    )

    parser.add_argument(
        "-l", "--limit",
        type=int,
        default=None,
        help="Maximum number of files to download"
    )

    parser.add_argument(
        "--token",
        default=None,
        help="HuggingFace API token for private datasets (or set HF_TOKEN env var)"
    )

    parser.add_argument(
        "-p", "--progress",
        action="store_true",
        help="Show progress bar during download"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files without downloading"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )

    return parser.parse_args()


def get_token(args_token: Optional[str]) -> Optional[str]:
    """Get HuggingFace token from argument or environment variable."""
    if args_token:
        return args_token
    return os.environ.get("HF_TOKEN")


def get_output_dir(dataset: str, output: Optional[str]) -> Path:
    """Generate output directory path."""
    if output:
        return Path(output)
    return Path(dataset.replace("/", "_"))


def filter_files(files: list, extensions: Optional[str], limit: Optional[int]) -> list:
    """Filter files by extension and limit count."""
    if extensions:
        ext_list = [ext.strip().lower() for ext in extensions.split(",")]
        ext_list = [ext if ext.startswith(".") else f".{ext}" for ext in ext_list]
        files = [f for f in files if any(f.rfilename.lower().endswith(ext) for ext in ext_list)]

    if limit:
        files = files[:limit]

    return files


def download_file(
    file_info: dict,
    output_dir: Path,
    token: Optional[str],
    verbose: bool = False
) -> tuple[str, bool, str]:
    """
    Download a single file.

    Returns:
        Tuple of (filename, success, message)
    """
    filename = file_info["rfilename"]
    url = file_info["url"]

    file_path = output_dir / filename
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        response = requests.get(url, headers=headers, stream=True, timeout=60)
        response.raise_for_status()

        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        if verbose:
            return filename, True, f"Downloaded: {filename}"
        return filename, True, ""

    except requests.exceptions.RequestException as e:
        return filename, False, f"Failed: {filename} - {str(e)}"
    except IOError as e:
        return filename, False, f"IO Error: {filename} - {str(e)}"


def list_dataset_files(api: HfApi, dataset: str, token: Optional[str]) -> list:
    """List all files in a dataset repository."""
    try:
        files = api.list_repo_files(
            repo_id=dataset,
            repo_type="dataset",
            token=token
        )

        file_infos = []
        for f in files:
            if f.startswith("."):
                continue
            url = hf_hub_url(
                repo_id=dataset,
                filename=f,
                repo_type="dataset"
            )
            file_infos.append({
                "rfilename": f,
                "url": url
            })

        return file_infos

    except RepositoryNotFoundError:
        print(f"Error: Dataset '{dataset}' not found.")
        print("Check if the dataset exists or if you need authentication for private datasets.")
        sys.exit(1)
    except GatedRepoError:
        print(f"Error: Dataset '{dataset}' is gated/private.")
        print("Use --token YOUR_TOKEN to authenticate.")
        sys.exit(1)
    except Exception as e:
        print(f"Error listing files: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    args = parse_arguments()

    token = get_token(args.token)
    output_dir = get_output_dir(args.dataset, args.output)

    if args.verbose:
        print(f"Dataset: {args.dataset}")
        print(f"Output directory: {output_dir}")
        print(f"Threads: {args.threads}")
        if args.filter:
            print(f"Filter: {args.filter}")
        if args.limit:
            print(f"Limit: {args.limit}")
        print()

    api = HfApi()

    print(f"Fetching file list from '{args.dataset}'...")
    files = list_dataset_files(api, args.dataset, token)

    if not files:
        print("No files found in dataset.")
        sys.exit(0)

    files = filter_files(files, args.filter, args.limit)

    if not files:
        print("No files match the specified filters.")
        sys.exit(0)

    print(f"Found {len(files)} file(s) to download.")

    if args.dry_run:
        print("\nFiles (dry run):")
        for f in files:
            print(f"  {f['rfilename']}")
        sys.exit(0)

    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    fail_count = 0

    if args.progress:
        try:
            from tqdm import tqdm
            progress_bar = tqdm(total=len(files), desc="Downloading", unit="file")
        except ImportError:
            print("Warning: tqdm not installed. Install with 'pip install tqdm' for progress bar.")
            print("Continuing without progress bar...\n")
            args.progress = False

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {
            executor.submit(download_file, f, output_dir, token, args.verbose): f
            for f in files
        }

        for future in as_completed(futures):
            filename, success, message = future.result()

            if success:
                success_count += 1
            else:
                fail_count += 1
                if message:
                    print(message)

            if args.progress:
                progress_bar.update(1)
            elif args.verbose and message:
                print(message)

    if args.progress:
        progress_bar.close()

    print(f"\nDownload complete!")
    print(f"  Success: {success_count}")
    print(f"  Failed:  {fail_count}")
    print(f"  Output:  {output_dir.absolute()}")


if __name__ == "__main__":
    main()
