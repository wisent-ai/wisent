#!/usr/bin/env python3
"""
Full Benchmark Downloader

Downloads complete benchmarks from lm-eval-harness and saves them in a structured format.
Downloads the ENTIRE benchmark datasets, not just samples.

Usage:
    python download_full_benchmarks.py --benchmarks glue mmlu --force
    python download_full_benchmarks.py --all  # Download all benchmarks
"""

import argparse
import json
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from wisent.core.utils.config_tools.constants import JSON_INDENT

from wisent.core.primitives.models.lm_harness_integration.only_benchmarks import CORE_BENCHMARKS
from wisent.core.utils.config_tools.constants import SECONDS_PER_MINUTE

from ._download_helpers import (
    BasicConvertersMixin,
    TextConvertersMixin,
    AdvancedConvertersMixin,
    QAConvertersMixin,
)


class FullBenchmarkDownloader(
    BasicConvertersMixin, TextConvertersMixin,
    AdvancedConvertersMixin, QAConvertersMixin,
):
    """Downloads complete benchmarks and saves them to disk."""

    UNAVAILABLE_BENCHMARKS = {}

    def __init__(self, download_dir: str = "full_benchmarks"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)

        # Create subdirectories
        self.data_dir = self.download_dir / "data"
        self.metadata_dir = self.download_dir / "metadata"
        self.data_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)

        print("🚀 Full Benchmark Downloader")
        print(f"📁 Download directory: {self.download_dir.absolute()}")

    def download_complete_benchmark(
        self, benchmark_name: str, benchmark_config: dict, force: bool = False
    ) -> Optional[str]:
        task_name = benchmark_config["task"]
        tags = benchmark_config.get("tags", [])

        # Check if already exists
        data_file = self.data_dir / f"{benchmark_name}.pkl"
        metadata_file = self.metadata_dir / f"{benchmark_name}_metadata.json"

        if data_file.exists() and metadata_file.exists() and not force:
            print(f"   ⏩ Skipping {benchmark_name} (already exists)")
            return str(data_file)

        print(f"   📥 Downloading complete benchmark: {benchmark_name}")
        print(f"      🔄 Loading full dataset for task: {task_name}")

        start_time = time.time()

        try:
            # Import lm_eval to download complete datasets
            from lm_eval import tasks

            # Get the task
            task_dict = tasks.get_task_dict([task_name])
            if task_name not in task_dict:
                print(f"      ❌ Task {task_name} not found in lm_eval")
                return None

            task = task_dict[task_name]

            # Download complete dataset - combine all splits into one unified dataset
            complete_data = {
                "benchmark_name": benchmark_name,
                "task_name": task_name,
                "config": benchmark_config,
                "all_samples": [],
                "total_samples": 0,
                "splits_found": [],
            }

            # Get all available document splits
            splits_to_try = ["test", "validation", "train", "dev"]

            for split in splits_to_try:
                try:
                    if hasattr(task, f"{split}_docs"):
                        docs_method = getattr(task, f"{split}_docs")
                        docs = list(docs_method())

                        if docs:
                            print(f"      📊 Found {len(docs)} samples in {split} split")
                            complete_data["splits_found"].append(split)

                            # Convert documents to serializable format and add to unified list
                            for i, doc in enumerate(docs):
                                if i % 1000 == 0 and i > 0:
                                    print(f"         Processing {split} {i}/{len(docs)}...")

                                # Convert doc to dict, handling different doc types
                                if hasattr(doc, "__dict__"):
                                    doc_dict = doc.__dict__.copy()
                                elif isinstance(doc, dict):
                                    doc_dict = doc.copy()
                                else:
                                    doc_dict = {"content": str(doc)}

                                # Add split origin info
                                doc_dict["_split_origin"] = split

                                # Ensure all values are serializable
                                serializable_doc = {}
                                for key, value in doc_dict.items():
                                    try:
                                        json.dumps(value)  # Test if serializable
                                        serializable_doc[key] = value
                                    except (TypeError, ValueError):
                                        serializable_doc[key] = str(value)

                                complete_data["all_samples"].append(serializable_doc)

                            complete_data["total_samples"] += len(docs)

                except Exception as e:
                    print(f"      ⚠️  Could not load {split} split: {e}")
                    continue

            if complete_data["total_samples"] == 0:
                print(f"      ❌ No data found for {benchmark_name}")
                return None

            processing_time = time.time() - start_time

            # Add metadata
            metadata = {
                "benchmark_name": benchmark_name,
                "task_name": task_name,
                "config": benchmark_config,
                "download_timestamp": datetime.now().isoformat(),
                "processing_time_seconds": processing_time,
                "total_samples": complete_data["total_samples"],
                "splits_found": complete_data["splits_found"],
                "task_info": {
                    "description": getattr(task, "DESCRIPTION", "No description available"),
                    "citation": getattr(task, "CITATION", "No citation available"),
                    "homepage": getattr(task, "HOMEPAGE", "No homepage available"),
                },
            }

            # Convert to contrastive pairs
            contrastive_data = self.convert_to_contrastive_pairs(benchmark_name, complete_data)

            # Save only the contrastive pairs
            data_file = self.data_dir / f"{benchmark_name}.pkl"
            with open(data_file, "wb") as f:
                pickle.dump(contrastive_data["contrastive_pairs"], f)

            # Save metadata as JSON
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=JSON_INDENT)

            print(f"      ✅ Saved benchmark: {benchmark_name}")
            print(f"         📊 Contrastive pairs: {len(contrastive_data['contrastive_pairs'])}")
            print(f"         ⏱️  Time: {processing_time:.1f}s")

            return str(data_file)

        except Exception as e:
            processing_time = time.time() - start_time
            print(f"      ❌ Failed to download {benchmark_name}: {e}")
            print(f"         ⏱️  Time: {processing_time:.1f}s")
            return None
    def download_all_benchmarks(self, benchmarks: Optional[List[str]] = None, force: bool = False) -> Dict[str, Any]:
        if benchmarks is None:
            # Filter out known unavailable benchmarks when downloading all
            available_benchmarks = {
                name: config for name, config in CORE_BENCHMARKS.items() if name not in self.UNAVAILABLE_BENCHMARKS
            }
            benchmarks_to_download = available_benchmarks

            # Report excluded benchmarks
            excluded_count = len(CORE_BENCHMARKS) - len(available_benchmarks)
            if excluded_count > 0:
                print(f"⏩ Excluding {excluded_count} known unavailable benchmarks")
                print(f"   📋 Available benchmarks: {len(available_benchmarks)}/{len(CORE_BENCHMARKS)}")
        else:
            benchmarks_to_download = {name: CORE_BENCHMARKS[name] for name in benchmarks if name in CORE_BENCHMARKS}

            # Check for invalid benchmarks
            invalid = [name for name in benchmarks if name not in CORE_BENCHMARKS]
            if invalid:
                print(f"⚠️  Invalid benchmarks (skipping): {invalid}")

            # Warn about unavailable benchmarks that were explicitly requested
            unavailable_requested = [name for name in benchmarks if name in self.UNAVAILABLE_BENCHMARKS]
            if unavailable_requested:
                print(f"⚠️  Requested benchmarks are known to be unavailable: {unavailable_requested}")
                print("   🔧 These will likely fail. Remove from list to avoid delays.")

        print(f"\n🏗️ Downloading {len(benchmarks_to_download)} complete benchmarks")
        print(f"   Force redownload: {force}")

        results = {
            "successful": [],
            "failed": [],
            "skipped": [],
            "excluded": list(self.UNAVAILABLE_BENCHMARKS) if benchmarks is None else [],
            "total_time": 0,
        }

        total_start_time = time.time()

        for i, (benchmark_name, benchmark_config) in enumerate(benchmarks_to_download.items(), 1):
            print(f"\n[{i:2d}/{len(benchmarks_to_download)}] 🎯 Processing benchmark: {benchmark_name}")
            print(f"   Task: {benchmark_config['task']}")
            print(f"   Tags: {benchmark_config.get('tags', [])}")

            try:
                result_path = self.download_complete_benchmark(benchmark_name, benchmark_config, force)

                if result_path:
                    results["successful"].append(benchmark_name)
                else:
                    results["failed"].append(benchmark_name)

            except Exception as e:
                print(f"   ❌ Exception downloading {benchmark_name}: {e}")
                results["failed"].append(benchmark_name)

            # Progress update
            elapsed = time.time() - total_start_time
            if i < len(benchmarks_to_download):
                eta = elapsed * (len(benchmarks_to_download) - i) / i
                print(f"\n📊 Progress: {i}/{len(benchmarks_to_download)} benchmarks completed")
                print(f"   ⏱️  Elapsed: {elapsed / SECONDS_PER_MINUTE:.1f}min, ETA: {eta / SECONDS_PER_MINUTE:.1f}min")

        results["total_time"] = time.time() - total_start_time
        return results
    def convert_to_contrastive_pairs(self, benchmark_name: str, complete_data: Dict[str, Any]) -> Dict[str, Any]:
        print("      🔄 Converting to contrastive pairs...")

        contrastive_pairs = []

        for i, sample in enumerate(complete_data["all_samples"]):
            try:
                pairs = self._convert_sample_to_pairs(sample, benchmark_name)
                if pairs:
                    contrastive_pairs.extend(pairs)
            except Exception as e:
                print(f"         ⚠️ Conversion error for sample {i}: {e}")

        return {"contrastive_pairs": contrastive_pairs}


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(description="Download benchmarks")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--benchmarks", nargs="+", help="Benchmarks to download")
    group.add_argument("--all", action="store_true", help="Download all")
    parser.add_argument("--force", action="store_true", help="Force redownload")
    parser.add_argument("--download-dir", default="full_benchmarks")
    args = parser.parse_args()
    downloader = FullBenchmarkDownloader(download_dir=args.download_dir)
    try:
        to_dl = None if args.all else args.benchmarks
        results = downloader.download_all_benchmarks(benchmarks=to_dl, force=args.force)
        ok, fail = len(results["successful"]), len(results["failed"])
        t = results["total_time"] / SECONDS_PER_MINUTE
        print(f"Summary: {ok} OK, {fail} failed, {t:.1f}min")
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
