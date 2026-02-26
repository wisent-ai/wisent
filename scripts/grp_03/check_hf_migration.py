"""Check HuggingFace migration status for wisent-ai/activations."""
import os
import sys

from huggingface_hub import HfApi


def main():
    api = HfApi(token=os.environ.get("HF_TOKEN", ""))

    try:
        info = api.dataset_info("wisent-ai/activations")
        print(f"Repo: {info.id}")
        print(f"Private: {info.private}")
        print(f"Last modified: {info.last_modified}")
        siblings = info.siblings or []
        print(f"Total files (from siblings): {len(siblings)}")
        total_size = sum(s.size for s in siblings if s.size)
        print(f"Total size: {total_size / (1024**3):.2f} GB")
    except Exception as e:
        print(f"Repo not found or error: {e}")
        sys.exit(1)

    # Categorize from siblings (no recursive API call needed)
    activations = []
    raw_acts = []
    pair_texts = []
    markers = []
    index_found = False
    other = []

    for s in siblings:
        fn = s.rfilename
        sz = s.size
        if fn.startswith("activations/"):
            activations.append((fn, sz))
        elif fn.startswith("raw_activations/"):
            raw_acts.append((fn, sz))
        elif fn.startswith("pair_texts/"):
            pair_texts.append((fn, sz))
        elif fn.startswith("markers/"):
            markers.append((fn, sz))
        elif fn == "index.json":
            index_found = True
        else:
            other.append((fn, sz))

    print(f"\n  Activations: {len(activations)} files")
    print(f"  Raw activations: {len(raw_acts)} files")
    print(f"  Pair texts: {len(pair_texts)} files")
    print(f"  Markers: {len(markers)} files")
    print(f"  Index: {'YES' if index_found else 'NO'}")
    print(f"  Other: {len(other)} files")

    models = set()
    benchmarks = set()
    strategies = set()
    for fname, _ in activations:
        parts = fname.split("/")
        if len(parts) >= 4:
            models.add(parts[1])
            benchmarks.add(parts[2])
            strategies.add(parts[3])

    if models:
        print(f"\n--- Models ({len(models)}) ---")
        for m in sorted(models):
            count = sum(1 for f, _ in activations if f.split("/")[1] == m)
            print(f"  {m} ({count} files)")
    if benchmarks:
        print(f"\n--- Benchmarks ({len(benchmarks)}) ---")
        for b in sorted(benchmarks):
            print(f"  {b}")
    if strategies:
        print(f"\n--- Strategies ({len(strategies)}) ---")
        for s in sorted(strategies):
            print(f"  {s}")

    if markers:
        print(f"\n--- Completed markers ({len(markers)}) ---")
        for fname, _ in sorted(markers):
            print(f"  {fname}")

    if pair_texts:
        print(f"\n--- Pair texts ({len(pair_texts)}) ---")
        for fname, sz in sorted(pair_texts):
            label = f"({sz / 1024:.1f} KB)" if sz else ""
            print(f"  {fname} {label}")

    if other:
        print(f"\n--- Other files ({len(other)}) ---")
        for fname, sz in sorted(other):
            print(f"  {fname}")


if __name__ == "__main__":
    main()
