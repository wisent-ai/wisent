#!/usr/bin/env python3
"""Search for Tag dataset on HuggingFace."""

from datasets import list_datasets

# Search for datasets with "tag" in the name
all_datasets = list_datasets()
tag_datasets = [d for d in all_datasets if 'tag' in d.lower()]

print(f"Found {len(tag_datasets)} datasets with 'tag' in name:")
for d in sorted(tag_datasets)[:50]:
    print(f"  - {d}")

# Check if "tag" exists exactly
if "tag" in all_datasets:
    print("\n✓ Found exact match: 'tag'")
else:
    print("\n✗ No exact match for 'tag'")
