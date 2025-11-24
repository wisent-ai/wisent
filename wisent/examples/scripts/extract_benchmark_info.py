#!/usr/bin/env python3
"""Extract benchmark information from README files."""

import json
import re
from pathlib import Path

def extract_info_from_readme(readme_path):
    """Extract title, description, paper, homepage from README."""
    content = readme_path.read_text()

    info = {
        "name": readme_path.stem,
        "description": "",
        "paper": "",
        "homepage": ""
    }

    # Extract title (first # heading)
    title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    if title_match:
        info["name"] = title_match.group(1).strip()

    # Extract paper link
    paper_match = re.search(r'(?:Paper|Abstract).*?https?://[^\s\)]+', content, re.IGNORECASE)
    if paper_match:
        info["paper"] = paper_match.group(0)

    # Extract homepage
    homepage_match = re.search(r'Homepage.*?https?://[^\s\)]+', content, re.IGNORECASE)
    if homepage_match:
        info["homepage"] = homepage_match.group(0)

    # Extract description (first paragraph after title or abstract)
    desc_match = re.search(r'(?:Abstract|##\s*Abstract)[:\s]*(.+?)(?:\n\n|\n#)', content, re.DOTALL | re.IGNORECASE)
    if desc_match:
        desc = desc_match.group(1).strip()
        # Clean up
        desc = re.sub(r'\s+', ' ', desc)
        desc = desc[:500]  # Limit length
        info["description"] = desc
    else:
        # Try to get first substantial paragraph
        paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 50]
        if paragraphs:
            info["description"] = paragraphs[0][:500]

    return info

def main():
    readmes_dir = Path(__file__).parent / "readmes"
    output_file = Path(__file__).parent / "results" / "benchmark_descriptions.json"

    output_file.parent.mkdir(exist_ok=True)

    all_info = {}

    for readme_path in sorted(readmes_dir.glob("*.md")):
        benchmark_name = readme_path.stem
        info = extract_info_from_readme(readme_path)
        all_info[benchmark_name] = info
        print(f"Processed {benchmark_name}")

    with open(output_file, 'w') as f:
        json.dump(all_info, f, indent=2)

    print(f"\nExtracted info for {len(all_info)} benchmarks")
    print(f"Output: {output_file}")

if __name__ == "__main__":
    main()
