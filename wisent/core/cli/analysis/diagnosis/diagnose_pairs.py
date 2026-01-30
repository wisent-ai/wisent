"""Diagnose contrastive pairs command execution logic."""

import sys
import json
import os


def execute_diagnose_pairs(args):
    """Execute the diagnose-pairs command - analyze existing contrastive pairs."""
    print(f"\n🔍 Diagnosing contrastive pairs")
    print(f"   Input file: {args.pairs_file}")

    try:
        # 1. Load pairs from JSON
        if not os.path.exists(args.pairs_file):
            raise FileNotFoundError(f"Pairs file not found: {args.pairs_file}")

        with open(args.pairs_file, 'r') as f:
            data = json.load(f)

        # Handle both formats: dict with 'pairs' key or direct list
        if isinstance(data, dict):
            pairs_list = data.get('pairs', [])
            trait_description = data.get('trait_description', 'N/A')
            trait_label = data.get('trait_label', 'N/A')
            num_pairs = data.get('num_pairs', len(pairs_list))
            requested = data.get('requested', 'N/A')
            kept_after_dedupe = data.get('kept_after_dedupe', 'N/A')
            diversity = data.get('diversity', None)
        else:
            pairs_list = data
            trait_description = 'N/A'
            trait_label = 'N/A'
            num_pairs = len(pairs_list)
            requested = 'N/A'
            kept_after_dedupe = 'N/A'
            diversity = None

        # 2. Basic statistics
        print(f"\n📊 Basic Statistics:")
        print(f"   Trait description: {trait_description}")
        print(f"   Trait label: {trait_label}")
        print(f"   Total pairs: {num_pairs}")
        if requested != 'N/A':
            print(f"   Requested: {requested}")
        if kept_after_dedupe != 'N/A':
            print(f"   Kept after dedupe: {kept_after_dedupe}")
            if requested != 'N/A' and isinstance(requested, int) and isinstance(kept_after_dedupe, int):
                retention = (kept_after_dedupe / requested) * 100
                print(f"   Retention rate: {retention:.1f}%")

        # 3. Diversity metrics
        if diversity:
            print(f"\n🎨 Diversity Metrics:")
            print(f"   Unique unigrams: {diversity.get('unique_unigrams', 'N/A'):.3f}")
            print(f"   Unique bigrams: {diversity.get('unique_bigrams', 'N/A'):.3f}")
            print(f"   Avg Jaccard similarity: {diversity.get('avg_jaccard', 'N/A'):.3f}")

        # 4. Content analysis
        if len(pairs_list) > 0:
            print(f"\n📝 Content Analysis:")

            # Calculate lengths
            prompt_lengths = []
            positive_lengths = []
            negative_lengths = []

            for pair in pairs_list:
                prompt_lengths.append(len(pair.get('prompt', '')))
                pos_resp = pair.get('positive_response', {})
                neg_resp = pair.get('negative_response', {})
                positive_lengths.append(len(pos_resp.get('model_response', '')))
                negative_lengths.append(len(neg_resp.get('model_response', '')))

            # Stats
            print(f"   Prompt lengths:")
            print(f"     Min: {min(prompt_lengths)} chars")
            print(f"     Max: {max(prompt_lengths)} chars")
            print(f"     Avg: {sum(prompt_lengths)/len(prompt_lengths):.1f} chars")

            print(f"   Positive response lengths:")
            print(f"     Min: {min(positive_lengths)} chars")
            print(f"     Max: {max(positive_lengths)} chars")
            print(f"     Avg: {sum(positive_lengths)/len(positive_lengths):.1f} chars")

            print(f"   Negative response lengths:")
            print(f"     Min: {min(negative_lengths)} chars")
            print(f"     Max: {max(negative_lengths)} chars")
            print(f"     Avg: {sum(negative_lengths)/len(negative_lengths):.1f} chars")

            # Quality warnings
            short_prompts = sum(1 for l in prompt_lengths if l < 10)
            short_positives = sum(1 for l in positive_lengths if l < 10)
            short_negatives = sum(1 for l in negative_lengths if l < 10)

            if short_prompts > 0 or short_positives > 0 or short_negatives > 0:
                print(f"\n⚠️  Quality Warnings:")
                if short_prompts > 0:
                    print(f"   {short_prompts} prompts are very short (<10 chars)")
                if short_positives > 0:
                    print(f"   {short_positives} positive responses are very short (<10 chars)")
                if short_negatives > 0:
                    print(f"   {short_negatives} negative responses are very short (<10 chars)")

        # 5. Schema validation
        print(f"\n✅ Schema Validation:")
        required_fields = ['prompt', 'positive_response', 'negative_response']
        missing_fields = []

        for i, pair in enumerate(pairs_list):
            for field in required_fields:
                if field not in pair:
                    missing_fields.append(f"Pair {i}: missing '{field}'")

            # Check nested fields
            if 'positive_response' in pair:
                if 'model_response' not in pair['positive_response']:
                    missing_fields.append(f"Pair {i}: positive_response missing 'model_response'")

            if 'negative_response' in pair:
                if 'model_response' not in pair['negative_response']:
                    missing_fields.append(f"Pair {i}: negative_response missing 'model_response'")

        if len(missing_fields) == 0:
            print(f"   All pairs have required fields")
        else:
            print(f"   ❌ Found {len(missing_fields)} schema issues:")
            for issue in missing_fields[:5]:  # Show first 5
                print(f"      {issue}")
            if len(missing_fields) > 5:
                print(f"      ... and {len(missing_fields) - 5} more")

        # 6. Show sample if requested
        if args.show_sample and len(pairs_list) > 0:
            print(f"\n📄 Sample Pair (first one):")
            sample = pairs_list[0]
            print(f"\n   Prompt:")
            print(f"   {sample.get('prompt', 'N/A')[:200]}{'...' if len(sample.get('prompt', '')) > 200 else ''}")
            print(f"\n   Positive Response:")
            pos_resp = sample.get('positive_response', {}).get('model_response', 'N/A')
            print(f"   {pos_resp[:200]}{'...' if len(pos_resp) > 200 else ''}")
            print(f"\n   Negative Response:")
            neg_resp = sample.get('negative_response', {}).get('model_response', 'N/A')
            print(f"   {neg_resp[:200]}{'...' if len(neg_resp) > 200 else ''}")

        print(f"\n✅ Diagnosis complete!\n")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
