"""
Test to verify extraction completeness for all models.

Checks against actual ContrastivePair data in the database:
1. Each model has activations for ALL ContrastivePairs (up to 500 cap)
2. Each extracted pair has ALL layers (no partial extractions)

OPTIMIZED: All queries filter by both modelId AND contrastivePairSetId.
"""

import pytest
from collections import defaultdict

from _extraction_helpers import (
    DB_CONFIG, MAX_PAIRS_PER_BENCHMARK, get_db_connection,
    TestExtractionIntegrity,
)
from _extraction_completeness_extra import (
    TestSpecificPairExtraction, TestBenchmarkCompleteness,
)

class TestExtractionCompleteness:
    """Tests to verify extraction data integrity against source data."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup database connection and fetch model info."""
        self.conn = get_db_connection()
        self.cur = self.conn.cursor()

        # Get all models with their layer counts from the Model table
        self.cur.execute("""
            SELECT id, name, "numLayers"
            FROM "Model"
            WHERE "numLayers" IS NOT NULL
        """)
        self.models = {row[1]: {"id": row[0], "layers": row[2]} for row in self.cur.fetchall()}

        # Get all benchmarks with their pair counts (capped at 500)
        self.cur.execute("""
            SELECT
                cps.id as set_id,
                cps.name as benchmark_name,
                LEAST(COUNT(cp.id), %s) as expected_pairs
            FROM "ContrastivePairSet" cps
            JOIN "ContrastivePair" cp ON cp."setId" = cps.id
            GROUP BY cps.id, cps.name
        """, (MAX_PAIRS_PER_BENCHMARK,))
        self.benchmarks = {row[1]: {"id": row[0], "expected_pairs": row[2]} for row in self.cur.fetchall()}

        yield
        self.cur.close()
        self.conn.close()

    def test_all_pairs_extracted_per_benchmark(self):
        """
        For each model and benchmark, verify we have activations for all ContrastivePairs.
        This is the core completeness check - it compares against actual pair data.

        OPTIMIZED: Queries per model AND per benchmark to avoid full table scans.
        """
        missing_extractions = []
        missing_by_model = defaultdict(list)

        for model_name, model_info in self.models.items():
            model_id = model_info["id"]
            print(f"\nChecking model: {model_name}")

            for benchmark_name, benchmark_info in self.benchmarks.items():
                benchmark_id = benchmark_info["id"]
                expected_count = benchmark_info["expected_pairs"]

                # Query ONLY this model + benchmark combination
                self.cur.execute("""
                    SELECT COUNT(DISTINCT "contrastivePairId")
                    FROM "Activation"
                    WHERE "modelId" = %s
                      AND "contrastivePairSetId" = %s
                      AND "contrastivePairId" IS NOT NULL
                """, (model_id, benchmark_id))

                actual_count = self.cur.fetchone()[0]

                if actual_count < expected_count:
                    issue = {
                        "model": model_name,
                        "benchmark": benchmark_name,
                        "expected": expected_count,
                        "actual": actual_count,
                        "missing": expected_count - actual_count
                    }
                    missing_extractions.append(issue)
                    missing_by_model[model_name].append(issue)

        if missing_extractions:
            report_lines = [
                f"TOTAL: {len(missing_extractions)} model/benchmark combinations with missing extractions",
                f"Total missing pairs: {sum(m['missing'] for m in missing_extractions)}",
                ""
            ]

            for model_name in sorted(missing_by_model.keys()):
                issues = missing_by_model[model_name]
                total_missing = sum(i['missing'] for i in issues)
                report_lines.append(f"\n=== {model_name} ({len(issues)} benchmarks, {total_missing} total missing pairs) ===")
                for i in sorted(issues, key=lambda x: -x['missing']):
                    report_lines.append(f"  {i['benchmark']}: {i['actual']}/{i['expected']} pairs ({i['missing']} missing)")

            assert False, "\n".join(report_lines)

    def test_each_pair_has_all_layers(self):
        """
        Verify no partial extractions - each extracted pair should have all layers.

        OPTIMIZED: Queries per model AND per benchmark.
        """
        incomplete_by_model = defaultdict(list)

        for model_name, model_info in self.models.items():
            model_id = model_info["id"]
            expected_layers = model_info["layers"]
            print(f"\nChecking layers for model: {model_name} (expects {expected_layers} layers)")

            for benchmark_name, benchmark_info in self.benchmarks.items():
                benchmark_id = benchmark_info["id"]

                # Find pairs in this benchmark that don't have all layers
                self.cur.execute("""
                    SELECT
                        "contrastivePairId",
                        COUNT(DISTINCT layer) as layer_count
                    FROM "Activation"
                    WHERE "modelId" = %s
                      AND "contrastivePairSetId" = %s
                      AND "contrastivePairId" IS NOT NULL
                    GROUP BY "contrastivePairId"
                    HAVING COUNT(DISTINCT layer) != %s
                """, (model_id, benchmark_id, expected_layers))

                for row in self.cur.fetchall():
                    incomplete_by_model[model_name].append({
                        "benchmark": benchmark_name,
                        "pair_id": row[0],
                        "actual_layers": row[1],
                        "expected_layers": expected_layers
                    })

        if any(incomplete_by_model.values()):
            total_incomplete = sum(len(v) for v in incomplete_by_model.values())
            report_lines = [
                f"TOTAL: {total_incomplete} pairs with incomplete layer coverage",
                ""
            ]

            for model_name in sorted(incomplete_by_model.keys()):
                pairs = incomplete_by_model[model_name]
                if pairs:
                    expected = pairs[0]['expected_layers']
                    report_lines.append(f"\n=== {model_name} (expected {expected} layers, {len(pairs)} incomplete pairs) ===")

                    # Group by actual layer count
                    by_layer_count = defaultdict(list)
                    for p in pairs:
                        by_layer_count[p['actual_layers']].append((p['benchmark'], p['pair_id']))

                    for layer_count in sorted(by_layer_count.keys()):
                        entries = by_layer_count[layer_count]
                        report_lines.append(f"  {layer_count} layers: {len(entries)} pairs")
                        for bench, pair_id in entries[:5]:
                            report_lines.append(f"    - {bench}: pair {pair_id}")
                        if len(entries) > 5:
                            report_lines.append(f"    ... and {len(entries) - 5} more")

            assert False, "\n".join(report_lines)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
