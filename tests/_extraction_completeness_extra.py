"""Specific pair extraction and benchmark completeness tests."""
import psycopg2
import pytest
from collections import defaultdict
from _extraction_helpers import (
    DB_CONFIG, MAX_PAIRS_PER_BENCHMARK, get_db_connection,
)

class TestSpecificPairExtraction:
    """Test that specific ContrastivePairs have been extracted for all models."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup database connection."""
        self.conn = get_db_connection()
        self.cur = self.conn.cursor()

        # Get all models
        self.cur.execute("""
            SELECT id, name, "numLayers"
            FROM "Model"
            WHERE "numLayers" IS NOT NULL
        """)
        self.models = {row[1]: {"id": row[0], "layers": row[2]} for row in self.cur.fetchall()}

        # Get all benchmarks
        self.cur.execute("""
            SELECT id, name FROM "ContrastivePairSet"
        """)
        self.benchmarks = [(row[0], row[1]) for row in self.cur.fetchall()]

        yield
        self.cur.close()
        self.conn.close()

    def test_first_n_pairs_per_benchmark_extracted(self):
        """
        Verify that the first N pairs (up to 500) of each benchmark are extracted.
        This checks the extraction order logic.

        OPTIMIZED: Queries per model AND per benchmark.
        """
        issues_by_model = defaultdict(list)

        for model_name, model_info in self.models.items():
            model_id = model_info["id"]
            print(f"\nChecking specific pairs for model: {model_name}")

            for benchmark_id, benchmark_name in self.benchmarks:
                # Get the first MAX_PAIRS_PER_BENCHMARK pair IDs for this benchmark
                self.cur.execute("""
                    SELECT id
                    FROM "ContrastivePair"
                    WHERE "setId" = %s
                    ORDER BY id
                    LIMIT %s
                """, (benchmark_id, MAX_PAIRS_PER_BENCHMARK))

                expected_pair_ids = {row[0] for row in self.cur.fetchall()}

                if not expected_pair_ids:
                    continue

                # Get actually extracted pair IDs for this model/benchmark
                self.cur.execute("""
                    SELECT DISTINCT "contrastivePairId"
                    FROM "Activation"
                    WHERE "modelId" = %s
                      AND "contrastivePairSetId" = %s
                      AND "contrastivePairId" IS NOT NULL
                """, (model_id, benchmark_id))

                actual_pair_ids = {row[0] for row in self.cur.fetchall()}

                # Check for missing pairs
                missing = expected_pair_ids - actual_pair_ids
                if missing:
                    issues_by_model[model_name].append({
                        "benchmark": benchmark_name,
                        "expected_count": len(expected_pair_ids),
                        "actual_count": len(actual_pair_ids),
                        "missing_count": len(missing),
                        "missing_ids": sorted(missing)
                    })

        if any(issues_by_model.values()):
            total_issues = sum(len(v) for v in issues_by_model.values())
            total_missing = sum(sum(i['missing_count'] for i in v) for v in issues_by_model.values())

            report_lines = [
                f"TOTAL: {total_issues} model/benchmark combinations with missing pairs",
                f"Total missing pairs: {total_missing}",
                ""
            ]

            for model_name in sorted(issues_by_model.keys()):
                issues = issues_by_model[model_name]
                if issues:
                    model_missing = sum(i['missing_count'] for i in issues)
                    report_lines.append(f"\n=== {model_name} ({len(issues)} benchmarks, {model_missing} missing pairs) ===")

                    for i in sorted(issues, key=lambda x: -x['missing_count']):
                        report_lines.append(f"  {i['benchmark']}: {i['actual_count']}/{i['expected_count']} ({i['missing_count']} missing)")
                        report_lines.append(f"    Missing IDs: {i['missing_ids'][:20]}{'...' if len(i['missing_ids']) > 20 else ''}")

            assert False, "\n".join(report_lines)
class TestBenchmarkCompleteness:
    """Test that all benchmarks have been processed."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup database connection."""
        self.conn = get_db_connection()
        self.cur = self.conn.cursor()

        # Get all models
        self.cur.execute("""
            SELECT id, name FROM "Model" WHERE "numLayers" IS NOT NULL
        """)
        self.models = {row[1]: row[0] for row in self.cur.fetchall()}

        # Get all benchmarks that have ContrastivePairs
        self.cur.execute("""
            SELECT DISTINCT cps.id, cps.name
            FROM "ContrastivePairSet" cps
            JOIN "ContrastivePair" cp ON cp."setId" = cps.id
        """)
        self.all_benchmarks = {row[1]: row[0] for row in self.cur.fetchall()}

        yield
        self.cur.close()
        self.conn.close()

    def test_all_benchmarks_have_extractions_per_model(self):
        """
        Verify each model has at least one extraction for every benchmark.

        OPTIMIZED: Check one benchmark at a time per model.
        """
        missing_by_model = {}

        for model_name, model_id in self.models.items():
            print(f"\nChecking benchmark coverage for model: {model_name}")
            missing = []

            for benchmark_name, benchmark_id in self.all_benchmarks.items():
                # Check if this model has ANY extraction for this benchmark
                self.cur.execute("""
                    SELECT 1
                    FROM "Activation"
                    WHERE "modelId" = %s
                      AND "contrastivePairSetId" = %s
                    LIMIT 1
                """, (model_id, benchmark_id))

                if self.cur.fetchone() is None:
                    missing.append(benchmark_name)

            if missing:
                missing_by_model[model_name] = sorted(missing)

        if missing_by_model:
            report_lines = [
                f"BENCHMARK COMPLETENESS REPORT",
                f"Total benchmarks with pairs: {len(self.all_benchmarks)}",
                ""
            ]

            for model_name in sorted(missing_by_model.keys()):
                missing = missing_by_model[model_name]
                report_lines.append(f"\n=== {model_name}: missing {len(missing)}/{len(self.all_benchmarks)} benchmarks ===")
                for bench in missing:
                    report_lines.append(f"  - {bench}")

            assert False, "\n".join(report_lines)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
