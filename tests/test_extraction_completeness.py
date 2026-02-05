"""
Test to verify extraction completeness for all models.

This test checks against the actual ContrastivePair data in the database:
1. Each model has activations for ALL ContrastivePairs (up to 500 cap per benchmark)
2. Each extracted pair has ALL layers (no partial extractions)
3. Each extracted pair has BOTH positive and negative activations
4. No duplicate activations exist

OPTIMIZED VERSION: All queries filter by both modelId AND contrastivePairSetId
to avoid full table scans on the Activation table.
"""

import psycopg2
import pytest
from collections import defaultdict

# Database configuration - direct connection (bypasses pooler timeout limits)
DB_CONFIG = {
    "host": "db.rbqjqnouluslojmmnuqi.supabase.co",
    "port": 5432,
    "database": "postgres",
    "user": "postgres",
    "password": "BsKuEnPFLCFurN4a",
    "sslmode": "require",
    "gssencmode": "disable",
}

# Maximum pairs per benchmark (as defined in extract_all.py)
MAX_PAIRS_PER_BENCHMARK = 500


def get_db_connection():
    """Create database connection with no statement timeout."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.close()
    conn.commit()
    return conn


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

    def test_each_pair_has_positive_and_negative(self):
        """
        Verify each extracted pair has both positive and negative activations.

        OPTIMIZED: Queries per model AND per benchmark.
        """
        incomplete_by_model = defaultdict(list)

        for model_name, model_info in self.models.items():
            model_id = model_info["id"]
            print(f"\nChecking polarities for model: {model_name}")

            for benchmark_name, benchmark_info in self.benchmarks.items():
                benchmark_id = benchmark_info["id"]

                # Find pairs missing positive or negative in this benchmark
                self.cur.execute("""
                    SELECT
                        "contrastivePairId",
                        ARRAY_AGG(DISTINCT "isPositive") as polarities
                    FROM "Activation"
                    WHERE "modelId" = %s
                      AND "contrastivePairSetId" = %s
                      AND "contrastivePairId" IS NOT NULL
                      AND "isPositive" IS NOT NULL
                    GROUP BY "contrastivePairId"
                    HAVING COUNT(DISTINCT "isPositive") != 2
                """, (model_id, benchmark_id))

                for row in self.cur.fetchall():
                    incomplete_by_model[model_name].append({
                        "benchmark": benchmark_name,
                        "pair_id": row[0],
                        "polarities": row[1]
                    })

        if any(incomplete_by_model.values()):
            total_incomplete = sum(len(v) for v in incomplete_by_model.values())
            report_lines = [
                f"TOTAL: {total_incomplete} pairs missing positive or negative activations",
                ""
            ]

            for model_name in sorted(incomplete_by_model.keys()):
                pairs = incomplete_by_model[model_name]
                if pairs:
                    report_lines.append(f"\n=== {model_name} ({len(pairs)} incomplete pairs) ===")

                    # Group by what's missing
                    only_positive = [p for p in pairs if p['polarities'] == [True]]
                    only_negative = [p for p in pairs if p['polarities'] == [False]]
                    other = [p for p in pairs if p['polarities'] not in [[True], [False]]]

                    if only_positive:
                        report_lines.append(f"  Only positive (missing negative): {len(only_positive)} pairs")
                        for p in only_positive[:5]:
                            report_lines.append(f"    - {p['benchmark']}: pair {p['pair_id']}")
                        if len(only_positive) > 5:
                            report_lines.append(f"    ... and {len(only_positive) - 5} more")
                    if only_negative:
                        report_lines.append(f"  Only negative (missing positive): {len(only_negative)} pairs")
                        for p in only_negative[:5]:
                            report_lines.append(f"    - {p['benchmark']}: pair {p['pair_id']}")
                        if len(only_negative) > 5:
                            report_lines.append(f"    ... and {len(only_negative) - 5} more")
                    if other:
                        report_lines.append(f"  Other (unexpected polarities): {len(other)} pairs")
                        for p in other[:5]:
                            report_lines.append(f"    - {p['benchmark']}: pair {p['pair_id']}, polarities={p['polarities']}")

            assert False, "\n".join(report_lines)

    def test_no_duplicate_activations(self):
        """
        Verify no duplicate activations exist for same model/pair/layer/polarity.

        OPTIMIZED: Queries per model AND per benchmark.
        """
        duplicates_by_model = defaultdict(list)

        for model_name, model_info in self.models.items():
            model_id = model_info["id"]
            print(f"\nChecking duplicates for model: {model_name}")

            for benchmark_name, benchmark_info in self.benchmarks.items():
                benchmark_id = benchmark_info["id"]

                self.cur.execute("""
                    SELECT
                        "contrastivePairId", layer, "isPositive", COUNT(*) as cnt
                    FROM "Activation"
                    WHERE "modelId" = %s
                      AND "contrastivePairSetId" = %s
                      AND "contrastivePairId" IS NOT NULL
                    GROUP BY "contrastivePairId", layer, "isPositive"
                    HAVING COUNT(*) > 1
                """, (model_id, benchmark_id))

                for row in self.cur.fetchall():
                    duplicates_by_model[model_name].append({
                        "benchmark": benchmark_name,
                        "pair_id": row[0],
                        "layer": row[1],
                        "is_positive": row[2],
                        "count": row[3]
                    })

        if any(duplicates_by_model.values()):
            total_duplicates = sum(len(v) for v in duplicates_by_model.values())
            total_extra = sum(sum(d['count'] - 1 for d in v) for v in duplicates_by_model.values())
            report_lines = [
                f"TOTAL: {total_duplicates} duplicate activation entries ({total_extra} extra rows)",
                ""
            ]

            for model_name in sorted(duplicates_by_model.keys()):
                dups = duplicates_by_model[model_name]
                if dups:
                    model_extra = sum(d['count'] - 1 for d in dups)
                    report_lines.append(f"\n=== {model_name} ({len(dups)} duplicated entries, {model_extra} extra rows) ===")

                    # Group by benchmark
                    by_benchmark = defaultdict(list)
                    for d in dups:
                        by_benchmark[d['benchmark']].append(d)

                    for bench in sorted(by_benchmark.keys()):
                        entries = by_benchmark[bench]
                        report_lines.append(f"  {bench}: {len(entries)} duplicates")
                        for e in entries[:3]:
                            report_lines.append(f"    pair={e['pair_id']}, layer={e['layer']}, positive={e['is_positive']}: {e['count']} copies")
                        if len(entries) > 3:
                            report_lines.append(f"    ... and {len(entries) - 3} more")

            assert False, "\n".join(report_lines)

    def test_extraction_coverage_per_model(self):
        """
        For each model, report total extraction coverage as percentage of all possible pairs.

        OPTIMIZED: Queries per model AND per benchmark.
        """
        total_possible_pairs = sum(b["expected_pairs"] for b in self.benchmarks.values())
        coverage_results = {}

        for model_name, model_info in self.models.items():
            model_id = model_info["id"]
            print(f"\nCalculating coverage for model: {model_name}")

            total_extracted = 0
            for benchmark_name, benchmark_info in self.benchmarks.items():
                benchmark_id = benchmark_info["id"]

                # Count unique pairs extracted for this benchmark
                self.cur.execute("""
                    SELECT COUNT(DISTINCT "contrastivePairId")
                    FROM "Activation"
                    WHERE "modelId" = %s
                      AND "contrastivePairSetId" = %s
                      AND "contrastivePairId" IS NOT NULL
                """, (model_id, benchmark_id))

                total_extracted += self.cur.fetchone()[0]

            coverage = (total_extracted / total_possible_pairs * 100) if total_possible_pairs > 0 else 0

            coverage_results[model_name] = {
                "extracted": total_extracted,
                "total": total_possible_pairs,
                "coverage": coverage
            }

        # All models should have at least 90% coverage
        low_coverage = [
            (name, data) for name, data in coverage_results.items()
            if data["coverage"] < 90
        ]

        if low_coverage:
            report_lines = [
                f"EXTRACTION COVERAGE REPORT",
                f"Total benchmarks: {len(self.benchmarks)}",
                f"Total possible pairs (capped at {MAX_PAIRS_PER_BENCHMARK}/benchmark): {total_possible_pairs}",
                ""
            ]

            for name, data in sorted(coverage_results.items(), key=lambda x: x[1]['coverage']):
                status = "LOW" if data['coverage'] < 90 else "OK"
                report_lines.append(f"  [{status}] {name}: {data['extracted']}/{data['total']} pairs ({data['coverage']:.1f}%)")

            report_lines.append("")
            report_lines.append(f"Models below 90% threshold: {len(low_coverage)}")

            assert False, "\n".join(report_lines)


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
