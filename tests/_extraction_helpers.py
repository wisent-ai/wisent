"""Shared DB config and extraction integrity tests."""
import psycopg2
import pytest
from collections import defaultdict

DB_CONFIG = {
    "host": "db.rbqjqnouluslojmmnuqi.supabase.co",
    "port": 5432,
    "database": "postgres",
    "user": "postgres",
    "password": "REDACTED_DB_PASSWORD",
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


class TestExtractionIntegrity:
    """Tests for polarity, duplicates, and coverage."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.conn = get_db_connection()
        self.cur = self.conn.cursor()
        self.cur.execute("""SELECT id, name, \"numLayers\" FROM \"Model\" WHERE \"numLayers\" IS NOT NULL""")
        self.models = {row[1]: {"id": row[0], "layers": row[2]} for row in self.cur.fetchall()}
        self.cur.execute("""
            SELECT cps.id, cps.name, LEAST(COUNT(cp.id), %s)
            FROM \"ContrastivePairSet\" cps
            JOIN \"ContrastivePair\" cp ON cp.\"setId\" = cps.id
            GROUP BY cps.id, cps.name
        """, (MAX_PAIRS_PER_BENCHMARK,))
        self.benchmarks = {row[1]: {"id": row[0], "expected_pairs": row[2]} for row in self.cur.fetchall()}
        yield
        self.cur.close()
        self.conn.close()

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
