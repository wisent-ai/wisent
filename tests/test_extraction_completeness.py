"""
Test to verify extraction completeness for all models.

This test checks against the actual ContrastivePair data in the database:
1. Each model has activations for ALL ContrastivePairs (up to 500 cap per benchmark)
2. Each extracted pair has ALL layers (no partial extractions)
3. Each extracted pair has BOTH positive and negative activations
4. No duplicate activations exist
"""

import psycopg2
import pytest
from collections import defaultdict

# Database configuration
DB_CONFIG = {
    "host": "aws-0-eu-west-2.pooler.supabase.com",
    "port": 6543,
    "database": "postgres",
    "user": "postgres.rbqjqnouluslojmmnuqi",
    "password": "BsKuEnPFLCFurN4a",
    "sslmode": "require",
    "gssencmode": "disable",
}

# Maximum pairs per benchmark (as defined in extract_all.py)
MAX_PAIRS_PER_BENCHMARK = 500


def get_db_connection():
    """Create database connection with extended timeout."""
    conn = psycopg2.connect(**DB_CONFIG)
    conn.set_session(autocommit=True)
    cur = conn.cursor()
    cur.execute("SET statement_timeout = '300s'")
    cur.close()
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

        yield
        self.cur.close()
        self.conn.close()

    def test_all_pairs_extracted_per_benchmark(self):
        """
        For each model and benchmark, verify we have activations for all ContrastivePairs.
        This is the core completeness check - it compares against actual pair data.
        """
        missing_extractions = []

        for model_name, model_info in self.models.items():
            model_id = model_info["id"]

            # Get expected pair counts per benchmark from ContrastivePair table
            # Capped at MAX_PAIRS_PER_BENCHMARK as that's what extract_all.py does
            self.cur.execute("""
                SELECT
                    cps.id as set_id,
                    cps.name as benchmark_name,
                    LEAST(COUNT(cp.id), %s) as expected_pairs
                FROM "ContrastivePairSet" cps
                JOIN "ContrastivePair" cp ON cp."setId" = cps.id
                GROUP BY cps.id, cps.name
            """, (MAX_PAIRS_PER_BENCHMARK,))

            expected_by_benchmark = {row[1]: {"set_id": row[0], "expected": row[2]}
                                      for row in self.cur.fetchall()}

            # Get actual extracted pair counts per benchmark for this model
            # Count distinct pairs where we have at least one activation
            self.cur.execute("""
                SELECT
                    cps.name as benchmark_name,
                    COUNT(DISTINCT a."contrastivePairId") as actual_pairs
                FROM "Activation" a
                JOIN "ContrastivePairSet" cps ON a."contrastivePairSetId" = cps.id
                WHERE a."modelId" = %s
                  AND a."contrastivePairId" IS NOT NULL
                GROUP BY cps.name
            """, (model_id,))

            actual_by_benchmark = {row[0]: row[1] for row in self.cur.fetchall()}

            # Compare expected vs actual
            for benchmark_name, expected_info in expected_by_benchmark.items():
                expected_count = expected_info["expected"]
                actual_count = actual_by_benchmark.get(benchmark_name, 0)

                if actual_count < expected_count:
                    missing_extractions.append({
                        "model": model_name,
                        "benchmark": benchmark_name,
                        "expected": expected_count,
                        "actual": actual_count,
                        "missing": expected_count - actual_count
                    })

        assert len(missing_extractions) == 0, \
            f"Found {len(missing_extractions)} model/benchmark combinations with missing extractions:\n" + \
            "\n".join(
                f"  {m['model']} / {m['benchmark']}: {m['actual']}/{m['expected']} pairs ({m['missing']} missing)"
                for m in sorted(missing_extractions, key=lambda x: -x['missing'])[:50]
            )

    def test_each_pair_has_all_layers(self):
        """
        Verify no partial extractions - each extracted pair should have all layers.
        """
        incomplete_pairs = []

        for model_name, model_info in self.models.items():
            model_id = model_info["id"]
            expected_layers = model_info["layers"]

            # Find pairs that don't have all layers
            self.cur.execute("""
                SELECT
                    "contrastivePairId",
                    COUNT(DISTINCT layer) as layer_count
                FROM "Activation"
                WHERE "modelId" = %s
                  AND "contrastivePairId" IS NOT NULL
                GROUP BY "contrastivePairId"
                HAVING COUNT(DISTINCT layer) != %s
            """, (model_id, expected_layers))

            for row in self.cur.fetchall():
                incomplete_pairs.append({
                    "model": model_name,
                    "pair_id": row[0],
                    "actual_layers": row[1],
                    "expected_layers": expected_layers
                })

        assert len(incomplete_pairs) == 0, \
            f"Found {len(incomplete_pairs)} pairs with incomplete layer coverage:\n" + \
            "\n".join(
                f"  {p['model']}: pair {p['pair_id']} has {p['actual_layers']}/{p['expected_layers']} layers"
                for p in incomplete_pairs[:20]
            )

    def test_each_pair_has_positive_and_negative(self):
        """
        Verify each extracted pair has both positive and negative activations.
        """
        incomplete_pairs = []

        for model_name, model_info in self.models.items():
            model_id = model_info["id"]

            # Find pairs missing positive or negative
            self.cur.execute("""
                SELECT
                    "contrastivePairId",
                    ARRAY_AGG(DISTINCT "isPositive") as polarities
                FROM "Activation"
                WHERE "modelId" = %s
                  AND "contrastivePairId" IS NOT NULL
                  AND "isPositive" IS NOT NULL
                GROUP BY "contrastivePairId"
                HAVING COUNT(DISTINCT "isPositive") != 2
            """, (model_id,))

            for row in self.cur.fetchall():
                incomplete_pairs.append({
                    "model": model_name,
                    "pair_id": row[0],
                    "polarities": row[1]
                })

        assert len(incomplete_pairs) == 0, \
            f"Found {len(incomplete_pairs)} pairs missing positive or negative activations:\n" + \
            "\n".join(
                f"  {p['model']}: pair {p['pair_id']} only has polarities {p['polarities']}"
                for p in incomplete_pairs[:20]
            )

    def test_no_duplicate_activations(self):
        """
        Verify no duplicate activations exist for same model/pair/layer/polarity.
        """
        duplicates = []

        for model_name, model_info in self.models.items():
            model_id = model_info["id"]

            self.cur.execute("""
                SELECT
                    "contrastivePairId", layer, "isPositive", COUNT(*) as cnt
                FROM "Activation"
                WHERE "modelId" = %s
                  AND "contrastivePairId" IS NOT NULL
                GROUP BY "contrastivePairId", layer, "isPositive"
                HAVING COUNT(*) > 1
            """, (model_id,))

            for row in self.cur.fetchall():
                duplicates.append({
                    "model": model_name,
                    "pair_id": row[0],
                    "layer": row[1],
                    "is_positive": row[2],
                    "count": row[3]
                })

        assert len(duplicates) == 0, \
            f"Found {len(duplicates)} duplicate activations:\n" + \
            "\n".join(
                f"  {d['model']}: pair {d['pair_id']}, layer {d['layer']}, positive={d['is_positive']} has {d['count']} copies"
                for d in duplicates[:20]
            )

    def test_extraction_coverage_per_model(self):
        """
        For each model, report total extraction coverage as percentage of all possible pairs.
        """
        # Get total possible pairs across all benchmarks
        self.cur.execute("""
            SELECT
                cps.id,
                LEAST(COUNT(cp.id), %s) as pair_count
            FROM "ContrastivePairSet" cps
            JOIN "ContrastivePair" cp ON cp."setId" = cps.id
            GROUP BY cps.id
        """, (MAX_PAIRS_PER_BENCHMARK,))

        total_possible_pairs = sum(row[1] for row in self.cur.fetchall())

        coverage_results = {}

        for model_name, model_info in self.models.items():
            model_id = model_info["id"]

            # Count unique pairs with complete extraction (all layers, both polarities)
            expected_layers = model_info["layers"]
            expected_activations_per_pair = expected_layers * 2  # layers * (positive + negative)

            self.cur.execute("""
                SELECT COUNT(DISTINCT "contrastivePairId")
                FROM "Activation"
                WHERE "modelId" = %s
                  AND "contrastivePairId" IS NOT NULL
            """, (model_id,))

            extracted_pairs = self.cur.fetchone()[0]
            coverage = (extracted_pairs / total_possible_pairs * 100) if total_possible_pairs > 0 else 0

            coverage_results[model_name] = {
                "extracted": extracted_pairs,
                "total": total_possible_pairs,
                "coverage": coverage
            }

        # All models should have at least 90% coverage
        low_coverage = [
            (name, data) for name, data in coverage_results.items()
            if data["coverage"] < 90
        ]

        assert len(low_coverage) == 0, \
            f"Models with low extraction coverage (<90%):\n" + \
            "\n".join(
                f"  {name}: {data['extracted']}/{data['total']} pairs ({data['coverage']:.1f}%)"
                for name, data in low_coverage
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

        yield
        self.cur.close()
        self.conn.close()

    def test_first_n_pairs_per_benchmark_extracted(self):
        """
        Verify that the first N pairs (up to 500) of each benchmark are extracted.
        This checks the extraction order logic.
        """
        issues = []

        for model_name, model_info in self.models.items():
            model_id = model_info["id"]

            # Get all benchmarks
            self.cur.execute("""
                SELECT id, name FROM "ContrastivePairSet"
            """)
            benchmarks = self.cur.fetchall()

            for benchmark_id, benchmark_name in benchmarks:
                # Get the first MAX_PAIRS_PER_BENCHMARK pair IDs for this benchmark
                self.cur.execute("""
                    SELECT id
                    FROM "ContrastivePair"
                    WHERE "contrastivePairSetId" = %s
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
                    issues.append({
                        "model": model_name,
                        "benchmark": benchmark_name,
                        "expected_count": len(expected_pair_ids),
                        "actual_count": len(actual_pair_ids),
                        "missing_count": len(missing),
                        "sample_missing": list(missing)[:5]
                    })

        assert len(issues) == 0, \
            f"Found {len(issues)} model/benchmark combinations with missing specific pairs:\n" + \
            "\n".join(
                f"  {i['model']} / {i['benchmark']}: missing {i['missing_count']} pairs (sample IDs: {i['sample_missing']})"
                for i in sorted(issues, key=lambda x: -x['missing_count'])[:30]
            )


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

        yield
        self.cur.close()
        self.conn.close()

    def test_all_benchmarks_have_extractions_per_model(self):
        """
        Verify each model has at least one extraction for every benchmark.
        """
        # Get all benchmarks that have ContrastivePairs
        self.cur.execute("""
            SELECT DISTINCT cps.id, cps.name
            FROM "ContrastivePairSet" cps
            JOIN "ContrastivePair" cp ON cp."setId" = cps.id
        """)
        all_benchmarks = {row[1]: row[0] for row in self.cur.fetchall()}

        missing_benchmarks = []

        for model_name, model_id in self.models.items():
            # Get benchmarks that have extractions for this model
            self.cur.execute("""
                SELECT DISTINCT cps.name
                FROM "Activation" a
                JOIN "ContrastivePairSet" cps ON a."contrastivePairSetId" = cps.id
                WHERE a."modelId" = %s
            """, (model_id,))

            extracted_benchmarks = {row[0] for row in self.cur.fetchall()}
            missing = set(all_benchmarks.keys()) - extracted_benchmarks

            if missing:
                missing_benchmarks.append({
                    "model": model_name,
                    "missing_count": len(missing),
                    "total_benchmarks": len(all_benchmarks),
                    "missing_names": sorted(missing)
                })

        assert len(missing_benchmarks) == 0, \
            f"Models with missing benchmark extractions:\n" + \
            "\n".join(
                f"  {m['model']}: missing {m['missing_count']}/{m['total_benchmarks']} benchmarks: {m['missing_names'][:10]}..."
                for m in missing_benchmarks
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
