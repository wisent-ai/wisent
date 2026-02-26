"""Compare Supabase combos vs HuggingFace migrated markers."""
import os
import psycopg2

# Use direct connection (port 5432) instead of pooler to avoid statement timeout
DB_URL = (
    "postgresql://postgres:BsKuEnPFLCFurN4a"
    "@db.rbqjqnouluslojmmnuqi.supabase.co:5432/postgres"
    "?sslmode=require"
)


def main():
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    # Get models
    print("--- Models in Supabase ---")
    cur.execute('SELECT id, "huggingFaceId" FROM "Model" ORDER BY id')
    models = cur.fetchall()
    for mid, hfid in models:
        print(f"  id={mid}: {hfid}")

    # Get pair sets
    print(f"\n--- ContrastivePairSets in Supabase ---")
    cur.execute('SELECT id, name FROM "ContrastivePairSet" ORDER BY id')
    pair_sets = cur.fetchall()
    print(f"  Total: {len(pair_sets)}")
    for pid, name in pair_sets:
        print(f"  {pid}: {name}")

    # Check if Activation table has an estimated row count
    cur.execute("""
        SELECT reltuples::bigint
        FROM pg_class
        WHERE relname = 'Activation'
    """)
    est = cur.fetchone()
    if est:
        print(f"\nEstimated Activation rows (pg_class): {est[0]:,}")

    cur.close()
    conn.close()

    # Now analyze HF data
    print("\n\n========== HuggingFace Analysis ==========")
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN", ""))
    info = api.dataset_info("wisent-ai/activations")

    hf_markers = {}
    hf_activations = {}
    for s in (info.siblings or []):
        fn = s.rfilename
        if fn.startswith("markers/") and fn.endswith(".json"):
            parts = fn[len("markers/"):-len(".json")].split("/")
            if len(parts) >= 3:
                model = parts[0]
                strategy = parts[-1]
                benchmark = "/".join(parts[1:-1])
                hf_markers.setdefault(model, {}).setdefault(
                    benchmark, set()
                ).add(strategy)
        elif fn.startswith("activations/") and fn.endswith(".safetensors"):
            parts = fn.split("/")
            if len(parts) >= 4:
                model = parts[1]
                hf_activations.setdefault(model, set()).add(fn)

    print(f"\nMarkers: {sum(len(s) for bm in hf_markers.values() for s in bm.values())} total")
    print(f"Activation files: {sum(len(v) for v in hf_activations.values())} total")

    # Expected: 4 models x 211 pair sets x 7 strategies = 5908 combos
    expected_strategies = [
        "chat_first", "chat_last", "chat_max_norm",
        "chat_mean", "chat_weighted", "mc_balanced", "role_play"
    ]
    print(f"\nExpected strategies: {expected_strategies}")
    print(f"Models: {len(models)}, PairSets: {len(pair_sets)}")
    max_possible = len(models) * len(pair_sets) * len(expected_strategies)
    print(f"Max possible combos: {max_possible}")

    # Per-model marker coverage
    for model_id, hfid in models:
        safe = hfid.replace("/", "__")
        model_markers = hf_markers.get(safe, {})
        total_markers = sum(len(v) for v in model_markers.values())
        covered_bms = len(model_markers)
        print(f"\n  {hfid}: {total_markers} markers, {covered_bms} benchmarks")

        # Find pair sets NOT covered
        missing_bms = []
        for _, ps_name in pair_sets:
            if ps_name not in model_markers:
                missing_bms.append(ps_name)
        if missing_bms:
            print(f"    Missing benchmarks ({len(missing_bms)}):")
            for bm in missing_bms[:20]:
                print(f"      {bm}")
            if len(missing_bms) > 20:
                print(f"      ... and {len(missing_bms) - 20} more")

        # Find partial strategy coverage
        partial = []
        for bm, strats in model_markers.items():
            missing_strats = set(expected_strategies) - strats
            extra_strats = strats - set(expected_strategies)
            if missing_strats:
                partial.append((bm, missing_strats, extra_strats))
        if partial:
            print(f"    Partial coverage ({len(partial)} benchmarks):")
            for bm, miss, extra in partial[:10]:
                print(f"      {bm}: missing {miss}")
            if len(partial) > 10:
                print(f"      ... and {len(partial) - 10} more")


if __name__ == "__main__":
    main()
