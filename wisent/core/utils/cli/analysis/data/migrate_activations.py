"""CLI executor for the migrate-activations command."""
import sys

from wisent.core.utils.config_tools.constants import EXIT_CODE_ERROR


def _create_hf_repo():
    """Create the wisent-ai/activations HF dataset repo."""
    from huggingface_hub import HfApi
    from wisent.core.reading.modules.utilities.data.sources.hf.hf_config import HF_REPO_ID, HF_REPO_TYPE
    import os

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("Error: HF_TOKEN environment variable required.")
        sys.exit(1)

    api = HfApi(token=token)
    url = api.create_repo(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        private=False,
        exist_ok=True,
    )
    print(f"Repository ready: {url}")


def execute_migrate_activations(args):
    """Dispatch migrate-activations subcommands."""
    action = getattr(args, "action", None)
    if not action:
        print("Error: specify an action (create-repo, all, single, pair-texts, raw, verify, verify-completeness, consolidate)")
        sys.exit(1)

    if action == "create-repo":
        _create_hf_repo()

    elif action == "all":
        from wisent.core.reading.modules.utilities.data.sources.hf.migration_verify import migrate_all
        migrate_all(
            database_url=args.database_url,
            dry_run=args.dry_run,
            combo_start=args.combo_start,
            combo_end=args.combo_end,
            skip_pair_texts=args.skip_pair_texts,
            reverse=args.reverse,

        )

    elif action == "consolidate":
        from wisent.core.reading.modules.utilities.data.sources.hf.hf_writers import consolidate_index
        consolidate_index(dry_run=args.dry_run)

    elif action == "single":
        from wisent.core.reading.modules.utilities.data.sources.hf.migration import migrate_activation_table
        layers = migrate_activation_table(
            model_name=args.model,
            task_name=args.benchmark,
            strategy=args.strategy,

            database_url=args.database_url,
            dry_run=args.dry_run,
        )
        print(f"\nMigrated {len(layers)} layers: {layers}")

    elif action == "pair-texts":
        from wisent.core.reading.modules.utilities.data.sources.hf.migration import migrate_pair_texts
        count = migrate_pair_texts(
            task_name=args.benchmark,

            database_url=args.database_url,
            dry_run=args.dry_run,
        )
        print(f"\nMigrated {count} pair texts")

    elif action == "raw":
        from wisent.core.reading.modules.utilities.data.sources.hf.migration import migrate_raw_activation_table
        chunks = migrate_raw_activation_table(
            model_name=args.model,
            task_name=args.benchmark,
            prompt_format=args.prompt_format,

            chunk_size=args.chunk_size,
            database_url=args.database_url,
            dry_run=args.dry_run,
        )
        print(f"\nUploaded {chunks} chunks")

    elif action == "verify-completeness":
        from wisent.core.reading.modules.utilities.data.sources.hf.migration_verify import verify_completeness
        ok = verify_completeness(database_url=args.database_url)
        if not ok:
            sys.exit(EXIT_CODE_ERROR)

    elif action == "verify":
        from wisent.core.reading.modules.utilities.data.sources.hf.migration_verify import verify_migration
        ok = verify_migration(
            model_name=args.model,
            task_name=args.benchmark,
            strategy=args.strategy,
            layer=args.layer,
            component=args.extraction_component,
            prompt_format=args.prompt_format,
            database_url=args.database_url,
        )
        if ok:
            print("\nVerification PASSED")
        else:
            print("\nVerification FAILED")
            sys.exit(1)

    else:
        print(f"Error: unknown action '{action}'")
        sys.exit(1)
