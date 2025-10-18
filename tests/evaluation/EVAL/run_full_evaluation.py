#!/usr/bin/env python3
"""
Full evaluation pipeline script.
Runs generate.py followed by eval_scorer.py and logs all output.
"""

import subprocess
import sys
import time
from datetime import datetime


def run_script(script_name, script_dir, log_file):
    """
    Run a Python script and log its output.

    Args:
        script_name: Name of the script to run
        script_dir: Directory where the script is located
        log_file: File object to write logs to

    Returns:
        bool: True if successful, False if failed
    """
    print(f"\n{'='*80}")
    print(f"Starting {script_name}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")

    log_file.write(f"\n{'='*80}\n")
    log_file.write(f"Starting {script_name}\n")
    log_file.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"{'='*80}\n\n")
    log_file.flush()

    start_time = time.time()

    try:
        # Run the script and capture output in real-time
        import os
        script_path = os.path.join(script_dir, script_name)
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=script_dir  # Run from script directory
        )

        # Print and log output in real-time
        for line in process.stdout:
            print(line, end='')
            log_file.write(line)
            log_file.flush()

        # Wait for process to complete
        return_code = process.wait()

        elapsed_time = time.time() - start_time

        if return_code == 0:
            success_msg = f"\n✓ {script_name} completed successfully in {elapsed_time:.2f} seconds\n"
            print(success_msg)
            log_file.write(success_msg)
            log_file.flush()
            return True
        else:
            error_msg = f"\n✗ {script_name} failed with return code {return_code} after {elapsed_time:.2f} seconds\n"
            print(error_msg)
            log_file.write(error_msg)
            log_file.flush()
            return False

    except Exception as e:
        error_msg = f"\n✗ Error running {script_name}: {e}\n"
        print(error_msg)
        log_file.write(error_msg)
        log_file.flush()
        return False


def main():
    """Main execution function."""
    # Create log file with timestamp in EVAL directory
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_filename = os.path.join(script_dir, f"evaluation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    print(f"\n{'#'*80}")
    print(f"# FULL EVALUATION PIPELINE")
    print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Log file: {log_filename}")
    print(f"{'#'*80}\n")

    overall_start = time.time()

    with open(log_filename, 'w', encoding='utf-8') as log_file:
        log_file.write(f"{'#'*80}\n")
        log_file.write(f"# FULL EVALUATION PIPELINE\n")
        log_file.write(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"{'#'*80}\n\n")
        log_file.flush()

        # Get script directory and project root
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Project root is 2 levels up from tests/EVAL/
        project_root = os.path.dirname(os.path.dirname(script_dir))

        # Step 1: Run generate.py from project root
        print("STEP 1/2: Generating steered responses")
        generate_success = run_script("tests/EVAL/generate.py", project_root, log_file)

        if not generate_success:
            error_msg = "\n✗ Pipeline failed: generate.py did not complete successfully\n"
            print(error_msg)
            log_file.write(error_msg)
            sys.exit(1)

        # Step 2: Run eval_scorer.py from project root
        print("\nSTEP 2/2: Evaluating responses with LLM judge")
        eval_success = run_script("tests/EVAL/eval_scorer.py", project_root, log_file)

        if not eval_success:
            error_msg = "\n✗ Pipeline failed: eval_scorer.py did not complete successfully\n"
            print(error_msg)
            log_file.write(error_msg)
            sys.exit(1)

        # Summary
        total_time = time.time() - overall_start
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)

        summary = f"""
{'#'*80}
# PIPELINE COMPLETED SUCCESSFULLY
# Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Total time: {hours}h {minutes}m {seconds}s
# Log file: {log_filename}
#
# Output files generated:
#   - output/happy_output.json (steered responses)
#   - output/evil_output.json (steered responses)
#   - output/happy_scores_txt.json (evaluations - txt format)
#   - output/happy_scores_markdown.json (evaluations - markdown format)
#   - output/happy_scores_json.json (evaluations - json format)
#   - output/evil_scores_txt.json (evaluations - txt format)
#   - output/evil_scores_markdown.json (evaluations - markdown format)
#   - output/evil_scores_json.json (evaluations - json format)
#   - output/happy_stats_*.json (aggregated statistics)
#   - output/evil_stats_*.json (aggregated statistics)
{'#'*80}
"""
        print(summary)
        log_file.write(summary)

    print(f"\nAll logs saved to: {log_filename}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Unexpected error: {e}")
        sys.exit(1)
