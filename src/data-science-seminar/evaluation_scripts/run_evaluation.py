import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

"""
    Runs the full evaluation pipeline for one experiment folder:
        1. avg_date_per_query.py    for the *_results.json in every top_k_X subfolder
        2. significance_test.py     baseline top_k_X vs. comparison top_k_X
                                    (Mann-Whitney-U if unpaired, Wilcoxon signed-rank if paired)
        3. significance_results_to_latex.py on the comparison folder -> values.tex

    Required Arguments:
        --baseline_dir {Folder with top_k_X subfolders, each containing one avg_date_per_query*.jsonl}
        --comparison_dir {Folder with top_k_X subfolders, each containing one *_results.json}
        --samples {'paired' or 'unpaired'}
"""

EXPECTED_TOP_K = [1, 10, 20, 50, 100]

SRC_DIR = Path(__file__).resolve().parent.parent
AVG_SCRIPT = SRC_DIR / "evaluation_scripts" / "avg_date_per_query.py"
SIGNIFICANCE_SCRIPT = SRC_DIR / "evaluation_scripts" / "significance_test.py"
LATEX_SCRIPT = SRC_DIR / "helpers" / "significance_results_to_latex.py"

# significance test used per sample design (value is passed as --test).
TEST_BY_SAMPLES = {
    "unpaired": "mwu",
    "paired": "wilcoxon",
}

AVG_GLOB = "avg_date_per_query*.jsonl"
SIGNIFICANCE_GLOBS = ["mwu_*.json", "wilcoxon_*.json"]
RESULTS_GLOB = "*_results.json"

# --------------------------------------------------------------------------- #
# Shell & Logging helpers
# --------------------------------------------------------------------------- #

def log_info(message: str) -> None:
    print(f"[INFO] {message}", flush=True)


def log_ok(message: str) -> None:
    print(f"[OK] {message}", flush=True)


def fail(message: str) -> None:
    print(f"[ERROR] {message}", file=sys.stderr, flush=True)
    sys.exit(1)


def ask_yes_no(question: str) -> bool:
    while True:
        answer = input(f"{question} ('yes' or 'no'): ").strip().upper()
        if answer in ["Y", "YES"]:
            return True
        if answer in ["N", "NO"]:
            return False
        print("Invalid input. Please type 'yes' or 'no'.")


def run_script(script: Path, args: list[str]) -> str:
    """Runs a pipeline script non-interactively, echoes its output and aborts on failure."""
    log_info(f"Running {script.name} {' '.join(args)}")
    completed = subprocess.run(
        [sys.executable, str(script), *args],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    for line in completed.stdout.splitlines():
        print(f"    {line}")
    if completed.returncode != 0:
        fail(f"{script.name} exited with code {completed.returncode}.")
    return completed.stdout


def find_single(folder: Path, pattern: str) -> Path:
    matches = sorted(folder.glob(pattern))
    if len(matches) != 1:
        fail(
            f"Expected exactly one '{pattern}' in {folder}, "
            f"found {len(matches)}: {[m.name for m in matches]}"
        )
    return matches[0]

# --------------------------------------------------------------------------- #
# Pre-flight checks.
# --------------------------------------------------------------------------- #

def find_topk_folders(folder: Path) -> dict[int, Path]:
    # same matching rule as significance_results_to_latex.py, so both see the same folders.
    topk_folders = {}
    for subfolder in folder.iterdir():
        if not subfolder.is_dir():
            continue
        match = re.search(r"top_k_(\d+)", subfolder.name)
        if match is None:
            continue
        topk = int(match.group(1))
        if topk in topk_folders:
            fail(f"Multiple folders for top_k={topk} in {folder}: {topk_folders[topk].name}, {subfolder.name}")
        topk_folders[topk] = subfolder
    if sorted(topk_folders) != EXPECTED_TOP_K:
        fail(f"top_k folders in {folder} are {sorted(topk_folders)}, expected {EXPECTED_TOP_K}.")
    return topk_folders


def delete_previous_outputs(comparison_dir: Path, comparison_folders: dict[int, Path]) -> None:
    stale = []
    for subfolder in comparison_folders.values():
        stale += sorted(subfolder.glob(AVG_GLOB))
        for pattern in SIGNIFICANCE_GLOBS:
            stale += sorted(subfolder.glob(pattern))
    values_tex = comparison_dir / "values.tex"
    if values_tex.exists():
        stale.append(values_tex)

    if not stale:
        return

    log_info("Found outputs of a previous evaluation run in the comparison folder:")
    for path in stale:
        print(f"    {path}")
    if not ask_yes_no("Delete these files and regenerate them?"):
        log_info("Terminating current run. Deleting previous outputs was denied by user.")
        sys.exit(1)
    for path in stale:
        path.unlink()
    log_ok(f"Deleted {len(stale)} files.")

# --------------------------------------------------------------------------- #
# Validation of significance test output.
# --------------------------------------------------------------------------- #

def find_significance_output(folder: Path, test: str) -> Path:
    return find_single(folder, f"{test}_*.json")


def validate_significance_output(output_path: Path, topk: int, test: str) -> None:
    with output_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)["metadata"]
    if metadata["test"] != test:
        fail(f"{output_path}: test is '{metadata['test']}', expected '{test}'.")
    inputs = metadata["inputs"]
    for group in ["baseline", "comparison"]:
        if inputs[group]["top_k"] != topk:
            fail(f"{output_path}: {group} top_k is {inputs[group]['top_k']}, expected {topk}.")
        if inputs[group]["n_rows_skipped"] != 0:
            fail(f"{output_path}: {inputs[group]['n_rows_skipped']} rows skipped in {group} input.")

# --------------------------------------------------------------------------- #
# Main function.
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run avg_date_per_query -> significance test -> LaTeX export "
            "for all top_k subfolders of one experiment."
        )
    )
    parser.add_argument("--baseline_dir", type=Path, required=True)
    parser.add_argument("--comparison_dir", type=Path, required=True)
    parser.add_argument(
        "--samples", choices=sorted(TEST_BY_SAMPLES), required=True,
        help="'unpaired' -> Mann-Whitney-U test, 'paired' -> Wilcoxon signed-rank test",
    )
    args = parser.parse_args()

    baseline_dir: Path = args.baseline_dir.resolve()
    comparison_dir: Path = args.comparison_dir.resolve()
    test = TEST_BY_SAMPLES[args.samples]
    log_info(f"Samples are {args.samples} -> using significance test '{test}'.")

    for folder in [baseline_dir, comparison_dir]:
        if not folder.is_dir():
            fail(f"Folder doesn't exist: {folder}")
    if baseline_dir == comparison_dir:
        fail("Baseline and comparison folder are identical.")

    # check all inputs before running anything.
    baseline_folders = find_topk_folders(baseline_dir)
    comparison_folders = find_topk_folders(comparison_dir)
    baseline_avg_files = {k: find_single(baseline_folders[k], AVG_GLOB) for k in EXPECTED_TOP_K}
    results_files = {k: find_single(comparison_folders[k], RESULTS_GLOB) for k in EXPECTED_TOP_K}
    log_ok("All input files found.")

    delete_previous_outputs(comparison_dir, comparison_folders)

    for topk in EXPECTED_TOP_K:
        subfolder = comparison_folders[topk]
        log_info(f"===== top_k = {topk} =====")

        run_script(AVG_SCRIPT, ["--results_path", str(results_files[topk])])
        comparison_avg_file = find_single(subfolder, AVG_GLOB)

        significance_output = run_script(
            SIGNIFICANCE_SCRIPT,
            [
                "--baseline_path", str(baseline_avg_files[topk]),
                "--comparison_path", str(comparison_avg_file),
                "--test", test,
            ],
        )
        if "[WARN]" in significance_output:
            fail(f"{SIGNIFICANCE_SCRIPT.name} reported warnings for top_k={topk} (see above).")
        validate_significance_output(find_significance_output(subfolder, test), topk, test)

    log_info("===== LaTeX export =====")
    run_script(LATEX_SCRIPT, [str(comparison_dir)])
    values_tex = comparison_dir / "values.tex"
    if not values_tex.exists():
        fail(f"{LATEX_SCRIPT.name} finished but {values_tex} was not created.")

    log_ok(f"Evaluation finished. LaTeX values written to {values_tex}")


if __name__ == "__main__":
    main()
