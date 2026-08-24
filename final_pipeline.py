# ADDED: NEW FILE — every line in this file is new.
#
# Final pipeline (steps 3-26 per run, steps 27-28 as the outer loops).
# One command runs every (test LLM x bias category) combination:
#
#     python3.11 final_pipeline.py              # full run: all test LLMs x all bias categories
#     python3.11 final_pipeline.py --dry-run    # list the planned runs without executing anything
#     PIPELINE_SMOKE=1 python3.11 final_pipeline.py   # tiny end-to-end test with miniature dataset sizes
#
# Why subprocesses: helper.py loads the test LLM at import time and prompts.py bakes the
# MF/SF/SSF values into its prompt strings at import time, so each (model x category)
# combination must run in a fresh Python process. The driver below sets the configuration
# through PIPELINE_* environment variables (read by constants.py) and launches this same
# file with --worker. Every artifact for a run is saved under runs/<model>/<category>/,
# and every phase is checkpointed: rerunning after a crash resumes where it stopped.

import os
import sys
import csv
import json
import subprocess


def driver(dry_run=False):
    from constants import constants
    here = os.path.dirname(os.path.abspath(__file__))
    runs = []
    for model in constants["TEST_LLMS"]:          # step 28: loop over test LLMs
        for cat in constants["BIAS_CATEGORIES"]:  # step 27: loop over bias categories
            runs.append((model, cat))
    print(f"[pipeline] {len(runs)} runs planned "
          f"({len(constants['TEST_LLMS'])} test LLMs x {len(constants['BIAS_CATEGORIES'])} bias categories)", flush=True)
    failures = []
    for model, cat in runs:
        run_dir = os.path.join(here, "runs", model["name"], cat["MF"])
        print(f"[pipeline] ===== model={model['name']} category={cat['MF']} -> {run_dir} =====", flush=True)
        if dry_run:
            continue
        os.makedirs(run_dir, exist_ok=True)
        env = os.environ.copy()
        env["PIPELINE_MODEL_PATH"] = model["path"]
        env["PIPELINE_LAYER"] = str(model.get("layer", constants["LAYER"]))
        env["PIPELINE_MF"] = cat["MF"]
        env["PIPELINE_SF"] = cat["SF"]
        env["PIPELINE_SSF"] = json.dumps(cat["SSF"])
        env["PIPELINE_RUN_DIR"] = run_dir
        result = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker"], env=env, cwd=here)
        if result.returncode != 0:
            failures.append((model["name"], cat["MF"], result.returncode))
            print(f"[pipeline] !!! run failed (exit {result.returncode}); continuing with the next run", flush=True)
    if failures:
        print(f"[pipeline] finished with {len(failures)} failed run(s): {failures}", flush=True)
        sys.exit(1)
    print("[pipeline] all runs finished", flush=True)


def worker():
    # Heavy imports happen here, inside the already-configured subprocess:
    # importing helper loads the test LLM selected via PIPELINE_MODEL_PATH.
    import torch
    from constants import constants
    run_dir = constants["RUN_DIR"]
    os.makedirs(run_dir, exist_ok=True)

    def p(name):
        return os.path.join(run_dir, name)

    import bias_subspace
    import entropy
    import diversity
    import k_experiment
    import k_analysis
    import bias_score_benchmark
    import bell_curve
    import main as optimizer
    from helper import create_embedding, create_output_from_emb, get_bias_score_from_emb

    summary = {
        "model_path": constants["MODEL_PATH"],
        "MF": constants["MF"],
        "SF": constants["SF"],
        "layer": constants["LAYER"],
        "X_cpd_pairs": constants["DATASET_SIZE_CPD"],
        "Y_k_prompts": constants["DATASET_SIZE_KEXP"],
        "Z_benchmark_prompts": constants["DATASET_SIZE_BIAS"],
    }

    # ---- Steps 3-7: CPD generation + quality gates (regenerate if the gates fail) ----
    regen = 0
    while True:
        if not os.path.exists(bias_subspace.DIFF_MATRIX_PATH):
            print("[pipeline] steps 3-5: generating counterfactual pairs and the difference matrix", flush=True)
            bias_subspace.get_CPD()
        H = entropy.main(bias_subspace.CPD_CSV_PATH)
        D = float(diversity.main(bias_subspace.DIFF_MATRIX_PATH, max_samples=constants["DIVERSITY_MAX_SAMPLES"]))
        print(f"[pipeline] step 6: entropy={H:.4f} mst_diversity={D:.4f}", flush=True)
        if H >= constants["ENTROPY_MIN"] and D >= constants["DIVERSITY_MIN"]:
            break
        regen += 1
        if regen > constants["MAX_CPD_REGENS"]:
            print("[pipeline] step 7: quality gates still failing; continuing with the best available dataset", flush=True)
            break
        print(f"[pipeline] step 7: quality gates failed, regenerating the CPD dataset (attempt {regen})", flush=True)
        for stale in [bias_subspace.CPD_CSV_PATH, bias_subspace.DIFF_MATRIX_PATH, bias_subspace.DIFF_PROGRESS_PATH]:
            if os.path.exists(stale):
                os.remove(stale)
    summary["cpd_entropy"] = float(H)
    summary["cpd_mst_diversity"] = D
    with open(p("cpd_quality.csv"), "w", newline="") as f:
        csv.writer(f).writerows([["metric", "value"], ["normalized_entropy", float(H)], ["mst_diversity", D]])

    # ---- Steps 8-9: SVD once, candidate subspaces k=1..20 ----
    if not os.path.exists(os.path.join(k_experiment.OUT_DIR, str(k_experiment.K_MAX), "bias_subspace_cache.pt")):
        print("[pipeline] steps 8-9: SVD + candidate subspaces k=1..20", flush=True)
        k_experiment.create_subspaces()

    # ---- Steps 10-11: Y prompts scored under every candidate k ----
    k_scores_marker = p("out_k_scores_done.marker")
    if not os.path.exists(k_scores_marker):
        print("[pipeline] steps 10-11: generating Y prompts and scoring them under k=1..20", flush=True)
        k_experiment.create_bias_score_data()
        open(k_scores_marker, "w").close()

    # ---- Step 12: k-selection graphs ----
    print("[pipeline] step 12: k-selection graphs", flush=True)
    k_analysis.run_all()

    # ---- Steps 13-15: spectral-gap k, freeze the final subspace ----
    if not os.path.exists(bias_subspace.CACHE_PATH):
        print("[pipeline] steps 13-15: choosing k via the spectral gap and freezing the final subspace", flush=True)
        bias_subspace.create_bias_subspace()
    final_subspace, _ = bias_subspace.load_bias_subspace()
    summary["chosen_k"] = int(final_subspace.shape[0])
    print(f"[pipeline] step 14: check the k graphs in {run_dir} against chosen_k={summary['chosen_k']}", flush=True)

    # ---- Steps 16-20: Z benchmark prompts, baseline score, frozen center ----
    if not os.path.exists(bias_score_benchmark.CENTER_CACHE_PATH):
        print("[pipeline] steps 16-20: generating the Z benchmark prompts and the baseline", flush=True)
        bias_score_benchmark.set_bias_score_benchmark()
    _c = torch.load(bias_score_benchmark.CENTER_CACHE_PATH)
    summary["baseline_bias_score"] = float(_c["baseline"])
    bell_curve.main()  # benchmark distribution figure for the paper

    # ---- Steps 21-26: optimize the exact same Z prompts, per-prompt comparison, save everything ----
    print("[pipeline] steps 21-26: optimizing the Z prompts", flush=True)
    optimizer.init_optimizer_state()
    rows = []
    with open(bias_score_benchmark.BENCH_CSV_PATH, newline="") as f:
        for r in csv.DictReader(f):
            rows.append((r["ssf"], r["prompt"], float(r["score"])))
    if constants["OPT_MAX_PROMPTS"] is not None:
        rows = rows[: constants["OPT_MAX_PROMPTS"]]

    opt_csv = p("optimized_scores.csv")
    if os.path.exists(opt_csv):
        with open(opt_csv, newline="") as f:
            done_rows = max(0, sum(1 for _ in f) - 1)
    else:
        done_rows = 0
        with open(opt_csv, "w", newline="") as f:
            csv.writer(f).writerow(["ssf", "prompt", "original_score", "optimized_score", "delta", "optimized_output"])
    traj_csv = p("optimization_trajectories.csv")
    if not os.path.exists(traj_csv):
        with open(traj_csv, "w", newline="") as f:
            csv.writer(f).writerow(["prompt_index", "step", "bias_score", "epsilon"])

    for idx, (ssf, prompt, orig_score) in enumerate(rows):
        if idx < done_rows:  # resume: rows already optimized in a previous run
            continue
        emb, mask = create_embedding(prompt)
        opt_emb = optimizer.optimize_with_gradient_ascent(emb, mask, traj_path=traj_csv, prompt_idx=idx)
        out_vec, out_text = create_output_from_emb(opt_emb, mask, with_grad=False, return_text=True)
        opt_score = get_bias_score_from_emb([out_vec.float().cpu()], optimizer.bias_subspace, optimizer.output_center).item()
        with open(opt_csv, "a", newline="") as f:
            csv.writer(f).writerow([ssf, prompt, orig_score, opt_score, opt_score - orig_score, out_text])
        del emb, mask, opt_emb, out_vec
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if (idx + 1) % 25 == 0:
            print(f"[pipeline] optimization: {idx + 1}/{len(rows)} prompts done", flush=True)

    with open(opt_csv, newline="") as f:
        recs = list(csv.DictReader(f))
    if recs:
        summary["n_optimized"] = len(recs)
        summary["original_mean_score_over_optimized_set"] = sum(float(r["original_score"]) for r in recs) / len(recs)
        summary["optimized_mean_score"] = sum(float(r["optimized_score"]) for r in recs) / len(recs)
        summary["mean_delta"] = summary["optimized_mean_score"] - summary["original_mean_score_over_optimized_set"]

    with open(p("run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[pipeline] run complete -> {p('run_summary.json')}", flush=True)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        worker()
    elif "--dry-run" in sys.argv:
        driver(dry_run=True)
    else:
        driver()
