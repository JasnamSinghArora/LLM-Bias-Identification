import os
import sys
import csv
import json
import subprocess


def driver(dry_run=False):
    from constants import constants
    here = os.path.dirname(os.path.abspath(__file__))

    # build the model x category run list
    runs = []
    for model in constants["TEST_LLMS"]:          
        for cat in constants["BIAS_CATEGORIES"]:  
            runs.append((model, cat))
    failures = []
    for model, cat in runs:
        run_dir = os.path.join(here, "runs", model["name"], cat["MF"])
        print(f"===== model={model['name']} category={cat['MF']} -> {run_dir} =====", flush=True)
        if dry_run:
            continue

        # configure and launch this run's subprocess
        os.makedirs(run_dir, exist_ok=True)
        env = os.environ.copy()
        env["PIPELINE_MODEL_PATH"] = model["path"]
        env["PIPELINE_LAYER"] = str(model.get("layer", constants["LAYER"]))
        env["PIPELINE_MF"] = cat["MF"]
        env["PIPELINE_SFS"] = json.dumps(cat["SFS"])
        env["PIPELINE_SSF"] = json.dumps(cat["SSF"])
        env["PIPELINE_SSF_BY_SF"] = json.dumps(cat["SSF_BY_SF"])
        env["PIPELINE_RUN_DIR"] = run_dir
        result = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker"], env=env, cwd=here)
        if result.returncode != 0:
            failures.append((model["name"], cat["MF"], result.returncode))
            print(f"run failed (exit {result.returncode}); continuing with the next run", flush=True)
    # report how all the runs ended
    if failures:
        print(f"finished with {len(failures)} failed run(s): {failures}", flush=True)
        sys.exit(1)
    print("all runs finished", flush=True)


def worker():

    # read this run's configuration and paths
    import torch
    from constants import constants
    run_dir = constants["RUN_DIR"]
    os.makedirs(run_dir, exist_ok=True)

    def p(name):
        return os.path.join(run_dir, name)

    # import the experiment modules and load the test LLM
    import bias_subspace
    import entropy
    import diversity
    import k_experiment
    import k_analysis
    import bias_score_benchmark
    import bell_curve
    import main as optimizer
    from helper import create_embedding, create_output_from_emb, get_bias_score_from_emb

    # start this run's summary record
    summary = {
        "model_path": constants["MODEL_PATH"],
        "MF": constants["MF"],
        "SFS": constants["SFS"],
        "layer": constants["LAYER"],
        "X_cpd_pairs": constants["DATASET_SIZE_CPD"],
        "Y_k_prompts": constants["DATASET_SIZE_KEXP"],
        "Z_benchmark_prompts": constants["DATASET_SIZE_BIAS"],
    }

    # generate counterfactual pairs and difference vectors
    if not os.path.exists(bias_subspace.DIFF_MATRIX_PATH):
        print("generating counterfactual pairs and the difference matrix", flush=True)
        bias_subspace.get_CPD()

    # measure dataset entropy and MST diversity
    H = entropy.main(bias_subspace.CPD_CSV_PATH)
    D = float(diversity.main(bias_subspace.DIFF_MATRIX_PATH, max_samples=constants["DIVERSITY_MAX_SAMPLES"]))
    print(f"entropy={H:.4f} mst_diversity={D:.4f}", flush=True)
    summary["cpd_entropy"] = float(H)
    summary["cpd_mst_diversity"] = D
    with open(p("cpd_quality.csv"), "w", newline="") as f:
        csv.writer(f).writerows([["metric", "value"], ["normalized_entropy", float(H)], ["mst_diversity", D]])

    # per SF entropy and diversity for scatterplots
    term_to_sf = {}
    for sf, terms in constants["SSF_BY_SF"].items():
        for t in terms:
            term_to_sf.setdefault(t, sf)
    with open(bias_subspace.CPD_CSV_PATH, newline="") as f:
        sentences = [r["sentence_A"] for r in csv.DictReader(f)]
    diff = torch.load(bias_subspace.DIFF_MATRIX_PATH).float().cpu()
    groups = {}
    for i, s in enumerate(sentences):
        t = entropy.label(s)
        groups.setdefault(term_to_sf.get(t, "other"), []).append((i, t))
    with open(p("cpd_metrics_by_sf.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["MF", "SF", "n_pairs", "entropy", "mst_diversity"])
        for sf in sorted(groups):
            idx = [i for i, t in groups[sf]]
            terms = [t for i, t in groups[sf]]
            H_sf = entropy.normalized_entropy(terms)
            D_sf = float(diversity.mst_diversity(diff[idx]))
            w.writerow([constants["MF"], sf, len(idx), H_sf, D_sf])

    # build candidate bias subspaces for k=1..20
    if not os.path.exists(os.path.join(k_experiment.OUT_DIR, str(k_experiment.K_MAX), "bias_subspace_cache.pt")):
        print("SVD + candidate subspaces k=1..20", flush=True)
        k_experiment.create_subspaces()

    # score the Y prompts under every candidate subspace
    k_scores_marker = p("out_k_scores_done.marker")
    if not os.path.exists(k_scores_marker):
        print("generating Y prompts and scoring them under k=1..20", flush=True)
        k_experiment.create_bias_score_data()
        open(k_scores_marker, "w").close()

    # draw the k selection graphs
    print("creating the k-selection graphs", flush=True)
    k_analysis.run_all()

    # choose k and freeze the final subspace
    if not os.path.exists(bias_subspace.CACHE_PATH):
        print("choosing k via the spectral gap and freezing the final subspace", flush=True)
        bias_subspace.create_bias_subspace()
    final_subspace, _ = bias_subspace.load_bias_subspace()
    summary["chosen_k"] = int(final_subspace.shape[0])
    print(f"check the k graphs in {run_dir} against chosen_k={summary['chosen_k']}", flush=True)

    # benchmark the Z prompts and freeze the baseline
    if not os.path.exists(bias_score_benchmark.CENTER_CACHE_PATH):
        print("generating the Z benchmark prompts and the baseline", flush=True)
        bias_score_benchmark.set_bias_score_benchmark()
    _c = torch.load(bias_score_benchmark.CENTER_CACHE_PATH)
    summary["baseline_bias_score"] = float(_c["baseline"])
    bell_curve.main()  # benchmark distribution figure for the paper

    # load optimization state and the starting prompts
    print("optimizing the Z prompts", flush=True)
    optimizer.init_optimizer_state()
    rows = []
    with open(bias_score_benchmark.BENCH_CSV_PATH, newline="") as f:
        for r in csv.DictReader(f):
            rows.append((r["prompt"], float(r["score"])))
    if constants["OPT_MAX_PROMPTS"] is not None:
        rows = rows[: constants["OPT_MAX_PROMPTS"]]

    # set up the output files with resume support
    opt_csv = p("optimized_scores.csv")
    if os.path.exists(opt_csv):
        with open(opt_csv, newline="") as f:
            done_rows = max(0, sum(1 for _ in f) - 1)
    else:
        done_rows = 0
        with open(opt_csv, "w", newline="") as f:
            csv.writer(f).writerow(["prompt", "original_score", "optimized_score", "delta", "optimized_output"])
    traj_csv = p("optimization_trajectories.csv")
    if not os.path.exists(traj_csv):
        with open(traj_csv, "w", newline="") as f:
            csv.writer(f).writerow(["prompt_index", "step", "bias_score", "epsilon"])

    # optimize each prompt and record its scores
    for idx, (prompt, orig_score) in enumerate(rows):
        if idx < done_rows:  # resume: rows already optimized in a previous run
            continue
        emb, mask = create_embedding(prompt)
        opt_emb = optimizer.optimize_with_gradient_ascent(emb, mask, traj_path=traj_csv, prompt_idx=idx)
        out_vec, out_text = create_output_from_emb(opt_emb, mask, with_grad=False, return_text=True)
        opt_score = get_bias_score_from_emb([out_vec.float().cpu()], optimizer.bias_subspace, optimizer.output_center).item()
        with open(opt_csv, "a", newline="") as f:
            csv.writer(f).writerow([prompt, orig_score, opt_score, opt_score - orig_score, out_text])
        del emb, mask, opt_emb, out_vec
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if (idx + 1) % 25 == 0:
            print(f"optimization: {idx + 1}/{len(rows)} prompts done", flush=True)

    # aggregate the results and save the run summary
    with open(opt_csv, newline="") as f:
        recs = list(csv.DictReader(f))
    if recs:
        summary["n_optimized"] = len(recs)
        summary["original_mean_score_over_optimized_set"] = sum(float(r["original_score"]) for r in recs) / len(recs)
        summary["optimized_mean_score"] = sum(float(r["optimized_score"]) for r in recs) / len(recs)
        summary["mean_delta"] = summary["optimized_mean_score"] - summary["original_mean_score_over_optimized_set"]

    with open(p("run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"run complete -> {p('run_summary.json')}", flush=True)


# choose worker, dry run, or full driver mode
if __name__ == "__main__":
    if "--worker" in sys.argv:
        worker()
    elif "--dry-run" in sys.argv:
        driver(dry_run=True)
    else:
        driver()
