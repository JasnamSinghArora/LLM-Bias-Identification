import os
import sys
import csv
import json
import time
import shutil
import subprocess


DATASET_FILES = ["cpd_dataset.csv", "kexp_prompts.pt", "benchmark_prompts.pt"]
N_STAGES = 9


def count_csv_rows(path):
    # data rows counted with the csv module, so a quoted newline inside a cell is not miscounted
    if not os.path.exists(path):
        return 0
    with open(path, newline="") as f:
        return max(0, sum(1 for _ in csv.reader(f)) - 1)


def run_status(run_dir, constants):
    """One line describing how far a run got, derived from the files it left behind."""
    from runlog import checkpoint_len

    def p(name):
        return os.path.join(run_dir, name)

    if os.path.exists(p("run_summary.json")):
        return "complete"
    if not os.path.isdir(run_dir):
        return "not started"
    if os.path.exists(p("optimized_scores.csv")):
        total = count_csv_rows(p("benchmark_scores.csv"))
        if constants["OPT_MAX_PROMPTS"] is not None:
            total = min(total, constants["OPT_MAX_PROMPTS"])
        return f"stage 9/{N_STAGES} optimization: {count_csv_rows(p('optimized_scores.csv'))}/{total} prompts done"
    if os.path.exists(p("output_center_cache.pt")):
        return f"stage 8/{N_STAGES} benchmark done; optimization not started"
    if os.path.exists(p("benchmark_progress.pt")):
        return f"stage 8/{N_STAGES} benchmark generation: {checkpoint_len(p('benchmark_progress.pt'))}/{constants['DATASET_SIZE_BIAS']} prompts done"
    if os.path.exists(p("bias_subspace_cache.pt")):
        return f"stage 7/{N_STAGES} final subspace chosen; benchmark not started"
    if os.path.exists(p("out_k_scores_done.marker")):
        return f"stage 6/{N_STAGES} k scores done; final subspace not chosen"
    if os.path.exists(p("kexp_progress.pt")):
        return f"stage 6/{N_STAGES} k-experiment generation: {checkpoint_len(p('kexp_progress.pt'))}/{constants['DATASET_SIZE_KEXP']} prompts done"
    if os.path.exists(os.path.join(run_dir, "out_k", "20", "bias_subspace_cache.pt")):  # 20 = k_experiment.K_MAX
        return f"stage 5/{N_STAGES} candidate subspaces built; k-experiment not started"
    if os.path.exists(p("cpd_diff_matrix.pt")):
        return f"stage 3/{N_STAGES} CPD embeddings done; SVD not started"
    if os.path.exists(p("cpd_diff_progress.pt")):
        return f"stage 3/{N_STAGES} CPD embeddings: {checkpoint_len(p('cpd_diff_progress.pt'))}/{count_csv_rows(p('cpd_dataset.csv'))} pairs done"
    if os.path.exists(p("cpd_dataset.csv")):
        return f"stage 1/{N_STAGES} datasets copied; CPD embeddings not started"
    return "not started"


def driver(dry_run=False):
    from constants import constants
    from runlog import log, fmt_duration
    here = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(here, "datasets")
    # run folders live next to the code unless PIPELINE_RUNS_DIR points somewhere persistent (e.g. Google Drive on Colab)
    runs_root = os.environ.get("PIPELINE_RUNS_DIR", os.path.join(here, "runs"))
    log(f"run folders: {runs_root}")

    def datasets_ready(mf):
        return all(os.path.exists(os.path.join(datasets_dir, mf, name)) for name in DATASET_FILES)

    # build the model x category run list and show where every run stands
    runs = []
    for model in constants["TEST_LLMS"]:
        for cat in constants["BIAS_CATEGORIES"]:
            runs.append((model, cat))
    log(f"===== {len(runs)} runs: {len(constants['TEST_LLMS'])} models x {len(constants['BIAS_CATEGORIES'])} categories =====")
    for i, (model, cat) in enumerate(runs, 1):
        run_dir = os.path.join(runs_root, model["name"], cat["MF"])
        log(f"  run {i}/{len(runs)}  {model['name']:<36} {cat['MF']:<9} {run_status(run_dir, constants)}")
    missing = [cat["MF"] for cat in constants["BIAS_CATEGORIES"] if not datasets_ready(cat["MF"])]
    if missing:
        log(f"datasets not generated yet for {missing}; those runs are skipped until generate_datasets.py has produced them")
    if dry_run:
        return

    failures, skipped = [], []
    for i, (model, cat) in enumerate(runs, 1):
        run_dir = os.path.join(runs_root, model["name"], cat["MF"])
        label = f"run {i}/{len(runs)}: model={model['name']} category={cat['MF']}"
        if os.path.exists(os.path.join(run_dir, "run_summary.json")):
            log(f"===== {label} already complete, skipping =====")
            continue
        if not datasets_ready(cat["MF"]):
            log(f"===== {label} skipped: datasets for {cat['MF']} not generated =====")
            skipped.append((model["name"], cat["MF"]))
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
        log(f"===== {label} -> {run_dir} =====")
        t0 = time.time()
        try:
            result = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker"], env=env, cwd=here)
        except KeyboardInterrupt:
            log("stopped by Ctrl+C; every finished checkpoint is saved in the run folder, run the same command again to resume")
            sys.exit(130)
        if result.returncode == 130:
            log("worker was interrupted; run the same command again to resume")
            sys.exit(130)
        if result.returncode != 0:
            failures.append((model["name"], cat["MF"], result.returncode))
            log(f"{label} failed (exit {result.returncode}); continuing with the next run")
        else:
            log(f"{label} finished in {fmt_duration(time.time() - t0)}")

    # report how all the runs ended
    if skipped:
        log(f"{len(skipped)} run(s) skipped for missing datasets: {skipped}")
    if failures:
        log(f"finished with {len(failures)} failed run(s): {failures}")
        sys.exit(1)
    log("all runs finished")


def purge_trajectories(traj_csv, done_rows):
    """Drop trajectory rows left by a prompt that was interrupted before its result row was written."""
    from runlog import log
    with open(traj_csv, newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    kept = [r for r in body if r and int(r[0]) < done_rows]
    if len(kept) != len(body):
        tmp = traj_csv + ".tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(kept)
        os.replace(tmp, traj_csv)
        log(f"dropped {len(body) - len(kept)} trajectory rows from the prompt that was interrupted last time")


def worker():

    # read this run's configuration and paths
    import torch
    from constants import constants
    from runlog import log, set_log_file, Stage, Progress, fmt_duration
    run_dir = constants["RUN_DIR"]
    os.makedirs(run_dir, exist_ok=True)

    def p(name):
        return os.path.join(run_dir, name)

    set_log_file(p("pipeline.log"))
    log(f"run start | model={constants['MODEL_PATH']} | category={constants['MF']} | layer={constants['LAYER']}")
    log(f"run folder: {run_dir} (full log in pipeline.log)")

    with Stage(1, N_STAGES, "copy the pre-generated datasets into the run folder") as st:
        datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", constants["MF"])
        copied = []
        for name in DATASET_FILES:
            if not os.path.exists(p(name)):
                src = os.path.join(datasets_dir, name)
                if not os.path.exists(src):
                    log(f"missing {src} - run generate_datasets.py first")
                    sys.exit(1)
                shutil.copy(src, p(name))
                copied.append(name)
        if copied:
            log(f"copied {copied}")
        else:
            st.skip("already in the run folder")
        log(f"datasets: {count_csv_rows(p('cpd_dataset.csv'))} counterfactual pairs, "
            f"{len(torch.load(p('kexp_prompts.pt')))} k-experiment prompts, {len(torch.load(p('benchmark_prompts.pt')))} benchmark prompts")

    with Stage(2, N_STAGES, f"load the test LLM {constants['MODEL_PATH']}"):
        import helper
        from helper import create_embedding, create_output_from_emb, get_bias_score_from_emb
        log(f"model loaded on {helper.model.device} as {helper.model.dtype}")
        import bias_subspace
        import entropy
        import diversity
        import k_experiment
        import k_analysis
        import bias_score_benchmark
        import bell_curve
        import main as optimizer

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

    with Stage(3, N_STAGES, "embed every counterfactual pair and build the difference matrix") as st:
        # the 246 MB difference matrix is not kept in git; after stages 4, 5 and 7 have consumed it, it is no longer needed
        consumers_done = all(os.path.exists(x) for x in [
            p("cpd_quality.csv"), p("cpd_metrics_by_sf.csv"), bias_subspace.CACHE_PATH,
            os.path.join(k_experiment.OUT_DIR, str(k_experiment.K_MAX), "bias_subspace_cache.pt")])
        if os.path.exists(bias_subspace.DIFF_MATRIX_PATH):
            st.skip(f"{os.path.basename(bias_subspace.DIFF_MATRIX_PATH)} exists")
        elif consumers_done:
            st.skip("difference matrix not on this machine, but every stage that needs it is already done")
        else:
            bias_subspace.get_CPD()

    with Stage(4, N_STAGES, "dataset quality: entropy and MST diversity, overall and per SF") as st:
        quality_csv = p("cpd_quality.csv")
        if os.path.exists(quality_csv):
            with open(quality_csv, newline="") as f:
                vals = {r["metric"]: float(r["value"]) for r in csv.DictReader(f)}
            H, D = vals["normalized_entropy"], vals["mst_diversity"]
            log("overall metrics already computed (cpd_quality.csv)")
        else:
            H = entropy.main(bias_subspace.CPD_CSV_PATH)
            D = float(diversity.main(bias_subspace.DIFF_MATRIX_PATH, max_samples=constants["DIVERSITY_MAX_SAMPLES"]))
            tmp = quality_csv + ".tmp"
            with open(tmp, "w", newline="") as f:
                csv.writer(f).writerows([["metric", "value"], ["normalized_entropy", float(H)], ["mst_diversity", D]])
            os.replace(tmp, quality_csv)
        log(f"entropy={H:.4f} mst_diversity={D:.4f}")
        summary["cpd_entropy"] = float(H)
        summary["cpd_mst_diversity"] = D

        by_sf_csv = p("cpd_metrics_by_sf.csv")
        if os.path.exists(by_sf_csv):
            log("per-SF metrics already computed (cpd_metrics_by_sf.csv)")
        else:
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
            rows_out = []
            for sf in sorted(groups):
                idx = [i for i, t in groups[sf]]
                terms = [t for i, t in groups[sf]]
                H_sf = entropy.normalized_entropy(terms)
                D_sf = float(diversity.mst_diversity(diff[idx]))
                rows_out.append([constants["MF"], sf, len(idx), H_sf, D_sf])
                log(f"  SF={sf}: {len(idx)} pairs, entropy={H_sf:.4f}, mst_diversity={D_sf:.4f}")
            tmp = by_sf_csv + ".tmp"
            with open(tmp, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["MF", "SF", "n_pairs", "entropy", "mst_diversity"])
                w.writerows(rows_out)
            os.replace(tmp, by_sf_csv)

    with Stage(5, N_STAGES, f"SVD of the difference matrix and candidate subspaces k=1..{k_experiment.K_MAX}") as st:
        if os.path.exists(os.path.join(k_experiment.OUT_DIR, str(k_experiment.K_MAX), "bias_subspace_cache.pt")):
            st.skip("out_k/<k>/bias_subspace_cache.pt exist")
        else:
            k_experiment.create_subspaces()

    with Stage(6, N_STAGES, "generate outputs for the k-experiment prompts and score them under every k") as st:
        k_scores_marker = p("out_k_scores_done.marker")
        if os.path.exists(k_scores_marker):
            st.skip("out_k_scores_done.marker exists")
        else:
            k_experiment.create_bias_score_data()
            open(k_scores_marker, "w").close()

    with Stage(7, N_STAGES, "k-selection graphs and the final bias subspace") as st:
        k_analysis.run_all()
        log("saved k_mean_bias_score.png, k_marginal_gain.png, k_bell_curves.png, k_bell_curves_individual.png")
        if os.path.exists(bias_subspace.CACHE_PATH):
            log("final subspace already chosen (bias_subspace_cache.pt)")
        else:
            bias_subspace.create_bias_subspace()
        final_subspace, _ = bias_subspace.load_bias_subspace()
        summary["chosen_k"] = int(final_subspace.shape[0])
        log(f"chosen_k={summary['chosen_k']} - check it against the k graphs in {run_dir}")

    with Stage(8, N_STAGES, "generate and score the benchmark prompts, freeze the baseline") as st:
        if os.path.exists(bias_score_benchmark.CENTER_CACHE_PATH):
            st.skip("output_center_cache.pt exists")
        else:
            bias_score_benchmark.set_bias_score_benchmark()
        _c = torch.load(bias_score_benchmark.CENTER_CACHE_PATH)
        summary["baseline_bias_score"] = float(_c["baseline"])
        log(f"baseline bias score = {summary['baseline_bias_score']:.4f}")
        if os.path.exists(p("bell_curve.png")):
            log("bell_curve.png already drawn")
        else:
            bell_curve.main()  # benchmark distribution figure for the paper
            log("saved bell_curve.png")

    with Stage(9, N_STAGES, "optimize every benchmark prompt with gradient ascent") as st:
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
            done_rows = count_csv_rows(opt_csv)
        else:
            done_rows = 0
            with open(opt_csv, "w", newline="") as f:
                csv.writer(f).writerow(["prompt", "original_score", "optimized_score", "delta", "optimized_output"])
        traj_csv = p("optimization_trajectories.csv")
        if os.path.exists(traj_csv):
            purge_trajectories(traj_csv, done_rows)
        else:
            with open(traj_csv, "w", newline="") as f:
                csv.writer(f).writerow(["prompt_index", "step", "bias_score", "epsilon"])

        if done_rows >= len(rows):
            st.skip(f"all {len(rows)} prompts already optimized")
        else:
            log(f"each prompt runs up to {constants['OPT_MAX_STEPS']} ascent steps; a prompt is saved the moment it finishes")
            prog = Progress("optimization", len(rows), done=done_rows, every=1)
            for idx, (prompt, orig_score) in enumerate(rows):
                if idx < done_rows:  # resume: rows already optimized in a previous run
                    continue
                t0 = time.time()
                log(f"prompt {idx + 1}/{len(rows)} | original score {orig_score:.4f} | {prompt[:90]!r}")
                emb, mask = create_embedding(prompt)
                opt_emb = optimizer.optimize_with_gradient_ascent(
                    emb, mask, traj_path=traj_csv, prompt_idx=idx, log_prefix=f"  prompt {idx + 1}/{len(rows)} ")
                out_vec, out_text = create_output_from_emb(opt_emb, mask, with_grad=False, return_text=True)
                opt_score = get_bias_score_from_emb([out_vec.float().cpu()], optimizer.bias_subspace, optimizer.output_center).item()
                with open(opt_csv, "a", newline="") as f:
                    csv.writer(f).writerow([prompt, orig_score, opt_score, opt_score - orig_score, out_text])
                del emb, mask, opt_emb, out_vec
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                prog.step(extra=f"last prompt {orig_score:.3f} -> {opt_score:.3f} ({opt_score - orig_score:+.3f}) in {fmt_duration(time.time() - t0)}")

    # aggregate the results and save the run summary
    with open(opt_csv, newline="") as f:
        recs = list(csv.DictReader(f))
    if recs:
        summary["n_optimized"] = len(recs)
        summary["original_mean_score_over_optimized_set"] = sum(float(r["original_score"]) for r in recs) / len(recs)
        summary["optimized_mean_score"] = sum(float(r["optimized_score"]) for r in recs) / len(recs)
        summary["mean_delta"] = summary["optimized_mean_score"] - summary["original_mean_score_over_optimized_set"]
        log(f"optimized {len(recs)} prompts: mean score {summary['original_mean_score_over_optimized_set']:.4f} -> "
            f"{summary['optimized_mean_score']:.4f} (mean delta {summary['mean_delta']:+.4f})")

    tmp = p("run_summary.json.tmp")
    with open(tmp, "w") as f:
        json.dump(summary, f, indent=2)
    os.replace(tmp, p("run_summary.json"))
    log(f"run complete -> {p('run_summary.json')}")


# choose worker, dry run, or full driver mode
if __name__ == "__main__":
    if "--worker" in sys.argv:
        try:
            worker()
        except KeyboardInterrupt:
            from runlog import log, Stage
            where = f"stage {Stage.current.idx}/{Stage.current.total} ({Stage.current.name})" if Stage.current else "startup"
            log(f"interrupted by Ctrl+C during {where}; every finished checkpoint is saved, run again to resume")
            sys.exit(130)
    elif "--dry-run" in sys.argv:
        driver(dry_run=True)
    else:
        driver()
