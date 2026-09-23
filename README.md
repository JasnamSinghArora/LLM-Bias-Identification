# Geometric Quantification of LLM Bias

Code for *Geometric Quantification of Large Language Model Bias and Semantically Constrained Gradient-Based Optimization of Bias-Eliciting Prompts* (Arora and Singhvi). For each test LLM and main field the pipeline:

1. embeds 15,000 counterfactual sentence pairs at hidden layer 16 and builds the difference matrix (Section III-A),
2. measures the MST diversity of the difference vectors (Section III-B),
3. recovers a bias subspace by SVD and selects its dimension k by the spectral-gap rule (Sections III-C, III-E),
4. scores 1,000 random benchmark prompts against the subspace to fix the baseline (Section III-D),
5. optimizes the input embedding of 700 benchmark prompts by gradient ascent in the null space of the semantic Jacobian, with a curvature cap, a first-local-maximum line search and Broyden updates (Section IV-D).

## Requirements

Python 3.11 and the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

Environment variables:

| variable | used by | purpose |
|---|---|---|
| `OPENAI_API_KEY` | `generate_datasets.py` | dataset generation with the generator model |
| `PIPELINE_RUNS_DIR` | `final_pipeline.py`, `results.py` | where run folders are written (default `./runs`) |
| `PIPELINE_RESULTS_DIR` | `results.py` | where tables and figures are written (default `./results`) |
| `PIPELINE_SMOKE=1` | all | tiny dataset sizes and step counts for a quick test |

The Llama test model is gated on Hugging Face, so run `huggingface-cli login` once before the pipeline.

## Reproduce

Step 1: generate the datasets. Each category gets 15,000 counterfactual pairs, 500 k-experiment prompts and 1,000 benchmark prompts under `datasets/<category>/`:

```
python generate_datasets.py
```

Step 2: run the pipeline over every test LLM and main field. Each of the nine runs writes to `runs/<model>/<category>/`:

```
python final_pipeline.py --dry-run   # print the status of every run
python final_pipeline.py             # run everything
```

Step 3: build the paper's tables and figures from the run folders:

```
python results.py
```

Both generation and the pipeline checkpoint after every batch, prompt, or stage. After an interruption, run the same command again and they resume where they stopped.

## Configuration

Everything lives in `constants.py`. The values match Table 5 of the paper:

| constant | paper symbol | value |
|---|---|---|
| `TEST_LLMS` | test LLMs | Llama-3.1-8B-Instruct, Qwen3-8B, Ministral-3-8B-Instruct (bfloat16) |
| `BIAS_CATEGORIES` | main fields and subfields | gender, politics, race with ten subfields each |
| `LAYER` | ℓ | 16 |
| `DATASET_SIZE_CPD` | m | 15,000 pairs |
| `DATASET_SIZE_KEXP` | k-experiment prompts | 500 |
| `DATASET_SIZE_BIAS` | N | 1,000 |
| `OPT_MAX_PROMPTS` | optimised prompts | 700 |
| `OPT_MAX_STEPS` | t_max | 20 |
| `OPT_STEP_MIN` | η_min | 0.1 |
| `OPT_RESTART_EVERY` | K | 10 |
| `OPT_REG` | λ | 1e-6 |
| `OPT_LINE_SEARCH_GROWTH` | γ | 2 |

The paper does not report numeric values for the per-step semantic tolerance τ, the drift tolerance τ̄, the relative-gain threshold ζ or the probe offset h. The code sets τ and τ̄ as fractions of the norm of the semantic representation of the original prompt (`OPT_SEMANTIC_TOL`, `OPT_DRIFT_TOL`), so one setting serves models whose hidden states differ in scale, and sets ζ and h with `OPT_GAIN_MIN` and `OPT_PROBE_OFFSET`.

## Pipeline stages

| stage | what it does | outputs |
|---|---|---|
| 1 | copy the datasets into the run folder | `cpd_dataset.csv`, `kexp_prompts.pt`, `benchmark_prompts.pt` |
| 2 | load the test LLM | |
| 3 | embed every counterfactual pair | `cpd_diff_matrix.pt` |
| 4 | MST diversity of the difference vectors | `cpd_quality.csv` |
| 5 | SVD and candidate subspaces k = 1..20 | `out_k/<k>/bias_subspace_cache.pt` |
| 6 | score the k-experiment prompts under every k | `out_k/<k>/benchmark_scores.csv` |
| 7 | spectral-gap choice of k, k-selection figures | `bias_subspace_cache.pt`, `singular_values.pt`, `k_*.png`, `singular_values.png` |
| 8 | score the benchmark prompts and fix the baseline | `benchmark_scores.csv`, `benchmark_stats.csv`, `output_center_cache.pt`, `bell_curve.png` |
| 9 | semantics-preserving gradient ascent on 700 prompts | `optimized_scores.csv`, `optimization_trajectories.csv`, `run_summary.json` |

`results.py` then reads every run folder and writes Tables 2, 3, 4, 6 and 7 as CSV files and Figures 3, 4, 5, 6, 7 and 12 as three-by-three grids.

## Files

| file | role |
|---|---|
| `final_pipeline.py` | entry point; runs the nine stages for every model and main field |
| `generate_datasets.py` | builds the three datasets per main field with the generator model |
| `results.py` | paper tables and figures from the run folders |
| `constants.py` | all configuration: models, main fields, sizes, layer, optimisation parameters |
| `prompts.py` | generator prompts for the counterfactual pairs and the random prompts |
| `helper.py` | test LLM loading, embeddings, generation, bias score, line search |
| `bias_subspace.py` | difference matrix, SVD, spectral-gap rule, final bias subspace |
| `diversity.py` | MST diversity of a difference matrix |
| `k_experiment.py` | candidate subspaces and k-experiment scoring |
| `k_analysis.py` | k-selection and singular value figures |
| `bias_score_benchmark.py` | benchmark scoring and baseline |
| `bell_curve.py` | benchmark distribution figure and descriptive statistics |
| `optimizer.py` | null-space gradient ascent with curvature cap, line search, Broyden updates |
| `runlog.py` | logging, progress, atomic checkpoints |
