import os
import json

GENDER_SFS = [
    "occupation and the workplace",
    "sports and athletics",
    "education and academics",
    "family and home life",
    "leadership and management",
    "science and technology",
    "arts and entertainment",
    "health and body image",
    "relationships and dating",
    "finance and money",
]
RACE_SFS = [
    "occupation and the workplace",
    "education and academics",
    "criminal justice and policing",
    "housing and neighborhoods",
    "healthcare and medicine",
    "immigration and citizenship",
    "sports and entertainment",
    "finance and lending",
    "food and culture",
    "media representation",
]
POLITICS_SFS = [
    "economic policy",
    "social and cultural issues",
    "environment and energy",
    "immigration and border policy",
    "media and institutions",
    "healthcare and welfare",
    "foreign policy and defense",
    "education policy",
    "technology and privacy",
    "crime and public safety",
]

constants = {
    "API_KEY": os.environ.get("OPENAI_API_KEY", os.environ.get("API_KEY", "")),
    "GEN_MODEL": "gpt-5.6-luna",
    "TEMP": 1.0,  # generator sampling temperature
    "TOKENS": 60,
    # target unique items per dataset
    "DATASET_SIZE_CPD": 15000,
    "DATASET_SIZE_BIAS": 1000,
    "DATASET_SIZE_KEXP": 500,
    # items requested per API call
    "BATCH_SIZE_CPD": 250,
    "BATCH_SIZE_BIAS": 50,
    "BATCH_SIZE_KEXP": 50,
    # cap on API calls per dataset
    "MAX_API_CALLS": 60,
    "MF": os.environ.get("PIPELINE_MF", "gender"),
    "SFS": GENDER_SFS,
    "LAYER": int(os.environ.get("PIPELINE_LAYER", 16)),
    "MODEL_PATH": os.environ.get("PIPELINE_MODEL_PATH", "Qwen/Qwen2.5-0.5B-Instruct"),
    "RUN_DIR": os.environ.get("PIPELINE_RUN_DIR", "."),
    "TEST_LLMS": [
        {"name": "llama-3.1-8b-instruct", "path": "meta-llama/Llama-3.1-8B-Instruct"},
        {"name": "qwen3-8b", "path": "Qwen/Qwen3-8B"},
        {"name": "ministral-3-8b-instruct-2512-bf16", "path": "mistralai/Ministral-3-8B-Instruct-2512-BF16"},
    ],
    # optimisation, Section IV-D of the paper
    "OPT_MAX_PROMPTS": 700,  # benchmark prompts optimised per run
    "OPT_MAX_STEPS": 20,  # t_max
    "OPT_STEP_MIN": 0.1,  # eta_min, step-size termination
    "OPT_GAIN_MIN": 0.0,  # zeta, relative-gain termination
    "OPT_RESTART_EVERY": 10,  # K, exact Jacobian period
    "OPT_REG": 1e-6,  # lambda, projector regularisation
    "OPT_PROBE_OFFSET": 1.0,  # h, above the bf16 noise floor
    "OPT_SEMANTIC_TOL": 0.2,  # tau, fraction of ||M(e0)||
    "OPT_DRIFT_TOL": 0.5,  # tau bar, fraction of ||M(e0)||
    "OPT_LINE_SEARCH_INIT": 0.05,  # first trial step
    "OPT_LINE_SEARCH_GROWTH": 2.0,  # gamma
    "OPT_LINE_SEARCH_EVALS": 8,  # evaluation budget per step
}
constants["BIAS_CATEGORIES"] = [
    {"MF": "gender", "SFS": GENDER_SFS},
    {"MF": "race", "SFS": RACE_SFS},
    {"MF": "politics", "SFS": POLITICS_SFS},
]
if os.environ.get("PIPELINE_SFS"):
    constants["SFS"] = json.loads(os.environ["PIPELINE_SFS"])
if os.environ.get("PIPELINE_SMOKE") == "1":
    constants.update({
        "DATASET_SIZE_CPD": 30, "BATCH_SIZE_CPD": 30,
        "DATASET_SIZE_BIAS": 7, "BATCH_SIZE_BIAS": 7,
        "DATASET_SIZE_KEXP": 7, "BATCH_SIZE_KEXP": 7,
        "OPT_MAX_PROMPTS": 2, "OPT_MAX_STEPS": 3,
    })
# nominal API calls per dataset
constants["BATCHES_CPD"] = max(1, constants["DATASET_SIZE_CPD"] // constants["BATCH_SIZE_CPD"])
constants["BATCHES_BIAS"] = max(1, constants["DATASET_SIZE_BIAS"] // constants["BATCH_SIZE_BIAS"])
constants["BATCHES_KEXP"] = max(1, constants["DATASET_SIZE_KEXP"] // constants["BATCH_SIZE_KEXP"])
