import os
import json  # ADDED: needed to parse the SSF list override passed via environment variable by final_pipeline.py
constants = {
    "API_KEY" : os.environ.get("OPENAI_API_KEY", os.environ.get("API_KEY", "")),  # EDITED: generation LLM is now GPT-5.6 Sol (OpenAI), so read OPENAI_API_KEY with API_KEY as fallback; .get() so importing without a key set no longer crashes
    "GEN_MODEL" : "gpt-5.6-sol",  # ADDED: generation LLM (GPT-5.6 Sol) used for every dataset-generation API call
    "DATASET_SIZE_CPD" : 100000,  # EDITED: X = 100,000 counterfactual pairs (was 300)
    "BATCHES_CPD" : 800,  # EDITED: 800 batches x 125 pairs = 100,000 (was 2)
    "DATASET_SIZE_BIAS" : 50000,  # EDITED: Z = 50,000 benchmark/optimization prompts (was 600)
    "BATCHES_BIAS" : 1000,  # EDITED: 1000 batches x 50 prompts = 50,000 (was 10)
    "DATASET_SIZE_KEXP" : 20000,  # ADDED: Y = 20,000 prompts used to evaluate the k=1..20 candidate subspaces
    "BATCHES_KEXP" : 400,  # ADDED: 400 batches x 50 prompts = 20,000
    "TEMP" : 1.0,
    "TOKENS" : 60,
    "MF" : os.environ.get("PIPELINE_MF", "gender"),  # EDITED: overridable per pipeline run via env var (set by final_pipeline.py)
    "SF" : os.environ.get("PIPELINE_SF", "occupation"),  # EDITED: overridable per pipeline run via env var (set by final_pipeline.py)
    "SSF" : [
        "software engineer",
        "nurse",
        "elementary school teacher",
        "firefighter",
        "chief executive officer",
        "attorney",
        "construction worker",
        "surgeon",
        "flight attendant",
        "auto mechanic",
        "professional chef",
        "research scientist",
        "police officer",
        "accountant",
        "social worker",
    ],
    "LAYER" : int(os.environ.get("PIPELINE_LAYER", 16)),  # EDITED: overridable per pipeline run via env var (set by final_pipeline.py)
    "MODEL_PATH" : os.environ.get("PIPELINE_MODEL_PATH", "/Users/jasnamsingharora/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"),  # EDITED: overridable per pipeline run via env var (set by final_pipeline.py)
    "RUN_DIR" : os.environ.get("PIPELINE_RUN_DIR", "."),  # ADDED: per-(test LLM x bias category) output directory; every artifact is read/written under it
    "TEST_LLMS" : [  # ADDED: the test LLMs the final pipeline loops over (pipeline step 28)
        {"name": "llama-3.1-8b-instruct", "path": "meta-llama/Llama-3.1-8B-Instruct"},  # ADDED: Test LLM 1
        {"name": "qwen3-8b", "path": "Qwen/Qwen3-8B"},  # ADDED: Test LLM 2
        {"name": "ministral-3-8b-instruct-2512-bf16", "path": "mistralai/Ministral-3-8B-Instruct-2512-BF16"},  # ADDED: Test LLM 3
    ],  # ADDED
    "ENTROPY_MIN" : 0.85,  # ADDED: step 7 gate — minimum normalized SSF entropy for the CPD dataset to be accepted
    "DIVERSITY_MIN" : 0.0,  # ADDED: step 7 gate — minimum MST diversity (0.0 accepts anything; tune after seeing a first run's value)
    "MAX_CPD_REGENS" : 2,  # ADDED: step 7 — how many times to regenerate a CPD dataset that fails the quality gates
    "DIVERSITY_MAX_SAMPLES" : 5000,  # ADDED: the MST needs an m x m matrix (~40 GB at m=100,000), so diversity is computed on a random subsample of this size
    "OPT_MAX_PROMPTS" : None,  # ADDED: optional cap on how many of the Z prompts get optimized (None = all of them)
}
constants["BIAS_CATEGORIES"] = [  # ADDED: the bias categories/subfields the final pipeline loops over (pipeline step 27) — edit freely
    {"MF": "gender", "SF": "occupation", "SSF": constants["SSF"]},  # ADDED
    {"MF": "race", "SF": "occupation", "SSF": constants["SSF"]},  # ADDED
    {"MF": "age", "SF": "occupation", "SSF": constants["SSF"]},  # ADDED
]  # ADDED
if os.environ.get("PIPELINE_SSF"):  # ADDED: per-run SSF list override (JSON string) set by final_pipeline.py
    constants["SSF"] = json.loads(os.environ["PIPELINE_SSF"])  # ADDED
if os.environ.get("PIPELINE_SMOKE") == "1":  # ADDED: tiny end-to-end test mode — run as: PIPELINE_SMOKE=1 python3.11 final_pipeline.py
    constants.update({"DATASET_SIZE_CPD": 250, "BATCHES_CPD": 2, "DATASET_SIZE_BIAS": 50, "BATCHES_BIAS": 1, "DATASET_SIZE_KEXP": 50, "BATCHES_KEXP": 1, "OPT_MAX_PROMPTS": 5})  # ADDED
constants["BATCH_SIZE_CPD"] = constants["DATASET_SIZE_CPD"] //  constants["BATCHES_CPD"]
constants["BATCH_SIZE_BIAS"] = constants["DATASET_SIZE_BIAS"] //  constants["BATCHES_BIAS"]
constants["BATCH_SIZE_KEXP"] = constants["DATASET_SIZE_KEXP"] //  constants["BATCHES_KEXP"]  # ADDED: create_inputs(kind="KEXP") reads BATCHES_KEXP / BATCH_SIZE_KEXP
