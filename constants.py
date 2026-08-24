import os
constants = {
    "API_KEY" : os.environ["API_KEY"],
    "DATASET_SIZE_CPD" : 300,
    "BATCHES_CPD" : 2,
    "DATASET_SIZE_BIAS" : 600,
    "BATCHES_BIAS" : 10,
    "TEMP" : 1.0,
    "TOKENS" : 60,
    "MF" : "gender",
    "SF" : "occupation",
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
    "LAYER" : 16,
    "MODEL_PATH" : "/Users/jasnamsingharora/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
}
constants["BATCH_SIZE_CPD"] = constants["DATASET_SIZE_CPD"] //  constants["BATCHES_CPD"]
constants["BATCH_SIZE_BIAS"] = constants["DATASET_SIZE_BIAS"] //  constants["BATCHES_BIAS"]