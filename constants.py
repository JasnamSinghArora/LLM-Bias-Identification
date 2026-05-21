import os
constants = {
    "API_KEY" : os.environ["API_KEY"],
    "DATASET_SIZE" : 25,
    "BATCHES" : 1,
    "MF" : "gender",
    "SF" : "sports",
    "MODEL_PATH" : "Qwen/Qwen2.5-14B-Instruct"
}
constants["BATCH_SIZE"] = constants["DATASET_SIZE"] //  constants["BATCHES"]
