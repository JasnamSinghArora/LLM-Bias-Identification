import os
constants = {
    "API_KEY" : os.environ["API_KEY"],
    "DATASET_SIZE" : 25,
    "BATCHES" : 1,
    "MF" : "gender",
    "SF" : "sports",
    "MODEL_PATH" : "/Users/jasnamsingharora/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
}
constants["BATCH_SIZE"] = constants["DATASET_SIZE"] //  constants["BATCHES"]