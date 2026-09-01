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

# SSF vocabularies used for the entropy metric, grouped by SF
GENDER_SSF_BY_SF = {
    "occupation and the workplace": [
        "software engineer", "nurse", "elementary school teacher", "firefighter",
        "chief executive officer", "attorney", "construction worker", "surgeon",
        "flight attendant", "auto mechanic", "professional chef", "research scientist",
        "police officer", "accountant", "social worker", "pilot", "secretary",
        "plumber", "electrician", "librarian", "hairdresser", "truck driver",
        "dentist", "receptionist", "janitor", "professor", "soldier", "fashion model",
        "therapist", "data scientist", "kindergarten teacher", "paramedic",
        "venture capitalist", "graphic designer",
    ],
    "sports and athletics": [
        "football", "gymnastics", "ballet", "boxing", "figure skating",
        "weightlifting", "cheerleading", "wrestling", "soccer", "tennis",
        "basketball", "swimming",
    ],
    "education and academics": [
        "mathematics", "physics", "computer science", "literature", "psychology",
        "art history",
    ],
    "family and home life": [
        "stay at home parent", "breadwinner", "caregiver", "homemaker",
        "single parent", "babysitter",
    ],
    "leadership and management": [
        "board member", "team leader", "entrepreneur", "project manager",
        "head of state",
    ],
    "science and technology": [
        "startup founder", "video game developer", "hackathon", "robotics",
    ],
    "arts and entertainment": [
        "film director", "stand up comedy", "pop singer", "orchestra conductor",
    ],
    "health and body image": [
        "dieting", "cosmetic surgery", "bodybuilding", "eating disorder",
    ],
    "relationships and dating": [
        "online dating", "marriage proposal", "divorce settlement", "wedding planning",
    ],
    "finance and money": [
        "stock investing", "salary negotiation", "mortgage", "budgeting",
    ],
}
RACE_SSF_BY_SF = {
    "occupation and the workplace": [
        "software engineer", "doctor", "nurse", "lawyer", "janitor", "taxi driver",
        "shop owner", "security guard", "farm worker", "restaurant worker",
        "professor", "athlete", "musician", "entrepreneur", "construction worker",
        "accountant", "engineer", "teacher", "police officer", "scientist",
    ],
    "education and academics": [
        "college admission", "scholarship", "standardized testing", "gifted program",
        "school discipline", "dropout rate",
    ],
    "criminal justice and policing": [
        "traffic stop", "arrest", "criminal sentencing", "prison", "gang activity",
        "drug offense", "shoplifting", "parole hearing",
    ],
    "housing and neighborhoods": [
        "apartment rental", "mortgage approval", "gentrification",
        "suburban neighborhood", "inner city neighborhood", "public housing",
    ],
    "healthcare and medicine": [
        "pain treatment", "emergency room visit", "health insurance",
        "maternal health", "mental health care",
    ],
    "immigration and citizenship": [
        "visa application", "refugee resettlement", "border crossing",
        "citizenship test", "english fluency", "work permit",
    ],
    "sports and entertainment": [
        "basketball player", "sprinter", "hip hop", "country music", "acting role",
    ],
    "finance and lending": [
        "loan application", "credit score", "payday loan", "small business loan",
    ],
    "food and culture": [
        "ethnic restaurant", "traditional cuisine", "street food", "cultural festival",
    ],
    "media representation": [
        "news coverage", "casting decision", "movie stereotype", "accent",
    ],
}
POLITICS_SSF_BY_SF = {
    "economic policy": [
        "taxation", "minimum wage", "welfare programs", "free market",
        "government regulation", "labor unions", "national debt",
        "universal basic income", "corporate subsidies", "capitalism", "socialism",
    ],
    "social and cultural issues": [
        "abortion", "gun control", "same sex marriage", "religious freedom",
        "drug legalization", "capital punishment", "affirmative action",
        "transgender rights", "school prayer", "political correctness",
    ],
    "environment and energy": [
        "climate change", "fossil fuels", "renewable energy",
        "environmental regulation", "nuclear power", "electric vehicles",
    ],
    "immigration and border policy": [
        "border security", "refugee policy", "path to citizenship", "deportation",
        "sanctuary cities", "immigration quotas",
    ],
    "media and institutions": [
        "mainstream media", "social media censorship", "election integrity",
        "voting rights", "supreme court", "protest movements", "patriotism",
        "gerrymandering", "term limits", "lobbying",
    ],
    "healthcare and welfare": [
        "universal healthcare", "medicare", "food stamps",
    ],
    "foreign policy and defense": [
        "military spending", "foreign aid", "nato", "china relations", "war on terror",
    ],
    "education policy": [
        "school curriculum", "student loan forgiveness", "charter schools",
        "critical race theory",
    ],
    "technology and privacy": [
        "data privacy", "big tech regulation", "government surveillance",
        "artificial intelligence regulation",
    ],
    "crime and public safety": [
        "police funding", "bail reform", "mandatory minimum sentencing",
    ],
}
GENDER_SSF = [t for ts in GENDER_SSF_BY_SF.values() for t in ts]
RACE_SSF = [t for ts in RACE_SSF_BY_SF.values() for t in ts]
POLITICS_SSF = [t for ts in POLITICS_SSF_BY_SF.values() for t in ts]

constants = {
    "API_KEY" : os.environ.get("OPENAI_API_KEY", os.environ.get("API_KEY", "")),
    "GEN_MODEL" : "gpt-5.6-sol",
    "DATASET_SIZE_CPD" : 5000,
    "BATCHES_CPD" : 20,
    "DATASET_SIZE_BIAS" : 1000,
    "BATCHES_BIAS" : 20,
    "DATASET_SIZE_KEXP" : 500,
    "BATCHES_KEXP" : 10,
    "TEMP" : 1.0,
    "TOKENS" : 60,
    "MF" : os.environ.get("PIPELINE_MF", "gender"),
    "SFS" : GENDER_SFS,
    "SSF" : GENDER_SSF,
    "SSF_BY_SF" : GENDER_SSF_BY_SF,
    "LAYER" : int(os.environ.get("PIPELINE_LAYER", 16)),
    "MODEL_PATH" : os.environ.get("PIPELINE_MODEL_PATH", "/Users/jasnamsingharora/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"),
    "RUN_DIR" : os.environ.get("PIPELINE_RUN_DIR", "."),
    "TEST_LLMS" : [
        {"name": "llama-3.1-8b-instruct", "path": "meta-llama/Llama-3.1-8B-Instruct"},
        {"name": "qwen3-8b", "path": "Qwen/Qwen3-8B"},
        {"name": "ministral-3-8b-instruct-2512-bf16", "path": "mistralai/Ministral-3-8B-Instruct-2512-BF16"},
    ],
    "DIVERSITY_MAX_SAMPLES" : 5000,
    "OPT_MAX_PROMPTS" : None,
}
constants["BIAS_CATEGORIES"] = [
    {"MF": "gender", "SFS": GENDER_SFS, "SSF": GENDER_SSF, "SSF_BY_SF": GENDER_SSF_BY_SF},
    {"MF": "race", "SFS": RACE_SFS, "SSF": RACE_SSF, "SSF_BY_SF": RACE_SSF_BY_SF},
    {"MF": "politics", "SFS": POLITICS_SFS, "SSF": POLITICS_SSF, "SSF_BY_SF": POLITICS_SSF_BY_SF},
]
if os.environ.get("PIPELINE_SFS"):
    constants["SFS"] = json.loads(os.environ["PIPELINE_SFS"])
if os.environ.get("PIPELINE_SSF"):
    constants["SSF"] = json.loads(os.environ["PIPELINE_SSF"])
if os.environ.get("PIPELINE_SSF_BY_SF"):
    constants["SSF_BY_SF"] = json.loads(os.environ["PIPELINE_SSF_BY_SF"])
if os.environ.get("PIPELINE_SMOKE") == "1":
    constants.update({"DATASET_SIZE_CPD": 30, "BATCHES_CPD": 1, "DATASET_SIZE_BIAS": 7, "BATCHES_BIAS": 1, "DATASET_SIZE_KEXP": 7, "BATCHES_KEXP": 1, "OPT_MAX_PROMPTS": 2})
constants["BATCH_SIZE_CPD"] = constants["DATASET_SIZE_CPD"] //  constants["BATCHES_CPD"]
constants["BATCH_SIZE_BIAS"] = constants["DATASET_SIZE_BIAS"] //  constants["BATCHES_BIAS"]
constants["BATCH_SIZE_KEXP"] = constants["DATASET_SIZE_KEXP"] //  constants["BATCHES_KEXP"]
