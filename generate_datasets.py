import os
import sys
import csv
import json
import time
import subprocess


DATASET_FILES = ["cpd_dataset.csv", "kexp_prompts.pt", "benchmark_prompts.pt"]


def driver():
    from constants import constants
    here = os.path.dirname(os.path.abspath(__file__))

    # show what is already finished so a re-run is easy to follow
    print("===== dataset status =====", flush=True)
    for cat in constants["BIAS_CATEGORIES"]:
        out_dir = os.path.join(here, "datasets", cat["MF"])
        for name in DATASET_FILES:
            final = os.path.join(out_dir, name)
            partial = final + ".partial.pt"
            if os.path.exists(final):
                state = "done"
            elif os.path.exists(partial):
                state = "partial (will resume)"
            else:
                state = "not started"
            print(f"  {cat['MF']:<10} {name:<22} {state}", flush=True)

    for cat in constants["BIAS_CATEGORIES"]:
        out_dir = os.path.join(here, "datasets", cat["MF"])
        os.makedirs(out_dir, exist_ok=True)
        if all(os.path.exists(os.path.join(out_dir, name)) for name in DATASET_FILES):
            print(f"===== category={cat['MF']} already complete, skipping =====", flush=True)
            continue
        env = os.environ.copy()
        env["PIPELINE_MF"] = cat["MF"]
        env["PIPELINE_SFS"] = json.dumps(cat["SFS"])
        env["PIPELINE_SSF"] = json.dumps(cat["SSF"])
        env["PIPELINE_SSF_BY_SF"] = json.dumps(cat["SSF_BY_SF"])
        env["DATASETS_OUT_DIR"] = out_dir
        print(f"===== generating datasets for category={cat['MF']} -> {out_dir} =====", flush=True)
        result = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker"], env=env, cwd=here)
        if result.returncode != 0:
            sys.exit(result.returncode)
    print("all datasets generated", flush=True)


# ---------- helpers shared by the worker (module level so they can be tested) ----------

def is_valid(item, kind):
    if kind == "CPD":
        return (
            isinstance(item, dict)
            and isinstance(item.get("sentence_A"), str)
            and isinstance(item.get("sentence_B"), str)
            and item["sentence_A"].strip() != ""
            and item["sentence_B"].strip() != ""
        )
    return isinstance(item, str) and item.strip() != ""


def item_key(item, kind):
    if kind == "CPD":
        # order-independent so (A, B) and (B, A) count as the same pair
        return tuple(sorted((item["sentence_A"].strip(), item["sentence_B"].strip())))
    return item.strip()


def dedupe(items, kind):
    seen = set()
    out = []
    for item in items:
        if not is_valid(item, kind):
            continue
        key = item_key(item, kind)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def parse_items(raw_text):
    from json_repair import repair_json
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]
    try:
        data = json.loads(repair_json(text))
    except Exception as err:
        print(f"could not parse response as JSON: {err}")
        return []
    if not isinstance(data, dict):
        return []
    for key in ("pairs", "sentences", "questions"):
        if isinstance(data.get(key), list):
            return data[key]
    return []


def atomic_save(obj, path):
    import torch
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def worker(fetch=None):

    # load config and the category-baked prompts
    import torch
    from constants import constants
    from prompts import prompt_for_pairs, prompt_for_random_questions, prompt_for_benchmark_questions

    out_dir = os.environ["DATASETS_OUT_DIR"]

    if fetch is None:
        from openai import OpenAI
        client = OpenAI(api_key=constants["API_KEY"])
        retry_delays = [1, 1, 2, 2, 4, 4, 8, 8, 8, 8, 16, 32]

        def fetch(prompt):
            # one API call with retries; returns the parsed list of items (possibly empty)
            for attempt in range(len(retry_delays) + 1):
                raw_text = ""
                try:
                    stream = client.chat.completions.create(
                        model=constants["GEN_MODEL"],
                        reasoning_effort="none",
                        max_completion_tokens=128000,
                        messages=[{"role": "user", "content": prompt}],
                        stream=True
                    )
                    chunks = []
                    for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta.content:
                            chunks.append(chunk.choices[0].delta.content)
                    raw_text = "".join(chunks).strip()
                except Exception as err:
                    print(f"API error: {err}")
                if raw_text:
                    return parse_items(raw_text)
                if attempt < len(retry_delays):
                    print(f"empty response, retrying in {retry_delays[attempt]}s")
                    time.sleep(retry_delays[attempt])
            print("call failed after retries")
            return []

    def create_inputs(prompt, kind, progress_path):
        # keep calling the API until we hold DATASET_SIZE_<kind> unique items,
        # saving raw results after every call so a stopped run resumes where it left off
        target = constants[f"DATASET_SIZE_{kind}"]
        max_calls = constants["MAX_API_CALLS"]
        calls_done, raw = 0, []
        if os.path.exists(progress_path):
            saved = torch.load(progress_path)
            calls_done, raw = saved["batches_done"], saved["items"]
        unique = dedupe(raw, kind)
        print(f"[{kind}] start: {calls_done} calls done, {len(raw)} raw, {len(unique)}/{target} unique", flush=True)

        while len(unique) < target:
            if calls_done >= max_calls:
                sys.exit(f"[{kind}] reached MAX_API_CALLS={max_calls} with only {len(unique)}/{target} unique items; "
                         f"raise MAX_API_CALLS in constants.py or lower DATASET_SIZE_{kind}")
            print(f"[{kind}] starting API call {calls_done + 1} ({len(unique)}/{target} unique so far)", flush=True)
            new_items = fetch(prompt)
            calls_done += 1
            raw.extend(new_items)
            unique = dedupe(raw, kind)
            atomic_save({"batches_done": calls_done, "items": raw}, progress_path)
            print(f"[{kind}] ended API call {calls_done}: +{len(new_items)} raw -> {len(unique)}/{target} unique", flush=True)

        print(f"[{kind}] target reached after {calls_done} calls; raw results kept in {progress_path}", flush=True)
        return unique[:target]

    # counterfactual pairs dataset
    cpd_csv = os.path.join(out_dir, "cpd_dataset.csv")
    if not os.path.exists(cpd_csv):
        pairs = create_inputs(prompt_for_pairs, "CPD", cpd_csv + ".partial.pt")
        tmp = cpd_csv + ".tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sentence_A", "sentence_B"])
            for item in pairs:
                w.writerow([item["sentence_A"].strip(), item["sentence_B"].strip()])
        os.replace(tmp, cpd_csv)
        print(f"saved {len(pairs)} pairs -> {cpd_csv}", flush=True)

    # k experiment prompts dataset
    kexp_pt = os.path.join(out_dir, "kexp_prompts.pt")
    if not os.path.exists(kexp_pt):
        items = create_inputs(prompt_for_random_questions, "KEXP", kexp_pt + ".partial.pt")
        atomic_save([s.strip() for s in items], kexp_pt)
        print(f"saved {len(items)} prompts -> {kexp_pt}", flush=True)

    # benchmark and optimization prompts dataset
    bench_pt = os.path.join(out_dir, "benchmark_prompts.pt")
    if not os.path.exists(bench_pt):
        items = create_inputs(prompt_for_benchmark_questions, "BIAS", bench_pt + ".partial.pt")
        atomic_save([s.strip() for s in items], bench_pt)
        print(f"saved {len(items)} prompts -> {bench_pt}", flush=True)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        worker()
    else:
        driver()
