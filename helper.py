import os
import json
from contextlib import nullcontext

import torch
from openai import OpenAI
from json_repair import repair_json
from transformers import AutoTokenizer, AutoModelForCausalLM

from constants import constants

client = OpenAI(api_key=constants["API_KEY"])

tokenizer_kwargs = {}
if "mistral" in constants["MODEL_PATH"].lower():
    tokenizer_kwargs["fix_mistral_regex"] = True  # Mistral pre-tokenizer regex fix
tokenizer = AutoTokenizer.from_pretrained(
    constants["MODEL_PATH"],
    local_files_only=os.path.isdir(constants["MODEL_PATH"]),
    trust_remote_code=True,
    **tokenizer_kwargs,
)

# empty system message suppresses Mistral default prompt
CHAT_NEEDS_EMPTY_SYSTEM = "default_system_message" in (tokenizer.chat_template or "")

try:
    model = AutoModelForCausalLM.from_pretrained(
        constants["MODEL_PATH"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
except ValueError as err:
    from transformers import AutoModelForImageTextToText

    print(
        f"AutoModelForCausalLM cannot load {constants['MODEL_PATH']} ({str(err).splitlines()[0][:120]}); "
        f"loading it with AutoModelForImageTextToText instead",
        flush=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        constants["MODEL_PATH"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )

model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def hidden_states(input):
    tensor = tokenizer(input, return_tensors="pt")
    tensor = {k: v.to(model.device) for k, v in tensor.items()}
    tensor["input_ids"] = tensor["input_ids"].long()
    with torch.no_grad():
        outputs = model(**tensor, output_hidden_states=True, return_dict=True)
    return outputs.hidden_states


def create_inputs(prompt, temp, tokens, kind):
    inputs = []

    for i in range(constants[f"BATCHES_{kind}"]):
        print("starting API call")
        raw_text = ""
        for attempt in range(3):
            response = client.chat.completions.create(
                model=constants["GEN_MODEL"],
                temperature=temp,
                max_completion_tokens=constants[f"BATCH_SIZE_{kind}"] * tokens + 30000,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_text = (response.choices[0].message.content or "").strip()
            if raw_text:
                break
            print("empty response, retrying batch")

        if not raw_text:
            print("batch failed after retries, skipping")
            continue

        if raw_text.startswith("```"):
            raw_text = raw_text.replace("```json", "").replace("```", "").strip()

        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1:
            raw_text = raw_text[start:end + 1]

        data = json.loads(repair_json(raw_text))

        if "sentences" in data:
            inputs.extend(data["sentences"])
        elif "questions" in data:
            inputs.extend(data["questions"])
        elif "pairs" in data:
            inputs.extend(data["pairs"])

        print("ended API call")

    return inputs


def create_inputs_with_grad(emb, attention_mask):
    emb = emb.to(model.device).to(model.dtype)
    attention_mask = attention_mask.to(model.device)

    outputs = model(
        inputs_embeds=emb,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )

    input_vec = outputs.hidden_states[constants["LAYER"]].mean(dim=1).squeeze()
    return input_vec


def create_output_from_emb(emb, attention_mask, with_grad, max_tokens=40, return_text=False):
    context = nullcontext() if with_grad else torch.no_grad()

    with context:
        emb = emb.to(model.device).to(model.dtype)
        attention_mask = attention_mask.to(model.device)

        with torch.no_grad():
            gen_ids = model.generate(
                inputs_embeds=emb,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        K = gen_ids.shape[1]

        gen_embeds = model.get_input_embeddings()(gen_ids).to(model.dtype).detach()

        full_embeds = torch.cat([emb, gen_embeds], dim=1)
        ones = torch.ones((emb.shape[0], K), device=model.device, dtype=attention_mask.dtype)
        full_mask = torch.cat([attention_mask, ones], dim=1)

        outputs = model(
            inputs_embeds=full_embeds,
            attention_mask=full_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        output_vec = outputs.hidden_states[constants["LAYER"]][:, -K:, :].mean(dim=1).squeeze()
    if return_text:
        gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
        return output_vec, gen_text
    return output_vec


def create_outputs_from_emb_batch(embs, attention_mask, max_tokens=40):
    """Batched no-grad version of create_output_from_emb."""
    with torch.no_grad():
        B = embs.shape[0]
        embs = embs.to(model.device).to(model.dtype)
        mask = attention_mask.to(model.device)
        if mask.shape[0] == 1:
            mask = mask.repeat(B, 1)
        gen_ids = model.generate(
            inputs_embeds=embs,
            attention_mask=mask,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        K = gen_ids.shape[1]

        # count tokens up to first EOS
        eos_ids = model.generation_config.eos_token_id
        eos_ids = [eos_ids] if isinstance(eos_ids, int) else list(eos_ids or [])
        eos_ids = torch.tensor(sorted(set(eos_ids + [tokenizer.eos_token_id])), device=gen_ids.device)
        is_eos = torch.isin(gen_ids, eos_ids)
        first_eos = torch.where(
            is_eos.any(dim=1),
            is_eos.int().argmax(dim=1),
            torch.full((B,), K - 1, device=gen_ids.device),
        )
        gen_mask = torch.arange(K, device=gen_ids.device)[None, :] <= first_eos[:, None]

        gen_embeds = model.get_input_embeddings()(gen_ids).to(model.dtype)
        full_embeds = torch.cat([embs, gen_embeds], dim=1)
        full_mask = torch.cat([mask, gen_mask.to(mask.dtype)], dim=1)
        outputs = model(
            inputs_embeds=full_embeds,
            attention_mask=full_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        h = outputs.hidden_states[constants["LAYER"]][:, -K:, :].float()
        w = gen_mask.float().unsqueeze(-1)
        return (h * w).sum(dim=1) / w.sum(dim=1)


def get_bias_score_from_emb(outputs, bias_subspace, center):
    bias_total = 0
    err_cnt = 0

    for output_vec in outputs:
        if output_vec is None:
            err_cnt += 1
            continue

        vector = output_vec.squeeze().float()
        local_center = center.squeeze().to(vector.device).float()
        local_bias_subspace = bias_subspace.to(vector.device).float()

        centered = vector - local_center
        coords = local_bias_subspace @ centered
        bias_total += torch.norm(coords)

    return (bias_total / (len(outputs) - err_cnt)).detach()


def create_embedding(input):
    messages = [{"role": "user", "content": input}]
    if CHAT_NEEDS_EMPTY_SYSTEM:
        messages.insert(0, {"role": "system", "content": ""})

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    tokens = tokenizer(text, return_tensors="pt").to(model.device)
    input_embeds = model.get_input_embeddings()(tokens["input_ids"])
    return input_embeds, tokens["attention_mask"]


def line_search(current_emb, direction, attention_mask, center_device, bias_subspace_device, eta_init, eta_cap):
    """First local maximum along the search ray."""
    growth = constants["OPT_LINE_SEARCH_GROWTH"]
    max_evals = constants["OPT_LINE_SEARCH_EVALS"]

    @torch.no_grad()
    def phi_batch(rs):
        embs = torch.cat([current_emb + r * direction for r in rs], dim=0)  # (B, T, H)
        outs = create_outputs_from_emb_batch(embs, attention_mask)  # (B, H)
        return ((outs - center_device) @ bias_subspace_device.T).norm(dim=1).tolist()

    # bracket: geometric steps until cap or budget
    rs = [0.0]
    r = eta_init
    while r <= eta_cap and len(rs) <= max_evals:
        rs.append(r)
        r *= growth
    if len(rs) == 1:
        return eta_cap  # first trial step exceeds the cap
    vals = phi_batch(rs)
    drop = next((i for i in range(1, len(rs)) if vals[i] < vals[i - 1]), None)
    if drop is None:
        return eta_cap if r > eta_cap else rs[-1]
    if drop < 2:
        return 0.0

    # refine: vertex of the bracketing parabola
    (a, fa), (b, fb), (c, fc) = [(rs[i], vals[i]) for i in (drop - 2, drop - 1, drop)]
    denom = (b - a) * (fb - fc) - (b - c) * (fb - fa)
    if abs(denom) < 1e-12:
        return b
    vertex = b - 0.5 * ((b - a) ** 2 * (fb - fc) - (b - c) ** 2 * (fb - fa)) / denom
    return min(max(vertex, a), c)
