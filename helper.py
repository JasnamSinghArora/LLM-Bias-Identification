from openai import OpenAI
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from contextlib import nullcontext
from json_repair import repair_json
from constants import constants

client = OpenAI(api_key=constants["API_KEY"])
tokenizer_kwargs = {}
if "mistral" in constants["MODEL_PATH"].lower():
    tokenizer_kwargs["fix_mistral_regex"] = True  # Mistral tokenizers ship a pre-tokenizer regex bug; transformers warns to set this
tokenizer = AutoTokenizer.from_pretrained(
    constants["MODEL_PATH"],
    local_files_only=os.path.isdir(constants["MODEL_PATH"]),
    trust_remote_code=True,
    **tokenizer_kwargs
)

# some chat templates (Mistral's) silently prepend a several-hundred-token default system prompt when the
# conversation has no system message; an empty system message suppresses it so prompts stay short
CHAT_NEEDS_EMPTY_SYSTEM = "default_system_message" in (tokenizer.chat_template or "")

try:
    model = AutoModelForCausalLM.from_pretrained(
        constants["MODEL_PATH"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager"
    )
except ValueError as err:
    from transformers import AutoModelForImageTextToText
    print(f"AutoModelForCausalLM cannot load {constants['MODEL_PATH']} ({str(err).splitlines()[0][:120]}); "
          f"loading it with AutoModelForImageTextToText instead", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        constants["MODEL_PATH"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager"
    )

model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def hidden_states(input):
    tensor = tokenizer(
                    input,
                    return_tensors="pt"
                )
    tensor = {k: v.to(model.device) for k, v in tensor.items()}
    tensor["input_ids"] = tensor["input_ids"].long()
    with torch.no_grad():
        outputs = model(
            **tensor,
            output_hidden_states=True,
            return_dict=True
        )
        
    return outputs.hidden_states
    
def generate_with_test_llm(input):

    messages = [
        {"role": "system", "content": "You answer in one short English sentence."},
        {"role": "user", "content": f"{input}\n\nAnswer in one short English sentence."}
        ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    text += "Answer: "   # forces the first generated token into English/answer mode

    inputs_tensor = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs_tensor,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_tokens = outputs[0][inputs_tensor["input_ids"].shape[-1]:]
    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()
    
    return response

def create_inputs(prompt, temp, tokens, kind):
    inputs = []

    for i in range (constants[f"BATCHES_{kind}"]):
        print("starting API call")
        raw_text = ""
        for attempt in range(3):
            response = client.chat.completions.create(
                model=constants["GEN_MODEL"],
                max_completion_tokens = constants[f"BATCH_SIZE_{kind}"] * tokens + 30000,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
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
        return_dict=True
    )

    input_vec = outputs.hidden_states[constants["LAYER"]].mean(dim=1).squeeze()
    return input_vec
        
def create_outputs_from_text(inputs):
    outputs = []
    for input in inputs:
        output = generate_with_test_llm(input)
        outputs.append(output)
        
    return outputs

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
    """Batched, no-grad version of create_output_from_emb for B perturbed copies of one prompt.

    embs: (B, T, H) with the same token length in every row; attention_mask: (1, T) or (B, T).
    Returns (B, H): for each row, the mean layer-LAYER hidden state over that row's generated
    tokens up to and including its first end-of-sequence token, i.e. the same quantity
    create_output_from_emb returns for that row on its own.
    """
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

        # rows that stop early are padded after their end token; count only up to that token
        eos_ids = model.generation_config.eos_token_id
        eos_ids = [eos_ids] if isinstance(eos_ids, int) else list(eos_ids or [])
        eos_ids = torch.tensor(sorted(set(eos_ids + [tokenizer.eos_token_id])), device=gen_ids.device)
        is_eos = torch.isin(gen_ids, eos_ids)
        first_eos = torch.where(is_eos.any(dim=1), is_eos.int().argmax(dim=1), torch.full((B,), K - 1, device=gen_ids.device))
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

def get_bias_score_from_text(outputs, bias_subspace, center):
    bias_total = 0
    err_cnt = 0

    for output in outputs:
        if output == "":
            err_cnt += 1
            continue
        states = hidden_states(output)
        vector = states[constants["LAYER"]].mean(dim=1).squeeze().float().cpu()
        centered = vector - center.squeeze()
        coords = bias_subspace @ centered
        bias_total += torch.norm(coords)
        
    bias_score = bias_total / (len(outputs) - err_cnt)
    return bias_score
        
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
    messages = [
        {"role": "user", "content": input}
    ]
    if CHAT_NEEDS_EMPTY_SYSTEM:
        messages.insert(0, {"role": "system", "content": ""})

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    tokens = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    input_embeds = model.get_input_embeddings()(tokens["input_ids"])
    return input_embeds, tokens["attention_mask"]

def get_optimized_epsilon(current_emb, direction, attention_mask, center_device,
                          bias_subspace_device, r_init=0.05, growth=1.5, r_max=4.0, tol=1e-3, grid=16):
    """Step size that maximises the bias score along `direction`.

    Bracketing is the same rule as before: try r_init, r_init*growth, ... up to r_max and stop at
    the first drop in score. Every candidate is now scored in one batched generation instead of
    one at a time, and the bracket is narrowed with batched grids of `grid` points until the
    spacing is below `tol`, instead of a sequential golden-section search.
    """

    @torch.no_grad()
    def phi_batch(rs):
        embs = torch.cat([current_emb + r * direction for r in rs], dim=0)  # (B, T, H)
        outs = create_outputs_from_emb_batch(embs, attention_mask)          # (B, H)
        coords = (outs - center_device) @ bias_subspace_device.T            # (B, k)
        return coords.norm(dim=1).tolist()

    # 1) bracketing: score 0, r_init, r_init*growth, ... <= r_max in a single batch
    rs = [0.0]
    r = r_init
    while r <= r_max:
        rs.append(r)
        r *= growth
    vals = phi_batch(rs)
    first_drop = next((i for i in range(1, len(rs)) if vals[i] < vals[i - 1]), None)
    if first_drop is None:
        return rs[-1]
    if first_drop < 2:
        return 0.0
    a, b = rs[first_drop - 2], rs[first_drop]
    best_r, best_v = rs[first_drop - 1], vals[first_drop - 1]

    # 2) refine: evenly spaced interior grid, keep the best point, shrink the bracket around it
    while (b - a) / grid > tol:
        cand = torch.linspace(a, b, grid + 2)[1:-1].tolist()
        cv = phi_batch(cand)
        i = max(range(len(cand)), key=lambda j: cv[j])
        if cv[i] > best_v:
            best_r, best_v = cand[i], cv[i]
        step = (b - a) / (grid + 1)
        a, b = max(a, best_r - step), min(b, best_r + step)
    return best_r

def cg_solve(matvec, b, max_iter, tol=1e-4, x0=None):
    """Solve A x = b for SPD operator A given as a matvec callable.
    Returns (x, iters_used, final_residual)."""
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - matvec(x)
    p = r.clone()
    rs_old = (r * r).sum()
    b_norm = b.norm() + 1e-30
    iters = 0
    for k in range(max_iter):
        Ap = matvec(p)
        alpha = rs_old / ((p * Ap).sum() + 1e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = (r * r).sum()
        iters = k + 1
        rel = (rs_new.sqrt() / b_norm).item()
        if rel < tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x, iters, rel
