from openai import OpenAI
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from contextlib import nullcontext
from json_repair import repair_json
from constants import constants

client = OpenAI(api_key=constants["API_KEY"])
tokenizer = AutoTokenizer.from_pretrained(
    constants["MODEL_PATH"],
    local_files_only=os.path.isdir(constants["MODEL_PATH"]),
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
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
                          bias_subspace_device, r_init=0.05, growth=1.5, r_max=4.0, tol=1e-3):

    @torch.no_grad()
    def phi(r, current_emb, direction, attention_mask, center_device, bias_subspace_device):
        e_r = current_emb + r * direction
        out = create_output_from_emb(e_r, attention_mask, with_grad=False)
        coords = bias_subspace_device @ (out - center_device)
        return torch.norm(coords).item()

    f = lambda r: phi(r, current_emb, direction, attention_mask,
                      center_device, bias_subspace_device)
    
    rs   = [0.0]
    vals = [f(0.0)]
    r = r_init
    while r <= r_max:
        vals.append(f(r))
        rs.append(r)
        if vals[-1] < vals[-2]:
            if len(rs) < 3:
                return 0.0
            a, b = rs[-3], rs[-1]
            break
        r *= growth
    else:
        return rs[-1] 
    
    gr = (5 ** 0.5 - 1) / 2   # ~0.618
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc, fd = f(c), f(d)
    while abs(b - a) > tol:
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = f(d)
    return (a + b) / 2

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
