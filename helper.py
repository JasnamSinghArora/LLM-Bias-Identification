from anthropic import Anthropic
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from json_repair import repair_json
from constants import constants

client = Anthropic(api_key=constants["API_KEY"])
tokenizer = AutoTokenizer.from_pretrained(
    constants["MODEL_PATH"],
    local_files_only=False,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    constants["MODEL_PATH"],
    torch_dtype=torch.float16,
    device_map="auto"
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
        add_generation_prompt=True
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

def create_inputs(prompt, temp, tokens):
    inputs = []

    for i in range (constants["BATCHES"]):
        print("starting API call")
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens = constants["BATCH_SIZE"] * tokens,
            temperature=temp,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        raw_text = response.content[0].text.strip()

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
        
def create_outputs_from_text(inputs):
    outputs = []
    for input in inputs:
        output = generate_with_test_llm(input)
        outputs.append(output)
        
    return outputs

def create_output_from_emb(emb, attention_mask, return_text, max_tokens=40):
    emb = emb.to(model.device).to(model.dtype)
    attention_mask = attention_mask.to(model.device)

    if not return_text:
        outputs = model(
            inputs_embeds=emb,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        output_vec = outputs.hidden_states[-1].mean(dim=1).squeeze()
        return output_vec

    generated_ids = model.generate(
        inputs_embeds=emb,
        attention_mask=attention_mask,
        max_new_tokens=max_tokens,
        do_sample=False
    )

    output_text = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True
    )

    outputs = model(
        input_ids=generated_ids,
        output_hidden_states=True,
        return_dict=True
    )

    output_vec = outputs.hidden_states[-1].mean(dim=1).squeeze()

    return output_vec, output_text

def get_bias_score_from_text(outputs, bias_subspace, bias_mean):
    bias_total = 0
    err_cnt = 0
    
    for output in outputs:
        if output == "":
            err_cnt += 1
            continue
        states = hidden_states(output)
        vector = states[-1].mean(dim=1).squeeze().float().cpu()
        centered = vector - bias_mean.squeeze()
        coords = bias_subspace @ centered
        bias_total += torch.norm(coords)
        
    bias_score = bias_total / (len(outputs) - err_cnt)
    return bias_score
        
def get_bias_score_from_emb(outputs, bias_subspace, bias_mean):
    bias_total = 0
    err_cnt = 0

    for output_vec in outputs:
        if output_vec is None:
            err_cnt += 1
            continue

        vector = output_vec.squeeze().float()

        local_bias_mean = bias_mean.squeeze().to(vector.device).float()
        local_bias_subspace = bias_subspace.to(vector.device).float()

        centered = vector - local_bias_mean
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
        add_generation_prompt=True
    )

    tokens = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    input_embeds = model.get_input_embeddings()(tokens["input_ids"])
    return input_embeds, tokens["attention_mask"]

def get_optimized_epsilon(current_emb, direction, attention_mask, bias_mean_device, 
                          bias_subspace_device, r_init=0.05, growth=1.5, r_max=4.0, tol=1e-3):
    
    @torch.no_grad()
    def phi(r, current_emb, direction, attention_mask, bias_mean_device, bias_subspace_device):
        e_r = current_emb + r * direction
        out = create_output_from_emb(e_r, attention_mask, False)
        coords = bias_subspace_device @ (out - bias_mean_device)
        return torch.norm(coords).item()
    
    f = lambda r: phi(r, current_emb, direction, attention_mask,
                      bias_mean_device, bias_subspace_device)
    
    rs   = [0.0]
    vals = [f(0.0)]
    r = r_init
    while r <= r_max:
        vals.append(f(r))
        rs.append(r)
        if vals[-1] < vals[-2]:
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
