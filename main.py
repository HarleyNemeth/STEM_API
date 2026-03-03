from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, json, re

app = FastAPI()

# CORS for Wix backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# LOAD THE MODEL
# ---------------------------
MODEL_ID = "microsoft/phi-3-mini-4k-instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
).to(device)

# ---------------------------
# SYSTEM PROMPT
# ---------------------------
SYSTEM_PROMPT = """
You collect information for poster generation.
Required fields:
- event_title
- event_description
- date
- time
- location
- target_audience
- tone
- colour_scheme

If ANY field is missing: reply ONLY with
{"follow_up":["field1", "field2"]}

If ALL fields are filled: reply ONLY with JSON containing exactly the required fields.
No markdown. No explanations.
"""

REQUIRED_FIELDS = [
    "event_title",
    "event_description",
    "date",
    "time",
    "location",
    "target_audience",
    "tone",
    "colour_scheme"
]


# ---------------------------
# RUN THE MODEL
# ---------------------------
def run_llm(conversation):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=False,
        temperature=0.0,
    )

    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text


# ---------------------------
# FASTAPI ENDPOINTS
# ---------------------------

@app.get("/test")
def test():
    return {"status": "azure-model-ok"}

@app.post("/generate")
async def generate(request: Request):
    body = await request.json()
    conversation = body.get("conversation", [])
    raw = run_llm(conversation)

    try:
        obj = json.loads(raw)
    except:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return {"error":"MODEL_NON_JSON","raw":raw}
        obj = json.loads(match.group(0))

    if "follow_up" in obj:
        fu = obj["follow_up"]
        fu_norm = [f for f in fu if f in REQUIRED_FIELDS] or fu
        return {"follow_up": fu_norm}

    final = {k: obj.get(k, "") for k in REQUIRED_FIELDS}
    missing = [k for k, v in final.items() if not v]

    if missing:
        return {"follow_up": missing}

    return final
