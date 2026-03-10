# main.py
import os, json, re
from typing import List, Literal
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential

# --- NEW: static serving & proxy imports ---
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
import httpx

# ----------------------------------------------------------------------
# FastAPI + CORS (tighten origins via env; fallback to *)
# ----------------------------------------------------------------------
WIX_ORIGIN = os.getenv("WIX_ORIGIN", "")            # e.g. https://your-site.wixsite.com
AZURE_ORIGIN = os.getenv("AZURE_ORIGIN", "")        # e.g. https://your-app.azurewebsites.net
ALLOWED = [o for o in [WIX_ORIGIN, AZURE_ORIGIN] if o]
ALLOW_ORIGINS = ALLOWED if ALLOWED else ["*"]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=True,
)

# ----------------------------------------------------------------------
# CONFIG (GitHub Models via Azure AI Inference SDK)
# ----------------------------------------------------------------------
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
if not GITHUB_TOKEN:
    raise RuntimeError("Missing GITHUB_TOKEN env var (PAT must include models:read).")

ENDPOINT = "https://models.inference.ai.azure.com"  # GitHub Models endpoint
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
USE_JSON_MODE = os.getenv("JSON_MODE", "1") == "1"

client = ChatCompletionsClient(
    endpoint=ENDPOINT,
    credential=AzureKeyCredential(GITHUB_TOKEN)
)

# ----------------------------------------------------------------------
# SYSTEM PROMPT + SCHEMA (kept from your version)
# ----------------------------------------------------------------------
SYSTEM_PROMPT = """
You collect information for poster generation and must output only JSON.

Required fields (all strings):
- event_title
- event_description
- date
- time
- location
- target_audience
- tone
- colour_scheme

Input can be:
1) Unstructured sentences across multiple user messages.
2) A single comma-separated line in this exact order:
   event_title, event_description, date, time, location, target_audience, tone, colour_scheme
3) Key:value lines using the exact field names.

Rules:
- Combine ALL user messages and extract the 8 fields.
- Normalize obvious typos in text (e.g., "ehtnic" -> "ethnic").
- If a field is strongly implied, infer it (e.g., "students" -> target_audience = "students").
- Trim trailing commas and extra punctuation.

If ANY required field is missing, reply ONLY with:
{"follow_up":["field1","field2", ...]}

If ALL fields are present, reply ONLY with a JSON object containing exactly the required fields (no extra keys).
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
    "colour_scheme",
]

class ChatMessage(BaseModel):
    role: Literal["system","user","assistant"]
    content: str

class GenerateRequest(BaseModel):
    conversation: List[ChatMessage] = Field(default_factory=list)

# ----------------------------------------------------------------------
# Helpers (kept from your version)
# ----------------------------------------------------------------------
def parse_json_strict_then_fallback(text: str):
    """Try strict JSON; if that fails, extract first {...} block."""
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise ValueError(f"Non-JSON model output: {text[:500]}")
        return json.loads(m.group(0))

def merge_user_messages(conversation: List[ChatMessage]) -> str:
    """Loop fix: keep ONLY user messages; merge to a single prompt."""
    return "\n".join(
        m.content.strip()
        for m in conversation
        if m.role == "user" and m.content and m.content.strip()
    )

def normalize_user_input(s: str) -> str:
    """Light normalization for common slips and trailing punctuation."""
    if not s:
        return s
    out = s
    out = re.sub(r"\behtnic\b", "ethnic", out, flags=re.IGNORECASE)
    out = re.sub(r"\bwemon\b", "women", out, flags=re.IGNORECASE)
    out = re.sub(r"(^|\n)\s*vent_title\s*:", r"\1event_title:", out, flags=re.IGNORECASE)
    out = re.sub(r"\s*,\s*,+", ", ", out)
    out = re.sub(r"\s+,", ", ", out)
    out = re.sub(r",\s*$", "", out, flags=re.MULTILINE)
    return out

def ensure_final_or_followup(obj: dict):
    """Normalize/validate the response and decide follow_up vs final JSON."""
    if isinstance(obj, dict) and "follow_up" in obj:
        fu = obj.get("follow_up", [])
        return {"follow_up": fu}

    final = {k: (obj.get(k, "") if isinstance(obj, dict) else "") for k in REQUIRED_FIELDS}
    missing = [k for k, v in final.items() if not v]
    if missing:
        return {"follow_up": missing}
    return final

# ----------------------------------------------------------------------
# Endpoints (kept + NEW: static + poster + proxy)
# ----------------------------------------------------------------------
@app.get("/test")
def test():
    return {"status":"ok","provider":"github-models","model":MODEL_NAME,"json_mode":USE_JSON_MODE}

@app.post("/generate")
def generate(req: GenerateRequest):
    combined_user_input = merge_user_messages(req.conversation).strip()
    if not combined_user_input:
        return {"follow_up": REQUIRED_FIELDS}

    combined_user_input = normalize_user_input(combined_user_input)

    messages = [
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":combined_user_input}
    ]

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 300
    }
    if USE_JSON_MODE:
        payload["response_format"] = {"type":"json_object"}

    try:
        resp = client.complete(payload)
        if not resp or not getattr(resp, "choices", None):
            raise HTTPException(status_code=502, detail="Empty response: choices missing")
        if not resp.choices[0].message or resp.choices[0].message.content is None:
            raise HTTPException(status_code=502, detail="Empty response: message.content is null")
        raw = resp.choices[0].message.content.strip()
        obj = parse_json_strict_then_fallback(raw)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ensure_final_or_followup(obj)

# --- NEW: static hosting ---
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/poster.html")
def poster_html():
    # Serves /static/poster.html
    return FileResponse("static/poster.html", media_type="text/html")

# --- NEW: simple, safe image proxy to avoid CORS tainting for html2canvas ---
SAFE_URL_RE = re.compile(r"^https?://", re.IGNORECASE)

@app.get("/img-proxy")
async def img_proxy(url: str):
    if not SAFE_URL_RE.match(url):
        raise HTTPException(status_code=400, detail="Invalid URL")
    try:
        async with httpx.AsyncClient(timeout=10) as http:
            r = await http.get(url)
            r.raise_for_status()
        content_type = r.headers.get("content-type", "application/octet-stream")
        headers = {
            "Cache-Control": "public, max-age=3600",
        }
        # If you set WIX_ORIGIN, you can add ACAO for strictness:
        if WIX_ORIGIN:
            headers["Access-Control-Allow-Origin"] = WIX_ORIGIN
        return Response(content=r.content, media_type=content_type, headers=headers)
    except httpx.RequestError:
        raise HTTPException(status_code=502, detail="Proxy fetch error")
