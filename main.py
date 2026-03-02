from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import httpx
import json
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# This will be set in Render dashboard
COLAB_URL = os.getenv("COLAB_URL")  # example: https://8000-m-s-xxxxx.colab.dev

@app.get("/test")
def test():
    return {"status": "render-ok"}

@app.post("/generate")
async def generate(req: Request):
    try:
        body = await req.json()
        conversation = body.get("conversation", [])

        async with httpx.AsyncClient(timeout=60) as client:
            colab_res = await client.post(
                f"{COLAB_URL.rstrip('/')}/generate",
                content=json.dumps({"conversation": conversation}),
                headers={"Content-Type": "text/plain"},
            )

        if colab_res.status_code != 200:
            return {
                "error": "COLAB_ERROR",
                "status": colab_res.status_code,
                "body": colab_res.text
            }

        return colab_res.json()

    except Exception as e:
        return {"error": "SERVER_ERROR", "details": str(e)}
