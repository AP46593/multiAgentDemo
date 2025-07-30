from __future__ import annotations
from typing import cast
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
import base64, re
from schema import GraphMessage

# ─── LLaVA client ────────────────────────────────────────────────────────
vision_llm = ChatOllama(model="llava:13b", base_url="http://localhost:11434")

SYSTEM_PROMPT = (
    "Return ONLY the licence‑plate string you see in the image, "
    "Output only the plate number with NO extra words. 7 to 11 charecters max Examples:\n"
    "- UK: KY69WMN\n"
    "- India: KA01AB1234\n"
    "- BH‑series: 22BH6517A\n"
    "If no plate is visible, reply exactly: NONE"
)

# ─── Agent ───────────────────────────────────────────────────────────────
def ocr_agent_llm(
    state: GraphMessage,
    _config: RunnableConfig | None = None
) -> GraphMessage:
    msg = state

    if msg.data is None or "image_path" not in msg.data:
        return GraphMessage(
            role="assistant",
            text="⚠️ No image received. Please upload a licence‑plate photo."
        )

    img_path: str = cast(str, msg.data["image_path"])

    # Base64‑embed image
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    image_dict = {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
    }

    llm_resp = vision_llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=[image_dict])
    ])

    # ── SAFE extraction: handle str | list union ─────────────────────────
    content = llm_resp.content
    if isinstance(content, str):
        plate_raw = content
    else:   # list – grab all strings and join
        plate_raw = " ".join(c for c in content if isinstance(c, str))

    plate_clean = re.sub(r"[^A-Z0-9]", "", plate_raw.strip().upper())

    if plate_clean == "NONE" or not plate_clean:
        return GraphMessage(
            role="assistant",
            text="Sorry, I couldn’t read a licence‑plate from that image."
        )

    return GraphMessage(
        role="delegate",
        text="process",
        data={"plate": plate_clean}
    )
