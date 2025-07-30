from __future__ import annotations

from typing import Dict, Optional
from schema import GraphMessage                        # <- your Pydantic model
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableConfig
import re

# ─────────────────────────  LLM SET‑UP  ──────────────────────────
llm = ChatOllama(model="llama3:8b", base_url="http://localhost:11434")
SYSTEM_PROMPT = (
    "You are Zen, a concise, helpful assistant for a multi‑agent demo. "
    "Capabilities:\n"
    "• Normal open‑domain chat.\n"
    "• If the user uploads a licence‑plate image, you delegate to the OCR agent.\n"
    "• After OCR + vehicle lookup, you summarise the details.\n"
    "If the user merely *asks about* the workflow, clearly explain how it operates.\n"
    "Avoid saying you cannot see images—just explain the process if needed."
)
# Anything that sounds like “look up a licence plate”
_PLATE_INTENT_RE = re.compile(
    r"\b(look\s*up|check|decode|identify|what\s+car|vehicle\s+details?)\b",
    re.I,
)
# ─────────────────────────  AGENT LOGIC  ─────────────────────────
def chat_agent(
    state: GraphMessage,
    _config: RunnableConfig | None = None
) -> GraphMessage:
    msg = state

    # ➊ Helper flags
    has_image = bool(msg.data and msg.data.get("image_path"))
    # ➋ Intent + image  → delegate once to OCR
    if has_image and ("vehicle" not in (msg.data or {})):
        return GraphMessage(
            role="delegate",
            text="ocr",
            data=msg.data
    )

    # ➌ Intent but NO image → prompt user to upload one
    if _PLATE_INTENT_RE.search(msg.text or "") and not has_image:
        return GraphMessage(
            role="assistant",
            text="Please upload a clear photo of the licence plate so I can analyse it."
    )

    # ➍ Vehicle data already present → summarise for the user
    if msg.data and "vehicle" in msg.data:
        v = msg.data["vehicle"]

        reply_lines = [
            "Here are the details I found:",
            f"• Reg: {v.get('registrationNumber', 'N/A')}",
            f"• Make/Model: {v.get('make', 'N/A')} {v.get('model', '')}".strip(),
            f"• Colour: {v.get('colour', 'N/A')}",
            f"• First Reg: {v.get('yearOfManufacture', 'N/A')}",
        ]
        return GraphMessage(role="assistant", text="\n".join(reply_lines))

    # ── Otherwise: normal small‑talk via LLM ─────────────────────
    human_text = msg.text or ""
    ai_msg = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=human_text)
    ])

    content = (
        ai_msg.content
        if isinstance(ai_msg.content, str)
        else " ".join(c if isinstance(c, str) else str(c) for c in ai_msg.content)
    )

    return GraphMessage(role="assistant", text=content)
