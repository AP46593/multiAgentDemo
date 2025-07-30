from __future__ import annotations
from schema import GraphMessage
from typing import Tuple, Any
from langgraph.graph import StateGraph, START, END
from agents.chat_agent import chat_agent # import the ChatAgent
from agents.processing_agent import processing_agent # import the DVLA processing agent
from agents.ocr_agent import ocr_agent  # import the easyOCR agent
#from agents.ocr_agent_llm import ocr_agent_llm as ocr_agent # import the LLM-based OCR agent


# ───── routers (unchanged) ─────
def route_from_chat(msg: GraphMessage) -> str:
    if msg.role == "delegate" and msg.text == "ocr":
        return "ocr"
    return "END"

def route_from_ocr(msg: GraphMessage) -> str:
    if msg.role == "delegate" and msg.text == "process":
        return "process"
    return "END"

def route_from_process(msg: GraphMessage) -> str:
    if msg.role == "delegate" and msg.text == "chat":
        return "chat"
    return "END"

# ───── builder ─────
def build_graph() -> Tuple[Any, StateGraph]:
    """Return (compiled_runnable, raw_state_graph)."""
    g = StateGraph(GraphMessage)

    g.add_node("chat", chat_agent)
    g.add_node("ocr", ocr_agent)
    g.add_node("process", processing_agent)

    g.add_edge(START, "chat")

    g.add_conditional_edges("chat", route_from_chat,
                            {"ocr": "ocr", "END": END})
    g.add_conditional_edges("ocr", route_from_ocr,
                            {"process": "process", "END": END})
    g.add_conditional_edges("process", route_from_process,
                            {"chat": "chat", "END": END})

    compiled = g.compile()
    return compiled, g
