# app.py  – compact chat UI + backend‑graph tab
from __future__ import annotations
import gradio as gr
import graphviz
from pathlib import Path
from typing import Any, List, Tuple

from schema import GraphMessage
from graph import build_graph

# ───────── Build graphs once ─────────
compiled_graph, raw_graph = build_graph()

# ───────── Render static PNG once ─────────
GRAPH_PNG = Path("langgraph_topology.png")

def render_graph_png() -> None:
    """Generate langgraph_topology.png from the raw StateGraph."""
    dot = graphviz.Digraph(
        "LangGraph",
        format="png",
        graph_attr={"rankdir": "LR", "splines": "ortho"},
        node_attr={"fontsize": "10"},
    )

    # Sentinel nodes
    dot.node("START", shape="circle", style="filled", fillcolor="#e0e0e0")
    dot.node("END",   shape="doublecircle", style="filled", fillcolor="#e0e0e0")

    # Agent nodes
    for node in raw_graph.nodes:
        dot.node(node, shape="box", style="rounded,filled", fillcolor="#ffffff")

    # High‑level logical edges
    logical_edges = [
        ("START",   "chat",     "entry"),
        ("chat",    "ocr",      "delegate: ocr"),
        ("chat",    "END",      "normal reply"),
        ("ocr",     "process",  "delegate: process"),
        ("ocr",     "END",      "no plate"),
        ("process", "chat",     "delegate: chat"),
        ("process", "END",      "lookup failed"),
    ]
    for src, dst, label in logical_edges:
        dot.edge(src, dst, label=label)

    dot.render(GRAPH_PNG.stem, cleanup=True)

render_graph_png()        # generate PNG once

# ───────── helper to pull assistant reply ─────────
def _extract_text(obj: Any) -> str:
    if isinstance(obj, GraphMessage):
        return obj.text or ""
    if isinstance(obj, dict):
        if "role" in obj and "text" in obj:
            return obj.get("text", "")
        for v in obj.values():
            t = _extract_text(v)
            if t:
                return t
    if isinstance(obj, (list, tuple)):
        for v in obj:
            t = _extract_text(v)
            if t:
                return t
    return "⚠️ Unexpected response format."

# ───────── chat callback ─────────
def chat_step(
    history: List[Tuple[str, str]],
    user_text: str,
    image_path: str | None
) -> Tuple[List[Tuple[str, str]], None, str]:
    history.append((user_text, ""))

    msg = GraphMessage(
        role="user",
        text=user_text,
        data={"image_path": image_path} if image_path else None,
    )
    reply = _extract_text(compiled_graph.invoke(msg))
    history[-1] = (user_text, reply)

    return history, None, ""   # clear image & textbox

# ───────── UI layout ─────────
with gr.Blocks(title="Multi‑agent Licence‑Plate Demo") as demo:
    with gr.Tabs():
        # ─── Chat tab ───
        with gr.Tab("Chat"):
            chatbot = gr.Chatbot(label="Conversation", height=420)
            with gr.Row():
                txt_in = gr.Textbox(
                    placeholder="Type your message...",
                    show_label=False,
                    lines=1,
                    scale=4,
                )
                img_in = gr.Image(
                    type="filepath",
                    show_label=False,
                    scale=1,
                    height=180,
                    width=180,
                )
                btn = gr.Button("Send", variant="primary", scale=1)

            btn.click(
                fn=chat_step,
                inputs=[chatbot, txt_in, img_in],
                outputs=[chatbot, img_in, txt_in],
                show_progress="minimal",
            )
            txt_in.submit(
                fn=chat_step,
                inputs=[chatbot, txt_in, img_in],
                outputs=[chatbot, img_in, txt_in],
                show_progress="minimal",
            )

        # ─── Graph tab ───
        with gr.Tab("Backend Graph"):
            gr.Image(
                value=str(GRAPH_PNG),
                label="LangGraph topology",
                interactive=False,
                height=420,
            )

if __name__ == "__main__":
    demo.launch()
