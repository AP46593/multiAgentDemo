# app.py
import tempfile
import gradio as gr
from textract_agent import extract_agent     # the object we just built

def ocr_pipeline(image):
    """Gradio callback."""
    # 1. Save upload to a temp file (Textract wants bytes, but our tool reads from path)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as t:
        image.save(t.name)

    # 2. Ask the agent to run the tool
    prompt = (
        f"Run textract_image on {t.name}. "
        "Return only the detected text; do not include any additional commentary."
    )
    result = extract_agent.run(prompt)
    return result

with gr.Blocks(title="Textract Demo") as demo:
    gr.Markdown("### 🖼️ Image‐to‐Text with Textract + LangChain Agent")
    inp = gr.Image(type="pil", label="Upload an image")
    out = gr.Textbox(label="Extracted text", lines=12)
    inp.change(ocr_pipeline, inputs=inp, outputs=out)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
