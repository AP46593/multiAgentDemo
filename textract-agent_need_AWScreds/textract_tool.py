# OCR using AWS Textract
# This tool extracts text from images using AWS Textract.
# Keep credentials in environment variables or ~/.aws/credentials; for demos, IAM user with TextractFullAccess is fine.


# tools/textract_tool.py
import boto3
from langchain.tools import BaseTool          # or langchain_community.tools if you’re on ≥0.2


class TextractTool(BaseTool):
    name: str = "textract_image"               # <‑‑ required
    description: str = (                      # <‑‑ required
        "Extract raw text from an image file using AWS Textract. "
        "Input: absolute file path to an image; output: plain text."
    )

    # Optional: you can declare args_schema for validation; skipped here.

    def _run(self, image_path: str) -> str:
        """Synchronous execution."""
        client = boto3.client("textract")

        with open(image_path, "rb") as f:
            img_bytes = f.read()

        resp = client.detect_document_text(Document={"Bytes": img_bytes})
        lines = [
            item["DetectedText"]
            for item in resp.get("Blocks", [])
            if item["BlockType"] == "LINE"
        ]
        return "\n".join(lines)

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError("TextractTool does not support async yet")
