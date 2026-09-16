"""
CXR Insight - Flask application entry point.

Routes
------
GET  /                 dashboard UI
POST /api/upload       upload an X-ray image, get back metadata + a preview URL
POST /api/report       generate a report for a previously-uploaded image
POST /api/chat         ask the assistant a question about a generated report
GET  /uploads/<file>   serve an uploaded image back to the browser
"""

from __future__ import annotations

import logging
import uuid

from flask import Flask, jsonify, request, send_from_directory, session
from flask_cors import CORS

from config import Config
from models.llm_assistant import LLMAssistant
from models.vlm_report_generator import ReportGenerator
from utils.image_utils import allowed_file, load_rgb_image, save_upload
from utils.metadata import extract_metadata

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("cxr_insight")

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

# Models are loaded once at process start-up (each falls back to a demo
# mode automatically if the real weights aren't configured/available -
# see models/vlm_report_generator.py and models/llm_assistant.py).
logger.info("Loading report generator...")
report_generator = ReportGenerator()
logger.info("Report generator mode: %s", report_generator.mode)

logger.info("Loading chat assistant...")
llm_assistant = LLMAssistant()
logger.info("Chat assistant mode: %s", llm_assistant.mode)

# In-memory session store: session_id -> {metadata, report, history}
# Fine for a single-process demo/dev deployment; swap for Redis/DB for
# multi-worker production use.
SESSIONS: dict = {}


@app.route("/")
def index():
    from flask import render_template

    return render_template(
        "index.html",
        vlm_mode=report_generator.mode,
        llm_mode=llm_assistant.mode,
    )


@app.route("/api/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify({"error": "No file part 'image' in request"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {sorted(Config.ALLOWED_EXTENSIONS)}"}), 400

    saved_path = save_upload(file)
    metadata = extract_metadata(file.filename)

    session_id = uuid.uuid4().hex
    SESSIONS[session_id] = {
        "image_path": str(saved_path),
        "metadata": metadata.as_dict(),
        "report": None,
        "history": [],
    }

    return jsonify(
        {
            "session_id": session_id,
            "image_url": f"/uploads/{saved_path.name}",
            "metadata": metadata.as_dict(),
        }
    )


@app.route("/api/report", methods=["POST"])
def generate_report():
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id")
    entry = SESSIONS.get(session_id)
    if not entry:
        return jsonify({"error": "Unknown or expired session_id. Upload an image first."}), 404

    try:
        image = load_rgb_image(entry["image_path"])
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to load image for report generation")
        return jsonify({"error": f"Could not read the uploaded image: {exc}"}), 500

    result = report_generator.generate_report(image, entry["metadata"])
    entry["report"] = result["report"]
    entry["history"] = []

    return jsonify(
        {
            "report": result["report"],
            "mode": result["mode"],
            "metadata": entry["metadata"],
        }
    )


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id")
    question = (data.get("question") or "").strip()

    entry = SESSIONS.get(session_id)
    if not entry:
        return jsonify({"error": "Unknown or expired session_id. Upload an image first."}), 404
    if not entry.get("report"):
        return jsonify({"error": "Generate a report before starting the chat."}), 400
    if not question:
        return jsonify({"error": "Question is empty."}), 400

    result = llm_assistant.answer(
        report=entry["report"],
        metadata=entry["metadata"],
        question=question,
        history=entry["history"],
    )

    entry["history"].append({"role": "user", "content": question})
    entry["history"].append({"role": "assistant", "content": result["reply"]})

    return jsonify({"reply": result["reply"], "mode": result["mode"]})


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(Config.UPLOAD_FOLDER, filename)


@app.route("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "vlm_mode": report_generator.mode,
            "llm_mode": llm_assistant.mode,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=Config.PORT, debug=Config.DEBUG)
