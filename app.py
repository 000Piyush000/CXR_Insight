import os
import io
import base64
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from dotenv import load_dotenv

from transformers import (
    SwinModel,
    T5ForConditionalGeneration,
    T5Tokenizer,
    AutoModelForCausalLM,
    AutoTokenizer
)

from transformers.modeling_outputs import BaseModelOutput

load_dotenv()

# ---------------- CONFIG ---------------- #

MODEL_PATH = "swin-t5-model.pth"
SWIN_MODEL_NAME = "microsoft/swin-base-patch4-window7-224"
T5_MODEL_NAME = "t5-base"
LLAMA_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALLOWED_EXTENSIONS = {'png','jpg','jpeg'}

# ---------------- MODEL ---------------- #

class ImageCaptioningModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.swin = SwinModel.from_pretrained(SWIN_MODEL_NAME)
        self.t5 = T5ForConditionalGeneration.from_pretrained(T5_MODEL_NAME)

        self.img_proj = nn.Linear(
            self.swin.config.hidden_size,
            self.t5.config.d_model
        )

    def forward(self, images):

        swin_outputs = self.swin(images)
        img_feats = swin_outputs.last_hidden_state

        img_feats = self.img_proj(img_feats)

        encoder_outputs = BaseModelOutput(last_hidden_state=img_feats)

        outputs = self.t5.generate(
            encoder_outputs=encoder_outputs,
            max_length=100,
            num_beams=4
        )

        return outputs


# ---------------- GLOBAL VARIABLES ---------------- #

swin_model = None
tokenizer = None
transform = None

llama_model = None
llama_tokenizer = None


# ---------------- LOAD SWIN T5 ---------------- #

def load_swin_model():

    global swin_model, tokenizer, transform

    print("Loading Swin-T5 model...")

    swin_model = ImageCaptioningModel()

    swin_model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )

    swin_model.to(DEVICE)
    swin_model.eval()

    tokenizer = T5Tokenizer.from_pretrained(T5_MODEL_NAME)

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])

    print("Swin-T5 loaded")


# ---------------- LOAD LLAMA ---------------- #

def load_llama():

    global llama_model, llama_tokenizer

    if not HF_TOKEN:
        print("HF token missing → chatbot disabled")
        return

    print("Loading Llama model...")

    llama_tokenizer = AutoTokenizer.from_pretrained(
        LLAMA_MODEL_NAME,
        token=HF_TOKEN
    )

    llama_model = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL_NAME,
        device_map="auto",
        token=HF_TOKEN
    )

    llama_model.eval()

    print("Llama loaded")


# ---------------- IMAGE REPORT ---------------- #

def generate_report(image_bytes):

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        swin_outputs = swin_model.swin(image)

        img_feats = swin_outputs.last_hidden_state

        img_feats = swin_model.img_proj(img_feats)

        encoder_outputs = BaseModelOutput(last_hidden_state=img_feats)

        ids = swin_model.t5.generate(
            encoder_outputs=encoder_outputs,
            max_length=100,
            num_beams=4
        )

    report = tokenizer.decode(ids[0], skip_special_tokens=True)

    return report


# ---------------- CHAT RESPONSE ---------------- #

def chat_answer(question, report):

    if llama_model is None:
        return "Chatbot unavailable"

    prompt = f"""
You are a medical assistant.

Report:
{report}

Question:
{question}
"""

    inputs = llama_tokenizer(prompt, return_tensors="pt").to(llama_model.device)

    outputs = llama_model.generate(
        **inputs,
        max_new_tokens=200
    )

    answer = llama_tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return answer.replace(prompt,"").strip()


# ---------------- PARSE PATIENT INFO ---------------- #

def parse_patient_info(filename):

    try:

        base = os.path.splitext(filename)[0]

        parts = base.split("-")

        ethnicity = parts[-1]
        gender = parts[-2]
        age = int(float(parts[-3]))

        view = "-".join(parts[2:-3])

        return {

            "view":view,
            "age":age,
            "gender":gender.capitalize(),
            "ethnicity":ethnicity.capitalize()

        }

    except:
        return None


# ---------------- FLASK APP ---------------- #

app = Flask(__name__)

app.secret_key = "secret"


# ---------------- ROUTES ---------------- #

@app.route("/")
def home():

    return render_template("upload.html")


# -------- REPORT GENERATION -------- #

@app.route("/predict", methods=["POST"])

def predict():

    if "image" not in request.files:

        flash("No image uploaded")

        return redirect("/")

    file = request.files["image"]

    if file.filename == "":
        flash("No selected file")
        return redirect("/")

    image_bytes = file.read()

    patient_info = parse_patient_info(file.filename)

    report = generate_report(image_bytes)

    image_data = base64.b64encode(image_bytes).decode("utf-8")

    return render_template(
        "result.html",
        report=report,
        image_data=image_data,
        patient_info=patient_info
    )


# -------- CHAT -------- #

@app.route("/chat", methods=["POST"])

def chat():

    data = request.get_json()

    question = data["question"]
    report = data["report_context"]

    answer = chat_answer(question, report)

    return jsonify({"answer":answer})


# ---------------- START SERVER ---------------- #

if __name__ == "__main__":

    print("Starting server...")

    load_swin_model()

    load_llama()

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )