# CXR Insight – Chest X-ray Report Generation & AI Assistant

A modular AI-powered web application that generates radiology-style reports from chest X-ray images using Vision-Language Models (VLMs) and provides interactive explanations through a Large Language Model (LLM).

The system allows medical students and practitioners to upload X-ray images, generate automated reports, and interact with an AI assistant to understand the results.

## Overview

This project integrates computer vision and natural language processing to assist users in interpreting chest X-ray images.

The application workflow:
1. Upload a chest X-ray image
2. Extract patient metadata from the filename
3. Generate a medical-style report using Swin-T5
4. Ask questions about the report using a LLaMA-3.1 AI assistant

Users can:
- Upload chest X-ray images
- Automatically generate radiology-style reports
- Chat with an AI assistant to understand the report
- View patient metadata extracted from the dataset filename
- Interact with the system through a clean medical-style interface

## Vision-Language Model (VLM)

The system uses a **Swin Transformer + T5** architecture to generate radiology-style reports from X-ray images (`models/vlm_report_generator.py`). The Swin Transformer encodes the image into patch-level features, which are linearly projected into T5's hidden size and fed to T5's decoder as cross-attention context, so T5 generates the report text conditioned on the image instead of on source tokens.

## Large Language Model (LLM)

Used for contextual medical explanations (`models/llm_assistant.py`).

- Model: **Meta LLaMA-3.1-8B-Instruct**
- HuggingFace: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct (gated — you must request access and use your own `HF_TOKEN`)

## Dataset

Training data was obtained from the **CheXpert dataset**:
https://stanfordaimi.azurewebsites.net/datasets/chexpert-chest-xray

From the filename/path the system extracts: **View, Age, Gender, Ethnicity** (see `utils/metadata.py`). CheXpert's public release keeps demographics in `train.csv` rather than the filename itself — the extractor supports both that CSV-lookup layout and flat filenames like `patient00007_58_Male_White_frontal.png` that bake the fields directly into the name.

## Features

- Automated Chest X-ray Report Generation using Swin-T5
- AI Chat Assistant powered by LLaMA-3.1
- Patient Metadata Extraction from dataset filenames
- X-ray Image Visualization within the web interface
- Medical-style structured reports
- Dark UI medical dashboard
- Real-time interaction through a Flask backend

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| Frontend | HTML5, CSS3, JavaScript |
| Deep Learning | PyTorch, HuggingFace Transformers |
| Vision Model | Swin Transformer |
| Language Model | LLaMA-3.1 |
| Image Processing | Torchvision, Pillow |
| Development | VS Code |
| Version Control | Git, GitHub |

## Project layout

```
project_CXR/
├── app.py                     Flask app (routes, session state)
├── config.py                  Env-driven configuration
├── requirements.txt
├── .env.example                Copy to .env and fill in what you have
├── models/
│   ├── vlm_report_generator.py Swin-T5 architecture + report generator wrapper
│   └── llm_assistant.py        LLaMA-3.1 chat wrapper + fallback assistant
├── utils/
│   ├── metadata.py              Filename/CSV -> patient metadata
│   └── image_utils.py           Upload validation, image loading
├── train/
│   ├── dataset.py               CheXpert Dataset for report-generation training
│   └── train_swin_t5.py         Fine-tuning script -> checkpoints/*.pt
├── scripts/
│   └── setup_models.py          Pre-download HF model weights
├── templates/index.html         Dashboard UI
├── static/{css,js}              Dashboard styling + client logic
├── uploads/                      Uploaded images (gitignored, kept via .gitkeep)
└── reports/                      Reserved for saved report exports
```

## Getting started

### 1. Install dependencies

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

### 2. Configure environment

```bash
copy .env.example .env        # Windows
# cp .env.example .env        # macOS/Linux
```

The app **runs immediately with no further setup**, in demo mode:
- the report generator uses a clearly-labeled template report derived from basic image statistics and the extracted metadata,
- the chat assistant answers by extracting the relevant sentence from the generated report.

This lets you exercise the full upload → report → chat flow, and the whole UI, without any GPU or model downloads.

### 3. (Optional) Enable the real Swin-T5 report generator

The public CheXpert release does not ship free-text reports, so there's no ready-made "Swin-T5 CXR" checkpoint to download — you train one:

```bash
python train/train_swin_t5.py \
  --csv CheXpert-v1.0/train.csv \
  --images-root CheXpert-v1.0 \
  --epochs 5 --batch-size 8 \
  --output checkpoints/swin_t5_cxr.pt
```

(Your `train.csv` needs a free-text `Report` column — pair CheXpert images with a report-generation split such as CheXpert-Plus, or your own annotations.)

Then in `.env`:
```
SWIN_T5_CHECKPOINT=checkpoints/swin_t5_cxr.pt
```

### 4. (Optional) Enable the real LLaMA-3.1 assistant

1. Request access to [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) on Hugging Face.
2. Create a Hugging Face access token and put it in `.env`:
   ```
   HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
   ```
3. Optionally pre-download the weights: `python scripts/setup_models.py --llm`

A CUDA GPU with ≥16GB VRAM is recommended (4-bit quantization is used automatically when `bitsandbytes` is installed); CPU inference works but is slow.

### 5. Run the app

```bash
python app.py
```

Open http://localhost:5000 — the header badges show whether each model is running live (`swin-t5` / `llama-3.1`) or in demo mode.

## API

| Endpoint | Method | Body | Returns |
|---|---|---|---|
| `/api/upload` | POST | multipart `image` file | `session_id`, `image_url`, extracted `metadata` |
| `/api/report` | POST | `{"session_id": ...}` | generated `report`, `mode` |
| `/api/chat` | POST | `{"session_id": ..., "question": ...}` | assistant `reply`, `mode` |
| `/api/health` | GET | – | current model modes |

## Disclaimer

This tool is for education/research. Generated reports — whether from the demo template or a trained Swin-T5 model — are **not a medical diagnosis** and must always be confirmed by a qualified radiologist before any clinical use.
