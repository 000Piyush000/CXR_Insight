# **CXR Insight – Chest X-ray Report Generation & AI Assistant**

A modular AI-powered web application that generates radiology-style reports from chest X-ray images using Vision-Language Models (VLMs) and provides interactive explanations through a Large Language Model (LLM).

The system allows medical students and practitioners to upload X-ray images, generate automated reports, and interact with an AI assistant to understand the results.

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/4e8e60c2-27a9-425b-b4ec-dd5901c908ef" />



Overview

This project integrates computer vision and natural language processing to assist users in interpreting chest X-ray images.

The application workflow:

1️⃣ Upload a chest X-ray image

2️⃣ Extract patient metadata from the filename

3️⃣ Generate a medical-style report using Swin-T5

4️⃣ Ask questions about the report using LLaMA-3.1 AI assistant

Users can:

Upload chest X-ray images

Automatically generate radiology-style reports

Chat with an AI assistant to understand the report

View patient metadata extracted from the dataset filename

Interact with the system through a clean medical-style interface


Vision-Language Model (VLM)

The system uses a Swin Transformer + T5 architecture to generate radiology-style reports from X-ray images.

Model:

Swin-T5 (Swin Transformer + T5)

Large Language Model (LLM)

Used for contextual medical explanations.

Model:

Meta LLaMA-3.1-8B Instruct

HuggingFace Model:

https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct

Dataset

Training data was obtained from the CheXpert dataset.

Dataset:

Stanford CheXpert Dataset
https://stanfordaimi.azurewebsites.net/datasets/chexpert-chest-xray

From this filename the system extracts:

View
Age
Gender
Ethnicity

Features

🔍 Automated Chest X-ray Report Generation using Swin-T5

💬 AI Chat Assistant powered by LLaMA-3.1

🧑‍⚕️ Patient Metadata Extraction from dataset filenames

🖼 X-ray Image Visualization within the web interface

📊 Medical-style structured reports

🌙 Dark UI medical dashboard

⚡ Real-time interaction through Flask backend



## Layer	------------Technology

Backend	-------------Python, Flask

Frontend	-----------HTML5, CSS3, JavaScript

Deep Learning	---------PyTorch, HuggingFace Transformers

Vision Model	--------Swin Transformer

Language Model	--------LLaMA-3.1

Image Processing-------------	Torchvision, Pillow

Development	----------VS Code
Version Control	-----------Git, GitHub

