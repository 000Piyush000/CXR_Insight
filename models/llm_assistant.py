"""
Chat assistant that answers questions about a generated report.

Uses meta-llama/Meta-Llama-3.1-8B-Instruct via transformers when it is
available (HF_TOKEN set with access granted, and enough RAM/VRAM to load
it - 4-bit quantization is used automatically on CUDA if bitsandbytes is
installed). Otherwise falls back to a lightweight extractive assistant
that answers directly from the report text and conversation, so the chat
feature works end-to-end even without the 8B model downloaded.
"""

from __future__ import annotations

import logging
import re

from config import Config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful radiology assistant embedded in a teaching tool called "
    "CXR Insight. You explain automated chest X-ray report findings in clear, "
    "educational language for medical students. You are not a physician and "
    "your output is not a diagnosis; always encourage confirmation by a "
    "qualified radiologist for real patient care."
)


class LLMAssistant:
    def __init__(self):
        self.mode = "demo"
        self.pipe = None

        if Config.FORCE_DEMO_MODE:
            logger.info("FORCE_DEMO_MODE set - using extractive fallback chat assistant.")
            return

        if not Config.HF_TOKEN:
            logger.warning(
                "No HF_TOKEN set - LLaMA-3.1 is a gated model and can't be "
                "downloaded. Using extractive fallback chat assistant."
            )
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            device = Config.DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
            quant_kwargs = {}
            if device == "cuda":
                try:
                    from transformers import BitsAndBytesConfig

                    quant_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
                except ImportError:
                    pass

            tokenizer = AutoTokenizer.from_pretrained(Config.LLM_MODEL_ID, token=Config.HF_TOKEN)
            model = AutoModelForCausalLM.from_pretrained(
                Config.LLM_MODEL_ID,
                token=Config.HF_TOKEN,
                device_map="auto" if device == "cuda" else None,
                **quant_kwargs,
            )
            if device != "cuda":
                model.to(device)

            self.pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
            self.mode = "llama-3.1"
            logger.info("Loaded %s for the chat assistant.", Config.LLM_MODEL_ID)
        except Exception:
            logger.exception("Failed to load LLaMA-3.1 - falling back to extractive assistant.")
            self.pipe = None
            self.mode = "demo"

    def answer(self, report: str, metadata: dict, question: str, history: list | None = None) -> dict:
        history = history or []
        if self.mode == "llama-3.1" and self.pipe is not None:
            messages = [{"role": "system", "content": f"{SYSTEM_PROMPT}\n\nReport:\n{report}"}]
            messages.extend(history)
            messages.append({"role": "user", "content": question})

            output = self.pipe(
                messages,
                max_new_tokens=Config.MAX_CHAT_TOKENS,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )
            generated = output[0]["generated_text"]
            reply = generated[-1]["content"] if isinstance(generated, list) else str(generated)
            return {"reply": reply, "mode": self.mode}

        return {"reply": self._extractive_answer(report, metadata, question), "mode": "demo"}

    @staticmethod
    def _extractive_answer(report: str, metadata: dict, question: str) -> str:
        q = question.lower().strip()
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", report) if s.strip()]

        def find(*keywords):
            for s in sentences:
                if any(k in s.lower() for k in keywords):
                    return s
            return None

        if any(k in q for k in ["age", "old"]):
            return f"The extracted patient age is {metadata.get('age', 'Unknown')}."
        if any(k in q for k in ["gender", "sex", "male", "female"]):
            return f"The extracted patient gender is {metadata.get('gender', 'Unknown')}."
        if any(k in q for k in ["view", "projection", "pa", "ap", "lateral"]):
            return f"This image was taken from the {metadata.get('view', 'Unknown')} view."
        if any(k in q for k in ["ethnicity", "race"]):
            return f"The extracted patient ethnicity is {metadata.get('ethnicity', 'Unknown')}."
        if any(k in q for k in ["summary", "summarize", "impression", "overall"]):
            impression = find("impression") or sentences[-1] if sentences else None
            return impression or "No impression could be extracted from the report."
        if any(k in q for k in ["effusion", "pneumothorax", "consolidation", "cardiomegaly", "opacity"]):
            match = find(*[k for k in ["effusion", "pneumothorax", "consolidation", "cardiomegaly", "opacity"] if k in q])
            return match or (
                "The report does not explicitly mention that finding; it may be "
                "absent or not assessed in this automated read."
            )

        return (
            "Here is the relevant part of the report I can find: "
            f"\"{sentences[0] if sentences else report}\"\n\n"
            "(Running in demo mode without the LLaMA-3.1 model loaded - set "
            "HF_TOKEN in .env with access to meta-llama/Meta-Llama-3.1-8B-Instruct "
            "for full conversational answers.)"
        )
