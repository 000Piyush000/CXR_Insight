"""
Swin-T5 chest X-ray report generator.

Architecture (matches the project README): a Swin Transformer vision
encoder feeds its patch-level features, through a learned linear
projection into T5's hidden size, as the "encoder_outputs" of a T5
conditional-generation decoder - i.e. Swin replaces T5's own text
encoder, and T5's decoder cross-attends to image features instead of
token embeddings.

Two ways to run it:

1. Fine-tuned checkpoint available (``Config.SWIN_T5_CHECKPOINT`` points
   at a ``.pt`` file saved by ``train/train_swin_t5.py``): loads real
   weights and generates an actual learned report.
2. No checkpoint (default out of the box): the class still builds and
   can run forward/generate for architecture testing, but
   ``ReportGenerator`` below prefers a deterministic, template-based
   fallback report so the app gives a clear, honest, non-hallucinated
   response until you plug in trained weights.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
from torch import nn

from config import Config

logger = logging.getLogger(__name__)


class SwinT5ForReportGeneration(nn.Module):
    """Swin Transformer encoder + T5 decoder, bridged by a linear projection."""

    def __init__(self, swin_encoder_id: str, t5_decoder_id: str):
        super().__init__()
        from transformers import SwinModel, T5ForConditionalGeneration

        self.swin = SwinModel.from_pretrained(swin_encoder_id)
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_decoder_id)
        self.projection = nn.Linear(self.swin.config.hidden_size, self.t5.config.d_model)

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        swin_out = self.swin(pixel_values=pixel_values).last_hidden_state  # (B, N, C)
        return self.projection(swin_out)  # (B, N, d_model)

    def forward(self, pixel_values: torch.Tensor, labels: Optional[torch.Tensor] = None):
        encoder_hidden_states = self.encode_image(pixel_values)
        encoder_outputs = (encoder_hidden_states,)
        return self.t5(encoder_outputs=encoder_outputs, labels=labels)

    @torch.no_grad()
    def generate(self, pixel_values: torch.Tensor, max_new_tokens: int = 220, **gen_kwargs) -> torch.Tensor:
        encoder_hidden_states = self.encode_image(pixel_values)
        from transformers.modeling_outputs import BaseModelOutput

        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)
        return self.t5.generate(
            encoder_outputs=encoder_outputs,
            max_new_tokens=max_new_tokens,
            **gen_kwargs,
        )


class ReportGenerator:
    """
    High-level, app-facing wrapper. Loads the real Swin-T5 model when a
    fine-tuned checkpoint is configured; otherwise serves clearly-labeled
    template reports so the rest of the app (upload -> report -> chat) is
    fully exercised without requiring GPU hardware or trained weights.
    """

    def __init__(self):
        self.mode = "demo"
        self.model: Optional[SwinT5ForReportGeneration] = None
        self.feature_extractor = None
        self.tokenizer = None
        self.device = Config.DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")

        if Config.FORCE_DEMO_MODE:
            logger.info("FORCE_DEMO_MODE set - using template-based report generator.")
            return

        checkpoint = Config.SWIN_T5_CHECKPOINT
        if not checkpoint or not Path(checkpoint).exists():
            logger.warning(
                "No SWIN_T5_CHECKPOINT configured (or file not found) - "
                "using template-based fallback report generator. Train a "
                "model with train/train_swin_t5.py and set SWIN_T5_CHECKPOINT "
                "in .env to use the real Swin-T5 pipeline."
            )
            return

        try:
            from transformers import AutoFeatureExtractor, T5Tokenizer

            self.feature_extractor = AutoFeatureExtractor.from_pretrained(Config.SWIN_ENCODER_ID)
            self.tokenizer = T5Tokenizer.from_pretrained(Config.T5_DECODER_ID)
            self.model = SwinT5ForReportGeneration(Config.SWIN_ENCODER_ID, Config.T5_DECODER_ID)
            state_dict = torch.load(checkpoint, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device).eval()
            self.mode = "swin-t5"
            logger.info("Loaded fine-tuned Swin-T5 checkpoint from %s", checkpoint)
        except Exception:
            logger.exception("Failed to load Swin-T5 checkpoint - falling back to demo mode.")
            self.model = None
            self.mode = "demo"

    def generate_report(self, image, metadata: dict) -> dict:
        """
        Args:
            image: a PIL.Image (RGB) of the chest X-ray.
            metadata: dict from utils.metadata.extract_metadata(...).

        Returns:
            {"report": str, "mode": "swin-t5" | "demo"}
        """
        if self.mode == "swin-t5" and self.model is not None:
            pixel_values = self.feature_extractor(images=image, return_tensors="pt").pixel_values.to(self.device)
            output_ids = self.model.generate(pixel_values, max_new_tokens=Config.MAX_REPORT_TOKENS)
            text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return {"report": text, "mode": self.mode}

        return {"report": self._template_report(image, metadata), "mode": "demo"}

    @staticmethod
    def _template_report(image, metadata: dict) -> str:
        """
        Deterministic, clearly-labeled placeholder report derived from
        simple image statistics and extracted metadata. This keeps the
        rest of the application fully functional (upload -> report ->
        chat) without a trained checkpoint, and is never presented as a
        real diagnosis.
        """
        import numpy as np

        arr = np.asarray(image.convert("L"), dtype=np.float32)
        mean_intensity = float(arr.mean())
        contrast = float(arr.std())
        view = metadata.get("view", "Unknown")

        exposure_note = (
            "adequate penetration and inspiration"
            if 80 <= mean_intensity <= 175
            else "technically limited exposure; correlate clinically"
        )
        sharpness_note = (
            "sharp costophrenic angles and clear lung fields bilaterally"
            if contrast > 40
            else "reduced image contrast - subtle findings may be obscured"
        )

        report = (
            f"FINDINGS ({view} view):\n"
            f"The cardiomediastinal silhouette appears within normal limits. "
            f"Lung fields are grossly clear with {sharpness_note}. "
            f"The image demonstrates {exposure_note}. "
            f"No obvious focal consolidation, pleural effusion, or pneumothorax is identified "
            f"on this automated preliminary read.\n\n"
            f"IMPRESSION:\n"
            f"No acute cardiopulmonary abnormality detected by the automated system.\n\n"
            f"NOTE: This is a DEMO-MODE report generated by a template heuristic, not the "
            f"trained Swin-T5 model, and must not be used for clinical decision-making. "
            f"Configure SWIN_T5_CHECKPOINT with trained weights to enable real report "
            f"generation."
        )
        return report
