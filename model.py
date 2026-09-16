from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import dolphin
from coral_components import (
    DoctorThresholdShift,
    MaskedAttentionPooling,
    MaskedTemporalConvBlock,
    coral_logits_to_class_probs,
)


class DolphinEncoderCORALClassifier(nn.Module):
    """Dolphin encoder with temporal pooling and clinician-aware CORAL heads."""

    def __init__(
        self,
        dolphin_model_name: str = "small",
        dolphin_model_dir: Optional[str] = None,
        device: str = "cuda",
        use_layer: int = 9,
        freeze_s2t: bool = True,
        freeze_layer: Optional[int] = 6,
        use_specaug: bool = True,
        head_hidden_size: int = 64,
        head_dropout_rate: float = 0.5,
        num_classes: int = 4,
        temporal_cnn_bottleneck_dim: int = 64,
        temporal_cnn_kernel_size: int = 5,
        temporal_cnn_dropout_rate: float = 0.1,
        attention_hidden_dim: int = 32,
        attention_dropout_rate: float = 0.1,
        num_doctors: int = 3,
        coral_min_threshold_gap: float = 1e-3,
        doctor_query_dim: int = 32,
        doctor_num_attention_heads: int = 2,
        doctor_num_self_attention_layers: int = 1,
        doctor_query_dropout: float = 0.1,
        doctor_ffn_multiplier: int = 2,
    ):
        super().__init__()
        
        self.use_layer = use_layer
        self.use_specaug = use_specaug
        self.num_classes = num_classes
        self.coral_min_threshold_gap = coral_min_threshold_gap

        dolphin_model = dolphin.load_model(
            dolphin_model_name,
            model_dir=dolphin_model_dir,
            device=device,
        )
        self.s2t_model = dolphin_model.s2t_model
        self.encoder = self.s2t_model.encoder

        num_encoder_layers = len(self.encoder.encoders)
        layer_index = self.use_layer - 1
        self._target_hidden: Optional[torch.Tensor] = None
        self.encoder.encoders[layer_index].register_forward_hook(
            self._capture_target_hidden
        )

        output_size = getattr(self.encoder, "output_size", None)
        feature_dim = int(output_size())
        self.temporal_cnn = MaskedTemporalConvBlock(
            feature_dim,
            temporal_cnn_bottleneck_dim,
            temporal_cnn_kernel_size,
            temporal_cnn_dropout_rate,
        )
        self.attention_pooling = MaskedAttentionPooling(
            feature_dim,
            attention_hidden_dim,
            attention_dropout_rate,
        )
        self.severity_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, head_hidden_size),
            nn.GELU(),
            nn.Dropout(head_dropout_rate),
            nn.Linear(head_hidden_size, 1),
        )
        self.doctor_shift_module = DoctorThresholdShift(
            num_doctors,
            doctor_query_dim,
            doctor_num_attention_heads,
            doctor_num_self_attention_layers,
            doctor_query_dropout,
            doctor_ffn_multiplier,
        )

        self.threshold_start = nn.Parameter(torch.tensor(-1.0))
        num_thresholds = self.num_classes - 1
        if num_thresholds > 1:
            initial_gap = 1.0 - self.coral_min_threshold_gap
            raw_gap = torch.log(torch.expm1(torch.tensor(initial_gap)))
            self.threshold_gaps = nn.Parameter(
                torch.full((num_thresholds - 1,), float(raw_gap))
            )
        else:
            self.register_parameter("threshold_gaps", None)

        self._configure_encoder_training(
            freeze_s2t,
            freeze_layer,
            layer_index,
            num_encoder_layers,
        )
        self.to(device)

    @staticmethod
    def _unwrap_tensor(value):
        while isinstance(value, (tuple, list)):
            value = value[0]
        return value

    def _capture_target_hidden(self, _module, _inputs, outputs):
        self._target_hidden = self._unwrap_tensor(outputs)

    def _configure_encoder_training(
        self,
        freeze_s2t: bool,
        freeze_layer: Optional[int],
        layer_index: int,
        num_encoder_layers: int,
    ):
        for parameter in self.s2t_model.parameters():
            parameter.requires_grad = not freeze_s2t
        if not freeze_s2t:
            return

        first_trainable_layer = (
            layer_index
            if freeze_layer is None
            else max(0, min(int(freeze_layer), num_encoder_layers))
        )
        for layer in self.encoder.encoders[first_trainable_layer:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True

    def get_global_thresholds(self) -> torch.Tensor:
        first = self.threshold_start.view(1)
        if self.threshold_gaps is None:
            return first
        gaps = F.softplus(self.threshold_gaps) + self.coral_min_threshold_gap
        return torch.cat((first, first + gaps.cumsum(dim=0)))

    def train(self, mode: bool = True):
        super().train(mode)
        for name in ("frontend", "normalize"):
            module = getattr(self.s2t_model, name, None)
            if module is not None:
                module.eval()
        specaug = getattr(self.s2t_model, "specaug", None)
        if specaug is not None:
            specaug.train(mode and self.use_specaug)
        self.encoder.train(
            mode
            and any(
                parameter.requires_grad
                for parameter in self.encoder.parameters()
            )
        )
        return self

    @staticmethod
    def _encoder_lengths(encoder_outputs, batch_size: int):
        if not isinstance(encoder_outputs, (tuple, list)):
            return None
        for value in encoder_outputs[1:]:
            if (
                torch.is_tensor(value)
                and value.ndim == 1
                and value.numel() == batch_size
            ):
                return value
        return None

    def _encode(self, speech, speech_lengths):
        self._target_hidden = None
        with torch.no_grad():
            features, feature_lengths = self.s2t_model.frontend(
                speech, speech_lengths
            )
            specaug = getattr(self.s2t_model, "specaug", None)
            if self.training and self.use_specaug and specaug is not None:
                features, feature_lengths = specaug(
                    features, feature_lengths
                )
            features, feature_lengths = self.s2t_model.normalize(
                features, feature_lengths
            )

        encoder_requires_grad = any(
            parameter.requires_grad for parameter in self.encoder.parameters()
        )
        with torch.set_grad_enabled(encoder_requires_grad):
            encoder_outputs = self.encoder(features, feature_lengths)
            hidden = self._target_hidden
            hidden = self._unwrap_tensor(hidden)
            if hidden.size(0) != speech.size(0) and hidden.size(1) == speech.size(0):
                hidden = hidden.transpose(0, 1)

            batch_size, time_steps, _ = hidden.shape
            output_lengths = self._encoder_lengths(
                encoder_outputs, batch_size
            )
            if output_lengths is None or output_lengths.max() > time_steps:
                feature_steps = max(features.size(1), 1)
                output_lengths = torch.ceil(
                    feature_lengths.to(hidden.device).float()
                    * (time_steps / feature_steps)
                ).long()
            else:
                output_lengths = output_lengths.to(hidden.device).long()
            output_lengths.clamp_(min=1, max=time_steps)

            padding_mask = (
                torch.arange(time_steps, device=hidden.device).unsqueeze(0)
                >= output_lengths.unsqueeze(1)
            )
            return hidden.masked_fill(padding_mask.unsqueeze(-1), 0), padding_mask

    def forward(self, speech, speech_lengths):
        features, padding_mask = self._encode(speech, speech_lengths)
        features = self.temporal_cnn(features, padding_mask)
        pooled, _ = self.attention_pooling(features, padding_mask)
        severity_score = self.severity_head(pooled)

        global_thresholds = self.get_global_thresholds()
        global_logits = severity_score - global_thresholds.unsqueeze(0)
        global_probs = coral_logits_to_class_probs(global_logits)

        doctor_shifts = self.doctor_shift_module()["doctor_shifts"]
        doctor_thresholds = (
            global_thresholds.unsqueeze(0) + doctor_shifts.unsqueeze(1)
        )
        doctor_logits = (
            severity_score.unsqueeze(1) - doctor_thresholds.unsqueeze(0)
        )
        return {
            "severity_score": severity_score,
            "global_thresholds": global_thresholds,
            "global_coral_logits": global_logits,
            "global_class_probs": global_probs,
            "global_predictions": (global_logits > 0).sum(dim=-1),
            "doctor_shifts": doctor_shifts,
            "doctor_thresholds": doctor_thresholds,
            "doctor_coral_logits": doctor_logits,
            "doctor_class_probs": coral_logits_to_class_probs(doctor_logits),
            "doctor_predictions": (doctor_logits > 0).sum(dim=-1),
        }
