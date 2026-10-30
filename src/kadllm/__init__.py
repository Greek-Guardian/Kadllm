"""WideTuning model adaptation and GRBeam shared-prefix decoding for Kadllm."""

from .grbeam import GRBeamAdapter, GRBeamResult, PrefillResult, grbeam_search
from .widetuning import GatedMLPSpec, WideTuningMLP, install_widetuning
from .ops import causal_conv1d, gated_delta, prefix_attention

__all__ = [
    "GRBeamResult",
    "PrefillResult",
    "GRBeamAdapter",
    "grbeam_search",
    "GatedMLPSpec",
    "WideTuningMLP",
    "install_widetuning",
    "causal_conv1d",
    "gated_delta",
    "prefix_attention",
]
