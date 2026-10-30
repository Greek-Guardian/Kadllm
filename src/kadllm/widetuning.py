"""WideTuning: function-preserving FFN width expansion for model adaptation.

The added down projection starts at zero. Old and new weights are separate
Parameters, so freezing survives AdamW weight decay without gradient masks.
"""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class GatedMLPSpec:
    """Attribute mapping for y = down(activation(gate(x)) * up(x)).

    Shapes alone do not prove this equation. The caller explicitly selects
    modules whose forward has these semantics. Fused/MoE projections need
    another adapter rather than heuristic matching.
    """

    gate: str = "gate_proj"
    up: str = "up_proj"
    down: str = "down_proj"
    activation: str = "act_fn"


DEFAULT_GATED_MLP_SPEC = GatedMLPSpec()


class WideTuningMLP(nn.Module):
    def __init__(
        self,
        mlp: nn.Module,
        extra_width: int,
        *,
        spec: GatedMLPSpec = DEFAULT_GATED_MLP_SPEC,
        freeze_base: bool = True,
    ):
        super().__init__()
        if (
            not isinstance(extra_width, int)
            or isinstance(extra_width, bool)
            or extra_width <= 0
        ):
            raise ValueError("extra_width must be a positive integer")
        gate, up, down = (
            getattr(mlp, name, None) for name in (spec.gate, spec.up, spec.down)
        )
        if not all(type(m) is nn.Linear for m in (gate, up, down)):
            raise TypeError("only plain nn.Linear projections are supported initially")
        if (
            gate.in_features != up.in_features
            or gate.out_features != up.out_features
            or down.in_features != gate.out_features
        ):
            raise ValueError("incompatible gated MLP projection shapes")
        if len({(m.weight.device, m.weight.dtype) for m in (gate, up, down)}) != 1:
            raise ValueError("projections must share dtype and device")
        if gate.weight.is_meta:
            raise ValueError("materialize the base weights before expansion")
        activation = getattr(mlp, spec.activation, None)
        if not callable(activation):
            raise TypeError("activation must be callable")
        self.base = deepcopy(mlp)
        self.spec = spec
        self.extra_width = extra_width
        self.original_width = gate.out_features
        self.freeze_base = freeze_base
        self.base.requires_grad_(not freeze_base)
        kwargs = {"device": gate.weight.device, "dtype": gate.weight.dtype}
        self.extra_gate = nn.Linear(
            gate.in_features, extra_width, bias=gate.bias is not None, **kwargs
        )
        self.extra_up = nn.Linear(
            up.in_features, extra_width, bias=up.bias is not None, **kwargs
        )
        self.extra_down = nn.Linear(
            extra_width, down.out_features, bias=False, **kwargs
        )
        with torch.no_grad():
            self.extra_down.weight.zero_()
            for module in (self.extra_gate, self.extra_up):
                if module.bias is not None:
                    module.bias.zero_()
        self.train(mlp.training)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        act = getattr(self.base, self.spec.activation)
        added = self.extra_down(act(self.extra_gate(x)) * self.extra_up(x))
        return self.base(x) + added

    @torch.no_grad()
    def to_dense(self) -> nn.Module:
        """Merge branches into a conventional wider MLP for inference.

        Forward equivalence is mathematical; GEMM rounding can differ. The
        returned module has ordinary projection state_dict keys. Architecture
        config/save_pretrained metadata remains the caller's responsibility.
        """
        result = deepcopy(self.base)
        for name, extra, dimension in (
            (self.spec.gate, self.extra_gate, 0),
            (self.spec.up, self.extra_up, 0),
            (self.spec.down, self.extra_down, 1),
        ):
            old = getattr(result, name)
            weight = torch.cat((old.weight, extra.weight), dim=dimension)
            new = nn.Linear(
                weight.shape[1],
                weight.shape[0],
                bias=old.bias is not None,
                device=weight.device,
                dtype=weight.dtype,
            )
            new.weight.copy_(weight)
            if old.bias is not None:
                new.bias.copy_(
                    torch.cat((old.bias, extra.bias)) if dimension == 0 else old.bias
                )
            setattr(result, name, new)
        if hasattr(result, "intermediate_size"):
            result.intermediate_size = self.original_width + self.extra_width
        result.train(self.training)
        return result


def install_widetuning(
    model: nn.Module,
    widths: Mapping[str, int],
    *,
    spec: GatedMLPSpec = DEFAULT_GATED_MLP_SPEC,
    freeze_base: bool = True,
) -> dict[str, int]:
    """Replace explicitly named MLPs. Construct the optimizer AFTER this call.

    Only selected base MLPs are frozen; freeze the rest of the model yourself
    if desired. Save the returned path/extra-width mapping with the state_dict,
    and apply the same mapping to a fresh base model before loading it.
    """
    paths = list(widths)
    if any(not path for path in paths):
        raise ValueError("use WideTuningMLP directly to wrap a root module")
    if any(a != b and b.startswith(a + ".") for a in paths for b in paths):
        raise ValueError("selected module paths may not overlap")
    # 先验证并构建全部替换项，避免输入错误时只修改一半模型。
    replacements = {
        path: WideTuningMLP(
            model.get_submodule(path), width, spec=spec, freeze_base=freeze_base
        )
        for path, width in widths.items()
    }
    for path, replacement in replacements.items():
        parent, _, name = path.rpartition(".")
        setattr(model.get_submodule(parent), name, replacement)
    return dict(widths)
