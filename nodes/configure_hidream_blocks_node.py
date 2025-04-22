"""
configure_hidream_blocks_node.py
================================
Patches HiDream / SD‑3 UNet so RF‑Edit, PAG, SEG work, while remaining
compatible with Flux checkpoints.

Strategy:
  • Ensure .double_blocks / .single_blocks exist
  • Run Flux's inject_blocks() to add inner‑block hooks
  • Wrap ANY module whose forward() doesn't accept extra kwargs
    (image_tokens, pag_mask, seg_mask) so they are ignored
"""

from importlib import import_module
import inspect
from functools import wraps

from ..flux.layers import inject_blocks

# --------------------------------------------------------------
# Find BasicTransformerBlock in any diffusers version (0.20‑0.27)
# --------------------------------------------------------------
_BLOCK_PATHS = [
    "diffusers.models.unets.transformer_2d",
    "diffusers.models.transformer_2d",
    "diffusers.models.attention",
]
BasicTransformerBlock = None
for _p in _BLOCK_PATHS:
    try:
        _m = import_module(_p)
        BasicTransformerBlock = getattr(_m, "BasicTransformerBlock", None)
        if BasicTransformerBlock:
            break
    except ImportError:
        pass
if BasicTransformerBlock is None:
    raise ImportError("diffusers too old – BasicTransformerBlock not found")

_EXTRA_KEYS = {"image_tokens", "pag_mask", "seg_mask"}


# --------------------------------------------------------------
# Helper: wrap forward so unknown kwargs are discarded
# --------------------------------------------------------------
def _wrap_forward(module):
    if hasattr(module, "_rfedit_kwpatched"):
        return
    sig = inspect.signature(module.forward)
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        # already has **kwargs
        return
    if _EXTRA_KEYS.intersection(sig.parameters):
        # forward already accepts the keys we care about
        return

    orig_fwd = module.forward

    @wraps(orig_fwd)
    def new_forward(*args, **kwargs):
        for k in list(kwargs.keys()):
            if k in _EXTRA_KEYS:
                kwargs.pop(k)
        return orig_fwd(*args, **kwargs)

    module.forward = new_forward
    module._rfedit_kwpatched = True


class ConfigureHiDreamBlocksNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "fluxtapoz"

    # ----------------------------------------------------------
    def _ensure_block_lists(self, unet):
        if hasattr(unet, "double_blocks") and hasattr(unet, "single_blocks"):
            return

        dbl, sgl = [], []

        for name in dir(unet):
            if "double" in name and "blocks" in name:
                attr = getattr(unet, name)
                if isinstance(attr, (list, tuple)):
                    dbl.extend(attr)
            if "single" in name and "blocks" in name:
                attr = getattr(unet, name)
                if isinstance(attr, (list, tuple)):
                    sgl.extend(attr)

        if not dbl and not sgl:
            # Fallback: traverse entire tree
            for m in unet.modules():
                if isinstance(m, BasicTransformerBlock):
                    dbl.append(m)

        unet.double_blocks = dbl
        unet.single_blocks = sgl

    # ----------------------------------------------------------
    def apply(self, model):
        unet = model.model.diffusion_model

        # 1. Prepare lists so inject_blocks() works
        self._ensure_block_lists(unet)

        # 2. Add RF‑Edit / PAG / SEG hooks to inner transformer blocks
        inject_blocks(unet)

        # 3. Wrap any module lacking the extra kwargs
        for m in unet.modules():
            _wrap_forward(m)

        return (model,)

