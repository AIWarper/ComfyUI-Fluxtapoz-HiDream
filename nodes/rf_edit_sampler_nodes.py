import torch
from tqdm import trange
from comfy.samplers import KSAMPLER

from ..utils.attn_bank import AttentionBank
from ..utils.const import DEFAULT_DOUBLE_LAYERS, DEFAULT_SINGLE_LAYERS

# --------------------------------------------------------------
def _num_transformer_blocks(model):
    """
    Walk through any number of .inner_model wrappers until we reach the
    object that exposes .modules(), then count sub‑modules that carry .idx.
    """
    while not hasattr(model, "modules") and hasattr(model, "inner_model"):
        model = model.inner_model
    if not hasattr(model, "modules"):
        return 0
    return sum(1 for m in model.modules() if hasattr(m, "idx"))

# ------------------------------------------------------------------
# Helper – convert UI text to the dict format the UNet expects
# ------------------------------------------------------------------
def _parse_layer_field(field, fallback):
    """
    • "" or None  →  dict()  (means “all layers” – we fill that later)
    • "0,17,35"   →  {"0":1,"17":1,"35":1}
    • {"0":1,...} →  unchanged
    """
    if field in ("", None):
        return {}                       # sentinel for ALL layers
    if isinstance(field, dict):
        return {str(k): int(v) for k, v in field.items()}
    if isinstance(field, str):
        nums = [n.strip() for n in field.split(",") if n.strip()]
        return {n: 1 for n in nums}
    return fallback


# ------------------------------------------------------------------
# Forward / reverse sample functions
# ------------------------------------------------------------------
def get_sample_forward(attn_bank, save_steps, single_layers, double_layers, order="second"):

    @torch.no_grad()
    def sample_forward(model, x, sigmas, extra_args=None, callback=None, disable=None):
        extra_args = {} if extra_args is None else extra_args
        N = len(sigmas) - 1
        s_in = x.new_ones([x.shape[0]])

        # Fill “all layers” dict the first time we see the model
        if not double_layers:
            total = getattr(model.inner_model, "total_transformer_blocks", None)
            if total is None:           # fallback: count manually
                total = _num_transformer_blocks(model)
            double_layers.update({str(i): 1 for i in range(total)})

        attn_bank.clear()
        attn_bank["save_steps"] = save_steps

        # Debug print once
        print(f"[RF‑Edit] forward save_steps={save_steps} dbl_keys={list(double_layers)[:5]} ...")

        prev_pred = None
        for i in trange(N, disable=disable):
            sigma, sigma_next = sigmas[i : i + 2]

            if N - i - 1 < save_steps:
                attn_bank[N - i - 1] = {"first": {}, "mid": {}}

            xf_opts = {
                "step": N - i - 1,
                "process": "forward" if N - i - 1 < save_steps else None,
                "pred": "first",
                "bank": attn_bank,
                "single_layers": single_layers,
                "double_layers": double_layers,
            }

            model_opts = extra_args.setdefault("model_options", {})
            model_opts.setdefault("transformer_options", {})["rfedit"] = xf_opts

            pred = prev_pred if (order == "fireflow" and prev_pred is not None) else model(x, s_in * sigma, **extra_args)
            xf_opts["pred"] = "mid"
            img_mid = x + (sigma_next - sigma) / 2 * pred
            sigma_mid = sigma + (sigma_next - sigma) / 2
            pred_mid = model(img_mid, s_in * sigma_mid, **extra_args)

            if order == "fireflow":
                prev_pred = pred_mid
                x = x + (sigma_next - sigma) * pred_mid
            else:
                first_order = (pred_mid - pred) / ((sigma_next - sigma) / 2)
                x = x + (sigma_next - sigma) * pred + 0.5 * (sigma_next - sigma) ** 2 * first_order

            if callback is not None:
                callback({"x": x, "denoised": x, "i": i, "sigma": sigma})

        return x

    return sample_forward


def get_sample_reverse(attn_bank, inject_steps, single_layers, double_layers, order="second"):

    @torch.no_grad()
    def sample_reverse(model, x, sigmas, extra_args=None, callback=None, disable=None):
        extra_args = {} if extra_args is None else extra_args
        N = len(sigmas) - 1
        s_in = x.new_ones([x.shape[0]])

        # Fill dict if still empty (should match forward)
        if not double_layers:
            total = getattr(model.inner_model, "total_transformer_blocks", None)
            if total is None:
                total = _num_transformer_blocks(model)
            double_layers.update({str(i): 1 for i in range(total)})

        print(f"[RF‑Edit] reverse inject_steps={inject_steps} dbl_keys={list(double_layers)[:5]} ...")

        prev_pred = None
        for i in trange(N, disable=disable):
            sigma, sigma_prev = sigmas[i : i + 2]

            xf_opts = {
                "step": i,
                "process": "reverse" if i < inject_steps else None,
                "pred": "first",
                "bank": attn_bank,
                "single_layers": single_layers,
                "double_layers": double_layers,
            }

            model_opts = extra_args.setdefault("model_options", {})
            model_opts.setdefault("transformer_options", {})["rfedit"] = xf_opts

            pred = prev_pred if (order == "fireflow" and prev_pred is not None) else model(x, s_in * sigma, **extra_args)
            xf_opts["pred"] = "mid"
            img_mid = x + (sigma_prev - sigma) / 2 * pred
            sigma_mid = sigma + (sigma_prev - sigma) / 2
            pred_mid = model(img_mid, s_in * sigma_mid, **extra_args)

            if order == "fireflow":
                prev_pred = pred_mid
                x = x + (sigma_prev - sigma) * pred_mid
            else:
                first_order = (pred_mid - pred) / ((sigma_prev - sigma) / 2)
                x = x + (sigma_prev - sigma) * pred + 0.5 * (sigma_prev - sigma) ** 2 * first_order

            if callback is not None:
                callback({"x": x, "denoised": x, "i": i, "sigma": sigma})

        return x

    return sample_reverse


# ------------------------------------------------------------------
# Nodes
# ------------------------------------------------------------------
class FlowEditForwardSamplerNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"save_steps": ("INT", {"default": 5, "min": 0})},
            "optional": {
                "single_layers": ("STRING",),
                "double_layers": ("STRING",),
                "order": (["second", "fireflow"],),
            },
        }

    RETURN_TYPES = ("SAMPLER", "ATTN_INJ")
    FUNCTION = "build"
    CATEGORY = "fluxtapoz"

    def build(self, save_steps, single_layers="", double_layers="", order="second"):
        single_layers = _parse_layer_field(single_layers, DEFAULT_SINGLE_LAYERS.copy())
        double_layers = _parse_layer_field(double_layers, DEFAULT_DOUBLE_LAYERS.copy())
        attn_bank = AttentionBank()
        sampler = KSAMPLER(get_sample_forward(attn_bank, save_steps, single_layers, double_layers, order))
        return (sampler, attn_bank)


class FlowEditReverseSamplerNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "attn_inj": ("ATTN_INJ",),
                "inject_steps": ("INT", {"default": 5, "min": 0}),
            },
            "optional": {
                "single_layers": ("STRING",),
                "double_layers": ("STRING",),
                "order": (["second", "fireflow"],),
            },
        }

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "build"
    CATEGORY = "fluxtapoz"

    def build(self, attn_inj, inject_steps, single_layers="", double_layers="", order="second"):
        single_layers = _parse_layer_field(single_layers, DEFAULT_SINGLE_LAYERS.copy())
        double_layers = _parse_layer_field(double_layers, DEFAULT_DOUBLE_LAYERS.copy())
        sampler = KSAMPLER(get_sample_reverse(attn_inj, inject_steps, single_layers, double_layers, order))
        return (sampler,)

# ------------------------------------------------------------------
#  PrepareAttnBankNode  (same behaviour as original)
# ------------------------------------------------------------------
class PrepareAttnBankNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "attn_inj": ("ATTN_INJ",),
            }
        }

    RETURN_TYPES = ("LATENT", "ATTN_INJ")
    FUNCTION     = "prepare"
    CATEGORY     = "fluxtapoz"

    def prepare(self, latent, attn_inj):
        # Forces ComfyUI execution order
        return (latent, attn_inj)

