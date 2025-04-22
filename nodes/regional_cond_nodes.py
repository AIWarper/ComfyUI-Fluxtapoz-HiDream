import torch

import comfy.sd
import comfy.model_sampling


DEFAULT_REGIONAL_ATTN = {
    'double': [i for i in range(1, 19, 2)],
    'single': [i for i in range(1, 38, 2)]
}


class RegionalMask(torch.nn.Module):
    def __init__(self, mask: torch.Tensor, start_percent: float, end_percent: float) -> None:
        super().__init__()
        self.register_buffer('mask', mask)
        self.start_percent = start_percent
        self.end_percent = end_percent

    def __call__(self, q, transformer_options, *args, **kwargs):
        if self.start_percent <= 1 - transformer_options['sigmas'][0] < self.end_percent:
            return self.mask
        
        return None
    

class RegionalConditioning(torch.nn.Module):
    def __init__(self, region_cond: torch.Tensor, start_percent: float, end_percent: float) -> None:
        super().__init__()
        self.register_buffer('region_cond', region_cond)
        self.start_percent = start_percent
        self.end_percent = end_percent

    def __call__(self, transformer_options, *args,  **kwargs):
        if self.start_percent <= 1 - transformer_options['sigmas'][0] < self.end_percent:
            return self.region_cond
        return None


class CreateRegionalCondNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "cond": ("CONDITIONING",),
            "mask": ("MASK",),
        }, "optional": {
            "prev_regions": ("REGION_COND",),
        }}

    RETURN_TYPES = ("REGION_COND",)
    FUNCTION = "create"

    CATEGORY = "fluxtapoz"

    def create(self, cond, mask, prev_regions=[]):
        prev_regions = [*prev_regions]
        prev_regions.append({
            'mask': mask,
            'cond': cond[0][0]
        })

        return (prev_regions,)


class ApplyRegionalCondsNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":      ("MODEL",),
                "region_conds": ("REGION_COND",),
                "latent":     ("LATENT",),
                "start_percent": ("FLOAT", {"default": 0,   "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01}),
                "end_percent":   ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01}),
            },
            "optional": {
                "attn_override": ("ATTN_OVERRIDE",),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION     = "patch"
    CATEGORY     = "fluxtapoz"

    def patch(
        self,
        model,
        region_conds,
        latent,
        start_percent,
        end_percent,
        attn_override=DEFAULT_REGIONAL_ATTN,
    ):
        # --------------------------------------------------------------------
        # 1. Clone the model so we can patch it without side‑effects
        # --------------------------------------------------------------------
        model = model.clone()

        # --------------------------------------------------------------------
        # 2. Derive dimensions
        # --------------------------------------------------------------------
        latent  = latent["samples"]                     # BCHW
        b, c, h, w = latent.shape
        h //= 2                                         # Fluxtapoz uses /2 latents
        w //= 2
        img_len = h * w

        # --------------------------------------------------------------------
        # 3. HiDream vs Flux: work out how many text tokens exist
        #    HiDream exposes clip_token_count (2 458), Flux falls back to 256
        # --------------------------------------------------------------------
        clip_tokens = getattr(model, "clip_token_count", 256)  # ★ NEW ★

        # --------------------------------------------------------------------
        # 4. Concat all regional text embeddings and build the attention mask
        # --------------------------------------------------------------------
        regional_conditioning = torch.cat(
            [rc["cond"] for rc in region_conds], dim=1
        )                                              # (1, R, 4096)

        text_len = clip_tokens + regional_conditioning.shape[1]

        regional_mask = torch.zeros(
            (text_len + img_len, text_len + img_len), dtype=torch.bool
        )

        # helpers for image‑image masking
        self_attend_masks = torch.zeros((img_len, img_len), dtype=torch.bool)
        union_masks       = torch.zeros((img_len, img_len), dtype=torch.bool)

        # --------------------------------------------------------------------
        # 5. Prepend a “global” region that covers the whole image
        #    Its cond tensor must match the REAL clip_token length
        # --------------------------------------------------------------------
        region_conds = [
            {
                "mask": torch.ones((1, h, w), dtype=torch.float16),
                "cond": torch.ones((1, clip_tokens, 4096), dtype=torch.float16),  # ★ NEW ★
            },
            *region_conds,
        ]

        # --------------------------------------------------------------------
        # 6. Fill the big attention‑mask matrix
        # --------------------------------------------------------------------
        current_seq_len = 0
        for rc in region_conds:
            cond_tokens  = rc["cond"].shape[1]
            next_seq_len = current_seq_len + cond_tokens

            region_mask  = 1 - rc["mask"][0]                               # (H,W)
            region_mask  = torch.nn.functional.interpolate(
                region_mask[None, None, :, :], (h, w), mode="nearest-exact"
            ).flatten().unsqueeze(1).repeat(1, cond_tokens)

            # txt ↔ txt
            regional_mask[current_seq_len:next_seq_len, current_seq_len:next_seq_len] = True
            # txt → img
            regional_mask[current_seq_len:next_seq_len, text_len:] = region_mask.T
            # img → txt
            regional_mask[text_len:, current_seq_len:next_seq_len] = region_mask

            # img ↔ img (self/union helpers)
            mask_full            = region_mask[:, :1].repeat(1, img_len)
            mask_full_T          = mask_full.T
            self_attend_masks    |= mask_full & mask_full_T
            union_masks          |= mask_full | mask_full_T

            current_seq_len = next_seq_len

        # everything else attends to its background
        background_masks = ~union_masks
        regional_mask[text_len:, text_len:] = background_masks | self_attend_masks

        # --------------------------------------------------------------------
        # 7. Wrap masks/conds in Fluxtapoz helper classes & patch model
        # --------------------------------------------------------------------
        regional_mask         = RegionalMask(regional_mask, start_percent, end_percent)
        regional_conditioning = RegionalConditioning(regional_conditioning, start_percent, end_percent)

        model.set_model_patch(regional_conditioning, "regional_conditioning")

        # override attention blocks
        for idx in attn_override["double"]:
            model.set_model_patch_replace(regional_mask, "double", "mask_fn", int(idx))
        for idx in attn_override["single"]:
            model.set_model_patch_replace(regional_mask, "single", "mask_fn", int(idx))

        return (model,)

