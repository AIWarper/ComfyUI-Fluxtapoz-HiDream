import torch
class PeekSigmasNode:
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"sigmas": ("SIGMAS",)}}
    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "peek"
    CATEGORY = "debug"
    def peek(self, sigmas: torch.Tensor):
        first, last = sigmas[0].item(), sigmas[-1].item()
        print(f"[PeekSigmas] first={first:.4f}  last={last:.4f}  len={len(sigmas)}")
        return (sigmas,)

