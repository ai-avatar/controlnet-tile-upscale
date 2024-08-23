from diffusers import FluxPipeline
from diffusers.models.controlnet_flux import FluxControlNetModel
import torch
from RealESRGAN import RealESRGAN


for scale in [2, 4]:
    model = RealESRGAN("cuda", scale=scale)
    model.load_weights(f"weights/RealESRGAN_x{scale}.pth", download=True)

SD15_WEIGHTS = "weights"
CONTROLNET_CACHE = "controlnet-cache"

controlnet = FluxControlNetModel.from_pretrained(
    'InstantX/FLUX.1-dev-Controlnet-Union-alpha', torch_dtype=torch.bfloat16, cache_dir=CONTROLNET_CACHE
)
controlnet.save_pretrained(CONTROLNET_CACHE)

pipe = FluxPipeline.from_pretrained(
    'black-forest-labs/FLUX.1-dev', torch_dtype=torch.bfloat16, cache_dir=SD15_WEIGHTS
)
pipe.save_pretrained(SD15_WEIGHTS)

