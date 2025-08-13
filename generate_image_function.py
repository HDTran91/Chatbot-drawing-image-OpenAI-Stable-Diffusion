from diffusers import DiffusionPipeline
import torch
import time

# Hôm nay mình đổi qua stablediffusionapi/anything-v5 để tạo ảnh anime cho nó bớt nhàm chán nha!
pipeline = DiffusionPipeline.from_pretrained("stablediffusionapi/anything-v5",
                                             use_safetensors=True, safety_checker=None, requires_safety_checker=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
# MPS chỉ có trên macOS dòng M1 trở đi nha
device = 'mps' if torch.backends.mps.is_available() else device
pipeline.to(device)

def generate_image(prompt: str) -> str:
    image = pipeline(
        prompt=prompt,
        # Hardcode negative prompt để ảnh đẹp hơn
        negative_prompt="ugly, deformed, disfigured, poor details, bad anatomy, low quality, worst quality",
        num_inference_steps=30
    ).images[0]

    # Tạo tên file để hiện thị nè
    file_name = f"image_{int(time.time())}.png"
    image.save(file_name)
    return file_name