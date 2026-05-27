import torch
from diffusers import DDPMPipeline, UNet2DModel, DDPMScheduler

model = UNet2DModel.from_pretrained("lesson1/my_ddpm")
scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    beta_start=0.0001,
    beta_end=0.02,
    beta_schedule="squaredcos_cap_v2"
    )

pipeline = DDPMPipeline(
    unet=model,
    scheduler=scheduler
)
device = "cuda" if torch.cuda.is_available() else "cpu"

pipeline.to(device)

image = pipeline().images[0]
image.show()