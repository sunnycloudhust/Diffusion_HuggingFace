import torch
from diffusers import UNet2DModel, DDPMScheduler
import torch.nn.functional as F
from data import train_dataloader

device = "cuda" if torch.cuda.is_availble() else "cpu"
model = UNet2DModel(
    sample_size=32,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(64,128,256,256),
    down_block_types=(
            "DownBlock2D",       
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
    up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",   
            "UpBlock2D",
            "UpBlock2D")
    )

model = model.to(device)
noise_scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    beta_start=0.0001,
    beta_end=0.02,
    beta_schedule="squaredcos_cap_v2"
    )


optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
epochs = 50 

print("Start training")

for epoch in range(epochs):
    for step, batch in enumerate(train_dataloader):
        clean_images = batch["images"].to(device)
        
        # 2. Tạo nhiễu ngẫu nhiên (noise) có cùng kích thước với batch ảnh
        noise = torch.randn_like(clean_images)
        
        # 3. Random một mốc thời gian t (từ 0 đến 999) cho mỗi bức ảnh trong batch
        bs = clean_images.shape[0] # Lấy số lượng ảnh hiện tại (batch size)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=device).long()
        
        # 4. Cộng nhiễu vào ảnh gốc theo tỷ lệ tại mốc thời gian t (Forward Process)
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
        
        # 5. Yêu cầu U-Net đoán xem nhiễu là gì (Nó dựa vào ảnh đã bị nhiễu và mốc thời gian t để dự đoán)
        noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
        
        # 6. Tính sai số (Loss) giữa nhiễu AI đoán và nhiễu thực tế mình đã bơm vào
        loss = F.mse_loss(noise_pred, noise)

        loss.backward()      
        optimizer.step()     
        optimizer.zero_grad()

        if step % 10 == 0:
            print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")
