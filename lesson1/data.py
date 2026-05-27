import torch
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader

dataset = load_dataset("cifar10", split="train")

preprocess = transforms.Compose([
    transforms.Resize((32, 32)),  
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),       
    transforms.Normalize([0.5], [0.5])
])

def transform_images(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["img"]]
    return {"images": images}

dataset.set_transform(transform_images)

batch_size = 64 
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
