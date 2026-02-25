import os
import json
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ShardedGeoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.samples = []
        for img_path in self.root_dir.rglob("*.jpg"):
            meta_path = img_path.with_suffix('.json')
            if meta_path.exists():
                self.samples.append((str(img_path), str(meta_path)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, meta_path = self.samples[idx]

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        with open(meta_path, 'r') as f:
            meta = json.load(f)

        pose = torch.tensor(meta["pose"], dtype=torch.float32)

        return {
            "image": image,
            "pose": pose,
            "geohash": meta["geohash"],
            "id": meta["id"]
        }

def get_dataloader(root_dir, batch_size=32, num_workers=4, shuffle=True, use_gpu=False):
    dataset = ShardedGeoDataset(root_dir)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=use_gpu,
        prefetch_factor=2 if num_workers > 0 else None
    )

    return dataloader

if __name__ == "__main__":
    import time

    test_dir = "test_data"
    if Path(test_dir).exists():
        test_loader = get_dataloader(test_dir, batch_size=2, num_workers=2)
        print(f"Dataset size: {len(test_loader.dataset)}")

        start = time.time()
        for batch in test_loader:
            print(f"Loaded batch of images: {batch['image'].shape}, poses: {batch['pose'].shape}")
            break
        print(f"First batch loaded in {time.time() - start:.3f} s")
    else:
        print("Run ingestion test first to generate test_data")
