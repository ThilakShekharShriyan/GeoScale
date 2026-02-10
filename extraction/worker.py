import os
import ray
import torch
import numpy as np
import argparse
import time
from pathlib import Path
from torch.utils.data import DataLoader

from dataset.loader import ShardedGeoDataset
from training.model import VisioPoseModel

@ray.remote
class EmbeddingWorker:
    def __init__(self, checkpoint_path):
        self.device = torch.device('cpu')
        self.model = VisioPoseModel().to(self.device)
        self.model.eval()

        if checkpoint_path and Path(checkpoint_path).exists():
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            state_dict = ckpt['model_state']
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k.replace('module.', ''): v for k,v in state_dict.items()}
            self.model.load_state_dict(state_dict)

    def process_shard(self, shard_path, batch_size=32):
        dataset = ShardedGeoDataset(shard_path)
        if len(dataset) == 0:
            return 0, 0.0

        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)

        all_embeddings = []
        all_ids = []

        start = time.time()
        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(self.device)
                poses = batch['pose'].to(self.device)

                v_emb, _ = self.model(images, poses)
                all_embeddings.append(v_emb.cpu().numpy())
                all_ids.extend(batch['id'])

        if not all_embeddings:
            return 0, 0.0

        embeddings_np = np.concatenate(all_embeddings, axis=0)

        out_path = Path(shard_path) / "embeddings.npy"
        np.save(out_path, embeddings_np)

        with open(Path(shard_path) / "embedding_ids.txt", "w") as f:
            for id_str in all_ids:
                f.write(f"{id_str}\n")

        duration = time.time() - start
        return len(all_ids), duration

def run_extraction(data_dir, checkpoint_path, num_workers=2):
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    root_path = Path(data_dir)
    shard_dirs = [d for d in root_path.iterdir() if d.is_dir()]

    print(f"Found {len(shard_dirs)} shards. Starting extraction...")

    workers = [EmbeddingWorker.remote(checkpoint_path) for _ in range(num_workers)]

    futures = []
    for i, shard_dir in enumerate(shard_dirs):
        worker = workers[i % num_workers]
        futures.append(worker.process_shard.remote(str(shard_dir)))

    start_time = time.time()
    results = ray.get(futures)
    total_time = time.time() - start_time

    total_extracted = sum(res[0] for res in results)
    throughput = total_extracted / total_time if total_time > 0 else 0

    print(f"Extraction complete!")
    print(f"Total shards: {len(shard_dirs)}")
    print(f"Total embeddings: {total_extracted}")
    print(f"Total time: {total_time:.2f} s")
    print(f"Throughput: {throughput:.2f} embeds/sec")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("GeoScale Embedding Extraction")
    parser.add_argument("--data-dir", default="data/dataset", type=str)
    parser.add_argument("--checkpoint", default="checkpoints/latest.pt", type=str)
    parser.add_argument("--num-workers", default=2, type=int)
    args = parser.parse_args()

    run_extraction(args.data_dir, args.checkpoint, args.num_workers)
