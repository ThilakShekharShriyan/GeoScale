import os
import csv
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path

from dataset.loader import get_dataloader, ShardedGeoDataset
from training.model import VisioPoseModel
from utils.config import load_config

def setup(rank, world_size, backend="gloo"):
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '12355')
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def contrastive_loss(v_emb, p_emb, temperature=0.07):
    batch_size = v_emb.size(0)
    logits = torch.matmul(v_emb, p_emb.T) / temperature
    labels = torch.arange(batch_size, device=v_emb.device)

    loss_v = nn.functional.cross_entropy(logits, labels)
    loss_p = nn.functional.cross_entropy(logits.T, labels)

    return (loss_v + loss_p) / 2

def main():
    parser = argparse.ArgumentParser("GeoScale Training")
    parser.add_argument("--data-dir", default="data/dataset", type=str)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--num-workers", default=2, type=int)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--checkpoint-dir", default="checkpoints", type=str)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--use-gpu", action="store_true", help="Enable GPU training if available")
    parser.add_argument("--checkpoint-vision", action="store_true", help="Enable activation checkpointing for ViT")
    args = parser.parse_args()

    config = load_config()

    use_gpu = args.use_gpu or getattr(config, 'use_gpu', False) == True or getattr(config, 'use_gpu', 'auto') == 'auto'
    if use_gpu and not torch.cuda.is_available():
        print("Warning: GPU requested but not available. Falling back to CPU in DDP")
        use_gpu = False

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    backend = "nccl" if (use_gpu and torch.cuda.is_available()) else "gloo"

    if world_size > 1 and not dist.is_initialized():
        setup(rank, world_size, backend)

    device = torch.device(f'cuda:{local_rank}' if use_gpu else 'cpu')
    if use_gpu:
        torch.cuda.set_device(device)

    if rank == 0:
        print(f"Process initialized. World size: {world_size}, Device: {device}, Backend: {backend}")

    model = VisioPoseModel(use_checkpointing=args.checkpoint_vision).to(device)

    if world_size > 1:
        if use_gpu:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import MixedPrecision
            mp_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16
            )
            model = FSDP(model, device_id=local_rank, mixed_precision=mp_policy)
        else:
            model = DDP(model)

    dataset = ShardedGeoDataset(args.data_dir)
    sampler = DistributedSampler(dataset) if world_size > 1 else None

    dataloader = get_dataloader(
        args.data_dir,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        use_gpu=use_gpu
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    checkpoint_dir = Path(args.checkpoint_dir)
    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    if args.resume:
        ckpt_path = checkpoint_dir / "latest.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            if world_size > 1 and not list(ckpt['model_state'].keys())[0].startswith('module.'):
                model.module.load_state_dict(ckpt['model_state'])
            elif world_size == 1 and list(ckpt['model_state'].keys())[0].startswith('module.'):
                new_state = {k.replace('module.', ''): v for k, v in ckpt['model_state'].items()}
                model.load_state_dict(new_state)
            else:
                model.load_state_dict(ckpt['model_state'])

            optimizer.load_state_dict(ckpt['optimizer_state'])
            start_epoch = ckpt['epoch'] + 1
            if rank == 0:
                print(f"Resumed from epoch {start_epoch}")

    profiler = None
    if args.profile and rank == 0:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if use_gpu and torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        profiler = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler'),
            record_shapes=False, profile_memory=False, with_stack=False
        )
        profiler.start()

    for epoch in range(start_epoch, args.epochs):
        if sampler:
            sampler.set_epoch(epoch)

        model.train()
        epoch_start_time = time.time()

        total_loss = 0.0
        num_batches = 0
        total_images = 0

        for batch_idx, batch in enumerate(dataloader):
            batch_start = time.time()

            images = batch['image'].to(device, non_blocking=use_gpu)
            poses = batch['pose'].to(device, non_blocking=use_gpu)

            optimizer.zero_grad()

            v_emb, p_emb = model(images, poses)
            loss = contrastive_loss(v_emb, p_emb)

            loss.backward()
            optimizer.step()

            if profiler:
                profiler.step()

            batch_time = time.time() - batch_start
            total_loss += loss.item()
            num_batches += 1
            images_in_batch = images.size(0) * world_size
            total_images += images_in_batch

            if rank == 0 and batch_idx % max(1, len(dataloader)//5) == 0:
                imgs_sec = images_in_batch / batch_time if batch_time > 0 else 0
                print(f"Epoch: [{epoch}][{batch_idx}/{len(dataloader)}] "
                      f"Loss: {loss.item():.4f} "
                      f"Batch Time: {batch_time:.3f}s "
                      f"Throughput: {imgs_sec:.1f} imgs/s")

        if rank == 0:
            epoch_time = time.time() - epoch_start_time
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            avg_imgs_sec = (total_images) / epoch_time if epoch_time > 0 else 0
            print(f"--- Epoch {epoch} complete in {epoch_time:.2f}s ---")
            print(f"Avg Loss: {avg_loss:.4f}, Overall Throughput: {avg_imgs_sec:.1f} imgs/s")

            metrics_file = Path("training_metrics.csv")
            file_exists = metrics_file.exists()
            with open(metrics_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['epoch', 'world_size', 'batch_size', 'avg_loss', 'epoch_time_sec', 'throughput_imgs_per_sec'])
                writer.writerow([epoch, world_size, args.batch_size, f"{avg_loss:.4f}", f"{epoch_time:.2f}", f"{avg_imgs_sec:.1f}"])

            state_dict = model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state': state_dict,
                'optimizer_state': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_dir / f"epoch_{epoch}.pt")

            torch.save({
                'epoch': epoch,
                'model_state': state_dict,
                'optimizer_state': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_dir / "latest.pt")

    if profiler:
        profiler.stop()

    cleanup()

if __name__ == "__main__":
    main()
