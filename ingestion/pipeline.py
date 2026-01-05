import os
import ray
import time
import json
import uuid
import geohash2
import argparse
from pathlib import Path

from dataset.synthetic import generate_gps, generate_pose, generate_intrinsics, is_valid_pose, generate_image
from utils.config import load_config

@ray.remote
def process_batch(batch_size, output_root, prefix_length, data_source="synthetic", waymo_bucket=None, segment_path=None):
    """
    Generates synthetic geospatial data or extracts authentic Waymo Frames and writes to sharded directories.
    Returns (num_processed, duration)
    """
    start_time = time.time()
    num_processed = 0
    root_path = Path(output_root)
    
    if data_source == "waymo":
        from dataset.waymo import WaymoGCSAdapter
        adapter = WaymoGCSAdapter(bucket_name=waymo_bucket)
        # Using a dummy or specific segment path if provided
        target_path = segment_path if segment_path else adapter._get_tfrecord_paths()[0]
        iterator = adapter.iter_frames(target_path, limit=batch_size)
        
        for frame_data in iterator:
            prefix = frame_data["geohash"][:prefix_length]
            shard_dir = root_path / prefix
            shard_dir.mkdir(parents=True, exist_ok=True)
            
            record_id = uuid.uuid4().hex
            img_path = shard_dir / f"{record_id}.jpg"
            meta_path = shard_dir / f"{record_id}.json"
            
            with open(img_path, 'wb') as f:
                f.write(frame_data["image_data"])
                
            meta = {
                "id": record_id,
                "lat": frame_data["lat"],
                "lon": frame_data["lon"],
                "geohash": frame_data["geohash"],
                "pose": frame_data["pose"].tolist(),
                "intrinsics": frame_data["intrinsics"].tolist(),
                "timestamp": frame_data["timestamp_micros"]
            }
            
            with open(meta_path, 'w') as f:
                json.dump(meta, f)
                
            num_processed += 1
            
    else:
        # Synthetic fallback
        for _ in range(batch_size):
            lat, lon = generate_gps()
            geohash_full = geohash2.encode(lat, lon, precision=12)
            prefix = geohash_full[:prefix_length]
            
            shard_dir = root_path / prefix
            shard_dir.mkdir(parents=True, exist_ok=True)
            
            img = generate_image(size=(224, 224))
            pose = generate_pose()
            if not is_valid_pose(pose):
                continue
                
            intrinsics = generate_intrinsics()
            
            record_id = uuid.uuid4().hex
            img_path = shard_dir / f"{record_id}.jpg"
            meta_path = shard_dir / f"{record_id}.json"
            
            img.save(img_path)
            
            meta = {
                "id": record_id,
                "lat": float(lat),
                "lon": float(lon),
                "geohash": geohash_full,
                "pose": pose.tolist(),
                "intrinsics": intrinsics.tolist()
            }
            
            with open(meta_path, 'w') as f:
                json.dump(meta, f)
                
            num_processed += 1
        
    duration = time.time() - start_time
    return num_processed, duration

def run_ingestion(total_images, batch_size_per_worker, output_dir, prefix_length, data_source="synthetic", waymo_bucket=None):
    """
    Runs the Ray-based ingestion pipeline.
    """
    print("Initializing Ray...")
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    num_tasks = total_images // batch_size_per_worker
    remainder = total_images % batch_size_per_worker
    
    tasks = [process_batch.remote(batch_size_per_worker, output_dir, prefix_length, data_source, waymo_bucket) for _ in range(num_tasks)]
    if remainder > 0:
        tasks.append(process_batch.remote(remainder, output_dir, prefix_length, data_source, waymo_bucket))
    
    print(f"Dispatched {len(tasks)} tasks to ingest {total_images} records (Source: {data_source}).")
    
    start_time = time.time()
    results = ray.get(tasks)
    total_time = time.time() - start_time
    
    total_processed = sum(res[0] for res in results)
    
    throughput = total_processed / total_time
    print(f"Ingestion complete!")
    print(f"Total processed: {total_processed}")
    print(f"Total time: {total_time:.2f} s")
    print(f"Throughput: {throughput:.2f} records/sec")
    
    return {
        "total_processed": total_processed,
        "total_time": total_time,
        "throughput": throughput
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GeoScale Ingestion Pipeline")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--total", type=int, default=1000, help="Total synthetic images to generate")
    parser.add_argument("--batch", type=int, default=250, help="Batch size per worker")
    parser.add_argument("--output", type=str, default="data/dataset", help="Output directory")
    args = parser.parse_args()
    
    config = load_config(args.config)
    prefix_length = config.get("shard_prefix_length", 3)
    data_source = config.get("data_source", "synthetic")
    waymo_bucket = config.get("waymo_gcs_bucket", None)
    
    run_ingestion(
        total_images=args.total,
        batch_size_per_worker=args.batch,
        output_dir=args.output,
        prefix_length=prefix_length,
        data_source=data_source,
        waymo_bucket=waymo_bucket
    )
