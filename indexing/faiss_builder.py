import os
import faiss
import numpy as np
import time
import argparse
from pathlib import Path

def build_shard_index(shard_path, embed_dim=128, M=32):
    """
    Builds an HNSW index for the embeddings in the shard.
    """
    embed_file = Path(shard_path) / "embeddings.npy"
    if not embed_file.exists():
        return None
        
    embeddings = np.load(embed_file).astype(np.float32)
    num_vectors = embeddings.shape[0]
    
    index = faiss.IndexHNSWFlat(embed_dim, M)
    index.hnsw.efConstruction = 64
    
    start_time = time.time()
    index.add(embeddings)
    build_time = time.time() - start_time
    
    index_path = Path(shard_path) / "faiss.index"
    faiss.write_index(index, str(index_path))
    
    return {
        "num_vectors": num_vectors,
        "build_time": build_time,
        "index_path": str(index_path)
    }

def evaluate_recall(index, embeddings, k=10, num_queries=100):
    """
    Evaluates recall@k and query latency since it's an approximate index.
    """
    num_queries = min(num_queries, embeddings.shape[0])
    if num_queries == 0:
        return 0.0, 0.0
        
    query_indices = np.random.choice(embeddings.shape[0], num_queries, replace=False)
    queries = embeddings[query_indices]
    
    start_time = time.time()
    D, I = index.search(queries, k)
    query_time = time.time() - start_time
    latency = query_time / num_queries
    
    hits = 0
    for i, q_idx in enumerate(query_indices):
        if q_idx in I[i]:
            hits += 1
            
    recall_at_k = hits / num_queries
    return recall_at_k, latency * 1000

def main():
    parser = argparse.ArgumentParser("GeoScale FAISS Indexing")
    parser.add_argument("--data-dir", default="data/dataset", type=str)
    args = parser.parse_args()
    
    root_path = Path(args.data_dir)
    shard_dirs = [d for d in root_path.iterdir() if d.is_dir()]
    
    print(f"Building indices for {len(shard_dirs)} shards...")
    
    total_vectors = 0
    total_build_time = 0
    
    for shard in shard_dirs:
        res = build_shard_index(shard)
        if res:
            total_vectors += res["num_vectors"]
            total_build_time += res["build_time"]
            
            index = faiss.read_index(res["index_path"])
            embeddings = np.load(shard / "embeddings.npy").astype(np.float32)
            
            index.hnsw.efSearch = 64
            
            recall, latency_ms = evaluate_recall(index, embeddings, k=10)
            print(f"Shard {shard.name}: {res['num_vectors']} vectors. Build: {res['build_time']:.2f}s. "
                  f"Recall@10: {recall:.2f}, Latency: {latency_ms:.2f}ms/query")
            
    print(f"Total Vectors Indexed: {total_vectors}")
    print(f"Total Build Time: {total_build_time:.2f}s")
    
if __name__ == "__main__":
    main()
