import os
import subprocess
import time
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def run_training_experiment(nprocs, data_dir, epochs, batch_size):
    cmd = [
        "torchrun",
        f"--nproc_per_node={nprocs}",
        "training/train.py",
        "--data-dir", data_dir,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size)
    ]
    
    start_time = time.time()
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    wall_time = time.time() - start_time
    
    if result.returncode != 0:
        print(f"Error running with {nprocs} processes:\n{result.stdout}")
        return None
        
    metrics_file = Path("training_metrics.csv")
    if not metrics_file.exists():
        return None
        
    df = pd.read_csv(metrics_file)
    last_run = df[df['world_size'] == nprocs].iloc[-1]
    throughput = last_run['throughput_imgs_per_sec']
    
    return {
        "nprocs": nprocs,
        "wall_time": wall_time,
        "throughput": float(throughput)
    }

def generate_plots(results):
    df = pd.DataFrame(results)
    
    base_time = df.loc[df['nprocs'] == 1, 'wall_time'].values[0]
    base_throughput = df.loc[df['nprocs'] == 1, 'throughput'].values[0]
    
    df['speedup_time'] = base_time / df['wall_time']
    df['speedup_throughput'] = df['throughput'] / base_throughput
    df['efficiency'] = df['speedup_throughput'] / df['nprocs']
    
    print("\nScaling Results:")
    print(df.to_string(index=False))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Throughput Plot
    ax1.plot(df['nprocs'], df['throughput'], marker='o', linewidth=2, color='blue')
    ax1.set_title("Training Throughput vs Num Processes")
    ax1.set_xlabel("Number of Processes")
    ax1.set_ylabel("Throughput (images/sec)")
    ax1.set_xticks(df['nprocs'])
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Efficiency Plot
    ax2.plot(df['nprocs'], df['efficiency'] * 100, marker='s', linewidth=2, color='green')
    ax2.set_title("Scaling Efficiency")
    ax2.set_xlabel("Number of Processes")
    ax2.set_ylabel("Efficiency (%)")
    ax2.set_xticks(df['nprocs'])
    ax2.set_ylim(0, 110)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("scaling_results.png")
    print("\nSaved plot to scaling_results.png")

def main():
    parser = argparse.ArgumentParser("GeoScale Scaling Benchmark")
    parser.add_argument("--data-dir", default="test_data", type=str)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--batch-size", default=2, type=int)
    args = parser.parse_args()
    
    if os.path.exists("training_metrics.csv"):
        os.remove("training_metrics.csv")
    
    print("Starting scaling benchmark suite...")
    
    process_counts = [1, 2, 4]
    results = []
    
    for nprocs in process_counts:
        res = run_training_experiment(nprocs, args.data_dir, args.epochs, args.batch_size)
        if res:
            results.append(res)
            
    if results:
        generate_plots(results)
    else:
        print("No valid results obtained.")

if __name__ == "__main__":
    main()
