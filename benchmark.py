import time
import sys
import platform
import psutil
import httpx
import json
import subprocess
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import settings

def get_system_info():
    """Gather system hardware information"""
    info = {
        "OS": f"{platform.system()} {platform.release()}",
        "Python": platform.python_version(),
        "CPU Cores": psutil.cpu_count(logical=True),
        "RAM Total": f"{psutil.virtual_memory().total / (1024**3):.1f} GB",
        "RAM Available": f"{psutil.virtual_memory().available / (1024**3):.1f} GB",
    }
    
    # Try to get GPU info via nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            gpus = result.stdout.strip().split('\n')
            info["GPUs"] = gpus
    except:
        info["GPUs"] = ["NVIDIA-SMI not found or no NVIDIA GPU"]
        
    return info

def benchmark_model(name: str, prompt: str = "Why is the sky blue? Answer in one sentence."):
    """Run a benchmark on a specific model"""
    print(f"\nTesting model: {name}...")
    
    url = "http://localhost:11434/api/generate"
    data = {
        "model": name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": 100  # Limit generation for consistent benchmarking
        }
    }
    
    start_time = time.time()
    try:
        response = httpx.post(url, json=data, timeout=60.0)
        response.raise_for_status()
        result = response.json()
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Ollama returns stats in the response
        eval_count = result.get("eval_count", 0)
        eval_duration = result.get("eval_duration", 0) # in nanoseconds
        
        if eval_duration > 0:
            tps = eval_count / (eval_duration / 1e9)
        else:
            tps = 0
            
        return {
            "success": True,
            "duration": f"{duration:.2f}s",
            "tokens_generated": eval_count,
            "tokens_per_second": f"{tps:.2f}",
            "response_preview": result.get("response", "")[:50] + "..."
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def main():
    print("="*60)
    print("LOCAL LLM BENCHMARK TOOL")
    print("="*60)
    
    # 1. System Info
    print("\n[System Information]")
    try:
        sys_info = get_system_info()
        for k, v in sys_info.items():
            if isinstance(v, list):
                print(f"{k}:")
                for item in v:
                    print(f"  - {item}")
            else:
                print(f"{k}: {v}")
    except Exception as e:
        print(f"Error getting system info: {e}")

    # 2. Model Benchmarks
    print("\n[Model Benchmarks]")
    print("Running inference tests (this may take a moment)...")
    
    # Get unique models to test (avoid duplicates if tiers share models)
    models_to_test = {}
    for tier, config in settings.models.items():
        models_to_test[config.name] = tier.value

    for model_name, tier_name in models_to_test.items():
        result = benchmark_model(model_name)
        
        if result["success"]:
            print(f"  Model: {model_name} ({tier_name})")
            print(f"  ├─ Speed: {result['tokens_per_second']} tokens/sec")
            print(f"  ├─ Total Time: {result['duration']}")
            print(f"  └─ Output: {result['tokens_generated']} tokens")
        else:
            print(f"  Model: {model_name} ({tier_name})")
            print(f"  └─ FAILED: {result['error']}")

    print("\n" + "="*60)
    print("Benchmark Complete")
    print("="*60)

if __name__ == "__main__":
    main()
