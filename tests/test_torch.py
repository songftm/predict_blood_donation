print("Before import torch", flush=True)
try:
    import torch
    print("Torch imported successfully", flush=True)
    print("Torch version:", torch.__version__, flush=True)
    print("CUDA available:", torch.cuda.is_available(), flush=True)
except ImportError as e:
    print(f"ImportError: {e}", flush=True)
except Exception as e:
    print(f"Error importing torch: {e}", flush=True)
