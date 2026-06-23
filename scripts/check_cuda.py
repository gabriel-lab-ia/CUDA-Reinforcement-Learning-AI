from __future__ import annotations

import platform
import sys
import time

import torch


def main() -> None:
    print("=" * 64)
    print("CUDA Reinforcement Learning AI — Environment Check")
    print("=" * 64)

    print(f"Python:               {sys.version.split()[0]}")
    print(f"Executable:           {sys.executable}")
    print(f"Platform:             {platform.platform()}")
    print(f"PyTorch:              {torch.__version__}")
    print(f"CUDA build:           {torch.version.cuda}")
    print(f"CUDA available:       {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA não está disponível para o PyTorch. "
            "Verifique driver, instalação e backend do PyTorch."
        )

    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(device)

    print(f"GPU:                   {torch.cuda.get_device_name(device)}")
    print(f"Compute capability:    {properties.major}.{properties.minor}")
    print(f"VRAM total:            {properties.total_memory / 1024**3:.2f} GiB")
    print(f"Multiprocessadores:    {properties.multi_processor_count}")

    size = 4096

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    matrix_a = torch.randn(size, size, device=device)
    matrix_b = torch.randn(size, size, device=device)

    torch.cuda.synchronize()
    start = time.perf_counter()

    result = matrix_a @ matrix_b

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    operations = 2 * size**3
    tflops = operations / elapsed / 1e12

    allocated = torch.cuda.memory_allocated(device) / 1024**2
    reserved = torch.cuda.memory_reserved(device) / 1024**2

    print("-" * 64)
    print(f"Teste:                 multiplicação {size} x {size}")
    print(f"Tempo CUDA:            {elapsed:.6f} s")
    print(f"Throughput estimado:   {tflops:.3f} TFLOP/s")
    print(f"Memória alocada:       {allocated:.2f} MiB")
    print(f"Memória reservada:     {reserved:.2f} MiB")
    print(f"Resultado médio:       {result.mean().item():.6f}")
    print("=" * 64)


if __name__ == "__main__":
    main()
