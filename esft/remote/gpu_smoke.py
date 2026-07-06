"""借り物 GPU の受け入れ検査: SM120 で実際に計算が通るか(driver 表示だけでは不十分)。"""
import torch

assert torch.cuda.is_available(), "CUDA not available — --gpus all / driver passthrough を確認"
n = torch.cuda.device_count()
print(f"GPUs: {n}")
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f"  [{i}] {p.name}  {p.total_memory/2**30:.0f}GB  sm_{p.major}{p.minor}")

for i in range(n):
    with torch.cuda.device(i):
        a = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
        c = a @ b
        torch.cuda.synchronize()
        assert torch.isfinite(c).all(), f"GPU{i}: matmul produced non-finite values"
        print(f"  [{i}] bf16 matmul OK (mean={c.float().mean():+.4f})")

if n >= 2:
    x = torch.randn(1024, 1024, device="cuda:0")
    y = x.to("cuda:1")
    assert torch.allclose(x.cpu(), y.cpu()), "P2P/copy mismatch"
    print("  GPU0->GPU1 transfer OK")
print("SMOKE PASS")
