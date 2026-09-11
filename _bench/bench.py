import time, sys, traceback
import torch
import torch.nn as nn
import torch.nn.functional as F

dev = torch.device("cuda")
name = torch.cuda.get_device_name(0)
print(f"=== Device: {name} ===")
print(f"torch {torch.__version__}, hip {torch.version.hip}")
print()

def sync():
    torch.cuda.synchronize()

def bench(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    t1 = time.perf_counter()
    return (t1 - t0) / iters

results = {}

# ---------- 1. Memory bandwidth ----------
print("--- Memory bandwidth (H2D / D2H / D2D copy) ---")
try:
    n = 512 * 1024 * 1024 // 4  # 512MB of float32
    a_cpu = torch.randn(n, dtype=torch.float32, pin_memory=True)
    a_gpu = torch.empty(n, dtype=torch.float32, device=dev)
    b_gpu = torch.empty(n, dtype=torch.float32, device=dev)

    t = bench(lambda: a_gpu.copy_(a_cpu, non_blocking=True))
    gbps = (n * 4) / t / 1e9
    print(f"H2D: {gbps:.1f} GB/s")

    t = bench(lambda: a_cpu.copy_(a_gpu, non_blocking=True))
    gbps = (n * 4) / t / 1e9
    print(f"D2H: {gbps:.1f} GB/s")

    t = bench(lambda: b_gpu.copy_(a_gpu))
    gbps = (n * 4 * 2) / t / 1e9
    print(f"D2D: {gbps:.1f} GB/s (read+write)")
except Exception as e:
    print("MEMORY BANDWIDTH TEST FAILED:", e)
    traceback.print_exc()
print()

# ---------- 2. Matmul FLOPs at various precisions ----------
print("--- Matmul (GEMM) throughput ---")
sizes = [4096]
for dtype_name, dtype in [("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16)]:
    for sz in sizes:
        try:
            a = torch.randn(sz, sz, dtype=dtype, device=dev)
            b = torch.randn(sz, sz, dtype=dtype, device=dev)
            t = bench(lambda: a @ b, warmup=10, iters=30)
            flops = 2 * sz**3
            tflops = flops / t / 1e12
            print(f"{dtype_name:5s} {sz}x{sz}: {t*1000:.3f} ms/iter -> {tflops:.2f} TFLOPS")
        except Exception as e:
            print(f"{dtype_name} {sz}x{sz} FAILED:", e)
print()

# ---------- 3. CNN training loop (ResNet18-like on synthetic ImageNet-shaped data) ----------
print("--- CNN training (torchvision resnet18, batch=64, 224x224) ---")
try:
    import torchvision.models as models
    model = models.resnet18(weights=None).to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    crit = nn.CrossEntropyLoss()

    batch = 64
    x = torch.randn(batch, 3, 224, 224, device=dev)
    y = torch.randint(0, 1000, (batch,), device=dev)

    def step():
        opt.zero_grad(set_to_none=True)
        out = model(x)
        loss = crit(out, y)
        loss.backward()
        opt.step()
        return loss

    losses = []
    for _ in range(5):
        losses.append(step().item())
    sync()

    t0 = time.perf_counter()
    n_steps = 30
    for _ in range(n_steps):
        loss = step()
    sync()
    t1 = time.perf_counter()

    imgs_per_sec = (batch * n_steps) / (t1 - t0)
    print(f"fp32: {imgs_per_sec:.1f} images/sec, last loss={loss.item():.4f}")
    if any(l != l for l in losses):  # NaN check
        print("WARNING: NaN detected in loss during fp32 CNN training!")

    # AMP (bf16) training
    scaler_dtype = torch.bfloat16
    def step_amp():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=scaler_dtype):
            out = model(x)
            loss = crit(out, y)
        loss.backward()
        opt.step()
        return loss

    for _ in range(5):
        l = step_amp()
    sync()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        loss = step_amp()
    sync()
    t1 = time.perf_counter()
    imgs_per_sec = (batch * n_steps) / (t1 - t0)
    print(f"bf16 autocast: {imgs_per_sec:.1f} images/sec, last loss={loss.item():.4f}")
    if loss.item() != loss.item():
        print("WARNING: NaN detected in loss during bf16 CNN training!")

    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"Peak GPU memory allocated: {peak_mem:.2f} GB")
except Exception as e:
    print("CNN TRAINING TEST FAILED:", e)
    traceback.print_exc()
print()
torch.cuda.reset_peak_memory_stats()
print()

# ---------- 4. Transformer block training (attention-heavy workload) ----------
print("--- Transformer block training (d_model=1024, seq=512, batch=16) ---")
try:
    d_model = 1024
    nhead = 16
    seq = 512
    batch = 16

    layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4096, batch_first=True).to(dev)
    opt = torch.optim.AdamW(layer.parameters(), lr=1e-4)
    x = torch.randn(batch, seq, d_model, device=dev)
    target = torch.randn(batch, seq, d_model, device=dev)

    def tstep():
        opt.zero_grad(set_to_none=True)
        out = layer(x)
        loss = F.mse_loss(out, target)
        loss.backward()
        opt.step()
        return loss

    losses = []
    for _ in range(5):
        losses.append(tstep().item())
    sync()

    t0 = time.perf_counter()
    n_steps = 30
    for _ in range(n_steps):
        loss = tstep()
    sync()
    t1 = time.perf_counter()

    steps_per_sec = n_steps / (t1 - t0)
    tokens_per_sec = steps_per_sec * batch * seq
    print(f"fp32: {steps_per_sec:.2f} steps/sec, {tokens_per_sec:.0f} tokens/sec, last loss={loss.item():.4f}")
    if any(l != l for l in losses):
        print("WARNING: NaN in transformer fp32 training!")

    def tstep_amp():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = layer(x)
            loss = F.mse_loss(out, target)
        loss.backward()
        opt.step()
        return loss

    for _ in range(5):
        l = tstep_amp()
    sync()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        loss = tstep_amp()
    sync()
    t1 = time.perf_counter()
    steps_per_sec = n_steps / (t1 - t0)
    tokens_per_sec = steps_per_sec * batch * seq
    print(f"bf16 autocast: {steps_per_sec:.2f} steps/sec, {tokens_per_sec:.0f} tokens/sec, last loss={loss.item():.4f}")
    if loss.item() != loss.item():
        print("WARNING: NaN in transformer bf16 training!")

    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"Peak GPU memory allocated: {peak_mem:.2f} GB")
except Exception as e:
    print("TRANSFORMER TEST FAILED:", e)
    traceback.print_exc()
print()

# ---------- 5. torch.compile smoke test ----------
print("--- torch.compile smoke test ---")
try:
    model2 = models.resnet18(weights=None).to(dev)
    compiled = torch.compile(model2)
    x = torch.randn(8, 3, 224, 224, device=dev)
    out = compiled(x)
    loss = out.sum()
    loss.backward()
    sync()
    print("torch.compile: OK (forward+backward succeeded)")
except Exception as e:
    print("torch.compile FAILED (this is a known problem area on ROCm/Windows):", repr(e))
print()

print("=== BENCHMARK COMPLETE ===")
