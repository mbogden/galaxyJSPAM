"""E1 checks 1-3: GPU visible to PyTorch, a small CNN trains in PyTorch, and in Keras (torch backend).

Prints PASS/FAIL per check and exits non-zero if any check fails.
Uses random data (no downloads), so it runs anywhere.
"""
import os
import socket
import subprocess
import sys
import time

import numpy as np
import torch

N_SAMPLES = 20_000
BATCH = 256
EPOCHS = 2
results = {}


def section(title):
    print(f"\n=== {title}", flush=True)


def check_gpu():
    section("1. GPU")
    print("host:", socket.gethostname())
    print("python:", sys.version.split()[0], "| torch:", torch.__version__, "| torch built for CUDA:", torch.version.cuda)
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
        print("nvidia-smi:", smi.stdout.strip() or smi.stderr.strip())
    except (OSError, subprocess.TimeoutExpired) as e:
        print("nvidia-smi not available:", e)
    if not torch.cuda.is_available():
        print("torch.cuda.is_available() = False")
        return False
    arch_list = torch.cuda.get_arch_list()
    print("compiled GPU archs:", " ".join(arch_list))
    ok = True
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        mem_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
        # GPU code built for sm_XY also runs on sm_XZ with Z >= Y (same major version), e.g. sm_86 on an sm_89 card.
        runs_as = [a for a in arch_list if a.startswith(f"sm_{major}") and int(a[len(f"sm_{major}"):]) <= minor]
        ok &= bool(runs_as)
        print(f"device {i}: {torch.cuda.get_device_name(i)}, sm_{major}{minor}, {mem_gb:.1f} GB, "
              f"runs wheel code for: {runs_as[-1] if runs_as else 'NONE'}")
    a = torch.randn(1024, 1024)
    b = torch.randn(1024, 1024)
    gpu = (a.cuda() @ b.cuda()).cpu()
    match = torch.allclose(gpu, a @ b, rtol=1e-3, atol=1e-2)
    print("GPU matmul matches CPU:", match)
    return ok and match


def fake_data():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((N_SAMPLES, 28, 28, 1), dtype=np.float32)
    y = rng.integers(0, 10, N_SAMPLES)
    return x, y


def check_torch_cnn():
    section("2. PyTorch CNN")
    x, y = fake_data()
    x = torch.from_numpy(x).permute(0, 3, 1, 2).cuda()
    y = torch.from_numpy(y).cuda()
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 32, 3), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(32, 64, 3), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
        torch.nn.Flatten(), torch.nn.Linear(64 * 5 * 5, 128), torch.nn.ReLU(), torch.nn.Linear(128, 10),
    ).cuda()
    opt = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(EPOCHS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(0, N_SAMPLES, BATCH):
            opt.zero_grad()
            loss = loss_fn(model(x[i:i + BATCH]), y[i:i + BATCH])
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
        print(f"epoch {epoch + 1}: {time.perf_counter() - t0:.2f} s, loss {loss.item():.3f}")
    on_gpu = next(model.parameters()).is_cuda
    print("model on GPU:", on_gpu)
    return on_gpu and bool(torch.isfinite(loss))


def check_keras_cnn():
    section("3. Keras CNN (torch backend)")
    import keras
    print("keras:", keras.__version__, "| backend:", keras.config.backend())
    if keras.config.backend() != "torch":
        return False
    x, y = fake_data()
    model = keras.Sequential([
        keras.Input((28, 28, 1)),
        keras.layers.Conv2D(32, 3, activation="relu"), keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, 3, activation="relu"), keras.layers.MaxPooling2D(),
        keras.layers.Flatten(), keras.layers.Dense(128, activation="relu"), keras.layers.Dense(10),
    ])
    model.compile(optimizer="adam", loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    for epoch in range(EPOCHS):
        t0 = time.perf_counter()
        hist = model.fit(x, y, batch_size=BATCH, epochs=1, verbose=0)
        print(f"epoch {epoch + 1}: {time.perf_counter() - t0:.2f} s, loss {hist.history['loss'][-1]:.3f}")
    device = model.weights[0].value.device
    print("weights on:", device)
    return device.type == "cuda" and np.isfinite(hist.history["loss"][-1])


for name, check in [("gpu", check_gpu), ("pytorch", check_torch_cnn), ("keras", check_keras_cnn)]:
    try:
        results[name] = bool(check())
    except Exception as e:  # report and keep going so one failure doesn't hide the others
        print(f"ERROR in {name}: {type(e).__name__}: {e}")
        results[name] = False

section("Summary")
for name, ok in results.items():
    print(f"{'PASS' if ok else 'FAIL'} {name}")
sys.exit(0 if all(results.values()) else 1)
