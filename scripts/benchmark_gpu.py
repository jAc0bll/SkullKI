import torch
import time

if not torch.cuda.is_available():
    print("CUDA not available — this benchmark requires a GPU")
    raise SystemExit(1)

dev = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name(0)}")

net = torch.nn.Sequential(
    torch.nn.Linear(244, 512), torch.nn.ReLU(),
    torch.nn.Linear(512, 512), torch.nn.ReLU(),
    torch.nn.Linear(512, 71),
).to(dev)
opt = torch.optim.Adam(net.parameters())

STEPS = 3000
BATCH = 8192

# warm-up
for _ in range(10):
    x = torch.randn(BATCH, 244, device=dev)
    net(x).sum().backward()
    opt.step(); opt.zero_grad()
torch.cuda.synchronize()

t0 = time.time()
for _ in range(STEPS):
    x = torch.randn(BATCH, 244, device=dev)
    loss = net(x).sum()
    opt.zero_grad()
    loss.backward()
    opt.step()
torch.cuda.synchronize()
elapsed = time.time() - t0

print(f"{STEPS} steps x batch {BATCH}: {elapsed:.2f}s  ({elapsed/STEPS*1000:.2f}ms/step)")
print(f"4 networks (bid+play adv+strat) would take: {elapsed*4:.2f}s per CFR iteration")
