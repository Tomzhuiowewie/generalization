import random
import torch

def grad(y, x):

    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y), create_graph=True
        )[0]

def sample_cases(dataset, count, seed=2032):
    rng = random.Random(seed)
    groups = {}
    for key in sorted(dataset):
        groups.setdefault(key.split("_")[0], []).append(key)

    for keys in groups.values():
        rng.shuffle(keys)

    selected = []
    while len(selected) < min(count, len(dataset)):
        active = [geo for geo, keys in groups.items() if keys]
        rng.shuffle(active)
        for geo in active:
            selected.append(groups[geo].pop())
            if len(selected) == min(count, len(dataset)):
                break

    return {key: dataset[key] for key in selected}

def relative_error(model, cases, time_step, time_batch):
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    depth_total = q_total = 0.0
    with torch.no_grad():
        for case in cases:
            x, bed = case["x"].to(device), case["bed"].to(device)[:, None]
            geo, mask = case["geo"].to(device), case["geo_mask"].to(device)
            indices = torch.arange(0, len(case["t"]), time_step)
            depth_sum = q_sum = count = 0
            for start in range(0, len(indices), time_batch):
                selected = indices[start:start + time_batch]
                nt, nx = len(selected), len(x)
                t = case["t"][selected].to(device)[:, None].expand(-1, nx).reshape(-1, 1)
                mx = x[None].expand(nt, -1).reshape(-1, 1)
                ic = case["ic"].to(device)[None].expand(nt * nx, -1)
                bc = case["bc"].to(device)[None].expand(nt * nx, -1)
                mg = geo[None].expand(nt, -1, -1, -1).reshape(nt * nx, geo.shape[1], 2)
                mm = mask[None].expand(nt, -1, -1).reshape(nt * nx, mask.shape[1])
                mb = bed[None].expand(nt, -1, -1).reshape(-1, 1)
                z, q = model(mx, t, ic, bc, mg, mm, mb)
                true_z = case["z"][selected].to(device).reshape(-1, 1)
                true_q = case["q"][selected].to(device).reshape(-1, 1)
                depth_sum += ((z - true_z).abs() / (true_z - mb).abs().clamp_min(1e-6)).sum().item()
                q_sum += ((q - true_q).abs() / true_q.abs().clamp_min(1e-6)).sum().item()
                count += true_z.numel()
            depth_total += 100 * depth_sum / count
            q_total += 100 * q_sum / count
    model.train(was_training)
    return depth_total / len(cases), q_total / len(cases)

class EarlyStopping:
    """指标越小越好；连续 patience 次调用无足够改善时返回 True。"""

    def __init__(self, patience=5, min_delta=1e-3):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0

    def step(self, value):
        if value < self.best - self.min_delta:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience
