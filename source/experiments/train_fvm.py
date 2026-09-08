"""Single-stage FV-Gauss PINN: IC, four boundaries and Saint-Venant PDE."""
import argparse
import json
import math
from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import train_dynamic_ga as ga
from loss import grad
from networks import GeometryEncoder, initialize_weights
from utils.geometry import water_area, water_area_at_x

NAMES = ("ic_z", "ic_q", "bc_q", "bc_z", "mass", "momentum")
KEYS = ("x", "t", "ic", "bc", "geo", "geo_mask", "bed", "manning_n",
        "upstream_z_bc", "downstream_q_bc")

def mlp(*widths):
    layers = []
    for i, (a, b) in enumerate(zip(widths, widths[1:])):
        layers += [nn.Linear(a, b)] + ([] if i == len(widths) - 2 else [nn.Tanh()])
    return nn.Sequential(*layers)

class TemporalEncoder(nn.Module):
    """Encode one 673-step boundary series into global and local tokens."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(1, 16, 7, padding=3), nn.Tanh(),
            nn.Conv1d(16, 32, 9, 2, 4), nn.Tanh(), nn.Conv1d(32, 32, 15, padding=7),
            nn.Tanh(), nn.Conv1d(32, 32, 7, padding=3), nn.Tanh())
        self.attention = nn.Conv1d(32, 1, 1)
        self.global_map, self.token_map = mlp(160, 64, 32), mlp(64, 32, 32)
        self.register_buffer("positions", torch.linspace(-1, 1, 64)[:, None])
    def forward(self, sequence):
        feature = self.conv(sequence.unsqueeze(1))
        weight = torch.softmax(self.attention(feature), -1)
        summary = torch.cat(((feature * weight).sum(-1), feature.mean(-1), feature.amax(-1), feature[..., 0], feature[..., -1]), -1)
        tokens = torch.cat((F.adaptive_avg_pool1d(feature, 64), F.adaptive_max_pool1d(feature, 64)), 1).transpose(1, 2)
        return self.global_map(summary), self.token_map(tokens)

class Query(nn.Module):
    """Query boundary tokens using learned space-time transport coordinates."""

    def __init__(self):
        super().__init__()
        self.query, self.key, self.value = mlp(6, 32, 32), nn.Linear(33, 32), nn.Linear(32, 32)
        self.output = nn.Sequential(nn.Linear(32, 32), nn.Tanh())
        self.locality = nn.Parameter(torch.tensor(1.0))
        self.transport = nn.Parameter(torch.zeros(4))
    def forward(self, coordinates, tokens, positions):
        token, (x, t) = tokens[0], coordinates.split(1, -1)
        query = self.query(torch.cat((coordinates, t - x * torch.tanh(self.transport)[None]), -1))
        score = query @ self.key(torch.cat((token, positions), -1)).T / math.sqrt(32)
        score = score - F.softplus(self.locality) * (t - positions.T).square()
        return self.output(torch.softmax(score, -1) @ self.value(token))

class PINN(nn.Module):
    """Fuse IC, boundary, geometry and coordinates, then predict Z and log-Q."""

    def __init__(self, condition_dim, scales):
        super().__init__()
        for name, value in scales.items():
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))
        self.ic = mlp(condition_dim - 1346, 64, 64, 32)
        self.q_encoder, self.z_encoder = TemporalEncoder(), TemporalEncoder()
        self.fuse_value = mlp(128, 64, 32)
        self.fuse_gate = nn.Sequential(nn.Linear(128, 64), nn.Sigmoid(), nn.Linear(64, 32), nn.Sigmoid())
        self.fuse_residual = nn.Sequential(nn.Linear(128, 32), nn.Tanh())
        self.geo = GeometryEncoder(32); self.trunk = mlp(14, 64, 64, 32)
        self.shared = nn.Sequential(mlp(96, 96, 64), nn.Tanh()); self.q_query, self.z_query = Query(), Query()
        self.query_fuse = mlp(96, 64, 32); self.terrain = GeometryEncoder(32)
        self.z_adapter = nn.Sequential(mlp(64, 64, 64), nn.Tanh())
        self.z_gate = nn.Parameter(torch.tensor(0.0))
        self.z_head, self.q_head = mlp(256, 128, 64, 1), mlp(256, 128, 64, 1)
        self.apply(initialize_weights)
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    def terrain_code(self, geo, mask):
        station, elevation = geo.unbind(-1)
        lo = station.masked_fill(~mask, torch.inf).amin(1, keepdim=True)
        hi = station.masked_fill(~mask, -torch.inf).amax(1, keepdim=True)
        bed = elevation.masked_fill(~mask, torch.inf).amin(1, keepdim=True)
        feature = torch.stack(((station - lo) / (hi - lo).clamp_min(1e-6), (elevation - bed) / self.elevation_std.clamp_min(1e-6)), -1)
        return self.terrain(torch.where(mask[..., None], feature, torch.zeros_like(feature)), mask)
    def forward(self, x, t, ic, bc, geo, mask, bed, geo_weight=None):
        xn = 2 * (x - self.x_min) / (self.x_max - self.x_min).clamp_min(1e-6) - 1
        tn = 2 * (t - self.t_min) / (self.t_max - self.t_min).clamp_min(1e-6) - 1
        coordinate = torch.cat((xn, tn), -1)
        encoded = [coordinate]
        for frequency in (1., 2., 4.):
            encoded += [torch.sin(math.pi * frequency * coordinate),
                        torch.cos(math.pi * frequency * coordinate)]
        iz, iq = ic.chunk(2, -1)
        bq, bz = bc.chunk(2, -1)
        ic_code = self.ic(torch.cat(((iz - self.z_mean) / self.z_std, (iq.clamp_min(1e-6).log() - self.q_log_mean) / self.q_log_std), -1)[:1]).expand(len(x), -1)
        q_global, q_tokens = self.q_encoder(
            ((bq.clamp_min(1e-6).log() - self.q_log_mean) / self.q_log_std)[:1])
        z_global, z_tokens = self.z_encoder(((bz - self.z_mean) / self.z_std)[:1])
        q_code, z_code = q_global.expand(len(x), -1), z_global.expand(len(x), -1)
        q_local, z_local = self.q_query(coordinate, q_tokens, self.q_encoder.positions), self.z_query(coordinate, z_tokens, self.z_encoder.positions)
        combined = torch.cat((ic_code, q_code, z_code, q_code * z_code), -1)
        gate = self.fuse_gate(combined)
        condition = gate * self.fuse_value(combined) + (1 - gate) * self.fuse_residual(combined) + self.query_fuse(torch.cat((q_local, z_local, q_local * z_local), -1))
        gn = torch.stack(((geo[..., 0] - self.station_mean) / self.station_std, (geo[..., 1] - self.elevation_mean) / self.elevation_std), -1)
        gn = torch.where(mask[..., None], gn, torch.zeros_like(gn))
        if geo_weight is None:
            geo_code, terrain = self.geo(gn, mask), self.terrain_code(geo, mask)
        else:
            gl, gr = gn.chunk(2, 1); ml, mr = mask.chunk(2, 1); rl, rr = geo.chunk(2, 1)
            geo_code = (1 - geo_weight) * self.geo(gl, ml) + geo_weight * self.geo(gr, mr)
            terrain = (1 - geo_weight) * self.terrain_code(rl, ml) + geo_weight * self.terrain_code(rr, mr)
        trunk = self.trunk(torch.cat(encoded, -1))
        shared = self.shared(torch.cat((condition, geo_code, trunk), -1))
        shared = shared + self.z_gate * self.z_adapter(torch.cat((terrain, z_local), -1))
        route = torch.cat((shared, q_code, q_local, z_local, geo_code, trunk, terrain), -1)
        z = bed + self.depth_mean * F.softplus(self.z_head(route))
        q = torch.exp(self.q_log_mean + self.q_log_std * self.q_head(route))
        return z, q

def make_scales(cases):
    z, q, depth, area, geometry = [], [], [], [], []
    for case in cases:
        z0, q0 = case["ic"].chunk(2); bq, bz = case["bc"].chunk(2); z += [z0, bz]; q += [q0, bq]
        depth += [z0 - case["bed"]]; area += [water_area(z0, case["geo"], case["geo_mask"])[0]]; geometry += [case["geo"][case["geo_mask"]]]
    z, q, depth, area, geometry = map(torch.cat, (z, q, depth, area, geometry)); x = torch.cat([c["x"] for c in cases]); t = torch.cat([c["t"] for c in cases])
    length, dref, aref = x.max() - x.min(), depth.median(), area.median(); velocity = (9.81 * dref).sqrt()
    return dict(x_min=x.min(), x_max=x.max(), t_min=t.min(), t_max=t.max(), z_mean=z.mean(), z_std=z.std(False).clamp_min(1e-6),
        q_log_mean=q.log().mean(), q_log_std=q.log().std().clamp_min(1e-6), station_mean=geometry[:, 0].mean(), station_std=geometry[:, 0].std(False),
        elevation_mean=geometry[:, 1].mean(), elevation_std=geometry[:, 1].std(False), depth_mean=depth.mean(), depth_ref=dref, area_ref=aref,
        length_ref=length, q_ref=aref * velocity, time_ref=length / velocity)

def state(model, case, x, t):
    right = torch.searchsorted(case["x"], x[:, 0]).clamp(1, len(case["x"]) - 1); left = right - 1
    weight = ((x[:, 0] - case["x"][left]) / (case["x"][right] - case["x"][left]))[:, None]; bed = (1 - weight) * case["bed"][left, None] + weight * case["bed"][right, None]
    z, q = model(x, t, case["ic"][None].expand(len(x), -1), case["bc"][None].expand(len(x), -1), torch.cat((case["geo"][left], case["geo"][right]), 1),
        torch.cat((case["geo_mask"][left], case["geo_mask"][right]), 1), bed, weight)
    area, perimeter = water_area_at_x(x, z, case["x"], case["geo"], case["geo_mask"]); area = area[:, None].clamp_min(1e-6)
    return z, q, area, (area / perimeter[:, None].clamp_min(1e-6)).clamp_min(1e-6)

def pinn_losses(model, cases, points):
    result, extra_z, extra_q = [], [], []
    for case in cases:
        device, count = case["x"].device, len(case["x"])
        z0, q0 = case["ic"].chunk(2)
        bq, bz = case["bc"].chunk(2)
        # Initial condition over every river section.
        z, q = model(case["x"][:, None], case["t"][:1].expand(count, 1), case["ic"][None].expand(count, -1), case["bc"][None].expand(count, -1), case["geo"], case["geo_mask"], case["bed"][:, None])
        terms = [((z[:, 0] - z0) / model.depth_ref).square().mean(), ((q[:, 0].log() - q0.log()) / model.q_log_std).square().mean()]
        # Encoded boundaries: upstream Q and downstream Z.
        index = torch.randint(len(case["t"]), (points,), device=device)
        time, boundary = case["t"][index, None], []
        for section, target, is_q in ((0, bq, True), (-1, bz, False)):
            zp, qp = model(torch.full((points, 1), case["x"][section].item(), device=device), time, case["ic"][None].expand(points, -1), case["bc"][None].expand(points, -1),
                case["geo"][section][None].expand(points, -1, -1), case["geo_mask"][section][None].expand(points, -1), case["bed"][section].expand(points, 1))
            residual = (qp[:, 0].log() - target[index].log()) / model.q_log_std if is_q else (zp[:, 0] - target[index]) / model.depth_ref
            boundary.append(residual.square().mean())
        # Random finite-volume cells: continuity on faces, momentum by Gauss integration.
        dx, dt = (case["x"][-1] - case["x"][0]) / 32, (case["t"][-1] - case["t"][0]) / 24
        xl = case["x"][0] + torch.rand(points, 1, device=device) * (case["x"][-1] - case["x"][0] - dx); xr = xl + dx
        tb = case["t"][0] + torch.rand(points, 1, device=device) * (case["t"][-1] - case["t"][0] - dt); tt = tb + dt; xm, tm = (xl + xr) / 2, (tb + tt) / 2
        _, qb, ab, _ = state(model, case, xm, tb); _, qt, at, _ = state(model, case, xm, tt); _, ql, al, _ = state(model, case, xl, tm); _, qr, ar, _ = state(model, case, xr, tm)
        mass = (at - ab)[:, 0] / dt * model.time_ref / model.area_ref + (qr - ql)[:, 0] / dx * model.length_ref / model.q_ref
        scale = model.area_ref * model.length_ref / model.q_ref.square()
        momentum = (qt - qb)[:, 0] / dt * model.time_ref / model.q_ref + (qr.square() / ar - ql.square() / al)[:, 0] / dx * scale
        nodes = torch.tensor([-math.sqrt(3 / 5), 0., math.sqrt(3 / 5)], device=device); weights = torch.tensor([5 / 9, 8 / 9, 5 / 9], device=device)
        gx = (xm + dx / 2 * nodes[None]).reshape(-1, 1).detach().requires_grad_(True)
        zg, qg, ag, rg = state(model, case, gx, tm.expand(-1, 3).reshape(-1, 1))
        ag, qg, rg = (v[:, 0].reshape(points, 3) for v in (ag, qg, rg))
        pressure = 9.81 * ag * grad(zg, gx)[:, 0].reshape(points, 3)
        friction = 9.81 * case["manning_n"] ** 2 * qg * qg.abs() / (ag * rg.pow(4 / 3))
        momentum = momentum + ((pressure + friction) * weights).sum(1) / 2 * scale
        result.append(torch.stack((terms[0], terms[1], boundary[0], boundary[1], mass.square().mean(), momentum.square().mean())))
    base = torch.stack(result).mean(0)
    # Extra observed boundaries affect the loss only, not the input encoder.
    for case in cases:
        device = case["x"].device
        index = torch.randint(len(case["t"]), (points,), device=device)
        time = case["t"][index, None]
        predictions = []
        for section in (0, -1):
            predictions.append(model(torch.full((points, 1), case["x"][section].item(), device=device), time,
                case["ic"][None].expand(points, -1), case["bc"][None].expand(points, -1), case["geo"][section][None].expand(points, -1, -1),
                case["geo_mask"][section][None].expand(points, -1), case["bed"][section].expand(points, 1)))
        extra_z.append(((predictions[0][0][:, 0] - case["upstream_z_bc"][index]) / model.depth_ref).square().mean())
        extra_q.append(((predictions[1][1][:, 0].log() - case["downstream_q_bc"][index].log()) / model.q_log_std).square().mean())
    return torch.stack((base[0], base[1], (base[2] + torch.stack(extra_q).mean()) / 2,
                        (base[3] + torch.stack(extra_z).mean()) / 2, base[4], base[5]))

def balance(model, task_losses):
    """Average six unit gradients so no task dominates solely by magnitude."""
    parameters, vectors = list(model.parameters()), []
    for i, loss in enumerate(task_losses):
        gs = torch.autograd.grad(loss, parameters, retain_graph=i < len(task_losses) - 1, allow_unused=True)
        vectors.append(torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(gs, parameters)]))
    gradients = torch.stack(vectors)
    norms = gradients.double().norm(dim=1).to(gradients.dtype)
    combined = (gradients / norms[:, None].clamp_min(1e-20)).mean(0)
    offset = 0
    for parameter in parameters:
        size = parameter.numel(); parameter.grad = combined[offset:offset + size].view_as(parameter).clone(); offset += size

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    seed, batch, points, manning = 2032, 4, 64, 0.020816
    torch.set_num_threads(4); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    name = "fixed_q_logq_fv_gauss_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or ga.project_dir / "outputs" / name
    output.mkdir(parents=True, exist_ok=False)
    load = lambda name: torch.load(ga.project_dir / f"data/pt/{name}_prepared.pt", map_location="cpu", weights_only=True)
    train_data, validation_data, test_data = load("train"), load("validation"), load("test")
    train_data = ga.sample_cases(train_data, 105, seed=seed)
    validation_data = ga.sample_cases(validation_data, 30, seed=seed)
    for case in train_data.values():
        case["manning_n"] = torch.full_like(case["manning_n"], manning)
        case["upstream_z_bc"] = case["z"][:, 0].clone()
        case["downstream_q_bc"] = case["q"][:, -1].clone()

    train_input = [{key: case[key] for key in KEYS} for case in train_data.values()]
    condition_dim = train_input[0]["ic"].numel() + train_input[0]["bc"].numel()
    model = PINN(condition_dim, make_scales(train_input)).to(device)
    train = [{key: value.to(device) for key, value in case.items()} for case in train_input]
    train_cases = list(train_data.values())
    validation = list(validation_data.values())
    ga.config["monitor"].update(time_step=24, time_batch=6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    history, best = [], float("inf")
    for epoch in range(1, args.epochs + 1):
        torch.manual_seed(seed + epoch)
        order = torch.randperm(len(train)).tolist()
        lr = 1e-4 + .5 * 9e-4 * (1 + math.cos(math.pi * (epoch - 1) / args.epochs))
        for group in optimizer.param_groups:
            group["lr"] = lr
        totals = torch.zeros(6, dtype=torch.float64, device=device)
        model.train()
        for start in range(0, len(train), batch):
            cases = [train[i] for i in order[start:start + batch]]
            task_losses = pinn_losses(model, cases, points)
            optimizer.zero_grad(set_to_none=True)
            balance(model, task_losses)
            optimizer.step()
            totals += task_losses.detach().double() * len(cases)
        mean_losses = (totals / len(train)).tolist()
        train_z, train_q = ga.relative_error(model, train_cases, device, 24)
        row = {"epoch": epoch, "lr": lr, **dict(zip(NAMES, mean_losses)),
               "train_z_error": train_z, "train_q_error": train_q}
        message = (f"epoch={epoch:02d} lr={lr:.2e} " +
                   " ".join(f"{name}={value:.3e}" for name, value in zip(NAMES, mean_losses)) +
                   f" train_z={train_z:.4f}% train_q={train_q:.4f}%")
        if epoch % 5 == 0 or epoch == args.epochs:
            z_error, q_error = ga.relative_error(model, validation, device, 24)
            row.update(z_error=z_error, q_error=q_error, score=max(z_error, q_error))
            if row["score"] < best:
                best = row["score"]
                state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                torch.save({"model_state_dict": state, "epoch": epoch, "validation": row}, output / "best.pt")
            message += f" val_z={z_error:.4f}% val_q={q_error:.4f}% best={best:.4f}%"
        print(message, flush=True)
        history.append(row)
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    selected = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(selected["model_state_dict"])
    test_z, test_q = ga.relative_error(model, list(test_data.values()), device,
                                       ga.config["monitor"]["final_test_time_step"])
    result = {"checkpoint_epoch": selected["epoch"], "validation": selected["validation"],
              "test_z": test_z, "test_q": test_q, "test_max": max(test_z, test_q)}
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(result, flush=True)

if __name__ == "__main__":
    main()
