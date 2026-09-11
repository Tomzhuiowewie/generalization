import json, math, yaml, sys
from datetime import datetime
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
config_path = SOURCE / "config.yaml"

import torch
import torch.nn as nn
import torch.nn.functional as F

from loss import grad
from utils.common import relative_error, sample_cases
from utils.geometry import water_area, water_area_at_x

with config_path.open(encoding="utf-8") as file:
    config = yaml.safe_load(file)

# 多层感知机
def mlp(*widths):
    layers = []
    for layer_index, (input_size, output_size) in enumerate(zip(widths[:-1], widths[1:])):
        layers.append(nn.Linear(input_size, output_size))
        if layer_index < len(widths) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)

# xavier初始化
def initialize_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=1.0)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

# 地形编码器
class GeometryEncoder(nn.Module):
    def __init__(self, geo_dim=64):
        super().__init__()

        self.point_net = nn.Sequential(
            nn.Linear(2, 128), nn.Tanh(),
            nn.Linear(128, geo_dim),
        )

    def forward(self, geo, mask):
        feature = self.point_net(geo)           # [B, P, geo_dim]
        mask = mask.unsqueeze(-1).float()       # [B, P, 1]

        # 对有效地形点取平均
        feature = (feature * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp_min(1.0)

        return feature / count                  # [B, geo_dim]

# 边界条件编码：使用一维卷积编码边界时间序列
class TemporalEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3),
            nn.Tanh(),
            nn.Conv1d(16, 32, 5, padding=2),
            nn.Tanh(),
            nn.Conv1d(32, 32, 3, padding=1),
            nn.Tanh(),
        )

        self.attention = nn.Conv1d(32, 1, 1)
        self.global_map = mlp(160, 64, 32)
        self.token_map = mlp(64, 32, 32)

        self.register_buffer("positions", torch.linspace(-1, 1, 64)[:, None],)

    def forward(self, sequence):
        feature = self.conv(sequence.unsqueeze(1))

        attention = torch.softmax(
            self.attention(feature),
            dim=-1,
        )

        summary = torch.cat((
            (feature * attention).sum(-1),
            feature.mean(-1),
            feature.amax(-1),
            feature[..., 0],
            feature[..., -1],
        ), dim=-1)

        tokens = torch.cat((
            F.adaptive_avg_pool1d(feature, 64),
            F.adaptive_max_pool1d(feature, 64),
        ), dim=1).transpose(1, 2)

        return (
            self.global_map(summary),
            self.token_map(tokens),
        )

# 初始条件编码：显式使用不规则空间坐标和相邻断面间距
class CoordinateSpatialEncoder(nn.Module):
    def __init__(self, source_x, token_count=64):
        super().__init__()
        source_x = source_x.float()
        length = (source_x[-1] - source_x[0]).clamp_min(1e-6)
        xn = 2 * (source_x - source_x[0]) / length - 1
        gaps = source_x[1:] - source_x[:-1]
        mean_gap = gaps.mean().clamp_min(1e-6)
        left_gap = torch.cat((gaps[:1], gaps)) / mean_gap
        right_gap = torch.cat((gaps, gaps[-1:])) / mean_gap

        target_x = torch.linspace(source_x[0], source_x[-1], token_count)
        right = torch.searchsorted(source_x, target_x).clamp(1, len(source_x) - 1)
        left = right - 1
        token_weight = (
            (target_x - source_x[left])
            / (source_x[right] - source_x[left]).clamp_min(1e-6)
        )

        cell_width = torch.empty_like(source_x)
        cell_width[0] = gaps[0] / 2
        cell_width[-1] = gaps[-1] / 2
        cell_width[1:-1] = (source_x[2:] - source_x[:-2]) / 2

        self.register_buffer("xn", xn)
        self.register_buffer("left_gap", left_gap)
        self.register_buffer("right_gap", right_gap)
        self.register_buffer("token_left", left)
        self.register_buffer("token_right", right)
        self.register_buffer("token_weight", token_weight)
        self.register_buffer("cell_weight", cell_width / cell_width.sum())
        self.register_buffer("positions", torch.linspace(-1, 1, token_count)[:, None])

        self.point_map = nn.Sequential(
            nn.Linear(4, 64), nn.Tanh(),
            nn.Linear(64, 32), nn.Tanh(),
        )
        self.updates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(96, 64), nn.Tanh(),
                nn.Linear(64, 32), nn.Tanh(),
            )
            for _ in range(2)
        ])
        self.attention = nn.Linear(32, 1)
        self.global_map = mlp(160, 64, 32)
        self.token_map = mlp(32, 32, 32)

    def forward(self, sequence):
        batch = sequence.shape[0]
        point_input = torch.stack((
            sequence,
            self.xn.expand(batch, -1),
            self.left_gap.log().expand(batch, -1),
            self.right_gap.log().expand(batch, -1),
        ), -1)
        feature = self.point_map(point_input)

        for update in self.updates:
            left = torch.cat((feature[:, :1], feature[:, :-1]), 1)
            right = torch.cat((feature[:, 1:], feature[:, -1:]), 1)
            left_slope = (feature - left) / self.left_gap[None, :, None]
            right_slope = (right - feature) / self.right_gap[None, :, None]
            feature = feature + update(
                torch.cat((feature, left_slope, right_slope), -1)
            )

        spatial_weight = self.cell_weight[None, :, None]
        attention = torch.softmax(
            self.attention(feature)
            + spatial_weight.clamp_min(1e-12).log(),
            1,
        )
        summary = torch.cat((
            (feature * attention).sum(1),
            (feature * spatial_weight).sum(1),
            feature.amax(1), feature[:, 0], feature[:, -1],
        ), -1)

        weight = self.token_weight[None, :, None]
        tokens = (
            feature[:, self.token_left] * (1 - weight)
            + feature[:, self.token_right] * weight
        )
        return self.global_map(summary), self.token_map(tokens)

# 查询器：使用学习得到的时空输运坐标来查询边界 token
class Query(nn.Module):
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

# 条件融合，预测
class PINN(nn.Module):
    def __init__(self, ic_dim, scales):
        super().__init__()
        for name, value in scales.items():
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))

        # IC 是沿河道的不规则空间场，水位和流量分别编码。
        self.ic_z_encoder = CoordinateSpatialEncoder(self.ic_x)
        self.ic_q_encoder = CoordinateSpatialEncoder(self.ic_x)
        self.ic_z_query, self.ic_q_query = Query(), Query()

        # BC 是规则采样的时间序列，采用一维卷积编码
        self.q_encoder = TemporalEncoder()
        self.z_encoder = TemporalEncoder()
        
        # 直接融合IC水位、IC流量、BC流量和BC水位
        self.global_fuse = mlp(128, 64, 32)
        self.local_fuse = mlp(128, 64, 32)

        # 地形编码器（断面内部相对地形）
        self.geo = GeometryEncoder(32)

        # 时空编码（坐标）
        self.trunk = mlp(14, 64, 64, 32)

        # 共享网络
        self.shared = nn.Sequential(mlp(96, 96, 64), nn.Tanh())

        # 查询器：提取与当前预测位置相关的边界局部信息。
        self.q_query, self.z_query = Query(), Query()
        # 水位/流量输出头
        self.z_head = mlp(192, 128, 64, 1)
        self.q_head = mlp(192, 128, 64, 1)  

        self.apply(initialize_weights)
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    # 相对地形编码
    def terrain_code(self, geo, mask):
        station, elevation = geo.unbind(-1)
        lo = station.masked_fill(~mask, torch.inf).amin(1, keepdim=True)
        hi = station.masked_fill(~mask, -torch.inf).amax(1, keepdim=True)
        bed = elevation.masked_fill(~mask, torch.inf).amin(1, keepdim=True)
        feature = torch.stack((
            (station - lo) / (hi - lo), 
            (elevation - bed) / self.elevation_std,
        ), -1)
        feature = torch.where(
            mask[..., None], feature, torch.zeros_like(feature),
        )
        return self.geo(feature, mask)

    def encode_conditions(self, ic, bc):
        iz, iq = ic.chunk(2, dim=-1)
        bq, bz = bc.chunk(2, dim=-1)

        iz = ((iz - self.z_mean) / self.z_std)[:1]
        iq = ((iq.clamp_min(1e-6).log() - self.q_log_mean) / self.q_log_std)[:1]
        bq = ((bq.clamp_min(1e-6).log() - self.q_log_mean) / self.q_log_std)[:1]
        bz = ((bz - self.z_mean) / self.z_std)[:1]

        iz_global, iz_tokens = self.ic_z_encoder(iz)
        iq_global, iq_tokens = self.ic_q_encoder(iq)
        bq_global, bq_tokens = self.q_encoder(bq)
        bz_global, bz_tokens = self.z_encoder(bz)

        global_condition = self.global_fuse(torch.cat((
            iz_global, iq_global, bq_global, bz_global,
        ), dim=-1))

        return (global_condition, iz_tokens, iq_tokens, bq_tokens, bz_tokens)

    def forward(self, x, t, ic, bc, geo, mask, bed, geo_weight=None, condition_cache=None):
        xn = 2 * (x - self.x_min) / (self.x_max - self.x_min).clamp_min(1e-6) - 1
        tn = 2 * (t - self.t_min) / (self.t_max - self.t_min).clamp_min(1e-6) - 1
        coordinate = torch.cat((xn, tn), -1)
        encoded = [coordinate]  # 原始坐标 + 频率1的sin、cos + 频率2的sin、cos + 频率4的sin、cos
        for frequency in (1., 2., 4.):
            encoded += [torch.sin(math.pi * frequency * coordinate),
                        torch.cos(math.pi * frequency * coordinate)]

        if condition_cache is None:
            condition_cache = self.encode_conditions(ic, bc)

        (global_condition, iz_tokens, iq_tokens, bq_tokens, bz_tokens) = condition_cache
        global_condition = global_condition.expand(x.shape[0], -1)

        # Query 的第二个坐标用于局部约束；交换 x/t 后即可查询空间 token。
        spatial_coordinate = coordinate.flip(-1)
        iz_local = self.ic_z_query(
            spatial_coordinate, iz_tokens, self.ic_z_encoder.positions
        )
        iq_local = self.ic_q_query(
            spatial_coordinate, iq_tokens, self.ic_q_encoder.positions
        )
        bq_local = self.q_query(
            coordinate, bq_tokens, self.q_encoder.positions
        )
        bz_local = self.z_query(
            coordinate, bz_tokens, self.z_encoder.positions
        )

        local_condition = self.local_fuse(torch.cat((
            iz_local, iq_local, bq_local, bz_local,
        ), dim=-1))
        condition = global_condition + local_condition

        if geo_weight is None:
            geo_code = self.terrain_code(geo, mask)
        else:
            left_geo, right_geo = geo.chunk(2, 1)
            left_mask, right_mask = mask.chunk(2, 1)

            geo_code = (
                (1 - geo_weight) * self.terrain_code(left_geo, left_mask)
                + geo_weight * self.terrain_code(right_geo, right_mask)
            )

        trunk = self.trunk(torch.cat(encoded, -1))
        shared = self.shared(torch.cat((condition, geo_code, trunk), -1))
        route = torch.cat((
            shared, global_condition, local_condition, geo_code, trunk,
        ), dim=-1)

        z = bed + self.depth_mean * F.softplus(self.z_head(route))
        q = torch.exp(self.q_log_mean + self.q_log_std * self.q_head(route))
        return z, q

def make_scales(cases):
    z, q, depth, area, geometry = [], [], [], [], []
    for case in cases:
        z0, q0 = case["ic"].chunk(2)    # 初始水位、初始流量
        bq, bz = case["bc"].chunk(2)    # 边界流量、边界水位
        z += [z0, bz]; q += [q0, bq]
        depth += [z0 - case["bed"]]
        area += [water_area(z0, case["geo"], case["geo_mask"])[0]]
        geometry += [case["geo"][case["geo_mask"]]]
    z, q, depth, area, geometry = map(torch.cat, (z, q, depth, area, geometry))
    x = torch.cat([c["x"] for c in cases])
    t = torch.cat([c["t"] for c in cases])
    length, dref, aref = x.max() - x.min(), depth.median(), area.median()
    velocity = (9.81 * dref).sqrt()
    return dict(
        x_min=x.min(), x_max=x.max(), t_min=t.min(), t_max=t.max(),
        ic_x=cases[0]["x"],
        z_mean=z.mean(), z_std=z.std(False),
        q_log_mean=q.log().mean(), q_log_std=q.log().std(), 
        station_mean=geometry[:, 0].mean(), station_std=geometry[:, 0].std(False),
        elevation_mean=geometry[:, 1].mean(), elevation_std=geometry[:, 1].std(False), 
        depth_mean=depth.mean(), depth_ref=dref, area_ref=aref, length_ref=length, 
        q_ref=aref * velocity, time_ref=length / velocity)


def state(model, case, x, t, condition_cache=None):   # 计算水力状态
    """计算任意位置：水位、流量、过水面积、水力半径"""
    right = torch.searchsorted(case["x"], x[:, 0]).clamp(1, len(case["x"]) - 1)
    left = right - 1
    weight = ((x[:, 0] - case["x"][left]) / (case["x"][right] - case["x"][left]))[:, None]
    bed = (1 - weight) * case["bed"][left, None] + weight * case["bed"][right, None]
    z, q = model(
        x, t, 
        case["ic"][None].expand(len(x), -1), case["bc"][None].expand(len(x), -1), 
        torch.cat((case["geo"][left], case["geo"][right]), 1),
        torch.cat((case["geo_mask"][left], case["geo_mask"][right]), 1), 
        bed, weight, condition_cache=condition_cache
    )
    area, perimeter = water_area_at_x(x, z, case["x"], case["geo"], case["geo_mask"])
    area = area[:, None].clamp_min(1e-6)
    return z, q, area, (area / perimeter[:, None].clamp_min(1e-6)).clamp_min(1e-6)

def pinn_losses(model, cases, points):
    result = []
    for case in cases:
        device, count = case["x"].device, len(case["x"])
        z0, q0 = case["ic"].chunk(2)    # 初始水位、初始流量
        bq, bz = case["bc"].chunk(2)    # 边界流量、边界水位
        condition_cache = model.encode_conditions(
            case["ic"][None],
            case["bc"][None],
        )
        # 初始损失
        z, q = model(
            case["x"][:, None], case["t"][:1].expand(count, 1), 
            case["ic"][None].expand(count, -1), case["bc"][None].expand(count, -1), 
            case["geo"], case["geo_mask"], 
            case["bed"][:, None], 
            condition_cache=condition_cache,
        )
        init_loss = [((z[:, 0] - z0) / model.depth_ref).square().mean(),    # 初始水位损失
                 ((q[:, 0].log() - q0.log()) / model.q_log_std).square().mean()]    # 初始流量损失

        # 边界损失
        index = torch.randint(len(case["t"]), (points,), device=device)
        time = case["t"][index, None]
        boundary_z, boundary_q = [], []
        boundaries = ((0,  case["upstream_z_bc"], bq), (-1, bz, case["downstream_q_bc"]))
        for section, z_target, q_target in boundaries:
            zp, qp = model(
                torch.full((points, 1), case["x"][section].item(), device=device), time, 
                case["ic"][None].expand(points, -1), case["bc"][None].expand(points, -1),
                case["geo"][section][None].expand(points, -1, -1), case["geo_mask"][section][None].expand(points, -1), 
                case["bed"][section].expand(points, 1),
                condition_cache=condition_cache,
            )
            boundary_z.append(((zp[:, 0] - z_target[index]) / model.depth_ref).square().mean()) # 上游/下游水位损失
            boundary_q.append(((qp[:, 0].log() - q_target[index].log()) / model.q_log_std).square().mean()) # 上游/下游损失

        # PDE损失：随机选取有限体积单元：在单元面上计算连续性方程，通过高斯积分计算动量方程
        dx, dt = (case["x"][-1] - case["x"][0]) / 64, (case["t"][-1] - case["t"][0]) / 48   # 时空区域划分
        xl = case["x"][0] + torch.rand(points, 1, device=device) * (case["x"][-1] - case["x"][0] - dx)  # 左边界
        xr = xl + dx    # 右边界
        tb = case["t"][0] + torch.rand(points, 1, device=device) * (case["t"][-1] - case["t"][0] - dt)  # 下边界
        tt = tb + dt    # 上边界
        xm, tm = (xl + xr) / 2, (tb + tt) / 2   # 中心点

        face_x = torch.cat((xm, xm, xl, xr), dim=0)
        face_t = torch.cat((tb, tt, tm, tm), dim=0)
        _, face_q, face_area, _ = state(model, case, face_x, face_t, condition_cache)

        qb, qt, ql, qr = face_q.chunk(4, dim=0)
        ab, at, al, ar = face_area.chunk(4, dim=0)

        # 质量方程
        mass = (
            (at - ab)[:, 0] / dt * model.time_ref / model.area_ref      # 过水面积随时间的变化
            + (qr - ql)[:, 0] / dx * model.length_ref / model.q_ref     # 流量随空间的变化
        )
        # 动量方程
        scale = model.area_ref * model.length_ref / model.q_ref.square()
        momentum_part = (
            (qt - qb)[:, 0] / dt * model.time_ref / model.q_ref         # 流量随时间的变化
            + (qr.square() / ar - ql.square() / al)[:, 0] / dx * scale  # 动量通量随空间的变化
        )
        # 三点高斯积分节点和权重
        nodes = torch.tensor([-math.sqrt(3 / 5), 0., math.sqrt(3 / 5)], device=device)
        weights = torch.tensor([5 / 9, 8 / 9, 5 / 9], device=device)
        # 映射到真实有限体积单元
        gx = (xm + dx / 2 * nodes[None]).reshape(-1, 1).detach().requires_grad_(True)
        # 高斯点预测水力状态
        zg, qg, ag, rg = state(model, case, gx, tm.expand(-1, 3).reshape(-1, 1), condition_cache)
        # 恢复成:单元 x 高斯点
        ag, qg, rg = (v[:, 0].reshape(points, 3) for v in (ag, qg, rg))
        # 压力项、Manning摩阻项
        pressure = 9.81 * ag * grad(zg, gx)[:, 0].reshape(points, 3)
        friction = 9.81 * case["manning_n"] ** 2 * qg * qg.abs() / (ag * rg.pow(4 / 3))
        momentum = momentum_part + ((pressure + friction) * weights).sum(1) / 2 * scale

        result.append(torch.stack((
            init_loss[0], init_loss[1], 
            torch.stack(boundary_q).mean(), torch.stack(boundary_z).mean(), 
            mass.square().mean(), momentum.square().mean()
        )))

    return torch.stack(result).mean(0)


def balance(model, batch_losses):
    """对六个单位化梯度取平均，使得不会有某个任务仅仅因为梯度幅值更大而占据主导"""
    parameters, vectors = list(model.parameters()), []
    for i, loss in enumerate(batch_losses):
        gs = torch.autograd.grad(
            loss, parameters, 
            retain_graph=i < len(batch_losses) - 1, 
            allow_unused=True)
        vectors.append(torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(gs, parameters)]))
    gradients = torch.stack(vectors)
    norms = gradients.double().norm(dim=1).to(gradients.dtype)
    combined = (gradients / norms[:, None].clamp_min(1e-20)).mean(0)
    offset = 0
    for parameter in parameters:
        size = parameter.numel() 
        parameter.grad = combined[offset:offset + size].view_as(parameter).clone()
        offset += size

def main():
    cfg = config["fvm2"]; seed = cfg["seed"]
    epochs, batch, points, manning = cfg["epochs"], cfg["cases_per_batch"], cfg["points_per_case"], cfg["manning_n"]
    torch.set_num_threads(cfg["num_threads"]); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_name = ("ic_z", "ic_q", "bc_q", "bc_z", "mass", "momentum")
    keys = ("x", "t", "ic", "bc", "geo", "geo_mask", "bed", "manning_n", "upstream_z_bc", "downstream_q_bc")

    output_root = (config_path.parent / cfg["output_path"]).resolve()
    output = output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)

    load = lambda name: torch.load(
        (config_path.parent.resolve().parent / config["paths"]["pt"][name]).resolve(),
        map_location="cpu", weights_only=True)
    train_data, val_data, test_data = load("train"), load("validation"), load("test")
    train_data, val_data = sample_cases(train_data, 210, seed), sample_cases(val_data, 60, seed)

    train_input = []
    for case in train_data.values():
        case["manning_n"] = torch.full_like(case["manning_n"], manning)
        case["upstream_z_bc"] = case["z"][:, 0].clone()     # 上游水位
        case["downstream_q_bc"] = case["q"][:, -1].clone()  # 下游流量
        train_input.append({key: case[key] for key in keys})

    ic_dim = train_input[0]["ic"].numel()
    model = PINN(ic_dim, make_scales(train_input)).to(device)

    train = [{key: value.to(device) for key, value in case.items()} for case in train_input]
    train_cases,val_cases = list(train_data.values()), list(val_data.values())
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])

    history, best = [], float("inf")
    for epoch in range(1, epochs + 1):
        torch.manual_seed(seed + epoch)
        order = torch.randperm(len(train)).tolist()
        lr = cfg["min_learning_rate"] + .5 * (cfg["learning_rate"] - cfg["min_learning_rate"]) * (1 + math.cos(math.pi * (epoch - 1) / epochs))
        for group in optimizer.param_groups: group["lr"] = lr

        total_losses = torch.zeros(6, dtype=torch.float64, device=device)
        model.train()
        for start in range(0, len(train), batch):
            cases = [train[i] for i in order[start:start + batch]]
            batch_losses = pinn_losses(model, cases, points)
            optimizer.zero_grad(set_to_none=True)
            balance(model, batch_losses)    # 梯度求平均
            optimizer.step()
            total_losses += batch_losses.detach().double() * len(cases)
        
        mean_losses = (total_losses / len(train)).tolist()
        row = {"epoch": epoch, "lr": lr, **dict(zip(loss_name, mean_losses))}
        message = (f"epoch={epoch:02d} lr={lr:.2e} " +
                " ".join(f"{name}={value:.3e}" for name, value in zip(loss_name, mean_losses)))

        # 验证误差
        if epoch % 5 == 0 or epoch == epochs:
            train_z, train_q = relative_error(
                model, train_cases,
                cfg["val_time_step"], cfg["val_time_batch"],
            )
            val_z_error, val_q_error = relative_error(
                model, val_cases, 
                cfg["val_time_step"], cfg["val_time_batch"]
            )
            row.update(
                train_z_error=train_z, train_q_error=train_q,
                val_z_error=val_z_error, val_q_error=val_q_error, 
                score=max(val_z_error, val_q_error)
            )
            
            if row["score"] < best:
                best = row["score"]
                state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                torch.save({"model_state_dict": state, "epoch": epoch, "validation": row}, output / "best.pt")

            message += (
                f" train_z={train_z:.4f}% train_q={train_q:.4f}%"
                f" val_z={val_z_error:.4f}% val_q={val_q_error:.4f}%"
                f" best={best:.4f}%"
            )

        print(message, flush=True)
        history.append(row)
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    # 测试误差
    selected = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(selected["model_state_dict"])
    test_z, test_q = relative_error(model, list(test_data.values()),
                                    config["monitor"]["final_test_time_step"],
                                    cfg["val_time_batch"])
    result = {
        "checkpoint_epoch": selected["epoch"], 
        "validation": selected["validation"],
        "test_z": test_z,  "test_q": test_q,  "test_max": max(test_z, test_q)
        }
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(result, flush=True)

if __name__ == "__main__":
    main()
