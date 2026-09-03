import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometryEncoder(nn.Module):
    def __init__(self, geo_dim=32):
        super().__init__()

        self.point_net = nn.Sequential(
            nn.Linear(2, 64), nn.GELU(),
            nn.Linear(64, geo_dim),
        )

    def forward(self, geo, mask):
        """
        geo:  [batch, max_points, 2]
        mask: [batch, max_points]
        """
        feature = self.point_net(geo)           # [B, P, geo_dim]
        mask = mask.unsqueeze(-1).float()       # [B, P, 1]

        # 对有效地形点取平均
        feature = (feature * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp_min(1.0)

        return feature / count                  # [B, geo_dim]


class OperatorPINN(nn.Module):
    def __init__(self, condition_dim, scales):
        super().__init__()

        # 归一化参数只由训练集计算。注册为 buffer 后，它们会随模型一起移动到 CPU/MPS/CUDA，但不会被优化器更新。
        for name, value in scales.items():
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))

        self.condition_branch = nn.Sequential(
            nn.Linear(condition_dim, 64), nn.GELU(),
            nn.Linear(64, 64), nn.GELU(),
            nn.Linear(64, 32),
        )

        self.geo_branch = GeometryEncoder(geo_dim=32)

        self.trunk = nn.Sequential(
            nn.Linear(2, 64), nn.GELU(),
            nn.Linear(64, 64), nn.GELU(),
            nn.Linear(64, 32),
        )

        self.shared_head = nn.Sequential(
            nn.Linear(32 * 3, 64), nn.GELU(),
            nn.Linear(64, 64), nn.GELU()
        )

        # 水深（水位）输出头
        self.z_head = nn.Linear(64, 1)

        # 流量输出头
        self.q_head = nn.Linear(64, 1)


    def forward(self, x, t, ic, bc, geo, geo_mask, bed, debug=False):

        # 坐标和时间映射到 [-1, 1] 区间
        x_range = (self.x_max - self.x_min).clamp_min(1e-6)
        t_range = (self.t_max - self.t_min).clamp_min(1e-6)

        x_net = (x - self.x_min) / x_range * 2.0 - 1.0
        t_net = (t - self.t_min) / t_range * 2.0 - 1.0

        # ic = [初始水位, 初始流量]，bc = [上游流量, 下游水位]
        # 水位与流量分别使用各自的统计量归一化。
        ic_z, ic_q = torch.chunk(ic, 2, dim=-1)
        bc_q, bc_z = torch.chunk(bc, 2, dim=-1)
        ic_net = torch.cat([
            (ic_z - self.z_mean) / self.z_std,
            (ic_q - self.q_mean) / self.q_std,
        ], dim=-1)
        bc_net = torch.cat([
            (bc_q - self.q_mean) / self.q_std,
            (bc_z - self.z_mean) / self.z_std,
        ], dim=-1)

        # 几何归一化
        station_net = (geo[..., 0] - self.station_mean) / self.station_std
        elevation_net = (geo[..., 1] - self.elevation_mean) / self.elevation_std
        geo_net = torch.stack([station_net, elevation_net], dim=-1)
        geo_net = torch.where(
            geo_mask.unsqueeze(-1), geo_net, torch.zeros_like(geo_net)
        )

        # 三个编码分支统一使用模型内部生成的归一化输入。
        condition_code = self.condition_branch(torch.cat([ic_net, bc_net], dim=-1))
        geo_code = self.geo_branch(geo_net, geo_mask)
        trunk_code = self.trunk(torch.cat([x_net, t_net], dim=-1))

        shared_feature = self.shared_head(
            torch.cat([condition_code, geo_code, trunk_code], dim=-1)
        )

        raw_depth = self.z_head(shared_feature)
        raw_q = self.q_head(shared_feature)

        if debug:
            print(
                "raw_q:",
                "min =", raw_q.min().item(),
                "max =", raw_q.max().item(),
                "mean =", raw_q.mean().item(),
                "std =", raw_q.std(unbiased=False).item(),
            )

            print(
                "q_mean =", self.q_mean.item(),
                "q_std =", self.q_std.item(),
            )

        depth = self.depth_mean * F.softplus(raw_depth)  # softplus 保证深度为正
        z = bed + depth
        
        q = self.q_mean + self.q_std * raw_q

        return z, q
