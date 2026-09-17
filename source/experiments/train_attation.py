import json, math, sys, yaml
from datetime import datetime
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
config_path = SOURCE / "config.yaml"

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks import GeometryEncoder, initialize_weights
from train_fvm import balance, make_scales, pinn_losses
from utils.common import relative_error, sample_cases

with config_path.open(encoding="utf-8") as file:
    config = yaml.safe_load(file)


def mlp(*widths):
    layers = []
    for index, (input_size, output_size) in enumerate(zip(widths[:-1], widths[1:])):
        layers.append(nn.Linear(input_size, output_size))
        if index < len(widths) - 2: layers.append(nn.Tanh())
    return nn.Sequential(*layers)

# 交叉注意力模块
class CrossAttention(nn.Module):
    """Paper PINTO CAU: two 64-D heads and a two-layer dense residual block."""

    def __init__(self, width=64, heads=2, key_dim=64):  # 输入/输出的特征维度、注意头、每个头的维度
        super().__init__()
        self.heads, self.key_dim = heads, key_dim   # 64 
        projection = heads * key_dim    # 128 
        self.query, self.key, self.value = (nn.Linear(width, projection) for _ in range(3))
        self.output = nn.Linear(projection, width)
        self.dense = nn.Sequential(nn.Linear(width, width), nn.SiLU(),
                                   nn.Linear(width, width), nn.SiLU()) # 残差网络

    def forward(self, query, key, value):
        count, length = len(query), len(key)    # 查询点数、条件点数
        q = self.query(query).view(count, self.heads, self.key_dim) # [N, 128] → [N, 2, 64]
        k = self.key(key).view(length, self.heads, self.key_dim)
        v = self.value(value).view(length, self.heads, self.key_dim)

        score = torch.einsum("nhd,lhd->nhl", q, k) / math.sqrt(self.key_dim)    # 每个查询点和条件点的注意力得分
        weights = torch.softmax(score, -1)  
        context = torch.einsum("nhl,lhd->nhd", weights, v)     # 加权求和
        attended = F.silu(query + self.output(context.reshape(count, -1)))      # 合并注意力头-->映射回原维度-->进行残差融合和激活
        return attended + self.dense(attended)


class PINN(nn.Module):
    """Paper-style PINTO adapted to river IC, boundary series and geometry."""

    def __init__(self, ic_dim, scales):
        super().__init__()
        for name, value in scales.items():
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))
        self.qpe = mlp(2, 64, 64)   # query point encoder
        self.bpe = mlp(2, 64, 64)   # boundary point encoder
        self.bve = mlp(2, 64, 64)   # boundary value encoder

        # 创建两个交叉注意力模块
        self.cross_attention = nn.ModuleList([CrossAttention() for _ in range(2)])

        # 河道几何数据编码
        self.geo = GeometryEncoder(32)  # 全局信息（绝对空间背景）-- 训练集统计量全局标准化
        self.terrain = GeometryEncoder(32)  # 河床局部形态 -- 单独进行归一化

        # 共享网络
        self.shared = mlp(128, 128, 64)
        # 水位/流量输出头
        self.z_head = mlp(128, 64, 1)
        self.q_head = mlp(128, 64, 1)
        
        self.apply(initialize_weights)

    def terrain_code(self, geo, mask):
        station, elevation = geo.unbind(-1)
        lo = station.masked_fill(~mask, torch.inf).amin(1, keepdim=True)
        hi = station.masked_fill(~mask, -torch.inf).amax(1, keepdim=True)
        bed = elevation.masked_fill(~mask, torch.inf).amin(1, keepdim=True)
        feature = torch.stack(((station - lo) / (hi - lo).clamp_min(1e-6),
                               (elevation - bed) / self.elevation_std), -1)
        return self.terrain(torch.where(mask[..., None], feature, torch.zeros_like(feature)), mask)

    def encode_conditions(self, ic, bc):
        iz, iq = ic[:1].chunk(2, -1)
        bq, bz = bc[:1].chunk(2, -1)
        xp = torch.linspace(-1, 1, iz.shape[1], device=iz.device)[None]
        tp = torch.linspace(-1, 1, bq.shape[1], device=bq.device)[None]
        zero = torch.zeros_like(tp)
        positions = torch.cat((
            torch.stack((xp, torch.full_like(xp, -1)), -1),
            torch.stack((torch.full_like(tp, -1), tp), -1),
            torch.stack((torch.full_like(tp, 1), tp), -1)), 1)
        values = torch.cat((
            torch.stack(((iz - self.z_mean) / self.z_std,
                         (iq.clamp_min(1e-6).log() - self.q_log_mean) / self.q_log_std), -1),
            torch.stack((zero, (bq.clamp_min(1e-6).log() - self.q_log_mean) / self.q_log_std), -1),
            torch.stack(((bz - self.z_mean) / self.z_std, zero), -1)), 1)
        return self.bpe(positions)[0], self.bve(values)[0]

    def forward(self, x, t, ic, bc, geo, mask, bed, geo_weight=None, condition_cache=None):
        xn = 2 * (x - self.x_min) / (self.x_max - self.x_min).clamp_min(1e-6) - 1
        tn = 2 * (t - self.t_min) / (self.t_max - self.t_min).clamp_min(1e-6) - 1
        query = self.qpe(torch.cat((xn, tn), -1))
        key, value = condition_cache if condition_cache is not None else self.encode_conditions(ic, bc)
        for unit in self.cross_attention: 
            query = unit(query, key, value)

        gn = torch.stack(((geo[..., 0] - self.station_mean) / self.station_std,
                          (geo[..., 1] - self.elevation_mean) / self.elevation_std
                        ), -1)
        gn = torch.where(
            mask[..., None], gn, torch.zeros_like(gn)
            )
        if geo_weight is None:
            geo_code = self.geo(gn, mask)
            terrain = self.terrain_code(geo, mask)
        else:
            gl, gr = gn.chunk(2, 1)
            ml, mr = mask.chunk(2, 1)
            rl, rr = geo.chunk(2, 1)
            geo_code = (1 - geo_weight) * self.geo(gl, ml) + geo_weight * self.geo(gr, mr)
            terrain = ((1 - geo_weight) * self.terrain_code(rl, ml)
                       + geo_weight * self.terrain_code(rr, mr))

        shared = self.shared(torch.cat((query, geo_code, terrain), -1))
        route = torch.cat((shared, query), -1)

        z = bed + self.depth_mean * F.softplus(self.z_head(route))
        q = torch.exp(self.q_log_mean + self.q_log_std * self.q_head(route))
        return z, q


def load(name):
    return torch.load((SOURCE.parent / config["paths"]["pt"][name]).resolve(),
                      map_location="cpu", weights_only=True)


def prepare(dataset, device, manning):
    keys = ("x", "t", "ic", "bc", "geo", "geo_mask", "bed", "manning_n",
            "upstream_z_bc", "downstream_q_bc")
    result = []
    for case in dataset.values():
        case["manning_n"] = torch.full_like(case["manning_n"], manning)
        case["upstream_z_bc"] = case["z"][:, 0].clone()
        case["downstream_q_bc"] = case["q"][:, -1].clone()
        result.append({key: case[key].to(device) for key in keys})
    return result


def main():
    cfg, run = config["fvm2"], config["attation"]
    seed = cfg["seed"]
    torch.set_num_threads(cfg["num_threads"]); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = sample_cases(load("train"), run["train_cases"], seed)
    val_data = sample_cases(load("validation"), run["val_cases"], seed)

    train = prepare(train_data, device, cfg["manning_n"])
    model = PINN(train[0]["ic"].numel(), make_scales(train)).to(device)

    output = (SOURCE / cfg["output_path"]).resolve().parent / "train_attation" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True)

    manifest = {
        "seed": seed, "epochs": run["epochs"], "device": str(device),
        "train_cases": len(train_data), "val_cases": len(val_data), "test_cases": run["test_cases"],
        "architecture": "paper_qpe_bpe_bve_2cau_2head_keydim64_2dense64",
        "parameters": sum(p.numel() for p in model.parameters()),
        "patience": run["patience"], "min_delta": run["min_delta"],
        "train_keys": list(train_data), "val_keys": list(val_data)
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"output={output}\nparameters={manifest['parameters']}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    names = ("ic_z", "ic_q", "bc_q", "bc_z", "mass", "momentum")
    history, best, convergence_best, stale = [], float("inf"), float("inf"), 0
    stop_reason = "max_epochs"
    for epoch in range(1, run["epochs"] + 1):
        torch.manual_seed(seed + epoch)
        order = torch.randperm(len(train)).tolist()
        lr = cfg["min_learning_rate"] + .5 * (cfg["learning_rate"] - cfg["min_learning_rate"]) * (1 + math.cos(math.pi * (epoch - 1) / run["epochs"]))
        for group in optimizer.param_groups: 
            group["lr"] = lr

        total = torch.zeros(6, dtype=torch.float64, device=device)
        model.train()
        for start in range(0, len(train), cfg["cases_per_batch"]):
            cases = [train[i] for i in order[start:start + cfg["cases_per_batch"]]]
            losses = pinn_losses(model, cases, cfg["points_per_case"])
            optimizer.zero_grad(set_to_none=True)
            balance(model, losses); optimizer.step()
            total += losses.detach().double() * len(cases)
        row = {"epoch": epoch, "lr": lr, **dict(zip(names, (total / len(train)).tolist()))}

        # 5epoch 验证
        if epoch % 5 == 0 or epoch == run["epochs"]:
            z_error, q_error = relative_error(
                model, list(val_data.values()), 
                cfg["val_time_step"], cfg["val_time_batch"]
            )
            row.update(z_error=z_error, q_error=q_error, score=max(z_error, q_error))

            if row["score"] < best:
                best = row["score"]
                torch.save({
                    "model_state_dict": model.state_dict(), 
                    "epoch": epoch, "validation": row
                    }, output / "best.pt")

            if row["score"] < convergence_best - run["min_delta"]: 
                convergence_best, stale = row["score"], 0
            else: 
                stale += 1
            row["stale_checks"] = stale

        history.append(row)
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        torch.save({"model_state_dict": model.state_dict(), 
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch, "history": history, "best": best,
                    "convergence_best": convergence_best, "stale_checks": stale
                    }, output / "last.pt")
        print(row, flush=True)
        if stale >= run["patience"]: 
            stop_reason = "validation_converged"
            break

    # 测试集
    selected = torch.load(output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(selected["model_state_dict"])
    test_data = sample_cases(load("test"), run["test_cases"], seed)
    manifest.update(test_keys=list(test_data), completed_epoch=history[-1]["epoch"], stop_reason=stop_reason)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    test_z, test_q = relative_error(
        model, list(test_data.values()),
        config["monitor"]["final_test_time_step"], cfg["val_time_batch"]
    )
    result = {"checkpoint_epoch": selected["epoch"], 
              "validation": selected["validation"],
              "test_z": test_z, "test_q": test_q, "test_max": max(test_z, test_q)
              }
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(result, flush=True)


if __name__ == "__main__":
    main()
