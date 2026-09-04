from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from utils.geometry import water_area
import yaml

config_path = Path(__file__).with_name("config.yaml")
with config_path.open("r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

project_dir = (config_path.parent / config["paths"]["project_dir"]).resolve()
ras_data_dir = (project_dir / config["paths"]["csv"]["ras_dir"]).resolve()
geo_data_dir = (project_dir / config["paths"]["csv"]["geo_dir"]).resolve()
max_geo_points = config["data"]["max_geo_points"]


def calculate_normalization_scales(dataset):
    """基于训练集计算归一化参数"""
    all_z = torch.cat([case["z"].reshape(-1) for case in dataset.values()]) # 水位
    all_q = torch.cat([case["q"].reshape(-1) for case in dataset.values()])   # 流量
    all_valid_geo = torch.cat([case["geo"][case["geo_mask"]] for case in dataset.values()], dim=0)   # 有效的河道剖面点
    all_x = torch.cat([case["x"].reshape(-1) for case in dataset.values()]) # 河道的纵向位置
    all_t = torch.cat([case["t"].reshape(-1) for case in dataset.values()]) # 模拟时间
    all_depth = torch.cat([(case["z"] - case["bed"][None, :]).reshape(-1) for case in dataset.values()])    # 最大水深 = 水位 - 河床最低高程

    z_mean = all_z.mean()   # 计算水位的均值
    z_std = all_z.std(unbiased=False).clamp_min(1e-6)   # 计算水位的标准差，避免为零

    q_mean = all_q.mean()   # 计算流量的均值
    q_std = all_q.std(unbiased=False).clamp_min(1e-6)   # 计算流量的标准差，避免为零

    station_mean = all_valid_geo[:, 0].mean()   # 计算河道剖面点的均值
    station_std = all_valid_geo[:, 0].std(unbiased=False).clamp_min(1e-6)   # 计算河道剖面点的标准差，避免为零

    elevation_mean = all_valid_geo[:, 1].mean() # 计算河道剖面高程的均值
    elevation_std = all_valid_geo[:, 1].std(unbiased=False).clamp_min(1e-6) # 计算河道剖面高程的标准差，避免为零

    depth_mean = all_depth.mean().clamp_min(1e-6)   # 计算水深的均值

    area_samples = []
    with torch.no_grad():
        for case in dataset.values():
            for water_level in case["z"]:
                area, _ = water_area(water_level, case["geo"], case["geo_mask"])
                valid_area = area[area > 0]
                if valid_area.numel() > 0:
                    area_samples.append(valid_area)

    all_area = torch.cat(area_samples)

    depth_ref = all_depth.median().clamp_min(1e-6)  # 计算水深的中位数
    area_ref = all_area.median().clamp_min(1e-6)    # 计算过水面积的中位数
    length_ref = (all_x.max() - all_x.min()).clamp_min(1e-6)    # 计算河道长度
    
    gravity_ref = all_depth.new_tensor(9.81)
    # 浅水重力波速度：重力加速度 * 水深
    velocity_ref = torch.sqrt(gravity_ref * depth_ref).clamp_min(1e-6)
    q_ref = area_ref * velocity_ref     # 计算流量的参考值 = 过水面积 * 浅水重力波速度
    time_ref = length_ref / velocity_ref    # 计算时间的参考值 = 河道长度 / 浅水重力波速度

    return {
        "x_min": all_x.min(),
        "x_max": all_x.max(),
        "t_min": all_t.min(),
        "t_max": all_t.max(),
        "z_mean": z_mean,
        "z_std": z_std,
        "q_mean": q_mean,
        "q_std": q_std,
        "station_mean": station_mean,
        "station_std": station_std,
        "elevation_mean": elevation_mean,
        "elevation_std": elevation_std,
        "depth_mean": depth_mean,

        # PDE无量纲尺度
        "depth_ref": depth_ref,
        "area_ref": area_ref,
        "length_ref": length_ref,
        "velocity_ref": velocity_ref,
        "q_ref": q_ref,
        "time_ref": time_ref,
    }


def prepare_case(split_name, manning_n=0.016):
    """直接读取原始 CSV，构造训练数据（PT）。"""
    prepared_data = {}
    geometry_cache = {}
    ras_files = sorted(ras_data_dir.glob(f"{split_name}_*_hydrodynamics.csv"))

    if not ras_files:
        raise FileNotFoundError(f"No {split_name} hydrodynamics CSV files under {ras_data_dir}")

    for file_index, file in enumerate(ras_files,start=1):
        data = pd.read_csv(file, encoding="utf-8-sig")
        case_id_array = data["combination_id"].to_numpy()
        time_text = data["time"].to_numpy()
        river_station = data["river_station"].to_numpy()
        water_surface = data["water_surface_m"].to_numpy()
        flow = data["flow_m3s"].to_numpy()

        case_ids = np.unique(case_id_array)
        case_id = str(case_ids[0])

        # 时间解析
        time = pd.to_datetime(time_text, format="%d%b%Y %H%M")

        # 去掉前三天，只保留整点
        start = (time.min() + pd.Timedelta(days=3))

        keep = ((time >= start) & (time.minute == 0))

        time = time[keep]
        river_station = river_station[keep]
        water_surface = water_surface[keep]
        flow = flow[keep]
        time_s = np.asarray((time - start).total_seconds(), dtype="float64")

        # 为每一行建立时间索引
        times, time_index = np.unique(time_s, return_inverse=True)

        # 断面从大到小排列
        sections_ascending, section_index = (np.unique(river_station, return_inverse=True))
        sections = sections_ascending[::-1]
        section_index = (len(sections_ascending) - 1 - section_index)
        time_count, section_count = len(times), len(sections)

        # 直接构造 [时间, 断面] 矩阵
        z = np.full((time_count, section_count), np.nan, dtype="float32")
        q = np.full((time_count, section_count), np.nan, dtype="float32")

        z[time_index, section_index] = water_surface
        q[time_index, section_index] = flow

        # 与原来的计算保持一致
        x = ((sections[0] - sections) * 1000).astype("float32")
        t = times.astype("float32")
        ic = np.concatenate([z[0], q[0]]).astype("float32")
        bc = np.concatenate([q[:, 0], z[:, -1]]).astype("float32")

        # 获取当前工况的地形编号，如 G000
        geometry_id = case_id.split("_", maxsplit=1)[0]
        if geometry_id not in geometry_cache:    # 相同地形只读取一次
            geo_files = list(geo_data_dir.glob(f"{geometry_id}_cross_section_geometry.csv"))
            if not geo_files:
                raise FileNotFoundError(f"No geometry CSV for {geometry_id} under {geo_data_dir}")
            geo_data = pd.read_csv(geo_files[0], encoding="utf-8-sig")
            geometry_cache[geometry_id] = {
                "river_station": geo_data["river_station"].to_numpy(),
                "point_index": geo_data["point_index"].to_numpy(),
                "station_m": geo_data["station_m"].to_numpy(),
                "elevation_m": geo_data["elevation_m"].to_numpy(),
            }

        geometry = geometry_cache[geometry_id]
        geo = np.zeros((section_count, max_geo_points, 2), dtype="float32")
        geo_mask = np.zeros((section_count, max_geo_points), dtype=bool)

        for section_position, section in enumerate(sections):
            selected = np.isclose(geometry["river_station"],section)

            # 按原始 point_index 排序
            order = np.argsort(geometry["point_index"][selected])
            profile = np.column_stack([
                geometry["station_m"][selected][order],
                geometry["elevation_m"][selected][order],
            ]).astype("float32")

            point_count = len(profile)
            geo[section_position, :point_count] = profile
            geo_mask[section_position, :point_count] = True

        bed = np.where(geo_mask, geo[..., 1], np.inf).min(axis=1).astype("float32")

        prepared_data[case_id] = {
            "x": torch.from_numpy(x),
            "t": torch.from_numpy(t),
            "z": torch.from_numpy(z),
            "q": torch.from_numpy(q),
            "ic": torch.from_numpy(ic),
            "bc": torch.from_numpy(bc),
            "geo": torch.from_numpy(geo),
            "geo_mask": torch.from_numpy(geo_mask),
            "bed": torch.from_numpy(bed),
            "manning_n": torch.tensor(manning_n, dtype=torch.float32),
        }
        print(
            f"[{file_index}/{len(ras_files)}] "
            f"loaded: {case_id}",
            flush=True,
        )

    return prepared_data


if __name__ == "__main__":
    prepared_datasets = {}

    # 重新生成训练集、验证集和测试集
    for split in ("train", "validation", "test"):
        print(f"preparing: {split}")
        dataset = prepare_case(split)

        prepared_datasets[split] = dataset
        output = (project_dir / config["paths"]["pt"][split]).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dataset, output)
        print(f"saved: {output}, cases={len(dataset)}")

    # 使用刚生成的训练集计算归一化参数
    normalization_scales = calculate_normalization_scales(prepared_datasets["train"])
    normalization_output = (project_dir / config["paths"]["pt"]["normalization"]).resolve()
    normalization_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(normalization_scales, normalization_output)

    print("saved:", normalization_output)