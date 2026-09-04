from datetime import datetime
from pathlib import Path

import torch, math, yaml

from loss import boundary_loss, initial_loss, pde_loss
from utils.plot import plot_error_contours, plot_loss_history
from networks import OperatorPINN

config_path = Path(__file__).with_name("config.yaml")
with config_path.open("r", encoding="utf-8") as file:
    config = yaml.safe_load(file)
project_dir = (config_path.parent / config["paths"]["project_dir"]).resolve()

def configured_path(*keys):
    value = config["paths"]
    for key in keys:
        value = value[key]
    return (project_dir / value).resolve()

def relative_error(model, case, device, time_step=24, time_batch=24):
    was_training = model.training
    model.eval()
    x = case["x"].to(device)
    section_count = len(x)
    geo = case["geo"].to(device)
    geo_mask = case["geo_mask"].to(device)
    bed = case["bed"].to(device)[:, None]
    indices = torch.arange(0, len(case["t"]), time_step)
    depth_error_sum = q_error_sum = 0.0
    point_count = 0

    with torch.no_grad():
        for start in range(0, len(indices), time_batch):
            selected = indices[start:start + time_batch]
            time_count = len(selected)
            t = case["t"][selected].to(device)[:, None].expand(-1, section_count).reshape(-1, 1)
            model_x = x[None].expand(time_count, -1).reshape(-1, 1)
            ic = case["ic"].to(device)[None].expand(time_count * section_count, -1)
            bc = case["bc"].to(device)[None].expand(time_count * section_count, -1)
            model_geo = geo[None].expand(time_count, -1, -1, -1).reshape(time_count * section_count, geo.shape[1], 2)
            model_geo_mask = geo_mask[None].expand(time_count, -1, -1).reshape(time_count * section_count, geo_mask.shape[1])
            model_bed = bed[None].expand(time_count, -1, -1).reshape(-1, 1)
            pred_z, pred_q = model(model_x, t, ic, bc, model_geo, model_geo_mask, model_bed)
            true_z = case["z"][selected].to(device).reshape(-1, 1)
            true_q = case["q"][selected].to(device).reshape(-1, 1)
            true_depth = true_z - model_bed
            depth_error_sum += ((pred_z - true_z).abs() / true_depth.abs().clamp_min(1e-6)).sum().item()
            q_error_sum += ((pred_q - true_q).abs() / true_q.abs().clamp_min(1e-6)).sum().item()
            point_count += true_z.numel()

    if was_training:
        model.train()
    return 100 * depth_error_sum / point_count, 100 * q_error_sum / point_count


def dataset_relative_error(model, cases, device, time_step=24, time_batch=24):
    errors = [relative_error(model, case, device, time_step, time_batch) for case in cases]
    return sum(error[0] for error in errors) / len(errors), sum(error[1] for error in errors) / len(errors)


def train():
    torch.manual_seed(config["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device)

    # train_data = prepare_case("train")  # 训练集数据
    train_data = torch.load(configured_path("pt", "train"), map_location="cpu", weights_only=True)
    # normalization_scales = load_normalization_scales(train_data)   # 加载或计算归一化尺度
    normalization_scales = torch.load(configured_path("pt", "normalization"), map_location="cpu", weights_only=True)

    example_case = list(train_data.values())[0] #  确定数据结构的示例工况
    condition_dim = example_case["ic"].numel() + example_case["bc"].numel() # 计算(初始条件+边界条件)的维度

    model = OperatorPINN(condition_dim, normalization_scales).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    cases = list(train_data.values())
    case_count = len(train_data)

    cases_per_batch, points_per_case  = 8, 512     # 每个 batch 选择几个工况, 每个工况采样多少个 PDE 点
    batch_count = math.ceil(case_count / cases_per_batch)   # 计算每个 epoch 的 batch 数量（方法向上取整）
    cases_per_batch, points_per_case = config["training"]["cases_per_batch"], config["training"]["points_per_case"]
    batch_count = math.ceil(case_count / cases_per_batch)


    history = {name: [] for name in ("ic_z", "ic_q", "bc_q", "bc_z", "mass", "momentum")}

    validation_data = torch.load(configured_path("pt", "validation"), map_location="cpu", weights_only=True)
    test_data = torch.load(configured_path("pt", "test"), map_location="cpu", weights_only=True)

    validation_cases = list(validation_data.values())
    monitor_case_count = config["monitor"]["case_count"]
    train_monitor_cases = [cases[index] for index in torch.linspace(0, len(cases) - 1, monitor_case_count).long()]
    validation_monitor_cases = [validation_cases[index] for index in torch.linspace(0, len(validation_cases) - 1, monitor_case_count).long()]
    relative_history = {"train_depth": [], "train_q": [], "validation_depth": [], "validation_q": []}

    for epoch in range(1, config["training"]["epochs"] + 1):

        epoch_loss = 0.0
        epoch_components = {name: 0.0 for name in history}
        case_reordered = torch.randperm(case_count)
        for start in range(0, case_count, cases_per_batch):
            selected_indices = case_reordered[start:start + cases_per_batch]
            selected_cases = [cases[index] for index in selected_indices]

            # PDE 采样点：在每个工况的空间范围内随机采样
            x_range = example_case["x"][-1] - example_case["x"][0]
            t_range = example_case["t"][-1] - example_case["t"][0]
            pde_x = torch.rand(points_per_case, 1) * x_range + example_case["x"][0]
            pde_t = torch.rand(points_per_case, 1) * t_range + example_case["t"][0]
            pde_x, pde_t = pde_x.to(device), pde_t.to(device)

            ic_z = ic_q = bc_q = bc_z = mass = momentum = 0.0 # 损失
            for case in selected_cases:

                # 初始条件和边界条件损失
                case_ic_z, case_ic_q = initial_loss(model, case)
                case_bc_q, case_bc_z = boundary_loss(model, case)
                ic_z += case_ic_z
                ic_q += case_ic_q
                bc_q += case_bc_q
                bc_z += case_bc_z

                # PDE 损失
                case_mass, case_momentum = pde_loss(
                    model, pde_x, pde_t,
                    case["ic"].to(device)[None].repeat(points_per_case, 1),
                    case["bc"].to(device)[None].repeat(points_per_case, 1),
                    case["x"].to(device),
                    case["geo"].to(device),
                    case["geo_mask"].to(device),
                    case["bed"].to(device),
                    case["manning_n"].to(device).repeat(points_per_case, 1),
                    debug=False,
                )
                mass += case_mass
                momentum += case_momentum

            selected_count = len(selected_cases)
            ic_z, ic_q = ic_z / selected_count, ic_q / selected_count
            bc_q, bc_z = bc_q / selected_count, bc_z / selected_count
            mass, momentum = mass / selected_count, momentum / selected_count

            if epoch <= config["training"]["pretrain_epochs"]:
                loss = ic_z + ic_q + bc_q + bc_z # 前 3个 epoch：只训练初始条件和边界条件
            else:
                loss = (
                    ic_z + ic_q + bc_q + bc_z
                    + config["loss_weights"]["mass"] * mass + config["loss_weights"]["momentum"] * momentum
                )

            optimizer.zero_grad()   # 每个 batch 训练前清零梯度，更新一次网络
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"])
            optimizer.step()
            epoch_loss += loss.item()   # 累计每个 batch 的损失，最后计算平均值

            epoch_components["ic_z"] += ic_z.item()
            epoch_components["ic_q"] += ic_q.item()
            epoch_components["bc_q"] += bc_q.item()
            epoch_components["bc_z"] += bc_z.item()
            epoch_components["mass"] += config["loss_weights"]["mass"] * mass.item()
            epoch_components["momentum"] += config["loss_weights"]["momentum"] * momentum.item()

        for name in history:
            history[name].append(epoch_components[name] / batch_count)

        print(
            f"epoch={epoch:03d}, loss={epoch_loss / batch_count:.6e}, "
            f"ic_z={history['ic_z'][-1]:.4e}, "
            f"ic_q={history['ic_q'][-1]:.4e}, "
            f"bc_q={history['bc_q'][-1]:.4e}, "
            f"bc_z={history['bc_z'][-1]:.4e}, "
            f"mass={history['mass'][-1]:.4e}, "
            f"momentum={history['momentum'][-1]:.4e}"
        )

        train_depth_error, train_q_error = dataset_relative_error(model, train_monitor_cases, device, config["monitor"]["time_step"], config["monitor"]["time_batch"])
        validation_depth_error, validation_q_error = dataset_relative_error(model, validation_monitor_cases, device, config["monitor"]["time_step"], config["monitor"]["time_batch"])
        relative_history["train_depth"].append(train_depth_error)
        relative_history["train_q"].append(train_q_error)
        relative_history["validation_depth"].append(validation_depth_error)
        relative_history["validation_q"].append(validation_q_error)
        print(f"relative error: train_depth={train_depth_error:.2f}%, train_q={train_q_error:.2f}%, validation_depth={validation_depth_error:.2f}%, validation_q={validation_q_error:.2f}%")

    test_depth_error, test_q_error = dataset_relative_error(model, list(test_data.values()), device, config["monitor"]["final_test_time_step"], config["monitor"]["time_batch"])
    print(f"final test relative error: depth={test_depth_error:.2f}%, q={test_q_error:.2f}%")

    # 误差等值线
    case_id, test_case = next(iter(test_data.items()))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_path = plot_error_contours(
        model,
        test_case,
        device=device,
        output_path=configured_path("figure_dir") / f"{case_id}_error_contours_{timestamp}.png",
        levels=config["plot"]["contour_levels"],
    )

    history_path = plot_loss_history(
        history,
        output_path=configured_path("figure_dir") / f"{case_id}_loss_history_{timestamp}.png",
    )
    print(f"loss history saved: {history_path}\nerror contour saved: {output_path}")


if __name__ == "__main__":
    train()
