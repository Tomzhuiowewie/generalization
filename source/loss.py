import torch
from utils.geometry import water_area_at_x


def grad(y, x):

    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y), create_graph=True
        )[0]


def initial_loss(model, case):
    """初始条件：约束 t=t0 时所有断面的水位和流量"""
    device = next(model.parameters()).device    # 获取模型所在的设备
    x = case["x"].to(device)[:, None]   # 额外增加一个维度，变为 (N, 1)
    t = torch.full_like(x, case["t"][0].item())
    count = len(x)

    pred_z, pred_q = model(
        x, t,
        case["ic"].to(device)[None].repeat(count, 1), # [92,184] 184 = 92 * 2(初始水位、初始流量)
        case["bc"].to(device)[None].repeat(count, 1),   # [92,1346] 1346 = 673 * 2(上游流量、下游水位)
        case["geo"].to(device),
        case["geo_mask"].to(device),
        case["bed"].to(device)[:, None],
    )

    true_z = case["z"].to(device)[0, :, None]
    true_q = case["q"].to(device)[0, :, None]

    # 无量纲化/标准化残差
    z_residual = (pred_z - true_z) / model.z_std
    q_residual = (pred_q - true_q) / model.q_std

    return torch.mean(z_residual**2), torch.mean(q_residual**2)


def boundary_loss(model, case):
    """边界条件：约束上游流量和下游水位过程"""
    device = next(model.parameters()).device
    t = case["t"].to(device)[:, None]
    count = len(t)
    ic = case["ic"].to(device)[None].repeat(count, 1)
    bc = case["bc"].to(device)[None].repeat(count, 1)

    x_up = torch.full_like(t, case["x"][0].item())
    _, pred_q_up = model(
        x_up, t, ic, bc,
        case["geo"].to(device)[0:1].repeat(count, 1, 1),
        case["geo_mask"].to(device)[0:1].repeat(count, 1),
        case["bed"].to(device)[0:1].repeat(count, 1),
    )

    x_down = torch.full_like(t, case["x"][-1].item())
    pred_z_down, _ = model(
        x_down, t, ic, bc,
        case["geo"].to(device)[-1:].repeat(count, 1, 1),
        case["geo_mask"].to(device)[-1:].repeat(count, 1),
        case["bed"].to(device)[-1:].repeat(count, 1),
    )

    true_q_up = case["q"].to(device)[:, 0, None]
    true_z_down = case["z"].to(device)[:, -1, None]

    # 无量纲化/标准化残差
    q_residual = (pred_q_up - true_q_up) / model.q_std
    z_residual = (pred_z_down - true_z_down) / model.z_std

    return torch.mean(q_residual**2), torch.mean(z_residual**2)


def pde_loss(
    model, x, t,
    ic, bc, section_x, geometry, geometry_mask, bed,
    manning_n, gravity=9.81, debug=False
):
    x = x.detach().clone().requires_grad_(True)
    t = t.detach().clone().requires_grad_(True)

    right = torch.searchsorted(section_x, x[:, 0]).clamp(1, len(section_x) - 1)
    left = right - 1
    weight = (x[:, 0] - section_x[left]) / (section_x[right] - section_x[left])
    bed_at_x = ((1 - weight) * bed[left] + weight * bed[right]).unsqueeze(-1)
    geo = torch.cat([geometry[left], geometry[right]], dim=1)
    geo_mask = torch.cat([geometry_mask[left], geometry_mask[right]], dim=1)

    water_level, discharge = model(
        x, t, ic, bc, geo, geo_mask, bed_at_x
    )

    area, perimeter = water_area_at_x(
        x, water_level, section_x, geometry, geometry_mask
    )
    area = area.reshape_as(water_level).clamp_min(1e-6)
    perimeter = perimeter.reshape_as(water_level).clamp_min(1e-6)  # 避免湿周为零
    radius = (area / perimeter).clamp_min(1e-6)  # 避免水力半径为零

    # Mass equation: ∂A/∂t + ∂Q/∂x = 0
    area_t = grad(area, t)  # 
    area_t_nd = (model.time_ref / model.area_ref) * area_t

    discharge_x = grad(discharge, x)
    discharge_x_nd = (model.length_ref / model.q_ref) * discharge_x

    mass = area_t + discharge_x
    mass_nd = area_t_nd + discharge_x_nd

    loss_mass = mass.square().mean()
    loss_mass_nd = mass_nd.square().mean()

    # Momentum equation: ∂Q/∂t + ∂(Q^2/A)/∂x + gA∂z/∂x + gA(Sf) = 0
    momentum_scale = model.area_ref * model.length_ref / model.q_ref.square()

    discharge_t = grad(discharge, t)
    discharge_t_nd = (model.time_ref / model.q_ref) * discharge_t

    flux = discharge**2 / area
    flux_x = grad(flux, x)
    flux_x_nd = momentum_scale * flux_x

    water_level_x = grad(water_level, x)
    pressure_term = gravity * area * water_level_x
    pressure_term_nd = momentum_scale * pressure_term

    friction = (
        manning_n**2 * discharge * discharge.abs()
        / (area**2 * radius.pow(4.0 / 3.0))
    )

    friction_term = gravity * area * friction
    friction_term_nd = momentum_scale * friction_term

    momentum = discharge_t + flux_x + pressure_term + friction_term
    momentum_nd = discharge_t_nd + flux_x_nd + pressure_term_nd + friction_term_nd

    loss_momentum = momentum.square().mean()
    loss_momentum_nd = momentum_nd.square().mean()


    if debug:
        diagnostic_eps = 1e-12

        mass_scale_local = (
            area_t_nd.abs()
            + discharge_x_nd.abs()
        )

        mass_relative = (
            mass_nd.abs()
            / (mass_scale_local + diagnostic_eps)
        )

        momentum_scale_local = (
            discharge_t_nd.abs()
            + flux_x_nd.abs()
            + pressure_term_nd.abs()
            + friction_term_nd.abs()
        )

        momentum_relative = (
            momentum_nd.abs()
            / (momentum_scale_local + diagnostic_eps)
        )

        def stats(name, tensor):
            v = tensor.detach()

            print(
                f"{name:22s}"
                f" mean_abs={v.abs().mean().item():.4e}"
                f"  rms={v.square().mean().sqrt().item():.4e}"
                f"  max_abs={v.abs().max().item():.4e}"
                f"  mean={v.mean().item():.4e}"
            )

        print("\n========== PDE DIAGNOSTICS ==========")

        print("\n[MASS ND]")
        stats("area_t_nd", area_t_nd)
        stats("discharge_x_nd", discharge_x_nd)
        stats("mass_nd", mass_nd)

        print("\n[MASS RELATIVE]")
        stats("mass_relative", mass_relative)

        print("\n[MOMENTUM ND]")
        stats("discharge_t_nd", discharge_t_nd)
        stats("flux_x_nd", flux_x_nd)
        stats("pressure_term_nd", pressure_term_nd)
        stats("friction_term_nd", friction_term_nd)
        stats("momentum_nd", momentum_nd)

        print("\n[MOMENTUM RELATIVE]")
        stats(
            "momentum_relative",
            momentum_relative,
        )

        print("\n[STATE]")
        stats("water_level", water_level)
        stats("discharge", discharge)
        stats("area", area)
        stats("perimeter", perimeter)
        stats("radius", radius)

        print("\n[RANGE]")
        print(
            f"Q: "
            f"min={discharge.detach().min().item():.4f}, "
            f"max={discharge.detach().max().item():.4f}, "
            f"range="
            f"{(discharge.detach().max() - discharge.detach().min()).item():.4f}"
        )

        print(
            f"Z: "
            f"min={water_level.detach().min().item():.4f}, "
            f"max={water_level.detach().max().item():.4f}"
        )

        print("\n[LOSS]")
        print(
            f"mass_loss     = {loss_mass_nd.item():.4e}"
        )
        print(
            f"momentum_loss = {loss_momentum_nd.item():.4e}"
        )


    return loss_mass_nd, loss_momentum_nd