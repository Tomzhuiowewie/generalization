from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_loss_history(history, output_path):
    """将六个损失项随 epoch 的变化分别绘制在独立子图中。"""
    names = ("ic_z", "ic_q", "bc_q", "bc_z", "mass", "momentum")
    titles = (
        "Initial water-level loss",
        "Initial discharge loss",
        "Upstream discharge loss",
        "Downstream water-level loss",
        "Mass-equation loss",
        "Momentum-equation loss",
    )
    epochs = np.arange(1, len(history["ic_z"]) + 1)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)

    for ax, name, title in zip(axes.flat, names, titles):
        values = np.asarray(history[name], dtype=float)
        ax.semilogy(epochs, np.maximum(values, 1e-16), linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, which="both", linestyle="--", alpha=0.3)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_error_contours(
    model,
    case,
    device=None,
    output_path="error_contours.png",
    levels=8,
):
    """绘制一个工况在每个 (x, t) 点上的水位和流量相对误差等值线。"""
    if device is None:
        device = next(model.parameters()).device

    was_training = model.training
    model.eval()

    x_values = case["x"].to(device)
    t_values = case["t"].to(device)
    true_z = case["z"].to(device)
    true_q = case["q"].to(device)
    section_count = len(x_values)

    ic = case["ic"].to(device)[None].expand(section_count, -1)
    bc = case["bc"].to(device)[None].expand(section_count, -1)
    geo = case["geo"].to(device)
    geo_mask = case["geo_mask"].to(device)
    bed = case["bed"].to(device)[:, None]

    pred_z_rows = []
    pred_q_rows = []

    with torch.no_grad():
        # 每次预测一个时刻的所有断面，避免一次展开整个时空网格。
        for time_value in t_values:
            x = x_values[:, None]
            t = torch.full_like(x, time_value.item())
            pred_z, pred_q = model(x, t, ic, bc, geo, geo_mask, bed)
            pred_z_rows.append(pred_z[:, 0].cpu())
            pred_q_rows.append(pred_q[:, 0].cpu())

    pred_z = torch.stack(pred_z_rows)
    pred_q = torch.stack(pred_q_rows)
    true_z_cpu = true_z.cpu()
    true_q_cpu = true_q.cpu()
    bed_cpu = bed[:, 0].cpu()[None, :]
    true_depth = true_z_cpu - bed_cpu
    pred_depth = pred_z - bed_cpu

    z_error = (
        (pred_depth - true_depth).abs()
        / true_depth.abs().clamp_min(1e-6)
        * 100.0
    ).numpy()
    q_error = (
        (pred_q - true_q_cpu).abs()
        / true_q_cpu.abs().clamp_min(1e-6)
        * 100.0
    ).numpy()
    x_km = x_values.cpu().numpy() / 1000.0
    t_hour = t_values.cpu().numpy() / 3600.0

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    def draw(ax, error, title, colorbar_label, label_format):
        max_error = float(np.nanmax(error))
        contour_levels = np.linspace(0.0, max(max_error, 1e-12), levels + 1)

        filled = ax.contourf(
            x_km,
            t_hour,
            error,
            levels=contour_levels,
            cmap="turbo",
            extend="max",
        )
        lines = ax.contour(
            x_km,
            t_hour,
            error,
            levels=contour_levels[1:],
            colors="black",
            linewidths=0.45,
            alpha=0.7
        )
        ax.clabel(lines, inline=True, fontsize=7, fmt=label_format)
        ax.set_title(title)
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("Time (hour)")
        fig.colorbar(filled, ax=ax, label=colorbar_label)

    draw(
        axes[0],
        z_error,
        "Water-depth relative error",
        "Relative error (%)",
        "%.2f%%",
    )
    draw(
        axes[1],
        q_error,
        "Discharge relative error",
        "Relative error (%)",
        "%.1f%%",
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    if was_training:
        model.train()

    return output_path
