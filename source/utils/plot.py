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


def plot_relative_error_history(history, output_path):
    """绘制训练集和验证集平均相对误差随 epoch 的变化。"""
    names = ("train_depth", "train_q", "validation_depth", "validation_q")
    titles = ("Training water-depth error", "Training discharge error", "Validation water-depth error", "Validation discharge error")
    epochs = np.arange(1, len(history["train_depth"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    for ax, name, title in zip(axes.flat, names, titles):
        ax.plot(epochs, np.asarray(history[name], dtype=float), linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean relative error (%)")
        ax.grid(True, linestyle="--", alpha=0.3)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_error_contours(
    model,
    cases,
    device=None,
    output_path="error_contours.png",
    levels=8,
):
    """绘制所有工况在每个 (x, t) 点上的平均相对误差等值线。"""
    if not cases:
        raise ValueError("cases cannot be empty")
    if device is None:
        device = next(model.parameters()).device

    was_training = model.training
    model.eval()

    reference_case = cases[0]
    x_values = reference_case["x"].to(device)
    t_values = reference_case["t"].to(device)
    section_count = len(x_values)
    z_error_sum = torch.zeros((len(t_values), section_count))
    q_error_sum = torch.zeros_like(z_error_sum)

    for case in cases:
        if not torch.equal(case["x"], reference_case["x"]) or not torch.equal(case["t"], reference_case["t"]):
            raise ValueError("All cases must use the same x and t grid")

        true_z = case["z"].to(device)
        true_q = case["q"].to(device)
        ic = case["ic"].to(device)[None].expand(section_count, -1)
        bc = case["bc"].to(device)[None].expand(section_count, -1)
        geo = case["geo"].to(device)
        geo_mask = case["geo_mask"].to(device)
        bed = case["bed"].to(device)[:, None]
        pred_z_rows = []
        pred_q_rows = []

        with torch.no_grad():
            for time_value in t_values:
                x = x_values[:, None]
                t = torch.full_like(x, time_value.item())
                pred_z, pred_q = model(x, t, ic, bc, geo, geo_mask, bed)
                pred_z_rows.append(pred_z[:, 0].cpu())
                pred_q_rows.append(pred_q[:, 0].cpu())

        pred_z = torch.stack(pred_z_rows)
        pred_q = torch.stack(pred_q_rows)
        bed_cpu = bed[:, 0].cpu()[None, :]
        true_depth = true_z.cpu() - bed_cpu
        pred_depth = pred_z - bed_cpu
        z_error_sum += (pred_depth - true_depth).abs() / true_depth.abs().clamp_min(1e-6) * 100.0
        q_error_sum += (pred_q - true_q.cpu()).abs() / true_q.cpu().abs().clamp_min(1e-6) * 100.0

    z_error = (z_error_sum / len(cases)).numpy()
    q_error = (q_error_sum / len(cases)).numpy()
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
        "Mean water-depth relative error across test cases",
        "Relative error (%)",
        "%.2f%%",
    )
    draw(
        axes[1],
        q_error,
        "Mean discharge relative error across test cases",
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
