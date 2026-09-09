"""用滚动遗传算法动态调整六项损失权重和学习率"""
import copy, csv, math, random, sys
from datetime import datetime
from pathlib import Path
import torch, yaml

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from loss import boundary_loss, initial_loss, pde_loss
from networks import OperatorPINN
from utils.common import sample_cases, EarlyStopping
from utils.plot import plot_error_contours, plot_loss_history, plot_relative_error_history
from utils.common import relative_error, sample_cases

with (SOURCE / "config.yaml").open(encoding="utf-8") as f: 
    config = yaml.safe_load(f)

project_dir = SOURCE.parent.resolve()

name = ("ic_z", "ic_q", "bc_q", "bc_z", "mass", "momentum")
GA = {
    "population": 4, "elite": 1, "parent_pool": 2, "immigrants": 1, "interval": 5,
    "weight_sigma": .30, "lr_sigma": .3, "smooth": .4, "initWeight_cases": 3, "validation_points": 128
      }


def initial_gene(losses):
    # 初始条件已由输出公式满足；校准边界和 PDE 四项
    weights = [1.0, 1.0] + [
        1.0 / max(value, 1e-12) for value in losses[2:]
    ]
    g = torch.tensor(
        [math.log10(value) for value in weights]
        + [math.log10(config["training"]["learning_rate"])],
        dtype=torch.float64,
    )
    return normalize_gene(g)

# 设置搜索上下限
def normalize_gene(g):
    g = g.clone().cpu()
    g[0] = 0.
    g[1:6].clamp_(-12., 8.)
    g[6].clamp_(-6., -2.5)
    return g

# 还原为实际权重和学习率
def decode(g):
    g = normalize_gene(g)
    return {n: 10. ** float(v) for n, v in zip(name, g[:6])}, 10. ** float(g[6])

# 随机扰动，生成新的权重和学习率
def mutate(g, rng, scale=1.):
    g = g.clone()
    g[1:6] += torch.randn(5, generator=rng, dtype=torch.float64) * GA["weight_sigma"] * scale
    g[6] += torch.randn(1, generator=rng, dtype=torch.float64).item() * GA["lr_sigma"] * scale
    return normalize_gene(g)

# 损失项计算
def component_losses(model, cases, x, t, device):
    values = [torch.zeros((), device=device) for _ in name]
    for case in cases:
        ic_z, ic_q = initial_loss(model, case)
        bc_q, bc_z = boundary_loss(model, case)

        count = len(x)
        mass, momentum = pde_loss(model, x, t, case["ic"].to(device)[None].repeat(count, 1),
            case["bc"].to(device)[None].repeat(count, 1), case["x"].to(device), case["geo"].to(device),
            case["geo_mask"].to(device), case["bed"].to(device), case["manning_n"].to(device).repeat(count, 1))

        for i, v in enumerate((ic_z, ic_q, bc_q, bc_z, mass, momentum)): 
            values[i] += v

    return [v / len(cases) for v in values]


def train_epoch(individual, cases, example, device, epoch):
    model, optimizer = individual["model"], individual["optimizer"]
    weights, lr = decode(individual["gene"])
    for group in optimizer.param_groups: 
        group["lr"] = lr

    model.train()
    per_batch = config["training"]["cases_per_batch"]
    points = config["training"]["points_per_case"]
    order = torch.randperm(len(cases))
    totals = [0.] * 7
    active = 4 if epoch <= config["training"]["pretrain_epochs"] else 6

    for start in range(0, len(cases), per_batch):
        batch = [cases[int(i)] for i in order[start:start + per_batch]]
        x = torch.rand(points, 1) * (example["x"][-1] - example["x"][0]) + example["x"][0]
        t = torch.rand(points, 1) * (example["t"][-1] - example["t"][0]) + example["t"][0]
        losses = component_losses(model, batch, x.to(device), t.to(device), device)
        loss = sum(weights[name[i]] * losses[i] for i in range(active))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"])
        optimizer.step()

        for i, value in enumerate([loss] + losses):
            totals[i] += value.detach().item() * len(batch)

    return [value / len(cases) for value in totals]



def metrics(model, validation, x, t, device):
    depth, q = relative_error(model, validation, config["monitor"]["time_step"], config["monitor"]["time_batch"])
    losses = component_losses(model, validation[:GA["initWeight_cases"]], x, t, device)
    return [depth, q] + [float(v.detach()) for v in losses]


def fitness(values, reference):
    importance = [.35, .35] + [.05] * 6
    return sum(a * math.log10((v + 1e-12) / (r + 1e-12)) for a, v, r in zip(importance, values, reference))


def new_individual(gene, state, scales, condition_dim, device, optimizer_state=None):
    model = OperatorPINN(condition_dim, scales).to(device); 
    model.load_state_dict(state)
    _, lr = decode(gene)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if optimizer_state:
        optimizer.load_state_dict(copy.deepcopy(optimizer_state))
        for group in optimizer.param_groups: 
            group["lr"] = lr

    return {"model": model, "optimizer": optimizer, "gene": gene, "fitness": math.inf, "metrics": None, "train": None}


def evolve(population, scales, condition_dim, device, rng, py_rng):
    population.sort(key=lambda p: p["fitness"])
    parents = population[:GA["parent_pool"]]
    children = population[:GA["elite"]]
    for index in range(len(children), GA["population"]):
        parent = py_rng.choice(parents)
        if index < GA["population"] - GA["immigrants"]:
            proposal = mutate(parent["gene"], rng)
            gene = normalize_gene((1 - GA["smooth"]) * parent["gene"] + GA["smooth"] * proposal)
        else:
            gene = mutate(parent["gene"], rng, scale=2.)
        children.append(new_individual(gene, parent["model"].state_dict(), scales, condition_dim, device, parent["optimizer"].state_dict()))
    return children


def history_row(epoch, rank, individual):
    weights, lr = decode(individual["gene"])
    row = {"epoch": epoch, "rank": rank, "fitness": individual["fitness"], "depth_error": individual["metrics"][0], "q_error": individual["metrics"][1], "train_loss": individual["train"][0]}
    for key, train_value, val_value in zip(weights, individual["train"][1:], individual["metrics"][2:]):
        row[f"{key}_train"] = train_value
        row[f"{key}_val"] = val_value
    for name, value in weights.items():
        row[f"{name}_weight"] = value
        row[f"{name}_log10"] = math.log10(value)
        row[f"{name}_order"] = math.floor(math.log10(value))

    row["learning_rate"] = lr
    row["learning_rate_log10"] = math.log10(lr)
    row["learning_rate_order"] = math.floor(math.log10(lr))
    return row


def main():
    seed = config["training"]["seed"]
    torch.manual_seed(seed); py_rng = random.Random(seed); rng = torch.Generator().manual_seed(seed + 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    load = lambda key: torch.load((project_dir / config["paths"]["pt"][key]).resolve(), map_location="cpu", weights_only=True)
    train_data, validation_data, test_data, scales = load("train"), load("validation"), load("test"), load("normalization")
    # 采样少部分数据用于测试不同方法
    train_data, validation_data, test_data = sample_cases(train_data, 105), sample_cases(validation_data, 30), sample_cases(test_data, 15)

    train_cases, validation = list(train_data.values()), list(validation_data.values())
    example = train_cases[0]
    condition_dim = example["ic"].numel() + example["bc"].numel()

    base = OperatorPINN(condition_dim, scales).to(device)   # 初始化模型
    base_state = copy.deepcopy(base.state_dict())

    xt = torch.rand(GA["validation_points"], 2, generator=torch.Generator().manual_seed(seed + 2)).to(device)
    x = example["x"][0].item() + xt[:, :1] * (example["x"][-1] - example["x"][0]).item()
    t = example["t"][0].item() + xt[:, 1:] * (example["t"][-1] - example["t"][0]).item()

    init_loss = component_losses(base, train_cases[:GA["initWeight_cases"]], x, t, device)  # 计算各个损失项的初始值大小
    g0 = initial_gene([float(value.detach()) for value in init_loss] )   # 生成初始权重

    reference = metrics(base, validation, x, t, device); population = []
    for i in range(GA["population"]):
        gene = g0 if i == 0 else mutate(g0, rng)
        population.append(new_individual(gene, base_state, scales, condition_dim, device))
    del base; 
    rows = []
    history = {key: [] for key in name}
    relative_history = {key: [] for key in ("train_depth", "train_q", "validation_depth", "validation_q")}
    best_fitness = math.inf
    early_stopping = EarlyStopping(patience=5, min_delta=1e-3)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = project_dir / config["paths"]["figure_dir"]; output.mkdir(exist_ok=True)
    best_path = output / f"dynamic_ga_best_{stamp}.pt"

    for epoch in range(1, config["training"]["epochs"] + 1):
        for individual in population:
            torch.manual_seed(seed + epoch)
            individual["train"] = train_epoch(individual, train_cases, example, device, epoch)    # 每个个体训练
        validate = epoch % GA["interval"] == 0 or epoch == config["training"]["epochs"]
        if validate:
            for individual in population:
                individual["metrics"] = metrics(individual["model"], validation, x, t, device)
                individual["fitness"] = fitness(individual["metrics"], reference)
            population.sort(key=lambda p: p["fitness"])
        candidate = population[0]
        if validate and candidate["fitness"] < best_fitness:
            best_fitness = candidate["fitness"]
            weights, lr = decode(candidate["gene"])
            best_checkpoint = {
                "model_state_dict": {key: value.detach().cpu().clone() for key, value in candidate["model"].state_dict().items()},
                "gene_log10": candidate["gene"].clone(), "weights": weights, "learning_rate": lr,
                "epoch": epoch, "fitness": best_fitness,
                "validation_metrics": dict(zip(("depth_error", "q_error") + name, candidate["metrics"])),
            }
            torch.save(best_checkpoint, best_path)

        if validate:
            for rank, individual in enumerate(population):
                rows.append(history_row(epoch, rank, individual))

        for key, value in zip(name, candidate["train"][1:]):
            history[key].append(value)

        train_error = relative_error(candidate["model"], train_cases, config["monitor"]["time_step"], config["monitor"]["time_batch"])
        validation_error = candidate["metrics"][:2] if validate else relative_error(candidate["model"], validation, config["monitor"]["time_step"], config["monitor"]["time_batch"])

        for key, value in zip(relative_history, (*train_error, *validation_error)):
            relative_history[key].append(value)

        train = candidate["train"]
        print(
            f"epoch={epoch:03d}, train={train[0]:.3e}, "
            f"L2: train_depth={train_error[0]:.2f}%, "
            f"train_q={train_error[1]:.2f}%, "
            f"validation_depth={validation_error[0]:.2f}%, "
            f"validation_q={validation_error[1]:.2f}%"
        )
        # print("  train: " + ", ".join(f"{key}={value:.2e}" for key, value in zip(name, train[1:])))

        if validate:
            weights, lr = decode(candidate["gene"])
            val = candidate["metrics"]
            print(f"           fitness={candidate['fitness']:.3e}, lr={lr:.2e}")
            print("           val components: " + ", ".join(f"{key}={value:.2e}" for key, value in zip(name, val[2:])))
            print("           weights: " + ", ".join(f"{key}={weights[key]:.2e}" for key in name))


        if validate and early_stopping.step(candidate["fitness"]):
            print(f"early stop: epoch={epoch:03d}, best_epoch={best_checkpoint['epoch']:03d}, best_fitness={best_fitness:.3e}")
            break
        if validate and epoch + GA["interval"] <= config["training"]["epochs"]:
            population = evolve(population, scales, condition_dim, device, rng, py_rng)

    best_model = population[0]["model"]
    best_model.load_state_dict(best_checkpoint["model_state_dict"])
    test = relative_error(best_model, list(test_data.values()), config["monitor"]["final_test_time_step"], config["monitor"]["time_batch"])
    csv_path = output / f"dynamic_ga_history_{stamp}.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)

    best_checkpoint.update(test_depth_error=test[0], test_q_error=test[1])
    torch.save(best_checkpoint, best_path)
    print(f"best: epoch={best_checkpoint['epoch']:03d}, fitness={best_fitness:.3e}, checkpoint={best_path}")
    print(f"test: depth={test[0]:.2f}%, q={test[1]:.2f}%\nhistory: {csv_path}")

    history_path = plot_loss_history(history, output / f"dynamic_ga_training_loss_history_{stamp}.png")
    relative_history_path = plot_relative_error_history(relative_history, output / f"dynamic_ga_train_validation_relative_error_history_{stamp}.png")
    print(f"loss history saved: {history_path}")
    print(f"relative error history saved: {relative_history_path}")

    for split_name, dataset in (("train", train_data), ("validation", validation_data), ("test", test_data)):
        contour_path = plot_error_contours(best_model, list(dataset.values()), device=device,
            output_path=output / f"dynamic_ga_{split_name}_mean_error_contours_{stamp}.png", levels=config["plot"]["contour_levels"])
        print(f"{split_name} error contour saved: {contour_path}")


if __name__ == "__main__": 
    main()