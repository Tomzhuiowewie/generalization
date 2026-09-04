"""用滚动遗传算法动态调整六项损失权重和学习率"""
import copy, csv, math, random, sys
from datetime import datetime
from pathlib import Path
import torch, yaml

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from loss import boundary_loss, initial_loss, pde_loss
from networks import OperatorPINN

with (SOURCE / "config.yaml").open(encoding="utf-8") as f: 
    config = yaml.safe_load(f)

project_dir = (SOURCE / config["paths"]["project_dir"]).resolve()

name = ("ic_z", "ic_q", "bc_q", "bc_z", "mass", "momentum")

GA = {
        "population": 8, "elite": 2, "interval": 3,
        "weight_sigma": .30, "lr_sigma": .3,
        "smooth": .4, "validation_cases": 4, "validation_points": 128
      }

def initial_gene(losses):
    reference_loss = max(losses[0], 1e-12)
    weights = [reference_loss / max(value, 1e-12) for value in losses]
    return torch.tensor([math.log10(value) for value in weights]
                        + [math.log10(config["training"]["learning_rate"])], dtype=torch.float64)


def normalize_gene(g):
    g = g.clone().cpu()
    g[0] = 0.
    g[1:6].clamp_(-12., 8.)
    g[6].clamp_(-6., -2.5)
    return g


def decode(g):
    g = normalize_gene(g)
    return {n: 10. ** float(v) for n, v in zip(name, g[:6])}, 10. ** float(g[6])


def mutate(g, rng):
    g = g.clone()
    g[1:6] += torch.randn(5, generator=rng, dtype=torch.float64) * GA["weight_sigma"]
    g[6] += torch.randn(1, generator=rng, dtype=torch.float64).item() * GA["lr_sigma"]
    return normalize_gene(g)


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
    total = 0.
    parts = [0.] * 6
    batches = 0
    for start in range(0, len(cases), per_batch):
        batch = [cases[int(i)] for i in order[start:start + per_batch]]
        x = torch.rand(points, 1) * (example["x"][-1] - example["x"][0]) + example["x"][0]
        t = torch.rand(points, 1) * (example["t"][-1] - example["t"][0]) + example["t"][0]
        losses = component_losses(model, batch, x.to(device), t.to(device), device)
        active = 4 if epoch <= config["training"]["pretrain_epochs"] else 6
        loss = sum(weights[name[i]] * losses[i] for i in range(active))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"])
        optimizer.step()

        total += loss.item()
        for i, value in enumerate(losses):
            parts[i] += value.detach().item()
        batches += 1

    return [total / batches] + [value / batches for value in parts]


def relative_error(model, cases, device, time_step):
    model.eval()
    depth_total = q_total = 0.
    with torch.no_grad():
        for case in cases:
            x, bed = case["x"].to(device), case["bed"].to(device)[:, None]
            geo, mask = case["geo"].to(device), case["geo_mask"].to(device)
            indices = torch.arange(0, len(case["t"]), time_step)
            depth_sum = q_sum = count = 0.

            for start in range(0, len(indices), config["monitor"]["time_batch"]):
                selected = indices[start:start + config["monitor"]["time_batch"]]
                nt, nx = len(selected), len(x)
                t = case["t"][selected].to(device)[:, None].expand(-1, nx).reshape(-1, 1)
                mx = x[None].expand(nt, -1).reshape(-1, 1)
                ic = case["ic"].to(device)[None].expand(nt * nx, -1)
                bc = case["bc"].to(device)[None].expand(nt * nx, -1)
                mg = geo[None].expand(nt, -1, -1, -1).reshape(nt * nx, geo.shape[1], 2)
                mm = mask[None].expand(nt, -1, -1).reshape(nt * nx, mask.shape[1])
                mb = bed[None].expand(nt, -1, -1).reshape(-1, 1)

                z, q = model(mx, t, ic, bc, mg, mm, mb)
                true_z = case["z"][selected].to(device).reshape(-1, 1)
                true_q = case["q"][selected].to(device).reshape(-1, 1)

                depth_sum += ((z - true_z).abs() / (true_z - mb).abs().clamp_min(1e-6)).sum().item()
                q_sum += ((q - true_q).abs() / true_q.abs().clamp_min(1e-6)).sum().item()
                count += true_z.numel()

            depth_total += 100 * depth_sum / count
            q_total += 100 * q_sum / count

    return depth_total / len(cases), q_total / len(cases)


def metrics(model, validation, x, t, device):
    depth, q = relative_error(model, validation, device, config["monitor"]["time_step"])
    losses = component_losses(model, validation[:GA["validation_cases"]], x, t, device)
    return [depth, q] + [float(v.detach()) for v in losses]


def fitness(values, reference):
    importance = [.35, .35] + [.05] * 6
    return sum(a * math.log10((v + 1e-12) / (r + 1e-12)) for a, v, r in zip(importance, values, reference))


def new_individual(gene, state, scales, condition_dim, device, optimizer_state=None):
    model = OperatorPINN(condition_dim, scales).to(device); 
    model.load_state_dict(copy.deepcopy(state))
    _, lr = decode(gene)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if optimizer_state:
        optimizer.load_state_dict(copy.deepcopy(optimizer_state))
        for group in optimizer.param_groups: 
            group["lr"] = lr

    return {"model": model, "optimizer": optimizer, "gene": gene, "fitness": math.inf, "metrics": None, "train": None}


def evolve(population, scales, condition_dim, device, rng, py_rng):
    population.sort(key=lambda p: p["fitness"])
    elites = population[:GA["elite"]]
    children = elites[:]
    while len(children) < GA["population"]:
        a, b = py_rng.choice(elites), py_rng.choice(elites)
        alpha = torch.rand(7, generator=rng, dtype=torch.float64)
        proposal = mutate(alpha * a["gene"] + (1 - alpha) * b["gene"], rng)
        gene = normalize_gene((1 - GA["smooth"]) * a["gene"] + GA["smooth"] * proposal)
        children.append(new_individual(gene, a["model"].state_dict(), scales, condition_dim, device, a["optimizer"].state_dict()))
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


def print_epoch(epoch, individual, validation=False):
    train = individual["train"]
    print(f"epoch={epoch:03d}, train={train[0]:.3e}")
    print("  train: " + ", ".join(f"{key}={value:.2e}" for key, value in zip(name, train[1:])))

    if validation:
        weights, lr = decode(individual["gene"])
        val = individual["metrics"]
        val_loss = sum(weights[key] * val[i + 2] for i, key in enumerate(name))
        print(f"  val: loss={val_loss:.3e}, depth={val[0]:.2f}%, q={val[1]:.2f}%, fitness={individual['fitness']:.3e}, lr={lr:.2e}")
        print("  val components: " + ", ".join(f"{key}={value:.2e}" for key, value in zip(name, val[2:])))
        print("  weights: " + ", ".join(f"{key}={weights[key]:.2e}" for key in name))


def main():
    seed = config["training"]["seed"]
    torch.manual_seed(seed); py_rng = random.Random(seed); rng = torch.Generator().manual_seed(seed + 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    load = lambda key: torch.load((project_dir / config["paths"]["pt"][key]).resolve(), map_location="cpu", weights_only=True)
    train_data, validation_data, test_data, scales = load("train"), load("validation"), load("test"), load("normalization")

    cases, validation = list(train_data.values()), list(validation_data.values())
    example = cases[0]
    condition_dim = example["ic"].numel() + example["bc"].numel()
    base = OperatorPINN(condition_dim, scales).to(device)

    base_state = copy.deepcopy(base.state_dict())
    points = GA["validation_points"]

    x = torch.linspace(example["x"][0], example["x"][-1], points, device=device)[:, None]
    t = torch.linspace(example["t"][0], example["t"][-1], points, device=device)[:, None]
    calibration_losses = [float(value.detach()) for value in
                          component_losses(base, cases[:GA["validation_cases"]], x, t, device)]
    g0 = initial_gene(calibration_losses)
    initial_weights, initial_lr = decode(g0)
    print(f"initial losses={dict(zip(name, calibration_losses))}")
    print(f"initial weights={initial_weights}, lr={initial_lr:.3e}")

    reference = metrics(base, validation, x, t, device); population = []
    for i in range(GA["population"]):
        gene = g0 if i == 0 else mutate(g0, rng)
        population.append(new_individual(gene, base_state, scales, condition_dim, device))
    del base; 
    rows = []

    for epoch in range(1, config["training"]["epochs"] + 1):
        for individual in population:
            torch.manual_seed(seed + epoch)
            individual["train"] = train_epoch(individual, cases, example, device, epoch)
        if epoch % GA["interval"] and epoch < config["training"]["epochs"]: 
            print_epoch(epoch, population[0])
            continue

        for individual in population:
            individual["metrics"] = metrics(individual["model"], validation, x, t, device)
            individual["fitness"] = fitness(individual["metrics"], reference)
        population.sort(key=lambda p: p["fitness"])

        for rank, individual in enumerate(population): 
            rows.append(history_row(epoch, rank, individual))

        print_epoch(epoch, population[0], validation=True)
        if epoch + GA["interval"] <= config["training"]["epochs"]: 
            population = evolve(population, scales, condition_dim, device, rng, py_rng)
    best = population[0]; 
    test = relative_error(best["model"], list(test_data.values()), device, config["monitor"]["final_test_time_step"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = project_dir / config["paths"]["figure_dir"]; output.mkdir(exist_ok=True)
    csv_path = output / f"dynamic_ga_history_{stamp}.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)

    weights, lr = decode(best["gene"])

    torch.save({"model_state_dict": best["model"].state_dict(), "gene_log10": best["gene"], "weights": weights,
                "learning_rate": lr, "test_depth_error": test[0], "test_q_error": test[1]}, output / f"dynamic_ga_best_{stamp}.pt")
    print(f"test: depth={test[0]:.2f}%, q={test[1]:.2f}%\nhistory: {csv_path}")


if __name__ == "__main__": 
    main()
