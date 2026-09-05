import random

def sample_cases(dataset, count, seed=2032):
    rng = random.Random(seed)
    groups = {}
    for key in sorted(dataset):
        groups.setdefault(key.split("_")[0], []).append(key)

    for keys in groups.values():
        rng.shuffle(keys)

    selected = []
    while len(selected) < min(count, len(dataset)):
        active = [geo for geo, keys in groups.items() if keys]
        rng.shuffle(active)
        for geo in active:
            selected.append(groups[geo].pop())
            if len(selected) == min(count, len(dataset)):
                break

    return {key: dataset[key] for key in selected}

class EarlyStopping:
    """指标越小越好；连续 patience 次调用无足够改善时返回 True。"""

    def __init__(self, patience=5, min_delta=1e-3):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0

    def step(self, value):
        if value < self.best - self.min_delta:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience
