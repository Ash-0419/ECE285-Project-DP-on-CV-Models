#!/usr/bin/env python3
"""
DINOv2 on Fitzpatrick17k

Predicts the 3-way label (benign / malignant / non-neoplastic) and reports balanced
accuracy overall and per skin-type group (I-II, III-IV, V-VI).

"""
import argparse
import json
import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from transformers import Dinov2Model

SKIN_GROUP = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
GROUP_NAMES = ["I-II", "III-IV", "V-VI"]
LABELS = {"benign": 0, "malignant": 1, "non-neoplastic": 2}
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

DEFAULTS = {  # per-method defaults; any of them can be overridden on the command line
    "linear": dict(epochs=50, lr=1e-3, batch_size=1024, physical_batch=1024, clip=1.0),
    "lora": dict(epochs=10, lr=5e-4, batch_size=256, physical_batch=32, clip=1.0),
}


#data
def load_metadata(csv_path, img_dir, ext):
    df = pd.read_csv(csv_path)
    df = df[df["fitzpatrick_scale"].isin(list(SKIN_GROUP))].copy()  # drops -1 = unknown type
    df["path"] = [os.path.join(img_dir, f"{h}.{ext}") for h in df["md5hash"]]
    exists = df["path"].map(os.path.exists)
    if (~exists).any():
        print(f"warning: {(~exists).sum()} images in the CSV are missing on disk; skipping them")
    df = df[exists].copy()
    df["group"] = df["fitzpatrick_scale"].map(SKIN_GROUP)
    df["y"] = df["three_partition_label"].map(LABELS)
    assert df["y"].notna().all(), "unexpected values in three_partition_label"
    return df.reset_index(drop=True)


def make_strata(sub, min_count):
    """Stratification key for `sub`: group x condition where that cell has >= min_count
    images, else group x 3-way label, else group alone (sklearn needs >= 2 per class)."""
    keys = [
        sub["group"].astype(str) + "|" + sub["label"],
        sub["group"].astype(str) + "|" + sub["three_partition_label"],
        sub["group"].astype(str),
    ]
    strata = keys[0]
    for fallback in keys[1:]:
        strata = strata.where(strata.map(strata.value_counts()) >= min_count, fallback)
    return strata


def split(df, seed):
    """70/15/15 stratified by skin-type group x condition (114 labels), with rare cells
    merged into coarser strata. Strata are recomputed for the val/test split because
    a cell that was big enough in the full data can have only 1 image in the 30% holdout."""
    tr, rest = train_test_split(df.index, test_size=0.30,
                                stratify=make_strata(df, 10), random_state=seed)
    rest_df = df.loc[rest]
    va, te = train_test_split(rest, test_size=0.50,
                              stratify=make_strata(rest_df, 4), random_state=seed)
    return df.loc[tr], df.loc[va], df.loc[te]


class Images(Dataset):
    def __init__(self, df, tf):
        self.paths, self.y, self.g = df["path"].tolist(), df["y"].tolist(), df["group"].tolist()
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB")), self.y[i], self.g[i]


TRAIN_TF = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
EVAL_TF = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),  # 224 = 16 x 16 patches of 14 px
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# models
def pool(h):
    return torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=-1)


@torch.no_grad()
def extract_features(backbone, df, device, workers):
    backbone.eval()
    loader = DataLoader(Images(df, EVAL_TF), batch_size=128, num_workers=workers)
    feats, ys, gs = [], [], []
    for x, y, g in loader:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            h = backbone(pixel_values=x.to(device)).last_hidden_state
        feats.append(pool(h).float().cpu())
        ys.append(y)
        gs.append(g)
    return torch.cat(feats), torch.cat(ys), torch.cat(gs)


class LoRAClassifier(nn.Module):
    def __init__(self, backbone, n_classes, r):
        from peft import LoraConfig, inject_adapter_in_model
        super().__init__()
        dim = backbone.config.hidden_size
        cfg = LoraConfig(r=r, lora_alpha=r, target_modules=["query", "value"], lora_dropout=0.0, bias="none")
        self.backbone = inject_adapter_in_model(cfg, backbone)
        for name, p in self.backbone.named_parameters(): 
            p.requires_grad = "lora_" in name
        self.head = nn.Linear(2 * dim, n_classes)

    def forward(self, x):
        return self.head(pool(self.backbone(pixel_values=x).last_hidden_state))


# training
def train(model, train_ds, args, device):
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable parameters: {sum(p.numel() for p in params):,}")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    private = math.isfinite(args.epsilon)
    engine = None
    if private:
        from opacus import PrivacyEngine
        engine = PrivacyEngine(accountant="prv")
        model, opt, loader = engine.make_private_with_epsilon(
            module=model, optimizer=opt, data_loader=loader,  # switches to Poisson sampling
            target_epsilon=args.epsilon, target_delta=args.delta,
            epochs=args.epochs, max_grad_norm=args.clip,
        )
        print(f"DP-SGD: noise multiplier sigma = {opt.noise_multiplier:.3f}, clip C = {args.clip}")

    loss_fn = nn.CrossEntropyLoss()
    for ep in range(args.epochs):
        model.train()
        total, n = 0.0, 0
        if private:
            from opacus.utils.batch_memory_manager import BatchMemoryManager
            # splits each logical (Poisson) batch into physical chunks; step() only fires on the last one
            with BatchMemoryManager(data_loader=loader, max_physical_batch_size=args.physical_batch,
                                    optimizer=opt) as mem_loader:
                for batch in mem_loader:
                    x, y = batch[0].to(device), batch[1].to(device)
                    if len(y) == 0:
                        continue
                    opt.zero_grad(set_to_none=True)
                    loss = loss_fn(model(x), y)
                    loss.backward()
                    opt.step()
                    total, n = total + loss.item() * len(y), n + len(y)
        else:
            for batch in loader:  # gradient accumulation so the logical batch matches the DP runs
                opt.zero_grad(set_to_none=True)
                X, Y = batch[0], batch[1]
                for x, y in zip(X.split(args.physical_batch), Y.split(args.physical_batch)):
                    x, y = x.to(device), y.to(device)
                    loss = loss_fn(model(x), y)
                    (loss * len(y) / len(Y)).backward()
                    total, n = total + loss.item() * len(y), n + len(y)
                opt.step()
        msg = f"epoch {ep + 1}/{args.epochs}  train loss {total / max(n, 1):.4f}"
        if private:
            msg += f"  eps spent {engine.get_epsilon(args.delta):.2f}"
        print(msg)
    return model, (opt.noise_multiplier if private else 0.0)


# evaluation
def group_metrics(pred, y, g):
    out = {"overall": balanced_accuracy_score(y, pred)}
    for k, name in enumerate(GROUP_NAMES):
        m = g == k
        out[name] = balanced_accuracy_score(y[m], pred[m]) if m.any() else float("nan")
        out[f"n_{name}"] = int(m.sum())
    return out


@torch.no_grad()
def predict_images(model, df, device, workers):
    model.eval()
    preds = []
    for x, _, _ in DataLoader(Images(df, EVAL_TF), batch_size=64, num_workers=workers):
        preds.append(model(x.to(device)).argmax(1).cpu())
    return torch.cat(preds).numpy()


# main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="fitzpatrick17k.csv")
    ap.add_argument("--img-dir", required=True, help="folder with <md5hash>.<ext> images")
    ap.add_argument("--ext", default="jpg")
    ap.add_argument("--model", default="facebook/dinov2-small",
                    help="dinov2-small / -base / -large (or *-with-registers variants)")
    ap.add_argument("--method", choices=["linear", "lora"], required=True)
    ap.add_argument("--epsilon", type=float, default=float("inf"), help="'inf' = non-private")
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--lora-r", type=int, default=8)
    for k in ["epochs", "batch_size", "physical_batch"]:
        ap.add_argument(f"--{k.replace('_', '-')}", type=int, default=None)
    for k in ["lr", "clip"]:
        ap.add_argument(f"--{k}", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    for k, v in DEFAULTS[args.method].items():
        if getattr(args, k) is None:
            setattr(args, k, v)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_metadata(args.csv, args.img_dir, args.ext)
    tr, va, te = split(df, args.seed)
    print(f"train {len(tr)}  val {len(va)}  test {len(te)}")
    print(pd.crosstab(tr["group"].map(dict(enumerate(GROUP_NAMES))), tr["three_partition_label"]))

    backbone = Dinov2Model.from_pretrained(args.model).to(device)
    dim = backbone.config.hidden_size

    if args.method == "linear":
        # Backbone is frozen, so features are computed once and cached; DP only touches the head.
        cache = os.path.join(args.out, f"feats_{args.model.split('/')[-1]}_seed{args.seed}.pt")
        if os.path.exists(cache):
            feats = torch.load(cache)
        else:
            feats = {name: extract_features(backbone, d, device, args.workers)
                     for name, d in [("train", tr), ("val", va), ("test", te)]}
            torch.save(feats, cache)
        head = nn.Linear(2 * dim, len(LABELS)).to(device)
        train_ds = TensorDataset(feats["train"][0], feats["train"][1])
        head, sigma = train(head, train_ds, args, device)
        head.eval()
        results = {}
        for name in ["val", "test"]:
            X, Y, G = feats[name]
            with torch.no_grad():
                pred = head(X.to(device)).argmax(1).cpu().numpy()
            results[name] = group_metrics(pred, Y.numpy(), G.numpy())
    else:
        model = LoRAClassifier(backbone, len(LABELS), args.lora_r).to(device)
        model, sigma = train(model, Images(tr, TRAIN_TF), args, device)
        results = {}
        for name, d in [("val", va), ("test", te)]:
            pred = predict_images(model, d, device, args.workers)
            results[name] = group_metrics(pred, d["y"].to_numpy(), d["group"].to_numpy())

    record = {"method": args.method, "model": args.model, "epsilon": args.epsilon, "delta": args.delta,
              "noise_multiplier": sigma, "seed": args.seed,
              "hparams": {k: getattr(args, k) for k in DEFAULTS[args.method]}, **results}
    print(json.dumps(record, indent=2))
    fname = f"{args.method}_eps{args.epsilon}_seed{args.seed}.json"
    with open(os.path.join(args.out, fname), "w") as f:
        json.dump(record, f, indent=2)


if __name__ == "__main__":
    main()
