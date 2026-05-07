import argparse
from pathlib import Path
import torch
from tqdm import tqdm

from resemattn.utils.io import load_yaml, save_json
from resemattn.utils.seed import set_seed
from resemattn.data.dataset import ReSemAttnDataset, move_question_to_device
from resemattn.model.resemattn import ReSemAttnModel
from resemattn.model.losses import combined_loss


def as_batch(q):
    return {k: v for k, v in q.__dict__.items() if torch.is_tensor(v)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--dataset", default="biohopr")
    ap.add_argument("--split", default="train")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    torch.set_num_threads(int(cfg.get("num_threads", 1)))
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if cfg["training"].get("device", "auto") == "auto" and torch.cuda.is_available() else "cpu")
    path = cfg["datasets"][args.dataset][args.split]
    ds = ReSemAttnDataset(path, cfg)
    model = ReSemAttnModel(cfg).to(device)
    
    opt_name = str(cfg["training"].get("optimizer", "sgd")).lower()
    if opt_name == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"].get("lr", 3e-4)), weight_decay=float(cfg["training"].get("weight_decay", 0.01)))
    else:
        opt = torch.optim.SGD(model.parameters(), lr=float(cfg["training"].get("lr", 1e-3)))
    history = []
    model.train()
    for epoch in range(int(cfg["training"].get("epochs", 5))):
        total = 0.0
        pbar = tqdm(range(len(ds)), desc=f"epoch {epoch+1}")
        for i in pbar:
            q = move_question_to_device(ds[i], device)
            batch = as_batch(q)
            out = model(batch, save_attention=False)
            losses = combined_loss(out["scores"], q.labels, cfg)
            opt.zero_grad()
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"].get("grad_clip", 1.0)))
            opt.step()
            total += float(losses["loss"].detach().cpu())
            pbar.set_postfix(loss=total/(i+1))
        history.append({"epoch": epoch+1, "loss": total/max(len(ds),1)})
    out_path = args.output or f"{cfg['output']['checkpoint_dir']}/{args.dataset}.pt"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg, "history": history}, out_path)
    save_json({"history": history, "checkpoint": out_path}, f"{cfg['output']['result_dir']}/{args.dataset}_train_history.json")
    print(f"saved checkpoint: {out_path}")

if __name__ == "__main__":
    main()
