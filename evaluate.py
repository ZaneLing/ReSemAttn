import argparse
from pathlib import Path
import torch

from resemattn.utils.io import load_yaml, save_json, write_jsonl
from resemattn.data.dataset import ReSemAttnDataset, move_question_to_device
from resemattn.model.resemattn import ReSemAttnModel
from resemattn.metrics.answer_metrics import multi_answer_ranking_metrics
from resemattn.metrics.coverage_metrics import coverage_diagnostics
from resemattn.metrics.path_validity import path_validity_metrics, attention_validity


def as_batch(q):
    return {k: v for k, v in q.__dict__.items() if torch.is_tensor(v)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--dataset", default="biohopr")
    ap.add_argument("--split", default="test")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    torch.set_num_threads(int(cfg.get("num_threads", 1)))
    device = torch.device("cuda" if cfg["training"].get("device", "auto") == "auto" and torch.cuda.is_available() else "cpu")
    ds = ReSemAttnDataset(cfg["datasets"][args.dataset][args.split], cfg)
    model = ReSemAttnModel(cfg).to(device)
    ckpt_path = args.checkpoint or f"{cfg['output']['checkpoint_dir']}/{args.dataset}.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    predictions = []
    save_attention = bool(cfg["training"].get("save_attention", True))
    with torch.no_grad():
        for i in range(len(ds)):
            q = move_question_to_device(ds[i], device)
            out = model(as_batch(q), save_attention=save_attention)
            scores = out["scores"].detach().cpu().tolist()
            prior = out["prior"].detach().cpu().tolist()
            alpha = out.get("alpha")
            if alpha is not None:
                # aggregate token attention to path attention per candidate.
                path_attn = alpha.detach().cpu().mean(dim=1).tolist()
            else:
                path_attn = [None] * len(scores)
            row = {"qid": q.qid, "question": q.raw.get("question", ""), "template": q.raw.get("template", ""), "raw": q.raw, "candidates": []}
            raw_cands = q.raw.get("candidates", [])[:len(scores)]
            for j, cand in enumerate(raw_cands):
                c = dict(cand)
                c["score"] = float(scores[j])
                c["prior"] = prior[j]
                if path_attn[j] is not None:
                    c["path_attention"] = path_attn[j]
                row["candidates"].append(c)
            predictions.append(row)
    ks = tuple(cfg.get("evaluation", {}).get("ks", [1,5,10]))
    metrics = {}
    metrics.update(multi_answer_ranking_metrics(predictions, ks=ks))
    metrics.update({"coverage/"+k: v for k, v in coverage_diagnostics(predictions, ks=ks).items()})
    metrics.update({"path/"+k: v for k, v in path_validity_metrics(predictions, ks=ks).items()})
    metrics.update({"path/"+k: v for k, v in attention_validity(predictions).items()})
    out_dir = Path(args.out_dir or f"{cfg['output']['result_dir']}/{args.dataset}_{args.split}")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(predictions, out_dir / "predictions.jsonl")
    save_json(metrics, out_dir / "metrics.json")
    print(metrics)
    print(f"wrote: {out_dir}")

if __name__ == "__main__":
    main()
