from pathlib import Path
import argparse
from resemattn.utils.io import write_jsonl


def make_rows():
    rows = []
    rows.append({
        "qid": "toy-ra",
        "question": "Name all effect phenotypes which are side effects of drug that can treat disease rheumatoid arthritis.",
        "template": "disease:drug:effect/phenotype",
        "anchor": {"id": "D_RA", "name": "rheumatoid arthritis", "type": "disease"},
        "answer_type": "effect/phenotype",
        "gold_answers": ["Rash", "Nausea"],
        "candidates": [
            {"id":"E_RASH","name":"Rash","type":"effect/phenotype","label":1,"paths":[{"nodes":[{"id":"D_RA","name":"rheumatoid arthritis","type":"disease"},{"id":"DRUG_MT","name":"Methotrexate","type":"drug"},{"id":"E_RASH","name":"Rash","type":"effect/phenotype"}],"relations":[{"name":"inverse indication","direction":"reverse"},{"name":"side effect","direction":"forward"}],"topology":{"bridge_degree":12,"endpoint_degree":2,"branch":4,"pathcount":1}}]},
            {"id":"E_NEURO","name":"Peripheral neuropathy","type":"effect/phenotype","label":0,"paths":[{"nodes":[{"id":"D_RA","name":"rheumatoid arthritis","type":"disease"},{"id":"D_HEP","name":"Hepatotoxicity","type":"disease"},{"id":"E_NEURO","name":"Peripheral neuropathy","type":"effect/phenotype"}],"relations":[{"name":"contraindication","direction":"forward"},{"name":"phenotype present","direction":"forward"}],"topology":{"bridge_degree":80,"endpoint_degree":20,"branch":45,"pathcount":3}}]},
            {"id":"E_NAUSEA","name":"Nausea","type":"effect/phenotype","label":1,"paths":[{"nodes":[{"id":"D_RA","name":"rheumatoid arthritis","type":"disease"},{"id":"DRUG_MT","name":"Methotrexate","type":"drug"},{"id":"E_NAUSEA","name":"Nausea","type":"effect/phenotype"}],"relations":[{"name":"inverse indication","direction":"reverse"},{"name":"side effect","direction":"forward"}],"topology":{"bridge_degree":12,"endpoint_degree":4,"branch":4,"pathcount":1}}]},
        ]})
    rows.append({
        "qid": "toy-drug-gene-disease",
        "question": "Name all diseases related to gene protein associated with drug trans hydroxycinnamic acid.",
        "template": "drug:gene/protein:disease",
        "anchor": {"id":"DRUG_TH","name":"trans hydroxycinnamic acid","type":"drug"},
        "answer_type": "disease",
        "gold_answers": ["Influenza"],
        "candidates": [
            {"id":"D_FLU","name":"Influenza","type":"disease","label":1,"paths":[{"nodes":[{"id":"DRUG_TH","name":"trans hydroxycinnamic acid","type":"drug"},{"id":"G_SLC22A8","name":"SLC22A8","type":"gene/protein"},{"id":"D_FLU","name":"Influenza","type":"disease"}],"relations":[{"name":"inverse transporter","direction":"reverse"},{"name":"associated with","direction":"forward"}],"topology":{"bridge_degree":7,"endpoint_degree":3,"branch":3,"pathcount":1}}]},
            {"id":"D_KIDNEY","name":"kidney disease","type":"disease","label":0,"paths":[{"nodes":[{"id":"DRUG_TH","name":"trans hydroxycinnamic acid","type":"drug"},{"id":"DRUG_ES","name":"Estradiol","type":"drug"},{"id":"D_KIDNEY","name":"kidney disease","type":"disease"}],"relations":[{"name":"synergistic interaction","direction":"forward"},{"name":"contraindication","direction":"forward"}],"topology":{"bridge_degree":150,"endpoint_degree":42,"branch":81,"pathcount":5}}]},
        ]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/biohopr")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = make_rows()
    for split in ["train", "dev", "test"]:
        write_jsonl(rows, out / f"{split}.jsonl")
    print(f"Wrote toy data to {out}")

if __name__ == "__main__":
    main()
