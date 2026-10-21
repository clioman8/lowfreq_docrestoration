"""Deterministic hard-case selection using a prespecified seed and input CER.

Selections are descriptive and do not select hyperparameters. Freeze the
threshold-adjacent interval on validation before final test selection.
"""
from __future__ import annotations
import argparse
from .common import read_json, read_jsonl, write_json


def select_cases(a_rows, b_rows, c_rows, input_rows, config, split="test"):
    analysis=config["analysis"]
    seed=analysis.get("example_seed",0)
    tau=float(analysis.get("threshold",.1))
    interval=analysis.get("threshold_adjacent_interval")
    if interval is None or len(interval)!=2 or not float(interval[0])<=tau<=float(interval[1]):
        raise ValueError("Prespecify analysis.threshold_adjacent_interval around the threshold")
    severities=[s["id"] for s in config["degradation"]["severities"]]
    def index(rows, learned):
        out={}
        for row in rows:
            if row.get("split") != split or (learned and row.get("seed") != seed):
                continue
            key=(row["sample_id"],row["severity_id"])
            if key in out:
                raise ValueError(f"Duplicate case key {key}")
            out[key]=row
        return out
    indexes=[index(rows,True) for rows in (a_rows,b_rows,c_rows)]
    inputs=index(input_rows,False)
    if not inputs or any(set(x)!=set(inputs) for x in indexes):
        raise ValueError("All methods and input must share the same sample/severity keys")
    for key,inp in inputs.items():
        if any(x[key].get("gt_normalized")!=inp.get("gt_normalized") for x in indexes):
            raise ValueError("Case ground-truth texts disagree")
    order={s:i for i,s in enumerate(severities)}
    sort_key=lambda key:(key[0],order[key[1]])
    reference=sorted([k for k in inputs if k[1]==severities[0]],key=sort_key)
    near=[k for k,r in inputs.items() if float(interval[0])<=float(r["cer"])<=float(interval[1])]
    near.sort(key=lambda k:(abs(float(inputs[k]["cer"])-tau),k[0],order[k[1]]))
    severe=[k for k in inputs if k[1]==severities[-1]]
    severe.sort(key=lambda k:(-float(inputs[k]["cer"]),k[0]))
    deterioration=lambda key:float(indexes[2][key]["cer"])-float(indexes[0][key]["cer"])
    failures=sorted(inputs,key=lambda k:(-deterioration(k),k[0],order[k[1]]))
    failure=failures[0] if failures and deterioration(failures[0])>0 else None
    choices={"reference":reference[0] if reference else None,
             "threshold_adjacent":near[0] if near else None,
             "severe_input":severe[0] if severe else None,
             "largest_C_deterioration_vs_A":failure}
    selected={}
    for label,key in choices.items():
        if key is None:
            selected[label]={"status":"not_found", "reason":"No example meets the prespecified condition"}
            continue
        selected[label]={"status":"selected","sample_id":key[0],"severity_id":key[1],
                         "input_cer":inputs[key]["cer"],"C_minus_A_cer":deterioration(key),
                         "input":inputs[key],"A":indexes[0][key],"B":indexes[1][key],"C":indexes[2][key]}
    return {"split":split,"example_seed":seed,"threshold":tau,"threshold_adjacent_interval":interval,
            "selection_rules":"reference: smallest ID at S0; near: degraded-input CER nearest threshold in interval; severe: maximum degraded-input CER at last severity; failure: largest positive C-A document CER, ties by ID/severity",
            "interpretation":"descriptive examples only; no authenticity or semantic recovery labels", "cases":selected}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("A","B","C","degraded","config","output"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--split",choices=["validation","test"],default="test")
    a=p.parse_args()
    report=select_cases(read_jsonl(a.A),read_jsonl(a.B),read_jsonl(a.C),read_jsonl(a.degraded),read_json(a.config),a.split)
    write_json(a.output,report)
    print(a.output)


if __name__=="__main__":
    main()
