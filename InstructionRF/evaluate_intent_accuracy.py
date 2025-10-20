import json
import argparse
import time
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from instruction_rf_client import InstructionRefinementClient
from pathlib import Path

VALID_INTENTS = ["ANOMALY_CHECK","PREDICTION","CONTROL","INFORMATION","OPTIMIZATION"]

DEFAULT_CSV = "/home/minjoo/Github/PRISM-Orch/InstructionRF/data/Semiconductor_intent_dataset__preview_.csv"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # 기본값(옵션으로 변경 가능)

def safe_intent(x):
    return x if isinstance(x, str) and x in VALID_INTENTS else None

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser(description="Evaluate intent classification accuracy using vLLM backend")
    ap.add_argument("--base_url", default=DEFAULT_BASE_URL, help="Base URL of vLLM OpenAI-compatible server (must end with /v1)")
    ap.add_argument("--csv", default=DEFAULT_CSV, help="Path to CSV dataset (id,intent,query)")
    ap.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds")
    ap.add_argument("--sleep", type=float, default=0.0, help="Optional delay between requests")
    ap.add_argument("--limit", type=int, default=0, help="Evaluate only the first N rows (0 = all)")
    ap.add_argument("--out", default="predictions.jsonl", help="Where to save detailed predictions (JSONL)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Model ID served by vLLM (e.g., unsloth/Qwen3-4B-Instruct-2507-bnb-4bit)")
    ap.add_argument("--no_raw", action="store_true", help="Do not include full model_output in JSONL rows")
    args = ap.parse_args()

    print(f"[info] Using dataset: {args.csv}")
    print(f"[info] Using base_url: {args.base_url}")
    print(f"[info] Using model:    {args.model}")

    client = InstructionRefinementClient(server_url=args.base_url, timeout=args.timeout, model=args.model)
    health = client.test_api_connection()
    if health.get("status") != "success":
        raise RuntimeError(f"Server not healthy: {health}")

    df = pd.read_csv(args.csv)
    if args.limit > 0:
        df = df.head(args.limit)

    # 출력 파일 경로들
    out_path = Path(args.out)
    stem = out_path.with_suffix("")  # predictions
    mis_path = Path(f"{stem}_misclassified.jsonl")
    ok_path  = Path(f"{stem}_correct.jsonl")
    csv_path = Path(f"{stem}_preds.csv")
    sum_path = Path(f"{stem}_summary.json")
    cm_path  = Path(f"{stem}_confusion_matrix.csv")

    gold, pred = [], []
    rows_all = []
    rows_ok, rows_mis = [], []
    errors = 0

    for _, row in df.iterrows():
        qid, query, true_intent = row["id"], row["query"], row["intent"]
        start = time.time()
        try:
            out = client.refine_instruction(query)
            latency = time.time() - start

            model_intent = safe_intent(out.get("intent_type"))
            final_pred = model_intent if model_intent else "INFORMATION"
            is_correct = (final_pred == true_intent)

            rec = {
                "id": qid,
                "query": query,
                "gold_intent": true_intent,
                "pred_raw": model_intent,          # 원래 모델이 뽑은 값(None 가능)
                "pred_final": final_pred,          # fallback 포함 최종 라벨
                "correct": is_correct,
                "latency_s": round(latency, 4),
                "error": None,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": args.model,
            }
            if not args.no_raw:
                rec["model_output"] = out

            rows_all.append(rec)
            (rows_ok if is_correct else rows_mis).append(rec)

            gold.append(true_intent)
            pred.append(final_pred)

        except Exception as e:
            latency = time.time() - start
            errors += 1

            rec = {
                "id": qid,
                "query": query,
                "gold_intent": true_intent,
                "pred_raw": None,
                "pred_final": "INFORMATION",       # 실패 시 기본값
                "correct": (true_intent == "INFORMATION"),
                "latency_s": round(latency, 4),
                "error": str(e),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": args.model,
            }
            rows_all.append(rec)
            (rows_ok if rec["correct"] else rows_mis).append(rec)

            gold.append(true_intent)
            pred.append("INFORMATION")

        if args.sleep > 0:
            time.sleep(args.sleep)

    # JSONL 저장 (전체/정답/오답)
    write_jsonl(out_path, rows_all)
    write_jsonl(ok_path, rows_ok)
    write_jsonl(mis_path, rows_mis)

    # CSV 저장(스프레드시트 확인용, model_output은 제외)
    df_out = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ("model_output",)}
        for r in rows_all
    ])
    df_out.to_csv(csv_path, index=False, encoding="utf-8")

    # 지표 계산 + 파일 저장
    acc = accuracy_score(gold, pred)
    report_dict = classification_report(
        gold, pred, labels=VALID_INTENTS, digits=3, zero_division=0, output_dict=True
    )
    cm = confusion_matrix(gold, pred, labels=VALID_INTENTS)

    summary = {
        "accuracy": acc,
        "errors": errors,
        "num_examples": len(gold),
        "labels": VALID_INTENTS,
        "classification_report": report_dict,  # per-class precision/recall/f1, macro/micro/weighted 포함
    }
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 혼동행렬 CSV
    pd.DataFrame(cm, index=VALID_INTENTS, columns=VALID_INTENTS).to_csv(cm_path, encoding="utf-8")

    # 콘솔 출력(기존 유지)
    print(f"\nAccuracy: {acc*100:.2f}%  (errors: {errors})\n")
    print(classification_report(gold, pred, labels=VALID_INTENTS, digits=3, zero_division=0))
    print("Confusion Matrix (labels in order):", VALID_INTENTS)
    print(cm)

    print("\n[Saved]")
    print(f"  • Detailed JSONL : {out_path}")
    print(f"  • Correct only   : {ok_path}")
    print(f"  • Miscls only    : {mis_path}")
    print(f"  • CSV (flat)     : {csv_path}")
    print(f"  • Summary (JSON) : {sum_path}")
    print(f"  • ConfMat (CSV)  : {cm_path}")

if __name__ == "__main__":
    main()