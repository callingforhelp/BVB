import json
from pathlib import Path

p = Path("eval/test.jsonl")
rr = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
bank = {str(r.get("id")): r for r in rr}

print(f"bank size: {len(rr)}")
for qid in ("1686", "2270", "2271", "2272", "2683"):
    r = bank.get(qid)
    if r is None:
        print(f"{qid}: NOT IN BANK")
        continue
    print(f"--- qa {qid} ({r.get('question_type')}) ---")
    print(f"  Q: {str(r.get('question'))[:170]}")
    print(f"  gold: {r.get('ground_truth')}")
    if r.get("options"):
        print(f"  options: {r['options']}")
    print()
