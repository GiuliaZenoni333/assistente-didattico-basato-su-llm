"""
Metriche RAG: valutazione manuale 0/0.5/1 e retrieval Hit@K/MRR.
"""
import json
from pathlib import Path
from collections import Counter

BASE = Path(__file__).resolve().parents[4]
DIR = BASE / "evaluation" / "phi4" / "sas"
QUESTIONS_FILE = BASE / "evaluation/questions/sas/questions_sas_final.jsonl"
RESULTS_FILE = DIR / "results/reranker/phi4_reranker.jsonl"
MANUAL_FILE = DIR / "manual/phi4_reranker_manual_final.jsonl"
OUTPUT_FILE = DIR / "metrics/phi4_reranker_metrics.json"

FIELDS = ("correctness", "completeness", "source_support", "abstention")

ALLOWED = {
    "correctness": (0, 0.5, 1, "N/A"),
    "completeness": (0, 0.5, 1, "N/A"),
    "source_support": (0, 0.5, 1, "N/A"),
    "abstention": (0, 1, "N/A")
}


def load(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def index(items, label):
    lookup = {}
    for item in items:
        idx = item["id"]
        if idx in lookup:
            raise ValueError(f"ID duplicato in {label}: {idx}")
        lookup[idx] = item
    return lookup


def pct(value, count):
    return round(value * 100 / count, 2) if count else 0.0


def summarize(items, field):
    counts = Counter(str(item[field]) for item in items)
    total = len(items)
    return {"total": total,
            "mean": round(sum(item[field] for item in items if isinstance(item[field], (int, float))) /
                          sum(isinstance(item[field], (int, float)) for item in items), 4)
                    if any(isinstance(item[field], (int, float)) for item in items) else None,
            "counts": {key: {"count": num, "percentage": pct(num, total)}
                       for key, num in sorted(counts.items())}}


def source_match(actual, expected):
    actual_name = actual.get("file")
    expected_name = expected.get("file")
    # Nel dataset SAS il PDF è chiamato ...Larman(1).pdf, ma l'indice
    # ChromaDB lo registra come ...Larman.pdf: stessi numeri di pagina.
    aliases = {"Applicare-Uml-E-I-Pattern(5ed)Larman(1).pdf":
                "Applicare-Uml-E-I-Pattern(5ed)Larman.pdf"}
    return (actual_name == aliases.get(expected_name, expected_name)
            and str(actual.get("page")) in {str(p) for p in expected.get("pages", [])})


def main():
    results = index(load(RESULTS_FILE), "risultati")
    manual = index(load(MANUAL_FILE), "manual")
    questions = index(load(QUESTIONS_FILE), "domande")
    ids = set(results)
    for label, group in (("manual", manual), ("questions", questions)):
        if set(group) != ids:
            raise ValueError(f"ID diversi per {label}: mancanti={sorted(ids-set(group))}, extra={sorted(set(group)-ids)}")
    for idx in ids:
        if results[idx]["question"] != questions[idx]["question"]:
            raise ValueError(f"Testo domanda differente all'ID {idx}")
        for field in FIELDS:
            val = manual[idx].get(field)
            if val not in ALLOWED[field] or (isinstance(val, bool)):
                raise ValueError(f"Valore non valido: ID {idx}, {field}={val!r}; ammessi={ALLOWED[field]}")
        
    ordered = [manual[i] for i in sorted(ids)]
    manual_scores = {f: summarize(ordered, f) for f in FIELDS}
    by_difficulty = {}
    by_question_type = {}
    # La difficolta' riguarda soltanto le 30 domande rispondibili.
    # Le 10 premesse false e le 10 non rispondibili restano in by_question_type.
    answerable_ids = [i for i in sorted(ids) if questions[i].get("question_type") == "answerable"]
    for difficulty in ("easy", "medium", "hard"):
        subset = [manual[i] for i in answerable_ids if questions[i].get("difficulty") == difficulty]
        by_difficulty[difficulty] = {"total": len(subset), **{f: summarize(subset, f) for f in FIELDS}}
    if len(answerable_ids) != 30 or any(by_difficulty[d]["total"] != 10 for d in ("easy", "medium", "hard")):
        raise ValueError(f"Distribuzione difficolta' inattesa: answerable={len(answerable_ids)}, "
                         f"gruppi={ {d: by_difficulty[d]['total'] for d in by_difficulty} }")
    for qtype in sorted({questions[i].get("question_type", "unknown") for i in ids}):
        subset = [manual[i] for i in sorted(ids) if questions[i].get("question_type", "unknown") == qtype]
        by_question_type[qtype] = {"total": len(subset), **{f: summarize(subset, f) for f in FIELDS}}
    evaluated = []
    for idx in sorted(ids):
        expected = questions[idx].get("expected_sources") or []
        if not expected:   # Domande NOT ANSWERABLE senza fonti attese: escluse dal retrieval
            continue
        ranks = [int(src.get("rank", pos)) for pos, src in enumerate(results[idx].get("retrieved_sources", []), 1)
                 if any(source_match(src, gold) for gold in expected)]
        evaluated.append({"id": idx, "first_correct_rank": min(ranks) if ranks else None})
    n = len(evaluated)
    retrieval = {"evaluated_questions": n, "excluded_questions": len(ids)-n}
    for k in (1, 3, 5, 8):
        count = sum(x["first_correct_rank"] is not None and x["first_correct_rank"] <= k for x in evaluated)
        retrieval[f"hit_at_{k}"] = {"count": count, "percentage": pct(count, n)}
    retrieval["mrr"] = round(sum(1 / x["first_correct_rank"] if x["first_correct_rank"] else 0 for x in evaluated)/n, 4) if n else None

    errors = [
        {
            "id": idx,
            "question": questions[idx]["question"],
            **{f: manual[idx][f] for f in FIELDS},
            "notes": manual[idx].get("notes", "")
        }
        for idx in sorted(ids)
        if isinstance(manual[idx]["correctness"], (int, float))
        and manual[idx]["correctness"] < 1
    ]

    out = {"experiment": "phi4_sas_reranker", "total_questions": len(ids),
           **manual_scores, "by_difficulty": by_difficulty, "by_question_type": by_question_type,
           "retrieval": retrieval, "representative_errors": errors}
    
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    
    print("Metriche salvate:", OUTPUT_FILE)
    
    for field in FIELDS:
        print(field, "media=", manual_scores[field]["mean"], "distribuzione=", manual_scores[field]["counts"])
    print("Retrieval:", retrieval)


if __name__ == "__main__":
    main()
