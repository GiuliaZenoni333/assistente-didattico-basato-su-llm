
"""
Metriche RAG 

I valori N/A sono esclusi dal calcolo delle medie.
Metriche retrieval: Hit@1, Hit@3, Hit@5, Hit@8 e MRR.
"""

import json
from pathlib import Path
from collections import Counter

BASE = Path(__file__).resolve().parents[4]
DIR = BASE / "evaluation" / "llama_3.2" / "progr1"

QUESTIONS_FILE = BASE / "evaluation/questions/progr1/questions_progr1_final.jsonl"
RESULTS_FILE = DIR / "results/reranker/llamaindex_chroma_reranker.jsonl"
MANUAL_FILE = DIR / "manual/progr1_reranker_manual_final.jsonl"
OUTPUT_FILE = DIR / "metrics/reranker_metrics.json"

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


def is_numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def summarize(items, field):
    """
    Calcola la media dei soli punteggi numerici.
    I valori N/A non contribuiscono al denominatore.
    """
    values = [
        item[field]
        for item in items
        if is_numeric(item[field])
    ]

    counts = Counter(str(item[field]) for item in items)

    return {
        "total": len(items),
        "mean": round(sum(values) / len(values), 4) if values else None,
        "counts": {
            key: {
                "count": num,
                "percentage": pct(num, len(items))
            }
            for key, num in sorted(counts.items())
        }
    }


def source_match(actual, expected):
    """
    Controlla se una fonte recuperata corrisponde
    a una fonte attesa per file e numero di pagina.
    """
    actual_name = actual.get("file")
    expected_name = expected.get("file")

    aliases = {
        "Applicare-Uml-E-I-Pattern(5ed)Larman(1).pdf":
        "Applicare-Uml-E-I-Pattern(5ed)Larman.pdf"
    }

    return (
        actual_name == aliases.get(expected_name, expected_name)
        and str(actual.get("page")) in {
            str(p) for p in expected.get("pages", [])
        }
    )


def main():
    results = index(load(RESULTS_FILE), "risultati")
    manual = index(load(MANUAL_FILE), "manual")
    questions = index(load(QUESTIONS_FILE), "domande")

    ids = set(results)

    # Controllo corrispondenza ID dei tre file
    for label, group in (("manual", manual), ("questions", questions)):
        if set(group) != ids:
            raise ValueError(
                f"ID diversi per {label}: "
                f"mancanti={sorted(ids - set(group))}, "
                f"extra={sorted(set(group) - ids)}"
            )

    # Controllo testi delle domande e valori manuali
    for idx in sorted(ids):
        if results[idx]["question"] != questions[idx]["question"]:
            raise ValueError(f"Testo domanda differente all'ID {idx}")

        for field in FIELDS:
            val = manual[idx].get(field)

            if isinstance(val, bool) or val not in ALLOWED[field]:
                raise ValueError(
                    f"Valore non valido: ID {idx}, "
                    f"{field}={val!r}; ammessi={ALLOWED[field]}"
                )

    # Valutazione qualitativa complessiva
    ordered = [manual[i] for i in sorted(ids)]

    manual_scores = {
        field: summarize(ordered, field)
        for field in FIELDS
    }

    # Valutazione per difficoltà
    by_difficulty = {}
    by_question_type = {}

    answerable_ids = [
        i for i in sorted(ids)
        if questions[i].get("question_type") == "answerable"
    ]

    for difficulty in ("easy", "medium", "hard"):
        subset = [
            manual[i]
            for i in answerable_ids
            if questions[i].get("difficulty") == difficulty
        ]

        by_difficulty[difficulty] = {
            "total": len(subset),
            **{
                field: summarize(subset, field)
                for field in FIELDS
            }
        }

    if len(answerable_ids) != 30 or any(
        by_difficulty[d]["total"] != 10
        for d in ("easy", "medium", "hard")
    ):
        raise ValueError(
            "Distribuzione delle difficoltà inattesa: "
            f"{ {d: by_difficulty[d]['total'] for d in by_difficulty} }"
        )

    # Valutazione per tipologia di domanda
    question_types = sorted({
        questions[i].get("question_type", "unknown")
        for i in ids
    })

    for qtype in question_types:
        subset = [
            manual[i]
            for i in sorted(ids)
            if questions[i].get("question_type", "unknown") == qtype
        ]

        by_question_type[qtype] = {
            "total": len(subset),
            **{
                field: summarize(subset, field)
                for field in FIELDS
            }
        }

    # Retrieval: Hit@K e MRR
    evaluated = []

    for idx in sorted(ids):
        expected = questions[idx].get("expected_sources") or []

        # Escludiamo le domande senza fonti attese
        if not expected:
            continue

        ranks = [
            int(src.get("rank", pos))
            for pos, src in enumerate(
                results[idx].get("retrieved_sources", []), 1
            )
            if any(
                source_match(src, gold)
                for gold in expected
            )
        ]

        evaluated.append({
            "id": idx,
            "first_correct_rank": min(ranks) if ranks else None
        })

    n = len(evaluated)

    retrieval = {
        "evaluated_questions": n,
        "excluded_questions": len(ids) - n
    }

    for k in (1, 3, 5, 8):
        count = sum(
            item["first_correct_rank"] is not None
            and item["first_correct_rank"] <= k
            for item in evaluated
        )

        retrieval[f"hit_at_{k}"] = {
            "count": count,
            "percentage": pct(count, n)
        }

    retrieval["mrr"] = (
        round(
            sum(
                1 / item["first_correct_rank"]
                if item["first_correct_rank"] else 0
                for item in evaluated
            ) / n,
            4
        )
        if n else None
    )

    # Analisi degli errori:
    # consideriamo soltanto correctness numerica inferiore a 1.
    # Le domande con correctness N/A non sono classificate come errate.
    errors = [
        {
            "id": idx,
            "question": questions[idx]["question"],
            **{field: manual[idx][field] for field in FIELDS},
            "notes": manual[idx].get("notes", "")
        }
        for idx in sorted(ids)
        if is_numeric(manual[idx]["correctness"])
        and manual[idx]["correctness"] < 1
    ]

    # Risultati finali
    out = {
        "experiment": "llama_3.2_progr1_reranker",
        "total_questions": len(ids),
        **manual_scores,
        "by_difficulty": by_difficulty,
        "by_question_type": by_question_type,
        "retrieval": retrieval,
        "representative_errors": errors
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_FILE.write_text(
        json.dumps(out, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("Metriche salvate:", OUTPUT_FILE)

    for field in FIELDS:
        print(
            field,
            "media=", manual_scores[field]["mean"],
            "distribuzione=", manual_scores[field]["counts"]
        )

    print("Retrieval:", retrieval)


if __name__ == "__main__":
    main()
