
"""
Error Analysis 

Punteggi:
- Correctness, Completeness, Source Support: 0, 0.5, 1, N/A
- Abstention: 0, 1, N/A

I valori N/A non sono considerati errori.
"""

import json
from pathlib import Path


# PERCORSI
BASE = Path(__file__).resolve().parents[4]
DIR = BASE / "evaluation" / "llama_3.2" / "progr1"

RESULTS_FILE = DIR / "results/reranker/llamaindex_chroma_reranker.jsonl"
MANUAL_FILE = DIR / "manual/progr1_reranker_manual_final.jsonl"
OUTPUT_FILE = DIR / "error_analysis/reranker_error_analysis.json"

FIELDS = ("correctness", "completeness", "source_support", "abstention")

ALLOWED = {
    "correctness": (0, 0.5, 1, "N/A"),
    "completeness": (0, 0.5, 1, "N/A"),
    "source_support": (0, 0.5, 1, "N/A"),
    "abstention": (0, 1, "N/A")
}


def load_jsonl(path):
    """Carica i record JSONL e verifica eventuali ID duplicati."""
    records = {}

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            record = json.loads(line)
            idx = record["id"]

            if idx in records:
                raise ValueError(
                    f"ID duplicato {idx} in {path}, riga {line_number}"
                )

            records[idx] = record

    return records


def is_numeric(value):
    """Distingue i punteggi numerici dai valori N/A."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def is_error(value):
    """Un punteggio è problematico soltanto se numerico e inferiore a 1."""
    return is_numeric(value) and value < 1


def main():
    results = load_jsonl(RESULTS_FILE)
    manual = load_jsonl(MANUAL_FILE)

    # Verifica della corrispondenza tra risultati e valutazioni
    if set(results) != set(manual):
        raise ValueError(
            "Gli ID dei risultati e delle valutazioni non coincidono. "
            f"Mancanti: {sorted(set(results) - set(manual))}. "
            f"Extra: {sorted(set(manual) - set(results))}."
        )

    if len(manual) != 50:
        raise ValueError(
            f"Attese 50 valutazioni manuali, trovate {len(manual)}."
        )

    # Verifica dei punteggi della professoressa
    for idx, evaluation in manual.items():
        for field in FIELDS:
            value = evaluation.get(field)

            if isinstance(value, bool) or value not in ALLOWED[field]:
                raise ValueError(
                    f"Valore non valido: ID {idx}, "
                    f"{field}={value!r}"
                )

    counters = {
        "incorrect_answer": 0,
        "incomplete_answer": 0,
        "unsupported_answer": 0,
        "incorrect_abstention": 0
    }

    errors = []

    print("\nERROR ANALYSIS - LLAMA 3.2 - PROGRAMMAZIONE I")
    print("Valutazione manuale della professoressa\n")

    for idx in sorted(manual):
        evaluation = manual[idx]
        result = results[idx]

        incorrect = is_error(evaluation["correctness"])
        incomplete = is_error(evaluation["completeness"])

        unsupported = (
            is_error(evaluation["source_support"])
            and evaluation["abstention"] != 1
        )

        incorrect_abstention = evaluation["abstention"] == 0

        categories = []

        if incorrect:
            counters["incorrect_answer"] += 1
            categories.append("incorrect_answer")

        if incomplete:
            counters["incomplete_answer"] += 1
            categories.append("incomplete_answer")

        if unsupported:
            counters["unsupported_answer"] += 1
            categories.append("unsupported_answer")

        if incorrect_abstention:
            counters["incorrect_abstention"] += 1
            categories.append("incorrect_abstention")

        # Se nessun criterio segnala problemi, passa alla domanda successiva
        if not categories:
            continue

        error = {
            "id": idx,
            "question": result["question"],
            "question_type": result.get("question_type"),
            "difficulty": result.get("difficulty"),
            "expected_answer": result.get("expected_answer"),
            "response": result.get("response"),
            "expected_sources": result.get("expected_sources", []),
            "retrieved_sources": result.get("retrieved_sources", []),
            "correctness": evaluation["correctness"],
            "completeness": evaluation["completeness"],
            "source_support": evaluation["source_support"],
            "abstention": evaluation["abstention"],
            "notes": evaluation.get("notes", ""),
            "error_categories": categories
        }

        errors.append(error)

        print(f"ID {idx}: {', '.join(categories)}")

    analysis = {
        "system": "Llama 3.2 + reranker",
        "course": "Programmazione 1",
        "total_questions": len(manual),
        "problematic_questions": len(errors),
        "error_categories": counters,
        "errors": errors
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_FILE.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("\nSTATISTICHE FINALI")
    print(f"Domande analizzate: {len(manual)}")
    print(f"Domande problematiche: {len(errors)}")

    for category, count in counters.items():
        print(f"{category}: {count}")

    print(f"\nFile salvato: {OUTPUT_FILE}")
    print("ANALISI COMPLETATA")


if __name__ == "__main__":
    main()
