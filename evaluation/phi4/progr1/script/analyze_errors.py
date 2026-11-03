
"""
Error Analysis 
Punteggi:
- Correctness, Completeness, Source Support: 0, 0.5, 1, N/A
- Abstention: 0, 1, N/A

I valori N/A non vengono considerati errori.
"""

import json
from pathlib import Path


# PERCORSI
PROJECT_ROOT = Path(__file__).resolve().parents[4]
BASE_DIR = PROJECT_ROOT / "evaluation" / "phi4" / "progr1"

RESULTS_FILE = BASE_DIR / "results/reranker/phi4_reranker.jsonl"
MANUAL_FILE = BASE_DIR / "manual/phi4_progr1_reranker_manual_final.jsonl"
OUTPUT_FILE = BASE_DIR / "error_analysis/phi4_error_analysis.json"

FIELDS = ("correctness", "completeness", "source_support", "abstention")

ALLOWED = {
    "correctness": (0, 0.5, 1, "N/A"),
    "completeness": (0, 0.5, 1, "N/A"),
    "source_support": (0, 0.5, 1, "N/A"),
    "abstention": (0, 1, "N/A")
}


def load_jsonl(path):
    """Carica un file JSONL verificando gli ID duplicati."""
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
    """Verifica che il valore sia numerico e non booleano."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def is_error(value):
    """Un punteggio è problematico se numerico e inferiore a 1."""
    return is_numeric(value) and value < 1


def main():
    results = load_jsonl(RESULTS_FILE)
    manual = load_jsonl(MANUAL_FILE)

    # Controllo della corrispondenza tra risultati e valutazioni
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

    # Validazione dei punteggi della professoressa
    for idx, evaluation in manual.items():
        for field in FIELDS:
            value = evaluation.get(field)

            if isinstance(value, bool) or value not in ALLOWED[field]:
                raise ValueError(
                    f"Valore non valido: ID {idx}, "
                    f"{field}={value!r}"
                )

    error_categories = {
        "incorrect_answer": 0,
        "incomplete_answer": 0,
        "unsupported_answer": 0,
        "incorrect_abstention": 0
    }

    errors = []

    # Analisi delle valutazioni
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
            error_categories["incorrect_answer"] += 1
            categories.append("incorrect_answer")

        if incomplete:
            error_categories["incomplete_answer"] += 1
            categories.append("incomplete_answer")

        if unsupported:
            error_categories["unsupported_answer"] += 1
            categories.append("unsupported_answer")

        if incorrect_abstention:
            error_categories["incorrect_abstention"] += 1
            categories.append("incorrect_abstention")

        if not categories:
            continue

        errors.append({
            "id": idx,
            "question": result["question"],
            "question_type": result.get("question_type"),
            "difficulty": result.get("difficulty"),
            "response": result.get("response"),
            "expected_answer": result.get("expected_answer"),
            "expected_sources": result.get("expected_sources", []),
            "retrieved_sources": result.get("retrieved_sources", []),
            "correctness": evaluation["correctness"],
            "completeness": evaluation["completeness"],
            "source_support": evaluation["source_support"],
            "abstention": evaluation["abstention"],
            "error_categories": categories,
            "notes": evaluation.get("notes", "")
        })

    # Statistiche per difficoltà
    errors_by_difficulty = {}

    for difficulty in ("easy", "medium", "hard"):
        subset = [
            error for error in errors
            if error["question_type"] == "answerable"
            and error["difficulty"] == difficulty
        ]

        errors_by_difficulty[difficulty] = {
            "total_errors": len(subset),
            "question_ids": [error["id"] for error in subset]
        }

    # Statistiche per tipologia di domanda
    errors_by_question_type = {}

    for question_type in ("answerable", "premise_error", "unanswerable"):
        subset = [
            error for error in errors
            if error["question_type"] == question_type
        ]

        errors_by_question_type[question_type] = {
            "total_errors": len(subset),
            "question_ids": [error["id"] for error in subset]
        }

    # Risultato finale
    analysis = {
        "experiment": "phi4_progr1_reranker",
        "total_questions": len(manual),
        "total_problematic_questions": len(errors),
        "error_categories": error_categories,
        "errors_by_difficulty": errors_by_difficulty,
        "errors_by_question_type": errors_by_question_type,
        "errors": errors
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_FILE.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # Riepilogo
    print("\nERROR ANALYSIS - PHI-4-MINI - PROGRAMMAZIONE I")
    print("Valutazione manuale della professoressa\n")

    print(f"Domande analizzate: {len(manual)}")
    print(f"Domande problematiche: {len(errors)}")

    for category, count in error_categories.items():
        print(f"{category}: {count}")

    print(f"\nFile salvato: {OUTPUT_FILE}")
    print("ANALISI COMPLETATA")


if __name__ == "__main__":
    main()
