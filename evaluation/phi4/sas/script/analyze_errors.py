
"""
Analisi degli errori della valutazione Phi-4 + reranker.

Legge i risultati generati dal sistema e la valutazione manuale
e produce un file strutturato con gli errori riscontrati.
"""

import json
from pathlib import Path
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parents[4]

BASE_DIR = PROJECT_ROOT / "evaluation" / "phi4" / "sas"

RESULTS_FILE = (BASE_DIR / "results" / "reranker" / "phi4_reranker.jsonl")

MANUAL_FILE = (BASE_DIR / "manual" / "phi4_reranker_manual_final.jsonl")

OUTPUT_FILE = (BASE_DIR / "error_analysis" / "phi4_error_analysis.json")


def load_jsonl(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


results = load_jsonl(
    RESULTS_FILE
)

manual = load_jsonl(
    MANUAL_FILE
)


# CONTROLLO VALUTAZIONI MANUALI
for evaluation in manual:

    for field in ["correctness", "completeness"]:
        if evaluation[field] not in (0, 0.5, 1, "N/A"):
            raise ValueError(
                f"Valore non valido: ID {evaluation['id']}, "
                f"{field}={evaluation[field]}"
            )

    if evaluation["source_support"] not in (0, 0.5, 1, "N/A"):
        raise ValueError(
            f"Source Support non valido: ID {evaluation['id']}"
        )


results_by_id = {
    item["id"]: item
    for item in results
}


errors = []

error_categories = Counter()


for evaluation in manual:

    result = results_by_id[
        evaluation["id"]
    ]

    problematic = (
        (
            isinstance(evaluation["correctness"], (int, float))
            and evaluation["correctness"] < 1
        )
        or (
            isinstance(evaluation["completeness"], (int, float))
            and evaluation["completeness"] < 1
        )
        or (
            isinstance(evaluation["source_support"], (int, float))
            and evaluation["source_support"] < 1
            and evaluation["abstention"] != 1
        )
        or evaluation["abstention"] == 0
    )

    if not problematic:
        continue

    categories = []

    if (
        isinstance(evaluation["correctness"], (int, float))
        and evaluation["correctness"] < 1
    ):
        categories.append(
            "incorrect_answer"
        )
        error_categories[
            "incorrect_answer"
        ] += 1

    if (
        isinstance(evaluation["completeness"], (int, float))
        and evaluation["completeness"] < 1
    ):
        categories.append(
            "incomplete_answer"
        )
        error_categories[
            "incomplete_answer"
        ] += 1

    if (
        isinstance(evaluation["source_support"], (int, float))
        and evaluation["source_support"] < 1
        and evaluation["abstention"] != 1
    ):
        categories.append(
            "unsupported_answer"
        )
        error_categories[
            "unsupported_answer"
        ] += 1

    if evaluation["abstention"] == 0:
        categories.append(
            "incorrect_abstention"
        )
        error_categories[
            "incorrect_abstention"
        ] += 1

    errors.append(
        {
            "id": result["id"],

            "question":
                result["question"],

            "question_type":
                result["question_type"],

            "difficulty":
                result["difficulty"],

            "response":
                result["response"],

            "expected_answer":
                result["expected_answer"],

            "retrieved_sources":
                result["retrieved_sources"],

            "correctness":
                evaluation["correctness"],

            "completeness":
                evaluation["completeness"],

            "source_support":
                evaluation["source_support"],

            "abstention":
                evaluation["abstention"],

            "error_categories":
                categories,

            "notes":
                evaluation["notes"]
        }
    )


errors_by_difficulty = {}

for difficulty in [
    "easy",
    "medium",
    "hard"
]:

    subset = [
        error
        for error in errors
        if error["question_type"] == "answerable"
        and error["difficulty"] == difficulty
    ]

    errors_by_difficulty[
        difficulty
    ] = {
        "total_errors": len(subset),
        "question_ids": [
            error["id"]
            for error in subset
        ]
    }


errors_by_question_type = {}

question_types = [
    "answerable",
    "premise_error",
    "unanswerable"
]

for question_type in question_types:

    subset = [
        error
        for error in errors
        if error["question_type"]
        == question_type
    ]

    errors_by_question_type[
        question_type
    ] = {
        "total_errors": len(subset),
        "question_ids": [
            error["id"]
            for error in subset
        ]
    }


analysis = {

    "experiment":
        "phi4_reranker",

    "total_questions":
        len(manual),

    "total_problematic_questions":
        len(errors),

    "error_categories": dict(
        error_categories
    ),

    "errors_by_difficulty":
        errors_by_difficulty,

    "errors_by_question_type":
        errors_by_question_type,

    "errors":
        errors
}


OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        analysis,
        file,
        ensure_ascii=False,
        indent=2
    )


print(
    "ANALISI ERRORI PHI-4 + RERANKER"
)

print(
    f"Domande problematiche: "
    f"{len(errors)}/{len(manual)}"
)

print("\nCATEGORIE ERRORI")

for category, count in error_categories.items():

    print(
        f"  {category}: {count}"
    )

print(
    f"\nFile salvato in: {OUTPUT_FILE}"
)
