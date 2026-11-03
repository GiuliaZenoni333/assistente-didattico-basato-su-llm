
"""
Analisi degli errori della valutazione finale RAG
con reranker per il corso SAS.

Stampa le domande problematiche, calcola
le statistiche degli errori e salva l'analisi
in formato JSON.
"""

import json
from pathlib import Path


# PERCORSI
RESULTS_FILE = Path(
    "evaluation/llama_3.2/sas/results/reranker/llamaindex_chroma_reranker.jsonl"
)

MANUAL_FILE = Path(
    "evaluation/llama_3.2/sas/manual/sas_reranker_manual_final.jsonl"
)

OUTPUT_FILE = Path(
    "evaluation/llama_3.2/sas/error_analysis/reranker_error_analysis.json"
)


# LETTURA JSONL
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


# INDICIZZAZIONE DEI RISULTATI
results_by_id = {
    item["id"]: item
    for item in results
}


# CONTATORI
problematic_questions = 0

incorrect_answers = 0
incomplete_answers = 0
unsupported_answers = 0
incorrect_abstentions = 0


errors = []


# ANALISI
print("ANALISI ERRORI VALUTAZIONE FINALE - SAS + RERANKER")
print("=" * 100)


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

    problematic_questions += 1

    # CONTATORI ERRORI
    if (
        isinstance(evaluation["correctness"], (int, float))
        and evaluation["correctness"] < 1
    ):
        incorrect_answers += 1

    if (
        isinstance(evaluation["completeness"], (int, float))
        and evaluation["completeness"] < 1
    ):
        incomplete_answers += 1

    if (
        isinstance(evaluation["source_support"], (int, float))
        and evaluation["source_support"] < 1
        and evaluation["abstention"] != 1
    ):
        unsupported_answers += 1

    if evaluation["abstention"] == 0:
        incorrect_abstentions += 1

    # STAMPA DOMANDA
    print(
        "\n" + "=" * 100
    )

    print(
        f"ID: {result['id']}"
    )

    print(
        f"\nDOMANDA:\n"
        f"{result['question']}"
    )

    print(
        f"\nRISPOSTA ATTESA:\n"
        f"{result['expected_answer']}"
    )

    print(
        f"\nRISPOSTA SISTEMA:\n"
        f"{result['response']}"
    )

    print(
        f"\nFONTI ATTESE:\n"
        f"{json.dumps(result['expected_sources'], ensure_ascii=False, indent=2)}"
    )

    print(
        "\nFONTI RECUPERATE:"
    )

    for source in result[
        "retrieved_sources"
    ]:

        print(
            f"  Rank {source['rank']}: "
            f"{source['file']} "
            f"(pagina {source['page']}) "
            f"retrieval_score={source['retrieval_score']} "
            f"rerank_score={source['rerank_score']}"
        )

    print(
        "\nVALUTAZIONE MANUALE:"
    )

    print(
        f"  Correctness: "
        f"{evaluation['correctness']}"
    )

    print(
        f"  Completeness: "
        f"{evaluation['completeness']}"
    )

    print(
        f"  Source support: "
        f"{evaluation['source_support']}"
    )

    print(
        f"  Abstention: "
        f"{evaluation['abstention']}"
    )

    print(
        f"\nNOTE:\n"
        f"{evaluation['notes']}"
    )

    # SALVATAGGIO DEGLI ERRORI
    errors.append(
        {
            "id": result["id"],
            "question": result["question"],
            "question_type": result["question_type"],
            "difficulty": result["difficulty"],
            "expected_answer": result["expected_answer"],
            "response": result["response"],
            "expected_sources": result["expected_sources"],
            "retrieved_sources": result["retrieved_sources"],
            "correctness": evaluation["correctness"],
            "completeness": evaluation["completeness"],
            "source_support": evaluation["source_support"],
            "abstention": evaluation["abstention"],
            "notes": evaluation["notes"]
        }
    )


# STATISTICHE FINALI
print(
    "STATISTICHE FINALI"
)

print(
    f"\nDomande problematiche: "
    f"{problematic_questions}/{len(manual)}"
)

print(
    "\nCATEGORIE ERRORI"
)

print(
    f"  incorrect_answer: "
    f"{incorrect_answers}"
)

print(
    f"  incomplete_answer: "
    f"{incomplete_answers}"
)

print(
    f"  unsupported_answer: "
    f"{unsupported_answers}"
)

print(
    f"  incorrect_abstention: "
    f"{incorrect_abstentions}"
)


# SALVATAGGIO JSON
OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

analysis = {
    "system": "Llama 3.2 + reranker",
    "course": "SAS",
    "total_questions": len(manual),
    "problematic_questions": problematic_questions,
    "error_categories": {
        "incorrect_answer": incorrect_answers,
        "incomplete_answer": incomplete_answers,
        "unsupported_answer": unsupported_answers,
        "incorrect_abstention": incorrect_abstentions
    },
    "errors": errors
}

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
    f"\nFile salvato in: {OUTPUT_FILE}"
)

print(
    "FINE ANALISI"
)
