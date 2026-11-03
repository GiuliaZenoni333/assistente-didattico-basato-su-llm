"""
Valuta il retrieval del sistema RAG con reranking
per il corso di Programmazione 1.

Per ogni domanda controlla se almeno una delle fonti attese
compare tra i risultati riordinati dal reranker e calcola
Hit@1, Hit@3, Hit@5, Hit@8 e MRR.
"""

import json
import sys
from pathlib import Path


# Aggiungo la root del progetto al percorso Python
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))



QUESTIONS_FILE = (PROJECT_ROOT / "evaluation" / "questions" / "progr1" / "questions_progr1_final.jsonl")

RESULTS_FILE = (PROJECT_ROOT / "evaluation" / "llama_3.2" / "progr1" / "results" / "reranker" / "llamaindex_chroma_reranker.jsonl")
def load_jsonl(path):

    data = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            if line.strip():

                data.append(
                    json.loads(line)
                )

    return data


# Carico domande e risultati
questions = load_jsonl(
    QUESTIONS_FILE
)

results = load_jsonl(
    RESULTS_FILE
)


# Controllo che domande e risultati
# siano dello stesso numero
if len(questions) != len(results):

    raise ValueError(
        f"Numero di domande ({len(questions)}) "
        f"diverso dal numero di risultati ({len(results)})."
    )

for question, result in zip(questions, results):

    if question["id"] != result["id"]:
        raise ValueError(
            f"ID non corrispondenti: "
            f"domanda {question['id']}, "
            f"risultato {result['id']}"
        )

    if question["question"] != result["question"]:
        raise ValueError(
            f"Testo differente all'ID {question['id']}"
        )


hit_1 = 0
hit_3 = 0
hit_5 = 0
hit_8 = 0

reciprocal_rank_sum = 0

evaluated_questions = 0


# Valuto il retrieval per ogni domanda
for question, result in zip(
    questions,
    results
):

    expected_sources = question[
        "expected_sources"
    ]


    # Le domande senza fonti attese
    # non vengono valutate automaticamente
    if not expected_sources:
        continue


    evaluated_questions += 1

    rank_found = None


    # Controllo le fonti recuperate
    # dopo il reranking
    for source in result[
        "retrieved_sources"
    ]:

        source_file = source[
            "file"
        ]

        source_page = str(
            source["page"]
        )


        # Confronto con le fonti attese
        for expected_source in expected_sources:

            expected_file = expected_source[
                "file"
            ]

            expected_pages = expected_source[
                "pages"
            ]


            same_file = (
                source_file == expected_file
            )

            same_page = (
                source_page
                in [
                    str(page)
                    for page in expected_pages
                ]
            )


            if same_file and same_page:

                rank_found = source[
                    "rank"
                ]

                break


        if rank_found is not None:
            break


    # Aggiorno le metriche
    if rank_found is not None:

        if rank_found <= 1:
            hit_1 += 1

        if rank_found <= 3:
            hit_3 += 1

        if rank_found <= 5:
            hit_5 += 1

        if rank_found <= 8:
            hit_8 += 1

        reciprocal_rank_sum += (
            1 / rank_found
        )


    print(
        f"Domanda {question['id']} - "
        f"rank fonte corretta: "
        f"{rank_found if rank_found is not None else 'non trovata'}"
    )


# Controllo che ci siano domande valutabili
if evaluated_questions == 0:

    raise ValueError(
        "Nessuna domanda con fonti attese da valutare."
    )


# Risultati finali
print(
    "\n--- RISULTATI RETRIEVAL RERANKER ---"
)

print(
    f"Domande valutate: "
    f"{evaluated_questions}"
)

print(
    f"Hit@1: "
    f"{hit_1 / evaluated_questions:.2f}"
)

print(
    f"Hit@3: "
    f"{hit_3 / evaluated_questions:.2f}"
)

print(
    f"Hit@5: "
    f"{hit_5 / evaluated_questions:.2f}"
)
print(
    f"Hit@8: "
    f"{hit_8 / evaluated_questions:.2f}"
)

print(
    f"MRR: "
    f"{reciprocal_rank_sum / evaluated_questions:.2f}"
)