"""
Valuta il retrieval del sistema RAG con Phi-4-mini
e reranker per Programmazione 1.

Per ogni domanda controlla se almeno una delle fonti attese
compare tra i risultati recuperati e calcola Hit@1,
Hit@3, Hit@5, Hit@8 e MRR.
"""

import json
from pathlib import Path


QUESTIONS_FILE = Path(
    "evaluation/questions/progr1/questions_progr1_final.jsonl"
)

RESULTS_FILE = Path(
    "evaluation/phi4/progr1/results/reranker/phi4_reranker.jsonl"
)


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


questions = load_jsonl(
    QUESTIONS_FILE
)

results = load_jsonl(
    RESULTS_FILE
)


# Controllo che domande e risultati corrispondano
if len(questions) != len(results):

    raise ValueError(
        "Il numero di domande e risultati non coincide."
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


    for source in result[
        "retrieved_sources"
    ]:

        source_file = source[
            "file"
        ]

        source_page = str(
            source["page"]
        )


        for expected_source in expected_sources:

            expected_file = expected_source[
                "file"
            ]

            expected_pages = [
                str(page)
                for page in expected_source[
                    "pages"
                ]
            ]


            same_file = (
                source_file
                == expected_file
            )

            same_page = (
                source_page
                in expected_pages
            )


            if same_file and same_page:

                rank_found = source[
                    "rank"
                ]

                break


        if rank_found is not None:

            break


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


print(
    "\n--- RISULTATI RETRIEVAL PHI-4 ---"
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