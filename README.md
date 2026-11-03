# Assistente didattico basato su LLM

**Bachelor's Thesis Project | Computer Science**

The public datasets contain selected evaluation information. Reference answers, retrieved source details and descriptive error annotations are excluded. Reproducing the original experiments requires access to the authorized course materials and complete private datasets.

## Project Overview

The project investigates two complementary functionalities:

- **Student assistance:** answering questions based on university teaching materials through a local RAG pipeline.
- **Question analysis:** grouping semantically similar student questions to help instructors identify frequently discussed topics.

The system was developed as a research prototype for a bachelor's thesis.

## RAG Architecture

The RAG pipeline consists of the following steps:

1. **Document processing:** extracting text from PDF files using PyMuPDF.
2. **Text chunking:** dividing documents into smaller overlapping segments.
3. **Embeddings:** converting text segments into vector representations using BGE-M3.
4. **Vector storage:** storing and retrieving document representations through ChromaDB.
5. **Retrieval and reranking:** identifying and prioritizing relevant document segments.
6. **Answer generation:** generating contextual responses using locally executed language models.

The implementation uses LlamaIndex to integrate the retrieval and generation components.

## Language Models

Two local language models were investigated:

- **Llama 3.2 (3B)** — Meta
- **Phi-4-mini** — Microsoft

Both models were evaluated within a RAG-based question-answering workflow using Ollama.

## Student Question Analysis

The project also explores semantic grouping of student questions through:

- **BGE-M3** for text embeddings.
- **DBSCAN** for clustering semantically related questions.

The objective is to identify recurring question topics and provide instructors with a structured overview of students' queries.

## Evaluation Methodology

The experimental evaluation involved two university courses:

- **Programmazione I**
- **Sviluppo delle Applicazioni Software (SAS)**

A dataset of **100 questions** was developed, with 50 questions per course.

Questions were divided into five categories:

| Category | Questions per course |
|---|---:|
| Easy | 10 |
| Medium | 10 |
| Hard | 10 |
| False premise | 10 |
| Unanswerable | 10 |
| **Total** | **50** |

The two language models generated a total of **200 responses** across the experiments.

### Answer Quality

Generated answers were assessed according to:

- Correctness
- Completeness
- Source support
- Appropriate abstention

### Retrieval Evaluation

The retrieval component was evaluated using:

- **Hit@K:** whether at least one expected source appears among the first K retrieved results.
- **Mean Reciprocal Rank (MRR):** the average reciprocal rank of the first relevant retrieved source.

## Technologies

- Python
- Ollama
- LlamaIndex
- PyMuPDF
- BGE-M3
- ChromaDB
- CrossEncoder / BGE reranker
- DBSCAN
- Llama 3.2
- Phi-4-mini

## Repository Contents

- `rag_system/` — RAG implementation and model configurations.
- `evaluation/questions/` — Evaluation questions and their categories.
- `evaluation/llama_3.2/` — Llama 3.2 responses, manual assessments, metrics, and error analysis.
- `evaluation/phi4/` — Phi-4-mini responses, manual assessments, metrics, and error analysis.
- `evaluation/comparison/` — Model comparison scripts.
- `webapp/` — Web interface for students and instructors.
- `requirements.txt` — Python dependencies.
- `LICENSE` — Software license.

This public version contains selected implementation files. Some evaluation scripts depend on locally generated datasets or experimental outputs that are not included.

## Teaching Materials and Data Availability

**The original teaching materials used for the experiments are not distributed in this public repository.**

Course slides, textbooks, processed document extracts, populated vector databases, and selected experimental datasets have been excluded to respect copyright, privacy, and redistribution restrictions.

Users should supply their own documents and generate their own vector indexes when adapting the project.

Some experimental components may require additional local files or configuration before they can be executed.

## Project Status

This repository represents an academic research prototype rather than a production-ready educational platform.

Future work may include expanding the supported courses, improving retrieval performance, evaluating the system with real users, and further validating the instructor dashboard.

## License

The software is distributed under the terms specified in the `LICENSE` file.

The license does not apply to third-party teaching materials, textbooks, or other content excluded from this repository.