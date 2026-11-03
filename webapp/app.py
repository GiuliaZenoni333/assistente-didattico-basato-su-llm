
import sqlite3
import uuid
import json
from werkzeug.utils import secure_filename
from functools import lru_cache
from pathlib import Path

from flask import Flask, render_template, request, jsonify

# Cartelle del progetto
WEBAPP_DIR = Path(__file__).resolve().parent
ROOT = WEBAPP_DIR.parent
DB_FILE = WEBAPP_DIR / "questions.db"

COURSES = {
    "progr1": "Programmazione I",
    "sas": "Sviluppo delle Applicazioni Software"
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024  # 15 MB per richiesta PDF


# DATABASE DELLE DOMANDE
def connect_db():
    con = sqlite3.connect(DB_FILE)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    with connect_db() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                course TEXT NOT NULL,
                student_code TEXT NOT NULL,
                question TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                student_code TEXT NOT NULL,
                course TEXT NOT NULL,
                title TEXT NOT NULL DEFAULT 'Nuova conversazione',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS conversation_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user','assistant')),
                content TEXT NOT NULL,
                sources_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id)
            )
        """)
        con.execute("""CREATE TABLE IF NOT EXISTS course_enrollment (
            course TEXT PRIMARY KEY,
            enrolled_students INTEGER NOT NULL CHECK(enrolled_students > 0)
        )""")
        con.execute("CREATE INDEX IF NOT EXISTS idx_conversations_owner ON conversations(student_code, course, updated_at)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_messages_conversation ON conversation_messages(conversation_id, id)")

# SISTEMA RAG CON RERANKER
@lru_cache(maxsize=2)
def get_pipeline(course):
    # Importiamo le librerie quando servono
    import chromadb
    from sentence_transformers import CrossEncoder

    from llama_index.core import VectorStoreIndex
    from llama_index.embeddings.ollama import OllamaEmbedding
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.llms.ollama import Ollama

    # Indice del corso selezionato
    chroma_path = (
        ROOT / "rag_system" / "Llama_3_2"
        / "chroma_db" / course
    )

    if not chroma_path.is_dir():
        raise FileNotFoundError(
            f"Indice ChromaDB non trovato: {chroma_path}"
        )

    # Modello per gli embedding
    embedding = OllamaEmbedding(
        model_name="bge-m3",
        base_url="http://localhost:11434"
    )

    # Colleghiamo ChromaDB
    client = chromadb.PersistentClient(
        path=str(chroma_path)
    )

    collection = client.get_collection(
        name="tesi_rag"
    )

    vector_store = ChromaVectorStore(
        chroma_collection=collection
    )

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embedding
    )

    # Recuperiamo 8 chunk
    retriever = index.as_retriever(
        similarity_top_k=8
    )

    # Reranker
    reranker = CrossEncoder(
        "BAAI/bge-reranker-v2-m3",
        max_length=512
    )

    # Modello LLM
    llm = Ollama(
        model="llama3.2:3b",
        request_timeout=600.0,
        temperature=0
    )

    return retriever, reranker, llm, embedding


def rag_answer(course, question):

    retriever, reranker, llm, _ = get_pipeline(course)

    # Recupero dei documenti
    chunks = retriever.retrieve(question)

    if not chunks:
        return (
            "Non trovo informazioni sufficienti "
            "nel materiale fornito.",
            []
        )

    # Calcolo dei punteggi del reranker
    pairs = [
        [question, item.node.get_content()]
        for item in chunks
    ]

    scores = reranker.predict(pairs)

    # Ordiniamo e manteniamo i primi 5 chunk
    ordered = sorted(
        zip(chunks, scores),
        key=lambda x: float(x[1]),
        reverse=True
    )[:5]

    sources = []
    contexts = []

    for i, (item, score) in enumerate(ordered, 1):

        metadata = item.node.metadata or {}

        file_name = metadata.get(
            "file_name", "Documento non indicato"
        )

        page = metadata.get(
            "source", "Pagina non indicata"
        )

        sources.append({
            "file": str(file_name),
            "page": str(page)
        })

        contexts.append(
            f"Fonte {i}\n"
            f"Documento: {file_name}\n"
            f"Pagina: {page}\n"
            f"{item.node.get_content()}"
        )

    context = "\n\n".join(contexts)

    # Prompt per il modello
    
    prompt = f"""
Sei un assistente didattico universitario.

Rispondi alla domanda dello studente utilizzando
esclusivamente le informazioni presenti nel contesto.

Regole:
- Usa un italiano semplice, chiaro e preciso.
- Rispondi direttamente alla domanda.
- Spiega i concetti in modo completo, ma senza ripetizioni.
- Usa paragrafi brevi quando necessario.
- Se utile, inserisci un piccolo esempio.
- Non aggiungere informazioni non presenti nel contesto.
- Se la domanda contiene una premessa errata, correggila.
- Se il materiale non è sufficiente per rispondere, scrivi:
  "Non trovo informazioni sufficienti nel materiale fornito."
- Non inserire l'elenco delle fonti nella risposta.
  Le fonti verranno mostrate separatamente dal sistema.

CONTESTO:
{context}

DOMANDA:
{question}

RISPOSTA:
"""


    answer = str(llm.complete(prompt)).strip()

    return answer, sources

# PAGINE WEB
@app.route("/")
def login():
    return render_template("login.html")


@app.route("/student")
def student():
    return render_template("student.html")


@app.route("/chat/<course>")
def chat(course):

    if course not in COURSES:
        return "Corso non trovato", 404

    return render_template(
        "chat.html",
        course=course,
        course_name=COURSES[course]
    )


@app.route("/teacher")
def teacher():
    return render_template("teacher.html")

# CRONOLOGIA CONVERSAZIONI (separata dai dati di analisi)
def valid_student(value):
    value = str(value or "").strip()
    return value if 0 < len(value) <= 100 else None


@app.route("/api/conversations", methods=["GET", "POST"])
def conversations_api():
    if request.method == "GET":
        owner = valid_student(request.args.get("student_code"))
        course = request.args.get("course")
        if not owner or course not in COURSES:
            return jsonify(error="Corso e identificativo obbligatori."), 400
        with connect_db() as con:
            rows = con.execute("""
                SELECT id, title, created_at, updated_at
                FROM conversations
                WHERE student_code = ? AND course = ?
                ORDER BY updated_at DESC, created_at DESC
                LIMIT 100
            """, (owner, course)).fetchall()
        return jsonify(conversations=[dict(row) for row in rows])

    body = request.get_json(silent=True) or {}
    owner = valid_student(body.get("student_code"))
    course = body.get("course")
    if not owner or course not in COURSES:
        return jsonify(error="Corso e identificativo obbligatori."), 400
    conversation_id = uuid.uuid4().hex
    with connect_db() as con:
        con.execute(
            "INSERT INTO conversations(id, student_code, course, title) VALUES (?, ?, ?, ?)",
            (conversation_id, owner, course, "Nuova conversazione")
        )
    return jsonify(id=conversation_id, title="Nuova conversazione"), 201


@app.route("/api/conversations/<conversation_id>", methods=["GET", "DELETE"])
def conversation_detail(conversation_id):
    owner = valid_student(request.args.get("student_code"))
    if not owner:
        return jsonify(error="Identificativo obbligatorio."), 400
    with connect_db() as con:
        row = con.execute(
            "SELECT id, title, course FROM conversations WHERE id = ? AND student_code = ?",
            (conversation_id, owner)
        ).fetchone()
        if not row:
            return jsonify(error="Conversazione non trovata."), 404
        if request.method == "DELETE":
            con.execute("DELETE FROM conversation_messages WHERE conversation_id = ?", (conversation_id,))
            con.execute("DELETE FROM conversations WHERE id = ? AND student_code = ?", (conversation_id, owner))
            # Le domande utilizzate per le statistiche restano nella tabella questions.
            return jsonify(ok=True)
        messages = con.execute("""
            SELECT role, content, sources_json, created_at
            FROM conversation_messages WHERE conversation_id = ?
            ORDER BY id
        """, (conversation_id,)).fetchall()
    return jsonify(id=row["id"], title=row["title"], course=row["course"],
                   messages=[{"role": m["role"], "content": m["content"],
                              "sources": json.loads(m["sources_json"] or "[]"),
                              "created_at": m["created_at"]} for m in messages])


# API PER LA CHAT
@app.route("/api/ask", methods=["POST"])
def ask():

    body = request.get_json(silent=True) or {}

    course = body.get("course", "")
    question = str(body.get("question", "")).strip()
    student_code = str(
        body.get("student_code", "")
    ).strip()

    if course not in COURSES or not question or not student_code:
        return jsonify(
            error="Corso, domanda e codice studente sono obbligatori."
        ), 400

    if len(question) > 2000 or len(student_code) > 100:
        return jsonify(
            error="Domanda o codice troppo lungo."
        ), 400

    conversation_id = body.get("conversation_id")
    if conversation_id:
        if not isinstance(conversation_id, str):
            return jsonify(error="Conversazione non valida."), 400
        with connect_db() as con:
            belongs = con.execute("""
                SELECT 1 FROM conversations
                WHERE id = ? AND student_code = ? AND course = ?
            """, (conversation_id, student_code, course)).fetchone()
        if not belongs:
            return jsonify(error="Conversazione non trovata."), 404

    try:
        answer, sources = rag_answer(
            course, question
        )

    except Exception:
        app.logger.exception("Errore del sistema RAG")

        return jsonify(
            error="RAG non disponibile. Controlla Ollama e gli indici."
        ), 503

    # Salviamo la domanda dopo la risposta del RAG
    with connect_db() as con:
        con.execute(
            """
            INSERT INTO questions
            (course, student_code, question)
            VALUES (?, ?, ?)
            """,
            (course, student_code, question)
        )

    if conversation_id:
        with connect_db() as con:
            con.execute("""
                INSERT INTO conversation_messages(conversation_id, role, content)
                VALUES (?, 'user', ?)
            """, (conversation_id, question))
            con.execute("""
                INSERT INTO conversation_messages(conversation_id, role, content, sources_json)
                VALUES (?, 'assistant', ?, ?)
            """, (conversation_id, answer, json.dumps(sources, ensure_ascii=False)))
            con.execute("""
                UPDATE conversations
                SET title = CASE WHEN title = 'Nuova conversazione'
                    THEN ? ELSE title END,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (question[:72] + ("…" if len(question) > 72 else ""), conversation_id))

    return jsonify(
        answer=answer,
        sources=sources,
        conversation_id=conversation_id
    )

# Numero di iscritti: inserito manualmente dal docente perché il login demo
# non conosce gli iscritti effettivi. Non è una misura degli utenti attivi.
@app.route("/api/enrollment/<course>", methods=["GET", "PUT"])
def enrollment_api(course):
    if course not in COURSES:
        return jsonify(error="Corso non trovato"), 404
    if request.method == "PUT":
        body = request.get_json(silent=True) or {}
        value = body.get("enrolled_students")
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 1000000:
            return jsonify(error="Inserisci un numero intero di iscritti maggiore di zero."), 400
        with connect_db() as con:
            con.execute("""INSERT INTO course_enrollment(course,enrolled_students) VALUES(?,?)
                ON CONFLICT(course) DO UPDATE SET enrolled_students=excluded.enrolled_students""", (course, value))
    with connect_db() as con:
        row = con.execute("SELECT enrolled_students FROM course_enrollment WHERE course=?", (course,)).fetchone()
    return jsonify(enrolled_students=row["enrolled_students"] if row else None)

# Upload PDF: copia in locale; l'indicizzazione resta esplicita e separata.
@app.route("/api/course-pdfs/<course>", methods=["GET", "POST"])
def course_pdfs(course):
    if course not in COURSES:
        return jsonify(error="Corso non trovato"), 404
    directory = ROOT / "data" / course / "documents"
    if request.method == "POST":
        if not ROOT.is_dir():
            return jsonify(error="Cartella TESI_RAG non trovata accanto a webapp 3."), 400
        files = request.files.getlist("files")
        if not files:
            return jsonify(error="Seleziona almeno un file PDF."), 400
        payloads = []
        for file in files:
            name = secure_filename(file.filename or "")
            if not name or not name.lower().endswith(".pdf"):
                return jsonify(error="Sono consentiti solo file PDF."), 400
            contents = file.read()
            if not contents.startswith(b"%PDF-"):
                return jsonify(error=f"Il file {name} non sembra un PDF valido."), 400
            payloads.append((name, contents))
        directory.mkdir(parents=True, exist_ok=True)
        uploaded = []
        for name, contents in payloads:
            target = directory / name
            if target.exists():
                target = directory / f"{Path(name).stem}_{uuid.uuid4().hex[:8]}.pdf"
            target.write_bytes(contents)
            uploaded.append(target.name)
        return jsonify(uploaded=uploaded, message="PDF salvati. Per renderli consultabili occorre ricostruire l'indice del corso."), 201
    if not directory.is_dir():
        return jsonify(files=[], available=ROOT.is_dir())
    return jsonify(files=sorted(f.name for f in directory.glob("*.pdf") if f.is_file()), available=True)

# ANALISI SEMANTICA DBSCAN
@app.route("/api/stats/<course>")
def stats(course):

    if course not in COURSES:
        return jsonify(error="Corso non trovato"), 404

    # Recuperiamo le domande del corso
    with connect_db() as con:
        rows = con.execute(
            """
            SELECT student_code, question
            FROM questions
            WHERE course = ?
            ORDER BY id
            """,
            (course,)
        ).fetchall()

    with connect_db() as con:
        enrollment_row = con.execute("SELECT enrolled_students FROM course_enrollment WHERE course=?", (course,)).fetchone()
    enrolled_students = enrollment_row["enrolled_students"] if enrollment_row else None

    students = {
        row["student_code"]
        for row in rows
    }

    active_students = len(students)

    if not rows:
        return jsonify(
            total_questions=0,
            active_students=0,
            enrolled_students=enrolled_students,
            groups=[],
            ungrouped_questions=0
        )

    try:
        import numpy as np
        from sklearn.cluster import DBSCAN

        # Usiamo lo stesso modello di embedding del RAG
        _, _, _, embedding = get_pipeline(course)

        questions = [
            row["question"]
            for row in rows
        ]

        vectors = np.array(
            embedding.get_text_embedding_batch(questions),
            dtype=float
        )

        # Raggruppiamo domande semanticamente simili
        clustering = DBSCAN(
            eps=0.27,
            min_samples=2,
            metric="cosine"
        )

        labels = clustering.fit_predict(vectors)

    except Exception:
        app.logger.exception("Errore durante DBSCAN")

        return jsonify(
            error="Analisi non disponibile. Controlla embedding e DBSCAN."
        ), 503

    # Il nome dell'argomento è un'etichetta descrittiva, non un risultato di DBSCAN.
    # Le regole non modificano né gli embedding né la composizione dei cluster.
    def topic_label(questions, course):
        import re
        combined = " ".join(questions).lower()
        rules = (
            [(r"\b(funzion[ei]|parametr[oi]|argoment[oi]|return|richiam[oa])\b", "Funzioni in C"),
             (r"\b(ricorsion[ei]|ricorsiv[oaie])\b", "Ricorsione"),
             (r"\b(array|vettor[ei]|matric[ei])\b", "Array e matrici"),
             (r"\b(cicl[oi]|while|for|do.while)\b", "Cicli in C"),
             (r"\b(puntator[ei]|pointer)\b", "Puntatori"),
             (r"\b(string[ah e]|char)\b", "Stringhe in C"),
             (r"\b(struct|struttur[ae])\b", "Strutture in C")]
            if course == "progr1" else
            [(r"\b(adapter|adattator[ei])\b", "Pattern Adapter"),
             (r"\b(facade|facciata)\b", "Pattern Facade"),
             (r"\b(observer|publisher|subscriber)\b", "Pattern Observer"),
             (r"\b(include|extend|cas[oi] d.uso|use.case)\b", "Casi d'uso UML"),
             (r"\b(composizion[ei]|aggregazion[ei]|associazion[ei])\b", "Relazioni UML"),
             (r"\b(sprint|scrum|product.owner)\b", "Scrum"),
             (r"\b(sequence|sequenza|messagg[io])\b", "Diagrammi di sequenza")]
        )
        scores = [(sum(bool(re.search(pattern, q.lower())) for q in questions), name)
                  for pattern, name in rules]
        best = max(scores, default=(0, ""), key=lambda item: item[0])
        if best[0] > 0:
            return best[1], "regola"
        first = re.sub(r"^(come|cosa|qual[ei]|quando|perch[eé]|spiega|descrivi|in che modo)\s+", "", questions[0], flags=re.I)
        first = re.sub(r"[?!.]+$", "", first).strip()
        words = first.split()
        return (" ".join(words[:6]).capitalize() + ("…" if len(words) > 6 else "")) or "Domande correlate", "estratto"

    groups = []

    for label in sorted(set(labels)):

        # -1 indica le domande non raggruppate
        if label == -1:
            continue

        members = [
            row
            for row, value in zip(rows, labels)
            if value == label
        ]

        # Studenti distinti nel gruppo
        group_students = {
            row["student_code"]
            for row in members
        }

        percentage = (
            len(group_students) / active_students * 100
        )

        topic, naming_method = topic_label([row["question"] for row in members], course)
        groups.append({
            "title": topic,
            "title_method": naming_method,
            "questions": [
                row["question"]
                for row in members
            ],
            "questions_count": len(members),
            "students_count": len(group_students),
            "percentage": round(percentage, 1)
        })

    # Mostriamo prima i gruppi più frequenti
    groups.sort(
        key=lambda x: (
            -x["students_count"],
            -x["questions_count"]
        )
    )

    ungrouped = sum(
        int(label == -1)
        for label in labels
    )

    return jsonify(
        total_questions=len(rows),
        active_students=active_students,
        groups=groups,
        enrolled_students=enrolled_students,
        ungrouped_questions=ungrouped
    )

# AVVIO
if __name__ == "__main__":
    init_db()
    app.run(debug=False, port=5001)     