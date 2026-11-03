
// LOGIN DIMOSTRATIVO
function loginDemo() {

    const role = document.getElementById("role").value;

    const email = document.getElementById("email")
        .value.trim().toLowerCase();

    if (!email) {
        alert("Inserisci un identificativo studente demo.");
        return;
    }

    // Salviamo l'identificativo nel browser
    localStorage.setItem("student_code", email);

    if (role === "teacher") {
        window.location.href = "/teacher";
    } else {
        window.location.href = "/student";
    }
}

// CHAT STUDENTE: cronologia persistente nel database, separata dalla dashboard
let currentConversationId = null;
let sendingQuestion = false;
const studentCode = () => localStorage.getItem("student_code");

function appendChatMessage(text, role, sources = []) {
    const container = document.getElementById("messages");
    if (!container) return;
    const welcome = container.querySelector(".welcome-chat");
    if (welcome) welcome.remove();
    const row = document.createElement("div");
    row.className = "chat-message-row " + (role === "user" ? "from-user" : "from-assistant");
    const avatar = document.createElement("span");
    avatar.className = "chat-avatar";
    avatar.textContent = role === "user" ? "Tu" : "✦";
    const bubble = document.createElement("div");
    bubble.className = "chat-bubble " + role;
    bubble.textContent = text;
    row.append(avatar, bubble);
    container.appendChild(row);
    if (role === "assistant" && sources.length) {
        const sourcePanel = document.createElement("div");
        sourcePanel.className = "chat-sources";
        const heading = document.createElement("strong");
        heading.textContent = "Fonti recuperate";
        sourcePanel.appendChild(heading);
        const unique = new Set();
        sources.forEach(source => {
            const label = `${source.file} · pagina ${source.page}`;
            if (unique.has(label)) return;
            unique.add(label);
            const line = document.createElement("div");
            line.textContent = label;
            sourcePanel.appendChild(line);
        });
        container.appendChild(sourcePanel);
    }
    container.scrollTop = container.scrollHeight;
}

function emptyChat() {
    const container = document.getElementById("messages");
    container.replaceChildren();
    const box = document.createElement("div");
    box.className = "welcome-chat";
    const mark = document.createElement("div"); mark.className = "welcome-symbol"; mark.textContent = "✦";
    const title = document.createElement("h2"); title.textContent = "Come posso aiutarti?";
    const text = document.createElement("p"); text.textContent = "Scrivi una domanda sul corso o prova uno dei suggerimenti.";
    box.append(mark, title, text);
    container.appendChild(box);
}

async function fetchJson(url, options = undefined) {
    const response = await fetch(url, options);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Operazione non riuscita");
    return data;
}

async function loadConversations() {
    const container = document.getElementById("conversationList");
    if (!container) return;
    if (!studentCode()) { container.textContent = "Effettua l'accesso per salvare la cronologia."; return; }
    try {
        const data = await fetchJson(`/api/conversations?course=${encodeURIComponent(COURSE)}&student_code=${encodeURIComponent(studentCode())}`);
        container.replaceChildren();
        if (!data.conversations.length) {
            const p = document.createElement("p"); p.className = "sidebar-placeholder";
            p.textContent = "Nessuna conversazione salvata."; container.appendChild(p);
        }
        data.conversations.forEach(item => {
            const group = document.createElement("div"); group.className = "conversation-entry" + (item.id === currentConversationId ? " selected" : "");
            const open = document.createElement("button"); open.type = "button"; open.className = "conversation-open";
            open.textContent = item.title; open.title = item.title;
            open.onclick = () => openConversation(item.id);
            const remove = document.createElement("button"); remove.type = "button"; remove.className = "conversation-delete";
            remove.textContent = "×"; remove.title = "Elimina conversazione";
            remove.onclick = () => deleteConversation(item.id);
            group.append(open, remove); container.appendChild(group);
        });
    } catch (error) { container.textContent = "Cronologia non disponibile: " + error.message; }
}

async function openConversation(id) {
    if (sendingQuestion) return;
    try {
        const data = await fetchJson(`/api/conversations/${encodeURIComponent(id)}?student_code=${encodeURIComponent(studentCode())}`);
        if (data.course !== COURSE) throw new Error("Conversazione di un altro corso.");
        currentConversationId = id;
        document.getElementById("messages").replaceChildren();
        if (!data.messages.length) emptyChat();
        data.messages.forEach(m => appendChatMessage(m.content, m.role === "user" ? "user" : "assistant", m.sources || []));
        await loadConversations();
    } catch (error) { alert(error.message); }
}

function newConversation() {
    if (sendingQuestion) return;
    currentConversationId = null;
    emptyChat();
    document.getElementById("question").value = "";
    loadConversations();
}

async function deleteConversation(id) {
    if (sendingQuestion) return;
    if (!confirm("Eliminare questa conversazione dalla cronologia? Le statistiche aggregate delle domande non saranno modificate.")) return;
    try {
        await fetchJson(`/api/conversations/${encodeURIComponent(id)}?student_code=${encodeURIComponent(studentCode())}`, {method: "DELETE"});
        if (currentConversationId === id) newConversation();
        await loadConversations();
    } catch (error) { alert(error.message); }
}

async function sendQuestion() {
    const input = document.getElementById("question");
    const button = document.getElementById("sendButton");
    if (!input || sendingQuestion) return;
    const question = input.value.trim();
    const owner = studentCode();
    if (!question) return;
    if (!owner) { alert("Torna alla pagina iniziale e inserisci un identificativo studente demo."); return; }
    sendingQuestion = true;
    button.disabled = true;
    input.disabled = true;
    let pending = null;
    try {
        if (!currentConversationId) {
            const created = await fetchJson("/api/conversations", {
                method: "POST", headers: {"Content-Type":"application/json"},
                body: JSON.stringify({course: COURSE, student_code: owner})
            });
            currentConversationId = created.id;
        }
        input.value = "";
        appendChatMessage(question, "user");
        pending = document.createElement("p");
        pending.className = "chat-loading"; pending.textContent = "L'assistente sta preparando la risposta…";
        document.getElementById("messages").appendChild(pending);
        const result = await fetchJson("/api/ask", {
            method: "POST", headers: {"Content-Type":"application/json"},
            body: JSON.stringify({course: COURSE, question, student_code: owner, conversation_id: currentConversationId})
        });
        pending.remove(); pending = null;
        appendChatMessage(result.answer, "assistant", result.sources || []);
        await loadConversations();
    } catch (error) {
        if (pending) pending.remove();
        // Il server salva la domanda solo se RAG è riuscito: rimetterla nel campo
        input.value = question;
        appendChatMessage("Errore: " + error.message, "assistant");
    } finally {
        sendingQuestion = false; button.disabled = false; input.disabled = false; input.focus();
    }
}

if (document.getElementById("chatForm")) {
    document.getElementById("chatForm").addEventListener("submit", event => {
        event.preventDefault(); sendQuestion();
    });
    document.getElementById("question").addEventListener("keydown", event => {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault(); sendQuestion();
        }
    });
    document.getElementById("newConversation").addEventListener("click", newConversation);
    document.querySelectorAll("[data-question]").forEach(button => {
        button.addEventListener("click", () => {
            document.getElementById("question").value = button.dataset.question;
            document.getElementById("question").focus();
        });
    });
    loadConversations();
}

// Dashboard: esempio isolato dai dati reali e vista reale selezionabile.
let teacherLastData = null;
let teacherEnrolled = null;
let teacherDemoMode = true;
const teacherExamples = {
  enrolled_students: 100, active_students: 34, ungrouped_questions: 0,
  groups: [
    {title: "Funzioni in C", title_method: "esempio", students_count: 12, questions_count: 15, questions: [
      "Come si definisce e richiama una funzione in C?",
      "Qual è la differenza tra parametri formali e parametri attuali?",
      "Come si restituisce un valore tramite return?",
      "Cosa succede se una funzione viene dichiarata void?",
      "Come si passa un array come parametro a una funzione?",
      "Che differenza c’è tra dichiarazione e definizione di una funzione?",
      "A cosa serve il prototipo di una funzione?",
      "Come si passano i parametri per valore?",
      "È possibile richiamare una funzione dentro un’altra funzione?",
      "Cosa significa il tipo di ritorno int?",
      "Come si utilizza una funzione senza parametri?",
      "Dove va dichiarata una funzione prima del main?",
      "Perché il compilatore segnala un errore nella chiamata a una funzione?",
      "Come si scrive una funzione che restituisce il massimo fra due numeri?",
      "Che differenza c’è fra variabili locali e globali in una funzione?"
    ]},
    {title: "Array e matrici", title_method: "esempio", students_count: 9, questions_count: 12, questions: [
      "Come si dichiara un array di interi in C?",
      "Come si accede all'elemento di una matrice bidimensionale?",
      "Qual è la differenza tra array e matrice?",
      "Come posso attraversare una matrice usando due cicli for?",
      "Come si inizializzano tutti gli elementi di un array?",
      "Perché gli indici degli array iniziano da zero?",
      "Come si determina la lunghezza di un array in C?",
      "Come si passa un array a una funzione?",
      "Come si sommano gli elementi di un array?",
      "Come si legge una matrice da tastiera?",
      "Come si calcola la diagonale principale di una matrice?",
      "Cosa succede se accedo a un indice fuori dai limiti?"
    ]},
    {title: "Ricorsione", title_method: "esempio", students_count: 5, questions_count: 8, questions: [
      "Che cos'è il caso base di una funzione ricorsiva?",
      "Come funziona una funzione ricorsiva che calcola il fattoriale?",
      "Quando è preferibile usare la ricorsione rispetto a un ciclo?",
      "Come si interrompono le chiamate ricorsive?",
      "Come si implementa Fibonacci con la ricorsione?",
      "Perché una funzione ricorsiva può causare stack overflow?",
      "Qual è la differenza fra ricorsione diretta e indiretta?",
      "Come si calcola una potenza con una funzione ricorsiva?"
    ]}
  ]
};
function initTeacherDashboard() {
    const cards = document.getElementById("topicCards");
    if (!cards) return;
    // Vista di presentazione: i tre gruppi sono simulati e non modificano il database.
    teacherDemoMode = true;
    teacherLastData = teacherExamples;
    teacherEnrolled = 100;
    document.getElementById("clusters").textContent = String(teacherExamples.groups.length);
    renderTeacherTopics(teacherExamples);
}
function renderTeacherTopics(data) {
    const container = document.getElementById("topicCards");
    container.replaceChildren();
    if (!data.groups?.length) {
        const empty = document.createElement("p");
        empty.className = "topics-empty";
        empty.textContent = "Non sono ancora stati individuati gruppi di almeno due domande simili.";
        container.append(empty);
    }
    if (!teacherDemoMode && teacherEnrolled && data.active_students > teacherEnrolled) {
        const warn = document.getElementById("enrollmentMessage");
        warn.textContent = "Attenzione: gli utenti attivi rilevati superano gli iscritti inseriti. Verifica il totale; i grafici restano disattivati.";
    } else if (!teacherDemoMode && !teacherEnrolled) {
        document.getElementById("enrollmentMessage").textContent = "Inserisci il numero totale degli iscritti per visualizzare percentuali corrette.";
    }
    data.groups.forEach((g, i) => {
        const box = document.createElement("details");
        box.className = "teacher-topic";
        const head = document.createElement("summary");
        const label = document.createElement("span"); label.className = "teacher-topic-name";label.textContent=g.title;
        const count = document.createElement("span"); count.className="teacher-topic-count";
        count.textContent = `${g.questions_count} domande · ${g.students_count} studenti`;
        const chevron = document.createElement("span");chevron.className="topic-chevron";chevron.textContent="⌄";
        head.append(label,count,chevron);
        const body = document.createElement("div");body.className="teacher-topic-body";
        const ratioGood = teacherEnrolled && data.active_students <= teacherEnrolled && g.students_count <= teacherEnrolled;
        const ratio = ratioGood ? Math.min(g.students_count / teacherEnrolled * 100,100) : null;
        const share = document.createElement("div");share.className="topic-share";
        const pie = document.createElement("div");pie.className="topic-pie";
        if (ratio !== null) pie.style.setProperty("--pie",ratio+"%"); else pie.classList.add("topic-pie-unknown");
        pie.setAttribute("role","img");pie.setAttribute("aria-label",ratio!==null?`${ratio.toFixed(1)} per cento degli iscritti` : "Percentuale non disponibile");
        const center = document.createElement("span");center.textContent=ratio!==null?ratio.toFixed(1).replace(".",",")+"%":"—";pie.append(center);
        const shareText = document.createElement("div");
        const h = document.createElement("strong");h.textContent="Studenti iscritti che hanno chiesto chiarimenti";
        const p = document.createElement("p");p.textContent = ratio !== null ? `${g.students_count} su ${teacherEnrolled} iscritti hanno fatto almeno una domanda in questo gruppo.` : "Imposta il totale degli iscritti per calcolare la percentuale. Il grafico non utilizza il numero degli utenti attivi.";
        shareText.append(h,p);share.append(pie,shareText);body.append(share);
        const questions = document.createElement("details");questions.className="teacher-question-list";
        const qs = document.createElement("summary"); qs.textContent=`Mostra le ${g.questions_count} domande associate`;
        const ol = document.createElement("ol");
        (g.questions||[]).forEach(q => {const li=document.createElement("li");li.textContent=q;ol.append(li);});
        questions.append(qs,ol);body.append(questions);
        if(g.title_method==="estratto") {const info=document.createElement("p");info.className="topic-method-note";info.textContent="Etichetta orientativa ricavata dal testo di una domanda; il gruppo è formato automaticamente da DBSCAN.";body.append(info);}
        box.append(head,body);container.append(box);
    });
    const loose = data.ungrouped_questions || 0;
    const ungrouped = document.getElementById("ungroupedSummary");
    if (ungrouped) {
        ungrouped.hidden = !loose;
        if (loose) ungrouped.textContent = `${loose} domande non sono state associate a un gruppo; DBSCAN le considera elementi isolati.`;
    }
}
