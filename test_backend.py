from flask import Flask, request, jsonify
from flask_cors import CORS
import time
from openai import OpenAI
import os
import sqlite3
from datetime import datetime
import re

from langchain_pinecone import PineconeVectorStore
from langchain_openai.embeddings import OpenAIEmbeddings
from pinecone import Pinecone
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

app = Flask(__name__)
CORS(app)

# ------------------------
# SYSTEM PROMPT
# ------------------------
system_prompt = """You are Websouls’ friendly marketing assistant. Help users choose the best hosting plan using only the accurate info provided in “Context”. You may rephrase for clarity and tone, but never invent, remove, or change any real features or pricing.

WHEN USER ASKS ABOUT HOSTING (plans, pricing, features, recommendations):
- If they ask about a category (WordPress, VPS, Business, etc.), show only that category.
- If multiple relevant plans exist, list them clearly with headings, price, and bullets.
- If the user describes their needs, recommend the single best plan and explain briefly.
- Prefer plans marked “recommended”, “most popular”, or “best”.
- Do NOT show all plans unless the user explicitly asks to compare or says “show all plans”.

WHEN USER ASKS ABOUT NON-HOSTING:
Answer normally. Do NOT mention hosting unless they do.

FORMAT FOR PLANS:
- **Plan Name**
- Price
- Bulleted features
- Optional short benefit line

Always be human, natural, helpful, and context-aware."""
# ------------------------
# SQLITE SETUP
# ------------------------
DB_PATH = "chatlogs.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            question TEXT,
            answer TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_log(session_id, question, answer):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO logs (session_id, question, answer, timestamp)
        VALUES (?, ?, ?, ?)
    """, (session_id, question, answer, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

init_db()

# ------------------------
# ENV + OPENAI
# ------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
print(PINECONE_API_KEY)

# Pinecone
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
INDEX_NAME = "websouls"
NAMESPACE = "websoulsRAGT2"

desc = pc.describe_index(name=INDEX_NAME)
pc_index = pc.Index(host=desc.host)

# Embeddings
embedding_function = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=os.getenv('OPENAI_API_KEY'),
    dimensions=768,
)

vector_store = PineconeVectorStore(
    index=pc_index,
    embedding=embedding_function,
    namespace=NAMESPACE,
)

# ------------------------
# SESSION STORE
# ------------------------
class ChatSession:
    def __init__(self):
        self.history = []
        self.last_used = time.time()

    def add(self, role, content):
        self.history.append({"role": role, "content": content})
        self.last_used = time.time()

    def trim(self, max_messages=10):
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

SESSIONS = {}
SESSION_EXPIRY = 3 * 3600

def cleanup_sessions():
    now = time.time()
    expired = [sid for sid, s in SESSIONS.items() if now - s.last_used > SESSION_EXPIRY]
    for sid in expired:
        del SESSIONS[sid]

def format_chunk(chunk):
    source = chunk['metadata'].get('source', 'Unknown source')
    text = chunk['metadata'].get('text', '')
    return f"-------\nSource: {source}\n\n{text}\n"

# ------------------------
# RAG RETRIEVAL
# ------------------------
def rag_chain(question):
    query_vector = embedding_function.embed_query(question)
    results = pc_index.query(
        vector=query_vector,
        top_k=3,
        namespace=NAMESPACE,
        include_metadata=True
    )
    context = ""
    for r in results.get("matches", []):
        context += format_chunk(r)

    return context

# ------------------------
# OPENAI CALL — WITH SYSTEM PROMPT
# ------------------------
def call_openai(history, context, question):

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": f"Context: {context}"}
    ] + history + [
        {"role": "user", "content": question}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7
    )

    response_content = response.choices[0].message.content.strip()
    final_answer = re.sub(
        r"<think>.*?</think>", 
        "", 
        response_content, 
        flags=re.DOTALL
    ).strip()

    return final_answer

# ------------------------
# /chat ENDPOINT
# ------------------------
@app.post("/chat")
def chat():
    cleanup_sessions()

    data = request.json
    session_id = data.get("session_id")
    message = data.get("message")

    if not session_id:
        return jsonify({"error": "session_id is required"}), 400
    if not message:
        return jsonify({"error": "message is required"}), 400

    if session_id not in SESSIONS:
        SESSIONS[session_id] = ChatSession()

    session = SESSIONS[session_id]
    session.add("user", message)
    session.trim()

    retrieved_context = rag_chain(message)
    answer = call_openai(session.history, retrieved_context, message)

    session.add("assistant", answer)

    save_log(session_id, message, answer)

    return jsonify({
        "response": answer,
        "session_id": session_id
    })

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
