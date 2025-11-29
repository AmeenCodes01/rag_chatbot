import os
import re
import gradio as gr
from langchain_pinecone import PineconeVectorStore
from langchain_openai.embeddings import OpenAIEmbeddings
from pinecone import Pinecone
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print(PINECONE_API_KEY)
# Initialize Pinecone
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
INDEX_NAME = "websouls"
NAMESPACE = "websoulsRAGT2"  # ⚠️ MUST match your upload namespace

# Get index
desc = pc.describe_index(name=INDEX_NAME)
pc_index = pc.Index(host=desc.host)

# Initialize embeddings - MUST match your upload config
embedding_function = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv('OPENAI_API_KEY'),
    dimensions=768,
)

# Create vector store with NAMESPACE
vector_store = PineconeVectorStore(
    index=pc_index, 
    embedding=embedding_function,
    namespace=NAMESPACE,  # 🔥 This is likely what's missing!
)

client = OpenAI(api_key=OPENAI_API_KEY)


# Create retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}  # Return top 5 results
)

system_prompt = (
    "You are a friendly and knowledgeable marketing assistant for Websouls, "
    "trained to help users find the best hosting plans for their needs. "
    "You must always stay helpful, conversational, and professional — never robotic or repetitive. "

    "You have access to Websouls' hosting plans and their detailed features, called 'Context'. "
    "Each feature, sentence, or benefit mentioned in the Context is accurate and must be preserved when describing plans. "
    "You can clean up the text, use natural language, and make it sound appealing — but never invent, exaggerate, or remove any real information. "

    "When the user asks directly about 'hosting', 'plans', or anything related to features, pricing, or recommendations: "
    "- If they ask about a specific type of hosting (e.g., WordPress, Business, VPS), only show that relevant plan or category. "
    "- If multiple plans are mentioned in the Context, list them separately with clear headings, prices, and features. "
    "- If they describe their needs (e.g., small business, student project, eCommerce), recommend the single most suitable plan and briefly explain why. "
    "- If the Context marks a plan as 'recommended', 'most popular', or 'best', prefer that plan when unsure. "

    "If the user asks about something unrelated to hosting, respond normally like a human assistant — do not show any hosting plans unless they specifically ask or mention related terms. "

    "When describing plans, you may use structured formatting such as:\n"
    "- **Plan Name**\n"
    "- Price\n"
    "- Key Features (bulleted)\n"
    "- Short benefit or recommendation line (optional)\n\n"

    "Always sound like a real assistant — helpful, engaging, and context-aware. "
    "Never dump all plans unless the user requests to compare them or says 'show all plans'. "
)

h_messages = [{"role":"system","content":system_prompt}]

def openai_llm(question, context):
    global h_messages
    # Add current question
    if "h_messages" not in globals() or not h_messages:
        h_messages = [{"role": "system", "content": system_prompt}]
    
    # Add new user message
    h_messages.append({"role": "user", "content": f"Question: {question}\n\n{context}"})

    # Keep only the system + last 6 messages (3 exchanges)
    if len(h_messages) > 7:  # 1 system + (3 user + 3 assistant)
        h_messages = [h_messages[0]] + h_messages[-6:]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=h_messages
    )
   # print("h_messages: ",h_messages)
    print("------------------")
    response_content = response.choices[0].message.content.strip()
    final_answer = re.sub(r"<think>.*?</think>", "", response_content, flags=re.DOTALL).strip()

    # Add assistant reply to conversation
    h_messages.append({"role": "assistant", "content": final_answer})
    return final_answer

def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def format_chunk(chunk):
    source = chunk['metadata'].get('source', 'Unknown source')
    text = chunk['metadata'].get('text', '')
    
    formatted = f"""-------
Source: {source}

{text}
"""
    return formatted



def rag_chain(question):
    query_vector = embedding_function.embed_query(question)
    results = pc_index.query(
        vector=query_vector,
        top_k=4,
        namespace=NAMESPACE,
        include_metadata=True
    )
    
    res = ""
    for r in results.get("matches", []):
        res += format_chunk(r)

    print("Retrieved context:\n", res)

    return openai_llm(question, res)


def answer_fn(message, history):
    
    return rag_chain(question=message)

demo = gr.ChatInterface(
    fn=answer_fn,
    title="Q/A Bot",
    type="messages",
    description="Ask me anything!",
    examples=[ "What are websouls hosting plans?"]
    
)

demo.launch(share=False)
