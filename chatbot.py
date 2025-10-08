import os
import re
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# Initialize APIs
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "websouls"
namespace = "websoulsRAGT1"
stats = pc.describe_index(index_name)
print(stats)

desc = pc.describe_index(name=index_name)
pc_index = pc.Index(host=desc.host)

# Embeddings (OpenAI hosted)
embedding_function = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=os.getenv("OPENAI_API_KEY"),
    dimensions=768,
)

# Vector store
vector_store = PineconeVectorStore(index=pc_index, embedding=embedding_function)
retriever = vector_store.as_retriever()
print(vector_store,retriever,"retriever")
# LLM (OpenAI remote)
def openai_llm(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    system_prompt = (
        "You are a helpful assistant for Websouls. "
        "You must strictly answer only using the information provided in the 'Context'. "
        "If the answer is not clearly present in the context, respond with: "
        "'I'm sorry, but I couldn’t find that information in Websouls' documentation.' "
        "Do not make up or guess information."
    )
    response = client.chat.completions.create(
        model="gpt-4o",  # or "gpt-4o-mini" if you want cheaper calls
        
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": formatted_prompt}
        ]
    )

    response_content = response.choices[0].message.content
    final_answer = re.sub(r"<think>.*?</think>", "", response_content, flags=re.DOTALL).strip()
    return final_answer

def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def rag_chain(question):
    retrieved_docs = retriever.invoke(question)
    formatted_content = combine_docs(retrieved_docs)
    if not formatted_content.strip():
        print( "⚠️ No relevant docs found in Pinecone.", question)
    return openai_llm(question, formatted_content)

def answer_fn(message, history):
    return rag_chain(question=message)

demo = gr.ChatInterface(
    fn=answer_fn,
    title="Q/A Bot",
    description="Ask me anything!",
    examples=["What is Python?", "Explain recursion", "What are websouls hosting plans?"]
)

demo.launch()
