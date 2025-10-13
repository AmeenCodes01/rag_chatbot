import os
import re
import gradio as gr
#from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
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
NAMESPACE = "websoulsRAGT1"  # ⚠️ MUST match your upload namespace

# Get index
desc = pc.describe_index(name=INDEX_NAME)
pc_index = pc.Index(host=desc.host)

# Initialize embeddings - MUST match your upload config
embedding_function = OpenAIEmbeddings(
    model="text-embedding-3-large",
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

# print(vector_store.similarity_search(query="What is websouls"))
# print(retriever.invoke("What is websouls"))


# # Initialize LLM
# llm = ChatOpenAI(
#     openai_api_key=os.getenv('OPENAI_API_KEY'),
#     model_name='gpt-4o-mini',
#     temperature=0.0
# )

# # Create chains
# retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
# combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
# retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

# # Test query
# result = retrieval_chain.invoke({"input": "What services do websouls offer"})
# print("---------")
# print(result)



# print("=" * 60)
# print("DEBUGGING PINECONE METADATA STRUCTURE")
# print("=" * 60)

# # Test 1: Direct Pinecone query
# print("\n1️⃣ DIRECT PINECONE QUERY:")
# query_vector = embedding_function.embed_query("What services do websouls offer")
# results = pc_index.query(
#     vector=query_vector,
#     top_k=2,
#     namespace=NAMESPACE,
#     include_metadata=True
# )

# for i, match in enumerate(results['matches']):
#     print(f"\n--- Match {i+1} ---")
#     print(f"ID: {match['id']}")
#     print(f"Score: {match['score']}")
#     print(f"Metadata keys: {list(match.get('metadata', {}).keys())}")
#     print(f"Metadata structure:")
#     for key, value in match.get('metadata', {}).items():
#         value_preview = str(value)[:100] if value else "None"
#         print(f"  - {key}: {value_preview}...")

# # Test 2: Try different text_key values
# print("\n" + "=" * 60)
# print("2️⃣ TESTING DIFFERENT text_key VALUES:")
# print("=" * 60)

# test_keys = [None, "text", "page_content", "content"]

# for text_key in test_keys:
#     print(f"\n--- Testing text_key='{text_key}' ---")
#     try:
#         if text_key is None:
#             vector_store = PineconeVectorStore(
#                 index=pc_index,
#                 embedding=embedding_function,
#                 namespace=NAMESPACE
#             )
#         else:
#             vector_store = PineconeVectorStore(
#                 index=pc_index,
#                 embedding=embedding_function,
#                 namespace=NAMESPACE,
#                 text_key=text_key
#             )
        
#         retriever = vector_store.as_retriever(search_kwargs={"k": 2})
#         docs = retriever.invoke("What services do websouls offer")
        
#         print(f"✅ Retrieved {len(docs)} documents")
#         if len(docs) > 0:
#             print(f"First doc preview: {docs[0].page_content[:150]}...")
#             print(f"First doc metadata keys: {list(docs[0].metadata.keys())}")
#         else:
#             print("❌ No documents retrieved")
#     except Exception as e:
#         print(f"❌ Error: {e}")

# # Test 3: Try similarity search directly
# print("\n" + "=" * 60)
# print("3️⃣ TESTING SIMILARITY_SEARCH METHOD:")
# print("=" * 60)

# vector_store = PineconeVectorStore(
#     index=pc_index,
#     embedding=embedding_function,
#     namespace=NAMESPACE,
#     text_key="text"
# )

# try:
#     docs = vector_store.similarity_search("What services do websouls offer", k=2)
#     print(f"similarity_search returned: {len(docs)} documents")
#     if len(docs) > 0:
#         print(f"First doc content preview: {docs[0].page_content[:150]}...")
# except Exception as e:
#     print(f"Error: {e}")

# # Test 4: Check if it's a LangChain version issue
# print("\n" + "=" * 60)
# print("4️⃣ VERSION INFO:")
# print("=" * 60)

# import langchain_pinecone
# import langchain_openai
# import langchain

# print(f"langchain_pinecone version: {langchain_pinecone.__version__}")
# print(f"langchain_openai version: {langchain_openai.__version__}")
# print(f"langchain version: {langchain.__version__}")



# # Check index stats
# stats = pc_index.describe_index_stats()
# print(f"Total vectors: {stats['total_vector_count']}")
# print(f"Namespaces: {stats.get('namespaces', {})}")

# # Direct query test
# query_vector = embedding_function.embed_query("What services do websouls offer")
# results = pc_index.query(
#     vector=query_vector,
#     top_k=5,
#     namespace=NAMESPACE,  # Make sure namespace is specified
#     include_metadata=True
# )
# print(f"Found {len(results['matches'])} matches")
# for match in results['matches']:
#     print(f"Score: {match['score']}")
#     print(f"Metadata: {match.get('metadata', {})}")
    
# # Test retriever
# docs = retriever.get_relevant_documents("What services do websouls offer")
# print(f"Retrieved {len(docs)} documents")
# for i, doc in enumerate(docs):
#     print(f"\nDoc {i+1}:")
#     print(doc.page_content[:200])  # First 200 chars
system_prompt = (
    "You are a helpful marketing assistant for Websouls. "
    "Your task is to present hosting plans and their features in a structured, promotional tone. "
    "You must NEVER omit or skip any information from the 'Context'. "
    "Every sentence, bullet point, and benefit listed in the context counts as a valid feature — even if it looks like descriptive or promotional text (for example, 'Powerful AI Website Builder allows you to create a professional website with an AI content generator'). "
    "List each plan exactly as described, with all its features preserved word-for-word whenever possible. "
    "If the context only shows one plan, display only that plan. If multiple plans appear, show all individually with their own titles, prices, and features. "
    "You may clean up formatting (e.g., add bullet points or bold titles), but you must not summarize, merge, or remove any features or benefits. "
    "Ignore irrelevant or navigational text like 'Click here', 'Register', or similar. "
    "When asked for a recommendation (like 'best plan' or 'most popular'), choose the plan explicitly labeled as 'recommended', 'most popular', or highest-tier in the context. "
    "If no such label exists, state that all plans are good options depending on user needs. "
    "Do not invent or assume anything not clearly stated in the context."
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
        messages=[
                  {"role": "system", "content": system_prompt},
                  {"role": "user", "content": f"Question: {question}\n\n{context}"}
                  
                  ]
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
    top_k=15,
    namespace=NAMESPACE,
    include_metadata=True
)
    
    res = ""
    for r in results.get("matches",[]):
        res+= format_chunk(r) 
    print( "_------")
    print(res,"res fr ques: ",question)
    print("_---------")
    # formatted_content = combine_docs(results)
    # if not formatted_content.strip():
    #     print( "⚠️ No relevant docs found in Pinecone.", question)
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

demo.launch(share=True)

