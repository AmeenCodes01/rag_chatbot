import os
import re
import time
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import json
import base64
# ------------------------------
# 1️⃣ Setup + Environment
# ------------------------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)
cloud = os.getenv("PINECONE_CLOUD", "aws")
region = os.getenv("PINECONE_REGION", "us-east-1")
spec = ServerlessSpec(cloud=cloud, region=region)

INDEX_NAME = "websouls"
NAMESPACE = "websoulsRAGT1"

embedding_function = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY,
    dimensions=768,
)

MD_FOLDER = "./websouls_scraped_md"   
files = ["https://websouls.com"]

# ------------------------------
# 2️⃣ Helpers
# ------------------------------
def filename_to_url(filename: str) -> str:
    """Convert filename back to real URL"""
    url = filename.replace("https___", "https://").replace("_", ".")
    url = url.replace(".md", "")
    return url

def url_to_filename(url: str) -> str:
    encoded = base64.urlsafe_b64encode(url.encode()).decode()
    return f"{encoded}.md"



# def process_md_file(file_path: str):
#     """Read, clean, and split Markdown into semantically coherent chunks"""
#     with open(file_path, encoding="utf-8") as f:
#         text = f.read()

#     # 1. Split by headings first (keep logical sections intact)
#     sections = re.split(r"(?:^|\n)(#{1,6} .+)", text)
#     merged_sections = []
#     current = ""
#     for part in sections:
#         if part.startswith("#"):
#             if current:
#                 merged_sections.append(current)
#             current = part
#         else:
#             current += part
#     if current:
#         merged_sections.append(current)

#     # 2. Further split long sections with RecursiveCharacterTextSplitter
#     final_chunks = []
#     splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=1500)
#     for sec in merged_sections:
#         sec_clean = clean_chunk_with_ai(sec)
#         split_chunks = splitter.split_text(sec_clean)
#         final_chunks.extend(split_chunks)
#         print(f"\n {docs}")
#     # 3. Convert to LangChain Documents
#     docs = []
#     filename = os.path.basename(file_path)
#     url = filename_to_url(filename)
#     for i, chunk in enumerate(final_chunks):
#         doc = Document(
#             page_content=chunk,
#             metadata={
#                 "source_file": filename,
#                 "url": url,
#                 "chunk_index": i,
#                 "total_chunks": len(final_chunks),
#             }
#         )
#         docs.append(doc)

#     print(f"✅ Processed {filename} → {len(docs)} chunks")
#     return docs

def group_blocks_by_heading(text, max_tokens=6000):
    """
    Split by ### headings, then keep adding blocks until token limit reached.
    Returns a list of groups. Each group is sent to OpenAI separately.
    """

    encoder = tiktoken.encoding_for_model("gpt-4o-mini")

    # Split by ### headings, keep heading with content
    parts = text.split("##")
    blocks = []

    for part in parts:
        cleaned = part.strip()
        if cleaned:
            blocks.append("### " + cleaned)

    groups = []
    current_group = []
    current_token_count = 0

    for block in blocks:
        block_tokens = len(encoder.encode(block))

        # If adding this block exceeds token limit → start a new group
        if current_token_count + block_tokens > max_tokens:
            if current_group:
                groups.append("\n\n".join(current_group))
            current_group = [block]
            current_token_count = block_tokens
        else:
            current_group.append(block)
            current_token_count += block_tokens

    # Add last group
    if current_group:
        groups.append("\n\n".join(current_group))

    return groups



def clean_json_output(raw: str):
    """Cleans LLM output to make it safe for json.loads."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()

    # Extract valid JSON portion
    start = min([i for i in [raw.find("["), raw.find("{")] if i != -1], default=0)
    end = max(raw.rfind("]"), raw.rfind("}"))
    if end != -1:
        raw = raw[start:end + 1]
    return raw



def process_chunk_with_ai(text: str, file_path: str, chunk_id_start: int):
    """
    Sends a chunk to AI to:
    - Clean fluff / repetitive text
    - Split into smaller logical chunks (if needed)
    Returns a list of Document objects.
    """
    prompt = f"""
You are given the full text of a scraped website page.

Your goal is to divide the text into logical chunks for vector embeddings,
while preserving *all important information*.

Guidelines:
1. Detect the main **category or service type** (e.g., "Shared Hosting", "WordPress Hosting", "VPS Hosting", etc.).
2. Identify all **individual plans or sub-sections** (e.g., Basic, Startup, Digital, etc.).
3. For each section or plan:
   - Include **all details** (features, pricing, benefits, comparisons, etc.).
   - Only remove navigation or repetitive footer text.
4. Each JSON object must follow this format:

{{
  "content": "<category> – <plan_name> – <full descriptive text of that plan and its features>",
  "metadata": {{
    "category": "<e.g., Shared Hosting>",
    "plan_name": "<e.g., Digital>",
    "detected_url": "<the URL found in text, or best guess>"
  }}
}}

5. Always return a valid JSON **array** with no markdown formatting.
6. Do not summarize — preserve all descriptive text for embeddings.
7. If you find general content (not plan-specific), include it as a separate chunk with
   `"plan_name": "overview"`.

Text:
{text}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    raw = response.choices[0].message.content
    cleaned = clean_json_output(raw)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"❌ JSON parse error in chunk {chunk_id_start}: {e}")
        return []
    
    print(f"\n{parsed}")

    docs = []
    for idx, obj in enumerate(parsed, start=chunk_id_start):
        obj["metadata"]["chunk_index"] = idx
        obj["metadata"]["total_chunks"] = len(parsed)
        obj["metadata"]["source_file"] = file_path

        doc = Document(
            page_content=obj["content"].strip(),
            metadata=obj["metadata"]
        )
        docs.append(doc)

    print(f"✅ Processed chunk {chunk_id_start} into {len(docs)} sub-chunks.")
    return docs



# ------------------------------
# 3️⃣ Process All Files
# ------------------------------
all_docs = []

for file in files:
    path = os.path.join(MD_FOLDER, url_to_filename( file))

    with open(path, encoding="utf-8") as f:
        text = f.read()

    url = filename_to_url(file)
    print(f"\n📄 Processing {file} (URL: {url})")

    # 1️⃣ Split into token-safe blocks by ### headings
    blocks = group_blocks_by_heading(text, max_tokens=6000)
    print(f"➡ Split into {len(blocks)} blocks based on headings/token limit.")
    for group in blocks:
        print(f"/n{ group}")
        print("----------------------")
    # chunk_id = 1
    # for block in blocks:
    #     # 2️⃣ Send each block to AI to clean fluff & optionally split into sub-chunks
    #     sub_docs = process_chunk_with_ai(block, file_path=file, chunk_id_start=chunk_id)

    #     # 3️⃣ Collect results
    #     all_docs.extend(sub_docs)
    #     chunk_id += len(sub_docs)

    #     # 4️⃣ Rate limit
    #     time.sleep(2)

print(f"\n📄 Total processed documents: {len(all_docs)}")

# # ------------------------------
# # 4️⃣ Pinecone Upload
# # ------------------------------
# existing_indexes = pc.list_indexes().names()
# if INDEX_NAME not in existing_indexes:
#     print(f"🆕 Creating index '{INDEX_NAME}'...")
#     pc.create_index(name=INDEX_NAME, dimension=768, metric="cosine", spec=spec)
# else:
#     print(f"ℹ️ Index '{INDEX_NAME}' already exists.")

# desc = pc.describe_index(name=INDEX_NAME)
# pc_index = pc.Index(host=desc.host)

# # Clear namespace before upload
# try:
#     pc_index.delete(delete_all=True, namespace=NAMESPACE)
#     print(f"✅ Cleared namespace '{NAMESPACE}'")
# except Exception as e:
#     print(f"⚠️ Could not clear namespace: {e}")

# vector_store = PineconeVectorStore(
#     index=pc_index,
#     embedding=embedding_function,
#     namespace=NAMESPACE
# )

# uuids = [str(uuid.uuid4()) for _ in range(len(all_docs))]
# vector_store.add_documents(documents=all_docs, ids=uuids)

# print("✅ Successfully uploaded all documents to Pinecone!")
