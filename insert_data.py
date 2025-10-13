import os
import uuid
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
import re
from openai import OpenAI
import json, time
#from textsplitter import LLMTextSplitter


# ------------------------------
# 1️⃣  Setup + Environment
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

# ------------------------------
# 2️⃣  Embedding + Chunking Setup
# ------------------------------
embedding_function = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=OPENAI_API_KEY,
    dimensions=768,
)

#text_splitter = LLMTextSplitter(count_tokens=True, prompt_type='wide')
# ------------------------------
# 3️⃣  Markdown Files
# ------------------------------
MD_FOLDER = "websouls_scraped_md"

urls = [
    "https://websouls.com.md",
    "https://websouls.com/website-security-management.md",
    "https://websouls.com/online-store-management.md",
    "https://websouls.com/360-degree-digital-marketing.md",
    "https://websouls.com/custom-software-development.md",
    "https://websouls.com/laravel-custom-development.md",
    "https://websouls.com/react-custom-development.md",
    "https://websouls.com/shopify-development.md",
    "https://websouls.com/wordpress-development.md",
    "https://websouls.com/social-media-marketing.md",
    "https://websouls.com/google-ads.md",
    "https://websouls.com/content-writing.md",
    "https://websouls.com/web-hosting-with-domain.md",
    "https://websouls.com/seo-services.md",
    "https://websouls.com/contactus.md",
    "https://websouls.com/about.md",
    "https://websouls.com/team.md",
    # "https://websouls.com/shared-hosting.md",
    "https://websouls.com/domain-transfer.md",
    "https://websouls.com/ecommerce-solution.md",
    "https://websouls.com/policy.md",
    "https://websouls.com/buy-pk-domain.md",
    "https://websouls.com/ssl-certificates.md",
    "https://websouls.com/pk-vps.md",
    "https://websouls.com/vps-hosting.md",
    "https://websouls.com/wordpress-hosting-in-pakistan.md",
    "https://websouls.com/reseller-hosting.md",
    "https://websouls.com/buy-ae-domains.md",
    "https://websouls.com/whyus.md",
    "https://websouls.com/dedicated-server.md",
    "https://websouls.com/privacy.md",
    "https://websouls.com/web-development.md",
    "https://websouls.com/domain-registration.md",
    "https://websouls.com/payment-methods.md"
]

all_docs = []

# ------------------------------
# 4️⃣  Helper: URL → Filename
# ------------------------------
def url_to_filename(url: str) -> str:
    url_no_protocol = re.sub(r'^https?://', '', url)
    base, ext = os.path.splitext(url_no_protocol)
    safe_base = re.sub(r'[^a-zA-Z0-9]', '_', base)
    return f"https___{safe_base}{ext}"

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


def process_docs_from_ai(file_path: str):
    """Reads a .md file, sends it to the AI, gets chunks & inferred URL."""
    with open(file_path, encoding="utf-8") as f:
        text = f.read()

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

    raw_output = response.choices[0].message.content
    raw_cleaned = clean_json_output(raw_output)

    try:
        parsed = json.loads(raw_cleaned)
    except json.JSONDecodeError as e:
        print(f"❌ JSON parse failed for {file_path}: {e}")
        return []

    print(raw_output)
    docs = []
    for idx, item in enumerate(parsed, start=1):
        metadata = item.get("metadata", {})
        metadata["chunk_index"] = idx
        metadata["total_chunks"] = len(parsed)

        doc = Document(page_content=item["content"].strip(), metadata=metadata)
        docs.append(doc)

    print(f"✅ Processed {file_path} into {len(docs)} chunks.")
    return docs



for url in urls:
    if url.endswith(".md"):
        path = os.path.join(MD_FOLDER, url_to_filename(url))
        docs = process_docs_from_ai(path)
        all_docs.extend(docs)
        time.sleep(20)
# for file_name in os.listdir(MD_FOLDER):
#     if file_name.endswith(".md"):
#         path = os.path.join(MD_FOLDER, file_name)
#         docs = process_docs_from_ai(path)
#         all_docs.extend(docs)
print(f"\n📄 Total processed documents: {len(all_docs)}")


# # ------------------------------
# # 6️⃣  Create/Connect to Pinecone Index
# # ------------------------------
existing_indexes = pc.list_indexes().names()
if INDEX_NAME not in existing_indexes:
    print(f"🆕 Creating index '{INDEX_NAME}'...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,
        metric="cosine",
        spec=spec
    )
else:
    print(f"ℹ️ Index '{INDEX_NAME}' already exists.")


desc = pc.describe_index(name=INDEX_NAME)
pc_index = pc.Index(host=desc.host)

# ------------------------------
# 7️⃣  Delete old namespace data (optional but recommended)
# ------------------------------
print(f"\n🗑️ Clearing existing data in namespace '{NAMESPACE}'...")
try:
    pc_index.delete(delete_all=True, namespace=NAMESPACE)
    print(f"✅ Cleared namespace '{NAMESPACE}'")
except Exception as e:
    print(f"⚠️ Could not clear namespace: {e}")

import time
time.sleep(2)  # Wait for delete to complete

# ------------------------------
# 8️⃣  Upload to Pinecone with proper configuration
# ------------------------------
print(f"\n📤 Uploading {len(all_docs)} documents to Pinecone...")

# 🔥 KEY FIX: Initialize PineconeVectorStore properly
vector_store = PineconeVectorStore(
    index=pc_index,
    embedding=embedding_function,
    namespace=NAMESPACE
)

# Generate UUIDs
uuids = [str(uuid.uuid4()) for _ in range(len(all_docs))]

# Upload documents
vector_store.add_documents(documents=all_docs, ids=uuids)

print("✅ Successfully uploaded all documents to Pinecone!")

# # ------------------------------
# # 9️⃣  Verify Upload
# # ------------------------------
# print("\n🔍 Verifying upload...")
# stats = pc_index.describe_index_stats()
# print(f"Total vectors in index: {stats['total_vector_count']}")
# print(f"Namespace '{NAMESPACE}' vector count: {stats.get('namespaces', {}).get(NAMESPACE, {}).get('vector_count', 0)}")

# # Test retrieval
# print("\n🧪 Testing retrieval...")
# test_docs = vector_store.similarity_search("What services does websouls offer", k=3)
# print(f"Retrieved {len(test_docs)} documents")
# if len(test_docs) > 0:
#     print("\nFirst result preview:")
#     print(test_docs[0].page_content[:200])
#     print(f"\nMetadata: {test_docs[0].metadata}")
# else:
#     print("⚠️ Warning: Retrieval test returned 0 documents")