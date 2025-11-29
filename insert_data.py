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
NAMESPACE = "websoulsRAGT2"

# ------------------------------
# 2️⃣  Embedding + Chunking Setup
# ------------------------------
embedding_function = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY,
    dimensions=768,
)

#text_splitter = LLMTextSplitter(count_tokens=True, prompt_type='wide')
# ------------------------------
# 3️⃣  Markdown Files
# ------------------------------
MD_FOLDER = "websouls_scraped_md"

urls = [
    "https://websouls.com/web-hosting",
    "https://websouls.com",
   "https://websouls.com/website-security-management",
     "https://websouls.com/online-store-management",
    "https://websouls.com/360-degree-digital-marketing",
###     "https://websouls.com/ui-ux-design",
    "https://websouls.com/custom-software-development",
    "https://websouls.com/laravel-custom-development",
    "https://websouls.com/react-custom-development",
    "https://websouls.com/shopify-development",
    "https://websouls.com/wordpress-development",
  "https://websouls.com/social-media-marketing",  
    "https://websouls.com/google-ads",
    "https://websouls.com/content-writing",
###    "https://websouls.com/mobile-app-development",
    "https://websouls.com/web-hosting-with-domain",
    "https://websouls.com/seo-services",
    "https://websouls.com/contactus",
    "https://websouls.com/about",
    "https://websouls.com/team",
    "https://websouls.com/shared-hosting",
    "https://websouls.com/domain-transfer",
    "https://websouls.com/ecommerce-solution",
    "https://websouls.com/policy",
     "https://websouls.com/buy-pk-domain",
    "https://websouls.com/ssl-certificates",
    "https://websouls.com/pk-vps",
    "https://websouls.com/vps-hosting",
    "https://websouls.com/wordpress-hosting-in-pakistan",
    "https://websouls.com/reseller-hosting",
    "https://websouls.com/buy-ae-domains",
    "https://websouls.com/whyus",
    "https://websouls.com/dedicated-server",
    "https://websouls.com/privacy",
    "https://websouls.com/web-development",
    "https://websouls.com/domain-registration",
    "https://websouls.com/payment-methods"
]


all_docs = []

# ------------------------------
# 4️⃣  Helper: URL → Filename
# ------------------------------
import base64

def url_to_filename(url: str) -> str:
    encoded = base64.urlsafe_b64encode(url.encode()).decode()
    return f"{encoded}.md"


def url_to_filename_fix(url: str) -> str:
    url_no_protocol = re.sub(r'^https?://', '', url)
    base, ext = os.path.splitext(url_no_protocol)
    safe_base = re.sub(r'[^a-zA-Z0-9]', '_', base)
    return f"https___{safe_base}{ext}.md"


def filename_to_url(filename: str) -> str:
    encoded = filename.replace(".md", "")
    print(f"Encoded part: {repr(encoded)}  Length: {len(encoded)}")
    return base64.urlsafe_b64decode(encoded.encode()).decode()

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


def process_docs_from_ai(url: str):
    """Reads a .md file, sends it to the AI, gets chunks & inferred URL."""
    file_path = "./websouls_scraped_md/"+url_to_filename_fix(url)
    with open(file_path, encoding="utf-8") as f:
        text = f.read()

    prompt = f"""
You are given the full text of a scraped website page.

Your goal is to divide the text into logical chunks for vector embeddings,
while preserving *all factual information* and removing marketing fluff.

Guidelines:
1. Detect the main **category or service type** (e.g., hosting, domains, email, VPS, etc.).
2. Collect all **sub-sections or offerings** (e.g., plans, packages, tiers, features) together into a single chunk.
   - Do not split into separate JSON objects per sub-section.
   - Merge all offerings into one combined chunk.
3. In that chunk:
   - Remove navigation text, repetitive slogans, and marketing fluff.
   - Keep **all factual details**: features, pricing, benefits, limitations, restrictions, unsupported features.
   - Explicitly preserve negative statements (e.g., "X not supported", "Y not included").
4. Each JSON object must follow this format:

{{
  "content": "<category> – all_offerings – <clean factual descriptive text of all offerings combined>",
  "metadata": {{
    "category": "<detected category>",
    "plan_name": "all_offerings"
  }}
}}

5. Always return a valid JSON **array** with no markdown formatting.
6. Do not summarize — preserve all factual descriptive text for embeddings.
7. If you find general content (not offering-specific), include it as a separate chunk with
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
        metadata["url"] = url

        doc = Document(page_content=item["content"].strip(), metadata=metadata)
        docs.append(doc)

    print(f"✅ Processed {file_path} into {len(docs)} chunks.")
    return docs

#testing


for url in urls:
   
    if url_to_filename(url).endswith(".md"):
        
        docs = process_docs_from_ai(url)
        all_docs.extend(docs)
        time.sleep(5)
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

# # # ------------------------------
# # # 9️⃣  Verify Upload
# # # ------------------------------
# # print("\n🔍 Verifying upload...")
# # stats = pc_index.describe_index_stats()
# # print(f"Total vectors in index: {stats['total_vector_count']}")
# # print(f"Namespace '{NAMESPACE}' vector count: {stats.get('namespaces', {}).get(NAMESPACE, {}).get('vector_count', 0)}")

# # # Test retrieval
# # print("\n🧪 Testing retrieval...")
# # test_docs = vector_store.similarity_search("What services does websouls offer", k=3)
# # print(f"Retrieved {len(test_docs)} documents")
# # if len(test_docs) > 0:
# #     print("\nFirst result preview:")
# #     print(test_docs[0].page_content[:200])
# #     print(f"\nMetadata: {test_docs[0].metadata}")
# # else:
# #     print("⚠️ Warning: Retrieval test returned 0 documents")