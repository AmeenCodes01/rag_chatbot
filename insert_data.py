import os
import uuid
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_pinecone import PineconeVectorStore
import re

# ------------------------------
# 1️⃣  Setup + Environment
# ------------------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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

text_splitter = SemanticChunker(OpenAIEmbeddings(api_key=OPENAI_API_KEY))

# ------------------------------
# 3️⃣  Markdown Files
# ------------------------------
MD_FOLDER = "websouls_scraped_md"

urls = [
  #  "https://websouls.com/web-hosting.md",
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
    "https://websouls.com/shared-hosting.md",
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
import re
import os
import re

def url_to_filename(url: str) -> str:
    # Remove protocol
    url_no_protocol = re.sub(r'^https?://', '', url)
    
    # Split path and extension
    base, ext = os.path.splitext(url_no_protocol)
    
    # Replace non-alphanumeric characters in base
    safe_base = re.sub(r'[^a-zA-Z0-9]', '_', base)
    
    # Reattach extension
    return f"https___{safe_base}{ext}"


# 5️⃣  Read & Chunk Files
#------------------------------
for url in urls:
    file_path = MD_FOLDER + "/"+ url_to_filename(url)
    #file_path = "websouls_scraped_md/https___websouls_com_shopify_development.md"
    try:
        with open(file_path, encoding="utf-8") as f:
            text = f.read()
            docs = text_splitter.create_documents([text])
            all_docs.extend(docs)
            print(f"✅ Processed {file_path} into {len(docs)} chunks")
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")

print(f"\n📄 Total chunks to upload: {len(all_docs)}")

# ------------------------------
# 6️⃣  Create Pinecone Index (if needed)
# ------------------------------
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
# 7️⃣  Upload to Pinecone
# ------------------------------
vector_store = PineconeVectorStore(index=pc_index, embedding=embedding_function)
uuids = [str(uuid.uuid4()) for _ in range(len(all_docs))]

vector_store.add_documents(documents=all_docs, ids=uuids, namespace=NAMESPACE)
print("✅ Successfully uploaded all documents to Pinecone!")
