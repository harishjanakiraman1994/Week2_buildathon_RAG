import os
import urllib3
from dotenv import load_dotenv  # <-- Add this line

# Load environment variables from the .env file
load_dotenv()  # <-- Add this line

from load_data import get_all_scheme_links, scrape_scheme_details
# ... rest of your code remains exactly the same ...

import os
import urllib3
from load_data import get_all_scheme_links, scrape_scheme_details
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Ensure API Key is available
if not os.environ.get("OPENAI_API_KEY"):
    print("[!] ERROR: Please set your OPENAI_API_KEY environment variable.")
    exit(1)

print("Step 1: Gathering and chunking content...")
all_schemes = get_all_scheme_links()
docs = []

# Using the first 3 schemes for our quick test setup
for scheme in all_schemes[:3]:
    detail_text = scrape_scheme_details(scheme['url'])
    doc = Document(page_content=detail_text, metadata={"title": scheme['title'], "source": scheme['url']})
    docs.append(doc)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)
print(f"-> Prepared {len(chunks)} chunks.")

print("\nStep 2: Initializing OpenAI Embeddings & FAISS Vector Store...")
# Initialize the embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Create the FAISS index from our document chunks
db_directory = "faiss_index"
vector_store = FAISS.from_documents(chunks, embeddings)

# Save the FAISS index locally to a folder
vector_store.save_local(db_directory)
print(f"-> FAISS Vector DB successfully created and saved locally at: '{db_directory}/'")

print("\nStep 3: Testing retrieval similarity search via FAISS...")
query = "Who should I submit the application to for farmer training?"

# Let's do a search using our local FAISS store
matching_docs = vector_store.similarity_search(query, k=2)

print(f"\nQuery: '{query}'")
for i, doc in enumerate(matching_docs):
    print(f"\n[Match {i+1}] From Scheme: {doc.metadata['title']}")
    print(f"Content snippet:\n{doc.page_content[:200]}...")
    