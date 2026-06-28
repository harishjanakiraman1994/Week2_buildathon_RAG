import os
import time
import urllib3
from dotenv import load_dotenv

# Load configurations first
load_dotenv()

from load_data import get_all_scheme_links, scrape_scheme_details
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

if not os.environ.get("OPENAI_API_KEY"):
    print("[!] ERROR: Please set your OPENAI_API_KEY inside the .env file.")
    exit(1)

def main():
    print("🚀 Starting BULK Data Extraction for TN Agriculture Schemes...")
    
    # 1. Fetch all link structures from the main portal
    all_schemes = get_all_scheme_links()
    total_schemes = len(all_schemes)
    print(f"-> Discovered a total of {total_schemes} schemes on the source page.")
    
    documents = []
    
    # 2. Iterate through ALL schemes and fetch content details
    print("\n⏳ Downloading content pages (this might take a minute)...")
    for index, scheme in enumerate(all_schemes, start=1):
        print(f"[{index}/{total_schemes}] Scraping: {scheme['title']}")
        
        detail_text = scrape_scheme_details(scheme['url'])
        
        # Verify we didn't extract an empty body
        if len(detail_text.strip()) > 50:
            doc = Document(
                page_content=detail_text, 
                metadata={"title": scheme['title'], "source": scheme['url']}
            )
            documents.append(doc)
        else:
            print(f"   ⚠️ Warning: Low or missing content for '{scheme['title']}', skipping.")
            
        # Polite crawling: pause for 1 second between requests
        time.sleep(1)

    print(f"\n✅ Successfully extracted {len(documents)} matching scheme documents.")

    # 3. Chunking strategy
    print("\n📦 Slicing documents into clean overlapping text chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    print(f"-> Created {len(chunks)} text chunks ready for vector processing.")

    # 4. Generate Embeddings and overwrite the FAISS local database
    print("\n🤖 Computing OpenAI Embeddings and compiling FAISS database...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    db_directory = "faiss_index"
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Save/overwrite the database locally
    vector_store.save_local(db_directory)
    print(f"\n🎉 SUCCESS! Your complete knowledge base has been written to: '{db_directory}/'")
    print("You can now restart your Streamlit App to search across ALL schemes!")

if __name__ == "__main__":
    main()
    