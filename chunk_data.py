from load_data import get_all_scheme_links, scrape_scheme_details
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Fetch and load the data exactly like before
print("Fetching and loading documents...")
all_schemes = get_all_scheme_links()
docs = []

# Scraping the first 3 for testing
for scheme in all_schemes[:3]:
    detail_text = scrape_scheme_details(scheme['url'])
    doc = Document(page_content=detail_text, metadata={"title": scheme['title'], "source": scheme['url']})
    docs.append(doc)

print(f"Loaded {len(docs)} large documents.")

# 2. Initialize the Text Splitter
# chunk_size: Maximum characters per chunk
# chunk_overlap: How many characters to repeat from the previous chunk (keeps context intact)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400, 
    chunk_overlap=50
)

# 3. Split the documents into chunks
chunks = text_splitter.split_documents(docs)

print(f"Split completed! Created {len(chunks)} smaller chunks from {len(docs)} documents.\n")

# Let's inspect a couple of chunks to see how they look
print("--- Chunk 1 Preview ---")
print(f"Metadata: {chunks[0].metadata}")
print(f"Content:\n{chunks[0].page_content}\n")

print("--- Chunk 2 Preview ---")
print(f"Metadata: {chunks[1].metadata}")
print(f"Content:\n{chunks[1].page_content}\n")
