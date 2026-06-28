import requests
from bs4 import BeautifulSoup
import urllib3
from langchain_core.documents import Document

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def get_all_scheme_links():
    """Fetches the list of all scheme URLs."""
    url = "https://www.tn.gov.in/scheme_list.php?dep_id=Mg=="
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, verify=False)
    response.encoding = 'utf-8'
    soup = BeautifulSoup(response.text, 'html.parser')
    
    schemes = []
    for link in soup.find_all('a'):
        href = link.get('href')
        text = link.text.strip()
        if href and 'scheme_details.php' in href:
            full_url = f"https://www.tn.gov.in/{href.lstrip('/')}"
            schemes.append({"title": text, "url": full_url})
    return schemes

def scrape_scheme_details(scheme_url):
    """Visits a scheme's page and extracts its descriptive text."""
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(scheme_url, headers=headers, verify=False, timeout=10)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Government scheme pages usually place text inside a main table or content div.
        # Let's target the main content block. If not found, grab the body text safely.
        content_div = soup.find('div', {'id': 'clear'}) or soup.find('table')
        if content_div:
            return content_div.get_text(separator="\n", strip=True)
        return soup.body.get_text(separator="\n", strip=True)
    except Exception as e:
        return f"Failed to retrieve content due to: {e}"

# --- RUNNING THE PIPELINE ---
if __name__ == "__main__":
    print("Step 1: Fetching scheme URLs...")
    all_schemes = get_all_scheme_links()
    
    langchain_documents = []
    
    # We will just process the first 3 schemes for testing to see how it looks
    test_limit = 3 
    print(f"\nStep 2: Scraped content from the first {test_limit} schemes...")
    
    for scheme in all_schemes[:test_limit]:
        print(f" -> Scraping: {scheme['title']}")
        detail_text = scrape_scheme_details(scheme['url'])
        
        # Wrap the data inside a LangChain Document object
        doc = Document(
            page_content=detail_text,
            metadata={"title": scheme['title'], "source": scheme['url']}
        )
        langchain_documents.append(doc)
        
    print(f"\nSuccessfully loaded {len(langchain_documents)} items into LangChain Documents!")
    print("\n--- Sample Document Content Peak ---")
    print(f"Title: {langchain_documents[0].metadata['title']}")
    print(f"Content Snippet:\n{langchain_documents[0].page_content[:300]}...")
    