import requests
from bs4 import BeautifulSoup
import urllib3

# This line turns off that annoying warning message you saw
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

URL = "https://www.tn.gov.in/scheme_list.php?dep_id=Mg=="

# Added standard headers so the website thinks a real web browser is visiting
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

print("Fetching schemes from the website...")
response = requests.get(URL, headers=headers, verify=False)
response.encoding = 'utf-8'

soup = BeautifulSoup(response.text, 'html.parser')

# Let's grab all links to see what the website is actually returning
all_links = soup.find_all('a')
print(f"Total links found on page: {len(all_links)}")

found_schemes = 0
for link in all_links:
    href = link.get('href')
    text = link.text.strip()
    
    # Check if the link looks like a scheme URL
    if href and ('scheme' in href.lower() or 'detail' in href.lower()):
        # If it's a relative URL, build the full URL
        full_url = href if href.startswith('http') else f"https://www.tn.gov.in/{href.lstrip('/')}"
        print(f"-> Found Scheme: {text}")
        print(f"   URL: {full_url}\n")
        found_schemes += 1

if found_schemes == 0:
    print("\n[!] No specific scheme links matched our filter. Let's print the first 5 links on the page to inspect them:")
    for link in all_links[:5]:
        print(f"Text: {link.text.strip()} | Link: {link.get('href')}")
        