"""
Web scraper for extracting additional HuggingFace dataset information
"""

import re
import requests
from bs4 import BeautifulSoup
from typing import Dict, Any


def scrape_dataset_page(dataset_id: str) -> Dict[str, Any]:
    """Scrape additional data from HuggingFace dataset page"""
    url = f"https://huggingface.co/datasets/{dataset_id}"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        data = {'dataset_id': dataset_id, 'url': url}

        # Full README
        readme = soup.find('article') or soup.find('div', class_='prose')
        if readme:
            text = readme.get_text(' ', strip=True)
            data['full_readme'] = re.sub(r'\s+', ' ', text)  # Replace all whitespace with single space
        else:
            data['full_readme'] = None

        # Citation
        bibtex = soup.find('code', class_='language-bibtex')
        if bibtex:
            data['citation'] = bibtex.get_text(strip=True)

        # License
        license_link = soup.find('a', href=lambda h: h and 'license' in h.lower())
        if license_link:
            data['license'] = license_link.get_text(strip=True)

        # Number of rows
        size_match = re.search(r'([\d,.]+k?)\s+rows', soup.get_text(), re.IGNORECASE)
        if size_match:
            data['num_rows'] = size_match.group(1)

        print(f"✅ Scraped {dataset_id}")
        return data

    except Exception as e:
        print(f"❌ Scraping failed for {dataset_id}: {e}")
        return {'dataset_id': dataset_id, 'error': str(e)}
