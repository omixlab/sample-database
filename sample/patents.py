import os
import re
import time
import json
import logging
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

logger = logging.getLogger("sample.patents")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ==========================================
# GOOGLE PATENTS
# ==========================================

def download_patent_pdf(patent_id: str, output_path: str) -> bool:
    """
    Downloads a patent PDF from Google Patents by extracting the direct PDF URL from its landing page.
    """
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        return True
        
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }
    
    try:
        # Fetch the patent details page
        url = f"https://patents.google.com/patent/{patent_id}/en"
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            match = re.search(r'https://patentimages\.storage\.googleapis\.com/[^"]+\.pdf', response.text, re.IGNORECASE)
            if match:
                pdf_url = match.group(0)
                # Download the PDF directly
                with requests.get(pdf_url, headers=headers, timeout=30, stream=True) as pdf_resp:
                    if pdf_resp.status_code == 200:
                        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                        with open(output_path, 'wb') as f:
                            for chunk in pdf_resp.iter_content(chunk_size=8192):
                                f.write(chunk)
                        return True
    except Exception as e:
        logger.debug(f"Error downloading Google Patent {patent_id}: {e}")
        
    return False

def download_google_patents(csv_path: str, output_dir: str, max_workers: int = 8) -> None:
    """
    Downloads patent PDFs listed in a Google Patents export.

    :param csv_path: CSV of patent records; the patent id column drives the download.
    :param output_dir: Directory to write PDFs into. Existing files are skipped.
    :param max_workers: Concurrent download threads.
    :returns: ``None``; progress and failures are logged.
    """
    logger.info(f"Starting Google Patents downloads from {csv_path} into {output_dir}...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Google Patents CSV not found: {csv_path}")
        
    df = pd.read_csv(csv_path, skiprows=[0])
    os.makedirs(output_dir, exist_ok=True)
    
    tasks = []
    for _, row in df.iterrows():
        link = row.get('result link')
        if pd.isna(link) or not link:
            continue
        try:
            parts = link.split('/')
            if len(parts) > 4:
                patent_id = parts[4]
                output_path = os.path.join(output_dir, f"{patent_id}.pdf")
                tasks.append((patent_id, output_path))
        except Exception as e:
            logger.warning(f"Error parsing link {link}: {e}")
            
    logger.info(f"Queued {len(tasks)} patents for downloading.")
    
    success_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_patent_pdf, pid, path): pid for pid, path in tasks}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading Patents"):
            pid = futures[future]
            try:
                if future.result():
                    success_count += 1
            except Exception as e:
                logger.error(f"Failed to download patent {pid}: {e}")
                
    logger.info(f"Finished downloading Google Patents. Successfully downloaded {success_count}/{len(tasks)} PDFs.")


