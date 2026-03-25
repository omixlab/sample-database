import epo_ops
import requests
import os
import xml.etree.ElementTree as ET
import re
import time

KEY = "mxBqwaL80roV5y3MIRs2o2J7mOxGwntd8E55KZe8eYb2AREf"
SECRET = "EI5WethGd0nR4pFwd67lMtCxujcwPyqNkGXW5G0vawyZOZTrE082aOMJkRlGGF29"
QUERY = open('data/queries/epo.txt').read()

OUTPUT_DIR = "data/patent_data/"

NS = {
    'ops': 'http://ops.epo.org',
    'ext': 'http://www.epo.org/exchange',
    'xlink': 'http://www.w3.org/1999/xlink'
}

def initialize_client():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return epo_ops.Client(key=KEY, secret=SECRET)

def get_text_or_none(element, path, namespaces):
    found = element.find(path, namespaces)
    return found.text if found is not None else None

def retrieve_patent_pdf(id_str):
    id_str = id_str.strip()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }

    fpo_url = f"https://www.freepatentsonline.com/{id_str}.html"
    response = requests.get(fpo_url, headers=headers, timeout=10)
    match = re.search(r'href="([^"]+\.pdf)"', response.text)
    if match:
        pdf_path = match.group(1)
        pdf_url = f"https://www.freepatentsonline.com{pdf_path}" if pdf_path.startswith("/") else f"https://www.freepatentsonline.com/{pdf_path}"
        pdf_response = requests.get(pdf_url, headers=headers, timeout=30)
        if 'The patent/application' in pdf_response.text and 'does not exist' not in pdf_response.text:
            with open(os.path.join(OUTPUT_DIR, f"{id_str}.pdf"), "wb") as f:
                f.write(pdf_response.content)
            print(f"File acquired (FPO): {id_str}.pdf")


    gp_url = f"https://patents.google.com/patent/{id_str}/"

    response = requests.get(gp_url, headers=headers, timeout=30)
    match = re.search(r"https://patentimages\.storage\.googleapis\.com/[^\"]+\.pdf", response.text, re.IGNORECASE)
    
    if match:
        file_url = match.group(0)
        file_response = requests.get(file_url, headers=headers, timeout=30)
        with open(os.path.join(OUTPUT_DIR, f"{id_str}.pdf"), "wb") as f:
            f.write(file_response.content)
        print(f"File acquired (Google Storage): {id_str}.pdf")
        return
        
    print(f"Failed to acquire PDF: {id_str}")

def process_patents():
    page_size = 10
    cursor_position = 1

    while True:
        client = initialize_client()
        response = client.published_data_search(cql=QUERY, range_begin=cursor_position, range_end=cursor_position + page_size - 1)
        
        if response.status_code != 200:
            break
            
        root = ET.fromstring(response.text)
        patents_in_page = 0
        
        publications = root.findall('.//ops:publication-reference', NS)
        if not publications:
            break

        for pub_ref in publications:
            doc_id = pub_ref.find('.//ext:document-id[@document-id-type="docdb"]', NS)
            if doc_id is None:
                continue

            country = get_text_or_none(doc_id, 'ext:country', NS)
            number = get_text_or_none(doc_id, 'ext:doc-number', NS)
            kind = get_text_or_none(doc_id, 'ext:kind', NS)
            
            if not (country and number and kind):
                continue

            id_str = f"{country}{number}{kind}"
            print(f"Processing: {id_str}")
            patents_in_page += 1
            
            retrieve_patent_pdf(id_str)
            time.sleep(1) 

        cursor_position += page_size
        if patents_in_page == 0:
            break

if __name__ == "__main__":
    process_patents()