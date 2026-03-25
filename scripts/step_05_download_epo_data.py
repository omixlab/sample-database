import epo_ops
import requests
import os
import xml.etree.ElementTree as ET
import re
import time
import json

KEY = "mxBqwaL80roV5y3MIRs2o2J7mOxGwntd8E55KZe8eYb2AREf"
SECRET = "EI5WethGd0nR4pFwd67lMtCxujcwPyqNkGXW5G0vawyZOZTrE082aOMJkRlGGF29"
QUERY = open('data/queries/epo.txt').read()

OUTPUT_DIR = "data/patent_data/epo/"

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

def xml_to_dict(element):
    node = {}
    if element.text and element.text.strip():
        node['text'] = element.text.strip()
    for key, value in element.attrib.items():
        node[f"@{key}"] = value
    for child in element:
        child_tag = child.tag.split('}')[-1] 
        child_dict = xml_to_dict(child)
        if child_tag not in node:
            node[child_tag] = child_dict
        else:
            if not isinstance(node[child_tag], list):
                node[child_tag] = [node[child_tag]]
            node[child_tag].append(child_dict)
    return node

def retrieve_and_save_data(client, country, number, kind, id_str):
    json_path = os.path.join(OUTPUT_DIR, f"{id_str}.json")
    if os.path.exists(json_path):
        return

    patent_data = {}
    docdb_input = epo_ops.models.Docdb(number, country, kind)
    epodoc_input = epo_ops.models.Epodoc(id_str)

    try:
        biblio_response = client.published_data(
            reference_type='publication',
            input=docdb_input,
            endpoint='biblio'
        )
        biblio_root = ET.fromstring(biblio_response.text)
        exchange_document = biblio_root.find('.//ext:exchange-document', NS)
        if exchange_document is not None:
            patent_data['biblio'] = xml_to_dict(exchange_document)
    except Exception as e:
        print(f"Biblio structural failure: {id_str} | {str(e)}")

    claims_acquired = False
    for input_format in [epodoc_input, docdb_input]:
        try:
            claims_response = client.published_data(
                reference_type='publication',
                input=input_format,
                endpoint='claims'
            )
            claims_root = ET.fromstring(claims_response.text)
            claims_node = claims_root.find('.//ops:claims', NS)
            
            patent_data['claims'] = xml_to_dict(claims_node) if claims_node is not None else xml_to_dict(claims_root)
            claims_acquired = True
            break
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                continue
            break
        except Exception:
            continue

    if not claims_acquired:
        patent_data['claims'] = {"status": "404 Not Found in EPO"}

    if patent_data.get('biblio'):
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(patent_data, f, indent=4, ensure_ascii=False)
        print(f"JSON acquired: {id_str}.json" + (" (Claims missing)" if not claims_acquired else ""))
    else:
        print(f"Fatal data absence: {id_str}")

def retrieve_patent_pdf(client, id_str):
    id_str = id_str.strip()
    pdf_path = os.path.join(OUTPUT_DIR, f"{id_str}.pdf")
    if os.path.exists(pdf_path):
        return

    epodoc_input = epo_ops.models.Epodoc(id_str)

    try:
        inquiry_response = client.published_data(
            reference_type='publication',
            input=epodoc_input,
            endpoint='images'
        )
        root = ET.fromstring(inquiry_response.text)
        
        pdf_link = None
        for instance in root.findall('.//ops:document-instance', NS):
            if instance.attrib.get('desc') == 'FullDocument':
                pdf_link = instance.attrib.get('link')
                break
        
        if pdf_link:
            pdf_response = client.image(pdf_link)
            with open(pdf_path, "wb") as f:
                f.write(pdf_response.content)
            print(f"File acquired (EPO OPS): {id_str}.pdf")
            return

    except Exception:
        pass

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }

    try:
        fpo_url = f"https://www.freepatentsonline.com/{id_str}.html"
        response = requests.get(fpo_url, headers=headers, timeout=10)
        match = re.search(r'href="([^"]+\.pdf)"', response.text)
        if match:
            pdf_path_match = match.group(1)
            pdf_url = f"https://www.freepatentsonline.com{pdf_path_match}" if pdf_path_match.startswith("/") else f"https://www.freepatentsonline.com/{pdf_path_match}"
            pdf_response = requests.get(pdf_url, headers=headers, timeout=30)
            if 'The patent/application' not in pdf_response.text:
                with open(pdf_path, "wb") as f:
                    f.write(pdf_response.content)
                print(f"File acquired (FPO): {id_str}.pdf")
                return
    except Exception:
        pass

    try:
        command = '''curl -sA "Mozilla/5.0" -L "https://patents.google.com/patent/%s/" | grep -oE 'https://patentimages\.storage\.googleapis\.com/[^"]+\.pdf' | head -1 | xargs -I {} wget {} -O "data/patent_data/%s.pdf"'''%(id_str, id_str)
        os.system(command)
    except:
        return

    print(f"Failed to acquire PDF: {id_str}")

def process_patents():
    client = initialize_client()
    page_size = 10
    cursor_position = 1

    while True:
        try:
            response = client.published_data_search(cql=QUERY, range_begin=cursor_position, range_end=cursor_position + page_size - 1)
        except Exception as e:
            print(f"Search API Error: {e}")
            break

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
            
            retrieve_and_save_data(client, country, number, kind, id_str)
            retrieve_patent_pdf(client, id_str)
            
            time.sleep(1) 

        cursor_position += page_size
        if patents_in_page == 0:
            break

if __name__ == "__main__":
    process_patents()