# Requirements: pip install biopython requests
import requests
import pandas as pd
from Bio import Entrez
import os
import sys
import glob
import json

Entrez.email = "fredk.cdtec@ufpel.edu.br"  # Required by NCBI
timeout = 60

def get_doi_from_pmid(pmid: str) -> str:
    try:
        handle = Entrez.efetch(db="pubmed", id=pmid, retmode="xml")
        records = Entrez.read(handle)
        handle.close()
        
        for article in records.get('PubmedArticle', []):
            article_ids = article.get('PubmedData', {}).get('ArticleIdList', [])
            for article_id in article_ids:
                if article_id.attributes.get('IdType') == 'doi':
                    return str(article_id)
    except Exception as e:
        return f"Entrez fetch failed: {e}"
    
    return None

def get_doi_from_title(title: str) -> str:
    url = "https://api.crossref.org/works"
    params = {
        "query.title": title,
        "rows": 1,
        "select": "DOI"
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        items = data.get("message", {}).get("items", [])
        
        if items:
            return items[0].get("DOI")
    except Exception as e:
        return f"Crossref API request failed: {e}"
        
    return None

def get_paper_doi(identifier: str, source: str = "title") -> str:
    if source.lower() == "pmid":
        return get_doi_from_pmid(identifier)
    elif source.lower() == "title":
        return get_doi_from_title(identifier)
    else:
        raise ValueError("Source must be 'pmid' or 'title'")

if __name__ == "__main__":
    #pmid_doi = get_paper_doi("23856986", source="pmid")
    #print(f"PMID DOI: {pmid_doi}")
    
    #title_doi = get_paper_doi("CRISPR/Cas9 for genome editing", source="title")
    #print(f"Title DOI: {title_doi}")


    if '--adp' in sys.argv:

        database = 'adp'
        os.system(f"mkdir -p data/antimicrobial_databases/papers/{database}")
        for r,row in pd.read_csv("data/antimicrobial_databases/adp/adb.csv", skiprows=[0]).iterrows():
            if not "Title:" in str(row['Title:']):
                try:
                    doi_number = get_paper_doi(row['Title:'], source='title')
                except:
                    continue
                os.system(
                    f'timeout {timeout} python scripts/utils/download_pdf.py --doi "{doi_number}" --output data/antimicrobial_databases/papers/{database}/{row["APD ID:"]}.pdf --email {Entrez.email}'
                )
    if '--dbamp' in sys.argv:

        database = 'dbamp'
        os.system(f"mkdir -p data/antimicrobial_databases/papers/{database}")
        for csv_file in glob.glob("data/antimicrobial_databases/dbamp/extracted_metadata/*/1.csv"):
            for r,row in pd.read_csv(csv_file, header=None).iterrows():
                if r == 0:
                    dbamp_id = row.values[1]
                    print(dbamp_id)
                if row.values[0] == 'PubMed':
                    doi_number = get_doi_from_pmid(row.values[1])
                    os.system(
                        f'timeout {timeout} python scripts/utils/download_pdf.py --doi "{doi_number}" --output data/antimicrobial_databases/papers/{database}/{dbamp_id}.pdf --email {Entrez.email}'
                    )
    if '--dramp' in sys.argv:

        database = 'dramp'
        for txt_file in glob.glob("data/antimicrobial_databases/dramp/Datasets/*.txt"):
            os.system(f"mkdir -p data/antimicrobial_databases/papers/{database}")
            df_dramp = pd.read_csv(txt_file, sep='\t')
            for r,row in df_dramp.iterrows():
                try:
                    doi_number = get_doi_from_title(row.Title)
                except:
                    pass
                os.system(
                        f'timeout {timeout} python scripts/utils/download_pdf.py --doi "{doi_number}" --output data/antimicrobial_databases/papers/{database}/{dbamp_id}.pdf --email {Entrez.email}'
                    )
    
    if '--dbaaspn' in sys.argv:
        database = 'dbaaspn'
        for json_file in glob.glob('data/antimicrobial_databases/dbaasp/*.json'):
            dbaaspn_id = json_file.split('/')[3].split('.')[0]
            json_data = json.loads(open(f'data/antimicrobial_databases/dbaasp/{dbaaspn_id}.json').read())
            for a, article in enumerate(json_data['articles']):
                try:
                    doi_number = get_doi_from_title(article['title'])
                except:
                    pass
                os.system(
                        f'timeout {timeout} python scripts/utils/download_pdf.py --doi "{doi_number}" --output data/antimicrobial_databases/papers/{database}/{dbaaspn_id}_{a}.pdf --email {Entrez.email}'
                    )