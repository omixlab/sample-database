import time
import requests
import pandas as pd
from Bio import Entrez
from xml.etree import ElementTree as ET
from datetime import date
from dateutil.relativedelta import relativedelta

MAX_RETRIES = 1000
SLEEP_TIME = 10

def search_pubmed_by_month(base_query, email, batch_size=500, start_year=1900):
    """
    Iterates month-by-month from 1900 to today, running a search for each month
    to bypass the 9999 record limit on very broad queries.
    """
    print("--- Starting PubMed Monthly Search (1900-Present) ---")
    Entrez.email = email
    
    start_date = date(1900, 1, 1)
    end_date = date.today()
    current_date = start_date
    
    all_results = []
    
    while current_date <= end_date:
        # Define the start and end of the current month
        month_start_str = current_date.strftime("%Y/%m/%d")
        month_end_date = current_date + relativedelta(months=1) - relativedelta(days=1)
        month_end_str = month_end_date.strftime("%Y/%m/%d")
        
        print(f"\nSearching month: {current_date.strftime('%Y-%m')}...")
        
        # Construct the query for the current month
        date_filter = f'("{month_start_str}"[Date - Publication] : "{month_end_str}"[Date - Publication])'
        monthly_query = f"({base_query}) AND ({date_filter})"
        
        for _ in range(MAX_RETRIES):
            try:
                # 1. ESEARCH for the current month
                handle = Entrez.esearch(db="pubmed", term=monthly_query, retmax=10000) # 10k is max for esearch
                record = Entrez.read(handle)
                handle.close()
                break
            except:
                time.sleep(SLEEP_TIME) 
        else:
            continue       
        id_list = record["IdList"]
        month_count = len(id_list)
        
        if month_count == 0:
            print("Found 0 records for this month.")
            # Move to the next month
            current_date += relativedelta(months=1)
            continue
            
        print(f"Found {month_count} records for this month. Fetching in batches...")
        
        # 2. EFETCH the records for the current month in batches
        for i in range(0, month_count, batch_size):
            batch_ids = id_list[i:i + batch_size]
            
            for i in range(MAX_RETRIES):
                try:
                    handle = Entrez.efetch(db="pubmed", id=batch_ids, rettype="xml", retmode="text")
                    xml_data = handle.read()
                    break
                except:
                    print('sleeping')
                    time.sleep(SLEEP_TIME)
            else:
                continue
            handle.close()
            
            # 3. PARSE the batch and append to results
            root = ET.fromstring(xml_data)
            for article in root.findall(".//PubmedArticle"):
                title_element = article.find(".//ArticleTitle")
                title = title_element.text if title_element is not None else "N/A"
                
                accession_ids = [(el.get("IdType"), el.text) for el in article.iter("ArticleId")]
                
                pmid = ''
                doi = ''

                for accession_id in accession_ids:
                    if accession_id[1] == 'pubmed':
                        pmid = accession_id[1]
                    elif accession_id[0] == 'doi':
                        doi = accession_id[1]
                
                row = {
                    "source": "PubMed",
                    "title": title,
                    "doi": doi,
                    "pmid": pmid,
                    "link": f"https://doi.org/{doi}" if doi else f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                }

                all_results.append(row)

            time.sleep(0.4) # Pause between batches
        
        print(f"Total records accumulated so far: {len(all_results)}")
        
        # Move to the next month
        current_date += relativedelta(months=1)
        
    print(f"\n✅ PubMed monthly search complete. Total records found: {len(all_results)}")
    return all_results

with open('data/queries/pubmed.txt') as reader:
    query = reader.read()
    df_results = pd.DataFrame(search_pubmed_by_month(query, email='1_1@gmail.com'))
    df_results.to_csv('data/literature_data/pubmed_metadata.csv', index=False)