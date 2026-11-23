import time
import os
import requests
import pandas as pd
from Bio import Entrez
from xml.etree import ElementTree as ET
from datetime import date
from dateutil.relativedelta import relativedelta
import logging
from tqdm import tqdm

logger = logging.getLogger("sample.metadata")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MAX_RETRIES = 10
SLEEP_TIME = 3

def download_pubmed_metadata(query_file: str, email: str, output_path: str,
                             batch_size: int = 500, start_year: int = 1900) -> None:
    """
    Downloads PubMed publication metadata for the queries in `query_file`.

    :param query_file: Text file of PubMed queries, one per line.
    :param email: Address registered with NCBI; E-utilities requires it for rate-limit accounting.
    :param output_path: Destination CSV.
    :param batch_size: Records fetched per E-utilities call.
    :param start_year: Earliest publication year to retrieve.
    :returns: ``None``; results are written to ``output_path``.
    """
    logger.info(f"Starting PubMed metadata download. Query: {query_file}, Output: {output_path}")
    
    if not os.path.exists(query_file):
        raise FileNotFoundError(f"PubMed query file not found: {query_file}")
        
    with open(query_file, 'r', encoding='utf-8') as f:
        base_query = f.read().strip()
        
    Entrez.email = email
    start_date = date(start_year, 1, 1)
    end_date = date.today()
    current_date = start_date
    
    all_results = []
    
    total_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1
    
    with tqdm(total=total_months, desc="PubMed Querying") as pbar:
        while current_date <= end_date:
            month_start_str = current_date.strftime("%Y/%m/%d")
            month_end_date = current_date + relativedelta(months=1) - relativedelta(days=1)
            month_end_str = month_end_date.strftime("%Y/%m/%d")
            
            logger.info(f"Searching PubMed month: {current_date.strftime('%Y-%m')}...")
            
            date_filter = f'("{month_start_str}"[Date - Publication] : "{month_end_str}"[Date - Publication])'
            monthly_query = f"({base_query}) AND ({date_filter})"
            
            id_list = []
            for attempt in range(MAX_RETRIES):
                try:
                    handle = Entrez.esearch(db="pubmed", term=monthly_query, retmax=10000)
                    record = Entrez.read(handle)
                    handle.close()
                    id_list = record.get("IdList", [])
                    break
                except Exception as e:
                    logger.warning(f"PubMed esearch error on attempt {attempt+1}/{MAX_RETRIES}: {e}")
                    time.sleep(SLEEP_TIME)
            else:
                logger.error(f"Failed to query PubMed for date range {month_start_str} to {month_end_str}")
                current_date += relativedelta(months=1)
                pbar.update(1)
                continue
                
            month_count = len(id_list)
            if month_count == 0:
                current_date += relativedelta(months=1)
                pbar.update(1)
                continue
                
            logger.info(f"Found {month_count} records. Fetching in batches of {batch_size}...")
            
            for i in range(0, month_count, batch_size):
                batch_ids = id_list[i:i + batch_size]
                xml_data = None
                
                for attempt in range(MAX_RETRIES):
                    try:
                        handle = Entrez.efetch(db="pubmed", id=batch_ids, rettype="xml", retmode="text")
                        xml_data = handle.read()
                        handle.close()
                        break
                    except Exception as e:
                        logger.warning(f"PubMed efetch error on attempt {attempt+1}/{MAX_RETRIES}: {e}")
                        time.sleep(SLEEP_TIME)
                else:
                    logger.error(f"Failed to fetch batch {i} to {i+len(batch_ids)} for PubMed")
                    continue
                    
                try:
                    root = ET.fromstring(xml_data)
                    for article in root.findall(".//PubmedArticle"):
                        pmid = article.findtext(".//PMID")
                        doi = article.findtext('.//ArticleId[@IdType="doi"]')
                        title = article.findtext(".//ArticleTitle")
                        abstract = article.findtext(".//AbstractText")
                        
                        year = article.findtext(".//PubDate/Year")
                        if not year:
                            medline_date = article.findtext(".//PubDate/MedlineDate")
                            if medline_date:
                                year = medline_date.split(" ")[0]
                                
                        all_results.append({
                            "internal_id": f"pubmed_{pmid}",
                            "pmid": pmid,
                            "doi": doi,
                            "title": title,
                            "abstract": abstract,
                            "year": year
                        })
                except Exception as e:
                    logger.error(f"XML parsing error in PubMed batch {i}: {e}")
            
            df = pd.DataFrame(all_results)
            df.to_csv(output_path, index=False)
            
            current_date += relativedelta(months=1)
            pbar.update(1)

def download_scopus_metadata(query_file: str, api_key: str, inst_token: str,
                             output_path: str, start_year: int = 1900) -> None:
    """
    Downloads Scopus publication metadata for the queries in `query_file`.

    :param query_file: Text file of Scopus queries, one per line.
    :param api_key: Elsevier API key.
    :param inst_token: Elsevier institutional token, for access outside the campus network.
    :param output_path: Destination CSV.
    :param start_year: Earliest publication year to retrieve.
    :returns: ``None``; results are written to ``output_path``.
    """
    logger.info(f"Starting Scopus metadata download. Query: {query_file}, Output: {output_path}")
    
    if not api_key:
        logger.error("Scopus API key not provided in environment variable X-ELS-APIKEY")
        return
        
    if not os.path.exists(query_file):
        raise FileNotFoundError(f"Scopus query file not found: {query_file}")
        
    with open(query_file, 'r', encoding='utf-8') as f:
        base_query = f.read().strip()
        
    headers = {
        "X-ELS-APIKey": api_key,
        "Accept": "application/json"
    }
    if inst_token:
        headers["X-ELS-Insttoken"] = inst_token

    start_date = date(start_year, 1, 1)
    end_date = date.today()
    current_date = start_date
    
    all_results = []
    
    total_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1
    
    with tqdm(total=total_months, desc="Scopus Querying") as pbar:
        while current_date <= end_date:
            month_start_str = current_date.strftime("%Y-%m-%d")
            month_end_date = current_date + relativedelta(months=1) - relativedelta(days=1)
            month_end_str = month_end_date.strftime("%Y-%m-%d")
            
            logger.info(f"Searching Scopus month: {current_date.strftime('%Y-%m')}...")
            
            monthly_query = f"({base_query}) AND PUBDATETXT(> {month_start_str} AND < {month_end_str})"
            
            start_index = 0
            while True:
                url = f"https://api.elsevier.com/content/search/scopus"
                params = {
                    "query": monthly_query,
                    "count": 200,
                    "start": start_index
                }
                
                response = None
                for attempt in range(MAX_RETRIES):
                    try:
                        response = requests.get(url, headers=headers, params=params, timeout=30)
                        if response.status_code == 200:
                            break
                        elif response.status_code == 429:
                            logger.warning(f"Scopus API rate limit reached on attempt {attempt+1}. Sleeping...")
                            time.sleep(60)
                        else:
                            logger.warning(f"Scopus API error {response.status_code} on attempt {attempt+1}: {response.text}")
                            time.sleep(SLEEP_TIME)
                    except Exception as e:
                        logger.warning(f"Scopus request error on attempt {attempt+1}: {e}")
                        time.sleep(SLEEP_TIME)
                else:
                    logger.error(f"Failed to query Scopus for {month_start_str} to {month_end_str} at start {start_index}")
                    break
                    
                data = response.json()
                results = data.get("search-results", {}).get("entry", [])
                
                if not results:
                    break
                    
                for entry in results:
                    eid = entry.get("eid")
                    doi = entry.get("prism:doi")
                    title = entry.get("dc:title")
                    abstract = entry.get("dc:description")
                    cover_date = entry.get("prism:coverDate", "")
                    year = cover_date.split("-")[0] if cover_date else ""
                    
                    all_results.append({
                        "internal_id": f"scopus_{eid}",
                        "eid": eid,
                        "doi": doi,
                        "title": title,
                        "abstract": abstract,
                        "year": year
                    })
                    
                total_results = int(data.get("search-results", {}).get("opensearch:totalResults", 0))
                start_index += 200
                if start_index >= total_results or start_index >= 5000:
                    break
                    
            df = pd.DataFrame(all_results)
            df.to_csv(output_path, index=False)
            
            current_date += relativedelta(months=1)
            pbar.update(1)
