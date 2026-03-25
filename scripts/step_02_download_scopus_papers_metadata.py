import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta
from elsapy.elsclient import ElsClient
from elsapy.elssearch import ElsSearch
import time
import os
import calendar

from dotenv import load_dotenv

load_dotenv()

MAX_RETRIES = 100
SLEEP_TIME = 10

def search_scopus_by_date_range(base_query, api_key, start_date):
    """
    Searches Scopus for a specific date range (e.g., a month).
    Dates should be in 'YYYY-MM-DD' format.
    """
    print("\n--- Starting Scopus Date Range Search (with elsapy) ---")

    client = ElsClient(api_key)
    client.inst_token = os.environ.get('X-ELS-INST')

    # Construct the query with the date range

    date_query = f"PUBDATETXT({calendar.month_name[start_date.month]} {start_date.year})"
    base_query = base_query.replace('\n', ' ')
    full_query = f"({base_query}) AND {date_query}"
    for _ in range(MAX_RETRIES):
        try:
            doc_srch = ElsSearch(full_query, 'scopus')
            doc_srch.execute(client, get_all=True)
            break
        except:
            time.sleep(SLEEP_TIME)
    else:
        return []

    print(f"Found {len(doc_srch.results)} records for the date range.")

    all_results = []

    for item in doc_srch.results:
        title = item.get("dc:title", "N/A")
        doi = item.get("prism:doi", None)
        link = item.get("link", [{}])[0].get("@href")
        row = {
            "source": "Scopus", "title": title, "doi": doi,
            "link": f"https://doi.org/{doi}" if doi else link
        }
        all_results.append(row)
    
    print(f"\n✅ Scopus date range search complete. Total records found: {len(all_results)}")
    return all_results

def get_monthly_scopus_data(query, start_date, end_date):
    """
    Iterates through each month from start_date to end_date,
    executes the Scopus search, and aggregates the results.
    """
    all_monthly_results = []
    current_date = start_date

    while current_date <= end_date:
        # Determine the end of the current month
        # This handles months with different numbers of days
        end_of_month = current_date + relativedelta(day=31)
        
        # Ensure the end_of_month does not exceed the overall end_date
        if end_of_month > end_date:
            end_of_month = end_date
            
        print(f"Processing month: {current_date.strftime('%Y-%m')}")
        
        monthly_results = search_scopus_by_date_range(
            base_query=query,
            api_key=os.environ.get('X-ELS-APIKEY'),
            start_date=current_date
        )
        all_monthly_results.extend(monthly_results)

        # Move to the first day of the next month
        current_date += relativedelta(months=1)
    
    return all_monthly_results

# Example usage to search from January 2024 to May 2024
with open('data/queries/scopus.txt') as reader:
    query = reader.read()
    
    start_date = date(1900, 1, 1)
    end_date = date.today()

    df_results = pd.DataFrame(get_monthly_scopus_data(query, start_date, end_date))
    df_results = df_results[df_results['title'] != "N/A"]
    df_results.to_csv('data/literature_data/scopus_metadata.csv', index=False)