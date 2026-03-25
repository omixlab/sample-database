import os
import time
import glob
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import random
import argparse

def download_pdf(doi: str, output_path: str, email: str) -> None:
    api_url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
    
    try:
        response = requests.get(api_url, timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"API Error [{doi}]: {e}")
        return

    if not data:
        return False

    pdf_url = data.get("best_oa_location", dict())
    if not pdf_url:
        print(f"Error [{doi}]: Open access record exists, but PDF URL is missing.")
        return
    pdf_url = data.get("best_oa_location").get("url_for_pdf")
    output_dir = os.path.abspath(os.path.dirname(output_path))
    os.makedirs(output_dir, exist_ok=True)

    options = Options()
    
    prefs = {
        "download.default_directory": output_dir,
        "download.prompt_for_download": False,
        "plugins.always_open_pdf_externally": True,
        "profile.default_content_settings.popups": 0,
    }
    options.add_experimental_option("prefs", prefs)
    
    driver = webdriver.Chrome(options=options)
    
    try:
        existing_pdfs = set(glob.glob(os.path.join(output_dir, "*.pdf")))
        
        driver.set_page_load_timeout(30)
        driver.get(pdf_url)
        timeout = 60
        new_pdf = None
        for _ in range(timeout):
            time.sleep(1)
            current_pdfs = set(glob.glob(os.path.join(output_dir, "*.pdf")))
            new_files = current_pdfs - existing_pdfs
            
            crdownloads = glob.glob(os.path.join(output_dir, "*.crdownload"))
            if new_files and not crdownloads:
                new_pdf = new_files.pop()
                break
        
        if new_pdf:
            os.rename(new_pdf, output_path)
            print(f"Success: File saved to {output_path}")
        else:
            print(f"Error [{doi}]: Download timed out or failed.")
            
    except Exception as e:
        print(f"Selenium Error [{doi}]: {e}")
    finally:
        driver.quit()

import argparse

if __name__ == '__main__':

    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--doi", help='digital object identifier (DOI) of the paper', required=True)
    argument_parser.add_argument("--output", help='output path for PDF', required=True)
    argument_parser.add_argument("--email", help='email to be used by unpaywall', required=True)

    arguments = argument_parser.parse_args()

    download_pdf(arguments.doi, arguments.output, arguments.email)