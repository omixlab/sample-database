# sAMPle: A web server of antimicrobial peptides

## Overview
This repository contains a sequence of Python scripts designed to systematically query, download, and extract structured data from academic literature (PubMed, Scopus) and patent databases (EPO) regarding antimicrobial peptides. It executes PDF acquisition and utilizes the Google Gemini API for structured data extraction via Large Language Models.

## Dependencies
Ensure the following Python packages are installed:
* `requests`
* `pandas`
* `biopython` (Bio)
* `python-dateutil`
* `elsapy`
* `python-dotenv`
* `tqdm`
* `epo-ops-client`
* `google-genai`

### Environment Variables
Create a `.env` file in the root directory containing:
* `X-ELS-INST`: Scopus Institutional Token.
* `X-ELS-APIKEY`: Scopus API Key.
* `GEMINI_KEY`: Google Gemini API Key.

## Directory Structure
The scripts expect the following directory structure to exist:
* `data/queries/`: Text files containing search queries (`pubmed.txt`, `scopus.txt`, `epo.txt`).
* `data/prompts/step_04/`: LLM prompt (`01_extraction_prompt.txt`) and response schema (`01_extraction_prompt_response_format.json`).
* `data/literature_data/`: Output directory for literature CSV metadata.
* `data/literature_data/pubmed/`: Output directory for PubMed PDFs.
* `data/literature_data/scopus/`: Output directory for Scopus PDFs.
* `data/patent_data/epo/`: Output directory for EPO JSON and PDF data.
* `data/llm_data_extraction/literature_data/`: Output directory for parsed LLM JSON results.

## Pipeline Execution Sequence

### Literature Metadata Acquisition
* **`step_01_download_pubmed_papers_metadata.py`**
    Queries PubMed from 1900 to present. Iterates month-by-month to bypass the 10,000 record ESEARCH limit. Outputs `pubmed_metadata.csv`.
* **`step_02_download_scopus_papers_metadata.py`**
    Queries Scopus utilizing `elsapy`. Iterates by month to circumvent API limits. Outputs `scopus_metadata.csv`.

### PDF Acquisition
* **`step_03_download_and_extract_papers_data.py`**
    Reads metadata CSVs, generates unique internal identifiers (UUIDs/Hashes), and attempts to download the corresponding PDFs using external utility scripts via DOI. Maintains state in `[source]_downloaded.csv` to prevent duplicate downloads.

### Patent Data Acquisition
* **`step_05_download_epo_data.py`**
    Queries the European Patent Office (EPO) Open Patent Services (OPS). Retrieves bibliographic data and claims as JSON. Attempts to retrieve full-text PDFs directly from EPO OPS, falling back to FreePatentsOnline or Google Patents via cURL/wget.

### Data Extraction
* **`step_10_extract_peptide_data_using_llm.py`**
    Iterates through the downloaded PDFs. Uploads each document to Google Gemini (`gemini-2.5-pro`) alongside an extraction prompt and JSON schema constraint. Outputs structured data to JSON files mapped to the internal identifiers.



