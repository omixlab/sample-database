from Bio import SeqIO
from warnings import simplefilter
from tqdm import tqdm
import pandas as pd
import requests
import os
import sys
import json
import time

simplefilter("ignore")

# wrapper for requests with retrying and sleep

def request_or_retry(method, url, data=None, json=None, params=None, timeout=30, max_retries=100, sleep=10, verify=False):
    for _ in range(max_retries):
        if method.upper() == 'POST':
            try:
                return requests.post(url, data=data, json=json, params=params, allow_redirects=True, timeout=timeout, verify=verify)
            except:
                time.sleep(sleep)
        elif method.upper() == 'GET':
            try:
                return requests.get(url, data=data, json=json, params=params, allow_redirects=True, timeout=timeout, verify=verify)
            except:
                time.sleep(sleep)

# ADP database

if '--adp' in sys.argv:

    print("Downloading ADP data")

    os.system("mkdir -p data/antimicrobial_databases/adp")

    request = request_or_retry('get', 'https://aps.unmc.edu/assets/sequences/naturalAMPs_APD2024a.fasta', verify=False)
    if request.status_code == 200:
        with open('data/antimicrobial_databases/adp/peptides.fasta','w') as writer:
            writer.write('\n'.join(request.text.split("\n")[1::]))

    peptides = {record.id:record for record in SeqIO.parse('data/antimicrobial_databases/adp/peptides.fasta','fasta')}

    all_data = []

    for peptide in tqdm(peptides):
        request = request_or_retry('post', 'https://aps.unmc.edu/database/peptide', data={'ID': peptide[2::]}, verify=False)
        df_data = pd.read_html(request.text)[0].T
        all_data.append(df_data)

    ADP_all_data = pd.concat(all_data)
    ADP_all_data.to_csv('data/antimicrobial_databases/adp/adb.csv', index=False)

# DBAASP database

if '--dbaasp' in sys.argv:

    print("Downloading DBAASP data")

    os.system("mkdir -p data/antimicrobial_databases/dbaasp/")
    dbaasp_url = 'https://dbaasp.org/'
    done = 0
    limit = 10

    get_total_peptides = request_or_retry('get', "https://dbaasp.org/peptides", params={'limit': limit}).json()['totalCount']
    
    for offset in tqdm(range(0, get_total_peptides, limit)):

        data = request_or_retry('get', "https://dbaasp.org/peptides", params={'limit': limit, 'offset': offset}).json()['data']
        for data_row in data:
            dbaasp_id = data_row.get('dbaaspId')
            data_row_complete = request_or_retry("get",  "https://dbaasp.org/peptides/" + dbaasp_id).json()
            ### ARRUMAR ISSO
            with open(f'data/antimicrobial_databases/dbaasp/{dbaasp_id}.json', 'w') as writer:
                writer.write(json.dumps(data_row_complete, indent=4))

if '--dramp' in sys.argv:

    print("Downloading DRAMP data")
    os.system("mkdir -p data/antimicrobial_databases/dramp/")

    urls = {
        'Datasets': [
            'http://dramp.cpu-bioinfor.org/downloads/download.php?filename=download_data/DRAMP3.0_new/general_amps.txt',
            'http://dramp.cpu-bioinfor.org/downloads/download.php?filename=download_data/DRAMP3.0_new/patent_amps.txt',
            'http://dramp.cpu-bioinfor.org/downloads/download.php?filename=download_data/DRAMP3.0_new/clinical_amps.xlsx',
            'http://dramp.cpu-bioinfor.org/downloads/download.php?filename=download_data/DRAMP3.0_new/specific_amps.txt',
            'http://dramp.cpu-bioinfor.org/downloads/download.php?filename=download_data/DRAMP3.0_new/stability_amps.txt',
            'http://dramp.cpu-bioinfor.org/downloads/download.php?filename=download_data/DRAMP3.0_new/expanded_amps.txt'
        ],
        'Activity': [
            'http://dramp.cpu-bioinfor.org/downloads/download.php?filename=download_data/DRAMP3.0_new/Antimicrobial_amps.txt',
            'http://dramp.cpu-bioinfor.org/downloads/download.php?filename=download_data/DRAMP3.0_new/Antibacterial_amps.txt',
            'http://dramp.cpu-bioinfor.org/downloads/download.php?filename=download_data/DRAMP3.0_new/Anti-Gram-positive_amps.txt',
            'http://dramp.cpu-bioinfor.org/downloads/download.php?filename=download_data/DRAMP3.0_new/Anti-Gram-_amps.txt',
            'http://dramp.cpu-bioinfor.org/downloads/download.php?filename=download_data/DRAMP3.0_new/Antifungal_amps.txt',
            'http://dramp.cpu-bioinfor.org/downloads/download.php?filename=download_data/DRAMP3.0_new/Antiviral_amps.txt',
            'http://dramp.cpu-bioinfor.org/downloads/download.php?filename=download_data/DRAMP3.0_new/Anticancer_amps.txt',
            'http://dramp.cpu-bioinfor.org/downloads/download.php?filename=download_data/DRAMP3.0_new/Anti-SARS-CoV-2_amps.txt',
            'http://dramp.cpu-bioinfor.org/downloads/download.php?filename=download_data/DRAMP3.0_new/Antiparasitic_amps.txt',
            'http://dramp.cpu-bioinfor.org/downloads/download.php?filename=download_data/DRAMP3.0_new/Insecticidal_amps.txt'
        ],
        'Subclass': [
            'http://dramp.cpu-bioinfor.org/downloads/download.php?filename=download_data/DRAMP3.0_new/natural_amps.txt',
            'http://dramp.cpu-bioinfor.org/downloads/download.php?filename=download_data/DRAMP3.0_new/synthetic_amps.txt',
            'http://dramp.cpu-bioinfor.org/downloads/download.php?filename=download_data/DRAMP3.0_new/plant_amps.txt',
            'http://dramp.cpu-bioinfor.org/downloads/download.php?filename=download_data/DRAMP3.0_new/stapled_amps.txt',
            'http://dramp.cpu-bioinfor.org/downloads/download.php?filename=download_data/DRAMP3.0_new/candidate_amps.txt',
        ]
    }

    for directory in urls:
        print(directory)
        os.system(f'mkdir -p data/antimicrobial_databases/dramp/{directory}')
        for url in tqdm(urls[directory]):
            request = request_or_retry('get', url).text
            filename,fileext = os.path.splitext(url.split('/')[-1])
            with open(f'data/antimicrobial_databases/dramp/{directory}/{filename}{fileext}','w') as writer:
                writer.write(request)

if '--dbamp' in sys.argv:

    print("Downloading DBAMP data")
    os.system("mkdir -p data/antimicrobial_databases/dbamp/")

    urls = {
        'sequences': [
            'https://ycclab.cuhk.edu.cn/dbAMP/download/3.0/dbAMP2024.zip'
        ],
        'functional_activity_data': [
            'https://ycclab.cuhk.edu.cn/dbAMP/download/3.0/activity/dbAMP_Antibacterial.fasta',
            'https://ycclab.cuhk.edu.cn/dbAMP/download/3.0/activity/dbAMP_AntiGram_p.fasta',
            'https://ycclab.cuhk.edu.cn/dbAMP/download/3.0/activity/dbAMP_AntiGram_n.fasta',
            'https://ycclab.cuhk.edu.cn/dbAMP/download/3.0/activity/dbAMP_Antifungal.fasta',
            'https://ycclab.cuhk.edu.cn/dbAMP/download/3.0/activity/dbAMP_Antiviral.fasta',
            'https://ycclab.cuhk.edu.cn/dbAMP/download/3.0/activity/dbAMP_Antiparasitic.fasta',
            'https://ycclab.cuhk.edu.cn/dbAMP/download/3.0/activity/dbAMP_AntiHIV.fasta',
            'https://ycclab.cuhk.edu.cn/dbAMP/download/3.0/activity/dbAMP_WoundHealing.fasta',
            'https://ycclab.cuhk.edu.cn/dbAMP/download/3.0/activity/dbAMP_Chemotactic.fasta',
            'https://ycclab.cuhk.edu.cn/dbAMP/download/3.0/activity/dbAMP_EnzymeInhibitor.fasta',
            'https://ycclab.cuhk.edu.cn/dbAMP/download/3.0/activity/dbAMP_Antimalarial.fasta',
            'https://ycclab.cuhk.edu.cn/dbAMP/download/3.0/activity/dbAMP_Antimicrobial.fasta',
            'https://ycclab.cuhk.edu.cn/dbAMP/download/3.0/activity/dbAMP_Antitumour.fasta',
            'https://ycclab.cuhk.edu.cn/dbAMP/download/3.0/activity/dbAMP_MammalianCells.fasta',
            'https://ycclab.cuhk.edu.cn/dbAMP/download/3.0/activity/dbAMP_SurfaceImmobilized.fasta',
            'https://ycclab.cuhk.edu.cn/dbAMP/download/3.0/activity/dbAMP_Antibiofilm.fasta',
            'https://ycclab.cuhk.edu.cn/dbAMP/download/3.0/activity/dbAMP_Antioxidant.fasta',
            'https://ycclab.cuhk.edu.cn/dbAMP/download/3.0/activity/dbAMP_Antiprotozoal.fasta',
            'https://ycclab.cuhk.edu.cn/dbAMP/download/3.0/activity/dbAMP_Spermicidal.fasta',
            'https://ycclab.cuhk.edu.cn/dbAMP/download/3.0/activity/dbAMP_Insecticidal.fasta',
            'https://ycclab.cuhk.edu.cn/dbAMP/download/3.0/activity/dbAMP_Antiyeast.fasta',
            'https://ycclab.cuhk.edu.cn/dbAMP/download/3.0/activity/dbAMP_Antiinflammatory.fasta',
            'https://ycclab.cuhk.edu.cn/dbAMP/download/3.0/activity/dbAMP_AntiMRSA.fasta'
        ],
        'sequences_by_specie_groups': [
            'https://ycclab.cuhk.edu.cn/dbAMP/download/3.0/category/dbAMP_bacteria.fasta',
            'https://ycclab.cuhk.edu.cn/dbAMP/download/3.0/category/dbAMP_fungi.fasta'
        ]
    }

    for directory in urls:
        if directory == 'sequences':
            for url in urls[directory]:
                filename = url.split("/")[-1]
                if url.endswith(".zip"):
                    os.chdir("data/antimicrobial_databases/dbamp/")
                    os.system(f"wget -O {filename} {url} > /dev/null 2> /dev/null")
                    os.system(f"unzip -o {filename} > /dev/null 2> /dev/null")
                    os.chdir("../../..")
        else:
            print(f'Downloading records of {directory}')
            for url in tqdm(urls[directory]):
                request = request_or_retry('get', url, verify=False)
                filename = url.split("/")[-1]
                with open(f'data/antimicrobial_databases/dbamp/{filename}','w') as writer:
                    writer.write(request.text)
        
    dbamp_records = list(SeqIO.parse("data/antimicrobial_databases/dbamp/dbAMP2024/dbAMP3.fasta", 'fasta'))

    print('Downloading peptide metadata')

    for record in tqdm(dbamp_records):
        os.system(f"mkdir -p data/antimicrobial_databases/dbamp/extracted_metadata/{record.id}")
        dbamp_record_metadata_tables = pd.read_html(
            request_or_retry('get',f'https://ycclab.cuhk.edu.cn/dbAMP/information.php?db={record.id}').text
        )

        for table_id, dbamp_record_metadata_table in enumerate(dbamp_record_metadata_tables):
            dbamp_record_metadata_table.to_csv(
                f'data/antimicrobial_databases/dbamp/extracted_metadata/{record.id}/{table_id}.csv',
                index=False
            )

# 

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from chromedriver_py import binary_path # this will get you the path variable
import re

chrome_options = Options()
chrome_options.add_argument("user-data-dir=selenium")
driver = webdriver.Chrome(options=chrome_options)

svc = webdriver.ChromeService(executable_path=binary_path)
driver = webdriver.Chrome(service=svc)

def get_from_camp_page(camp_page_url):

    driver.get(camp_page_url)
    loaded = False
    while not loaded:
        loaded = (driver.execute_script('return document.readyState;') == 'complete')
    html = driver.page_source
    with open('teste.html', 'w') as writer:
        writer.write(html)
    for seq_href in get_seqdisp_links(html):
        url = 'camp3.bicnirrh.res.in/'+seq_href
        camp_id = seq_href.split('id=')[1]
        df_page_data = pd.read_html(request_or_retry('get', f'https://camp.bicnirrh.res.in/{seq_href}').text)
        df_metadata = df_page_data[2][1::]
        df_metadata = df_metadata.rename(columns={0: 'key', 1:'value'})
        df_metadata['key'] = df_metadata['key'].map(lambda key: str(key).strip(":").strip(' '))
        df_metadata.to_csv(f'data/antimicrobial_databases/camp/{camp_id}.csv', index=False)

if '--camp' in sys.argv:

    os.system('mkdir -p data/antimicrobial_databases/camp/')

    def get_seqdisp_links(html_content: str) -> list[str]:
        return re.findall(r'href="(seqDisp\.php\?id=[^"]+)"', html_content)

    url = f'https://camp.bicnirrh.res.in/seqDb.php?page=0'

    # Example limited to 1 for testing; change back to range(0, 1000) for full execution
    for page in tqdm(range(0, 1000)):

        for _ in range(100):
            try:
                get_from_camp_page(url)
                break
            except:
                time.sleep(10)

        url = f'https://camp.bicnirrh.res.in/seqDb.php?page={page+1}'