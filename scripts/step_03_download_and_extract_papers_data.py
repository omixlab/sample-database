from utils.connection import CheckThereIsConnection
import pandas as pd
import os
import subprocess
import re
from tqdm import tqdm
import multiprocessing as mp
import hashlib
import uuid
import random
import string
import json
import datetime
import time

EMAILS = [
    'fred.s.kremer@gmail.com',
    'fredericok.cdtec@ufpel.edu.br',
    'fred.s.kremer@hotmail.com'
]

def download_paper(identifier, filename, directory, timeout):
    email = random.choice(EMAILS)
    output_path = os.path.join(directory, filename)
    os.system(
        f'timeout {timeout} python scripts/utils/download_pdf.py --doi "{identifier}" --output {directory}/{filename} --email {email}'
    )
    if os.path.isfile(output_path):
        return True
    else:
        return False

uuid_sample = uuid.uuid1()

internal_ids = set([])

for source in ['pubmed', 'scopus']:

    try:
        df_downloaded = pd.read_csv(f"data/literature_data/{source}_downloaded.csv")
    except:
        df_downloaded = pd.DataFrame(columns=[
            'source','title','doi','pmid','link','internal_id','success,pdf_path'
        ])

    
    for internal_id in df_downloaded['internal_id']:
        internal_ids.add(internal_id)

    print(f"Downloading PDF of {source} papers")
    metadata_csv = f'data/literature_data/{source}_metadata.csv'
    df_metadata = pd.read_csv(metadata_csv)
    metadata_rows = []
    for r,row in tqdm(df_metadata.iterrows(), total=df_metadata.shape[0]):
        if pd.isna(row['doi']):
            continue
        if row['doi'] in df_downloaded['doi'].values:
            continue
        hash = hashlib.sha1(datetime.datetime.now().strftime("%Y-%m-d-%A-%H-%M-%S-%f-%z").encode()).hexdigest()
        row['internal_id'] = "%s-%s-%s-%s-%s"%(hash[0:8], hash[8:12], hash[12:16], hash[16:20], hash[20::])
        for i in range(3):        
            success = download_paper(row['doi'], filename=f'{row.internal_id}.pdf', directory=f'data/literature_data/{source}/', timeout=60)
            if not success and not CheckThereIsConnection():
                time.sleep(60)
            if success:
                break
        output_file = os.path.join(f'data/literature_data/{source}/', f'{row.internal_id}.pdf')      
        if os.path.isfile(output_file):
            row['pdf_path'] = f'data/literature_data/{source}/{row.internal_id}.pdf'
        row['success'] = success
        metadata_rows.append(row)
        df_metadata_updated = pd.concat(
            [
                df_downloaded,
                pd.DataFrame(metadata_rows)
            ]
        )
        df_metadata_updated.drop_duplicates(subset=['doi'], keep='last', inplace=True)
        df_metadata_updated.to_csv(f"data/literature_data/{source}_downloaded.csv",index=False)
        