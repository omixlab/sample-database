import pandas as pd
import os
import time
df_google_patents = pd.read_csv('data/patent_data/google_patents.csv',skiprows=[0])

for r,row in df_google_patents.iterrows():
    patent_id = row['result link'].split('/')[4]
    output_path = "data/patent_data/google_patents/%s.pdf"%(patent_id)
    command = '''curl -sA "Mozilla/5.0" -L "https://patents.google.com/patent/%s/" | grep -oE 'https://patentimages\.storage\.googleapis\.com/[^"]+\.pdf' | head -1 | xargs -I {} wget {} -O "%s"'''%(patent_id, output_path)
    os.system(command)