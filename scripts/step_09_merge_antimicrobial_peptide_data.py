from utils.peptide import sequence_to_helm
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from Bio import SeqIO
import glob
import pandas as pd
import json

# ADP database

df_adp_peptides = pd.DataFrame(
    [
        {'adp_id':record.id, 'sequence':str(record.seq).upper()} 
        for record 
        in SeqIO.parse('data/antimicrobial_databases/adp/peptides.fasta', 'fasta')
    ]
)

# CAMP database

camp_data_rows = []

for csv_file in glob.glob("data/antimicrobial_databases/camp/*.csv"):
    camp_id = csv_file.split('/')[3].split('.')[0]
    df_camp_entry = pd.read_csv(csv_file).T
    data_row = {'camp_id': camp_id, **dict(zip(df_camp_entry.iloc[0], df_camp_entry.iloc[1]))}
    camp_data_rows.append(data_row)

df_camp_peptides = pd.DataFrame(camp_data_rows)

# DBAMP database

dbamp_data_rows = []

for csv_file in glob.glob("data/antimicrobial_databases/dbamp/extracted_metadata/*/1.csv"):
    dbamp_id = csv_file.split('/')[4]
    df_dbamp_entry = pd.read_csv(csv_file).T
    data_row = {'dbamp_id':dbamp_id, **dict(zip(df_dbamp_entry.iloc[0], df_dbamp_entry.iloc[1]))}
    dbamp_data_rows.append(data_row)

df_dbamp_peptides = pd.DataFrame(dbamp_data_rows)

# DRAMP database

df_dramp_peptides = pd.concat([
    pd.read_csv(csv_file, sep='\t') for csv_file in glob.glob('data/antimicrobial_databases/dramp/Datasets/*.txt')
])

df_dramp_peptides = df_dramp_peptides.drop_duplicates(subset=['DRAMP_ID'])

# DBAASP database

df_dbaasp_peptides = pd.DataFrame([
    json.loads(open(json_file).read()) for json_file in glob.glob("data/antimicrobial_databases/dbaasp/*.json")
])

# Merge Normalized Datasets

df_adp_peptides_normalized                   = pd.DataFrame()
df_adp_peptides_normalized['internal_id']    = list(df_adp_peptides['adp_id'].values)
df_adp_peptides_normalized['source']         = 'ADP'
df_adp_peptides_normalized['sequence']       = list(df_adp_peptides['sequence'].values)

df_camp_peptides_normalized                  = pd.DataFrame()
df_camp_peptides_normalized['internal_id']   = list(df_camp_peptides['camp_id'])
df_camp_peptides_normalized['source']        = 'CAMP'
df_camp_peptides_normalized['sequence']      = list(df_camp_peptides['Sequence'].values)

df_dbamp_peptides_normalized                 = pd.DataFrame()
df_dbamp_peptides_normalized['internal_id']  = list(df_dbamp_peptides['dbamp_id'])
df_dbamp_peptides_normalized['source']       = 'DBAMP'
df_dbamp_peptides_normalized['sequence']     = list(df_dbamp_peptides['Sequence'].values)

df_dramp_peptides_normalized                 = pd.DataFrame()
df_dramp_peptides_normalized['internal_id']  = list(df_dramp_peptides['DRAMP_ID'].values)
df_dramp_peptides_normalized['source']       = 'DRAMP'
df_dramp_peptides_normalized['sequence']     = list(df_dramp_peptides['Original_Sequence'].values)

df_dbaasp_peptides_normalized                = pd.DataFrame()
df_dbaasp_peptides_normalized['internal_id'] = list(df_dbaasp_peptides['dbaaspId'].values)
df_dbaasp_peptides_normalized['source']      = 'DBAASP'
df_dbaasp_peptides_normalized['sequence']    = list(df_dbaasp_peptides['sequence'].values)

df_consensus_datasets = pd.concat(
    [
        df_adp_peptides_normalized,
        df_camp_peptides_normalized,
        df_dbamp_peptides_normalized,
        df_dramp_peptides_normalized,
        df_dbaasp_peptides_normalized
    ], ignore_index=True
)

df_consensus_datasets = df_consensus_datasets.drop_duplicates(subset=['sequence', 'source'])
df_consensus_datasets['sequence'] = df_consensus_datasets['sequence'].str.upper()
df_consensus_datasets['sequence'] = df_consensus_datasets['sequence'].replace(' ', '').replace('\t', '')

ID_PREFIX='SAMPLE'
ID_ZFILL=10

consensus_dataset_rows = []

ENTRY_COUNT = 0

for _, (sequence, df_sources) in enumerate(tqdm(df_consensus_datasets.groupby(by=['sequence']), total=len(df_consensus_datasets['sequence'].unique()))):

    entry_data = {}
    entry_data['entry_id']=None
    entry_data['sequence']=sequence[0].strip(',').replace(' ', '').replace('\t', '')
    for r,row in df_sources.iterrows():
        entry_data[row.source] = row.internal_id
    mol = Chem.MolFromSequence(entry_data['sequence'])

    if 'X' in entry_data['sequence']:
        continue

    if not mol:
        helm = sequence_to_helm(entry_data['sequence'])
        mol = Chem.MolFromHELM(helm)

    if not mol:
        continue

    ENTRY_COUNT += 1

    entry_id=ID_PREFIX+str(ENTRY_COUNT).zfill(ID_ZFILL)

    #with Chem.SDWriter(f'data/raw_release/{entry_id}.sdf') as writer:
    #    writer.write(mol)
    
    entry_data['entry_id']=entry_id
    entry_data['SMILES']=Chem.MolToSmiles(mol)

    consensus_dataset_rows.append(entry_data)

df_consensus_dataset = pd.DataFrame(consensus_dataset_rows)
df_consensus_dataset.to_csv("data/sample_raw_release/merged_from_database.csv", index=False)