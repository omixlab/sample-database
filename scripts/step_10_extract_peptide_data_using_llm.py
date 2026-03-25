import os
import sys
import json
import time
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

api_key = os.environ.get("GEMINI_KEY")
if not api_key:
    sys.exit("Error: GEMINI_KEY environment variable not set.")

client = genai.Client(api_key=api_key)

# Fix: Read file content into a string, do not pass the IO object to the model
with open('data/prompts/step_04/01_extraction_prompt.txt', 'r', encoding='utf-8') as f:
    prompt_text = f.read()

# Fix: Parse the JSON file into a Python dictionary. json.loads() on a file path string raises JSONDecodeError.
with open('data/prompts/step_04/01_extraction_prompt_response_format.json', 'r', encoding='utf-8') as f:
    response_schema = json.load(f)

for source in ['pubmed', 'scopus']:
    print(f'extracting data using LLM for {source} papers')
    df_source = pd.read_csv('data/literature_data/pubmed_downloaded.csv')
    source_rows = []

    for r, row in tqdm(df_source.iterrows(), total=df_source.shape[0]):
        if pd.isna(row['pdf_path']):
            continue
        if not os.path.isfile(row['pdf_path']):
            continue
        # Updated upload syntax
        uploaded_file = client.files.upload(file=row['pdf_path'])

        # Updated polling syntax
        file_info = client.files.get(name=uploaded_file.name)
        
        # Pydantic enum fallback logic for state validation
        def get_state(state_obj):
            return getattr(state_obj, 'name', str(state_obj))

        while get_state(file_info.state) == "PROCESSING":
            time.sleep(2)
            file_info = client.files.get(name=uploaded_file.name)

        if get_state(file_info.state) == "FAILED":
            sys.exit(f"Error: Document processing failed on the server for {row['pdf_path']}.")

        # Updated model generation syntax
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=[uploaded_file, prompt_text],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=response_schema
            )
        )

        with open(f'data/llm_data_extraction/literature_data/{source}/{row.internal_id}.json','w') as writer:
            print(response.text)
            writer.write(json.dumps(json.loads(response.text), indent=4))
        
        # Updated delete syntax
        client.files.delete(name=uploaded_file.name)