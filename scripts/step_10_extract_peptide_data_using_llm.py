import os
import sys
import json
import time
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from google import genai
from google.genai import types
import glob

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

papers = glob.glob("data/literature_data/*/*.pdf")
patents =  glob.glob("data/patent_data/*/*.pdf") + glob.glob("data/patent_data/*/*.json")
antimicrobial_databases = glob.glob("data/antimicrobial_databases/papers/*/*.pdf")

rows = []

for extracted_id, extracted_paper_file in enumerate(tqdm(patents + papers + antimicrobial_databases)):
    print(extracted_paper_file)
    if extracted_paper_file.endswith(".pdf"):
        uploaded_file = client.files.upload(file=extracted_paper_file)

        # Updated polling syntax
        file_info = client.files.get(name=uploaded_file.name)
        
        # Pydantic enum fallback logic for state validation
        def get_state(state_obj):
            return getattr(state_obj, 'name', str(state_obj))

        while get_state(file_info.state) == "PROCESSING":
            time.sleep(2)
            file_info = client.files.get(name=uploaded_file.name)

        if get_state(file_info.state) == "FAILED":
            sys.exit(f"Error: Document processing failed on the server for {extracted_paper_file}.")

    if extracted_paper_file.endswith('.json'):
        contents = [prompt_text, open(extracted_paper_file).read()]
    else:
        contents = [prompt_text, uploaded_file]

    # Updated model generation syntax
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=response_schema
        )
    )

    output_json = f'data/llm_data_extraction/raw_extraction_json/{extracted_id}.json'

    rows.append({'extracted_id':extracted_id, 'uploaded_file': extracted_paper_file, 'output_json': output_json})

    json_content = json.dumps(json.loads(response.text), indent=4)

    print(json_content)

    with open(output_json,'w') as writer:
        writer.write(json.dumps(json.loads(response.text), indent=4))
    


    if extracted_paper_file.endswith(".pdf"):
        # Updated delete syntax
        client.files.delete(name=uploaded_file.name)

    pd.DataFrame(rows).to_csv('data/llm_data_extraction/raw_extraction.csv', index=False)