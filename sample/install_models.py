# sample/install_models.py

import os
import sys

def main() -> None:
    """
    Downloads the NLP models the pipeline loads at runtime.

    Fetches the sentence-transformers encoder and the scispaCy ``en_core_sci_sm`` model used by
    :mod:`sample.nlp`. Safe to re-run: each download is skipped when already cached. Not wired into
    a pipeline step -- run it once after creating the environment.

    :returns: ``None``. Each failure is reported and does not abort the remaining downloads.
    """
    print("Downloading required machine learning models for Semantic and Relational extraction...")
    
    # 1. sentence-transformers model
    try:
        from sentence_transformers import SentenceTransformer
        print("Downloading all-MiniLM-L6-v2...")
        _ = SentenceTransformer('all-MiniLM-L6-v2')
        print("[+] Sentence-transformers downloaded.")
    except Exception as e:
        print(f"[-] Failed to download sentence-transformers: {e}")
        
    # 2. spaCy model
    try:
        import spacy
        try:
            spacy.load("en_core_sci_sm")
            print("[+] en_core_sci_sm already downloaded.")
        except OSError:
            print("Downloading en_core_sci_sm...")
            os.system(f"{sys.executable} -m pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz")
    except Exception as e:
        print(f"[-] Failed to setup spaCy models: {e}")

if __name__ == "__main__":
    main()
