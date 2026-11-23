import os
import re
import logging
import pandas as pd
from typing import List, Dict, Any, Optional

logger = logging.getLogger("sample.pdf_tables")

def extract_tables_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extracts structured 2D tables from PDF pages using pdfplumber or PyMuPDF,
    identifying antimicrobial activity tables (MIC, IC50, ZOI, MBC) and converting
    rows into structured biological activity extractions.
    """
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        return []

    extracted_records = []
    
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for table in tables:
                    if not table or len(table) < 2:
                        continue
                        
                    # Clean table cells
                    cleaned_table = []
                    for row in table:
                        cleaned_row = [str(cell).strip() if cell is not None else "" for cell in row]
                        if any(cleaned_row):
                            cleaned_table.append(cleaned_row)
                            
                    if len(cleaned_table) < 2:
                        continue
                        
                    # Header detection
                    headers = [h.lower() for h in cleaned_table[0]]
                    header_str = " ".join(headers)
                    
                    # Check if table is an antimicrobial assay table
                    is_assay_table = any(kw in header_str for kw in ["mic", "ic50", "mbc", "mfc", "zoi", "zone", "inhibition", "ug/ml", "um", "organism", "strain", "peptide", "sequence"])
                    if not is_assay_table:
                        continue
                        
                    # Identify column indices
                    seq_col_idx = -1
                    pep_name_col_idx = -1
                    
                    for idx, h in enumerate(headers):
                        if any(k in h for k in ["sequence", "seq", "structure"]):
                            seq_col_idx = idx
                        elif any(k in h for k in ["peptide", "name", "compound", "analogue", "code"]):
                            pep_name_col_idx = idx
                            
                    # Process data rows
                    for row in cleaned_table[1:]:
                        if len(row) != len(headers):
                            continue
                            
                        seq_val = row[seq_col_idx] if seq_col_idx != -1 and seq_col_idx < len(row) else ""
                        name_val = row[pep_name_col_idx] if pep_name_col_idx != -1 and pep_name_col_idx < len(row) else ""
                        
                        # Search for peptide sequences in the row
                        from sample.nlp import is_valid_sequence
                        cleaned_seq = seq_val.replace(" ", "").replace("-", "").upper()
                        
                        if not is_valid_sequence(cleaned_seq):
                            # Try to find sequence in other cells of the row
                            cleaned_seq = ""
                            for cell in row:
                                potential = cell.replace(" ", "").replace("-", "").upper()
                                if len(potential) >= 5 and is_valid_sequence(potential):
                                    cleaned_seq = potential
                                    break
                                    
                        if not cleaned_seq and not name_val:
                            continue
                            
                        # Extract activities from remaining columns
                        activities = []
                        for col_idx, (h, cell_val) in enumerate(zip(headers, row)):
                            if col_idx in (seq_col_idx, pep_name_col_idx) or not cell_val:
                                continue
                                
                            # Check if cell contains numerical activity value
                            from sample.nlp import parse_activity_value_fast, infer_activity_group
                            op, v_low, v_high = parse_activity_value_fast(cell_val)
                            if v_low is not None or v_high is not None or re.search(r'\d+', cell_val):
                                # Infer metric type and organism from column header
                                metric_type = "MIC"
                                if "ic50" in h: metric_type = "IC50"
                                elif "ec50" in h: metric_type = "EC50"
                                elif "mbc" in h: metric_type = "MBC"
                                elif "zoi" in h or "zone" in h: metric_type = "Growth Inhibition"
                                
                                act_grp = infer_activity_group(h, metric_type)
                                
                                # The unit is READ from the column header, never assumed. This
                                # stamped every activity "ug/mL" regardless of what the header
                                # said, so a "MIC ( uM)" column -- which this corpus really does
                                # contain -- was recorded in micrograms per millilitre, wrong by
                                # roughly a factor of 1000 for a typical peptide. Omitting the
                                # field when the header declares no unit is honest; a consumer can
                                # tell "unknown" from "micromolar", but cannot tell a wrong unit
                                # from a right one.
                                from sample.docling_tables import unit_from_header
                                activity = {
                                    "activity_group": act_grp,
                                    "target_organism": h.strip().title(),
                                    "activity_type": metric_type,
                                    "value_raw": cell_val,
                                    "operator": op,
                                    "value_lower": v_low,
                                    "value_upper": v_high,
                                }
                                unit = unit_from_header(h)
                                if unit:
                                    activity["activity_unit"] = unit
                                activities.append(activity)
                                
                        if cleaned_seq or name_val:
                            from sample.nlp import extract_context_window
                            page_text = page.extract_text() or ""
                            target_term = cleaned_seq or name_val
                            rec = {
                                "peptide_name": name_val if name_val else None,
                                "peptide_sequence": cleaned_seq if cleaned_seq else None,
                                "biological_activities": activities,
                                "context": extract_context_window(page_text, target_term=target_term, token_window=1000),
                                "source_page": page_idx + 1
                            }
                            extracted_records.append(rec)
                            
    except Exception as e:
        logger.error(f"Error extracting tables from PDF '{pdf_path}': {e}")
        
    return extracted_records
