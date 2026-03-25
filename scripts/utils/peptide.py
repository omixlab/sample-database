import re

def sequence_to_helm(sequence):
    # 1. Input Sanitization
    if not sequence or sequence in ["None", "/", "O", "OO", "OOO", "OOOO", "OOOOO"]:
        return ""
    sequence = str(sequence).strip()
    
    # Auto-correct malformed multipliers (e.g., missing opening parenthesis)
    if sequence.count(')') > sequence.count('('):
        sequence = '(' + sequence

    # 2. Multiplier Expansion
    # Recursively expand (Sequence)N to SequenceSequence...
    def expand_match(match):
        seq_block = match.group(1)
        multiplier = int(match.group(2))
        return "".join([seq_block] * multiplier)

    while re.search(r"\(([^)]+)\)(\d+)", sequence):
        sequence = re.sub(r"\(([^)]+)\)(\d+)", expand_match, sequence)
        
    # Strip residual formatting parentheses
    sequence = sequence.replace('(', '').replace(')', '')

    # 3. Topology Splitting
    # Comma indicates separate chains; Asterisk indicates branching
    chain_strings = re.split(r'([,*])', sequence)
    
    helm_chains = []
    connections = []
    parsed_chains = []
    operators = []
    
    chain_idx = 1
    standard_aa = set("ACDEFGHIKLMNPQRSTVWY")
    
    # Tokenizer identifies standard AAs, unnatural numeric AAs (C10R, S5), termini, and staple hyphens
    token_regex = re.compile(r"(C\d+R?|RC\d+|[A-Z]\d+|-NH2|-OH|NH2|OH|[A-Z]|\?|-)")
    
    # 4. Monomer Parsing & Chain Construction
    for part in chain_strings:
        if part in ['*', ',']:
            operators.append(part)
            continue
        if not part.strip():
            continue
            
        tokens = token_regex.findall(part)
        monomers = []
        monomer_idx = 1
        chain_id = f"PEPTIDE{chain_idx}"
        
        for token in tokens:
            if token == '-':
                # Bypass staple logic if hyphen follows an N-terminal lipid (C followed by numbers only)
                if monomer_idx == 2 and re.fullmatch(r"C\d+", tokens[0]):
                    continue
                # Standard side-chain staple logic
                if monomer_idx > 1:
                    connections.append(f"{chain_id},{chain_id},{monomer_idx-1}:R3-{monomer_idx}:R3")
                continue
            
            # Map terminal modifications and un-natural amino acids
            if token in ["-NH2", "NH2"]:
                monomers.append("[am]")
            elif token in ["-OH", "OH"]:
                monomers.append("[OH]")
            elif token in standard_aa:
                monomers.append(token)
            else:
                # Wrap custom components in HELM bracket syntax
                monomers.append(f"[{token}]")
                
            monomer_idx += 1
            
        if monomers:
            helm_chains.append(f"{chain_id}{{{'.'.join(monomers)}}}")
            parsed_chains.append(chain_id)
            chain_idx += 1

    # 5. Cross-Chain Linkage Assembly
    for i, op in enumerate(operators):
        if i + 1 < len(parsed_chains):
            chain_a = parsed_chains[i]
            chain_b = parsed_chains[i+1]
            if op == '*':
                # Default branch geometry: Chain A sidechain to Chain B N-terminus
                connections.append(f"{chain_a},{chain_b},3:R3-1:R1")

    # 6. Final String Compilation
    helm_str = "|".join(helm_chains)
    conn_str = "|".join(connections)
    
    if not helm_str:
        return ""
        
    return f"{helm_str}${conn_str}$$$"