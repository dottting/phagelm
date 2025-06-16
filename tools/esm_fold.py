"""
Generates protein structure predictions using ESM3 and saves the resulting .pdb files.

This scrit requires a huggingface token to use. Input is a csv with the columns "id" and "sequence".
"""

import pandas as pd
from pathlib import Path
from huggingface_hub import login
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig, GenerationConfig

df = pd.read_csv("./classification_model/data/proteins.csv")  # Input file


login(token="")
client = ESM3.from_pretrained("esm3-open").to("cuda")

print("Login and model loading successful!")

# Batch folding
outdir = Path("./classification_model/data/protein_structures")
outdir.mkdir(exist_ok=True)

proteins = [ESMProtein(sequence=seq[:1500]) for seq in df["sequence"]]
protein_ids = df["id"].tolist()
configs = [
    GenerationConfig(
        track="structure", schedule="cosine", num_steps=int(len(prot) / 16)
    )
    for prot in proteins
]
for protein, id, config in zip(proteins, protein_ids, configs):
    # Generate for the structure track.
    p = client.generate(protein, config)
    # Now write sequence and structure to PDB files.
    assert isinstance(p, ESMProtein), f"ESMProtein was expected but got {p}"
    p.to_pdb(f"./{outdir}/{id}.pdb")
