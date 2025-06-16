"""
Generates ESM3 protein embeddings and saves them as .npy array files.

This scrit requires a huggingface token to use. Input is a csv with the columns "id" and "sequence".
"""

import pickle
import pandas as pd
from pathlib import Path
from huggingface_hub import login
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig, GenerationConfig
import tqdm
import numpy as np


login(token="")
client = ESM3.from_pretrained("esm3-open").to("cuda")

df = pd.read_csv("./classification_model/data/proteins.csv")  # Input file


def get_esm_embedding(sequence):
    protein = ESMProtein(sequence=sequence)
    protein_tensor = client.encode(protein)

    result = client.forward_and_sample(
        protein_tensor,
        SamplingConfig(return_per_residue_embeddings=True, return_mean_embedding=True),
    )

    return result


def process_sequences(df):
    ids = []
    embeddings = []

    for index, row in tqdm.tqdm(
        df.iterrows(), total=len(df), desc="Processing protein sequences"
    ):
        seq = row["sequence"]
        protein_id = row["id"]

        seq = seq[:1500]  # Limit protein length
        embedding = get_esm_embedding(seq)
        ids.append(protein_id)
        embeddings.append(embedding)

    np.save("embedding_ids.npy", np.array(ids))
    np.save("embeddings.npy", np.array(embeddings))


process_sequences(df)
