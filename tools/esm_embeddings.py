#!/usr/bin/env python3
"""
Generates ESM3 protein embeddings and saves them as .npy array files, one with the embeddings and one with ids.

This script requires a huggingface token to use.
Input is a csv with the columns "protein_id" and "sequence".
"""

import argparse, logging, tqdm
import pandas as pd
import numpy as np
from pathlib import Path
from huggingface_hub import login
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig


def setup_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )


def get_esm_embedding(sequence, client):
    """Generates ESM3 embeddings for a given protein sequence."""
    protein = ESMProtein(sequence=sequence)
    protein_tensor = client.encode(protein)

    result = client.forward_and_sample(
        protein_tensor,
        SamplingConfig(return_per_residue_embeddings=True, return_mean_embedding=True),
    )
    return result


def process_sequences(df, client, outputdir, loop):
    """
    Generate ESM3 embeddings for sequences (limited to length 1500) in the input DataFrame.
    Saves embeddings and ids to .npy files.
    """
    ids = []
    embeddings = []

    for index, row in tqdm.tqdm(
        df.iterrows(), total=len(df), desc="Processing protein sequences"
    ):
        seq = row["sequence"]
        protein_id = row["protein_id"][:1500]

        try:
            embedding = get_esm_embedding(seq, client)
            ids.append(protein_id)
            embeddings.append(embedding.mean_embedding.cpu())
        except Exception as e:
            logging.error(f"Failed to processing sequence {protein_id}: {e}")

    np.save(outputdir / f"{loop}_embedding_id.npy", np.array(ids))
    np.save(outputdir / f"{loop}_embedding.npy", np.array(embeddings))
    logging.info(f"Saved {len(ids)} embeddings to {outputdir}/{loop}_")


def main():
    parser = argparse.ArgumentParser(
        description="Generate ESM3 embeddings for protein sequences."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Path to input TSV with 'protein_id' and 'sequence' columns.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("./data/embeddings"),
        help="Output directory for embedding files. ['./data/embeddings']",
    )
    parser.add_argument(
        "-t",
        "--token",
        type=str,
        required=True,
        help="Hugging Face token for authentication.",
    )
    parser.add_argument(
        "-c",
        "--chunksize",
        type=int,
        required=False,
        default=10000,
        help="How many sequences should be processed per file. [10000]",
    )

    args = parser.parse_args()
    setup_logger()

    # Login and load model
    login(token=args.token)
    logging.info("Logged into Hugging Face Hub.")
    client = ESM3.from_pretrained("esm3-open").to("cuda")
    logging.info("Loaded ESM3 model.")

    # Read input
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    df = pd.read_csv(args.input, sep="\t")

    if not {"protein_id", "sequence"}.issubset(df.columns):
        raise ValueError(
            "Input DataFrame must contain 'protein_id' and 'sequence' columns."
        )

    # Create output dir if it doesnt exist
    args.output.mkdir(parents=True, exist_ok=True)

    # Process sequences
    chunk_size = args.chunksize
    chunks = [df.iloc[i : i + chunk_size] for i in range(0, len(df), chunk_size)]
    logging.info(f"Processing {len(df)} sequences in {len(chunks)} chunks.")
    for i, chunk in enumerate(chunks):
        process_sequences(chunk, client, args.output, i)
        logging.info(
            f"Processed {(i + 1) * chunk_size} out of {len(df)} sequences total."
        )


if __name__ == "__main__":
    main()
