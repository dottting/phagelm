#!/bin/bash

input_dir="data/protein_structures"
output_dir="data/protein_dssp"

mkdir -p "$output_dir"

for pdb_file in "$input_dir"/*.pdb; do
    base_name=$(basename "$pdb_file" .pdb)

    output_file="$output_dir/$base_name.dssp"

    # Run mkdssp
    mkdssp "$pdb_file" "$output_file"

    echo "Processed $pdb_file -> $output_file"
done