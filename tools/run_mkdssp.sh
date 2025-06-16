#!/bin/bash

# input is relative to execution location
input_dir="classification_model/data/protein_structures"
output_dir="classification_model/data/protein_dssp"

mkdir -p "$output_dir"

# Get total number of .pdb files
total=$(ls "$input_dir"/*.pdb 2>/dev/null | wc -l)
count=0

for pdb_file in "$input_dir"/*.pdb; do
    base_name=$(basename "$pdb_file" .pdb)
    output_file="$output_dir/$base_name.dssp"

    # Run mkdssp
    mkdssp "$pdb_file" "$output_file"

    ((count++))
    echo "Processed $pdb_file -> $output_file"
    echo "$count of $total done"
done