#!/bin/sh

# Script used to collect rst-formatted files saved as markdown ones, remove @sphinxdirective lines and save them as rst.

source_dir=$1

for file in $(find "$source_dir" -iname "*.md" -type f ! -path "*/thirdparty/*" ! -path "*/bin/*" ! -path "*/tests/*" ! -path "*/temp/*" ! -path "*/docs/ops/internal/*")
do
    firstline=$(head -n1 "$file")
    if (echo "$firstline" | grep -qo '{#.*}'); then
        echo "Label found in $file file. Cleaning up file and saving as rst..."
        sed -i -E '/^@(end)?sphinxdirective/d' $file
        directory=$(dirname $file)
        filename=$(basename $file .md)
        mv $file "$directory/$filename.rst" 
    else
        echo "Skipped $file"
    fi
done