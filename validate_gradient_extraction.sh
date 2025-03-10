#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <directory>"
  exit 1
fi
echo "This tool will prompt you with all files that have unusual file sizes (likely due to the gradient extraction script crashing during file transfer)."
echo "searching ..."
directory="$1"

if [ ! -d "$directory" ]; then
  echo "Error: '$directory' not a dir"
  exit 1
fi


file_sizes=$(find "$directory" -type f -exec stat --format="%s" {} \;)
size_counts=$(echo "$file_sizes" | sort | uniq -c)


sizes_to_filter=$(echo "$size_counts" | awk '$1 < 3 {print $2}')


for file in $(find "$directory" -type f); do

  size=$(stat --format="%s" "$file")

  if echo "$sizes_to_filter" | grep -q "^$size$" && [ "$(basename "$file")" != "mean" ]; then # at least 10 times this size in dir?

    echo "File: $(realpath "$file")"
    echo "Size: $size bytes"


    read -p "Delete ? (y/n): " input
    echo $input
    if [[ $input == "Y" || $input == "y" ]]; then
      rm "$file"
      echo "File deleted: $(realpath "$file")"
    else
      echo "File not deleted: $(realpath "$file")"
    fi
    echo "..."
  fi
done
echo "done: no more unusual file sizes"