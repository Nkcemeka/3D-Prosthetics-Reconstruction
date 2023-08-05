#!/bin/bash
: '
Project Author: Chukwuemeka L. Nkama
Date: August 5, 2023

heights.sh is a file that allows the CAD
designer get the height between two or more
lines.
'

usage(){
  cat << EOF
USAGE
-----
./height.sh id1 id2 

EOF
}

# Path to csv file
heights_csv="line_height.csv"

# Check if the CSV file exists
if [ ! -f "$heights_csv" ]; then
  echo "CSV file not found! Run the script: run.sh!"
  exit 1
fi

# Check if two command line arguments are passed
if [ "$#" -ne 2 ]; then
  echo "Number of command line arguments should be two!"
  echo "See USAGE details below!"
  usage
  exit 1
fi

# Get commnad line arguments
id1="$1"
id2="$2"

# Get the first and second id y values from file
y1=$(awk -F, -v id="$id1" '$1 == id {print $2}' "$heights_csv")
y2=$(awk -F, -v id="$id2" '$1 == id {print $2}' "$heights_csv")

# Check if both y coordinates are found
if [ -n "$y1" ] && [ -n "$y2" ]; then
  # calculate the height
  height=$((y2 - y1))

  # Calculate the absolute height
  abs_height=$(awk -v h="$height" 'BEGIN {print (h < 0) ? -h : h}')
  echo "Height between ID $1 and ID $2 is $abs_height"
else
  echo "$y1"
  echo "$y2"
  echo "One or both of the IDs are missing."
fi
