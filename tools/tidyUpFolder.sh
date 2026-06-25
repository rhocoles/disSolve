#!/bin/bash

if [[ $# -eq 0 ]] ; then
    echo 'dont you want to set the step number?'
    exit 0
fi

echo 'processing the results of step' $1
mkdir -p dataStep$1
cwd=$(pwd)

mv data_* ./dataStep$1/

file="experiment_overview.txt"

#orga_info is a line that stores the following info parallel_process_number energy frameNumber temparature overlapratio eta

echo "lowest energy scores:"
while IFS= read -r line
do
    read -ra orga_info <<< "$line"
    cp "./polyFiles/test${orga_info[0]}_${orga_info[2]}.poly" .
    cp "./polyFiles/test${orga_info[0]}__${orga_info[2]}.txt" .
    echo "test ${orga_info[0]} is of lowest energy ${orga_info[1]} at temp ${orga_info[3]}"

done < "$file"

mv "polyFiles" ./dataStep$1/
mv screenlog.0 dataStep$1

cp results.ipynb results_step$1.ipynb
today=$(date +"%d-%m-%Y")
mv experiment_overview.txt experiment_step$1_$today

zip -r dataStep$1 dataStep$1/

rm -r __pycache__
rm -r morph_local
rm -r *.py
rm -r dataStep$1
