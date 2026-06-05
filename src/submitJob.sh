#!/bin/bash

#geometryType="Biarcs"
geometryType="ThreadedBeads"

structure='test'

inputFile='/Users/harmon/programming/disSolve/program/current/disSolve/src/test_1.txt'    # ---> each parallel process gets the same initial configuration

overlapRatio=0.08
eta=0.05

numberParallelProcesses=10

#temperature range
T_top=1.0
T_bot=0.1

#annealing parameters
time_to_swop=3600
total_number_rounds=12

# check current director
dir_origin=${PWD}

# first make an directory .../structure/..
dir=$dir_origin/${structure}
#echo $dir

mkdir -p $dir
cd $dir

#echo "an experiment with ${structure}"
cp ${dir_origin}/main_parallelTempering.py .
cp ${dir_origin}/geometryClass.py .
cp ${dir_origin}/simple_functions.py .
cp ${dir_origin}/morphometry.py .
cp ${dir_origin}/libmorphometry.so .
cp ${dir_origin}/biarcs.py .
cp ${dir_origin}/pointFilaments.py .

#may be that you don't want to do this...
rm -rf polyFiles
mkdir polyFiles
rm -rf screenlog.0
rm -rf data
mkdir data


#screen -S ${structure:0:3}_rs0_${overlapRatio:2:3}_eta0_${eta:2:3} -L -d -m mpirun -np $numberParallelProcesses python3 main.py $structure $T_0 $overlapRatio $eta $T_step $numberSecondsPerTemp $numberOfRounds $varyT $numberRoundsVaryT $numberSecondsBetweenUpdatingTempByVaryT $inputFile
mpirun -np $numberParallelProcesses python3.9 main_parallelTempering.py $overlapRatio $eta $T_top $T_bot $time_to_swop $total_number_rounds $inputFile
