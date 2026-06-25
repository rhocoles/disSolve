#!/bin/bash

#geometryType="Biarcs"
geometryType="ThreadedBeads"

structure='test'

inputFile='/Users/harmon/programming/disSolve/program/current/disSolve/src/test/polyFiles/test_2_13.poly'    # ---> each parallel process gets the same initial configuration

overlapRatio=0.3
eta=0.0
alpha=0.0

numberParallelProcesses=3

#annealing parameters for the decreasing temp part
numberSecondsPerTemp=800
numberOfRounds=18
T_top=2.0
T_bot=0.1

# check current director
dir_origin=${PWD}

# first make an directory .../structure/..
dir=$dir_origin/${structure}
#echo $dir

mkdir -p $dir
cd $dir

cp ${dir_origin}/main.py .
cp ${dir_origin}/geometryClass.py .
cp ${dir_origin}/simple_functions.py .
cp ${dir_origin}/morphometry.py .
cp ${dir_origin}/libmorphometry.so .
cp ${dir_origin}/self_distance_c.py .
cp ${dir_origin}/libself_distance_c.so .
cp ${dir_origin}/pointFilaments.py .

#may be that you don't want to do this...
rm -rf polyFiles
mkdir polyFiles
rm -rf screenlog.0
rm -rf data
mkdir data

mpirun -np $numberParallelProcesses python3.9 main.py  $overlapRatio $eta $alpha $T_top $T_bot $numberSecondsPerTemp $numberOfRounds $structure $inputFile

#screenExperimentName=${test}
#screen -S ${screenExperimentName} -L -d -m bash -lc "
#mpirun -np $numberParallelProcesses ~/miniconda3/bin/python3 main_parallel_annealing.py \
#$overlapRatio $eta $alpha $T_top $T_bot $numberSecondsPerTemp $numberOfRounds $structure $inputFile
#python3 evaluate_experiment.py
#"
