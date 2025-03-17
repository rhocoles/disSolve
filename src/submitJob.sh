#!/bin/bash

#geometryType="Biarcs"
geometryType="ThreadedBeads"

structure='test'

inputFile='/Users/harmon/programming/disSolve/program/current/disSolve/src/test_0__2801.txt' # ---> each parallel process gets the same initial configuration
#inputFile='/Users/harmon/programming/disSolve/program/current/disSolve/src/openChain_dl0_25_50.txt' # ---> each parallel process gets the same initial configuration

overlapRatio=0.12
eta=0.1

numberParallelProcesses=1

#annealing parameters for the decreasing temp part
numberSecondsPerTemp=2400
numberOfRounds=0
T_0=0.5
T_step=0.75

#annealing parameters for the variable T part of experiment --- no swopping
varyT=1 #0 means this doesn't happen, 1 means this does happen.
numberRoundsVaryT=1 #this brings the temp up to prob(med(deltaE>0)) = 0.6, I wouldn't increase this, instead increase the time between temp updates 
numberSecondsBetweenUpdatingTempByVaryT=2400 #updates temp every 40 mins of iterations 

# check current director
dir_origin=${PWD}

# first make an directory .../structure/..
dir=$dir_origin/${structure}
#echo $dir

mkdir -p $dir
cd $dir

#echo "an experiment with ${structure}"
cp ${dir_origin}/main.py .
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


#screen -S ${structure:0:3}_rs0_${overlapRatio:2:3}_eta0_${eta:2:3} -L -d -m mpirun -np $numberParallelProcesses ~/miniconda3/bin/python3 main.py $structure $T_0 $overlapRatio $ropelength $eta $inputFile $T_step $numberSecondsPerTemp $totalNumberOfRounds $varyT $numberRoundsVaryT $numberSecondsBetweenUpdatingTempByVaryT
#screen -S ${structure:0:3}_rs0_${overlapRatio:2:3}_eta0_${eta:2:3} -L -d -m mpirun -np $numberParallelProcesses python3 main.py $structure $T_0 $overlapRatio $eta $T_step $numberSecondsPerTemp $numberOfRounds $varyT $numberRoundsVaryT $numberSecondsBetweenUpdatingTempByVaryT $inputFile
python3.9 main.py $overlapRatio $eta $T_0 $T_step $numberSecondsPerTemp $numberOfRounds $varyT $numberRoundsVaryT $numberSecondsBetweenUpdatingTempByVaryT $structure $inputFile
#mpirun -np 1 python3.9 main.py $inputFile $overlapRatio $eta $T_0 $T_step $numberSecondsPerTemp $numberOfRounds $inputFile

#echo "experiment will end in $(((varyT*(numberRoundsVaryT*numberSecondsBetweenUpdatingTempByVaryT + 2) + numberSecondsPerTemp*numberSecondsBetweenUpdatingTempByVaryT)/(60*60*24))) days"
