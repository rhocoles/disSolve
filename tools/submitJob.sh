#!/bin/bash

#structure='circle10'
#structure='circle14'
structure='circle36'
#structure='circle22'
#structure='hopfLink'
#structure='hopfLink32'
#structure='trefoil40'
#structure='trefoil36'
#structure='trefoil56'
#structure='torusKnot2_5'
#structure='thBe_circle36'
#structure='openChain_dl25_25'

#inputFile='test0__4500.txt'# ---> to continue an experiment from frame_number of a previous run through ... files should be in the form testi__f.txt where i=number_of_parallel_process f=frame_number then uncomment lins 170 and 177 of main.py
inputFile='fileName.txt' # ---> each parallel process gets the same initial configuration saved from fileName.txt then uncomment lines 171 and 177 of main.py

overlapRatio=0.08
eta=0.0

numberParallelProcesses=4

#annealing parameters for the decreasing temp part
numberSecondsPerTemp=3600
numberOfRounds=15
T_0=26.0
T_step=0.95

#annealing parameters for the variable T part of experiment --- no swopping
varyT=1 #0 means this doesn't happen, 1 means this does happen.
numberRoundsVaryT=15 #this brings the temp up to prob(med(deltaE>0)) = 0.6, I wouldn't increase this, instead increase the time between temp updates 
numberSecondsBetweenUpdatingTempByVaryT=2400 #updates temp every 40 mins of iterations 

# check current director
dir_origin=${PWD}

#echo "${dir_origin}"
#echo "${dir_origin}/main.py"

# first make an directory .../structure/gridPoint../
#dir=/LOCAL/coles/testing/structures/${structure}/${structure}_rs0_${overlapRatio:2:3}_eta0_${eta:2:3}
dir=$dir_origin/${structure}/${structure}_rs0_${overlapRatio:2:3}_eta0_${eta:2:3}
echo $dir

mkdir -p $dir
cd $dir

echo "an experiment with ${structure}"
cp ${dir_origin}/main.py .
cp ${dir_origin}/geometryClass.py .
cp ${dir_origin}/simple_functions.py .
cp ${dir_origin}/morph_local .
cp ${dir_origin}/pointFilaments.py .

#may be that you don't want to do this...
rm -rf polyFiles
mkdir polyFiles
rm -rf screenlog.0

#tidy up the old data files
rm -rf data*

#screen -S ${structure:0:3}_rs0_${overlapRatio:2:3}_eta0_${eta:2:3} -L -d -m mpirun -np $numberParallelProcesses ~/miniconda3/bin/python3 main.py $structure $T_0 $overlapRatio $ropelength $eta $inputFile $T_step $numberSecondsPerTemp $totalNumberOfRounds $varyT $numberRoundsVaryT $numberSecondsBetweenUpdatingTempByVaryT
screen -S ${structure:0:3}_rs0_${overlapRatio:2:3}_eta0_${eta:2:3} -L -d -m mpirun -np $numberParallelProcesses python3 main.py $structure $T_0 $overlapRatio $eta $T_step $numberSecondsPerTemp $numberOfRounds $varyT $numberRoundsVaryT $numberSecondsBetweenUpdatingTempByVaryT $inputFile

echo "experiment will end in $(((varyT*(numberRoundsVaryT*numberSecondsBetweenUpdatingTempByVaryT + 2) + numberSecondsPerTemp*numberSecondsBetweenUpdatingTempByVaryT)/(60*60*24))) days"
