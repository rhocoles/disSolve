#!/bin/bash

#structure='circle10'
#structure='circle14'
#structure='circle36'
#structure='circle22'
#structure='hopfLink'
#structure='hopfLink32'
#structure='trefoil40'
#structure='trefoil36'
#structure='trefoil56'
#structure='torusKnot2_5'
#structure='thBe_circle36'
structure='openChain_dl25_25'
#structure='openChain_dl25_50'
#shape='compact_stacked_not_sure'
#shape='tied_parallel'
#shape='tied'
#shape='parallel_twice_folded'
#shape='parallel_rolled_up'
#shape='treble_clef'
shape='treble_clef_flat_end'
#shape='double_helix_flat_end'
#shape='double_helix_symmetric'
#shape='placebo'
#shape='optimal_helix'

inputFile="${shape}.txt"
#inputFile="blah"
# ---> to continue an experiment from frame_number of a previous run through ... files should be in the form testi__f.txt where i=number_of_parallel_process f=frame_number then uncomment lins 170 and 177 of main.py
#inputFile='fileName.txt' # ---> each parallel process gets the same initial configuration saved from fileName.txt then uncomment lines 171 and 177 of main.py

overlapRatio=0.02
eta=0.125

numberParallelProcesses=18

#annealing parameters for the decreasing temp part
numberSecondsPerTemp=4680
numberOfRounds=120
T_0=16.0
T_step=0.96

#annealing parameters for the variable T part of experiment --- no swopping
varyT=0 #0 means this doesn't happen, 1 means this does happen.
numberRoundsVaryT=0 #if 30 this brings the temp up to prob(med(deltaE>0)) = 0.6, I wouldn't increase this, instead increase the time between temp updates 
numberSecondsBetweenUpdatingTempByVaryT=69120

# check current director
dir_origin=${PWD}

#echo "${dir_origin}"
#echo "${dir_origin}/main.py"

# first make an directory .../structure/gridPoint../
dir=/LOCAL/coles/testing/structures/${structure}/$shape/${structure}_rs0_${overlapRatio:2:3}_eta0_${eta:2:3}
#dir=/LOCAL/coles/testing/structures/${structure}/${structure}_rs0_${overlapRatio:2:3}_eta0_${eta:2:3}
echo $dir

mkdir -p $dir
cd $dir

echo "an experiment with ${structure}"
echo "${dir}"
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
#screen -S ${structure:0:3}_rs0_${overlapRatio:2:3}_eta0_${eta:2:3} -L -d -m mpirun -np $numberParallelProcesses ~/miniconda3/bin/python3 main.py $structure $T_0 $overlapRatio $eta $T_step $numberSecondsPerTemp $numberOfRounds $varyT $numberRoundsVaryT $numberSecondsBetweenUpdatingTempByVaryT $inputFile
#screen -S ${structure:0:3}_rs0_${overlapRatio:2:3}_eta0_${eta:2:3} -L -d -m mpirun --oversubscribe -np $numberParallelProcesses ~/miniconda3/bin/python3 main.py $structure $T_0 $overlapRatio $eta $T_step $numberSecondsPerTemp $numberOfRounds $varyT $numberRoundsVaryT $numberSecondsBetweenUpdatingTempByVaryT $inputFile
screen -S ${shape:0:4}_${shape:(-4)}_rs0_${overlapRatio:2:3}_eta0_${eta:2:3} -L -d -m mpirun --oversubscribe -np $numberParallelProcesses ~/miniconda3/bin/python3 main.py $structure $T_0 $overlapRatio $eta $T_step $numberSecondsPerTemp $numberOfRounds $varyT $numberRoundsVaryT $numberSecondsBetweenUpdatingTempByVaryT $inputFile

