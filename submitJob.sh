#!/bin/bash

#geometryType="Biarcs"
geometryType="ThreadedBeads"
#structure='circle10'
#structure='circle14'
structure='circle36'
#structure='circle22'
#structure='hopfLink'
#structure='hopfLink32'
#structure='trefoil40'
#structure='trefoil56'
#structure='torusKnot2_5'
#structure='thBe_circle36'
#structure='openChain_dl25_25'
#structure='openChain_dl25_50'
#shape='compact_stacked_not_sure'
#shape='tied_parallel'
#shape='tied'
#shape='parallel_twice_folded'
#shape='parallel_rolled_up'
#shape='treble_clef'
#shape='treble_clef_flat_end'
#shape='double_helix_flat_end'
#shape='double_helix_symmetric'
#shape='placebo'
#shape='optimal_helix'
#structure="test"

#inputFile="${shape}.txt"
inputFile="/HOME1/users/personal/coles/program/dissolve/initialConfigs/circle36Biarcs_52arcs_"

overlapRatio=0.1
eta=0.35

numberParallelProcesses=6

#annealing parameters for the decreasing temp part
numberSecondsPerTemp=3600
numberOfRounds=0
T_0=0.5
T_step=0.75

#annealing parameters for the variable T part of experiment --- no swopping
varyT=1 #0 means this doesn't happen, 1 means this does happen.
numberRoundsVaryT=23 #if 30 this brings the temp up to prob(med(deltaE>0)) = 0.6, I wouldn't increase this, instead increase the time between temp updates 
numberSecondsBetweenUpdatingTempByVaryT=25920 #20*25920=518400=6*24*60*60 i.e. 6 days

# check current director
dir_origin=${PWD}

#echo "${dir_origin}"
#echo "${dir_origin}/main.py"

# first make an directory .../structure/gridPoint../
#dir=/LOCAL/coles/testing/structures/${structure}/$shape/${structure}_rs0_${overlapRatio:2:3}_eta0_${eta:2:3}
dir=/LOCAL/coles/testing/structures/${structure}/${geometryType}/${structure}_rs0_${overlapRatio:2:3}_eta0_${eta:2:3}
#dir=${dir_origin}/${structure}
echo $dir

mkdir -p $dir
cd $dir

echo "an experiment with ${structure}"
echo "${dir}"
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
#screen -S ${structure:0:3}_rs0_${overlapRatio:2:3}_eta0_${eta:2:3} -L -d -m mpirun -np $numberParallelProcesses ~/miniconda3/bin/python3 main.py $structure $T_0 $overlapRatio $eta $T_step $numberSecondsPerTemp $numberOfRounds $varyT $numberRoundsVaryT $numberSecondsBetweenUpdatingTempByVaryT $inputFile
#screen -S ${structure:0:3}_rs0_${overlapRatio:2:3}_eta0_${eta:2:3} -L -d -m mpirun --oversubscribe -np $numberParallelProcesses ~/miniconda3/bin/python3 main.py $structure $T_0 $overlapRatio $eta $T_step $numberSecondsPerTemp $numberOfRounds $varyT $numberRoundsVaryT $numberSecondsBetweenUpdatingTempByVaryT $inputFile
#screen -S ${shape:0:4}_${shape:(-4)}_rs0_${overlapRatio:2:3}_eta0_${eta:2:3} -L -d -m mpirun --oversubscribe -np $numberParallelProcesses ~/miniconda3/bin/python3 main.py $structure $T_0 $overlapRatio $eta $T_step $numberSecondsPerTemp $numberOfRounds $varyT $numberRoundsVaryT $numberSecondsBetweenUpdatingTempByVaryT $inputFile

#screen -S "testRun" -L -d -m mpirun -np $numberParallelProcesses ~/miniconda3/bin/python3 main.py $overlapRatio $eta $T_0 $T_step $numberSecondsPerTemp $numberOfRounds $varyT $numberRoundsVaryT $numberSecondsBetweenUpdatingTempByVaryT $structure $inputFile

screenExperimentName=${structure:0:3}${structure: -2}_${geometryType:0:2}_rs0_${overlapRatio:2:3}_eta0_${eta:2:3}
#echo "${screenExperimentName}"
screen -S ${screenExperimentName} -L -d -m mpirun -np $numberParallelProcesses ~/miniconda3/bin/python3 main.py $overlapRatio $eta $T_0 $T_step $numberSecondsPerTemp $numberOfRounds $varyT $numberRoundsVaryT $numberSecondsBetweenUpdatingTempByVaryT $structure $inputFile

