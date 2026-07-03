#!/bin/bash

#geometryType="Biarcs"
geometryType="ThreadedBeads"
#structure='circle36'
structure='circleTB' #label the directory with the edgelength line 37
#structure='hopfLink40'
#structure='trefoil50'

overlapRatio=0.3
eta=0.0
alpha=0.0

#inputFile="/LOCAL/coles/testing/structures/${structure}/${geometryType}/${structure}_rs0_${overlapRatio:2:3}_eta0_${eta:2:3}/initialConfigs/test_"
#inputFile="/LOCAL/coles/testing/structures/${structure}/${geometryType}/${structure}_rs0_${overlapRatio:2:3}_volMin_dl0_08/initialConfigs/test_"
inputFile="/LOCAL/coles/testing/structures/${structure}/${geometryType}/${structure}_rs0_${overlapRatio:2:3}_volMin_dl0_25/initialConfigs/test_"
#inputFile="/LOCAL/coles/testing/structures/${structure}/${geometryType}/${structure}_rs0_${overlapRatio:2:3}_eta0_${eta:2:3}/initialConfigs/test.txt"
#inputFile="/LOCAL/coles/testing/structures/${structure}/${geometryType}/${structure}_rs0_${overlapRatio:2:3}_eta0_${eta:2:3}__/initialConfigs/test.txt"
#inputFile="/LOCAL/coles/testing/structures/${structure}/${geometryType}/${structure}_rs0_${overlapRatio:2:3}_eta0_${eta:2:3}_frnber_18500/initialConfigs/test_"

numberParallelProcesses=32

#annealing parameters for the decreasing temp part
numberSecondsPerTemp=8000
numberOfRounds=120
T_top=2.0
T_bot=0.1

# check current director
dir_origin=${PWD}


# first make an directory .../structure/gridPoint../
#dir=/LOCAL/coles/testing/structures/${structure}/${geometryType}/${structure}_rs0_${overlapRatio:2:3}_eta0_${eta:2:3}
#dir=/LOCAL/coles/testing/structures/${structure}/${geometryType}/${structure}_rs0_${overlapRatio:2:3}_alpha0_${alpha:2:4}_dl0_25
#dir=/LOCAL/coles/testing/structures/${structure}/${geometryType}/${structure}_rs0_${overlapRatio:2:3}_alpha0_${alpha:2:4}_dl0_08
dir=/LOCAL/coles/testing/structures/${structure}/${geometryType}/${structure}_rs0_${overlapRatio:2:3}_volMin_dl0_25
#dir=/LOCAL/coles/testing/structures/${structure}/${geometryType}/${structure}_rs0_${overlapRatio:2:3}_volMin_dl0_08
#dir=/LOCAL/coles/testing/structures/${structure}/${geometryType}/${structure}_rs0_${overlapRatio:2:3}_eta0_${eta:2:3}_frnber_18500
echo $dir

mkdir -p $dir
cd $dir

cp ${dir_origin}/main_parallel_annealing.py .
cp ${dir_origin}/geometryClass_.py .
cp ${dir_origin}/simple_functions.py .
cp ${dir_origin}/morphometry.py .
cp ${dir_origin}/libmorphometry.so .
cp ${dir_origin}/biarcs.py .
cp ${dir_origin}/pointFilaments.py .
cp ${dir_origin}/evaluate_experiment.py .
cp ${dir_origin}/evaluate_experiment.ipynb .

#may be that you don't want to do this...
rm -rf polyFiles
mkdir polyFiles
rm -rf screenlog.0
rm -rf data
mkdir data

rm ${structure}.db

#screenExperimentName=${structure:0:3}${structure: -2}_${geometryType:0:2}_rs0_${overlapRatio:2:3}_eta0_${eta:2:3}
#screenExperimentName=${structure:0:3}${structure: -2}_${geometryType:0:2}_rs0_${overlapRatio:2:3}_alpha0_${alpha:2:4}
screenExperimentName=${structure:0:3}${structure: -2}_${geometryType:0:2}_rs0_${overlapRatio:2:3}_volMin
#screenExperimentName=${structure:0:3}${structure: -2}_${geometryType:0:2}_rs0_${overlapRatio:2:3}_eta0_${eta:2:3}_18500
#screen -S ${screenExperimentName} -L -d -m mpirun -np $numberParallelProcesses ~/miniconda3/bin/python3 main_parallel_annealing.py $overlapRatio $eta $alpha $T_top $T_bot $numberSecondsPerTemp $numberOfRounds $structure $inputFile
screen -S ${screenExperimentName} -L -d -m bash -lc "
mpirun -np $numberParallelProcesses ~/miniconda3/bin/python3 main_parallel_annealing.py \
$overlapRatio $eta $alpha $T_top $T_bot $numberSecondsPerTemp $numberOfRounds $structure $inputFile
python3 evaluate_experiment.py
"
