#!/bin/bash

#structure='circle36'
#structure='circleTB' #label the directory with the edgelength line 37
#structure='hopfLink40'
#structure='trefoil50'
structure='openChain50_dl0_25'
path_to_BIG_db="/HOME1/users/personal/coles/BIG_databases/"${structure}"_BIG.db"

experimentID= # set to a queued id in BIG to claim it (overlapRatio/eta then come from BIG); leave blank for a fresh local run

overlapRatio=0.1
eta=0.05
alpha=0.0

numberParallelProcesses=12

#annealing parameters for the decreasing temp part
numberSecondsPerTemp=43200 #10800
numberOfRounds=36
T_top=0.02
T_bot=0.002
temp_options=(geometric linear temp_scan)
temperature_description=${temp_options[1]}

# initial curve configs: only used when experimentID is blank (a claimed-from-BIG job always
# starts from the basic embedded structure); "source"/"option" match select_initial_curve_configs.py
initial_config_source="BIG"   # "local" or "BIG"
initial_config_option="min-energy"   # "min-energy" | "last-frame-per-rank" | "min-energy-per-rank"

# optional: when claiming from BIG (experimentID set above), you can also start from specific
# known curveIDs instead of the basic embedded structure. Leave static_curve_ids blank to use
# the basic structure as before.
static_curve_config_source="local"   # "local" or "BIG"
static_curve_ids=""        # comma-separated curveIDs WITHOUT SPACES, e.g. "104,205,301" - blank reverts to basic structure



# ---- all variables to set ------------------------------------------------------------------------------- ----


# if claiming a specific BIG experiment, its rs/eta take priority over the hardcoded ones above
# (read-only lookup, safe to run before the run directory exists)
if [ -n "$experimentID" ]; then
    claimed=$(python3 -c "
import sqlite3
db = sqlite3.connect('$path_to_BIG_db')
row = db.execute('SELECT rs, eta FROM _experiments WHERE id = ?', ($experimentID,)).fetchone()
if row is None:
    raise SystemExit(f'No experiment with id $experimentID found in $path_to_BIG_db')
print(row[0], row[1])
")
    if [ -z "$claimed" ]; then
        echo "Failed to resolve rs/eta for experimentID=$experimentID (see error above) — aborting."
        exit 1
    fi
    read overlapRatio eta <<< "$claimed"
    echo beginning experiment ${experimentID} with parameters rs=${overlapRatio} and eta=${eta}
fi


# check current director
tools_dir=${PWD}
src_dir="$(dirname "$tools_dir")/src"

echo $tools_dir
echo $src_dir

# first make an directory .../structure/gridPoint../
LOCAL_TESTING_DIR=/LOCAL/coles/testing/
dir=${LOCAL_TESTING_DIR}${structure}/${structure}_rs0_${overlapRatio:2:3}_eta0_${eta:2:3}
echo $dir

mkdir -p $dir
cd $dir

#may be that you don't want to do this...
rm -rf polyFiles
mkdir polyFiles
rm -rf screenlog.0
rm -rf data
mkdir data


# get the initial curve configurations either from BIG db or local db
# claiming a specific BIG job always starts from the basic embedded structure, unless curveIDs from BIG/local are given;
# otherwise curveIDs are determined via select_initial_curve_configs.py
if [ -n "$experimentID" ]; then
    if [ -n "$static_curve_ids" ]; then
        curve_config_source=$static_curve_config_source
        curve_config_ids=$static_curve_ids
    else
        curve_config_source=0
        curve_config_ids=0
    fi
else
    curve_config_source=$initial_config_source
    curve_config_ids=$(python3 ${tools_dir}/select_initial_curve_configs.py $structure --source $initial_config_source --option $initial_config_option --rs $overlapRatio --eta $eta)
fi

[ -f "${structure}.db" ] && cp ${structure}.db ${structure}"_prev_version.db"

cp ${src_dir}/main.py .
cp ${src_dir}/geometryClass.py .
cp ${src_dir}/simple_functions.py .
cp ${src_dir}/morphometry.py .
cp ${src_dir}/libmorphometry.so .
cp ${src_dir}/self_distance_c.py .
cp ${src_dir}/libself_distance_c.so .
cp ${src_dir}/pointFilaments.py .
cp ${tools_dir}/evaluate_experiment.py .
cp ${tools_dir}/evaluate_experiment.ipynb .
cp ${tools_dir}/evaluate_experiment_temperatures.ipynb .
cp ${tools_dir}/evaluate_experiment_temperature_scan.ipynb .
cp ${tools_dir}/experiment_logging.py .


#get an experimentID via insertion into local/BIG db or claiming from BIG db
#NOTE: if extra_args="--path_to_BIG_db $path_to_BIG_db" and path_to_BIG_db is empty the experimentID is generated from insertion into local db, BIG db is not updated
#NOTE: if extra_args="--path_to_BIG_db $path_to_BIG_db" experimentID is generated from insertion into BIG db, local db is updated
#NOTE: if extra_args="--experimentID $experimentID" experimentID is inserted into local db. this is not used here
extra_args="--path_to_BIG_db $path_to_BIG_db"
if [ -n "$experimentID" ]; then
    extra_args="$extra_args --experimentID $experimentID"
fi
experimentID=$(python3 ${tools_dir}/initialise_experiment_database.py $structure $extra_args)
if [ -z "$experimentID" ]; then
    echo "Failed to initialise/claim experiment (see error above) — aborting."
    exit 1
fi
echo "Initialised experiment on $(hostname) with rs=$overlapRatio, eta=$eta, experimentID=$experimentID"

# to safely kill a running job (no orphaned mpirun): screen -S <name> -X stuff $'\003'
#screen -ls; echo "---"; ps aux | grep -iE 'main\.py|mpirun|prterun' | grep -v grep
screenExperimentName=${structure:0:3}${structure: -2}_rs0_${overlapRatio:2:3}_eta0_${eta:2:3}
screen -S ${screenExperimentName} -L -d -m bash -lc "
mpirun -np $numberParallelProcesses ~/miniconda3/bin/python3 main.py \
$overlapRatio $eta $alpha $T_top $T_bot $temperature_description $numberSecondsPerTemp $numberOfRounds $curve_config_source $curve_config_ids $structure $experimentID
if [ \$? -ne 0 ]; then
    python3 ${tools_dir}/mark_experiment_failed.py $structure $experimentID
else
    python3 evaluate_experiment.py $structure
fi
"

# original (pre-2026-07-21): mpirun was screen's direct child, so `screen -X -S <name> quit`
# killed it cleanly — no bash wrapper in between, but also no auto failure-marking.
# screen -S ${structure:0:3}_rs0_${overlapRatio:2:3}_eta0_${eta:2:3} -L -d -m mpirun -np $numberParallelProcesses python3 main.py $structure $T_0 $overlapRatio $eta $T_step $numberSecondsPerTemp $numberOfRounds $varyT $numberRoundsVaryT $numberSecondsBetweenUpdatingTempByVaryT $inputFile
