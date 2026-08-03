import numpy as np
np.set_printoptions(legacy='1.21')
import random
#from copy import deepcopy

import decimal as dec
from datetime import datetime

import time
import sys
import os
import shutil
from mpi4py import MPI
import socket
import sqlite3
import json

import geometryClass as geoClass

import pointFilaments
#import biarcs
from experiment_logging import BIG_DIR

comm = MPI.COMM_WORLD
rank = comm.Get_rank() #number of this parallel process
size = comm.Get_size() #number of parallel processes in total
host = socket.gethostname() #name of computer

status = MPI.Status()


def update_BIG_metadata_early(structure, experimentID, rs, eta, T_bot, T_top, temperature_description, numberOfRounds, allgather_time, initial_curve_configs, size):
    """Pushes early-known metadata to BIG right after experimentData.txt is written, so a
       multi-week experiment shows useful info in BIG long before evaluate_experiment.py runs.
       Retries once after a minute if the db is locked; exits if still locked."""
    for attempt in range(2):
        try:
            BIG_db_path = os.path.join(BIG_DIR, structure+"_BIG.db")
            db = sqlite3.connect(BIG_db_path)
            db.execute('''
                UPDATE _experiments SET
                    rs = ?, eta = ?, T_bot = ?, T_top = ?, temperature_description = ?,
                    number_of_rounds = ?, allgather_time = ?, initial_curve_configs = ?, size = ?
                WHERE id = ?
            ''', (rs, eta, T_bot, T_top, temperature_description, numberOfRounds,
                  allgather_time, initial_curve_configs, size, experimentID))
            db.commit()
            db.close()
            return None
        except sqlite3.OperationalError as e:
            if attempt == 0:
                print(f"BIG db locked ({e}), retrying in 60s...")
                time.sleep(60)
            else:
                sys.exit(f"BIG db still locked after retry, giving up on early metadata push: {e}")
    return None

def read_polyfile(filepath):
    """
    Function: read_polyfile
    ------------------------------------
    filepath is string
    """
    points = {}
    polygons = []
    section = None
    configType = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line == 'POINTS':
                section = 'POINTS'
                continue
            elif line == 'POLYS':
                section = 'POLYS'
                continue
            elif line == 'END':
                break

            colon = line.index(':')
            idx = int(line[:colon])
            rest = line[colon + 1:].split()

            if section == 'POINTS':
                # rest = ['x', 'y', 'z', 'c(r,g,b,a)'] — just take first 3
                x, y, z = float(rest[0]), float(rest[1]), float(rest[2])
                points[idx] = [x, y, z]

            elif section == 'POLYS':
                indices = [int(i) for i in rest]
                if  indices[0] == indices[-1]:
                    configType.append('closed')
                    indices = indices[:-1]
                else:
                    configType.append('open')
                polygons.append([points[i] for i in indices])

    return polygons, configType

def measureTime():
    """Returns the current date and time up to the nearest second"""

    time = str(datetime.now()).split(' ')
    #print time[0], time[1]
    date = time[0].split('-')
    time = time[1].split(':')
    return (int(date[1]), int(date[2]), int(time[0]), int(time[1]),int(float(time[2])))

def measureTimeDifference(startTime, endTime):
    """Returns the difference in time of startTime and endTime in seconds."""

    if startTime[0]!=endTime[0]:
        print("these have started at different months, you need to extend the method to deal with this")
    else:
        #return (endTime[1] - startTime[1])*(24 -  startTime[2])*60*60 - startTime[3]*60 - startTime[4] + endTime[2]*60*60 + endTime[3]*60 + endTime[4]
        return (endTime[1] - startTime[1])*24*60*60 + (endTime[2]- startTime[2])*60*60 + (endTime[4]- startTime[4]) + (endTime[3] - startTime[3])*60
    return 0

def writeData(dataList, fileName):
    '''
    Function: writeData
    -------------------------------------------------
    Function writes the content of dataList as the last line in the file called filename in current directory
    dataList list of floats
    filename string
    Return None
    '''
    f1 = open('./data/'+str(fileName)+'.txt', 'a')
    for k in range(len(dataList)):
        if type(dataList[k])==str:
            f1.write(dataList[k] + ' ')
        else:
            f1.write(str(dec.Decimal(str(round(dataList[k],10))))+' ')
    f1.write('\n')
    f1.close()
    return None

def acceptOrReject(x, T):
    """
    Function: acceptOrReject
    -----------------------------------------------------------------------------------------------------------
    method returns a 0 or 1. P(return = 1) = P_T(X <= x) where P_T(X<=x) = exp(-x/T)
    x float
    T float
    return 0 or 1
    """
    prob = np.exp(np.clip(-x/T, -100000000, 100)) #if -x/T is positive then the value is irrelevant because we always accept
    z = random.uniform(0,1)
    if z<=prob:
        return (1, prob)
    else:
        return (0, prob)


def swopOrNot(xj, Tj, xi, Ti):
    """
    Function: swopOrNot
    -------------------------------------------------------------------------------------------------------------------------------------------------------------
    method returns a 0 or 1. P(return = 1) = P_T(X <= x) where P_T(X<=x) = min(1, exp((1/Ti - 1/Tj)*(xj - xi)))
    x float (energy)
    T float (temp)
    Suppose Ti < Tj so that 1/Ti - 1/Tj > 0
    xi < xj <--> (xj - xi) > 0 <--> (1/Ti - 1/Tj)*(xj - xi) > 0 <--> exp((1/Ti - 1/Tj)*(xj - xi)) > 1 :chains always swop
        --> helps to avoid configurations of low energy remaining stationary at a low temperature
    xi > xj <--> (xj - xi) < 0 <--> (1/Ti - 1/Tj)*(xj - xi) < 0 <--> exp((1/Ti - 1/Tj)*(xj - xi)) < 1 :chains swop with a certain likelihood  
        --> xj should migrate to the lower temperature system
    return 0 or 1
    """
    prob = np.exp((1/Ti - 1/Tj)*(xi - xj))
    z = random.uniform(0,1)
    if z<=prob:
        return (1, min(round(prob, 3), 1))
    else:
        return (0, round(prob, 3))

def do_move_and_check_energy_and_accept_or_reject(geometry, dMin, dMax, T):
    """ Function: do_move_and_check_energy_and_accept_or_reject
    ----------------------------------------------------------------------------------------------
    This is the bit from the annealing algorithm which really can't be interuppted
    geometry object of type biarc or threadedBeadCurve
    dMax float maximum distance moved over one iteration
    T positive float temperature
    """

    moveReturn = geometry.curve_object.move(dMin=dMin, dMax=dMax)
    if moveReturn == 1:#configuration jammed all processes should end
        print('jammed on rank'+str(rank))
        geometry.curve_object.save_data('./','jammedGeometry')
        sys.exit()
    else:
        tmpGeometryData =  moveReturn
    
    tmpCurveVertices = geometry.curve_object.evaluate_curve_vertices(tmpGeometryData)
    deltaE, size_measures = geometry.evaluate_energy_difference(tmpCurveVertices) #if ThreadedBeads size_measures =  measures if Biarc size_measures = (measures, embedded_measures, length)    
    deltaE = deltaE/geometry.curve_object.length

    #accept or reject energy if increased
    (tmp_0or1, prob) = acceptOrReject(deltaE, T)#calculate probability of achieving deltaE choose via random number generation
    if tmp_0or1==1:
        geometry.update_geometry(tmpGeometryData, tmpCurveVertices, size_measures)
            
    return (tmp_0or1, prob, deltaE)

if rank==0:
    print(sys.argv, len(sys.argv))
fileLocation = './polyFiles/' #need to move one down from the current directory directoryName
polyFileName = 'test_'+str(rank)+'_'
fileName = 'test_'+str(rank)
frameNumber = 1

structure = str(sys.argv[11])#this variable is passed to automatically title the jupyter notebook
experimentID = int(sys.argv[-1])#updates BIG with the metadata used in experiment. 
overlapRatio = float(sys.argv[1])
eta = float(sys.argv[2])
alpha = float(sys.argv[3])

initial_config_source = str(sys.argv[9])    # "local", "BIG", or "0"
initial_curve_ids_str = str(sys.argv[10])   # comma-separated ids, or "0"

#update the last version here with the configType
structure_defaults = {
    "openChain50_dl0_25": ("open_chain_50_dl0_25", 0.25, 1),
    "openChain25_dl0_25": ("open_chain_25_dl0_25", 0.25, 1),
    "circle36":            ("circle36_thBe_dl25", 0.25, 0),
    "hopfLink40":          ("hopfLink40_thBe_dl25", 0.25, 0),
    "trefoil50":           ("trefoil50_thBe_dl4", 0.4, 0),
    "circleTB":            ("circleTB_dl0_25", 0.25, 0),
    "circleTB_dl0_08":     ("circleTB_dl0_08", 0.08, 0)
}

pointFilaments_name, edgeLength, configType = structure_defaults[structure]
if initial_config_source in ("0", "") or initial_curve_ids_str in ("0", ""):
    curveData = getattr(pointFilaments, pointFilaments_name)
    initial_configs = str([0])
else:
    initial_curve_ids = [int(x) for x in initial_curve_ids_str.split(",")]
    my_curveID = initial_curve_ids[rank * len(initial_curve_ids) // size]

    db_path = os.path.join(BIG_DIR, structure + "_BIG.db") if initial_config_source == "BIG" else structure + ".db"
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    row = db.execute("SELECT pointCoordinates, edgeLength FROM _experiments_all_curves WHERE id = ?", (my_curveID,)).fetchone()
    db.close()
    if row is None:
        sys.exit(f"No curve found with id {my_curveID} in {db_path}")
    curveData = json.loads(row["pointCoordinates"])
    if initial_config_source == "BIG":
        initial_configs = json.dumps([0] + initial_curve_ids, separators=(',', ':'))
    else:
        initial_configs = json.dumps([1] + initial_curve_ids, separators=(',', ':'))
geometry = geoClass.TubularGeometry(overlapRatio, eta, geoClass.ThreadedBeads(configType, curveData=curveData, edgeLength=edgeLength))

#geometry = geoClass.TubularGeometry(overlapRatio, eta, geoClass.ThreadedBeads(0, fileName=pathToInitialConfig, edgeLength=0.25))
#geometry = geoClass.TubularGeometry(overlapRatio, eta, geoClass.ThreadedBeads(0, curveData=pointFilaments.circle36_thBe_dl25, edgeLength=0.25))
#geometry = geoClass.TubularGeometry(overlapRatio, eta, geoClass.ThreadedBeads(0, fileName=pathToInitialConfig, edgeLength=0.4))
#geometry = geoClass.TubularGeometry(overlapRatio, eta, geoClass.ThreadedBeads(0, curveData=pointFilaments.trefoil50_thBe_dl4, edgeLength=0.4))
#geometry = geoClass.TubularGeometry(overlapRatio, eta, geoClass.ThreadedBeads(0, curveData=pointFilaments.hopfLink40_thBe_dl25, edgeLength=0.25))

#geometry = geoClass.TubularGeometry(overlapRatio, eta, geoClass.ThreadedBeads(0, curveData=pointFilaments.circleTB_dl0_08, edgeLength=0.08))
#geometry = geoClass.TubularGeometry(overlapRatio, eta, geoClass.ThreadedBeads(0, curveData=pointFilaments.circleTB_dl0_08, edgeLength=0.08))
#geometry = geoClass.TubularGeometry(overlapRatio, eta, geoClass.ThreadedBeads(0, fileName=pathToInitialConfig, edgeLength=0.08))
#geometry = geoClass.TubularGeometry(overlapRatio, eta, geoClass.ThreadedBeads(0, curveData=pointFilaments.circleTB_dl0_25, edgeLength=0.25), alpha=alpha)
#geometry = geoClass.TubularGeometry(overlapRatio, eta, geoClass.ThreadedBeads(0, fileName=pathToInitialConfig, edgeLength=0.25))
#################### END THREADED BEADS

##################### BIARCS
#geometry = geoClass.TubularGeometry(overlapRatio, eta, geoClass.Biarcs(fileName=pathToInitialConfig))
#geometry = geoClass.TubularGeometry(overlapRatio, eta, geoClass.Biarcs(curveData = biarcs.circle36Biarcs_52arcs))
#geometry.set_uniform_tube_and_energy_specs_by_overriding_edgeLength(0.225)
#geometry = geoClass.TubularGeometry(overlapRatio, eta, geoClass.Biarcs(curveData = biarcs.trefoil50Biarcs_56arcs, sphereDensity=4))
#geometry.set_uniform_tube_and_energy_specs_by_overriding_edgeLength(0.226)
#geometry.curve_object.rescale_geometry(40/39.68504)
#geometry = geoClass.TubularGeometry(overlapRatio, eta, geoClass.Biarcs(curveData = biarcs.hopfLink40_44arcs, sphereDensity=4))
#geometry.set_uniform_tube_and_energy_specs_by_overriding_edgeLength(0.22722)
################## END BIARCS

#geoClass.makePointCloudPoly([pt for subList in geometry.curve_object.curve_vertices for pt in subList], fileLocation,'test_'+str(frameNumber))
#geoClass.makeFilFile([pt for subList in geometry.curve_object.curve_vertices for pt in subList], 'trefoil40_'+str(frameNumber), geometry.input_R)
geometry.evaluate_embedded_measures()
geometry.evaluate_measures()


#annealing schedule variables

#set Temperature
T_top = float(sys.argv[4])
T_bot = float(sys.argv[5])
temperature_description = str(sys.argv[6]) #geometric/linear/temp_scan (temp_scan divides temp interval linearly)
if temperature_description == "geometric":
    temp_of_bin = lambda i : round(T_bot*pow((T_top/T_bot), i/(size-1)), 6)
elif temperature_description == "linear":
    temp_of_bin = lambda i : round(T_bot + ((T_top - T_bot)/size)*i, 6)

#time between exchanging temperatures between systems
allgather_time = int(sys.argv[7])#number secs computing between systems may be exchanged
numberOfRounds = int(sys.argv[8])

dMin = 0.01*overlapRatio
dMax = 2*overlapRatio + dMin
if rank==0:
    print("(dMin, dMax) = ", (dMin, dMax))

if rank ==0:
    if geometry.curve_object.geometryType == "Biarc":
        print("strand lengths ", list(map(lambda x: round(x, 5), geometry.curve_object.evaluate_curve_length(geometry.curve_object.data))))
    print("curve is of ", len(geometry.curve_object.data), "strands and each strand has ", [len(geometry.curve_object.data[i]) for i in range(len(geometry.curve_object.data))])
    print("edge length is fixed to ", geometry.curve_object.edgeLength)
    print("computing with ball radius ", geometry.input_R)
    print(geometry.V_0, geometry.A_0, geometry.C_0, geometry.X_0)
    print(geometry.V, geometry.A, geometry.C, geometry.X)
    print("Initialised curve", rank, "of length", geometry.curve_object.length, "(E - E0)/L = ", geometry.evaluate_normalised_energy(), "(minRads, minSelfDist) = ", geometry.curve_object.check_reach())
    print("(dMin, dMax) = ", dMin, dMax, "with", len(geometry.curve_object.index_intervals_to_be_rotated), "possible rotations")
    print("energy is set with coefficients", geometry.coefficients, " using ", geometry.f1_f2_f3_f4)

it_no =0
it_no_=0
accept=0
rounds = 0       #even rounds (T_{i} for i even communicates with T_{i+1}) odd rounds (T_{i} for i odd communicates with T_{i+1}) i is BIN_INDEX

if temperature_description != "temp_scan":
    #set up the data to be communicated between processes
    allgather_data = {
        "bin_index": rank,
        "energy": geometry.evaluate_normalised_energy(),
        "prob Ti --> Ti+1": 0,
        "swop": 0, 
    }
    T = temp_of_bin(allgather_data["bin_index"])
    if rank==0:
        print("annealing within temperature range [",T_bot, ",", T_top, "],  ", size, " systems are exchanged every ", allgather_time, "seconds,  over a total number of ", numberOfRounds, " rounds")
        print("\nexperiment ends in ", round(allgather_time*numberOfRounds/(60*60*24), 3), " days")
        writeData([size, overlapRatio, eta, geometry.R, geometry.r_s, geometry.input_R] + geometry.coefficients + [geoClass.countNumberOf(geometry.curve_object.curve_vertices), geometry.curve_object.edgeLength, T_bot, T_top, numberOfRounds, allgather_time, structure, alpha, time.time(), temperature_description, initial_configs], 'experimentData')
        writeData([str(temp_of_bin(k)) for k in range(size)], 'temperatures')
        update_BIG_metadata_early(structure, experimentID, overlapRatio, eta, T_bot, T_top, temperature_description, numberOfRounds, allgather_time, initial_configs, size)
    
    start_time = time.time()
    writeData([it_no, T, 1, 0, geometry.V, geometry.A, geometry.C, geometry.X, geometry.V_0, geometry.A_0, geometry.C_0, geometry.X_0, geometry.curve_object.length, geometry.evaluate_normalised_energy(), frameNumber, 1.0, start_time, rank, allgather_data["bin_index"], rounds, it_no_] + list(map(lambda x: round(x, 5), geometry.curve_object.check_reach())), fileName)
    geometry.curve_object.make_curve_polyFile(fileLocation, polyFileName+str(frameNumber))
    frameNumber+=1
    while rounds < numberOfRounds:
        (tmp_0or1, prob, deltaE) = do_move_and_check_energy_and_accept_or_reject(geometry, dMin, dMax, T)
        accept+=tmp_0or1
        it_no+=1
        it_no_+=1
    
        #save iteration data
        if (it_no + accept)%2000==0:
            geometry.evaluate_embedded_measures()
            geometry.evaluate_measures()
            geometry.evaluate_normalised_energy()
            writeData([it_no, T, prob, deltaE, geometry.V, geometry.A, geometry.C, geometry.X, geometry.V_0, geometry.A_0, geometry.C_0, geometry.X_0, geometry.curve_object.length, geometry.evaluate_normalised_energy(), frameNumber, accept/it_no_, time.time(), rank, allgather_data["bin_index"], rounds, it_no_] + list(map(lambda x: round(x, 5), geometry.curve_object.check_reach())), fileName)
            geometry.curve_object.make_curve_polyFile(fileLocation, polyFileName+str(frameNumber))
            frameNumber+=1
    
        if it_no%2000==0:
            print("Info from rank", rank, "at bin ", allgather_data["bin_index"], " (E - E0)/L", round(geometry.evaluate_normalised_energy(),6), "T =", round(T,5), "acceptRatio", round(accept/it_no_, 3), "(minRads, minSelfDist) = ", list(map(lambda x: round(x, 5), geometry.curve_object.check_reach())), frameNumber, " for round", rounds)
    
        if time.time() - start_time > allgather_time:
    
            allgather_data["energy"] = geometry.evaluate_normalised_energy()
            alldatas=comm.allgather(allgather_data)
            if (rank==0):
                print("finished round", rounds)
    
            is_even_or_odd = rounds%2 #even bin_indices exchange with odd
            if rank==0:
                if is_even_or_odd==0:
                    print("even round")
                else:
                    print("odd round")
            
            allgather_data["prob Ti --> Ti+1"] = -1
            allgather_data["swop"] = -1
            if allgather_data["bin_index"]%2==is_even_or_odd:# T_{i} decides whether to swop with T_{i+1}
                if allgather_data["bin_index"]<(size - 1):#highest temperate bin swops only to lower temperature bins
                    rank_bin_index_right = [alldatas[i]["bin_index"] for i in range(size)].index(allgather_data["bin_index"] + 1)
                    tmpSwopOrNot, prob = swopOrNot(alldatas[rank_bin_index_right]["energy"], temp_of_bin(allgather_data["bin_index"] + 1), allgather_data["energy"], T)
                    allgather_data["prob Ti --> Ti+1"] = prob #prob belongs to how the bin index became what it is at the s
                    print(f"rank {rank} @bin {allgather_data['bin_index']} is engaged in swopping with {rank_bin_index_right} @bin {alldatas[rank_bin_index_right]['bin_index']}", "swopOrNot prob =", prob)
                    comm.send((tmpSwopOrNot, prob), dest=rank_bin_index_right, tag=1)
                    if tmpSwopOrNot==1:
                        allgather_data["swop"]=1 
                        comm.send(allgather_data["bin_index"], dest=rank_bin_index_right, tag=1)
                        allgather_data["bin_index"]+=1
                        T = temp_of_bin(allgather_data["bin_index"])
                        print(f"{rank} has swopped temperature with {rank_bin_index_right}")
                    else:#send back the same
                        allgather_data["swop"]=0 
                        comm.send(allgather_data["bin_index"] + 1, dest=rank_bin_index_right, tag=1)
            else:
                if allgather_data["bin_index"] > 0: #lowest temperature bin swops only with higher temperatures i.e. bin 0 recieves no message
                    rank_bin_index_left = [alldatas[i]["bin_index"] for i in range(size)].index(allgather_data["bin_index"] - 1)
                    tmpSwopOrNot, prob = comm.recv(source=rank_bin_index_left, tag=1)
                    allgather_data["prob Ti --> Ti+1"] = prob
                    allgather_data["swop"] = tmpSwopOrNot
                    if tmpSwopOrNot:
                        allgather_data["bin_index"] = comm.recv(source=rank_bin_index_left, tag=1)
                        print(rank, "recieved bin index from rank", rank_bin_index_left)
                        T = temp_of_bin(allgather_data["bin_index"])
    
            alldatas=comm.allgather(allgather_data)
            if (rank==0):
                writeData([rounds, is_even_or_odd, size] + [alldatas[i]["bin_index"] for i in range(size)] +  [alldatas[i]["swop"] for i in range(size)] + [alldatas[i]["prob Ti --> Ti+1"] for i in range(size)] + [alldatas[i]["energy"] for i in range(size)] + [experimentID], "temp_exchange_stats")
                for (i,a) in enumerate(alldatas):
                    print(i,a)
    
            comm.Barrier()
            it_no_=0
            accept=0
            rounds+=1
            start_time = time.time()



if temperature_description=="temp_scan":
    def triangle_fraction(i):
        half = numberOfRounds // 2
        pos = i % numberOfRounds
        return pos / half if pos <= half else (numberOfRounds - pos) / half

    temp_of_round = lambda i: round(T_bot + (T_top - T_bot) * triangle_fraction(i), 6)

    if rank==0:
        print("Annealing within temperature range [",T_bot, ",", T_top, "].\nTemperature interval is divided into ", numberOfRounds//2, " fixed temperatures, all ", size, " parallel systems anneal at the same fixed temperature. \nEach temperature is annealed for ", allgather_time, "seconds,  over a total number of ", numberOfRounds, "rounds (temperature interval is visited twice).")
        print("\nexperiment ends in ", round(allgather_time*numberOfRounds/(60*60*24), 3), " days")
        writeData([size, overlapRatio, eta, geometry.R, geometry.r_s, geometry.input_R] + geometry.coefficients + [geoClass.countNumberOf(geometry.curve_object.curve_vertices), geometry.curve_object.edgeLength, T_bot, T_top, numberOfRounds, allgather_time, structure, alpha, time.time(), temperature_description, initial_configs], 'experimentData')
        writeData([str(temp_of_round(k)) for k in range(numberOfRounds)], 'temperatures')
        update_BIG_metadata_early(structure, experimentID, overlapRatio, eta, T_bot, T_top, temperature_description, numberOfRounds, allgather_time, initial_configs, size)

    T = temp_of_round(rounds)
    start_time = time.time()
    writeData([it_no, T, 1, 0, geometry.V, geometry.A, geometry.C, geometry.X, geometry.V_0, geometry.A_0, geometry.C_0, geometry.X_0, geometry.curve_object.length, geometry.evaluate_normalised_energy(), frameNumber, 1.0, start_time, rank, rank, rounds, it_no_] + list(map(lambda x: round(x, 5), geometry.curve_object.check_reach())), fileName)
    geometry.curve_object.make_curve_polyFile(fileLocation, polyFileName+str(frameNumber))
    frameNumber+=1
    while rounds < numberOfRounds:
        (tmp_0or1, prob, deltaE) = do_move_and_check_energy_and_accept_or_reject(geometry, dMin, dMax, T)
        accept+=tmp_0or1
        it_no+=1
        it_no_+=1
    
        #save iteration data
        if (it_no + accept)%1000==0:
            geometry.evaluate_embedded_measures()
            geometry.evaluate_measures()
            geometry.evaluate_normalised_energy()
            writeData([it_no, T, prob, deltaE, geometry.V, geometry.A, geometry.C, geometry.X, geometry.V_0, geometry.A_0, geometry.C_0, geometry.X_0, geometry.curve_object.length, geometry.evaluate_normalised_energy(), frameNumber,  accept/it_no, time.time(), rank, rank, rounds, it_no_] + list(map(lambda x: round(x, 5), geometry.curve_object.check_reach())), fileName)
            geometry.curve_object.make_curve_polyFile(fileLocation, polyFileName+str(frameNumber))
            frameNumber+=1
    
        if it_no%1000==0:
            print("Info from rank", rank, " (E - E0)/L", round(geometry.evaluate_normalised_energy(),6), "T =", round(T,8), "acceptRatio", round(accept/it_no, 3), "(minRads, minSelfDist) = ", list(map(lambda x: round(x, 5), geometry.curve_object.check_reach())), frameNumber)
    
        if time.time() - start_time > allgather_time:
    
            if (rank==0):
                print("finished round", rounds)
    
            comm.Barrier()
            it_no_=0
            rounds+=1
            T = temp_of_round(rounds)
            start_time = time.time()
