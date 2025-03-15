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

#from geometryClass import ThreadedBeads
import geometryClass as geoClass

import pointFilaments
import biarcs

comm = MPI.COMM_WORLD
rank = comm.Get_rank() #number of this parallel process
size = comm.Get_size() #number of parallel processes in total
host = socket.gethostname() #name of computer

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
    ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
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

#def returnProbability(x, T):
#    return  np.clip(np.exp(np.clip(-x/float(T),-100000000, 100)), 0, 1)

def returnWeight(x, T):
    return  np.exp(np.clip(-x/float(T),-1000000000, 1000000000))

def updateExpectation(historyList, newEntry, expectation):
    """
    Function: updateExpectation
    ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    function inserts newEntry into 0th position of the history list and removes the last entry. 
    expectation is returned as expectation + newEntry/(len(historyList)) - historyList[-1]/len(historyList)
    historyList list of floats
    newEntry float
    expectation float
    return float
    """
    historyList.insert(0, newEntry)
    return expectation + (newEntry - historyList.pop())/len(historyList)

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

    #accept or reject energy if increased
    (tmp_0or1, prob) = acceptOrReject(deltaE, T)#calculate probability of achieving deltaE choose via random number generation
    tmp_0or1 = 1
    if tmp_0or1==1:
        geometry.update_geometry(tmpGeometryData, tmpCurveVertices, size_measures)
            
    return (tmp_0or1, prob, deltaE)

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

if rank==0:
    print(sys.argv, len(sys.argv))
fileLocation = './polyFiles/' #need to move one down from the current directory directoryName
polyFileName = 'test_'+str(rank)+'_'
fileName = 'test_'+str(rank)
frameNumber = 1

pathToInitialConfig = sys.argv[-1]
#pathToInitialConfig = sys.argv[-1]+str(rank)+'.txt'
structure = str(sys.argv[-1])
overlapRatio = float(sys.argv[1])
eta = float(sys.argv[2])

#annealing schedule variables
T = float(sys.argv[3])
decrease = lambda x: x*float(sys.argv[4])
allgather_time = int(sys.argv[5])#number secs computing between systems may be exchanged
numberOfRounds = int(sys.argv[6])
dragFactor = 5000 #number of iterations over which expectations are computed.
dMin = 0.00001*overlapRatio
dMax = 0.1*(overlapRatio + dMin)
#print(dMin, dMax)

updateT0 = lambda x, y: -max(x, 0.005)/np.log(y)
varyT = int(sys.argv[7])
print(varyT)
numberRoundsVaryT=int(sys.argv[8])
numberSecondsBetweenUpdatingTempByVaryT=int(sys.argv[9])
print(numberRoundsVaryT, numberSecondsBetweenUpdatingTempByVaryT)
deltaE_increasing_list = []
deltaE_increasing_expectation = 0.0
probability_list = []
probability_expectation = 1.0
acceptRatio = 1.0
medianProbability = 0.1

#define the geometry
#geometry = geoClass.TubularGeometry(overlapRatio, eta, geoClass.ThreadedBeads(0, curveData=pointFilaments.circle36_thBe_dl25, edgeLength=0.25))
#geometry = geoClass.TubularGeometry(overlapRatio, eta, geoClass.ThreadedBeads(0, curveData=pointFilaments.trefoil50_thBe_dl4, edgeLength=0.4))
geometry = geoClass.TubularGeometry(overlapRatio, eta, geoClass.ThreadedBeads(0, fileName=pathToInitialConfig, edgeLength=0.4))
print([[tuple(geometry.curve_object.data[i][j]) for j in range(len(geometry.curve_object.data[i]))] for i in range(len(geometry.curve_object.data))])
quit()

#geometry = geoClass.TubularGeometry(overlapRatio, eta, geoClass.Biarcs(fileName=pathToInitialConfig))

#geometry = geoClass.TubularGeometry(overlapRatio, eta, geoClass.Biarcs(curveData = biarcs.circle36Biarcs_52arcs))
#geometry.set_uniform_tube_and_energy_specs_by_overriding_edgeLength(0.225)

#geometry = geoClass.TubularGeometry(overlapRatio, eta, geoClass.Biarcs(curveData = biarcs.trefoil50Biarcs_56arcs, sphereDensity=4))
#geometry.set_uniform_tube_and_energy_specs_by_overriding_edgeLength(0.226)
#geometry.curve_object.rescale_geometry(40/39.68504)

#geometry = geoClass.TubularGeometry(overlapRatio, eta, geoClass.Biarcs(curveData = biarcs.trefoil50Biarcs_56arcs, sphereDensity=5))
#geometry.curve_object.rescale_geometry(40/39.68504)

geometry.curve_object.make_curve_polyFile(fileLocation, polyFileName+str(frameNumber))
geoClass.makePointCloudPoly([pt for subList in geometry.curve_object.curve_vertices for pt in subList], fileLocation,'test_'+str(frameNumber))
#geoClass.makeFilFile([pt for subList in geometry.curve_object.curve_vertices for pt in subList], 'trefoil40_'+str(frameNumber), geometry.input_R)
geometry.evaluate_embedded_measures()
geometry.evaluate_measures()

if rank ==0:
    if geometry.curve_object.geometryType == "Biarc":
        print("strand lengths ", list(map(lambda x: round(x, 5), geometry.curve_object.evaluate_curve_length(geometry.curve_object.data))))
    print("curve is of ", len(geometry.curve_object.data), "strands and each strand has ", [len(geometry.curve_object.data[i]) for i in range(len(geometry.curve_object.data))])
    print("edge length is fixed to ", geometry.curve_object.edgeLength)
    print("computing with ball radius ", geometry.input_R)
    print(geometry.V_0, geometry.A_0, geometry.C_0, geometry.X_0)
    print(geometry.V, geometry.A, geometry.C, geometry.X)
    print("Initialised curve", rank, "of length", geometry.curve_object.length, "(E - E0)/L = ", geometry.evaluate_normalised_energy(), "(minRads, minSelfDist) = ", geometry.curve_object.check_reach())

#set up the data to be communicated between processes
allgather_data={}
allgather_data["geometry"] = geometry
allgather_data["energy"] = geometry.evaluate_normalised_energy()
#allgather_data["temp"] = 0.0

if rank==0:
     print("annealing at T_0 = ", T, " systems are exchanged every ", allgather_time, "over a total number of ", numberOfRounds, "rounds")
     print("starting experiment which will end in ", round((numberRoundsVaryT*numberSecondsBetweenUpdatingTempByVaryT + numberOfRounds*allgather_time)/(60*60*24), 3), "days from now")
     writeData([size, overlapRatio, eta, geometry.R, geometry.r_s, geometry.input_R] + geometry.coefficients + [geoClass.countNumberOf(geometry.curve_object.curve_vertices), geometry.curve_object.edgeLength, varyT, numberRoundsVaryT, numberSecondsBetweenUpdatingTempByVaryT, T, numberOfRounds, allgather_time, structure], 'experimentData')

#print(size, rank, overlapRatio, eta, geometry.R, geometry.r_s, geometry.input_R, geometry.coefficients, geoClass.countNumberOf(geometry.curve_object.curve_vertices), geometry.curve_object.edgeLength, varyT, numberRoundsVaryT, numberSecondsBetweenUpdatingTempByVaryT, T, numberOfRounds, allgather_time)
    
starting_time=time.time()
last_gather=time.time()
new_rank = rank
    
if varyT:
    rounds = 0   
    T = 0.01 #start T of varyT could be still too large for energies with parameterised with high eta
    while rounds < numberRoundsVaryT:
        it_no = 0
        accept=0
    
        writeData([it_no, T, 1, 0, geometry.V, geometry.A, geometry.C, geometry.X, geometry.V_0, geometry.A_0, geometry.C_0, geometry.X_0, geometry.curve_object.length, geometry.evaluate_normalised_energy(), frameNumber, 1.0, time.time(), rank, probability_expectation, deltaE_increasing_expectation], fileName+'varyT')
        while (time.time()-last_gather < numberSecondsBetweenUpdatingTempByVaryT):
            it_no+=1

            (tmp_0or1, prob, deltaE) = do_move_and_check_energy_and_accept_or_reject(geometry, dMin, dMax, T)
            accept+=tmp_0or1

            #save iteration data
            if (it_no)%100==0:
                writeData([it_no, T, prob, deltaE, geometry.V, geometry.A, geometry.C, geometry.X, geometry.V_0, geometry.A_0, geometry.C_0, geometry.X_0, geometry.curve_object.length, geometry.evaluate_normalised_energy(), frameNumber,  accept/it_no, time.time(), rank, probability_expectation, deltaE_increasing_expectation], fileName+'varyT')

            #print out poly file
            if (it_no + accept)%50==0:
                geometry.curve_object.make_curve_polyFile(fileLocation, polyFileName+str(frameNumber))
                geometry.curve_object.save_data(fileLocation, polyFileName+'_'+str(frameNumber))
                #geoClass.makeFilFile([pt for subList in geometry.curve_object.curve_vertices for pt in subList], 'trefoil40_'+str(frameNumber+63), geometry.input_R)
                frameNumber+=1

            if it_no%50==0:
                print("Info from rank", rank, " (E - E0)/L", round(geometry.evaluate_normalised_energy(),3), "T =", round(T,3), "acceptRatio", round(accept/it_no, 3), "(minRads, minSelfDist) = ", list(map(lambda x: round(x, 5), geometry.curve_object.check_reach())), frameNumber)

    
            #compute expected prob and deltaE_>0 ---> this could be just a massive waste of computation power since you don't actually use this to pitch the temperature...
            if deltaE>0:
                if len(deltaE_increasing_list) > dragFactor:
                    deltaE_increasing_expectation = updateExpectation(deltaE_increasing_list, deltaE, deltaE_increasing_expectation)
                    probability_expectation = updateExpectation(probability_list, prob, probability_expectation)
                else:
                    deltaE_increasing_list.append(deltaE)
                    probability_list.append(prob)
                    deltaE_increasing_expectation = sum(deltaE_increasing_list)/len(deltaE_increasing_list)
                    probability_expectation = sum(probability_list)/len(probability_list)

        #varyT
        median = np.median(np.array(deltaE_increasing_list))
        T = updateT0(median, medianProbability)
        medianProbability*1.06
        print(rank, "median(deltaE_>0)=", median, "new temp set to T=",T)

        last_gather=time.time()
        rounds+=1

#what you could do is adopt the temperatures from varyT --> if you are planning to change the set up though here should be the point where the new parallel implementation comes in.
T = float(sys.argv[3])
rounds = 0
while rounds < numberOfRounds:
    it_no = 0
    accept=0
    
    writeData([it_no, T, 1, 0, geometry.V, geometry.A, geometry.C, geometry.X, geometry.V_0, geometry.A_0, geometry.C_0, geometry.X_0, geometry.curve_object.length, geometry.evaluate_normalised_energy(), frameNumber, 1.0, time.time(), new_rank], fileName)
    while (time.time()-last_gather < allgather_time):
        it_no+=1
    
        (tmp_0or1, prob, deltaE) = do_move_and_check_energy_and_accept_or_reject(geometry, dMin, dMax, T)
        accept+=tmp_0or1

        #save iteration data
        if (it_no)%100==0:
            writeData([it_no, T, prob, deltaE, geometry.V, geometry.A, geometry.C, geometry.X, geometry.V_0, geometry.A_0, geometry.C_0, geometry.X_0, geometry.curve_object.length, geometry.evaluate_normalised_energy(), frameNumber,  accept/it_no, time.time(), new_rank], fileName)

        #print out poly file
        if (it_no + accept)%50==0:
            geometry.curve_object.make_curve_polyFile(fileLocation, polyFileName+str(frameNumber))
            geometry.curve_object.save_data(fileLocation, polyFileName+'_'+str(frameNumber))
            #geoClass.makeFilFile([pt for subList in geometry.curve_object.curve_vertices for pt in subList], 'trefoil40_'+str(frameNumber+63), geometry.input_R)
            frameNumber+=1

        if it_no%50==0:
            print("Info from rank", rank, " (E - E0)/L", geometry.evaluate_normalised_energy(), "T =", T, "acceptRatio", accept/it_no, "(minRads, minSelfDist) = ", geometry.curve_object.check_reach())

        #compute expected prob and deltaE_>0 --- the problem here is that you swop the geometries... so you also should swop the gathered data, otherwise it doesn't make sense.

    #mix the states
    geometry.evaluate_measures()
    allgather_data["geometry"] = geometry
    allgather_data["energy"] = geometry.evaluate_normalised_energy()

    alldatas=comm.allgather(allgather_data)
    
    #compute probabilities as list
    tmp_list = [returnWeight(x, T) for x in [alldatas[i]["energy"] for i in range(size)]]
    tmp_float = sum(tmp_list)
    mixing_probabilities = [tmp_list[i]/tmp_float for i in range(size)]
    if rank==0:
        print("---> now mixing the states \n")
        print([alldatas[i]["energy"] for i in range(size)])
        print('mixing probabilities', mixing_probabilities)
    #bias the pdf to choose the current strand with 0.5 liklihood over swopping 
    dont_swop_bias = [0.5 if i == rank else 0.5/(size - 1) for i in range(size)] 
    biased_probabilities = [0.5*(mixing_probabilities[i] + dont_swop_bias[i]) for i in range(size)]
    if rank==0:
        print("\n biased probabilties", biased_probabilities)
    new_rank = random.choices(range(size), mixing_probabilities, k=1)[0]
    #make an extra check to make sure we have not chucked out the lowest energy configuration
    if abs(allgather_data["energy"] - min([alldatas[i]["energy"] for i in range(size)]))< 0.05:
        new_rank = rank
        print("rank ", rank, "is more--or--less the minimum so not swopping")
    else:
        new_rank = random.choices(range(size), biased_probabilities, k=1)[0]
        print(rank, " will adopt", new_rank)
    geometry = alldatas[new_rank]["geometry"]
    
    #check
    geometry.evaluate_measures()
    allgather_data["geometry"] = geometry
    allgather_data["energy"] = geometry.evaluate_normalised_energy()
    print(rank, " is of energy", allgather_data["energy"])

    #decrease T
    T = decrease(T)

    last_gather=time.time()
    rounds+=1

