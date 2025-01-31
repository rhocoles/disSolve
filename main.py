import numpy as np
import random
from copy import deepcopy

import decimal as dec
from datetime import datetime

import time
import sys
import os
import shutil
from mpi4py import MPI
import socket

from geometryClass import ThreadedBeads

import pointFilaments

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
    """ Function: do_move_and_check_energy
    ----------------------------------------------------------------------------------------------
    This is the bit from the annealing algorithm which really can't be interuppted
    geometry object of type biarc or threadedBeadCurve
    dMax float maximum distance moved over one iteration
    T positive float temperature
    """
    moveReturn = geometry.move(dMin=dMin, dMax=dMax)
    if moveReturn == 1:#configuration jammed all processes should end
        print('jammed on rank'+str(rank))
        beadedCurve.save_data('./','jammedGeometry')
        sys.exit()
    else:
        d, indices, newPos =  moveReturn
    
    #update new positions to generate a neighbouring configuration
    tmpGeometryData = deepcopy(geometry.data)
    for (i,j) in indices[1:-1:]:
        tmpGeometryData[i][j]=newPos.pop(0)
    
    (deltaE, measures) = geometry.evaluate_energy_difference(tmpGeometryData)

    #accept or reject energy if increased
    (tmp_0or1, prob) = acceptOrReject(deltaE, T)#calculate probability of achieving deltaE choose via random number generation
    if tmp_0or1==1:
        geometry.data = tmpGeometryData
        geometry.V = measures[0]
        geometry.A = measures[1]
        geometry.C = measures[2]
        geometry.X = measures[3]
            
    return (tmp_0or1, prob, deltaE, d)

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

comm = MPI.COMM_WORLD
rank = comm.Get_rank() #number of this parallel process
size = comm.Get_size() #number of parallel processes in total
host = socket.gethostname() #name of computer
expName = sys.argv[1]+'rs'+str(round(float(sys.argv[3]), 2))+'_eta'+str(round(float(sys.argv[5]), 2))#structure + gridNumber ---> make some acronym out of this as appropriate.

if rank==0:
    print(sys.argv, len(sys.argv))
directoryName = './data_'+str(rank)
fileLocation = '../polyFiles/' #need to move one down from the current directory directoryName
#polyFileName = expName+'_test'+str(rank)+"_"
polyFileName = 'test'+str(rank)+"_"
fileName = "test_"+str(rank)+"_"
frameNumber = 1

#Each parallel experiment (numbered by rank) is given its own directory called "data_#rank" in which the evaluation of the energy occurs and the data of the experiment is stored.
#At the beginning, the data_#Rank directory is removed, inputFiles directory removed, the directorys data_#Rank, inputFiles are made, the program switches to the directory data_#Rank, copies morphLf into current working directory
#directoryList = os.listdir()
#print(rank, directoryList, directoryName)
if 'data_'+str(rank) in os.listdir():
    os.chdir(directoryName)
    if 'data' in os.listdir():
        shutil.rmtree('data')
    os.mkdir('data')
    if 'inputFiles' in os.listdir():
        shutil.rmtree('inputFiles')
    os.mkdir('inputFiles')
else:
    os.mkdir(directoryName)
    os.chdir(directoryName)
    os.mkdir('data')
    os.mkdir('inputFiles')
shutil.copy2('../morph_local', '.')
#in directory directoryName from here on


#define coefficents which define the energy, define the input sphere radius
edgeLength = 0.25
rTube = 1.0
T = float(sys.argv[2])
overlapRatio = float(sys.argv[3]) #change this in the submit file 
eta = float(sys.argv[4])
#inputFileString = sys.argv[-1][:4:]+str(rank)+'.txt'#same directory level as polyFile directory
#inputFileString = sys.argv[-1][:4:]+str(rank%8)+sys.argv[-1][5::] #same directory level as polyFile directory
inputFileString = sys.argv[-1]

ThreadedBeads.set_radii(overlapRatio, rTube, edgeLength)
ThreadedBeads.set_coefficients(eta=eta)

#define the geometry
beadedCurve = ThreadedBeads(1, fileName='../../'+inputFileString)
#beadedCurve = ThreadedBeads(1, fileName='../initialConfigs/test'+str(rank)+'.txt')
#if rank%2 ==0:
#    beadedCurve = ThreadedBeads(1, fileName='../'+inputFileString)
#else:
#    beadedCurve = ThreadedBeads(1, curveData = pointFilaments.open_chain_25_dl0_25)
#beadedCurve = ThreadedBeads(1, curveData = pointFilaments.open_chain_50_dl0_25)
#beadedCurve = ThreadedBeads(1, curveData = pointFilaments.open_chain_25_dl0_25)
#beadedCurve = ThreadedBeads(1, fileName='../'+inputFileString)
#beadedCurve = ThreadedBeads(0, curveData = pointFilaments.circle36_thBe_dl10)
#beadedCurve = ThreadedBeads(0, curveData = pointFilaments.circle36_thBe_dl25)
beadedCurve.evaluate_embedded_measures()
beadedCurve.evaluate_measures()
beadedCurve.make_curve_polyFile(fileLocation, polyFileName+str(frameNumber))

#print(rank, beadedCurve.V_0, beadedCurve.A_0, beadedCurve.C_0, beadedCurve.X_0)
#print(rank, beadedCurve.V, beadedCurve.A, beadedCurve.C, beadedCurve.X)
#print("Initialised curve", rank,"of length", beadedCurve.length, "(E - E0)/L = ", beadedCurve.evaluate_normalised_energy(), "(minRads, minSelfDist) = ", beadedCurve.check_reach())


#set up the data to be communicated between processes
allgather_data={}
allgather_data["geometry"] = beadedCurve
allgather_data["energy"] = beadedCurve.evaluate_normalised_energy()
#allgather_data["temp"] = 0.0

#annealing schedule variables
#THIS NEEDS TO BE AUTOMATED!!!#TODO need to vary the temperature every so many iterations for the temperature to be effective. Then decrease in linear time steps. The amount of time spent at each time step may or maynot be adaptive.
allgather_time = int(sys.argv[6])
numberOfRounds = int(sys.argv[7])
dragFactor = 5000 #number of iterations over which expectations are computed.
dMin = 0.00001*overlapRatio
return_dMax = lambda x : 2*x + dMin
theta_bar = 0.5*overlapRatio
dMax = return_dMax(theta_bar)
deltaE_increasing_list = []
deltaE_increasing_expectation = 0.0
probability_list = []
probability_expectation = 1.0
acceptRatio = 1.0
medianProbability = 0.1
newRank = rank

#decrease = lambda x: x - float(sys.argv[5])
decrease = lambda x: x*float(sys.argv[5])
updateT0 = lambda x, y: -max(x, 0.005)/np.log(y)
varyT = int(sys.argv[8])
numberRoundsVaryT=int(sys.argv[9])
numberSecondsBetweenUpdatingTempByVaryT=int(sys.argv[10])
T = float(sys.argv[2])

if rank==0:
    print('each round is ', allgather_time, ' seconds')
    print('experiment lasts for ', numberOfRounds)
    print('initial temperature is T0 = ', T)
    print("starting experiment which will end in ", (numberRoundsVaryT*numberSecondsBetweenUpdatingTempByVaryT + numberOfRounds*allgather_time)/(60*60*24), "days from now")

writeData([size, rank, 1.0, overlapRatio, ThreadedBeads.input_R, sum(beadedCurve.sphereCount), T, allgather_time] + ThreadedBeads.prefactors + [beadedCurve.length, beadedCurve.compute_average_edge_length(), beadedCurve.V_0, beadedCurve.A_0, beadedCurve.C_0, beadedCurve.X_0, beadedCurve.evaluate_normalised_energy(), beadedCurve.V, beadedCurve.A, beadedCurve.C, beadedCurve.X, eta] , 'experimentData')
    
starting_time=time.time()
last_gather=time.time()
    
if varyT:
    rounds = 0   
    T = 0.01 #start T of varyT could be still too large for energies with parameterised with high eta
    while rounds < numberRoundsVaryT:
        it_no = 0
        accept=0
    
        beadedCurve.evaluate_measures()
        writeData([it_no, T, 1, 0, 0, beadedCurve.V, beadedCurve.A, beadedCurve.C, beadedCurve.X, frameNumber, 0, probability_expectation, deltaE_increasing_expectation, time.time(), newRank], fileName+'varyT')
        while (time.time()-last_gather < numberSecondsBetweenUpdatingTempByVaryT):
            it_no+=1
    
            (tmp_0or1, prob, deltaE, d) = do_move_and_check_energy_and_accept_or_reject(beadedCurve,dMin, dMax, T)
            accept+=tmp_0or1

            #save iteration data
            if (it_no)%50==0:
                writeData([it_no, T, prob, deltaE, d, beadedCurve.V, beadedCurve.A, beadedCurve.C, beadedCurve.X, frameNumber, accept/it_no, probability_expectation, deltaE_increasing_expectation, time.time(), newRank], fileName+'varyT')

            #print out poly file
            if (it_no + accept)%500==0:
                beadedCurve.make_curve_polyFile(fileLocation, polyFileName+str(frameNumber))
                beadedCurve.save_data(fileLocation, polyFileName+'_'+str(frameNumber))
                frameNumber+=1

            if it_no%500==0:
                beadedCurve.evaluate_measures()
                print("Info from rank", rank, " (E - E0)/L", beadedCurve.evaluate_normalised_energy(), "T =", T, "acceptRatio", accept/it_no, "expected probability", probability_expectation, "(minRads, minSelfDist) = ", beadedCurve.check_reach())

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

#allgather_data["temp"] = t
#if varyt:#only enter the intermediate round if varyt ... swop all configs which are still embedded with the lowest energy curve
#
#    beadedcurve.evaluate_measures()
#    allgather_data["geometry"] = beadedcurve
#    allgather_data["energy"] = beadedcurve.evaluate_normalised_energy()
#
#    alldatas=comm.allgather(allgather_data)
#    #get the curve of minimum energy
#    minrank = [alldatas[i]["energy"] for i in range(size)].index(min([alldatas[i]["energy"] for i in range(size)]))
#    embeddedenergy = beadedcurve.evaluate_embedded_energy()
#    #if the curve is less than 0.1% of the energy of the embedded curve then replace this geometry with the curve of least energy
#    if allgather_data["energy"] > - 0.001*embeddedenergy/beadedcurve.length:
#        #print(allgather_data["energy"], - 0.001*embeddedenergy/beadedcurve.length, embeddedEnergy, beadedCurve.length)
#        beadedcurve = alldatas[minrank]["geometry"]
#        t = alldatas[minrank]["temp"]
#        print("curve at process ", rank, "is more--or--less embedded and is therefore replace by the curve at process ", minRank)
#        #print(beadedcurve.evaluate_energy()/embeddedenergy)
#
##intermediate round without swopping but average temp of all rounds
##allgather_data["temp"] = t
#beadedcurve.evaluate_measures()
#allgather_data["geometry"] = beadedcurve
#allgather_data["energy"] = beadedcurve.evaluate_normalised_energy()
#
#alldatas=comm.allgather(allgather_data)
#t = sum([alldatas[i]["temp"] for i in range(size)])/float(size)
#if rank==0:
#    print("temp set to ", t)
#del allgather_data['temp'] #only cos you don't use the temp after this
#
#if varyt:#only enter the intermediate round if varyt
#    it_no = 0
#    accept=0
#    beadedcurve.evaluate_measures()
#    writedata([it_no, t, 1, 0, 0, beadedcurve.v, beadedcurve.a, beadedcurve.c, beadedCurve.X, frameNumber, 0, probability_expectation, deltaE_increasing_expectation, time.time(), newRank], fileName+'varyT')
#    while (time.time()-last_gather < numbersecondsbetweenupdatingtempbyvaryt):
#        it_no+=1
#    
#        (tmp_0or1, prob, deltae, d) = do_move_and_check_energy_and_accept_or_reject(beadedCurve,dMin, dMax, T)
#        accept+=tmp_0or1
#
#        #save iteration data
#        if (it_no)%50==0:
#            writedata([it_no, t, prob, deltae, d, beadedcurve.v, beadedcurve.a, beadedCurve.C, beadedCurve.X, frameNumber, accept/it_no, probability_expectation, deltaE_increasing_expectation, time.time(), newRank], fileName+'varyT')
#
#        #print out poly file
#        if (it_no + accept)%500==0:
#            beadedcurve.make_curve_polyfile(filelocation, polyfilename+str(framenumber))
#            beadedcurve.save_data(filelocation, polyfilename+'_'+str(framenumber))
#            framenumber+=1
#
#        if it_no%500==0:
#            beadedcurve.evaluate_measures()
#            print("info from rank", rank, " (e - e0)/l", beadedcurve.evaluate_normalised_energy(), "T =", T, "acceptRatio", accept/it_no, "expected probability", probability_expectation, "(minRads, minSelfDist) = ", beadedCurve.check_reach())
#
#        #compute expected prob and deltae_>0 ---> this could be just a massive waste of computation power since you don't actually use this to pitch the temperature...
#        if deltae>0:
#            if len(deltae_increasing_list) > dragfactor:
#                deltae_increasing_expectation = updateexpectation(deltae_increasing_list, deltaE, deltaE_increasing_expectation)
#                probability_expectation = updateexpectation(probability_list, prob, probability_expectation)
#            else:
#                deltaE_increasing_list.append(deltaE)
#                probability_list.append(prob)
#                deltaE_increasing_expectation = sum(deltaE_increasing_list)/len(deltaE_increasing_list)
#                probability_expectation = sum(probability_list)/len(probability_list)

#now cooling without swopping
rounds = 0
while rounds < 0:
    it_no = 0
    accept=0
    
    beadedCurve.evaluate_measures()
    writeData([it_no, T, 1, 0, 0, beadedCurve.V, beadedCurve.A, beadedCurve.C, beadedCurve.X, frameNumber, 0, probability_expectation, deltaE_increasing_expectation, time.time(), newRank], fileName)
    while (time.time()-last_gather < allgather_time):
        it_no+=1
    
        (tmp_0or1, prob, deltaE, d) = do_move_and_check_energy_and_accept_or_reject(beadedCurve,dMin, dMax, T)
        accept+=tmp_0or1

        #save iteration data
        if (it_no)%50==0:
            writeData([it_no, T, prob, deltaE, d, beadedCurve.V, beadedCurve.A, beadedCurve.C, beadedCurve.X, frameNumber, accept/it_no, probability_expectation, deltaE_increasing_expectation, time.time(), newRank], fileName)

        #print out poly file
        if (it_no + accept)%500==0:
            beadedCurve.make_curve_polyFile(fileLocation, polyFileName+str(frameNumber))
            beadedCurve.save_data(fileLocation, polyFileName+'_'+str(frameNumber))
            frameNumber+=1

        if it_no%500==0:
            beadedCurve.evaluate_measures()
            print("Info from rank", rank, " (E - E0)/L", beadedCurve.evaluate_normalised_energy(), "T =", T, "acceptRatio", accept/it_no, "expected probability", probability_expectation, "(minRads, minSelfDist) = ", beadedCurve.check_reach())

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

    #mix the states
    beadedCurve.evaluate_measures()
    allgather_data["geometry"] = beadedCurve
    allgather_data["energy"] = beadedCurve.evaluate_normalised_energy()

    alldatas=comm.allgather(allgather_data)
    
    if rank==0:
        print("---> reducing the temperature all energies: \n")
        print([alldatas[i]["energy"] for i in range(size)])
    #decrease T
    T = decrease(T)

    last_gather=time.time()
    rounds+=1

#cooling with swopping 
rounds = 0
while rounds < numberOfRounds:
    it_no = 0
    accept=0
    
    beadedCurve.evaluate_measures()
    writeData([it_no, T, 1, 0, 0, beadedCurve.V, beadedCurve.A, beadedCurve.C, beadedCurve.X, frameNumber, 0, probability_expectation, deltaE_increasing_expectation, time.time(), newRank], fileName)
    while (time.time()-last_gather < allgather_time):
        it_no+=1
    
        (tmp_0or1, prob, deltaE, d) = do_move_and_check_energy_and_accept_or_reject(beadedCurve,dMin, dMax, T)
        accept+=tmp_0or1

        #save iteration data
        if (it_no)%50==0:
            writeData([it_no, T, prob, deltaE, d, beadedCurve.V, beadedCurve.A, beadedCurve.C, beadedCurve.X, frameNumber, accept/it_no, probability_expectation, deltaE_increasing_expectation, time.time(), newRank], fileName)

        #print out poly file
        if (it_no + accept)%500==0:
            beadedCurve.make_curve_polyFile(fileLocation, polyFileName+str(frameNumber))
            beadedCurve.save_data(fileLocation, polyFileName+'_'+str(frameNumber))
            frameNumber+=1

        if it_no%500==0:
            beadedCurve.evaluate_measures()
            print("Info from rank", rank, " (E - E0)/L", beadedCurve.evaluate_normalised_energy(), "T =", T, "acceptRatio", accept/it_no, "expected probability", probability_expectation, "(minRads, minSelfDist) = ", beadedCurve.check_reach())

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

    #mix the states
    beadedCurve.evaluate_measures()
    allgather_data["geometry"] = beadedCurve
    allgather_data["energy"] = beadedCurve.evaluate_normalised_energy()

    alldatas=comm.allgather(allgather_data)
    
    #compute probabilities as list
    tmp_list = [returnWeight(x, T) for x in [alldatas[i]["energy"] for i in range(size)]]
    tmp_float = sum(tmp_list)
    mixing_probabilities = [tmp_list[i]/tmp_float for i in range(size)]
    if rank==0:
        print("---> now mixing the states \n")
        print([alldatas[i]["energy"] for i in range(size)])
        print(mixing_probabilities)
    #bias the pdf to choose the current strand with 0.5 liklihood over swopping 
    dont_swop_bias = [0.5 if i == rank else 0.5/(size - 1) for i in range(size)] 
    biased_probabilities = [0.5*(mixing_probabilities[i] + dont_swop_bias[i]) for i in range(size)]
    if rank==0:
        print("going with the dont--swop biased pdf\n", biased_probabilities)
    #make an extra check to make sure we have not chucked out the lowest energy configuration
    if abs(allgather_data["energy"] - min([alldatas[i]["energy"] for i in range(size)]))< 0.05:
        newRank = rank
        print("rank ", rank, "is more--or--less the minimum so not swopping")
    else:
        newRank = random.choices(range(size), biased_probabilities, k=1)[0]
    print(rank, " will adopt", newRank)
    beadedCurve = alldatas[newRank]["geometry"]
    
    #check
    beadedCurve.evaluate_measures()
    allgather_data["geometry"] = beadedCurve
    allgather_data["energy"] = beadedCurve.evaluate_normalised_energy()
    print(rank, " is of energy", allgather_data["energy"])

    #decrease T
    T = decrease(T)

    last_gather=time.time()
    rounds+=1
