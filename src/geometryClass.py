import numpy as np
import random
#from scipy import optimize
import decimal as dec
import math as m
import sys
import subprocess
import os

import simple_functions as simp_func

def randomOrder(dataList , repetitions = 1):
    """Returns a shuffled list of all indicies in dataList with repetitions."""
    
    result=[]
    for i in range(len(dataList)):
        for j in range(len(dataList[i])):
            result.append((i,j))
    n=1
    while n<repetitions:
        result+=result
        n+=1

    random.shuffle(result)
    return result

def makePointCloudPoly(pointList, fileLocation, fileName):
    """Writes a each point position in a .poly file suitable for viewing in houdini."""

    point_count = 1
    f1 = open(str(fileLocation)+str(fileName)+'.poly', 'w')
    f1.write('POINTS''\n')
    for i in range(len(pointList)):
        pt = pointList[i]
        f1.write(str(point_count)+': '+str(dec.Decimal(str(round(pt[0],5))))+' '+str(dec.Decimal(str(round(pt[1],5))))+' '+str(dec.Decimal(str(round(pt[2],5))))+'\n')
        point_count = point_count + 1
    
    f1.write('POLYS\n')     
    f1.write('\nEND')       
    f1.close()
    return None

def makeCopyOf(dataList):
    """ Returns a copy of the dataList."""

    result = []
    for i in range(len(dataList)):
        strand = []
        for j in range(len(dataList[i])):
            strand.append(dataList[i][j])
        result.append(strand)
    return result

def countNumberOf(dataList):
    """Returns the total number of arcs/triangles/nodes contained in the dataList."""
    runningTotal = 0
    for i in range(len(dataList)):
        runningTotal+=len(dataList[i])
    return runningTotal

def returnMinimumInNestedListOfFloats(dataList):
    """Returns the minimum in nested list of floats."""
    minimum = 100000.0
    for i in range(len(dataList)):
        for j in range(len(dataList[i])):
            if minimum>dataList[i][j]:
                minimum=dataList[i][j]
    return minimum

def returnLengthOfPointList(self, pointList):
    """Returns the total distance between the points on the pointList."""

    runningTotal = 0
    if self.configType == "open":
        print("free energy line returnLengthOfInterpolation you haven't dealt with this yet")
        quit()
    for i in range(len(pointList)):
        for j in range(len(pointList[i])):
            runningTotal+=np.linalg.norm(np.array(pointList[i][j]) - np.array(pointList[i][(j+1)%len(pointList[i])]))
    #print("total spheres used to compute measures ", sum([len(pointList[i]) for i in range(len(pointList))]))
    return runningTotal

def returnRadiusOfSphere(self, edgeLength):
    """Set the sphere radius of the interpolating spheres."""
    rTube = 1.0
    return round(m.sqrt(rTube**2 + 0.25*(edgeLength)**2), 5)

def returnEmbeddedMeasures(pointList, radius, endCaps=True):
    """
    Function: returnEmbeddedMeasures
    --------------------------------------------------------------------------------------------------------------------------
    returns the volume of the union of spheres centred at points on pointList of radius radius
    the embedded volume assumes each Sphere intersects with the neighbouring adjacent spheres and no other sphere
    end points of arcs may or may not intersect and is calculated appropriately.
    pointList nestedList of three tuples
    radius float
    return float
    """
    volume = lambda x: m.pi*(x*radius**2 - pow(x,3)/12.0)
    area = lambda x: 2*m.pi*radius*x
    intMeanCurv = lambda x: 2*m.pi*(x - m.sqrt(np.clip(radius**2 - 0.25*x**2, 0, 10))*np.arcsin(np.clip(0.5*x/radius, -1, 1)))
    V = 0
    A = 0
    C = 0
    X = 0
    for i in range(len(pointList)):
        for k in range(len(pointList[i])-1):
            V+=volume(np.linalg.norm(np.around(np.array(pointList[i][k]), decimals=5) - np.around(np.array(pointList[i][k+1]), decimals=5)))
            A+=area(np.linalg.norm(np.around(np.array(pointList[i][k]), decimals=5) - np.around(np.array(pointList[i][k+1]), decimals=5)))
            C+=intMeanCurv(np.linalg.norm(np.around(np.array(pointList[i][k]), decimals=5) - np.around(np.array(pointList[i][k+1]), decimals=5)))
        if endCaps:
            V+=4*m.pi*pow(radius, 3)/3.0
            A+=4*m.pi*radius**2
            C+=4*m.pi*radius
            X+=4*m.pi
        else:
            V+=volume(np.linalg.norm(np.around(np.array(pointList[i][k+1]), decimals=5) - np.around(np.array(pointList[i][0]), decimals=5)))
            A+=area(np.linalg.norm(np.around(np.array(pointList[i][k+1]), decimals=5) - np.around(np.array(pointList[i][0]), decimals=5)))
            C+=intMeanCurv(np.linalg.norm(np.around(np.array(pointList[i][k+1]), decimals=5) - np.around(np.array(pointList[i][0]), decimals=5)))
            #X=0 if endCaps==False we assume the structure is closed and hence represents a torus which has euler characteristic 0
    return [V,A,C,X]

def pressure(R, eta):
    """
    Function: pressure
    -----------------------------------
    returns float calculated from the formula expression for pressure given in equation.24 Density functional theory for hard-sphere mixtures: the White Bear version mark II
    R float (solvent radius)
    eta float (packing fraction)
    return float
    """
    fraction = (1 + eta + eta**2 - pow(eta, 3))/pow(1- eta, 3)
    prefactor = 3/(4*m.pi)
    return prefactor*pow(R, -3)*eta*fraction

def sigma(R, eta):
    """
    Function: sigma
    -----------------------------------
    returns float calculated from the formula expression for sigma given in equation.24 Density functional theory for hard-sphere mixtures: the White Bear version mark II
    R float (solvent radius)
    eta float (packing fraction)
    return float
    """
    fraction = (1 + 2*eta + 8*eta**2 - 5*pow(eta, 3))/(3*pow(1 - eta, 3))
    prefactor = 3/(4*m.pi)
    return -prefactor*pow(R, -2)*eta*(fraction + m.log(1-eta)/(3*eta))

def kappa(R, eta):
    """
    Function: kappa
    -----------------------------------
    returns float calculated from the formula expression for kappa given in equation.24 Density functional theory for hard-sphere mixtures: the White Bear version mark II
    R float (solvent radius)
    eta float (packing fraction)
    return float
    """
    fraction = (4 - 10*eta + 20*eta**2 - 8*pow(eta, 3))/(3*pow(1- eta, 3))
    prefactor = 3/(4*m.pi)
    return prefactor*pow(R, -1)*eta*(fraction + 4*m.log(1-eta)/(3*eta))

def kappaBar(eta):
    """
    Function: kappaBar
    -----------------------------------
    returns float calculated from the formula expression for kappaBar given in equation.24 Density functional theory for hard-sphere mixtures: the White Bear version mark II
    eta float (packing fraction)
    return float
    """
    fraction = (-4 + 11*eta - 13*eta**2 + 4*pow(eta, 3))/(3*pow(1- eta, 3))
    prefactor = 3/(4*m.pi)
    return prefactor*eta*(fraction - 4*m.log(1-eta)/(3*eta))

def makeFilFile(pointList,fileName, radius):
    """
    Function: makeFilFile
    ---------------------------------------------------------------------------------------------------------------------
    writes the point list to .fil file as input for Roland's program
    nested list of points
    r float
    fileName string
    """
    f1 = open('./inputFiles/'+str(fileName)+'.fil', 'w')
    for i in range(len(pointList)):
        f1.write(str(dec.Decimal(str(round(pointList[i][0],5))))+' '+str(dec.Decimal(str(round(pointList[i][1],5))))+' '+str(dec.Decimal(str(round(pointList[i][2],5))))+' '+str(dec.Decimal(str(radius)))+'\n')
    f1.close()
    return None

def makeObjFile(pointList, fileName):
    """
    Function: makeObjFile
    ---------------------------------------------------------------------------------------------------------------------
    writes the point list to .fil file as input for Roland's program
    nested list of points
    r float
    fileName string
    """
    f1 = open('./inputFiles/'+str(fileName)+'.obj', 'w')
    for i in range(len(pointList)):
        f1.write('v '+str(dec.Decimal(str(round(pointList[i][0],5))))+' '+str(dec.Decimal(str(round(pointList[i][1],5))))+' '+str(dec.Decimal(str(round(pointList[i][2],5))))+'\n')
    f1.close()
    return None


class Biarcs:
    """ Geometry class of biarcs, this is a set of isoceles triangles sharing common verticies and parallel short edges with adjacent triangles """ 

    # here you can put class variables shared by all instances
    
    def __init__(self,  fileName=''):
        if fileName=='':
            self.data = []
        else:
            f = open(str(fileName), 'r')
            readList = f.read().split('\n')
            f.close()
            self.data = []
            while len(readList)>1:
                strand = []
                line = readList.pop(0)
                while line != 'END':
                    line = line.split(' ')
                    b0 = (float(line[0]), float(line[1]), float(line[2]))
                    b1 = (float(line[3]), float(line[4]), float(line[5]))
                    b2 = (float(line[6]), float(line[7]), float(line[8]))
                    strand.append((b0,b1,b2))
                    line = readList.pop(0)
                    if np.linalg.norm(np.array(b2) - np.array(strand[0][0]))<0.00001:#float conversion is accurate up to five decimal places
                        self.data.append(strand)
                        strand=[]
                if len(strand)>0:
                    self.data.append(strand)

        self.ropelength = 0.0    # instance variables unique to each instance, these values are then changed in a method
        self.length = 0.0
        self.configType = "closed"
        self.arcPairs = []
        self.edgeWeights = []
        self.sphereCount = [3*len(self.data[i]) for i in range(len(self.data))]#number of spheres per stand to interpolate the curve

    def set_ropelength(self, value):
        self.ropelength = value

    def evaluate_curve_length(self):
        """Evaluates the length of the curve and saves in the variable length."""
        
        result = []
        runningTotal = 0
        for i in range(len(self.data)):
            result.append(runningTotal)
            for j in range(len(self.data[i])):
                (b0,b1,b2) = self.data[i][j]
                runningTotal+= simp_func.returnArcLengthForBezierTriangle(b0, b1, b2)
            result[-1] = runningTotal - result[-1]
        self.length = runningTotal
        #print(result)
        return result
    
    def recenter_the_curve(self):
        """Curve is translated so that the centre of mass is at the origin"""

        avgPos = np.zeros(3)
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                avgPos+=np.array(self.data[i][j][1])
        avgPos = avgPos/float(countNumberOf(self.data))

        for i in range(len(self.data)):
            strand = []
            for j in range(len(self.data[i])):
                strand.append((tuple(np.array(self.data[i][j][0]) - avgPos), tuple(np.array(self.data[i][j][1]) - avgPos), tuple(np.array(self.data[i][j])[2] - avgPos)))
            self.data[i] = strand
            
        return None
        
    def bisect(self):
        """All triangles are bisected, as a result the number of arcs doubles."""
        
        for i in range(len(self.data)):
            strand = []
            for j in range(len(self.data[i])):
                (b0,b1,b2) = self.data[i][j]
                (tri0, tri1) = simp_func.bisectBezierTraingle(b0, b1, b2)
                strand.append(tri0)
                strand.append(tri1)
            self.data[i]=strand

    def rescale_curve_length_to_ropelength(self, ropelength=10.0, force=False):
        """The curve data points are scalar multiplied such that the length equals ropelength if the length was less than ropelength, if force=True then also if the length is larger than ropelength."""

        if self.ropelength==0:
            print("ropelength is set to ", ropelength)
            self.set_ropelength(ropelength)

        self.evaluate_curve_length()

        if force or self.length<self.ropelength:
            scalingFactor = self.ropelength/self.length
            for i in range(len(self.data)):
                strand = []
                for j in range(len(self.data[i])):
                    (a0, a1, a2) = self.data[i][j]
                    strand.append((tuple(np.array(a0)*scalingFactor), tuple(np.array(a1)*scalingFactor), tuple(np.array(a2)*scalingFactor)))
                self.data[i] = strand
            self.evaluate_curve_length()

    def evaluate_arc_pairs_to_be_checked(self, radius = 1.0):
        """Appends an integer pair ((i,j), (l,k)) to the list arcPairs if the corresponding arcs have an arc length separation of at least pi."""

        indexNextJ = lambda ind : (ind[1] + 1)%len(self.data[ind[0]])
        indexPrevJ = lambda ind : (ind[1] - 1) + len(self.data[ind[0]])*(ind[1]==0)
        tightestCurve = lambda edgeLength: 2*radius*np.arctan(edgeLength)

        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                indiciesToSkip = [(i,j)]
                runningTotal = 0
                s = j
                while runningTotal < m.pi:
                    s = indexNextJ((i,s))
                    runningTotal+=tightestCurve(np.linalg.norm(np.array(self.data[i][s][0]) - np.array(self.data[i][s][1])))
                    indiciesToSkip.append((i,s))
                runningTotal = 0
                s = j
                while runningTotal < m.pi:
                    s = indexPrevJ((i,s))
                    runningTotal+=tightestCurve(np.linalg.norm(np.array(self.data[i][s][0]) - np.array(self.data[i][s][1])))
                    indiciesToSkip.insert(0, (i,s))

                for l in range(i, len(self.data)):#interested only in unordered pairs
                    if l==i:
                        stIndex = j+1
                    else:
                        stIndex = 0
                    for k in range(stIndex, len(self.data[l])):
                        if (l,k) in indiciesToSkip:
                            continue
                        self.arcPairs.append(((i,j), (l,k)))

    def store_edge_weight_information(self):
        """Stores the scalar value l0/(l0 + l1) where l0+l1 is the length of the two parallel shorts sides of adjacent triangles with l1 being the edge of the next triangle in clockwise rotation."""

        if self.configType=='open':
            print("to do, edge weights list will be missing the first and last entries-->or just the last?")
            return None

        for i in range(len(self.data)):
            strand = []
            for j in range(len(self.data[i])):
                l0 = np.linalg.norm(np.array(self.data[i][j][2]) - np.array(self.data[i][j][1]))
                l1 = np.linalg.norm(np.array(self.data[i][(j+1)%len(self.data[i])][1]) - np.array(self.data[i][(j+1)%len(self.data[i])][0]))
                #print(l0, l1, l0+l1, "l0 + l1 = dl")
                strand.append(l0/(l0+l1))
            self.edgeWeights.append(strand)

    
    def check_new_positions_do_not_cause_overlaps(self, radius, epsilon, newTrianglePositions, indicies):
        """Return True if the curve, defined by self with the newTrianglePositions of triangles given at index locations indicies, is such that distinct arcs are further than 2radius apart."""
        closestDistanceBound = 2.*radius
        #later you need to change the curve.data set so that it is a set of numpy arrays
        for ((i,j), (l,k)) in self.arcPairs:
            if (i,j) in indicies[1:-1:]:
                if (l,k) in indicies[1:-1:]:#both arc pairs belong to the rotated group
                    continue
                #check distance
                (a0, a1, a2) = newTrianglePositions[indicies.index((i,j))]
                a = np.array(a0 + a1 + a2).reshape(3,3)
                if (l,k) in [indicies[0], indicies[-1]]:#second pair belongs to a join arc, first pair to rotated arc
                    (b0, b1, b2) = newTrianglePositions[indicies.index((l,k))]
                else:#second pair is unmoved, first pair rotated.
                    (b0, b1, b2) = self.data[l][k]
                b = np.array(b0 + b1 + b2).reshape(3,3) 
                if simp_func.cDistArcToArcWithinErrorBound(a, b, closestDistanceBound, epsilon):#returns 1 if dl2l + error_a + error_b < closestDistanceBound or dl2l + error_a + error_b < closestDistanceBound and error_a + error_b < epsilon
                    return 0
            elif (i,j) in [indicies[0], indicies[-1]]:#first pair belongs to a join arc, second pair to any arc
                (a0, a1, a2) = newTrianglePositions[indicies.index((i,j))]
                a = np.array(a0 + a1 + a2).reshape(3,3)
                if (l,k) in indicies:
                    (b0, b1, b2) = newTrianglePositions[indicies.index((l,k))]
                else:
                    (b0, b1, b2) = self.data[l][k]
                b = np.array(b0 + b1 + b2).reshape(3,3) 
                if simp_func.cDistArcToArcWithinErrorBound(a, b, closestDistanceBound, epsilon):#returns 1 if dl2l + error_a + error_b < closestDistanceBound or dl2l + error_a + error_b < closestDistanceBound and error_a + error_b < epsilon
                    return 0
            elif (l,k) in indicies:#first pair is unmoved, second pair join or rotated arc
                (a0, a1, a2) = self.data[i][j]
                a = np.array(a0 + a1 + a2).reshape(3,3)
                (b0, b1, b2) = newTrianglePositions[indicies.index((l,k))]
                b = np.array(b0 + b1 + b2).reshape(3,3) 
                if simp_func.cDistArcToArcWithinErrorBound(a, b, closestDistanceBound, epsilon):#returns 1 if dl2l + error_a + error_b < closestDistanceBound or dl2l + error_a + error_b < closestDistanceBound and error_a + error_b < epsilon
                    return 0
        return 1


    def move(self, radius, epsilon, dMax=0.015, Nmax=0):
        """A new position of the curve is evaluated close to the original position such that the tube surrounding the curve of radius radius is embedded

        arguments:
        radius float
        epsilon float defines the accuracy with which the arc-to-arc distance between arc pairs included in the list self.arcPairs should be by the line-to-line distance between said pairs

        keyword arguments:
        dMax float maximum distance points belonging to the curve are moved
        Nmax integer maximum additional number of arcs moved

        return:
        list of the same stucture as self.data, integer equally the number of attempts, float distance moved
        """ 
        dAngle = lambda  x, y: 2*np.arcsin(np.clip(0.5*y/x, 0, 1))
        angleChoice = lambda x : [x, 2*m.pi - x]
        indexNext = lambda ind : (ind[0], (ind[1] + 1)%len(self.data[ind[0]]))
        randomStartIndex = randomOrder(self.data, repetitions = 5)
        while len(randomStartIndex)>0:
            stIndex = randomStartIndex.pop()
            indicies= [stIndex]
            N = np.random.randint(3, 4 + Nmax)
            while len(indicies)<N:#you need to make sure Nmax + 4 < half the number of arcs in the smallest strand
                indicies.append(indexNext(indicies[-1]))
                #if shape.configType=='open' and indicies[-1]==(stIndex[0], len(stIndex[0])-1):#this is probably wrong
                if self.configType=='open':
                    print("move to dos")
                    quit()

            #define the transformation
            normalVector = np.array(self.data[indicies[-1][0]][indicies[-1][1]][1]) - np.array(self.data[indicies[0][0]][indicies[0][1]][1])
            normalVector = normalVector/np.linalg.norm(normalVector)

            rCircle = [0]
            posVector= [np.zeros(3)]
            dotProd = [0]
            for n in range(len(indicies)-1):
                #dataList[i][j] = (b0, b1, b2) we want to move b1
                posVector.append(posVector[-1] + np.array(self.data[indicies[n+1][0]][indicies[n+1][1]][1]) - np.array(self.data[indicies[n][0]][indicies[n][1]][1]))
                dotProd.append(np.dot(posVector[-1], normalVector))
                rCircle.append(m.sqrt(np.clip(np.dot(posVector[-1], posVector[-1]) - dotProd[-1]**2,0, 100)))

            rc = max(rCircle)
            if rc <0.00025:
                continue
            d_upperBound = dMax
            while d_upperBound>0.001:
                d = np.random.uniform(0.001, d_upperBound)
                angle = angleChoice(dAngle(rc, d))
                random.shuffle(angle)
        
                while len(angle)>0:
                    a = angle.pop(0)
                    #for each point in the index list other than the end nodes which are stationary rotate the point by angle about the normal vector
                    newPos = [np.array(self.data[indicies[0][0]][indicies[0][1]][1])]
                    for n in range(len(indicies)-1):
                        newPos.append(np.array(self.data[indicies[0][0]][indicies[0][1]][1]) + simp_func.rotate(posVector[n+1], a, normalVector))
                    
                    newTrianglePositions = []
                    
                    #check first if the local radius of curvature for the join arcs is larger than radius
                    b0 = self.data[indicies[0][0]][indicies[0][1]][0]
                    b1 = self.data[indicies[0][0]][indicies[0][1]][1]
                    b2 = tuple(np.array(self.data[indicies[0][0]][indicies[0][1]][1]) + self.edgeWeights[indicies[0][0]][indicies[0][1]]*(newPos[1] - np.array(self.data[indicies[0][0]][indicies[0][1]][1])))
                    #check curvature constraint
                    if simp_func.returnCircleRadiusForBezierTriangle(b0, b1, b2)<radius:
                        continue
                    newTrianglePositions.append((b0, b1, b2))
                    
                    bb0 = tuple(np.array(self.data[indicies[-1][0]][indicies[-1][1]][1]) - (1 - self.edgeWeights[indicies[-2][0]][indicies[-2][1]])*(np.array(self.data[indicies[-1][0]][indicies[-1][1]][1]) - newPos[-2]))
                    bb1 = self.data[indicies[-1][0]][indicies[-1][1]][1]
                    bb2 = self.data[indicies[-1][0]][indicies[-1][1]][2]
                    if simp_func.returnCircleRadiusForBezierTriangle(bb0, bb1, bb2)<radius:
                        continue
                    
                    n = 1
                    while n < len(indicies)-1:
                        b0 = b2
                        b1 = tuple(newPos[n])
                        b2 = tuple(newPos[n] + self.edgeWeights[indicies[n][0]][indicies[n][1]]*(newPos[n+1] - newPos[n])) 
                        newTrianglePositions.append((b0, b1, b2))
                        n+=1
                    
                    newTrianglePositions.append((bb0, bb1, bb2))
               
                    #check the overlapping arc condition
                    if self.check_new_positions_do_not_cause_overlaps(radius, epsilon, newTrianglePositions, indicies):
                        return d, indicies, newTrianglePositions

                d_upperBound=0.9*d_upperBound

        print("All Points Are Jammed!")
        return 1

    def make_curve_polyFile(self, fileLocation, fileName, pointsPerArcInterpolation=3, resolveArcs=False):
        """Each arc is interpolated with pointsPerArcInterpolation points and the resulting set of concatenated points is written as a .poly file to be read with Houdini"""

        r=str(round(204.0/255.0, 3))
        g=str(round(0/255.0, 3))
        b=str(round(204.0/255.0, 3))
        if resolveArcs==True:
            alpha = lambda x: x%2
        else:
            alpha = lambda x: 1

        pointList = []
        for i in range(len(self.data)):
            strand = []
            for j in range(len(self.data[i])):
                strand+= simp_func.circleInterpolationBezierTriangle(self.data[i][j], pointsPerArcInterpolation)
            pointList.append(strand)

        point_count = 1
        f1 = open(str(fileLocation)+str(fileName)+'.poly', 'w')
        f1.write('POINTS''\n')
        for i in range(len(pointList)):
            for j in range(len(pointList[i])):
                f1.write(str(point_count)+': '+str(dec.Decimal(str(round(pointList[i][j][0],5))))+' '+str(dec.Decimal(str(round(pointList[i][j][1],5))))+' '+str(dec.Decimal(str(round(pointList[i][j][2],5))))+' c('+r+','+g+','+b+','+str(alpha(int(j/pointsPerArcInterpolation)))+')\n')
                point_count = point_count + 1
    
        point_count = 1
        poly_count = 1
        f1.write('POLYS')
        for i in range(len(pointList)):
            f1.write('\n'+str(poly_count)+': ')
            for j in range(len(pointList[i])):
                f1.write(str(point_count)+' ')
                point_count+=1
            if self.configType=="closed":#visually close the loop
                f1.write(str(point_count - len(pointList[i]))+'')#bit of a hack
            poly_count+=1

        f1.write('\nEND')       
        f1.close()

    def make_biarc_polyFile(self, fileLocation, fileName):
        """Writes each triangle in the data to a .poly file to be read with Houdini."""

        r = 69/255.0
        g = 57/255.0
        b = 159/255.0
        f1 = open(str(fileLocation)+str(fileName)+'.poly', 'w')
        f1.write('POINTS''\n')
        pointCount = 1
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                (b0, b1, b2) = self.data[i][j]
                f1.write(str(pointCount)+': '+str(dec.Decimal(str(round(b0[0],5))))+' '+str(dec.Decimal(str(round(b0[1],5))))+' '+str(dec.Decimal(str(round(b0[2],5))))+' c('+str(r)+','+str(g)+','+str(b)+',1)'+'\n')
                f1.write(str(1+pointCount)+': '+str(dec.Decimal(str(round(b1[0],5))))+' '+str(dec.Decimal(str(round(b1[1],5))))+' '+str(dec.Decimal(str(round(b1[2],5))))+' c('+str(r)+','+str(g)+','+str(b)+',1)'+'\n')
                f1.write(str(2+pointCount)+': '+str(dec.Decimal(str(round(b2[0],5))))+' '+str(dec.Decimal(str(round(b2[1],5))))+' '+str(dec.Decimal(str(round(b2[2],5))))+' c('+str(r)+','+str(g)+','+str(b)+',1)'+'\n')
                pointCount+=3
    
        f1.write('POLYS\n')
        pointCount = 1
        polyCount = 1
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                f1.write(str(polyCount)+': '+str(pointCount)+' '+str(pointCount + 1)+'\n')
                f1.write(str(1 + polyCount)+': '+str(pointCount + 1)+' '+str(pointCount + 2)+'\n')
                f1.write(str(2 + polyCount)+': '+str(pointCount)+' '+str(pointCount + 2)+'\n')
                polyCount+=3
                pointCount+=3
        f1.write('\nEND')       
        f1.close()

    def save_data(self, fileLocation, fileName):
        """Write the position of the verticies for each triangle to file."""
        f1 = open(str(fileLocation)+str(fileName)+'.txt', 'w')
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                (b0, b1, b2) = self.data[i][j]
                f1.write(str(dec.Decimal(str(round(b0[0],5))))+' '+str(dec.Decimal(str(round(b0[1],5))))+' '+str(dec.Decimal(str(round(b0[2],5))))+' ')
                f1.write(str(dec.Decimal(str(round(b1[0],5))))+' '+str(dec.Decimal(str(round(b1[1],5))))+' '+str(dec.Decimal(str(round(b1[2],5))))+' ')
                f1.write(str(dec.Decimal(str(round(b2[0],5))))+' '+str(dec.Decimal(str(round(b2[1],5))))+' '+str(dec.Decimal(str(round(b2[2],5))))+'\n')
        f1.write('END\n')
        f1.close()


    def generates_list_arc_radii(self):
        """Creates list floats. Each float gives the radius of the arc defined by the bezier triangle at that index."""
        result = []
        for i in range(len(self.data)):
            strand = []
            for j in range(len(self.data[i])):
                (b0, b1, b2) = self.data[i][j]
                strand.append(simp_func.returnCircleRadiusForBezierTriangle(b0, b1, b2))
            result.append(strand)
        return result

    def generate_all_arc_to_arc_distances_faffing(self, epsilon=0.0025):
        """Creates list of floats. Each float is the closest distance of an arc to the arc at that index, such that the arc contains is a critical point of the distance function retricted to the curve."""

        for ((i,j), (l,k)) in [((0,7), (0,15)), ((0,4), (0,8))]:
            #da2a = simp_func.cReturnDistArcToArcBounds(np.array(self.data[i][j]), np.array(self.data[l][k]), epsilon)
            a = np.array(self.data[i][j][0] + self.data[i][j][1] + self.data[i][j][2]).reshape(3,3)
            b = np.array(self.data[l][k][0] + self.data[l][k][1] + self.data[l][k][2]).reshape(3,3)
            dl2l = simp_func.cLineToLineDistance(a, b)
            dl2l_old = simp_func.returnLineToLineDistance((self.data[i][j][0], self.data[i][j][2]), (self.data[l][k][0], self.data[l][k][2]))
            tri_list = [a]
            eps, om = simp_func.returnCurveToLineErrorFromBezierTriangle(self.data[i][j])
            error = eps
            print((i,j), eps, om)
            while eps > epsilon:
                tmp = []
                for tri in tri_list:
                    tmp+=simp_func.cBisectBezierTraingle(a)
                eps = eps/(2 + m.sqrt(2*(1 + om)))
                om = m.sqrt(0.5*(1 + om))
                tri_list = tmp
            print((i,j), (eps, om), len(tri_list) )
            tri_list = [b]
            eps, om = simp_func.returnCurveToLineErrorFromBezierTriangle(self.data[l][k])
            error+=eps
            print((l,k), eps, om)
            while eps > epsilon:
                tmp = []
                for tri in tri_list:
                    tmp+=simp_func.cBisectBezierTraingle(b)
                eps = eps/(2 + m.sqrt(2*(1 + om)))
                om = m.sqrt(0.5*(1 + om))
                tri_list = tmp
            print((l,k), (eps, om), len(tri_list) )
            da2a= simp_func.cReturnDistArcToArcBounds(a, b, epsilon)
            print(da2a)
            print("(", dl2l - error, " , ", dl2l + error , ")")
            print("d(a",(i, j),", a",(l, k),") =? "," dl2l = ", dl2l, dl2l_old, "\n" )
            #print("d(a",(i, j),", a",(l, k),") = ", a, b," dl2l = ", dl2l,"\n" )
        return None

    def generate_all_arc_to_arc_distances(self, epsilon=0.0025):
        """Creates list of floats. Each float is the closest distance of an arc to the arc at that index, such that the arc contains is a critical point of the distance function retricted to the curve."""

        for ((i,j), (l,k)) in self.arcPairs:
            #da2a = simp_func.cReturnDistArcToArcBounds(np.array(self.data[i][j]), np.array(self.data[l][k]), epsilon)
            a = np.array(self.data[i][j][0] + self.data[i][j][1] + self.data[i][j][2]).reshape(3,3)
            b = np.array(self.data[l][k][0] + self.data[l][k][1] + self.data[l][k][2]).reshape(3,3)
            da2a= simp_func.cReturnDistArcToArcBounds(a, b, epsilon)
            print("d(a",(i, j),", a",(l, k),") =",da2a, "\n" )
            #print("d(a",(i, j),", a",(l, k),") = ", a, b," dl2l = ", dl2l,"\n" )
        return None

    def c_generate_closest_arc_to_arc_distances(self, epsilon=0.0025, withInfo=False):
        """Creates list of floats. Each float is the closest distance of an arc to the arc at that index, such that the arc contains is a critical point of the distance function retricted to the curve."""

        if withInfo:
            print('\n')
        result = [[1000.0 for k in range(len(self.data[i]))] for i in range(len(self.data))]
        index = [[(i,j) for j in range(len(self.data[i]))] for i in range(len(self.data))]
        for ((i,j), (l,k)) in self.arcPairs:
            dij = result[i][j]
            dlk = result[l][k]
            a = np.array(self.data[i][j][0] + self.data[i][j][1] + self.data[i][j][2]).reshape(3,3)
            b = np.array(self.data[l][k][0] + self.data[l][k][1] + self.data[l][k][2]).reshape(3,3)
            (lb, ub) = simp_func.cReturnDistArcToArcBounds(a, b, epsilon)
            if lb<dij:
                result[i][j] = lb
                index[i][j] = (l,k)
            if lb<dlk:
                result[l][k] = lb
                index[l][k] = (i,j)
        if withInfo:
            for i in range(len(self.data)):
                for j in range(len(self.data[i])):
                    print((i,j), index[i][j], result[i][j])
            print('\n')
        return result

    def generate_closest_arc_distances(self, epsilon=0.0025, withInfo = False):
        """Creates list of floats. Each float is the closest distance of an arc to the arc at that index, such that the arc contains is a critical point of the distance function retricted to the curve."""

        if withInfo:
            print('\n')
        result = []
        for i in range(len(self.data)):
            strand = []
            for j in range(len(self.data[i])):
                d = 1000.0
                ind = (0,0)
                for l in range(len(self.data)):
                    for k in range(len(self.data[l])):
                        error_a, omega_a = simp_func.returnCurveToLineErrorFromBezierTriangle(self.data[i][j])
                        if (l,k) in [(i, j - 1 + len(self.data[i])*(j==0)), (i,j), (i, (j+1)%len(self.data[i]))]:
                            continue
                        error_b, omega_b =  simp_func.returnCurveToLineErrorFromBezierTriangle(self.data[l][k])
                        trianglePairsToCheck = [(self.data[i][j], self.data[l][k])]
                        while epsilon < error_a + error_b:
                            trianglePairsToBisect = []
                            while len(trianglePairsToCheck)>0:
                                ((a0,a1,a2), (b0,b1,b2)) = trianglePairsToCheck.pop()
                                if simp_func.doesNotContainASingleCriticalPoint(a0, a2, (b0,b1,b2)):
                                    continue
                                trianglePairsToBisect.append(((a0,a1,a2), (b0,b1,b2)))
                            if len(trianglePairsToBisect)==0:
                                break
                            trianglePairsToCheck = simp_func.bisectAllTrianglePairsInList(trianglePairsToBisect)
                            #evaluate the new error term
                            error_a = error_a/(2 + m.sqrt(2*(1 + omega_a)))
                            error_b = error_b/(2 + m.sqrt(2*(1 + omega_b)))
                            omega_a = 0.5*m.sqrt(0.5*(1 + omega_a))
                            omega_b = 0.5*m.sqrt(0.5*(1 + omega_b))
                        #print(error_a + error_b, len(trianglePairList))
                        while len(trianglePairsToCheck)>0:
                            ((a0,a1,a2), (b0,b1,b2)) = trianglePairsToCheck.pop()
                            dtmp = simp_func.returnLineToLineDistance((a0, a2), (b0, b2)) - error_a - error_b #dArcToArc is a least as large as this number
                            if dtmp<d:
                                d = dtmp
                                ind = (l,k)
                if withInfo:
                    print((i,j), ind, d)
                strand.append(d)
            result.append(strand)
        if withInfo:
            print('\n')
        return result
    
    def check_reach(self, withInfo=False):
        """Function returns the minimum arc radius and half the minimum distance of the closest arc to each arc along the curve."""

        return (returnMinimumInNestedListOfFloats(self.generates_list_arc_radii()), returnMinimumInNestedListOfFloats(self.generate_closest_arc_distances(withInfo=withInfo)), returnMinimumInNestedListOfFloats(self.c_generate_closest_arc_to_arc_distances(withInfo=withInfo)))

    def make_point_interpolation(self, enlarge=False):
        """Returns a nested list of points spaced such that the arclength separation between points along the curve self is equal."""

        strandLengths = self.evaluate_curve_length()
        result = []
        for i in range(len(self.data)):
            x = strandLengths[i]/float(self.sphereCount[i])#target sphere separation
            strand = []
            N = self.sphereCount[i]
            
            #pick random start index
            j = np.random.randint(0, len(self.data[i]))
            if self.configType == "open":
                print("sphere interpolation to do")
                quit()
            else:
                indexList = [k for k in range(j, len(self.data[i]))] + [k for k in range(0, j)]
            
            restLength = 0
            for j in indexList:
                (b0,b1,b2)=self.data[i][j]
                #get the relevant information to define the circumcircle
                t = np.array(b1) - np.array(b0)
                l = np.linalg.norm(t)
                t = t/l
                e = np.array(b2) - np.array(b0)
                D = np.linalg.norm(e)
                e = e/D
                delta = np.arccos(np.clip(np.dot(t, e), -1, 1))
                if delta < 0.00632456586131289:#np.arccos(0.99998) treat as a straight line
                    tau = restLength
                    deltaTau = x
                    while tau < D:
                        strand.append(tuple(np.array(b0) + tau*e))
                        tau+=deltaTau
                    restLength = tau - D
                else:   
                    r =l/np.tan(delta)
                    normal = np.cross(t, e)
                    #sinDelta = np.linalg.norm(normal)
                    normal = normal/np.linalg.norm(normal)
                    #binormal = np.cross(normal, e)
                    centre = 0.5*(np.array(b0) + np.array(b2)) + r*np.dot(t,e)*np.cross(normal, e)
                    u = np.array(b0) - centre
                    u=u/np.linalg.norm(u)
                    tau = restLength/r#caculates the start angle
                    deltaTau = x/r
                    while tau < 2*delta:
                        strand.append(tuple(np.array(centre) + r*simp_func.rotate(u, tau, normal)))
                        tau+=deltaTau
                    restLength = r*(tau - 2*delta)
            if len(strand)>N:
                strand.pop(-1)
            if len(strand)!=N:
                print("changed the number of points ?")
                print(triangles)
                print(N, len(strand)+1, np.linalg.norm(np.array(strand[0]) - np.array(strand[-1])))
                quit()
            result.append(strand)

        if enlarge:
            lengthOfPointList = returnLengthOfPointList(self, result)
            if lengthOfPointList<self.ropelength:
                scalingFactor = self.ropelength/lengthOfPointList
                for i in range(len(result)):
                    strand = []
                    for j in range(len(result[i])):
                        strand.append(tuple(np.array(result[i][j])*scalingFactor))
                result[i] = strand

        #print(lengthOfPointList, self.ropelength/lengthOfPointList, returnLengthOfPointList(self, result))

        return result

    length_of_point_interpolation = returnLengthOfPointList

    def set_point_interpolation_sphere_radius(self):
        return returnRadiusOfSphere(self, self.ropelength/float(sum([self.sphereCount[i] for i in range(len(self.data))])))

    def get_all_the_values_fixed_before_you_start(self, ropelength, forceSphereCount=[], enlarge=False, bisect=False, center=False):

        if enlarge:
            self.rescale_curve_length_to_ropelength(ropelength=ropelength)
        else:
            self.set_ropelength(ropelength)
            self.evaluate_curve_length()
    
        if bisect:
            self.bisect()

        if center:
            self.recenter_the_curve()

        self.store_edge_weight_information()
        #print(curve.edgeWeights)
        if len(forceSphereCount)==len(self.data):
            self.sphereCount = forceSphereCount
        else:#use the convention that forceSphereCount=[#] means use a total of # spheres.
            if len(forceSphereCount)>0:
                strandLengths = self.evaluate_curve_length()
                approximatePointSpacing = self.length/float(forceSphereCount[0])
                self.sphereCount = [int(strandLengths[i]/approximatePointSpacing) for i in range(len(self.data))]
            else:
                self.sphereCount = [3*len(self.data[i]) for i in range(len(self.data))]#number of spheres per stand to interpolate the curve
        print("Total number of interpolating spheres ",sum([self.sphereCount[i] for i in range(len(self.data))]), "with approximate sphere spacing", self.length/sum(self.sphereCount[i] for i in range(len(self.sphereCount))))
        
        #epsilon = (edgeLength/np.sin(np.arctan(edgeLength/rTube)) - rTube)/3.0#tightest curves you need to bisect approximately three times
        #print("set error margin", epsilon)
        
        self.evaluate_arc_pairs_to_be_checked()

    def __copy__(self):
        """Returns a copy of the object and a number of variables.""" 

        newone = type(self)()
        #newone.__dict__.update(self.__dict__) -> makes a copy of all the instance variables which may be unnecessary
        #for item in ['ropelength', 'configType', 'arcPairs', 'edgeWeights', 'sphereCount']: -> this also kind of overkill -> to make the pointList only need sphereCount, configType and poss. ropelength
        for item in ['ropelength', 'configType', 'sphereCount']:
            newone.__dict__[item] = self.__dict__[item]
        return newone

    def update_new_geometry(self, geometry):
        """updates self with certain attributes of geometry."""

        for item in ['arcPairs', 'edgeWeights']:
            self.__dict__[item] = geometry.__dict__[item]


class ThreadedBeads:
    """ Geometry class of threaded beads, this is a polygonal curve of equal edges.""" 
    
    prefactors = [0,0,0,0] #do this so that you save an if loop each time you evaluate the energy

    def __init__(self, openOrClosed, fileName='', curveData=[]):
        if fileName=='':
            if len(curveData)==0:
                print("You need to initialise the shape of the curve either from a .txt file or from a pointFilaments.pointListName")
            else:
                self.data = [[np.array(curveData[i][j]) for j in range(len(curveData[i]))] for i in range(len(curveData))]
        else:
            f = open(str(fileName), 'r')
            readList = f.read().split('\n')
            f.close()
            self.data = []
            while len(readList)>1:
                strand = []
                line = readList.pop(0)
                while line != 'END':
                    line = line.split(' ')
                    pt = np.array([float(line[0]), float(line[1]), float(line[2])])
                    strand.append(pt)
                    line = readList.pop(0)
                self.data.append(strand)

        self.configType = "open" if (openOrClosed) else "closed" #options are closed or open
        self.length = sum(self.compute_length())

        self.deltaStar = self.compute_deltaStar(1.0, self.compute_average_edge_length())
        self.skippedInteger = m.floor(0.5*m.pi/self.deltaStar)
        self.arcPairs = self.evaluate_arc_pairs_to_be_checked(self.skippedInteger)

        nMax = int(0.3*min([len(self.data[i]) for i in range(len(self.data))]))
        self.index_intervals_to_be_rotated = self.generate_index_intervals_to_be_rotated(nMax)
        self.sphereCount = [len(self.data[i]) for i in range(len(self.data))]
        

    def compute_average_edge_length(self):
        """computes the average of all edge lengths for each strand."""

        edgeTally = 0
        runningTotal = 0
        for i in range(len(self.data)):
            j = 1 
            while j<len(self.data[i]):
                runningTotal+=np.linalg.norm(self.data[i][j] - self.data[i][j-1]) 
                edgeTally+=1
                j+=1
            if self.configType=="closed":
                runningTotal+=np.linalg.norm(self.data[i][0] - self.data[i][-1]) 
                edgeTally+=1
        return round(runningTotal/float(edgeTally), 10)

    def compute_length(self):
        """Sums the length of each edge and saves in the variable length."""
        
        result = []
        runningTotal = 0
        for i in range(len(self.data)):
            result.append(runningTotal)
            j = 1 
            while j<len(self.data[i]):
                runningTotal+=np.linalg.norm(self.data[i][j] - self.data[i][j-1]) 
                j+=1
            if self.configType=='closed':
                runningTotal+=np.linalg.norm(self.data[i][0] - self.data[i][-1]) 
            result[-1] = runningTotal - result[-1]
        #self.length = runningTotal
        return result

    def check_equal_edges(self):
        """Prints the smallest and largest edge to screen."""
        
        dlMax = 0
        dlMin = 10
        for i in range(len(self.data)):
            j = 1 
            while j<len(self.data[i]):
                tmpFloat=np.linalg.norm(self.data[i][j] - self.data[i][j-1]) 
                #print(tmpFloat)
                if tmpFloat>dlMax:
                    dlMax = tmpFloat
                if tmpFloat<dlMin:
                    dlMin = tmpFloat
                j+=1
            if self.configType=='closed':
                tmpFloat=np.linalg.norm(self.data[i][0] - self.data[i][-1]) 
                #print(tmpFloat)
                if tmpFloat>dlMax:
                    dlMax = tmpFloat
                if tmpFloat<dlMin:
                    dlMin = tmpFloat
        print(dlMin,'< dl <', dlMax)
        return None
    
    def recenter_the_curve(self):
        """Curve is translated so that the centre of mass is at the origin"""
        
        avgPos = np.zeros(3)
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                avgPos+=self.data[i][j]
        avgPos = avgPos/float(countNumberOf(self.data))

        for i in range(len(self.data)):
            strand = []
            for j in range(len(self.data[i])):
                strand.append(self.data[i][j] - avgPos)
            self.data[i] = strand
            
        return None

    def rescale(self, scalingFactor):
        """The data points are scalar multiplied such that the edgeLength equals value."""

        for i in range(len(self.data)):
            strand = []
            for j in range(len(self.data[i])):
                strand.append(self.data[i][j]*scalingFactor)
            self.data[i] = strand

        self.length = self.compute_length()
        return None

    def evaluate_arc_pairs_to_be_checked(self, skippedInteger):
        """appends an integer pair ((i,j), (l,k)) to the list arcpairs if the corresponding arcs have an arc length separation of at least pi."""

        tmpList = []
        if self.configType=='open':
             for i in range(len(self.data)):
                #print(i, skippedinteger)
                for j in range(len(self.data[i])):
                    indiciesToSkip = [(i,j)]
                    s = 1
                    while s < min(skippedInteger, len(self.data[i]) - j):
                        indiciesToSkip.append((i,j+s))
                        s+=1
                    s = 1
                    while s < min(skippedInteger, j+1):
                        indiciesToSkip.insert(0, (i,(j-s)))
                        s+=1
                    #print((i, j), indiciestoskip)

                    for l in range(i, len(self.data)):#interested only in unordered pairs
                        if l==i:
                            stIndex = j+1
                        else:
                            stIndex = 0
                        for k in range(stIndex, len(self.data[l])):
                            if (l,k) in indiciesToSkip:
                                continue
                            tmpList.append(((i,j), (l,k)))
        else:
             for i in range(len(self.data)):
                for j in range(len(self.data[i])):
                    indicesToSkip = [(i,j)]
                    s = 1
                    while s < skippedInteger:
                        indicesToSkip.append((i,(j+s)%len(self.data[i])))
                        s+=1
                    s = 1
                    while s < skippedInteger:
                        indicesToSkip.insert(0, (i,(j-s) + len(self.data[i])*(s>j)))
                        s+=1
                    #print((i, j), indiciestoskip)

                    for l in range(i, len(self.data)):#interested only in unordered pairs
                        if l==i:
                            stIndex = j+1
                        else:
                            stIndex = 0
                        for k in range(stIndex, len(self.data[l])):
                            if (l,k) in indicesToSkip:
                                continue
                            tmpList.append(((i,j), (l,k)))
        return tmpList

    def compute_deltaStar(self, radius, edgeLength):
        return np.arcsin(0.5*edgeLength/m.sqrt(radius**2 + 0.25*edgeLength**2))

    def compute_radius_R(self, radius, edgeLength):
        return m.sqrt(radius**2 + 0.25*edgeLength**2)
        
    def check_new_positions_do_not_cause_overlaps(self, radius, newPositions, indices):
        """Return True if the polygonal curve defined by self with the newPositions given at index locations indicies, spheres separated by at least a skippedInteger are at least 2radius apart."""

        for ((i,j), (l,k)) in self.arcPairs:
            if (i,j) in indices[1:-1:]:
                if (l,k) in indices[1:-1:]:#both arc pairs belong to the rotated group
                    continue
                #check distance
                p = newPositions[indices.index((i,j))-1] #indices[1:-1:].index((i,j)) = indices.index(i,j) - 1
                q = self.data[l][k]
                if np.linalg.norm(p - q)<2*radius:
                    return 0
            elif (i,j) in [indices[0], indices[-1]]:#first pair belongs to a join arc, second pair to any arc
                p = self.data[i][j]
                if (l,k) in indices[1:-1:]:
                    q = newPositions[indices.index((l,k))-1]
                else:
                    q = self.data[l][k]
                if np.linalg.norm(p - q)<2*radius:
                    return 0
            elif (l,k) in indices[1:-1:]:#first pair is unmoved, second pair rotated
                p = np.array(self.data[i][j])
                q = np.array(newPositions[indices.index((l,k))-1])
                if np.linalg.norm(p - q)<2*radius:
                    return 0
            elif (l,k) in [indices[0], indices[1]]:#first pair is unmoved, second pair join
                p = np.array(self.data[i][j])
                q = np.array(self.data[l][k])
                if np.linalg.norm(p - q)<2*radius:
                    return 0
        return 1

    def generate_index_intervals_to_be_rotated(self, Nmax):
        """ generates a list of index intervals, this list is shuffled and an interval list is drawn at random. if the curve is open intervals containing the end vertices to be extended with a dummy vertex are included"""   
        
        if self.configType=='open':
            indexList = []
            for i in range(len(self.data)):
                for j in range(len(self.data[i])):
                    for n in range(1, int(Nmax/2)+1):
                        tmpList = []
                        #print((i, len(self.data[i])-1))
                        for k in range(n+1):
                            if j + k < len(self.data[i]):
                                tmpList.append((i, j + k))
                            else:
                                tmpList.append((i,'x', n - (k-1)))
                                break
                        for k in range(1, n+1):
                            if j - k > -1:
                                tmpList.insert(0, (i, j - k))
                            else:
                                tmpList.insert(0, (i, 'x', n - (k-1)))
                                break
                        indexList.append(tmpList)
        else:
            indexList = []
            for i in range(len(self.data)):
                for j in range(len(self.data[i])):
                    for n in range(3, Nmax + 1):
                        indexList.append([(i, (j + k)%len(self.data[i])) for k in range(n)])
        return indexList

    def move(self, radius=1.0, epsilon=1.0, dMin = 0.0008, dMax=0.015):
        """a new position of the curve is evaluated close to the original position such that the tube surrounding the curve of radius radius is embedded

        arguments:

        keyword arguments:
        radius float  dummy variable hack to make the call the same as for the biarc model
        epsilon float dummy variable hack to make the call the same as for the biarc model
        dMax float maximum distance points belonging to the curve are moved
        Nmax integer maximum additional number of arcs moved

        return:
        list of the same stucture as self.data, integer equally the number of attempts, float distance moved
        """ 
        indexNextJ = lambda ind : (ind[1] + 1)%len(self.data[ind[0]])
        indexPrevJ = lambda ind : (ind[1] - 1) + len(self.data[ind[0]])*(ind[1]==0)
       # indexnext = lambda ind : (ind[0], (ind[1] + 1)%len(self.data[ind[0]]))

        dAngle = lambda  x, y: 2*np.arcsin(np.clip(0.5*y/x, 0, 1))
        angleChoice = lambda x : [x, 2*m.pi - x]
        random.shuffle(self.index_intervals_to_be_rotated)
        numberOfTries = 0
        while numberOfTries < len(self.index_intervals_to_be_rotated):
            ind = self.index_intervals_to_be_rotated[numberOfTries]
            if self.configType=='open':
                if ind[0][1] == 'x':
                    x = ind[0][2]
                    #add a dummy position for the start vertex
                    t = self.data[ind[0][0]][0] - self.data[ind[0][0]][1]
                    e = np.linalg.norm(t)
                    t = t/e
                    randomUnitVectorInCone = simp_func.returnRandomUnitVectorInCone(t, x*self.deltaStar)
                    dummyPos = self.data[ind[0][0]][0] + x*e*randomUnitVectorInCone
                    
                    startPoint = dummyPos
                    rotationAxis = self.data[ind[-1][0]][ind[-1][1]] - dummyPos
                    rotationAxis = rotationAxis/np.linalg.norm(rotationAxis)

                elif ind[-1][1] == 'x':
                    x = ind[-1][2]
                    #add a dummy position for the end vertex
                    t = self.data[ind[-1][0]][-1] - self.data[ind[-1][0]][-2]
                    e = np.linalg.norm(t)
                    t = t/e
                    randomUnitVectorInCone = simp_func.returnRandomUnitVectorInCone(t, x*self.deltaStar)
                    dummyPos = self.data[ind[-1][0]][-1] + x*e*randomUnitVectorInCone
                    
                    startPoint = self.data[ind[0][0]][ind[0][1]]
                    rotationAxis = dummyPos - self.data[ind[0][0]][ind[0][1]]
                    rotationAxis = rotationAxis/np.linalg.norm(rotationAxis)

                else:
                    startPoint = self.data[ind[0][0]][ind[0][1]]
                    rotationAxis = self.data[ind[-1][0]][ind[-1][1]] - self.data[ind[0][0]][ind[0][1]]
                    rotationAxis = rotationAxis/np.linalg.norm(rotationAxis)
            else:
                    startPoint = self.data[ind[0][0]][ind[0][1]]
                    rotationAxis = self.data[ind[-1][0]][ind[-1][1]] - self.data[ind[0][0]][ind[0][1]]
                    rotationAxis = rotationAxis/np.linalg.norm(rotationAxis)

            posVector= [self.data[ind[1][0]][ind[1][1]] - startPoint]
            n = 2
            while n < len(ind)-1:
                posVector.append(posVector[-1] + self.data[ind[n][0]][ind[n][1]] - self.data[ind[n-1][0]][ind[n-1][1]])
                n+=1
        
            rCircle = map(lambda x: m.sqrt(np.clip(np.dot(x, x) - np.dot(x, rotationAxis)**2, 0, 100)), posVector)             
            rc = max(rCircle)
            if rc <2*dMin:
                numberOfTries+=1
                continue
#            #d = dMax
#            d = np.random.uniform(dMin, dMax)
#            #d = dMax/1000 + np.random.exponential(scale=dMax)
#            angle = angleChoice(dAngle(rc, d))
#            random.shuffle(angle)
#
#            while len(angle)>0:
#                a = angle.pop(0)
#
#                newPos = []
#                for n in range(len(posVector)):
#                    newPos.append(startPoint + simp_func.rotate(posVector[n], a, rotationAxis))
#
#                #check curvature constraint
#                if self.configType=='open':
#                    if (ind[0][1] not in  ['x', 0]):#start vertex is not an interior vertex if ind[0][1] = 'x' or 0 and is never n-1
#                        if simp_func.returnTurningAngleForControlTriangle(self.data[ind[0][0]][indexPrevJ(ind[0])], self.data[ind[0][0]][ind[0][1]], newPos[0]) > 2*self.deltaStar:
#                            continue
#                    if (ind[-1][1] not in ['x', (len(self.data[ind[-1][0]])-1)]):#last vertex is the beginning vertex of the chain
#                        if simp_func.returnTurningAngleForControlTriangle(newPos[-1], self.data[ind[-1][0]][ind[-1][1]], self.data[ind[-1][0]][indexNextJ(ind[-1])]) > 2*self.deltaStar:
#                            continue
#                else:
#                    if simp_func.returnTurningAngleForControlTriangle(self.data[ind[0][0]][indexPrevJ(ind[0])], self.data[ind[0][0]][ind[0][1]], newPos[0]) > 2*self.deltaStar:
#                        continue
#                    if simp_func.returnTurningAngleForControlTriangle(newPos[-1], self.data[ind[-1][0]][ind[-1][1]], self.data[ind[-1][0]][indexNextJ(ind[-1])]) > 2*self.deltaStar:
#                        continue
#
#                #check the overlapping arc condition
#                if self.check_new_positions_do_not_cause_overlaps(self.R, newPos, ind):
#                    return d, ind, newPos

            d_upperBound = dMax
            while d_upperBound>1000*dMin:
                d = np.random.uniform(dMin, d_upperBound)
                angle = angleChoice(dAngle(rc, d))
                random.shuffle(angle)

                while len(angle)>0:
                    a = angle.pop(0)
                    #print(tries, d_upperBound)

                    #for each point in the index list other than the end nodes which are stationary rotate the point by angle about the normal vector
                    newPos = []
                    for n in range(len(posVector)):
                        newPos.append(startPoint + simp_func.rotate(posVector[n], a, rotationAxis))

                    #check curvature constraint
                    if self.configType=='open':
                        if (ind[0][1] not in  ['x', 0]):#start vertex is not an interior vertex if ind[0][1] = 'x' or 0 and is never n-1
                            if simp_func.returnTurningAngleForControlTriangle(self.data[ind[0][0]][indexPrevJ(ind[0])], self.data[ind[0][0]][ind[0][1]], newPos[0]) > 2*self.deltaStar:
                                continue
                        if (ind[-1][1] not in ['x', (len(self.data[ind[-1][0]])-1)]):#last vertex is the beginning vertex of the chain
                            if simp_func.returnTurningAngleForControlTriangle(newPos[-1], self.data[ind[-1][0]][ind[-1][1]], self.data[ind[-1][0]][indexNextJ(ind[-1])]) > 2*self.deltaStar:
                                continue
                    else:
                        if simp_func.returnTurningAngleForControlTriangle(self.data[ind[0][0]][indexPrevJ(ind[0])], self.data[ind[0][0]][ind[0][1]], newPos[0]) > 2*self.deltaStar:
                            continue
                        if simp_func.returnTurningAngleForControlTriangle(newPos[-1], self.data[ind[-1][0]][ind[-1][1]], self.data[ind[-1][0]][indexNextJ(ind[-1])]) > 2*self.deltaStar:
                            continue

                    #check the overlapping arc condition
                    if self.check_new_positions_do_not_cause_overlaps(self.R, newPos, ind):
                        return d, ind, newPos
                d_upperBound = 0.25*d_upperBound

            numberOfTries+=1
              
        print("All Points Are Jammed!")
        return 1

    def make_curve_polyFile(self, fileLocation, fileName):
        """Set of concatenated data points is written as a .poly file to be read with Houdini"""

        r=str(round(204.0/255.0, 3))
        g=str(round(12.0/255.0, 3))
        b=str(round(204.0/255.0, 3))

        r_1=str(round(14.0/255.0, 3))
        g_1=str(round(120.0/255.0, 3))
        b_1=str(round(204.0/255.0, 3))

        point_count = 1
        f1 = open(str(fileLocation)+str(fileName)+".poly", "w")
        f1.write('POINTS''\n')
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                if point_count%2==0:
                    f1.write(str(point_count)+': '+str(dec.Decimal(str(round(self.data[i][j][0],5))))+' '+str(dec.Decimal(str(round(self.data[i][j][1],5))))+' '+str(dec.Decimal(str(round(self.data[i][j][2],5))))+' c('+r+','+g+','+b+',1.0)'+'\n')
                else:
                    f1.write(str(point_count)+': '+str(dec.Decimal(str(round(self.data[i][j][0],5))))+' '+str(dec.Decimal(str(round(self.data[i][j][1],5))))+' '+str(dec.Decimal(str(round(self.data[i][j][2],5))))+' c('+r_1+','+g_1+','+b_1+',1.0)'+'\n')
                point_count = point_count + 1
    
        point_count = 1
        poly_count = 1
        f1.write('POLYS')
        for i in range(len(self.data)):
            f1.write('\n'+str(poly_count)+': ')
            for j in range(len(self.data[i])):
                f1.write(str(point_count)+' ')
                point_count+=1
            if self.configType=="closed":#visually close the loop
                f1.write(str(point_count - len(self.data[i]))+'')#bit of a hack
            poly_count+=1

        f1.write('\nEND')       
        f1.close()

    def save_data(self, fileLocation, fileName):
        """Write the position of the verticies for each triangle to file."""
        f1 = open(str(fileLocation)+str(fileName)+'.txt', 'w')
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                pt = self.data[i][j]
                f1.write(str(dec.Decimal(str(round(pt[0],5))))+' '+str(dec.Decimal(str(round(pt[1],5))))+' '+str(dec.Decimal(str(round(pt[2],5))))+'\n')
        f1.write('END\n')
        f1.close()


    def generates_list_arc_radii(self):
        """Creates list floats. Each float gives the radius of the arc defined by the bezier triangle at that index."""
        result = []
        if self.configType=='open':
            for i in range(len(self.data)):
                strand = []
                for j in range(1, len(self.data[i])-1):
                    p0 = self.data[i][j-1]
                    p = self.data[i][j]
                    p1 = self.data[i][j+1]
                    strand.append(simp_func.returnCircleRadiusForControlTriangle(p0, p, p1))
                result.append(strand)
        else:
            for i in range(len(self.data)):
                strand = []
                for j in range(len(self.data[i])):
                    p0 = self.data[i][j-1]
                    p = self.data[i][j]
                    p1 = self.data[i][(j+1)%len(self.data[i])]
                    strand.append(simp_func.returnCircleRadiusForControlTriangle(p0, p, p1))
                result.append(strand)
        return result

    def generate_closest_distances(self, withInfo = False):
        """Creates list of floats. Each float is the closest to another point on the polygonal curve separtaed by at least a skipped integer."""

        if withInfo:
            print('\n')
        result = []
        tmpList = []
        for i in range(len(self.data)):
            #skippedInteger = int(0.5*m.pi/np.arcsin(0.5*self.edgeLength/self.R))
            #print(skippedInteger)
            for j in range(len(self.data[i])):
                d = 1000.0
                ind = (0,0)
                for l in range(i, len(self.data)):
                    stInd = j if l==i else 0
                    for k in range(stInd, len(self.data[l])):
                        if self.configType=='open':
                            if l==i and (k - j < self.skippedInteger):
                                #print('for', (i,j), 'skipping ', (l,k))
                                continue
                        else:
                            if l==i and ((k - j < self.skippedInteger) or (len(self.data[i]) - k < self.skippedInteger - j)):
                                #print('for', (i,j), 'skipping ', (l,k))
                                continue
                        dtmp = np.linalg.norm(self.data[i][j] - self.data[l][k])
                        if dtmp<d:
                            d = dtmp
                            ind = (l,k)
                    if d<1000.0:
                        tmpList.append([(i,j), ind, d])
                result.append(d)
        if withInfo:
            tmpList.sort(key=lambda x:x[2])
            for data in tmpList:
                print(data)
            print('\n')
        return result
    
    def check_reach(self, withInfo=False):
        """Function returns the minimum arc radius and half the minimum distance of the closest arc to each arc along the curve."""
        #return (returnMinimumInNestedListOfFloats(self.generates_list_arc_radii()))
        return (returnMinimumInNestedListOfFloats(self.generates_list_arc_radii()), min(self.generate_closest_distances(withInfo=withInfo)))

#    def update_minRad_closestDistance(self):
#        """"Function computes the minimum arc radius and the minimum self distance and sets this as varaibles minRad and selfDist. """
#        (minRad, selfDist) = self.check_reach()
#        self.minRad = minRad
#        self.selfDist = selfDist
#        return None
    
    @classmethod
    def set_radii(cls, overlapRatio, rTube, edgeLength):
        """"Sets the input sphere radius used to evaluate the energy, since you are now swopping curve data this needs to be the same across all instances of the class"""
        cls.R = round(m.sqrt(rTube**2 + 0.25*edgeLength**2), 5)
        cls.r_s = round(m.sqrt(rTube**2 + 0.25*edgeLength**2)*overlapRatio, 5)  
        cls.input_R = round(m.sqrt(rTube**2 + 0.25*edgeLength**2)*(1 + overlapRatio), 5)
        return None

    def evaluate_embedded_measures(self):
        """ Function computes the embedded measures and saves them as variables of the curve"""
        
        if self.configType=='open':
            endCaps=True
        else:
            endCaps=False
        if hasattr(ThreadedBeads, 'input_R'): 
            measures = returnEmbeddedMeasures(self.data, ThreadedBeads.input_R, endCaps=endCaps)

            self.V_0 = measures[0]
            self.A_0 = measures[1]
            self.C_0 = measures[2]
            self.X_0 = measures[3]
            return None
        else:
            print('You need to call ThreadedBeads.set_radii(overlapRatio, rTube, edgeLength) to set the radius of the spheres used to compute the measures')
            sys.exit()
        

    def evaluate_measures(self):
        """Function computes the measures of the curveData point list with balls of radius input_R """
    
        if hasattr(ThreadedBeads, 'input_R'): 
            pointList = [pt for subList in self.data for pt in subList]
            makeFilFile(pointList, 'input', ThreadedBeads.input_R)
            #makeObjFile(pointList, 'input_'+fileName)

            #call morph with the input files
            #proc = subprocess.Popen('export LD_LIBRARY_PATH=/usr/site-local/lib/gcc/x86_64-unknown-linux-gnu/4.1.1:$LD_LIBRARY_PATH \n ./morphLf ./inputFiles/input.fil', shell=True)
            proc = subprocess.Popen('./morph_local ./inputFiles/input.fil', shell=True)
            proc.wait()
            
            #read line
            measures = []
            with open("./data.txt", "r") as f1:
                line = f1.readline().split('  ')
                measures=[float(line[l]) for l in range(4)]
                f1.close()

            self.V = measures[0]
            self.A = measures[1]
            self.C = measures[2]
            self.X = measures[3]    

            #delete file data.txt and input file
            os.remove("data.txt")
            os.remove("./inputFiles/input.fil")
            return None
        else:
            print('You need to call ThreadedBeads.set_radii(overlapRatio, rTube, edgeLength) to set the radius of the spheres used to compute the measures')
            sys.exit()
        
    @classmethod
    def set_coefficients(cls, alpha=0, eta=0, real=False):
        """Sets the coefficients used to evaluate the energy."""

        if hasattr(cls, 'r_s'):
            if eta>0:
                f1 = eta*(1 + eta + eta**2 - pow(eta, 3))/pow(1 - eta, 3)
                f2 = eta*((1 + 2*eta + 8*eta**2 - 5*pow(eta, 3))/(3*pow(1 - eta, 3)) + m.log(1 - eta)/(3*eta))
                f3 = eta*((4 - 10*eta + 20*eta**2 - 8*pow(eta, 3))/(3*pow(1- eta, 3)) + 4*m.log(1 - eta)/(3*eta))
                f4 = eta*((-4 + 11*eta - 13*eta**2 + 4*pow(eta, 3))/(3*pow(1 - eta, 3)) - 4*m.log(1 - eta)/(3*eta))
                cls.prefactors = [round(1.0/pow(cls.r_s, 3), 5), round((-f2/f1)/pow(cls.r_s, 2), 5), round((f3/f1)/cls.r_s, 5), round(f4/f1, 5)]
                #cls.prefactors = (np.array([f1/pow(cls.r_s, 3),-f2/pow(cls.r_s, 2), f3/cls.r_s, f4])*(3/(4*m.pi))).round(5).tolist()
            elif alpha>0:
                cls.prefactors = [round(1.0/pow(cls.r_s, 3), 5), round(-alpha/pow(cls.r_s, 2), 5), 0, 0]
            elif real:#here you shall put the prefactors as they compare to HAN07
                f1 = eta*(1 + eta + eta**2 - pow(eta, 3))/pow(1 - eta, 3)
                f2 = eta*((1 + 2*eta + 8*eta**2 - 5*pow(eta, 3))/(3*pow(1 - eta, 3)) + m.log(1 - eta)/(3*eta))
                f3 = eta*((4 - 10*eta + 20*eta**2 - 8*pow(eta, 3))/(3*pow(1- eta, 3)) + 4*m.log(1 - eta)/(3*eta))
                f4 = eta*((-4 + 11*eta - 13*eta**2 + 4*pow(eta, 3))/(3*pow(1 - eta, 3)) - 4*m.log(1 - eta)/(3*eta))
                cls.prefactors = [round(f1/pow(cls.r_s, 3), 5), round(-f2/pow(cls.r_s, 2), 5), round(f3/cls.r_s, 5), round(f4, 5)]
                print("WARNING still missing a factor 3/4pi or something")
            else:
                cls.prefactors = [round(1.0/pow(cls.r_s, 3), 5),0,0,0]

            return None
        else:
            print('You need to call ThreadedBeads.set_radii(overlapRatio, rTube, edgeLength) to set the radius of the spheres used to compute the measures')
            sys.exit()

    def evaluate_energy_difference(self, neighbouringCurveCoordinates):
        """Function computes the measures of the neighbouringCurveCoordinates with the radius input_R as associated to the class variable. 
           The energy of self is compared to the energy of a class instance with vertex positions given by the neighbouringCurveCoordinates."""

        if hasattr(ThreadedBeads, 'input_R'): 
            pointList = [pt for subList in neighbouringCurveCoordinates for pt in subList]
            makeFilFile(pointList, 'input', ThreadedBeads.input_R)
            #makeObjFile(pointList, 'input_'+fileName)

            #call morph with the input files
            #proc = subprocess.Popen('export LD_LIBRARY_PATH=/usr/site-local/lib/gcc/x86_64-unknown-linux-gnu/4.1.1:$LD_LIBRARY_PATH \n ./morphLf ./inputFiles/input.fil', shell=True)
            proc = subprocess.Popen('./morph_local ./inputFiles/input.fil', shell=True)
            proc.wait()
            
            #read line
            measures = []
            with open("./data.txt", "r") as f1:
                line = f1.readline().split('  ')
                measures=[float(line[l]) for l in range(4)]
                f1.close()

            #delete file data.txt and input file
            os.remove("data.txt")
            os.remove("./inputFiles/input.fil")
            
            
            #deltaE = (tmpE[0] - tmpE[1]) - (E[0] - E[1])
            #deltaE = tmpE[0] - E[0] #the above is pointless because the embedded energies are exactly equal 
            deltaE = ThreadedBeads.prefactors[0]*(measures[0] - self.V) + ThreadedBeads.prefactors[0]*(measures[0] - self.V) + ThreadedBeads.prefactors[1]*(measures[1] - self.A) + ThreadedBeads.prefactors[2]*(measures[2] - self.C) + ThreadedBeads.prefactors[3]*(measures[3] - self.X)
            return (deltaE, measures)
        else:
            print('You need to call ThreadedBeads.set_radii(overlapRatio, rTube, edgeLength) to set the radius of the spheres used to compute the measures')
            sys.exit()

                
            # E = measures[0]/pow(solventSphereRadius, 3) + prefactors[1]*measures[1]/pow(solventSphereRadius, 2) + prefactors[2]*measures[2]/(solventSphereRadius) + prefactors[3]*measures[3]
            # E_0 = embeddedMeasures[0]/pow(solventSphereRadius, 3) + prefactors[1]*embeddedMeasures[1]/pow(solventSphereRadius, 2) + prefactors[2]*embeddedMeasures[2]/(solventSphereRadius) + prefactors[3]*embeddedMeasures[3]

    def evaluate_energy(self):
        """ Computes the energy from the measures associated to the class object """
        
        return ThreadedBeads.prefactors[0]*self.V + ThreadedBeads.prefactors[1]*self.A + ThreadedBeads.prefactors[2]*self.C + ThreadedBeads.prefactors[3]*self.X

    def evaluate_embedded_energy(self):
        """ Computes the energy from the measures associated to the class object """
        
        return ThreadedBeads.prefactors[0]*self.V_0 + ThreadedBeads.prefactors[1]*self.A_0 + ThreadedBeads.prefactors[2]*self.C_0 + ThreadedBeads.prefactors[3]*self.X_0

    def evaluate_normalised_energy(self):
        """ Computes the energy from the measures associated to the class object """
        
        return (ThreadedBeads.prefactors[0]*(self.V - self.V_0) + ThreadedBeads.prefactors[1]*(self.A - self.A_0) + ThreadedBeads.prefactors[2]*(self.C - self.C_0) + ThreadedBeads.prefactors[3]*(self.X - self.X_0))/self.length

    def __copy__(self):
        """Returns a copy of the object and a number of variables.""" 

        newone = type(self)()
        #newone.__dict__.update(self.__dict__) -> makes a copy of all the instance variables which may be unnecessary
        #for item in ['ropelength', 'configType', 'arcPairs', 'edgeWeights', 'sphereCount']: -> this also kind of overkill -> to make the pointList only need sphereCount, configType and poss. ropelength
        for item in ['configType']:
            newone.__dict__[item] = self.__dict__[item]
        return newone

    def update_new_geometry(self, geometry):
        """updates self with certain attributes of geometry."""

        for item in ['edgeLength', 'length', 'deltaStar','R','arcPairs']:
            self.__dict__[item] = geometry.__dict__[item]
