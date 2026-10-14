"""
Restructured geometryClass.py -- built from geometryClass_redo_pilot.py with every method
body copied from the original (geometryClass_original.py, in this folder).

Deviations from a pure copy (all forced by the new structure):
    - TubularGeometry is gone; its methods sit on ThickenedEquilateralPolygon (TEP) or
      OpenChain/ClosedChain as in the pilot, self.curve_object.X -> self.X.
    - TubularGeometry.__init__ -> TEP.set_energy_parameters (classmethod, once per run).
      set_radii / set_coefficients became classmethods because they only touch class-level
      state now (edgeLength, r_s_star, f1_f2_f3_f4).
    - rTube is the module-level constant.
    - OpenChain/ClosedChain.__init__(curveData, edgeLength=0): the fileName branch of the
      original ThreadedBeads.__init__ is not carried over (signature as in the pilot).
    - OpenChain/ClosedChain.__init__ checks edgeLength against the class-level one
      (relative tolerance 0.01, sys.exit) and computes all measures at construction.
    - umbrella_sampling is stored but not yet used (NotImplementedError if set).
    - __copy__ is the original, still broken, body (see PROMPT_MERGED.md).

--> this is the verion from test_6 major updates are restructuring the file as outlined above and introducing the method move to the TEP class
--> move then got an update, rotation is done via numpy in built functions, the stack of index_intervals_to_be_rotated is no longer shuffled but an interval is drawn at random. The curvature constraint is checked via a slightly conciser version.

Planned updates not yet completed: I would like to change the way the move is applied for the open curve case as I think the 'x' labelling of the end points is messy. This could mean that the index_intervals_to_be_rotated is redesigned -- this fits into incorperating the lasso into this class.

Then I would like to have some data on the best choice of dMin/dMax/d_interval_subdivision

Then tidy up--> update is complete and could be merged into main branch.

"""

import math as m
import numpy as np
import random
import decimal as dec
import sys
#import subprocess
#import os

import morphometry as mm
import self_distance_c  as self_dist

# ---- functions this file used from simple_functions.py, pasted here verbatim (simple_functions.py is no longer imported) ----
# rotate() is not here any more: move() rotates all the vertices at once with Rodrigues' formula (see the comment there).

def returnRandomPointOnCircle(centrePoint, normalVector, radius):
    """
    Function: returnRandomPointOnCircle
    --------------------------------------------------------------------------------------------------------------------------------------------------------
    function returns a point on a circle with given centre radius and normal vector
    centrePoint three tuple
    normal vector np array
    radius float
    return three tuple
    """
    #check the normalVector is a unit vector
    n = normalVector/np.linalg.norm(normalVector)
    direction1 = 2*np.random.rand(3) - np.ones(3)
    direction = direction1 - np.dot(direction1, n)*n
    vector =  direction/np.linalg.norm(direction) if np.linalg.norm(direction)>0 else np.array(direction1)/np.linalg.norm(direction1)
    newPoint = np.array(centrePoint) + radius*vector

    return newPoint

def returnRandomUnitVectorInCone(normalVector, openingAngle):
    """
    Function: returnRandomUnitVectorInCone
    -------------------------------------------------------------------------------
    function returns a random vector in the cone with angle openingAngle from the vector normalVector
    return np.array
    """
    openingAngle = min(openingAngle, 0.5*m.pi)
    #helpVector = returnRandomUnitVector()
    #h = random.uniform(0, m.sin(openingAngle))
    #return  h*(helpVector - np.dot(helpVector, normalVector)*normalVector)/m.sqrt(1 - np.dot(helpVector, normalVector)**2) + normalVector*m.sqrt(1 - h**2)
    #same as above, this over samples points in the direction of the normal vector. 
    #in the below method the disk of radius h=sin(theta) is sample uniformly. theta must be sampled proportional to sin(theta)
    
    s = random.uniform(-1, -m.cos(openingAngle))
    h = m.sin(m.acos(-s))
    return returnRandomPointOnCircle(m.sqrt(1 - h**2)*normalVector, normalVector, h)

# (returnTurningAngleForControlTriangle is kept for reference; check_curvature_constraint now uses turning_angle_exceeds below)
def returnTurningAngleForControlTriangle(p0, p, p1):
    """
    Function: returnCircleRadiusForControlTriangle
    --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    function returns the radii of the circular arc defined by the control triangle (p0, p, p1)
    p0,p, p1 three tuples
    return float
    """
    t0 = p - p0
    t1 = p1 - p
    dl0 = np.linalg.norm(t0)
    dl1 = np.linalg.norm(t1)
    t0 = t0/dl0
    t1 = t1/dl1
    t0Dott1 = np.clip(np.dot(t0, t1), -0.999999999999, 0.999998)
	
    return np.arccos(t0Dott1)

def turning_angle_exceeds(p0, p, p1, cosLimit):
    """
    True if the turning angle at p of the control triangle (p0, p, p1) is larger than the limit, cosLimit = cos(limit).
    The same test as returnTurningAngleForControlTriangle(p0, p, p1) > limit, without the arccos, the clip and the normalisations:
    the angle between the tangents t0 = p - p0 and t1 = p1 - p is larger than the limit exactly when cos(angle) = t0.t1/(|t0||t1|) < cos(limit)
    (arccos is decreasing). The old clip of the cosine to [-0.999999999999, 0.999998] does not matter for a limit of 2*deltaStar (about 0.24 rad).
    """
    t0 = p - p0
    t1 = p1 - p
    return np.dot(t0, t1) < cosLimit*m.sqrt(np.dot(t0, t0)*np.dot(t1, t1))

def returnCircleRadiusForControlTriangle(p0, p, p1):
    """
    Function: returnCircleRadiusForControlTriangle
    --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    function returns the radii of the circular arc defined by the control triangle (p0, p, p1)
    p0,p, p1 three tuples
    return float
    """
    t0 = p - p0
    t1 = p1 - p
    dl0 = np.linalg.norm(t0)
    dl1 = np.linalg.norm(t1)
    t0 = t0/dl0
    t1 = t1/dl1
    t0Dott1 = np.clip(np.dot(t0, t1), -0.999999999999, 0.999998)
	
    #return (0.25*(dl0 + dl1)*m.sqrt(1 + t0Dott1))/m.sqrt(1 - t0Dott1)
    return 0.5*(dl0 + dl1)/m.sqrt(2*(1 - t0Dott1))

# ---- end of the pasted functions ----

rTube = 1.0


# ---- module-level helpers from the original geometryClass.py ----

# -- DROPPED: randomOrder, makeCopyOf, returnLengthOfPointList,
#    returnAverageEdgeLengthOfPointList, makeFilFile.

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

# pressure, sigma, kappa, kappaBar (reference for printing the prefactors) but are not used to define the coefficients
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


# -- MOVED OUT of the class hierarchy: zero configType dependence and no self needed.
def save_data(data, fileLocation, fileName):
    """Write the position of the verticies for each triangle to file."""
    f1 = open(str(fileLocation)+str(fileName)+'.txt', 'w')
    for i in range(len(data)):
        for j in range(len(data[i])):
            pt = data[i][j]
            f1.write(str(dec.Decimal(str(round(pt[0],5))))+' '+str(dec.Decimal(str(round(pt[1],5))))+' '+str(dec.Decimal(str(round(pt[2],5))))+'\n')
    f1.write('END\n')
    f1.close()


class ThickenedEquilateralPolygon:
    """Methods that are either zero-branching, or depend on topology only
    abstractly (through other methods that themselves are split). Never instantiated;
    the energy parameters are class-level state set once per run by set_energy_parameters.

    Coding notes:
    @classmethod marks a method that belongs to the class, not to one object. 
    It receives the class itself as its first argument, called cls instead of self.

    An ordinary method works on one object:
    geometry.move()   # i.e. geometry is the object set via geometry = geoClass.OpenChain(curveData, ...)
    A class method works on the class, so you can call it before any curve exists:
    ThickenedEquilateralPolygon.set_energy_parameters(edgeLength, rs, eta)
    """

    umbrella_sampling = None #incase we want to add in the spring constant thing... 

    # -- REPLACES TubularGeometry.__init__. Class-level, called once per run.
    #    umbrella_sampling = spring constant, None = off.
    @classmethod
    def set_energy_parameters(cls, edgeLength, overlapRatio, eta, alpha=0, umbrella_sampling=None):

        #compute the functions used to define the linear combination
        if eta>0:
            f1 = round(eta*(1 + eta + eta**2 - pow(eta, 3))/pow(1 - eta, 3), 5)
            f2 = round(eta*((1 + 2*eta + 8*eta**2 - 5*pow(eta, 3))/(3*pow(1 - eta, 3)) + m.log(1 - eta)/(3*eta)), 5)
            f3 = round(eta*((4 - 10*eta + 20*eta**2 - 8*pow(eta, 3))/(3*pow(1- eta, 3)) + 4*m.log(1 - eta)/(3*eta)),5)
            f4 = round(eta*((-4 + 11*eta - 13*eta**2 + 4*pow(eta, 3))/(3*pow(1 - eta, 3)) - 4*m.log(1 - eta)/(3*eta)), 5)
        elif alpha>0:
            f1 = 1.0
            f2 = alpha
            f3 = 0
            f4 = 0
        else:
            f1 = 1.0
            f2 = 0
            f3 = 0
            f4 = 0

        cls.f1_f2_f3_f4 = [f1, -1*f2, f3, f4]

        #set scale invariant tube dimensions, the class-level edgeLength fixes the radius
        cls.r_s_star = overlapRatio
        cls.edgeLength = edgeLength
        cls.umbrella_sampling = umbrella_sampling #still a new idea: memory jog "spring_connection_matrix" matrix of 0, 1's R01

        #set tube dimensions depending on the density of vertices on the curve
        cls.set_radii()

        #compute coefficients used to define energy as linear combination of measures
        cls.set_coefficients()

    # -- classmethod (the pilot had self): only touches class-level state.
    @classmethod
    def set_radii(cls):
        """"Sets the input sphere radius used to evaluate the energy which depends on the edgeLength spacing of the curve"""

        cls.R = round(m.sqrt(rTube + 0.25*cls.edgeLength**2), 5)
        cls.r_s = round(m.sqrt(rTube + 0.25*cls.edgeLength**2)*cls.r_s_star, 5)
        cls.input_R = round(m.sqrt(rTube + 0.25*cls.edgeLength**2)*(1 + cls.r_s_star), 5)

        return None

    @classmethod
    def set_coefficients(cls):
        """Sets the coefficients used to define the specific linear combination of measures defining the energy"""

        if not hasattr(cls, 'r_s'):
            cls.set_radii()

        f1, f2, f3, f4 = cls.f1_f2_f3_f4 #f2 is already set as negative

        cls.coefficients = [round(f1/pow(cls.r_s, 3), 5), round(f2/pow(cls.r_s, 2), 5), round(f3/cls.r_s, 5), round(f4, 5)]
        return None

    # -- DROPPED: set_uniform_tube_and_energy_specs_by_overriding_edgeLength.

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

        self.length = self.compute_length() #compute_length() to be updated and pushed into TEP
        return None

    # -- PROMOTED from OpenChain/ClosedChain: one move() for every topology. What differs is in prev_point/next_point and check_curvature_constraint,
    #    which each class defines for itself.
    def move(self, dMin = 0.0008, dMax=0.015):
        """a new position of the curve is evaluated close to the original position such that the reach of the curve is lower bounded by reach variable
        arguments:

        keyword arguments:
        dMax float maximum distance points belonging to the curve are moved
        Nmax integer maximum additional number of arcs moved

        return:
        list of the same stucture as self.data, integer equally the number of attempts, float distance moved
        """
       # indexnext = lambda ind : (ind[0], (ind[1] + 1)%len(self.data[ind[0]]))

        dAngle = lambda  x, y: 2*np.arcsin(np.clip(0.5*y/x, 0, 1))
        angleChoice = lambda x : [x, 2*m.pi - x]
        numberOfTries = 0
        while numberOfTries < len(self.index_intervals_to_be_rotated):
            # LAZY DRAW: idea here is you draw a variable in the list at random and reshuffle the list at the same time by alway putting the varible you draw to the "drawn pile" so you never draw the same variable twice
            # let A = self.index_intervals_to_be_rotated and k=0 be the numberOfTries
            # concretely draw at random _j in [0, n-1], giving A[_j] as the interval, let A = [A[_j], A[0,..., _j-1, _j+1,...n-1]]=:A_1 k+=1
            # draw _j in [1, 1] giving A[_j] as the interval and let A = [A_1[0], A_1[_j], A_1[1,...,_j-1, _j+1,...n-1]=:A_2 k+=1
            # in general draw _j in [k, n-1] giving A[_j] and let A = A_k[0,..., k-1] u A[_j] u A_[k+1, n-1]
            # all that needs to happen computationally is that A[k] and A[_j] swop positions in the list
            # Same distribution as the full shuffle, but the cost is
            # O(1) per interval tried instead of O(length of the list) per move.
            _j = random.randrange(numberOfTries, len(self.index_intervals_to_be_rotated))
            self.index_intervals_to_be_rotated[numberOfTries], self.index_intervals_to_be_rotated[_j] = self.index_intervals_to_be_rotated[_j], self.index_intervals_to_be_rotated[numberOfTries]
            ind = self.index_intervals_to_be_rotated[numberOfTries]
            # the fixed ends of the rotation: the point before the first rotated vertex and the point after the last one. At an open chain's end these are
            # dummy points x edge lengths beyond the end (random in a cone), x = how far the interval overshoots the end, stored as the marker 'x' in the interval.
            # Called once per interval: the dummy is drawn once and used for every attempt on this interval.
            x0 = ind[0][2] if ind[0][1] == 'x' else 1
            x1 = ind[-1][2] if ind[-1][1] == 'x' else 1
            v0 = self.prev_point(ind[1][0], ind[1][1], x0)
            v1 = self.next_point(ind[-2][0], ind[-2][1], x1)
            rotationAxis = v1 - v0
            rotationAxis = rotationAxis/np.linalg.norm(rotationAxis)

            # relative positions of the vertices that are rotated (the end nodes ind[0], ind[-1] stay), as a k x 3 array
            posVector = np.array([self.data[ind[n][0]][ind[n][1]] for n in range(1, len(ind)-1)]) - v0
            # distance of every rotated vertex from the rotation axis, all at once; rc is the largest
            #what the ?? is einsum:
            # let P be k x 3 array, k being the number of points in the rotated interval P[i]=(x,y,z) of v_i
            # einsum = "sum of products" the first input is the string telling which sum of products to take
            #The same rule explains the variants:
            #   - 'ij->i' keeps i and sums j, giving row sums.
            #   - 'ij->j' keeps j and sums i, giving column sums.
            #   - 'ij->' keeps nothing, so it sums everything into one number.
            #   - 'ij->ij' keeps both and sums nothing. It just returns the array.
            #   - 'ij,ij->i' product of all P[i,j]**2 sum over j  i.e. so einsum takes the norm of each vector in P at once
            #  P @ w = (P w)_i = Σ_j P_ij w_j. normal matrix rotation
            rc = float(np.max(np.sqrt(np.clip(np.einsum('ij,ij->i', posVector, posVector) - (posVector @ rotationAxis)**2, 0, 100))))
            if rc <2*dMin:
                numberOfTries+=1
                continue

            d_upperBound = dMax
            while d_upperBound>10*dMin:
                d = np.random.uniform(dMin, d_upperBound)
                angle = angleChoice(dAngle(rc, d))
                random.shuffle(angle)

                while len(angle)>0:
                    a = angle.pop(0)
                    #print(tries, d_upperBound)

                    #for each point in the index list other than the end nodes which are stationary rotate the point by angle about the normal vector
                    # Rodrigues' rotation formula, for all k vertices at once: rotating v by the angle a about the unit vector w gives
                    #     v_rot = v cos(a) + (w x v) sin(a) + w (w.v) (1 - cos(a))       (the same as the 3x3 rotation matrix rotate() built for every vertex)
                    # posVector is k x 3, rotationAxis is w; the new position of each vertex is v0 + v_rot
                    newPos = v0 + posVector*np.cos(a) + np.cross(rotationAxis, posVector)*np.sin(a) + np.outer(posVector @ rotationAxis, rotationAxis)*(1 - np.cos(a))
                    newPos = list(newPos)      # a list of the k new positions, the code below takes them from the front

                    #check curvature constraint (open and closed chains differ here, see check_curvature_constraint)
                    if not self.check_curvature_constraint(ind, newPos): #only uses the end positions of the chain. 
                        continue

                    #check the overlapping arc condition if self.check_new_positions_do_not_cause_overlaps(self.upper_bound_closest_self_distance, newPos, ind):
                    if self_dist.check_new_positions_do_not_cause_overlaps(self.data, self.configType, self.skippedInteger, newPos, ind, self.upper_bound_closest_self_distance):

                        #return d, ind, newPos ---> Tidy: #update new positions to generate a neighbouring configuration
                        # a copy of the LISTS only (the outer list and every strand list are new), not of the numpy arrays: assigning into it below does not change self.data,
                        # which is only replaced by this data if the move is accepted (update_geometry). The arrays of the vertices that do not move are shared and never modified in place.
                        tmpGeometryData = [list(strand) for strand in self.data]
                        for (i,j) in ind[1:-1:]:
                            tmpGeometryData[i][j]=newPos.pop(0)
                        return tmpGeometryData

                d_upperBound = 0.5*d_upperBound
            
            numberOfTries+=1

        print("All Points Are Jammed!")
        return 1

    def compute_deltaStar(self, radius, edgeLength):
        return np.arcsin(0.5*edgeLength/m.sqrt(radius**2 + 0.25*edgeLength**2))

    def compute_upper_bound_closest_self_distance(self, radius, edgeLength):
        return 2*m.sqrt(radius**2 + 0.25*edgeLength**2)

    def check_reach(self, withInfo=False):
        """Function returns the minimum arc radius and half the minimum distance of the closest arc to each arc along the curve."""
        return (returnMinimumInNestedListOfFloats(self.generates_list_arc_radii()), min(self.generate_closest_distances(withInfo=withInfo)))
        #return (returnMinimumInNestedListOfFloats(self.generates_list_arc_radii()), min(self_dist.generate_closest_distances(self.data, self.configType, self.skippedInteger, withInfo=withInfo)))

    def evaluate_curve_vertices(self, geometryData):
        """Function returns the curve vertices corresponding to the data of the geometry object.
           If the geometry type is ThreadedBeads (as it is the  case here) then the geometryData is the curve vertices.
           For the biarc curve this is different.
        """
        return geometryData

    # -- former TubularGeometry methods that stay on TEP (zero branching).
    def evaluate_normalised_energy(self):
        """Computes the normalised energy"""

        return np.dot(np.array(self.coefficients), np.array([self.V - self.V_0, self.A - self.A_0, self.C - self.C_0, self.X - self.X_0]))/self.length

    # -- geometryType branch collapses: only the ThreadedBeads path survives.
    #    The umbrella_sampling spring term is not implemented yet.
    def evaluate_energy_difference(self, neighbouring_curve_vertices):
        """Function computes the measures of the neighbouringCurveVertices with the radius input_R as associated to the class variable.
           The energy of self is compared to the energy of a geeometry who's curve_object would produce the curve_vertices given by neighbouring_curve_vertices."""

        pointList = [pt for subList in neighbouring_curve_vertices for pt in subList]

        measures = mm.morph_(pointList, self.input_R)

        #deltaE = (tmpE - E)
        deltaE = np.dot(np.array(self.coefficients), np.array(measures) - np.array([self.V, self.A, self.C, self.X]))

        if self.umbrella_sampling is not None:
            raise NotImplementedError("umbrella_sampling spring energy is not implemented yet")

        return (deltaE, measures)


class OpenChain(ThickenedEquilateralPolygon):

    configType = "open"

    def __init__(self, curveData, edgeLength=0):

        self.geometryType = "ThreadedBeads"

        if len(curveData)==0:
            print("You need to initialise the shape of the curve either from a .txt file or from a pointFilaments.pointListName")
        else:
            self.data = [[np.array(curveData[i][j]) for j in range(len(curveData[i]))] for i in range(len(curveData))]

        self.length = sum(self.compute_length())
        #returnLengthOfPointList(self.curve_vertices, self.configType)

        if edgeLength > 0:
            self.edgeLength = edgeLength
        else:
            self.edgeLength = self.compute_average_edge_length()

        # -- check this curve against the class-level edgeLength (energy parameters must be set first)
        if not hasattr(ThickenedEquilateralPolygon, 'edgeLength'):
            sys.exit("call ThickenedEquilateralPolygon.set_energy_parameters(...) before creating a curve")
        if abs(self.compute_average_edge_length() - ThickenedEquilateralPolygon.edgeLength) > 0.01*ThickenedEquilateralPolygon.edgeLength:
            sys.exit("edge length mismatch: expected " + str(ThickenedEquilateralPolygon.edgeLength) + " but measured average edge length " + str(self.compute_average_edge_length()))

        self.deltaStar = self.compute_deltaStar(rTube, self.compute_average_edge_length())

        self.skippedInteger = m.floor(0.5*m.pi/self.deltaStar)

        self.arcPairs = self.evaluate_arc_pairs_to_be_checked(self.skippedInteger)

        self.upper_bound_closest_self_distance = self.compute_upper_bound_closest_self_distance(rTube, self.compute_average_edge_length())

        nMax = int(0.5*min([len(self.data[i]) for i in range(len(self.data))]))

        self.index_intervals_to_be_rotated = self.generate_index_intervals_to_be_rotated(nMax)

        self.curve_vertices = self.evaluate_curve_vertices(self.data)

        # -- compute all measures at construction
        self.evaluate_embedded_measures()
        self.evaluate_measures()

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
        return round(runningTotal/float(edgeTally), 10)

    def compute_length(self):
        """Sums the length of each edge and saves in the variable length."""

        result = []
        for i in range(len(self.data)):
            runningTotal = 0
            j = 1
            while j<len(self.data[i]):
                runningTotal+=np.linalg.norm(self.data[i][j] - self.data[i][j-1])
                j+=1
            result.append(runningTotal)
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
        print(dlMin,'< dl <', dlMax)
        return None

    def evaluate_arc_pairs_to_be_checked(self, skippedInteger):
        """appends an integer pair ((i,j), (l,k)) to the list arcpairs if the corresponding arcs have an arc length separation of at least pi."""

        tmpList = []
        for i in range(len(self.data)):
            #print(i, skippedinteger)
            for j in range(len(self.data[i])):
                indicesToSkip = [(i,j)]
                s = 1
                while s < min(skippedInteger, len(self.data[i]) - j):
                    indicesToSkip.append((i,j+s))
                    s+=1
                s = 1
                while s < min(skippedInteger, j+1):
                    indicesToSkip.insert(0, (i,(j-s)))
                    s+=1
                #print((i, j), indicestoskip)

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

    # -- kept per your call, NOT dead code: a Python-level prototype for
    #    developing overlap-check logic before porting to the C++ library.
    def check_new_positions_do_not_cause_overlaps(self, upper_bound_closest_self_distance, newPositions, indices):
        """Return True if the polygonal curve defined by self with the newPositions given at index locations indices, spheres separated by at least a skippedInteger are at least 2radius apart."""

        for ((i,j), (l,k)) in self.arcPairs:
            if (i,j) in indices[1:-1:]:
                if (l,k) in indices[1:-1:]:#both arc pairs belong to the rotated group
                    continue
                #check distance
                p = newPositions[indices.index((i,j))-1] #indices[1:-1:].index((i,j)) = indices.index(i,j) - 1
                q = self.data[l][k]
                if np.linalg.norm(p - q)<upper_bound_closest_self_distance:
                    return 0
            elif (i,j) in [indices[0], indices[-1]]:#first pair belongs to a join arc, second pair to any arc
                p = self.data[i][j]
                if (l,k) in indices[1:-1:]:
                    q = newPositions[indices.index((l,k))-1]
                else:
                    q = self.data[l][k]
                if np.linalg.norm(p - q)<upper_bound_closest_self_distance:
                    return 0
            elif (l,k) in indices[1:-1:]:#first pair is unmoved, second pair rotated
                p = np.array(self.data[i][j])
                q = np.array(newPositions[indices.index((l,k))-1])
                if np.linalg.norm(p - q)<upper_bound_closest_self_distance:
                    return 0
            elif (l,k) in [indices[0], indices[1]]:#first pair is unmoved, second pair join
                p = np.array(self.data[i][j])
                q = np.array(self.data[l][k])
                if np.linalg.norm(p - q)<upper_bound_closest_self_distance:
                    return 0
        return 1

    def generate_index_intervals_to_be_rotated(self, Nmax):
        """ generates a list of index intervals, this list is shuffled and an interval list is drawn at random. if the curve is open intervals containing the end vertices to be extended with a dummy vertex are included"""

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
        return indexList

    # -- STAGED: duplicate now (here) -> test -> promote into TEP with a
    #    next_index/prev_index hook once validated -> benchmark.
    def prev_point(self, i, j, x=1):
        """position of the point before vertex j of strand i: Data(i, j-1). The first vertex has no previous vertex: it gets a dummy point
        x edge lengths before it, at a random place in a cone around the direction of the extension of the first edge."""
        if j == 0:
            t = self.data[i][0] - self.data[i][1]
            e = np.linalg.norm(t)
            t = t/e
            randomUnitVectorInCone = returnRandomUnitVectorInCone(t, x*self.deltaStar)
            return self.data[i][0] + x*e*randomUnitVectorInCone
        return self.data[i][j-1]

    def next_point(self, i, j, x=1):
        """position of the point after vertex j of strand i: Data(i, j+1). The last vertex has no next vertex: it gets a dummy point
        x edge lengths after it, at a random place in a cone around the direction of the extension of the last edge."""
        if j == len(self.data[i]) - 1:
            t = self.data[i][-1] - self.data[i][-2]
            e = np.linalg.norm(t)
            t = t/e
            randomUnitVectorInCone = returnRandomUnitVectorInCone(t, x*self.deltaStar)
            return self.data[i][-1] + x*e*randomUnitVectorInCone
        return self.data[i][j+1]

    def check_curvature_constraint(self, ind, newPos):
        """True if the turning angle at both ends of the rotated interval stays below 2*deltaStar. An end vertex of the chain (or a dummy 'x' end) is not tested."""
        cosLimit = m.cos(2*self.deltaStar)      # the turning angle exceeds 2*deltaStar exactly when cos(angle) < cos(2*deltaStar)
        if (ind[0][1] not in  ['x', 0]):#start vertex is not an interior vertex if ind[0][1] = 'x' or 0 and is never n-1
            if turning_angle_exceeds(self.prev_point(ind[0][0], ind[0][1]), self.data[ind[0][0]][ind[0][1]], newPos[0], cosLimit):
                return False
        if (ind[-1][1] not in ['x', (len(self.data[ind[-1][0]])-1)]):#last vertex is the beginning vertex of the chain
            if turning_angle_exceeds(newPos[-1], self.data[ind[-1][0]][ind[-1][1]], self.next_point(ind[-1][0], ind[-1][1]), cosLimit):
                return False
        return True

    # -- needs the end points to make the polyFile image correctly.
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
            poly_count+=1

        f1.write('\nEND')
        f1.close()

    # -- needs the end points (open trims the first/last index, no wraparound neighbor).
    def generates_list_arc_radii(self):
        """Creates list floats. Each float gives the radius of the arc defined by the bezier triangle at that index."""
        result = []
        for i in range(len(self.data)):
            strand = []
            for j in range(1, len(self.data[i])-1):
                p0 = self.data[i][j-1]
                p = self.data[i][j]
                p1 = self.data[i][j+1]
                strand.append(returnCircleRadiusForControlTriangle(p0, p, p1))
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
                        if l==i and (k - j < self.skippedInteger):
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

    # -- from TubularGeometry. Branched on configType (endCaps) in the original;
    #    open version uses endCaps=True.
    def evaluate_embedded_measures(self):
        """ Function computes the embedded measures and saves them as variables of the curve"""

        endCaps=True

        self.V_0, self.A_0, self.C_0, self.X_0 = returnEmbeddedMeasures(self.curve_vertices, self.input_R, endCaps=endCaps)
        return None

    # -- from TubularGeometry. Relies on curve_vertices flattening to the sphere centres.
    def evaluate_measures(self):
        """Function computes the measures of the curveData point list with balls of radius input_R """

        pointList = [pt for subList in self.curve_vertices for pt in subList]

        self.V, self.A, self.C, self.X=mm.morph_(pointList, self.input_R)

        return None

    # -- from TubularGeometry. Updates the INSTANCE state; Biarcs branch dropped.
    def update_geometry(self, data, curve_vertices, size_measures):
        """updates self with attributes of geometry.
           measures, embedded_measures = [], length
        """

        self.data = data
        self.curve_vertices = curve_vertices

        #measures = size_measures
        self.V, self.A, self.C, self.X = size_measures

        return None

    # -- copies an INSTANCE of a specific object; TEP never has instances.
    #    (original body, still broken: type(self)() needs curveData)
    def __copy__(self):
        """Returns a copy of the object and a number of variables."""

        newone = type(self)()
        for item in ['configType', 'other object attributes to pass over to new instance', 'etc']:
            newone.__dict__[item] = self.__dict__[item]
        return newone


# -- same shape as OpenChain throughout; configType="closed".
class ClosedChain(ThickenedEquilateralPolygon):

    configType = "closed"

    def __init__(self, curveData, edgeLength=0):

        self.geometryType = "ThreadedBeads"

        if len(curveData)==0:
            print("You need to initialise the shape of the curve either from a .txt file or from a pointFilaments.pointListName")
        else:
            self.data = [[np.array(curveData[i][j]) for j in range(len(curveData[i]))] for i in range(len(curveData))]

        self.length = sum(self.compute_length())
        #returnLengthOfPointList(self.curve_vertices, self.configType)

        if edgeLength > 0:
            self.edgeLength = edgeLength
        else:
            self.edgeLength = self.compute_average_edge_length()

        # -- check this curve against the class-level edgeLength (energy parameters must be set first)
        if not hasattr(ThickenedEquilateralPolygon, 'edgeLength'):
            sys.exit("call ThickenedEquilateralPolygon.set_energy_parameters(...) before creating a curve")
        if abs(self.compute_average_edge_length() - ThickenedEquilateralPolygon.edgeLength) > 0.01*ThickenedEquilateralPolygon.edgeLength:
            sys.exit("edge length mismatch: expected " + str(ThickenedEquilateralPolygon.edgeLength) + " but measured average edge length " + str(self.compute_average_edge_length()))

        self.deltaStar = self.compute_deltaStar(rTube, self.compute_average_edge_length())

        self.skippedInteger = m.floor(0.5*m.pi/self.deltaStar)

        self.arcPairs = self.evaluate_arc_pairs_to_be_checked(self.skippedInteger)

        self.upper_bound_closest_self_distance = self.compute_upper_bound_closest_self_distance(rTube, self.compute_average_edge_length())

        nMax = int(0.5*min([len(self.data[i]) for i in range(len(self.data))]))

        self.index_intervals_to_be_rotated = self.generate_index_intervals_to_be_rotated(nMax)

        self.curve_vertices = self.evaluate_curve_vertices(self.data)

        # -- compute all measures at construction
        self.evaluate_embedded_measures()
        self.evaluate_measures()

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
            runningTotal+=np.linalg.norm(self.data[i][0] - self.data[i][-1])
            edgeTally+=1
        return round(runningTotal/float(edgeTally), 10)

    def compute_length(self):
        """Sums the length of each edge and saves in the variable length."""

        result = []
        for i in range(len(self.data)):
            runningTotal = 0
            j = 1
            while j<len(self.data[i]):
                runningTotal+=np.linalg.norm(self.data[i][j] - self.data[i][j-1])
                j+=1
            runningTotal+=np.linalg.norm(self.data[i][-1] - self.data[i][0])
            result.append(runningTotal)
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
            tmpFloat=np.linalg.norm(self.data[i][0] - self.data[i][-1])
            #print(tmpFloat)
            if tmpFloat>dlMax:
                dlMax = tmpFloat
            if tmpFloat<dlMin:
                dlMin = tmpFloat
        print(dlMin,'< dl <', dlMax)
        return None

    def evaluate_arc_pairs_to_be_checked(self, skippedInteger):
        """appends an integer pair ((i,j), (l,k)) to the list arcpairs if the corresponding arcs have an arc length separation of at least pi."""

        tmpList = []
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
                #print((i, j), indicestoskip)

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

    def check_new_positions_do_not_cause_overlaps(self, upper_bound_closest_self_distance, newPositions, indices):
        """Return True if the polygonal curve defined by self with the newPositions given at index locations indices, spheres separated by at least a skippedInteger are at least 2radius apart."""

        for ((i,j), (l,k)) in self.arcPairs:
            if (i,j) in indices[1:-1:]:
                if (l,k) in indices[1:-1:]:#both arc pairs belong to the rotated group
                    continue
                #check distance
                p = newPositions[indices.index((i,j))-1] #indices[1:-1:].index((i,j)) = indices.index(i,j) - 1
                q = self.data[l][k]
                if np.linalg.norm(p - q)<upper_bound_closest_self_distance:
                    return 0
            elif (i,j) in [indices[0], indices[-1]]:#first pair belongs to a join arc, second pair to any arc
                p = self.data[i][j]
                if (l,k) in indices[1:-1:]:
                    q = newPositions[indices.index((l,k))-1]
                else:
                    q = self.data[l][k]
                if np.linalg.norm(p - q)<upper_bound_closest_self_distance:
                    return 0
            elif (l,k) in indices[1:-1:]:#first pair is unmoved, second pair rotated
                p = np.array(self.data[i][j])
                q = np.array(newPositions[indices.index((l,k))-1])
                if np.linalg.norm(p - q)<upper_bound_closest_self_distance:
                    return 0
            elif (l,k) in [indices[0], indices[1]]:#first pair is unmoved, second pair join
                p = np.array(self.data[i][j])
                q = np.array(self.data[l][k])
                if np.linalg.norm(p - q)<upper_bound_closest_self_distance:
                    return 0
        return 1

    def generate_index_intervals_to_be_rotated(self, Nmax):
        """ generates a list of index intervals, this list is shuffled and an interval list is drawn at random. if the curve is open intervals containing the end vertices to be extended with a dummy vertex are included"""

        indexList = []
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                for n in range(3, Nmax + 1):
                    indexList.append([(i, (j + k)%len(self.data[i])) for k in range(n)])
        return indexList

    def prev_point(self, i, j, x=1):
        """position of the point before vertex j of strand i: Data(i, j-1), wrapping around at the start (x is not used: a closed curve has no ends)"""
        return self.data[i][(j - 1) + len(self.data[i])*(j==0)]

    def next_point(self, i, j, x=1):
        """position of the point after vertex j of strand i: Data(i, j+1), wrapping around at the end (x is not used: a closed curve has no ends)"""
        return self.data[i][(j + 1)%len(self.data[i])]

    def check_curvature_constraint(self, ind, newPos):
        """True if the turning angle at both ends of the rotated interval stays below 2*deltaStar"""
        cosLimit = m.cos(2*self.deltaStar)      # the turning angle exceeds 2*deltaStar exactly when cos(angle) < cos(2*deltaStar)
        if turning_angle_exceeds(self.prev_point(ind[0][0], ind[0][1]), self.data[ind[0][0]][ind[0][1]], newPos[0], cosLimit):
            return False
        if turning_angle_exceeds(newPos[-1], self.data[ind[-1][0]][ind[-1][1]], self.next_point(ind[-1][0], ind[-1][1]), cosLimit):
            return False
        return True

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
            f1.write(str(point_count - len(self.data[i]))+'')#bit of a hack
            poly_count+=1

        f1.write('\nEND')
        f1.close()

    def generates_list_arc_radii(self):
        """Creates list floats. Each float gives the radius of the arc defined by the bezier triangle at that index."""
        result = []
        for i in range(len(self.data)):
            strand = []
            for j in range(len(self.data[i])):
                p0 = self.data[i][j-1]
                p = self.data[i][j]
                p1 = self.data[i][(j+1)%len(self.data[i])]
                strand.append(returnCircleRadiusForControlTriangle(p0, p, p1))
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

    # -- from TubularGeometry; closed version uses endCaps=False.
    def evaluate_embedded_measures(self):
        """ Function computes the embedded measures and saves them as variables of the curve"""

        endCaps=False

        self.V_0, self.A_0, self.C_0, self.X_0 = returnEmbeddedMeasures(self.curve_vertices, self.input_R, endCaps=endCaps)
        return None

    def evaluate_measures(self):
        """Function computes the measures of the curveData point list with balls of radius input_R """

        pointList = [pt for subList in self.curve_vertices for pt in subList]

        self.V, self.A, self.C, self.X=mm.morph_(pointList, self.input_R)

        return None

    def update_geometry(self, data, curve_vertices, size_measures):
        """updates self with attributes of geometry.
           measures, embedded_measures = [], length
        """

        self.data = data
        self.curve_vertices = curve_vertices

        #measures = size_measures
        self.V, self.A, self.C, self.X = size_measures

        return None

    def __copy__(self):
        """Returns a copy of the object and a number of variables."""

        newone = type(self)()
        for item in ['configType', 'other object attributes to pass over to new instance', 'etc']:
            newone.__dict__[item] = self.__dict__[item]
        return newone


# -- fixed endpoints; easy extension of OpenChain.
class ClampedChain(OpenChain):
    pass  # Phase 1 target -- not implemented yet


# -- closed loop with a trailing open tail.
class Lasso(ThickenedEquilateralPolygon):
    pass  # Phase 1 target -- not implemented yet
