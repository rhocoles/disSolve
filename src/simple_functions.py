import math as m
import numpy as np
from numba import njit
import random
import itertools

def Di(x, y): 
    """ 
    Function: Di
    -------------------------------------------------------------------------------------------------------------------------------------------------------------
    calculates the distance between two points
    (x1,y1,z1): coordinate postion
    (x2,y2,z2): coordinate position
    returns: scalar
    """
    (x1, x2, x3) = x
    (y1, y2, y3) = y
    
    return m.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)

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

def rotate(v, angle, w):
    """
    Function: rotate
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    returns the vector v rotated by angle about w, assumed w is a unit vector
    angle float
    v list float
    w list float
    return np array
    """
    rotMatrix = lambda n, theta: np.array([[np.cos(theta) + (1-np.cos(theta))*n[0]**2, n[0]*n[1]*(1-np.cos(theta)) - n[2]*np.sin(theta), n[0]*n[2]*(1 - np.cos(theta)) + n[1]*np.sin(theta)],[n[0]*n[1]*(1-np.cos(theta)) + n[2]*np.sin(theta), np.cos(theta) + (1-np.cos(theta))*n[1]**2, n[1]*n[2]*(1-np.cos(theta)) - n[0]*np.sin(theta)], [n[0]*n[2]*(1-np.cos(theta)) - n[1]*np.sin(theta), n[1]*n[2]*(1-np.cos(theta)) + n[0]*np.sin(theta), np.cos(theta) + (1-np.cos(theta))*n[2]**2]])
    return np.dot(rotMatrix(w, angle), np.array(v))

def circleRadiusMatchedPointTangentData(pt0, pt1):
    """
    Function: circleRadiusMatchedPointTangentData
    -----------------------------------------------------------------------------------------------------------------------
    function returns radius of the circular arc interpolating the matched point tangent pairs (p0,t0), (p1, t1)
    (p, t) point tangent data
    return float
    """
    (p0, t0) = pt0
    (p1, t1) = pt1
    D = np.linalg.norm(np.array(p0) - np.array(p1))
    t0Dott1 = np.clip(np.dot(t0, t1), -0.999999999999, 0.999998)
    return D/m.sqrt(2*(1 - t0Dott1))

def returnCircleRadiusForBezierTriangle(b0, b1, b2):
    """
    Function: returnCircleRadiusForBezierTriangle
    --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    function returns the radii of the circular arc defined by the bezier triangle (b0, b, b1)
    b0,b1, b2 three tuples
    return float
    """
    e0 = np.linalg.norm(np.array(b0) - np.array(b1))
    e1 = np.linalg.norm(np.array(b2) - np.array(b1))
    t0 = np.array(b1) - np.array(b0)
    t0 = t0/np.linalg.norm(t0)
    t1 = np.array(b2) - np.array(b1)
    t1 = t1/np.linalg.norm(t1)
    t0Dott1 = np.clip(np.dot(t0, t1), -0.999999999999, 0.999998)
    
    return (0.5*(e0 + e1)*m.sqrt(1 + t0Dott1))/m.sqrt(1 - t0Dott1)

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

def returnArcLengthForBezierTriangle(b0, b1, b2):
    """
    Function: returnArcLengthForBezierTriangle
    --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    function returns the radii of the circular arc defined by the bezier triangle (b0, b, b1)
    b0,b1, b2 three tuples
    return float
    """
    e0 = np.linalg.norm(np.array(b0) - np.array(b1))
    e1 = np.linalg.norm(np.array(b2) - np.array(b1))
    t0 = np.array(b1) - np.array(b0)
    t0 = t0/np.linalg.norm(t0)
    t1 = np.array(b2) - np.array(b1)
    t1 = t1/np.linalg.norm(t1)
    t0Dott1 = np.clip(np.dot(t0, t1), -0.999999999999, 0.999998)
    
    return np.arccos(t0Dott1)*(0.5*(e0 + e1)*m.sqrt(1 + t0Dott1))/m.sqrt(1 - t0Dott1)

def returnPointOnArcAtDistance(a, separationDistance, normal, tangent, centre, circleRadius):
    """
    Function: returnPointOnArcAtDistance
    ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    function returns a three tuple. The return is one of the two intersection points of the circle defined by the centre, circleRadius and normal vector and the sphere centred at a
    of radius separationDistance. The return point from the choice of two is that point which is in the direction of the tangent vector t0.
    a three tuple
    separationDistance float
    normal numpy three array
    tangent numpy three array
    centre three tuple
    circleRadius float
    return three tuple
    """
    e = np.array(centre) - np.array(a)
    D = np.linalg.norm(e)
    e = e/D
    n = np.array(a) + e*(separationDistance**2 + D**2 - circleRadius**2)/(2*D)
    r = m.sqrt((2*separationDistance*D)**2 - (separationDistance**2 + D**2 - circleRadius**2)**2)/(2*D)

    b = np.cross(normal, e)
    if np.linalg.norm(b)<0.001:
        print('problem with joining the end points')
        quit()
    if np.dot(b, tangent)<0:
        b = -b

    b = b/np.linalg.norm(b)
    newPoint = tuple(n + r*b)
    #correct newPoint to be on arc
    direction = np.array(newPoint) - np.array(centre) - np.dot(np.array(newPoint) - np.array(centre), normal)*normal
    direction = direction/np.linalg.norm(direction)
    return tuple(np.array(centre) + circleRadius*direction)
    #return tuple(n + r*b)

def returnPointInterpolationMatchedPointTangentData(pointTangentList, pointSeparation):
    """
    Function: returnPointInterpolationMatchedPointTangentData
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    returns a list of three tuples spaced equidistant (pointSeparation) along the piecewise circular arc defined by the point tangent data in the pointTangentList
    pointTangentList nested list of point tangent data
    pointSeparation float
    return list of three tuples
    """
    deltaTau = lambda circleRadius: np.arcsin(0.5*pointSeparation/circleRadius)
    result = []
    for i in range(len(pointTangentList)):
        strand = [pointTangentList[i][0][0]]
        for j in range(len(pointTangentList[i])-1):
            circleRadius = circleRadiusMatchedPointTangentData(pointTangentList[i][j], pointTangentList[i][j+1])
            normal = np.cross(pointTangentList[i][j][1], pointTangentList[i][j+1][1])/np.linalg.norm(np.cross(pointTangentList[i][j][1], pointTangentList[i][j+1][1]))
            centre = tuple( 0.5*(np.array(pointTangentList[i][j][0]) + np.array(pointTangentList[i][j+1][0])) + circleRadius*m.sqrt(0.5*(1 + np.dot(pointTangentList[i][j][1], pointTangentList[i][j+1][1])))*(np.cross(normal, np.array(pointTangentList[i][j+1][0]) - np.array(pointTangentList[i][j][0]))/np.linalg.norm(np.cross(normal, np.array(pointTangentList[i][j+1][0]) - np.array(pointTangentList[i][j][0])))))
            strand.append(returnPointOnArcAtDistance(strand[-1], pointSeparation, normal, pointTangentList[i][j][1], centre, circleRadius))
            dTau = deltaTau(circleRadius)
            while np.linalg.norm(np.array(strand[-1]) - np.array(pointTangentList[i][j+1][0]))> pointSeparation:
                strand.append(tuple(np.array(centre) + rotate(np.array(strand[-1]) - np.array(centre), 2*dTau, normal)))
        if np.linalg.norm(np.array(pointTangentList[i][0][0]) - np.array(pointTangentList[i][-1][0]))>pointSeparation:#strand is open
            result.append(strand + [pointTangentList[i][-1][0]])
        else:
            result.append(strand)

    return result

def returnPointInterpolationFromBezierTriangle(pointTangentList, pointSeparation):
    """
    Function: returnPointInterpolationMatchedPointTangentData
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    returns a list of three tuples spaced equidistant (pointSeparation) along the piecewise circular arc defined by the point tangent data in the pointTangentList
    pointTangentList nested list of point tangent data
    pointSeparation float
    return list of three tuples
    """
    deltaTau = lambda circleRadius: np.arcsin(0.5*pointSeparation/circleRadius)
    result = []
    for i in range(len(pointTangentList)):
        strand = [pointTangentList[i][0][0]]
        for j in range(len(pointTangentList[i])-1):
            circleRadius = circleRadiusMatchedPointTangentData(pointTangentList[i][j], pointTangentList[i][j+1])
            normal = np.cross(pointTangentList[i][j][1], pointTangentList[i][j+1][1])/np.linalg.norm(np.cross(pointTangentList[i][j][1], pointTangentList[i][j+1][1]))
            centre = tuple( 0.5*(np.array(pointTangentList[i][j][0]) + np.array(pointTangentList[i][j+1][0])) + circleRadius*m.sqrt(0.5*(1 + np.dot(pointTangentList[i][j][1], pointTangentList[i][j+1][1])))*(np.cross(normal, np.array(pointTangentList[i][j+1][0]) - np.array(pointTangentList[i][j][0]))/np.linalg.norm(np.cross(normal, np.array(pointTangentList[i][j+1][0]) - np.array(pointTangentList[i][j][0])))))
            strand.append(returnPointOnArcAtDistance(strand[-1], pointSeparation, normal, pointTangentList[i][j][1], centre, circleRadius))
            dTau = deltaTau(circleRadius)
            while np.linalg.norm(np.array(strand[-1]) - np.array(pointTangentList[i][j+1][0]))> pointSeparation:
                strand.append(tuple(np.array(centre) + rotate(np.array(strand[-1]) - np.array(centre), 2*dTau, normal)))
        if np.linalg.norm(np.array(pointTangentList[i][0][0]) - np.array(pointTangentList[i][-1][0]))>pointSeparation:#strand is open
            result.append(strand + [pointTangentList[i][-1][0]])
        else:
            result.append(strand)

    return result

def returnRandomUnitVector():
    """
    Function: returnRandomUnitVector
    -------------------------------------------------------------------------------
    function returns a random 3dim unit vector
    return np.array
    """
    a = random.uniform(0, 6.283185307179586)
    b = random.uniform(-1.5707963267948966, 1.5707963267948966)
    return np.array([m.cos(a)*m.cos(b), m.sin(a)*m.cos(b), m.sin(b)])

def returnRandomUnitVectorInCone(normalVector, openingAngle):
    """
    Function: returnRandomUnitVectorInCone
    -------------------------------------------------------------------------------
    function returns a random vector in the cone with angle openingAngle from the vector normalVector
    return np.array
    """
    openingAngle = min(openingAngle, 0.5*m.pi)
    helpVector = returnRandomUnitVector()
    h = random.uniform(0, m.sin(openingAngle))
    return  h*(helpVector - np.dot(helpVector, normalVector)*normalVector)/m.sqrt(1 - np.dot(helpVector, normalVector)**2) + normalVector*m.sqrt(1 - h**2)

def returnRandomPointInIntersectionOfSphereCaps(c, r, c1, r1, c2, r2):
    """
    Function: returnRandomPointInIntersectionOfSphereCaps
    -----------------------------------------------------------------------------------------------------------------------------------------------
    returns a random point in the intersection of the spherical caps bounded by two intersection circles on the sphere at c of radius r
    the first circle is centred at c1 of radius r1
    the second circle is the intersection of S(c2,r2) and S(c,r)
    c,c1,c2 three tuple
    r, r1, r2 float
    return three tuple
    """
    N1 = np.array(c1)- np.array(c)
    x1 = np.linalg.norm(N1)
    N1 = N1/x1
    N2 = np.array(c2)- np.array(c)
    x2 = np.linalg.norm(N2)
    N2 =  N2/x2
    l = np.cross(N1, N2)
    #get parameters of the boundary circles
    #x1 = (r**2 + np.linalg.norm(np.array(c) - np.array(c1))**2 - r1**2)/(2*np.linalg.norm(np.array(c) - np.array(c1)))
    #r1 = m.sqrt(r**2 - x1**2)#radius of circle 1
    #c1 = tuple(np.array(c) + x1*N1)#centre of circle 2
    x2 = (r**2 + x2**2 - r2**2)/(2*x2)
    if x2>r:
        print('spheres dont intersect ?', x2, np.linalg.norm(np.array(c) - np.array(c2)), '>', r+r2)
    r2 = m.sqrt(r**2 - x2**2)#radius of circle 2
    c2 = tuple(np.array(c) + x2*N2)#centre of circle 2
    #print('check viable ', np.linalg.norm(np.array(c1) - np.array(c2)), '<', r1+ r2)
    
    if np.arcsin(np.linalg.norm(l)) + np.arcsin(r2/r) < np.arcsin(r1/r):#spherical cap c2, r2 sits inside spherical cap c1, r1
        u = returnRandomUnitVectorInCone(N2, np.linalg.norm(r2/r))
        return tuple(np.array(c) +  r*u)
    else:
        sinAlpha = np.linalg.norm(l)
        l = l/sinAlpha
        e = np.array(c2) - np.array(c1)
        d = np.linalg.norm(e)
        e = e/d
        c1_n = (r1**2 + d**2 - r2**2)/(2*d)
        c1_i = c1_n/(np.dot(N2 - np.dot(N1, N2)*N1, e)/sinAlpha)
        ha = m.sqrt(np.clip(r1**2 - c1_i**2, 0, r1))
        x = np.array(c1) + c1_i*(N2 - np.dot(N1, N2)*N1)/sinAlpha + np.random.uniform(-ha, ha)*l
        y = np.array(c) + r*(x - np.array(c))/np.linalg.norm(x - np.array(c))
        #print('check', np.linalg.norm(y - np.array(c)), '=', r, 'inside c1 check', np.linalg.norm(y - np.array(c1)), '<', r1, 'inside c2 check', np.linalg.norm(y - np.array(c2)), '<', r2)
        return y

def skippedInteger(nodeSeparation, radius):
    """
    Function: skippedInteger
    --------------------------------------------
    returns an integer which is the number of nodes mapping the circle of radius rTube and opening angle pi.
    all nodes within the skippedInteger should be not closet than the closestDistance function
    nodeSeparation float
    radius float
    """
    return int(m.pi/(2*np.arctan(0.5*nodeSeparation/radius)))

def returnClosestPointOnSphereIntersection(point, a, b, radius):
    """
    Function: returnClosestPointOnSphereIntersection
    --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    function returns a random point on the intersection of the spheres centered at a with radius aRadius and b of radius bRadius
    a three tuple
    b three tuple
    aRadius float (dl)
    bRadius float (d)
    return three tuple
    """
    point = np.array(point)
    a = np.array(a)
    b = np.array(b)
    
    circleRadius = m.sqrt(np.clip(radius**2 - 0.25*np.dot(b-a, b-a), 0, 5.0))
    if circleRadius < 0.001:
        return tuple(0.5*(a + b))
    else:
        normalVector = b - a
        normalVector = normalVector/np.linalg.norm(normalVector)
        circleCentre = 0.5*(a + b)
        u = point - circleCentre
        u = u - np.dot(u, normalVector)*normalVector
        u = u/np.linalg.norm(u)
    return tuple(circleCentre + circleRadius*u)
    
def returnRandomPointOnSphereIntersection(a, aRadius, b, bRadius):
    """
    Function: returnRandomPointOnSphereIntersection
    --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    function returns a random point on the intersection of the spheres centered at a with radius aRadius and b of radius bRadius
    a three tuple
    b three tuple
    aRadius float (dl)
    bRadius float (d)
    return three tuple
    """
    a = np.array(a)
    b = np.array(b)
    
    circleRadius = m.sqrt(bRadius**2 - (0.5*bRadius**2/aRadius)**2)
    normalVector = a - b
    normalVector = normalVector/np.linalg.norm(normalVector)
    circleCentre = b + (0.5*bRadius**2/aRadius)*normalVector
    
    return returnRandomPointOnCircle(circleCentre, normalVector, circleRadius)


def circleInterpolationMatchedPointTangentData(pt0, pt1,  pointSeparation):
    """
    Function: circleInterpolationMatchedPointTangentData
    -----------------------------------------------------------------------------------------------------------------------
    function returns a list of  points which lie of the circular arc interpolating the matched point tangent pairs (p0,t0), (p1, t1)
    points are arrange qually spaced at a distance close to pointSeparation
    the last point p1 is excluded from the return list
    (p, t) point tangent data
    pointSeparation
    return list of three tuples
    """
    (p0, t0) = pt0
    (p1, t1) = pt1
    D = np.linalg.norm(np.array(p0) - np.array(p1))
    t0Dott1 = np.clip(np.dot(t0, t1), -0.999999999999, 0.999999999999)
    if t0Dott1>0.999998:
        N = int(D/pointSeparation)
        correctSeparation = D/float(N)
        v = (np.array(p1) - np.array(p0))/D
        result = []
        for i in range(N):
            result.append(tuple(np.array(p0) + i*correctSeparation*v))
    else:
        r = D/m.sqrt(2*(1 - t0Dott1))
        #halfDl = D/m.sqrt(2*(1 + t0Dott1))

        #calculate the number of points
        N = int(0.5*np.arccos(t0Dott1)/np.arcsin(0.5*pointSeparation/r))
        deltaTau = 0.5*np.arccos(t0Dott1)/float(N)

        binormal = np.cross(t0, t1)/m.sqrt(1 - t0Dott1**2)
        result = []
        p = np.array(p0)
        tangent = t0
        correctSeparation = 2*r*m.sin(deltaTau)
        for i in range(N):
            #p = p + correctSeparation*rotate(tangent, deltaTau, binormal)
            p = p0 + 2*r*m.sin(i*deltaTau)*rotate(t0, i*deltaTau, binormal)
            result.append(p)

    return result

def circleInterpolationBezierTriangle(b,  N):
    """
    Function: circleInterpolationBezierTriangle
    -----------------------------------------------------------------------------------------------------------------------
    function returns a list of  points which lie of the circular arc defined by the bezier triangle
    N points are equally spaced along the arc excluding the end point 
    the last point p1 is excluded from the return list
    (b0, b1, b2) tuple of three tuples
    N integer
    return list of three tuples
    """
    (b0, b1, b2) = b
    t0 = np.array(b1) - np.array(b0)
    t0 = t0/np.linalg.norm(t0)
    t1 = np.array(b2) - np.array(b1)
    t1 = t1/np.linalg.norm(t1)
    D = np.linalg.norm(np.array(b0) - np.array(b2))
    t0Dott1 = np.clip(np.dot(t0, t1), -0.999999999999, 0.999999999999)
    if t0Dott1>0.999998:
        #N = int(D/pointSeparation)
        correctSeparation = D/float(N)
        v = (np.array(b2) - np.array(b0))/D
        result = []
        for i in range(N):
            result.append(tuple(np.array(b0) + i*correctSeparation*v))
    else:
        r = D/m.sqrt(2*(1 - t0Dott1))
        #halfDl = D/m.sqrt(2*(1 + t0Dott1))

        #calculate the number of points
        #N = int(0.5*np.arccos(t0Dott1)/np.arcsin(0.5*pointSeparation/r))
        deltaTau = 0.5*np.arccos(t0Dott1)/float(N)

        binormal = np.cross(t0, t1)/m.sqrt(1 - t0Dott1**2)
        result = []
        p0 = np.array(b0)
        tangent = t0
        correctSeparation = 2*r*m.sin(deltaTau)
        for i in range(N):
            #p = p + correctSeparation*rotate(tangent, deltaTau, binormal)
            p = p0 + 2*r*m.sin(i*deltaTau)*rotate(t0, i*deltaTau, binormal)
            result.append(tuple(p))

    return result

def returnLineToLineDistance(a0a1, b0b1):
    """
    Function: returnLineToLineDistance
    ----------------------------------------------------------------------------------------------------------------------------------------------------
    function returns the distance between the line segments a0a1 and b0b1
    a0 three tuple
    a1 three tuple
    b0 three tuple
    b1 three tuple
    return float
    """
    (a0, a1) = a0a1
    (b0, b1) = b0b1
    m1 = (np.array(a0) + np.array(a1))*0.5
    m2 = (np.array(b0) + np.array(b1))*0.5
    e1 = np.array(a1) - np.array(a0)
    L1 = np.linalg.norm(e1)
    e1 = e1/L1
    e2 = np.array(b1) - np.array(b0)
    L2 = np.linalg.norm(e2)
    e2 = e2/L2
    e1dote2 = np.dot(e1, e2)
    if abs(e1dote2)>0.9998:#lines are parallel
        #first calculate the distace between the infinite lines
        Dinf = m.sqrt(np.dot(m2 - m1, m2 - m1) - np.dot(m2 - m1, e1)**2)
        #if the projection of the segements overlap then return Dinf, otherwise return the distance between the closest end points
        t2 = lambda x: np.dot(m1 - m2, e1) + x
        if t2(-0.5*L1)>0.5*L2:
            if e1dote2 > 0:
                return np.linalg.norm(np.array(a1) - np.array(b0))
            else:
                return np.linalg.norm(np.array(a1) - np.array(b1))
        elif t2(0.5*L1)<-0.5*L2:
            if e1dote2 > 0:
                return np.linalg.norm(np.array(a0) - np.array(b1))
            else:
                return np.linalg.norm(np.array(a0) - np.array(b0))
        else:
            return Dinf
    else:
        #first calculate the distace between the infinite lines
        t1 = np.dot(m1 - m2, e1dote2*e2 - e1)/(1 - e1dote2**2)
        t2 = np.dot(m1 - m2, e2 - e1dote2*e1)/(1 - e1dote2**2)
        Dinf = np.linalg.norm(m1 + t1*e1 - m2 - t2*e2)
    
        #calculate the distance in the plane projected along the vector connecting the closest two points
        if (((-t1 - L1*0.5)*(-t1 + L1*0.5))<0)*(((-t2 - L2*0.5)*(-t2 + L2*0.5))<0)>0:#closest point is contained in the line segment
            return Dinf
        else:#closest point will envolve one of the end points
            #find the closest edge point to the points realising the minimal distance of the infinite lines
            t1_ = [-t1 - L1*0.5, -t1 + L1*0.5][np.argmin(np.array([abs(-t1 - L1*0.5), abs(-t1 + L1*0.5)]))]
            t2_ = t1_*e1dote2
            if ((-t2 - L2*0.5 - t2_)*(-t2 + L2*0.5 - t2_))<0:
                f1 = (t1_**2)*(1 - e1dote2**2)
            else:
                t2_ = [-t2 - L2*0.5, -t2 + L2*0.5][np.argmin(np.array([abs(-t2 - L2*0.5 - t2_), abs(-t2 + L2*0.5 - t2_)]))]
                f1 = t1_**2 + t2_**2 - 2*t1_*t2_*e1dote2

            t2_ = [-t2 - L2*0.5, -t2 + L2*0.5][np.argmin(np.array([abs(-t2 - L2*0.5), abs(-t2 + L2*0.5)]))]
            t1_ = t2_*e1dote2
            if ((-t1 - L1*0.5 - t1_)*(-t1 + L1*0.5 - t1_))<0:
                f2 = (t2_**2)*(1 - e1dote2**2)
            else:
                t1_ = [-t1 - L1*0.5, -t1 + L1*0.5][np.argmin(np.array([abs(-t1 - L1*0.5 - t1_), abs(-t1 + L1*0.5 - t1_)]))]
                f2 = t1_**2 + t2_**2 - 2*t1_*t2_*e1dote2
        
            return m.sqrt(Dinf**2 + min(f1, f2))


@njit
def cLineToLineDistance(a, b):
    """
    Function: cLineToLineDistance
    -----------------------------------------------------------------------------------------------------
    function returns the distance between the line segments a and b
    a  2 x 3 numpy.array
    b  2 x 3 numpy.array
    return float
    """

    a_0_0 = a[0,0];
    a_0_1 = a[0,1];
    a_0_2 = a[0,2];
    a_1_0 = a[2,0];
    a_1_1 = a[2,1];
    a_1_2 = a[2,2];    
    
    # compute edge length and the edge unit vector
    u_0    = a_1_0 - a_0_0;
    u_1    = a_1_1 - a_0_1;
    u_2    = a_1_2 - a_0_2;
    K      = np.sqrt( u_0 * u_0 + u_1 * u_1 + u_2 * u_2 );
    K_inv  = 1. / K;
    u_0   *= K_inv;
    u_1   *= K_inv;
    u_2   *= K_inv;
    K_half = 0.5 * K;

    # edge midpoint
    M_0    = 0.5 * (a_0_0 + a_1_0);
    M_1    = 0.5 * (a_0_1 + a_1_1);
    M_2    = 0.5 * (a_0_2 + a_1_2);
    
    b_0_0 = b[0,0];
    b_0_1 = b[0,1];
    b_0_2 = b[0,2];
    b_1_0 = b[2,0];
    b_1_1 = b[2,1];
    b_1_2 = b[2,2];  

    # compute edge length and the edge unit vector            
    v_0    = b_1_0 - b_0_0;
    v_1    = b_1_1 - b_0_1;
    v_2    = b_1_2 - b_0_2;
    L      = np.sqrt( v_0 * v_0 + v_1 * v_1 + v_2 * v_2 );
    L_inv  = 1. / L;
    v_0   *= L_inv;
    v_1   *= L_inv;
    v_2   *= L_inv;
    L_half = 0.5 * L;

    # edge midpoint                
    N_0    = 0.5 * (b_0_0 + b_1_0);
    N_1    = 0.5 * (b_0_1 + b_1_1);
    N_2    = 0.5 * (b_0_2 + b_1_2);

    w_0    = N_0 - M_0;
    w_1    = N_1 - M_1;
    w_2    = N_2 - M_2;

    # compute several scalar products that will be used frequently
    uv     = u_0 * v_0 + u_1 * v_1 + u_2 * v_2;
    uw     = u_0 * w_0 + u_1 * w_1 + u_2 * w_2;
    vw     = v_0 * w_0 + v_1 * w_1 + v_2 * w_2;

    denom  = (1.0 - uv * uv);
    
    if denom < 0.0000000000000001 : #lines are parallel
        #first calculate the distace between the infinite lines
        Dinf = m.sqrt( w_0 * w_0 + w_1 * w_1 + w_2 * w_2 - uw * uw )
        #if the projection of the segements overlap then return Dinf, otherwise return the distance between the closest end points

        if -K_half -uw  > L_half :

            if uv > 0:#lines are oriented
            
                d_0 = a_0_0 - b_1_0;
                d_1 = a_0_1 - b_1_1;
                d_2 = a_0_2 - b_1_2;   
            
                return np.sqrt( d_0 * d_0 + d_1 * d_1 + d_2 * d_2 );
                
            else:
                d_0 = a_0_0 - b_0_0;
                d_1 = a_0_1 - b_0_1;
                d_2 = a_0_2 - b_0_2;   
            
                return np.sqrt( d_0 * d_0 + d_1 * d_1 + d_2 * d_2 );
        
        elif uw - K_half > L_half:
                
            if uv > 0:#lines are oriented

                d_0 = a_1_0 - b_0_0;
                d_1 = a_1_1 - b_0_1;
                d_2 = a_1_2 - b_0_2;   
            
                return np.sqrt( d_0 * d_0 + d_1 * d_1 + d_2 * d_2 );

            else:
                d_0 = a_1_0 - b_1_0;
                d_1 = a_1_1 - b_1_1;
                d_2 = a_1_2 - b_1_2;   
            
                return np.sqrt( d_0 * d_0 + d_1 * d_1 + d_2 * d_2 );
        else:
            return Dinf
    else:
        #first calculate the distace between the infinite lines
        factor = 1. / denom;
        s = (uw - uv * vw) * factor;
        t = (uv * uw - vw) * factor;
        
        d_0 = w_0 - s * u_0 + t * v_0;
        d_1 = w_1 - s * u_1 + t * v_1;
        d_2 = w_2 - s * u_2 + t * v_2;        
        
        Dinf = np.sqrt( d_0 * d_0 + d_1 * d_1 + d_2 * d_2 );
    
        s_0 = -s - K_half;
        s_1 = -s + K_half;
        t_0 = -t - L_half;
        t_1 = -t + L_half;
        
        #calculate the distance in the plane projected along the vector connecting the closest two points
        if ( s_0 * s_1 < 0 ) and ( t_0 * t_1 < 0 ) : #closest point is contained in the line segment
            return Dinf
        else:#closest point will envolve one of the end points
            #find the closest edge point to the points realising the minimal distance of the infinite lines

            s_ = s_0 if abs(s_0) <= abs(s_1) else s_1        
            t_ = s_*uv;
            
            if t_0 < t_ < t_1 :
                
                f1 = (s_ * s_) * denom
                
            else :
                
                t_ = t_0 if abs(t_0 - t_) <= abs(t_1 - t_) else t_1;                
                f1 = s_ * s_ + t_ * t_ - 2.0 * s_ * t_ * uv

            t_ = t_0 if abs(t_0) <= abs(t_1) else t_1;            
            s_ = t_ * uv;
            
            if s_0 < s_ < s_1 :
                
                f2 = (t_ * t_) * denom;
                
            else :
                
                s_ = s_0 if abs(s_0 - s_) <= abs(s_1 - s_) else s_1;                
                f2 = s_ * s_ + t_ * t_ - 2.0 * s_ * t_ * uv;
        
            return np.sqrt( Dinf*Dinf + min(f1, f2) )

def bisectBezierTraingle(b0, b1, b2):
    """
    Function: bisectBezierTriangle
    -------------------------------------------------------------------------------------------------------------------------------------------
    used to bisect the curve once during the simulation
    returns two three tuples which defined the bezier triangles for the two arcs arising from besecting the arc difined by the bzier triangle (b0, b1, b2)
    """
    #find cos(delta) 
    e1 = np.array(b1) - np.array(b0)
    l = np.linalg.norm(e1)
    e = np.array(b2) - np.array(b0)
    D = np.linalg.norm(e)
    omega = np.dot(e1/l, e/D)
    triangle_0 = (b0, tuple((np.array(b0) +  omega*np.array(b1))/(1 + omega)), tuple((np.array(b0) + 2*omega*np.array(b1) + np.array(b2))/(2 + 2*omega)))
    triangle_1 = (tuple((np.array(b0) + 2*omega*np.array(b1) + np.array(b2))/(2 + 2*omega)), tuple((np.array(b2) +  omega*np.array(b1))/(1 + omega)), b2)
    
    return (triangle_0, triangle_1)

#below method used in finding overlaps....?
@njit
def cBisectBezierTraingle(a):
    """
    Function: bisectBezierTriangle
    ----------------------------------------------------------------------------------------------------------------------------------
    returns tuple of (a0, a1) ai is numpy 3 x 3 array
    two arcs arising from besecting the arc difined by the bzier triangle (b0, b1, b2)
    """
    a_0_0 = a[0,0];
    a_0_1 = a[0,1];
    a_0_2 = a[0,2];
    a_1_0 = a[1,0];
    a_1_1 = a[1,1];
    a_1_2 = a[1,2];
    a_2_0 = a[2,0];
    a_2_1 = a[2,1];
    a_2_2 = a[2,2];
    
    e1_0 = a_1_0 - a_0_0
    e1_1 = a_1_1 - a_0_1
    e1_2 = a_1_2 - a_0_2
    
    e_0 = a_2_0 - a_0_0
    e_1 = a_2_1 - a_0_1
    e_2 = a_2_2 - a_0_2
    
    denom = np.sqrt((e_0*e_0 + e_1*e_1 + e_2*e_2)*(e1_0*e1_0 + e1_1*e1_1 + e1_2*e1_2))
    
    e1dote = e_0*e1_0 + e_1*e1_1 + e_2*e1_2
    M = 1./denom
    omega = e1dote*M
    
    #evaluate the new points
    om_plus_1_inv = 1./(1 + omega)
    _2om_plus_2_inv = 1./(2 + 2*omega)
    om_a_1_0 = omega*a_1_0
    om_a_1_1 = omega*a_1_1
    om_a_1_2 = omega*a_1_2
    
    
    #b00 = b0
    
    #b01 = (np.array(b0) +  omega*np.array(b1))/(1 + omega)
    a01_0 = a_0_0 + om_a_1_0
    a01_1 = a_0_1 + om_a_1_1
    a01_2 = a_0_2 + om_a_1_2
    a01_0 *= om_plus_1_inv
    a01_1 *= om_plus_1_inv
    a01_2 *= om_plus_1_inv
    
    #a01 = [a01_0, a01_1, a01_2]
    
    #b02 = (np.array(b0) + 2*omega*np.array(b1) + np.array(b2))/(2 + 2*omega)
    a02_0 = a_0_0 + 2*om_a_1_0 + a_2_0
    a02_1 = a_0_1 + 2*om_a_1_1 + a_2_1
    a02_2 = a_0_2 + 2*om_a_1_2 + a_2_2
    a02_0 *= _2om_plus_2_inv
    a02_1 *= _2om_plus_2_inv
    a02_2 *= _2om_plus_2_inv
    
    #a02 = [a02_0, a02_1, a02_2]

    #b10 = (np.array(b0) + 2*omega*np.array(b1) + np.array(b2))/(2 + 2*omega) = b02
    
    #b11 = (np.array(b2) +  omega*np.array(b1))/(1 + omega)
    a11_0 = a_2_0 + om_a_1_0
    a11_1 = a_2_1 + om_a_1_1
    a11_2 = a_2_2 + om_a_1_2
    a11_0 *= om_plus_1_inv
    a11_1 *= om_plus_1_inv
    a11_2 *= om_plus_1_inv
    
    #a11 = [a11_0, a11_1, a11_2]
    
    #b12 = b2
    
    return ( np.array([a_0_0, a_0_1 , a_0_2, a01_0, a01_1, a01_2, a02_0, a02_1, a02_2]).reshape(3,3), np.array([a02_0, a02_1, a02_2, a11_0, a11_1, a11_2, a_2_0, a_2_1, a_2_2]).reshape(3,3) ) 

def bisectAllTrianglePairsInList(triangleList):
    """
    Function: bisectAllTriangleParsInList
    --------------------------------------------------------------------------------------------------------------------------------------------
    function bisects all the triangle pairs appearing the the triangle list keeping the pairing order
    triangleList list with elements ((a0, a1, a2), (b0, b1, b2))
    return list
    """
    result = []
    for ((a0, a1, a2), (b0, b1, b2)) in triangleList:
        result+=list(itertools.product(list(bisectBezierTraingle(a0, a1, a2)), list(bisectBezierTraingle(b0, b1, b2))))
    return result

def returnCurveToLineErrorFromBezierTriangle(b):
    """
    Function: returnCurveToLineErrorFromBezierTriangle
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    function returns the error associated with the line to line approximation of the arc to arc distance
    (b0, b1, b2) tuple of three tuples
    return float
    """
    (b0, b1, b2) = b
    t0 = np.array(b1) - np.array(b0)
    t0 = t0/np.linalg.norm(t0)
    t1 = np.array(b2) - np.array(b1)
    t1 = t1/np.linalg.norm(t1)
    t0Dott1 = np.clip(np.dot(t0, t1), 0, 1.0)
    omega =  m.sqrt(0.5*(1 + t0Dott1))
    return (omega*m.sqrt((1 - omega)/(1 + omega))*np.linalg.norm(np.array(b0) - np.array(b2)), omega)

def returnLowerBoundInArcToArcDistance(a,b):
    """
    Function: returnLowerBoundInArcToArcDistance
    ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    returns dl2l((a0, a1, a2), (b0, b1, b2)) - error_a - error_b
    (b0, b1, b2) tuple of three tuples
    (a0, a1, a2) tuple of three tuples
    return float
    """
    (a0,a1,a2)=a
    (b0,b1,b2)=b
    error_a, omega_a = returnCurveToLineErrorFromBezierTriangle((a0, a1, a2))
    error_b, omega_b = returnCurveToLineErrorFromBezierTriangle((b0, b1, b2))
    return returnLineToLineDistance((a0, a1), (b0, b1)) - error_a - error_b

def returnArcToArcDistanceForBezierTrianglesWithinErrorBound(b, a, epsilon):
    """
    Function: returnArcToArcDistanceForControlTrianglesWithinErrorBound
    ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    returns the line to line approximation of the arc to arc distance between the arcs defined by the bezier triangles (b0, b1, b2), (a0, a1, a2) up to an error given by epsilon
    (b0, b1, b2) tuple of three tuples
    (a0, a1, a2) tuple of three tuples
    return float
    """
    (a0,a1,a2)=a
    (b0,b1,b2)=b
    #test the distance to decide if the arcs need to be subdivded
    error_p, omega_p = returnCurveToLineErrorFromBezierTriangle((b0, b1, b2))
    error_q, omega_q =  returnCurveToLineErrorFromBezierTriangle((a0, a1, a2))
    d = returnLineToLineDistance((a0, a2), (b0, b2))
    upperBound = d + error_q + error_p
    if error_p + error_q > epsilon:#need to bisect the triangles
        arcPairs = [((a0, a1, a2), (b0, b1, b2))]
        while error_p + error_q > epsilon:
            d = 10.0
            #bisect bezier triangle pairs
            arcPairs = bisectAllTrianglePairsInList(arcPairs)
            #print("number pairs", len(arcPairs))

            #evaluate the new error term
            error_p = error_p/(2 + m.sqrt(2*(1 + omega_p)))
            error_q = error_q/(2 + m.sqrt(2*(1 + omega_q)))
            omega_p = m.sqrt(0.5*(1 + omega_p))
            omega_q = m.sqrt(0.5*(1 + omega_q))
            #print("reduced error ", error_p + error_q, " > ", epsilon)
                
            arcPairsNeedingFurtherBisection = []
            for ((a0, a1, a2), (b0, b1, b2)) in arcPairs:
                dl2l = returnLineToLineDistance((a0, a2), (b0, b2))
                if dl2l - error_p - error_q > upperBound:#don't need to bisect this arc again
                    continue
                arcPairsNeedingFurtherBisection+=[((a0, a1, a2), (b0, b1, b2))]
                if dl2l < d:
                    d = dl2l
            upperBound = d + error_p + error_q
            arcPairs = arcPairsNeedingFurtherBisection

    #return d, error_p + error_q
    return d - error_p - error_q

def distArcToArcWithinBoundForBezierCurve(b, a, closestDistanceBound, epsilon, withInfo = False):
    """
    Function: distArcToArcWithinBoundForBezierCurve
    ----------------------------------------------------------------------------------------------------------------------------------------------
    function returns 1 if the arcs defined by the bezier triangles (b0,b1,b2) and (b0, b1, b2) are closer than closestDistanceBound + epsilon and otherwise 0
    (b0, b1, b2) tuple of three tuples
    (a0, a1, a2) tuple of three tuples
    closestDistanceBound float
    epsilon float
    """
    #returnLineToLineDistance2.counter = 0
    (a0,a1,a2)=a
    (b0,b1,b2)=b
    #test the distance to decide if the arcs need to be subdivded
    error_b, omega_b = returnCurveToLineErrorFromBezierTriangle((b0, b1, b2))
    error_a, omega_a =  returnCurveToLineErrorFromBezierTriangle((a0, a1, a2))
    dl2l = returnLineToLineDistance((a0, a2), (b0, b2))
    if dl2l - error_a - error_b >= closestDistanceBound:
        return 0
    if dl2l + error_a + error_b < closestDistanceBound:#too close reject the move
        return 1
    if error_a+error_b<epsilon:#too close within the tolerence of epsilon
        return 1
    else:#you need to bisect the arcs
        arcPairs = [((a0, a1, a2), (b0, b1, b2))]
        while error_a + error_b > epsilon:
            if withInfo:
                lb = smallestDl2l - error_a - error_b
                print(lb, "<  dArcToArc < ", ub)
                print("further processing ", len(arcPairs), "pairs")
            #bisect bezier triangle pairs
            arcPairs = bisectAllTrianglePairsInList(arcPairs)
            arcPairsNeedingFurtherBisection = arcPairs[::]
            #print("number pairs", len(arcPairs))

            #evaluate the new error term
            error_a = error_a/(2 + m.sqrt(2*(1 + omega_a)))
            error_b = error_b/(2 + m.sqrt(2*(1 + omega_b)))
            omega_p = m.sqrt(0.5*(1 + omega_p))
            omega_q = m.sqrt(0.5*(1 + omega_q))

            for ((a0, a1, a2), (b0, b1, b2)) in arcPairs:
                dl2l = returnLineToLineDistance((a0, a2), (b0, b2))
                if dl2l - error_a - error_a >= closestDistanceBound:#arc pair either far enough apart or further apart than the minimising arc pair
                    arcPairsNeedingFurtherBisection.remove(((a0, a1, a2), (b0, b1, b2)))
                if dl2l + error_a + error_a < closestDistanceBound:#reject move
                    return 1
            arcPairs = arcPairsNeedingFurtherBisection
            if len(arcPairs)==0:
                return 0

        if withInfo:
            for ((a0, a1, a2), (b0, b1, b2)) in arcPairs:
                dl2l = returnLineToLineDistance2((a0, a2), (b0, b2))
                print(dl2l - error_a - error_b, '< dArcToArc <', dl2l + error_a + error_b)
                print('not decernable within error bound')
                
        return 1

@njit
def cDistArcToArcWithinErrorBound(a, b, closestDistanceBound, epsilon):
    
    """
    used in curve_object.check_new_positions_do_not_cause_overlaps for curve_object type biarc
    a 3 x 3 array
    b 3 x 3 numpy array
    """
    #compute errors in the verbose c way
    a_0_0 = a[0,0];
    a_0_1 = a[0,1];
    a_0_2 = a[0,2];
    a_1_0 = a[1,0];
    a_1_1 = a[1,1];
    a_1_2 = a[1,2];
    a_2_0 = a[2,0];
    a_2_1 = a[2,1];
    a_2_2 = a[2,2];
    
    t0_0 = a_1_0 - a_0_0
    t0_1 = a_1_1 - a_0_1
    t0_2 = a_1_2 - a_0_2
    t0_len = np.sqrt(t0_0*t0_0 + t0_1*t0_1 + t0_2*t0_2)
    t0_len_inv = 1./t0_len
    t0_0*=t0_len_inv
    t0_1*=t0_len_inv
    t0_2*=t0_len_inv
    
    t1_0 = a_2_0 - a_1_0
    t1_1 = a_2_1 - a_1_1
    t1_2 = a_2_2 - a_1_2
    t1_len = np.sqrt(t1_0*t1_0 + t1_1*t1_1 + t1_2*t1_2)
    t1_len_inv = 1./t0_len
    t1_0*=t1_len_inv
    t1_1*=t1_len_inv
    t1_2*=t1_len_inv
    
    t0Dott1 = t0_0*t1_0 + t0_1*t1_1 +t0_2*t1_2
    om_a = np.sqrt(min(0.5*(1 + t0Dott1), 0.99999598))
    eps_a = om_a*np.sqrt(((1 - om_a)/(1 + om_a))*((a_0_0 - a_2_0)*(a_0_0 - a_2_0) + (a_0_1 - a_2_1)*(a_0_1 - a_2_1) + (a_0_2 - a_2_2)*(a_0_2 - a_2_2)))
    
    b_0_0 = b[0,0];
    b_0_1 = b[0,1];
    b_0_2 = b[0,2];
    b_1_0 = b[1,0];
    b_1_1 = b[1,1];
    b_1_2 = b[1,2];
    b_2_0 = b[2,0];
    b_2_1 = b[2,1];
    b_2_2 = b[2,2];
    
    t0_0 = b_1_0 - b_0_0
    t0_1 = b_1_1 - b_0_1
    t0_2 = b_1_2 - b_0_2
    t0_len = np.sqrt(t0_0*t0_0 + t0_1*t0_1 + t0_2*t0_2)
    t0_len_inv = 1./t0_len
    t0_0*=t0_len_inv
    t0_1*=t0_len_inv
    t0_2*=t0_len_inv
    
    t1_0 = b_2_0 - b_1_0
    t1_1 = b_2_1 - b_1_1
    t1_2 = b_2_2 - b_1_2
    t1_len = np.sqrt(t1_0*t1_0 + t1_1*t1_1 + t1_2*t1_2)
    t1_len_inv = 1./t0_len
    t1_0*=t1_len_inv
    t1_1*=t1_len_inv
    t1_2*=t1_len_inv
    
    t0Dott1 = t0_0*t1_0 + t0_1*t1_1 +t0_2*t1_2
    om_b = np.sqrt(min(0.5*(1 + t0Dott1), 0.99999598))
    eps_b = om_b*np.sqrt(((1 - om_b)/(1 + om_b))*((b_0_0 - b_2_0)*(b_0_0 - b_2_0) + (b_0_1 - b_2_1)*(b_0_1 - b_2_1) + (b_0_2 - b_2_2)*(b_0_2 - b_2_2)))
    
    #now the depth lifo
    stack=[((a, b), 0)] #inialised the queue
    err_a = [(eps_a, om_a)]
    err_b = [(eps_b, om_b)]

    while len(stack)>0:
        
        ((a,b), depth) = stack.pop()
        
        #first compute the line 2 line distance as in cLineToLineDistance_2
        a_0_0 = a[0,0];
        a_0_1 = a[0,1];
        a_0_2 = a[0,2];
        a_2_0 = a[2,0];
        a_2_1 = a[2,1];
        a_2_2 = a[2,2];    
    
        # compute edge length and the edge unit vector
        u_0    = a_2_0 - a_0_0;
        u_1    = a_2_1 - a_0_1;
        u_2    = a_2_2 - a_0_2;
        K      = np.sqrt( u_0 * u_0 + u_1 * u_1 + u_2 * u_2 );
        K_inv  = 1. / K;
        u_0   *= K_inv;
        u_1   *= K_inv;
        u_2   *= K_inv;
        K_half = 0.5 * K;

        # edge midpoint
        M_0    = 0.5 * (a_0_0 + a_2_0);
        M_1    = 0.5 * (a_0_1 + a_2_1);
        M_2    = 0.5 * (a_0_2 + a_2_2);
    
        b_0_0 = b[0,0];
        b_0_1 = b[0,1];
        b_0_2 = b[0,2];
        b_2_0 = b[2,0];
        b_2_1 = b[2,1];
        b_2_2 = b[2,2];  

        # compute edge length and the edge unit vector            
        v_0    = b_2_0 - b_0_0;
        v_1    = b_2_1 - b_0_1;
        v_2    = b_2_2 - b_0_2;
        L      = np.sqrt( v_0 * v_0 + v_1 * v_1 + v_2 * v_2 );
        L_inv  = 1. / L;
        v_0   *= L_inv;
        v_1   *= L_inv;
        v_2   *= L_inv;
        L_half = 0.5 * L;

        # edge midpoint                
        N_0    = 0.5 * (b_0_0 + b_2_0);
        N_1    = 0.5 * (b_0_1 + b_2_1);
        N_2    = 0.5 * (b_0_2 + b_2_2);

        w_0    = N_0 - M_0;
        w_1    = N_1 - M_1;
        w_2    = N_2 - M_2;

        # compute several scalar products that will be used frequently
        uv     = u_0 * v_0 + u_1 * v_1 + u_2 * v_2;
        uw     = u_0 * w_0 + u_1 * w_1 + u_2 * w_2;
        vw     = v_0 * w_0 + v_1 * w_1 + v_2 * w_2;

        denom  = (1.0 - uv * uv);
    
        if denom < 0.0000000000000001 : #lines are parallel
            #first calculate the distace between the infinite lines
            Dinf = m.sqrt( w_0 * w_0 + w_1 * w_1 + w_2 * w_2 - uw * uw )

            #if the projection of the segements overlap then return Dinf, otherwise return the distance between the closest end points
            if -K_half - uw  > L_half :
                if uv > 0:
            
                    d_0 = a_0_0 - b_2_0;
                    d_1 = a_0_1 - b_2_1;
                    d_2 = a_0_2 - b_2_2;   
            
                    dl2l = np.sqrt( d_0 * d_0 + d_1 * d_1 + d_2 * d_2 );
                
                else:
                    d_0 = a_0_0 - b_0_0;
                    d_1 = a_0_1 - b_0_1;
                    d_2 = a_0_2 - b_0_2;   
            
                    dl2l = np.sqrt( d_0 * d_0 + d_1 * d_1 + d_2 * d_2 );
        
            elif uw - K_half > L_half:
                
                if uv > 0:#lines are oriented
                    d_0 = a_2_0 - b_0_0;
                    d_1 = a_2_1 - b_0_1;
                    d_2 = a_2_2 - b_0_2;   
            
                    dl2l = np.sqrt( d_0 * d_0 + d_1 * d_1 + d_2 * d_2 );

                else:
                    d_0 = a_2_0 - b_2_0;
                    d_1 = a_2_1 - b_2_1;
                    d_2 = a_2_2 - b_2_2;   
            
                    dl2l = np.sqrt( d_0 * d_0 + d_1 * d_1 + d_2 * d_2 );
        
            else:
                dl2l = Dinf
        else:
            #first calculate the distace between the infinite lines
            factor = 1. / denom;
            s = (uw - uv * vw) * factor;
            t = (uv * uw - vw) * factor;
        
            d_0 = w_0 - s * u_0 + t * v_0;
            d_1 = w_1 - s * u_1 + t * v_1;
            d_2 = w_2 - s * u_2 + t * v_2;        
        
            Dinf = np.sqrt( d_0 * d_0 + d_1 * d_1 + d_2 * d_2 );
    
            s_0 = -s - K_half;
            s_1 = -s + K_half;
            t_0 = -t - L_half;
            t_1 = -t + L_half;
        
            #calculate the distance in the plane projected along the vector connecting the closest two points
            if ( s_0 * s_1 < 0 ) and ( t_0 * t_1 < 0 ) : #closest point is contained in the line segment
                 dl2l = Dinf
            else:#closest point will envolve one of the end points
            #find the closest edge point to the points realising the minimal distance of the infinite lines

                s_ = s_0 if abs(s_0) <= abs(s_1) else s_1        
                t_ = s_*uv;
            
                if t_0 < t_ < t_1 :
                
                    f1 = (s_ * s_) * denom
                
                else :
                
                    t_ = t_0 if abs(t_0 - t_) <= abs(t_1 - t_) else t_1;                
                    f1 = s_ * s_ + t_ * t_ - 2.0 * s_ * t_ * uv

                t_ = t_0 if abs(t_0) <= abs(t_1) else t_1;#seems like Henrik has made mistake here, check     
                s_ = t_ * uv;
            
                if s_0 < s_ < s_1 :
                
                    f2 = (t_ * t_) * denom;
                
                else :
                
                    s_ = s_0 if abs(s_0 - s_) <= abs(s_1 - s_) else s_1;                
                    f2 = s_ * s_ + t_ * t_ - 2.0 * s_ * t_ * uv;
        
                dl2l = np.sqrt( Dinf*Dinf + min(f1, f2) )
    
        # update errors in the verbose c way by using depth
        if len(err_a) < depth + 1:#need to update the errors and append them to the list
            tmp1 = np.sqrt(0.5*(1 + err_a[-1][1]))
            tmp0 = err_a[-1][0]*(1./(2 + np.sqrt(2*(1 + err_a[-1][1]))))
            err_a.append((tmp0, tmp1))
            
            tmp1 = np.sqrt(0.5*(1 + err_b[-1][1]))
            tmp0 = err_b[-1][0]*(1./(2 + np.sqrt(2*(1 + err_b[-1][1]))))
            err_b.append((tmp0, tmp1))
        
        error = err_a[depth][0] + err_b[depth][0]

        if dl2l + error < closestDistanceBound:#violates reach constraint
            return 1
        elif dl2l - error < closestDistanceBound:#need to process arc pair further
            if error < epsilon:#violates contraint within accuracy bounds
                return 1
        
            #bisect the arc pair and get four arc pair children
            a_1_0 = a[1,0];
            a_1_1 = a[1,1];
            a_1_2 = a[1,2];
        
            omega = err_a[depth][1]  

            #evaluate the new points
            om_plus_1_inv = 1./(1 + omega)
            _2om_plus_2_inv = 1./(2 + 2*omega)
            om_a_1_0 = omega*a_1_0
            om_a_1_1 = omega*a_1_1
            om_a_1_2 = omega*a_1_2

            #b00 = b0

            #b01 = (np.array(b0) +  omega*np.array(b1))/(1 + omega)
            a01_0 = a_0_0 + om_a_1_0
            a01_1 = a_0_1 + om_a_1_1
            a01_2 = a_0_2 + om_a_1_2
            a01_0 *= om_plus_1_inv
            a01_1 *= om_plus_1_inv
            a01_2 *= om_plus_1_inv
            #a01 = [a01_0, a01_1, a01_2]

            #b02 = (np.array(b0) + 2*omega*np.array(b1) + np.array(b2))/(2 + 2*omega)
            a02_0 = a_0_0 + 2*om_a_1_0 + a_2_0
            a02_1 = a_0_1 + 2*om_a_1_1 + a_2_1
            a02_2 = a_0_2 + 2*om_a_1_2 + a_2_2
            a02_0 *= _2om_plus_2_inv
            a02_1 *= _2om_plus_2_inv
            a02_2 *= _2om_plus_2_inv
            #a02 = [a02_0, a02_1, a02_2]

            #b10 = (np.array(b0) + 2*omega*np.array(b1) + np.array(b2))/(2 + 2*omega) = b02

            #b11 = (np.array(b2) +  omega*np.array(b1))/(1 + omega)
            a11_0 = a_2_0 + om_a_1_0
            a11_1 = a_2_1 + om_a_1_1
            a11_2 = a_2_2 + om_a_1_2
            a11_0 *= om_plus_1_inv
            a11_1 *= om_plus_1_inv
            a11_2 *= om_plus_1_inv
            #a11 = [a11_0, a11_1, a11_2]

            #b12 = b2

            a0 = np.array([a_0_0, a_0_1 , a_0_2, a01_0, a01_1, a01_2, a02_0, a02_1, a02_2]).reshape(3,3)
            a1 = np.array([a02_0, a02_1, a02_2, a11_0, a11_1, a11_2, a_2_0, a_2_1, a_2_2]).reshape(3,3)


            b_1_0 = b[1,0];
            b_1_1 = b[1,1];
            b_1_2 = b[1,2];

            omega = err_b[depth][1]  

            #evaluate the new points
            om_plus_1_inv = 1./(1 + omega)
            _2om_plus_2_inv = 1./(2 + 2*omega)
            om_b_1_0 = omega*b_1_0
            om_b_1_1 = omega*b_1_1
            om_b_1_2 = omega*b_1_2

            #b00 = b0

            #b01 = (np.array(b0) +  omega*np.array(b1))/(1 + omega)
            b01_0 = b_0_0 + om_b_1_0
            b01_1 = b_0_1 + om_b_1_1
            b01_2 = b_0_2 + om_b_1_2
            b01_0 *= om_plus_1_inv
            b01_1 *= om_plus_1_inv
            b01_2 *= om_plus_1_inv
            #a01 = [a01_0, a01_1, a01_2]

            #b02 = (np.array(b0) + 2*omega*np.array(b1) + np.array(b2))/(2 + 2*omega)
            b02_0 = b_0_0 + 2*om_b_1_0 + b_2_0
            b02_1 = b_0_1 + 2*om_b_1_1 + b_2_1
            b02_2 = b_0_2 + 2*om_b_1_2 + b_2_2
            b02_0 *= _2om_plus_2_inv
            b02_1 *= _2om_plus_2_inv
            b02_2 *= _2om_plus_2_inv
            #a02 = [a02_0, a02_1, a02_2]

            #b10 = (np.array(b0) + 2*omega*np.array(b1) + np.array(b2))/(2 + 2*omega) = b02

            #b11 = (np.array(b2) +  omega*np.array(b1))/(1 + omega)
            b11_0 = b_2_0 + om_b_1_0
            b11_1 = b_2_1 + om_b_1_1
            b11_2 = b_2_2 + om_b_1_2
            b11_0 *= om_plus_1_inv
            b11_1 *= om_plus_1_inv
            b11_2 *= om_plus_1_inv
            #a11 = [a11_0, a11_1, a11_2]

            #b12 = b2

            b0 = np.array([b_0_0, b_0_1 , b_0_2, b01_0, b01_1, b01_2, b02_0, b02_1, b02_2]).reshape(3,3)
            b1 = np.array([b02_0, b02_1, b02_2, b11_0, b11_1, b11_2, b_2_0, b_2_1, b_2_2]).reshape(3,3)       

            #append arc pair children to stack [((a002, b002), depth+1) ((a102, b002), depth+1) ...]
            stack.append(((a0, b0), depth+1))
            stack.append(((a0, b1), depth+1))
            stack.append(((a1, b0), depth+1))
            stack.append(((a1, b1), depth+1))
    return 0

@njit
def cReturnDistArcToArcBounds(a, b, epsilon):
    
    """
    a 3 x 3 array
    b 3 x 3 numpy array
    """
    #compute errors in the verbose c way
    a_0_0 = a[0,0];
    a_0_1 = a[0,1];
    a_0_2 = a[0,2];
    a_1_0 = a[1,0];
    a_1_1 = a[1,1];
    a_1_2 = a[1,2];
    a_2_0 = a[2,0];
    a_2_1 = a[2,1];
    a_2_2 = a[2,2];
    
    t0_0 = a_1_0 - a_0_0
    t0_1 = a_1_1 - a_0_1
    t0_2 = a_1_2 - a_0_2
    t0_len = np.sqrt(t0_0*t0_0 + t0_1*t0_1 + t0_2*t0_2)
    t0_len_inv = 1./t0_len
    t0_0*=t0_len_inv
    t0_1*=t0_len_inv
    t0_2*=t0_len_inv
    
    t1_0 = a_2_0 - a_1_0
    t1_1 = a_2_1 - a_1_1
    t1_2 = a_2_2 - a_1_2
    t1_len = np.sqrt(t1_0*t1_0 + t1_1*t1_1 + t1_2*t1_2)
    t1_len_inv = 1./t0_len
    t1_0*=t1_len_inv
    t1_1*=t1_len_inv
    t1_2*=t1_len_inv
    
    t0Dott1 = t0_0*t1_0 + t0_1*t1_1 +t0_2*t1_2
    om_a = np.sqrt(min(0.5*(1 + t0Dott1), 0.99999598))
    eps_a = om_a*np.sqrt(((1 - om_a)/(1 + om_a))*((a_0_0 - a_2_0)*(a_0_0 - a_2_0) + (a_0_1 - a_2_1)*(a_0_1 - a_2_1) + (a_0_2 - a_2_2)*(a_0_2 - a_2_2)))
    
    b_0_0 = b[0,0];
    b_0_1 = b[0,1];
    b_0_2 = b[0,2];
    b_1_0 = b[1,0];
    b_1_1 = b[1,1];
    b_1_2 = b[1,2];
    b_2_0 = b[2,0];
    b_2_1 = b[2,1];
    b_2_2 = b[2,2];
    
    t0_0 = b_1_0 - b_0_0
    t0_1 = b_1_1 - b_0_1
    t0_2 = b_1_2 - b_0_2
    t0_len = np.sqrt(t0_0*t0_0 + t0_1*t0_1 + t0_2*t0_2)
    t0_len_inv = 1./t0_len
    t0_0*=t0_len_inv
    t0_1*=t0_len_inv
    t0_2*=t0_len_inv
    
    t1_0 = b_2_0 - b_1_0
    t1_1 = b_2_1 - b_1_1
    t1_2 = b_2_2 - b_1_2
    t1_len = np.sqrt(t1_0*t1_0 + t1_1*t1_1 + t1_2*t1_2)
    t1_len_inv = 1./t0_len
    t1_0*=t1_len_inv
    t1_1*=t1_len_inv
    t1_2*=t1_len_inv
    
    t0Dott1 = t0_0*t1_0 + t0_1*t1_1 +t0_2*t1_2
    om_b = np.sqrt(min(0.5*(1 + t0Dott1), 0.99999598))
    eps_b = om_b*np.sqrt(((1 - om_b)/(1 + om_b))*((b_0_0 - b_2_0)*(b_0_0 - b_2_0) + (b_0_1 - b_2_1)*(b_0_1 - b_2_1) + (b_0_2 - b_2_2)*(b_0_2 - b_2_2)))
    
    #now the depth info
    stack=[((a, b), 0)] #inialised the stack
    err_a = [(eps_a, om_a)]
    err_b = [(eps_b, om_b)]

    upper_bound = 100.0
    while len(stack)>0:
        
        ((a,b), depth) = stack.pop()
        lower_bound = 0.0 # the lower_bound is not monotonically increasing with the depth
        
        #first compute the line 2 line distance as in cLineToLineDistance_2
        a_0_0 = a[0,0];
        a_0_1 = a[0,1];
        a_0_2 = a[0,2];
        a_2_0 = a[2,0];
        a_2_1 = a[2,1];
        a_2_2 = a[2,2];    
    
        # compute edge length and the edge unit vector
        u_0    = a_2_0 - a_0_0;
        u_1    = a_2_1 - a_0_1;
        u_2    = a_2_2 - a_0_2;
        K      = np.sqrt( u_0 * u_0 + u_1 * u_1 + u_2 * u_2 );
        K_inv  = 1. / K;
        u_0   *= K_inv;
        u_1   *= K_inv;
        u_2   *= K_inv;
        K_half = 0.5 * K;

        # edge midpoint
        M_0    = 0.5 * (a_0_0 + a_2_0);
        M_1    = 0.5 * (a_0_1 + a_2_1);
        M_2    = 0.5 * (a_0_2 + a_2_2);
    
        b_0_0 = b[0,0];
        b_0_1 = b[0,1];
        b_0_2 = b[0,2];
        b_2_0 = b[2,0];
        b_2_1 = b[2,1];
        b_2_2 = b[2,2];  

        # compute edge length and the edge unit vector            
        v_0    = b_2_0 - b_0_0;
        v_1    = b_2_1 - b_0_1;
        v_2    = b_2_2 - b_0_2;
        L      = np.sqrt( v_0 * v_0 + v_1 * v_1 + v_2 * v_2 );
        L_inv  = 1. / L;
        v_0   *= L_inv;
        v_1   *= L_inv;
        v_2   *= L_inv;
        L_half = 0.5 * L;

        # edge midpoint                
        N_0    = 0.5 * (b_0_0 + b_2_0);
        N_1    = 0.5 * (b_0_1 + b_2_1);
        N_2    = 0.5 * (b_0_2 + b_2_2);

        w_0    = N_0 - M_0;
        w_1    = N_1 - M_1;
        w_2    = N_2 - M_2;

        # compute several scalar products that will be used frequently
        uv     = u_0 * v_0 + u_1 * v_1 + u_2 * v_2;
        uw     = u_0 * w_0 + u_1 * w_1 + u_2 * w_2;
        vw     = v_0 * w_0 + v_1 * w_1 + v_2 * w_2;

        denom  = (1.0 - uv * uv);
    
        if denom < 0.0000000000000001 : #lines are parallel
            #first calculate the distace between the infinite lines
            Dinf = m.sqrt( w_0 * w_0 + w_1 * w_1 + w_2 * w_2 - uw * uw )
            
            #if the projection of the segements overlap then return Dinf, otherwise return the distance between the closest end points
            if -K_half - uw  > L_half :
                if uv > 0:
            
                    d_0 = a_0_0 - b_2_0;
                    d_1 = a_0_1 - b_2_1;
                    d_2 = a_0_2 - b_2_2;   
            
                    dl2l = np.sqrt( d_0 * d_0 + d_1 * d_1 + d_2 * d_2 );
                
                else:
                    d_0 = a_0_0 - b_0_0;
                    d_1 = a_0_1 - b_0_1;
                    d_2 = a_0_2 - b_0_2;   
            
                    dl2l = np.sqrt( d_0 * d_0 + d_1 * d_1 + d_2 * d_2 );
        
            elif uw - K_half > L_half:
                
                if uv > 0:#lines are oriented
                    d_0 = a_2_0 - b_0_0;
                    d_1 = a_2_1 - b_0_1;
                    d_2 = a_2_2 - b_0_2;   
            
                    dl2l = np.sqrt( d_0 * d_0 + d_1 * d_1 + d_2 * d_2 );

                else:
                    d_0 = a_2_0 - b_2_0;
                    d_1 = a_2_1 - b_2_1;
                    d_2 = a_2_2 - b_2_2;   
            
                    dl2l = np.sqrt( d_0 * d_0 + d_1 * d_1 + d_2 * d_2 );
        
            else:
                dl2l = Dinf
        else:
            #first calculate the distace between the infinite lines
            factor = 1. / denom;
            s = (uw - uv * vw) * factor;
            t = (uv * uw - vw) * factor;
        
            d_0 = w_0 - s * u_0 + t * v_0;
            d_1 = w_1 - s * u_1 + t * v_1;
            d_2 = w_2 - s * u_2 + t * v_2;        
        
            Dinf = np.sqrt( d_0 * d_0 + d_1 * d_1 + d_2 * d_2 );
    
            s_0 = -s - K_half;
            s_1 = -s + K_half;
            t_0 = -t - L_half;
            t_1 = -t + L_half;
        
            #calculate the distance in the plane projected along the vector connecting the closest two points
            if ( s_0 * s_1 < 0 ) and ( t_0 * t_1 < 0 ) : #closest point is contained in the line segment
                dl2l = Dinf
            else:#closest point will envolve one of the end points
                #find the closest edge point to the points realising the minimal distance of the infinite lines

                s_ = s_0 if abs(s_0) <= abs(s_1) else s_1        
                t_ = s_*uv;
            
                if t_0 < t_ < t_1 :
                
                    f1 = (s_ * s_) * denom
                
                else :
                
                    t_ = t_0 if abs(t_0 - t_) <= abs(t_1 - t_) else t_1;                
                    f1 = s_ * s_ + t_ * t_ - 2.0 * s_ * t_ * uv

                t_ = t_0 if abs(t_0) <= abs(t_1) else t_1;            
                s_ = t_ * uv;
            
                if s_0 < s_ < s_1 :
                
                    f2 = (t_ * t_) * denom;
                
                else :
                
                    s_ = s_0 if abs(s_0 - s_) <= abs(s_1 - s_) else s_1;                
                    f2 = s_ * s_ + t_ * t_ - 2.0 * s_ * t_ * uv;
        
                dl2l = np.sqrt( Dinf*Dinf + min(f1, f2) )
           
        # update errors in the verbose c way by using depth
        if len(err_a) < depth + 1:#need to update the errors and append them to the list
            tmp1 = np.sqrt(0.5*(1 + err_a[-1][1]))
            tmp0 = err_a[-1][0]*(1./(2 + np.sqrt(2*(1 + err_a[-1][1]))))
            err_a.append((tmp0, tmp1))
            
            tmp1 = np.sqrt(0.5*(1 + err_b[-1][1]))
            tmp0 = err_b[-1][0]*(1./(2 + np.sqrt(2*(1 + err_b[-1][1]))))
            err_b.append((tmp0, tmp1))
        
        error = err_a[depth][0] + err_b[depth][0]
        
        #print(depth, dl2l, dl2l - error, dl2l + error)
        if dl2l - error < upper_bound:#arc sub pair to further process
            if upper_bound > dl2l + error:
                upper_bound = dl2l + error
                #print(depth, (lower_bound, upper_bound), "least upper bound")
            if lower_bound < dl2l -error:#take the greatest lower bound of those lower bounds less than the upper bound
                lower_bound = dl2l - error
                #print(depth, (lower_bound, upper_bound), "greatest lower bound")
            if error < epsilon:#evaluate within accuracy bounds
                #print("depth = ", depth, "error", err_a[depth], err_b[depth])
                return (lower_bound, upper_bound)
            
            #bisect the arc pair and get four arc pair children
            a_1_0 = a[1,0];
            a_1_1 = a[1,1];
            a_1_2 = a[1,2];
        
            omega = err_a[depth][1]  

            #evaluate the new points
            om_plus_1_inv = 1./(1 + omega)
            _2om_plus_2_inv = 1./(2 + 2*omega)
            om_a_1_0 = omega*a_1_0
            om_a_1_1 = omega*a_1_1
            om_a_1_2 = omega*a_1_2

            #b00 = b0

            #b01 = (np.array(b0) +  omega*np.array(b1))/(1 + omega)
            a01_0 = a_0_0 + om_a_1_0
            a01_1 = a_0_1 + om_a_1_1
            a01_2 = a_0_2 + om_a_1_2
            a01_0 *= om_plus_1_inv
            a01_1 *= om_plus_1_inv
            a01_2 *= om_plus_1_inv
            #a01 = [a01_0, a01_1, a01_2]

            #b02 = (np.array(b0) + 2*omega*np.array(b1) + np.array(b2))/(2 + 2*omega)
            a02_0 = a_0_0 + 2*om_a_1_0 + a_2_0
            a02_1 = a_0_1 + 2*om_a_1_1 + a_2_1
            a02_2 = a_0_2 + 2*om_a_1_2 + a_2_2
            a02_0 *= _2om_plus_2_inv
            a02_1 *= _2om_plus_2_inv
            a02_2 *= _2om_plus_2_inv
            #a02 = [a02_0, a02_1, a02_2]

            #b10 = (np.array(b0) + 2*omega*np.array(b1) + np.array(b2))/(2 + 2*omega) = b02

            #b11 = (np.array(b2) +  omega*np.array(b1))/(1 + omega)
            a11_0 = a_2_0 + om_a_1_0
            a11_1 = a_2_1 + om_a_1_1
            a11_2 = a_2_2 + om_a_1_2
            a11_0 *= om_plus_1_inv
            a11_1 *= om_plus_1_inv
            a11_2 *= om_plus_1_inv
            #a11 = [a11_0, a11_1, a11_2]

            #b12 = b2

            a0 = np.array([a_0_0, a_0_1 , a_0_2, a01_0, a01_1, a01_2, a02_0, a02_1, a02_2]).reshape(3,3)
            a1 = np.array([a02_0, a02_1, a02_2, a11_0, a11_1, a11_2, a_2_0, a_2_1, a_2_2]).reshape(3,3)


            b_1_0 = b[1,0];
            b_1_1 = b[1,1];
            b_1_2 = b[1,2];

            omega = err_b[depth][1]  

            #evaluate the new points
            om_plus_1_inv = 1./(1 + omega)
            _2om_plus_2_inv = 1./(2 + 2*omega)
            om_b_1_0 = omega*b_1_0
            om_b_1_1 = omega*b_1_1
            om_b_1_2 = omega*b_1_2

            #b00 = b0

            #b01 = (np.array(b0) +  omega*np.array(b1))/(1 + omega)
            b01_0 = b_0_0 + om_b_1_0
            b01_1 = b_0_1 + om_b_1_1
            b01_2 = b_0_2 + om_b_1_2
            b01_0 *= om_plus_1_inv
            b01_1 *= om_plus_1_inv
            b01_2 *= om_plus_1_inv
            #a01 = [a01_0, a01_1, a01_2]

            #b02 = (np.array(b0) + 2*omega*np.array(b1) + np.array(b2))/(2 + 2*omega)
            b02_0 = b_0_0 + 2*om_b_1_0 + b_2_0
            b02_1 = b_0_1 + 2*om_b_1_1 + b_2_1
            b02_2 = b_0_2 + 2*om_b_1_2 + b_2_2
            b02_0 *= _2om_plus_2_inv
            b02_1 *= _2om_plus_2_inv
            b02_2 *= _2om_plus_2_inv
            #a02 = [a02_0, a02_1, a02_2]

            #b10 = (np.array(b0) + 2*omega*np.array(b1) + np.array(b2))/(2 + 2*omega) = b02

            #b11 = (np.array(b2) +  omega*np.array(b1))/(1 + omega)
            b11_0 = b_2_0 + om_b_1_0
            b11_1 = b_2_1 + om_b_1_1
            b11_2 = b_2_2 + om_b_1_2
            b11_0 *= om_plus_1_inv
            b11_1 *= om_plus_1_inv
            b11_2 *= om_plus_1_inv
            #a11 = [a11_0, a11_1, a11_2]

            #b12 = b2

            b0 = np.array([b_0_0, b_0_1 , b_0_2, b01_0, b01_1, b01_2, b02_0, b02_1, b02_2]).reshape(3,3)
            b1 = np.array([b02_0, b02_1, b02_2, b11_0, b11_1, b11_2, b_2_0, b_2_1, b_2_2]).reshape(3,3)       

            #append arc pair children to stack [((a002, b002), depth+1) ((a102, b002), depth+1) ...]
            stack.append(((a0, b0), depth+1))
            stack.append(((a0, b1), depth+1))
            stack.append(((a1, b0), depth+1))
            stack.append(((a1, b1), depth+1))
            
    return (0,0)#falls dieses zurueck gegeben wird is irgendwas schief gegangen.
                
def returnPointTangentDataForBezierTriangle(b):
    """
    Function: returnPointTangentDataForBezierTriangle
    ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    function returns the point tangent data relating to the bezier triangle
    """
    (b0,b1,b2)=b
    t0 = np.array(b1) - np.array(b0)
    t0 = t0/np.linalg.norm(t0)
    t1 = np.array(b2) - np.array(b1)
    t1 = t1/np.linalg.norm(t1)
    return ((b0, t0), (b2, t1))

def returnPointTangentDataTripleForBezierTriangles(a,b,c,d):
    """
    Function: returnPointTangentDataTripleForBezierTriangles
    --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    returns three point tangent data correspoding to the pwc curve defined by the three bezier triangles
    (a0, a1, a2)... tuple of three tuples, data should be matched, a2 = b0, b2=c0, c2=d0
    return tuple of three point tangent data
    """
    (a0,a1,a2)=a
    (b0, b1, b2)=b
    (c0, c1, c2)=c
    (d0,d1,d2)=d
    t0 = np.array(a1) - np.array(a0)
    t0 = t0/np.linalg.norm(t0)
    t = np.array(c1) - np.array(b1)
    t = t/np.linalg.norm(t)
    t1 = np.array(d2) - np.array(d1)
    t1 = t1/np.linalg.norm(t1)
    return (a0, t0), (b2, t), (d2, t1)
    
def returnMatchingPointTangentData(qt0, qt1, gamma):
    """
    Function: returnMatchingPointTangentData
    -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    function returns the matching point and tangent vector of the biarc interpolating the point tangent data pair (q0, t0), (q1, t1) via the parameter gamma in (0, 1)
    """
    (q0, t0)=qt0
    (q1, t1)=qt1
    #needed floats
    q0Bar = np.array(q0)
    q1Bar = np.array(q1)
    d = np.array(q1Bar) - np.array(q0Bar)
    dSquared = np.dot(d,d)
    t0dotd = np.dot(t0, d)
    t1dotd = np.dot(t1, d)

    #gammaBar
    gammaBar = (2*(1- gamma)*t0dotd*t1dotd)/(gamma*(1-np.dot(t0,t1))*dSquared +  2*t0dotd*t1dotd)
    
    #matchingpoint
    mBar = (gammaBar*t0dotd*q0Bar + gamma*t1dotd*q1Bar)/(gamma*t1dotd + gammaBar*t0dotd) + 0.5*(gamma*gammaBar*dSquared*(t0-t1))/(gamma*t1dotd + gammaBar*t0dotd)
    m = tuple(mBar)

    #tangent vector
    tangent = q1Bar - (gammaBar*0.5*dSquared/t1dotd)*t1 -  q0Bar - (gamma*0.5*dSquared/t0dotd)*t0
    tangent = tangent/np.linalg.norm(tangent)
    
    return (m, tangent)

def returnBezierTriangleForMatchedPointTangentData(qt0, qt1):
    """
    Function: returnBezierTriangleForMatchedPointTangentData
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    function returns the bezier triangle (b0, b1, b2) corresponding to the point tangent data ((q0, t0), (q1, t1))
    position of the bezier triangle are given with respect to the standard frame
    (q,t) point tangent data, point is given with repect to the periodic frame
    return tuple of three tuples
    """
    (q0, t0)=qt0
    (q1, t1)=qt1
    b0 = q0
    b2 = q1
    e = np.array(b2) - np.array(b0)
    D = np.linalg.norm(e)
    if D<0.0001:
        print("curvature what?", (b0, b2), (q0, t1), (q0, t1))
        quit()
    e = e/D
    n = np.cross(t1, t0)
    sinAlpha = np.linalg.norm(n)
    if sinAlpha < 0.0001:#all lies on a line
        return (b0, tuple(0.5*(np.array(b0) + np.array(b2))), b2)
    else:
        n = n/sinAlpha
        delta = 0.5*np.arcsin(sinAlpha)
        b1 = tuple(0.5*(np.array(b0) + np.array(b2)) + 0.5*D*m.tan(delta)*np.cross(n,e))
        return (b0, b1, b2)

def doesNotContainACriticalPair(a, b):
    """
    Function: doesNotContainACriticalPair
    --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    If the arcs defined by the bezier triangles (a0,a1,a2), (b0,b0,b1) pass the single critical point test given in cor.7.8 then the value 1 is returned 
    Otherwise 0 is returned which means that the arcs could contain a single critical pair.
    (a0, a1, a2) three tuple of three tuples
    (b0, b1, b2) three tuple of three tuples
    return 0 or 1
    """
    (a0,a1,a2)=a
    (b0, b1, b2)=b
    containedInDisjointBallsConstraint = lambda x, y, u, v : (np.linalg.norm(np.array(x) + np.array(y) - np.array(u) - np.array(v)) > np.linalg.norm(np.array(x) - np.array(y)) + np.linalg.norm(np.array(u) - np.array(v)))
    sinLambda = lambda x, y, u, v : (np.linalg.norm(np.array(x) - np.array(y)) + np.linalg.norm(np.array(u) - np.array(v)))/np.linalg.norm(np.array(x) + np.array(y) - np.array(u) - np.array(v))
    wVector = lambda x, y, u, v : (np.array(x) + np.array(y) - np.array(u) - np.array(v))/np.linalg.norm(np.array(x) + np.array(y) - np.array(u) - np.array(v))
    if containedInDisjointBallsConstraint(a0, a2, b0, b2):#should be able to check the criterium
        t0 = np.array(a1) - np.array(a0)
        t0 = t0/np.linalg.norm(t0)
        t1 = np.array(a2) - np.array(a1)
        t1 = t1/np.linalg.norm(t1)
        t0Bar = np.array(b1) - np.array(b0)
        t0Bar = t0Bar/np.linalg.norm(t0Bar)
        t1Bar = np.array(b2) - np.array(b1)
        t1Bar = t1Bar/np.linalg.norm(t1Bar)
        s = sinLambda(a0, a2, b0, b2)
        w = wVector(a0, a2, b0, b2)
        if (np.dot(w,t0)>s)*(np.dot(w,t1)>s):#no single crtical point
            return 1
        if (np.dot(w,t0)<-s)*(np.dot(w,t1)<-s):#no single crtical point
            return 1
        if (np.dot(w,t0Bar)>s)*(np.dot(w,t1Bar)>s):#no single crtical point
            return 1
        if (np.dot(w,t0Bar)<-s)*(np.dot(w,t1Bar)<-s):#no single crtical point
            return 1
    return 0

def doesNotContainASingleCriticalPoint(a0, a2, b):
    """
    Function: doesNotContainASingleCriticalPair
    --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    If the arcs defined by the bezier triangles (a0,a1,a2), (b0,b0,b1) pass the single critical point test given in cor.7.8 then the value 1 is returned 
    Otherwise 0 is returned which means that the arcs could contain a single critical pair.
    a0 three tuple
    a2 three tuple
    (b0, b1, b2) three tuple of three tuples
    return 0 or 1
    """
    (b0, b1, b2)=b
    if (np.linalg.norm(np.array(a0) + np.array(a2) - np.array(b0) - np.array(b2)) > np.linalg.norm(np.array(a0) - np.array(a2)) + np.linalg.norm(np.array(b0) - np.array(b2))):
        t0 = np.array(b1) - np.array(b0)
        t0 = t0/np.linalg.norm(t0)
        t1 = np.array(b2) - np.array(b1)
        t1 = t1/np.linalg.norm(t1)
        s = (np.linalg.norm(np.array(a0) - np.array(a2)) + np.linalg.norm(np.array(b0) - np.array(b2)))/np.linalg.norm(np.array(a0) + np.array(a2) - np.array(b0) - np.array(b2))
        w = (np.array(a0) + np.array(a2) - np.array(b0) - np.array(b2))/np.linalg.norm(np.array(a0) + np.array(a2) - np.array(b0) - np.array(b2))
        if (np.dot(w,t0)>s)*(np.dot(w,t1)>s):#no single crtical point
            return 1
        if (np.dot(w,t0)<-s)*(np.dot(w,t1)<-s):#no single crtical point
            return 1
    return 0

#def distanceFunctionRestrictedBetweenTheTwoArcsDoesNotHaveACritcialPointWithRespectToArcB((a0,a1,a2), (b0,b1,b2)):
#   """
#   Function: distanceFunctionRestrictedBetweenTheTwoArcsDoesNotHaveACriticalPointWithRespectToArcB
#   --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#   If the arcs defined by the bezier triangles (a0,a1,a2), (b0,b0,b1) pass the single critical point test given in cor.7.8 then the value 1 is returned 
#   Otherwise 0 is returned which means that the distance function restricted to the two arcs may contain a single critical point with respect to an argument of arc b
#   (a0, a1, a2) three tuple of three tuples
#   (b0, b1, b2) three tuple of three tuples
#   return 0 or 1
#   """
#   if (np.linalg.norm(np.array(a0) + np.array(a2) - np.array(b0) - np.array(b2)) > np.linalg.norm(np.array(a0) - np.array(a2)) + np.linalg.norm(np.array(b0) - np.array(b2))):
#       t0Bar = np.array(b1) - np.array(b0)
#        t0Bar = t0Bar/np.linalg.norm(t0Bar)
#       t1Bar = np.array(b2) - np.array(b1)
#       t1Bar = t1Bar/np.linalg.norm(t1Bar)
#       s = (np.linalg.norm(np.array(a0) - np.array(a2)) + np.linalg.norm(np.array(b0) - np.array(b2)))/np.linalg.norm(np.array(a0) + np.array(a2) - np.array(b0) - np.array(b2))
#       w = (np.array(a0) + np.array(a2) - np.array(b0) - np.array(b2))/np.linalg.norm(np.array(a0) + np.array(a2) - np.array(b0) - np.array(b2))
#       if (np.dot(w,t0Bar)>s)*(np.dot(w,t1Bar)>s):#no single crtical point
#           return 1
#       if (np.dot(w,t0Bar)<-s)*(np.dot(w,t1Bar)<-s):#no single crtical point
#           return 1
#   return 0

def arcPairMayContainCriticalPointAndDistanceIsCloserThanErrorToLowerBound(a, b,  lowerBound, error):
    """
    Function: arcPairMayContainCriticalPointAndDistanceIsCloserThanErrorToLowerBound
    -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    function checks if a pair in the list [((a0, a1, a2),(b0,b1, b2)), ...] contains a double critical point if so it is removed from the list
    the distance bounds between each pair in the list is evaluated 
    if dl2l - error_a - error_b > lowerBound then the pair is removed from the list
    if dl2l + error_a + error_b < lowerBound then (dl2l - error_a - error_b) is returned
    if list empty return 0
    list is rediscretised if nothing is returned, loop continues while error_a + error_b > error
    if error_a + error_b < error and list is not empty the smallest lower bound of the list is returned
    (a0, a1, a2) tuple of three tuples
    (b0, b1, b2) tuple of three tuples
    lowerBound float
    error float
    return float
    """
    (a0,a1,a2)=a
    (b0, b1, b2)=b
    error_a, omega_a = returnCurveToLineErrorFromBezierTriangle((a0, a1, a2))
    error_b, omega_b =  returnCurveToLineErrorFromBezierTriangle((b0, b1, b2))
    if doesNotContainACriticalPair((a0,a1,a2), (b0,b1,b2)):
        return 0
    trianglePairList = [((a0, a1, a2), (b0, b1, b2))]
    while error < error_a + error_b:
        trianglePairsToBisect = []
        while len(trianglePairList)>0:
            ((a0, a1, a2), (b0, b1, b2))=trianglePairList.pop()
            if doesNotContainACriticalPair((a0,a1,a2), (b0,b1,b2)):
                continue
            dl2l = returnLineToLineDistance((a0, a2), (b0, b2))
            #if dl2l + error_a + error_b < lowerBound: #can't do this because you may be dealing with an arc pair that does not contain a critical pair but is not bisected enough for this to be detected
            #return dl2l - error_a - error_b
            if dl2l - error_a - error_b > lowerBound:
                continue
            trianglePairsToBisect.append(((a0, a1, a2), (b0, b1, b2)))
            if len(trianglePairsToBisect)==0:
                return 0
            trianglePairList = bisectAllTrianglePairsInList(trianglePairsToBisect)
        #evaluate the new error term
        error_a = error_a/(2 + m.sqrt(2*(1 + omega_a)))
        error_b = error_b/(2 + m.sqrt(2*(1 + omega_b)))
        #print(error_a + error_b, len(trianglePairList))

    d = lowerBound
    for ((a0, a1, a2), (b0, b1, b2)) in trianglePairList:
        dl2l = returnLineToLineDistance((a0, a2), (b0, b2))
        if dl2l - error_a - error_a < d:
            d = dl2l - error_a - error_a
    if d==lowerBound:
        return 0
    else:
        return d        


