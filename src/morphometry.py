import ctypes
import numpy as np

# create the library with
# g++ -Wall  -fPIC morphometry.cpp -shared -Wl,-soname,libmorphometry.so -o libmorphometry.so
#/opt/homebrew/bin/g++-14 -Wall  -fPIC morphometry.cpp -shared -Wl,-install_name,libmorphometry.so -o libmorphometry.so

lib = ctypes.CDLL('./libmorphometry.so')

lib.pythonentry.argtypes=(ctypes.c_int, ctypes.POINTER(ctypes.c_double) , ctypes.POINTER(ctypes.c_double) , ctypes.POINTER(ctypes.c_double) , ctypes.POINTER(ctypes.c_double) , ctypes.POINTER(ctypes.c_double) )
lib.pythonentry.restype = ctypes.c_int


def morph_(pointList, input_R):
    """Function: morph
       pointList centres of balls i.e. concantenation of curve_vertices the form [np.array(p0_x, p0_y, p0_z), np.array(p1_x, p1_y, p1_z), np.array(p2_x, p2_y, p2_z), ....]  
       input_R postive float ball radius
    """ 
    global lib
    
    n=len(pointList)
    cx=np.array([pointList[i][0] for i in range(n)])
    cy=np.array([pointList[i][1] for i in range(n)])
    cz=np.array([pointList[i][2] for i in range(n)])
    w=input_R*np.ones(n)

    p_cx=ctypes.cast( cx.ctypes.data , ctypes.POINTER(ctypes.c_double))
    p_cy=ctypes.cast( cy.ctypes.data , ctypes.POINTER(ctypes.c_double))
    p_cz=ctypes.cast( cz.ctypes.data , ctypes.POINTER(ctypes.c_double))
    p_w=ctypes.cast( w.ctypes.data , ctypes.POINTER(ctypes.c_double))

    vec_res=np.zeros(4)
    p_res=ctypes.cast(vec_res.ctypes.data , ctypes.POINTER(ctypes.c_double))

    r=lib.pythonentry(ctypes.c_int(n),p_res,p_cx,p_cy,p_cz,p_w)
    return vec_res

