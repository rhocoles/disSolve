import math as m
import geometryClass as geoClass

def read_poly_file(full_path_poly):
    with open(full_path_poly, 'r') as f:
        lines = f.read().splitlines()

    # split into POINTS and POLYS sections
    point_lines = lines[lines.index('POINTS') + 1: lines.index('POLYS')]
    strand_lines = lines[lines.index('POLYS') + 1: lines.index('END')]

    #points into a dict
    points = {}
    for line in point_lines:
        idx, rest = line.split(': ')
        coords = list(map(float,  rest.split(' c(')[0].split(' ')))
        points[int(idx)] = coords

    #points into strand
    curve = []
    configType = []
    for line in strand_lines:
        #print(line.split(': ')[1].strip().split(' '))
        indices = list(map(int, line.split(': ')[1].strip().split(' ')))
        if indices[0] == indices[-1]:
            configType.append('closed')
            indices = indices[:-1]
        else:
            configType.append('open')
        curve.append([points[i] for i in indices])

    numberOfBalls = sum(len(c) for c in curve)

    return curve, numberOfBalls, configType

#define coefficents which define the energy, define the input sphere radius
overlapRatio = 0.1
eta = 0.05

#define the geometry
curveData, numberOfBalls, configType = read_poly_file('test_0_3000.poly')
geometry = geoClass.TubularGeometry(overlapRatio, eta, geoClass.ThreadedBeads(1, curveData=curveData, edgeLength=0.25))
geometry.evaluate_embedded_measures()
geometry.evaluate_measures()
print(geometry.coefficients)

print(geometry.V_0, geometry.A_0, geometry.C_0, geometry.X_0)
print(geometry.V, geometry.A, geometry.C, geometry.X)
print("Initialised curve of length", geometry.curve_object.length, "(E - E0)/L = ", geometry.evaluate_normalised_energy(), geometry.evaluate_normalised_energy()*((4*m.pi)/3), "(minRads, minSelfDist) = ", geometry.curve_object.check_reach())

