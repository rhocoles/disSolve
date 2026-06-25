import math as m
from geometryClass import ThreadedBeads


#define coefficents which define the energy, define the input sphere radius
edgeLength = 0.25
rTube = 1.0
#overlapRatio = float(sys.argv[3]) #change this in the submit file 
#eta = float(sys.argv[4])

ThreadedBeads.set_radii(0.1, rTube, edgeLength)
ThreadedBeads.set_coefficients(eta=0.125)
print(ThreadedBeads.prefactors)

#define the geometry
#beadedCurve = ThreadedBeads(1, fileName='test2__31428.txt')
beadedCurve = ThreadedBeads(1, fileName='packed_helix.txt')
beadedCurve.evaluate_embedded_measures()
beadedCurve.evaluate_measures()

print(beadedCurve.V_0, beadedCurve.A_0, beadedCurve.C_0, beadedCurve.X_0)
print(beadedCurve.V, beadedCurve.A, beadedCurve.C, beadedCurve.X)
print("Initialised curve of length", beadedCurve.length, "(E - E0)/L = ", beadedCurve.evaluate_normalised_energy(), beadedCurve.evaluate_normalised_energy()*((4*m.pi)/3), "(minRads, minSelfDist) = ", beadedCurve.check_reach(), beadedCurve.evaluate_energy())

