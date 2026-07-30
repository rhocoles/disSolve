import numpy as np

def generate_closest_distances(data, configType, skippedInteger, withInfo=False):
    """Creates list of floats. Each float is the closest distance to another point on the polygonal curve separated by at least skippedInteger steps."""

    if withInfo:
        print('\n')
    result = []
    tmpList = []
    for i in range(len(data)):
        for j in range(len(data[i])):
            d = 1000.0
            ind = (0, 0)
            for l in range(i, len(data)):
                stInd = j if l == i else 0
                for k in range(stInd, len(data[l])):
                    if configType == 'open':
                        if l == i and (k - j < skippedInteger):
                            continue
                    else:
                        if l == i and ((k - j < skippedInteger) or (len(data[i]) - k < skippedInteger - j)):
                            continue
                    dtmp = np.linalg.norm(data[i][j] - data[l][k])
                    if dtmp < d:
                        d = dtmp
                        ind = (l, k)
                if d < 1000.0:
                    tmpList.append([(i, j), ind, d])
            result.append(d)
    if withInfo:
        tmpList.sort(key=lambda x: x[2])
        for entry in tmpList:
            print(entry)
        print('\n')
    return result

def is_arc_adjacent(i, j, l, k, n, configType, skippedInteger):
    """
    Function: is_arc_adjacent
    ---------------------------------------
    returns True is (i,j) and (l,k) belong to the same strand (i=l) and indices j and k are within a skipped integer along the strand of length n
    if strand is open this is abs(j-k) < skippedInteger
    if strand is closed this is min( abs(j-k), n - (j-k))
    otherwise return False
    """
    if i != l:
        return False
    d = abs(j - k)
    if configType == 'open':
        return d < skippedInteger
    else:
        return min(d, n - d) < skippedInteger

def is_not_in_A(i, j, A):
    for (l, k) in A:
        if l == i and k == j:
            return False
    return True

def initialise_stack(data, configType, skippedInteger, indices, newPositions):
    """
    Constructs the initial BVH stack for a given index interval.
    Returns a list of ((P, P_positions, (Q, Q_Positions)) pairs of sets to be checked for overlap.
    P, Q index set of the for [(i,j), (i, j+1), ...]
    P, Q are disjoint subsets with no arc-adjacency, all vertices in P are not within a skipped integer of those in Q
    P_positions, Q_positions (x,y,z) coordinates of points in P, Q. Convention, P is moved set, Q is unmoved
    Can be precomputed once per interval at simulation init.
    indices: full interval including fixed endpoints
    newPostions, list of (x,y,z) coordinates 
    """
    strand = [len(data[i]) for i in range(len(data))]
    A = indices[1:-1:]
    i_A = A[0][0]
    n = strand[i_A]

    B = []
    for i in range(len(strand)):
        for j in range(strand[i]):
            if is_not_in_A(i, j, A):
                B.append((i, j))

    B_prime = []
    B_prime_positions = []
    B_complement = []
    for (i, j) in B:
        if is_arc_adjacent(i, j, i_A, A[0][1], n, configType, skippedInteger) or \
           is_arc_adjacent(i, j, i_A, A[-1][1], n, configType, skippedInteger):
            B_complement.append((i, j))
        else:
            B_prime.append((i, j))
            B_prime_positions.append(data[i][j])

    stack = []
    for (i, j) in B_complement:
        A_b = []
        A_b_positions = []
        for s in range(len(A)):
            #(l,k) = s
            if not is_arc_adjacent(i, j, A[s][0], A[s][1], n, configType, skippedInteger):   #strand of (i,j) is of length n since (i,j) is arc adjacent to A i.e. i==l==i_A
                A_b.append(A[s])
                A_b_positions.append(newPositions[s]) 
        if len(A_b) > 0:
            stack.append(((A_b, A_b_positions), ([(i,j)], [data[i][j]])))
    stack.append(((A, newPositions), (B_prime, B_prime_positions)))

    return stack

def closest_distance_and_pair_new(data, configType, skippedInteger, newPositions, indices):
    print("Set A=", indices[1:-1:])
    stack = initialise_stack(data, configType, skippedInteger, indices, newPositions)
    d = 1000.0
    closest_pair = None
    while len(stack) > 0:
        ((P, P_positions), (Q, Q_positions)) = stack.pop()
        for s in range(len(P_positions)):
            for t in range(len(Q_positions)):
                dist = np.linalg.norm(P_positions[s] - Q_positions[t])
                #print(P[s], Q[t], "at distance ", dist)
                if dist < d:
                    d = dist
                    closest_pair = (P[s], Q[t])
    return (d, closest_pair)

#Note re. C++ implementation: computing min max simultaneously for x,y,z, see second method
def aabb(positions):
    xmin = min([p[0] for p in positions])
    ymin = min([p[1] for p in positions])
    zmin = min([p[2] for p in positions])
    xmax = max([p[0] for p in positions])
    ymax = max([p[1] for p in positions])
    zmax = max([p[2] for p in positions])
    return (xmin, ymin, zmin, xmax, ymax, zmax)

def aabbs_overlap(box_p, box_q, upper_bound_closest_self_distance):
    """
        box_p = (xpmin, ypmin, zpmin, xpmax, ypmax, zpmax)
        box_q = (xqmin, yqmin, zqmin, xqmax, yqmax, zqmax)
        box_p and box_q intersect if xpmin < xqmax and  xqmin < xpmax and yqmin < ypmax and ypmin < ypmax and zpmin < zpmax and zqmin < zqmax   
        box_p_extended_by_radius_r and box_q intersect_extended_by_radius_r intersect if xpmin - xqmax < upper_bound_closest_self_distance etc
        return true false
    """
    return ((box_p[0] - box_q[3] < upper_bound_closest_self_distance) and (box_q[0] - box_p[3] < upper_bound_closest_self_distance) and 
            (box_p[1] - box_q[4] < upper_bound_closest_self_distance) and (box_q[1] - box_p[4] < upper_bound_closest_self_distance) and
            (box_p[2] - box_q[5] < upper_bound_closest_self_distance) and (box_q[2] - box_p[5] < upper_bound_closest_self_distance))


def subdivide_box_along_longest_edge(box,  index_set_and_coords):

    (index_set, point_coords) = index_set_and_coords
    n = len(index_set)
    (xmin, ymin, zmin, xmax, ymax, zmax) = box

    axis = [xmax - xmin, ymax - ymin, zmax - zmin].index(max([xmax - xmin, ymax - ymin, zmax - zmin]))
    order_along_axis = sorted(range(n), key = lambda i : point_coords[i][axis])
    index_set_left_child = [index_set[i] for i in order_along_axis[:n//2:]]
    point_coords_left_child = [point_coords[i] for i in order_along_axis[:n//2:]]
    index_set_right_child = [index_set[i] for i in order_along_axis[n//2::]]
    point_coords_right_child = [point_coords[i] for i in order_along_axis[n//2::]]
    
    return (index_set_left_child, point_coords_left_child), (index_set_right_child, point_coords_right_child)

def check_new_positions_do_not_cause_overlaps(data, configType, skippedInteger, newPositions, indices, upper_bound_closest_self_distance):
    """
    Returns 1 if no overlap, 0 if overlap detected.
    data: nested list of numpy arrays (curve_obj.data)
    configType: 'open' or 'closed'
    skippedInteger: int
    newPositions: list of numpy arrays, positions of indices[1:-1:]
    indices: full interval including fixed endpoints
    upper_bound_closest_self_distance: threshold for overlap detection
    """
    stack = initialise_stack(data, configType, skippedInteger, indices, newPositions)
    
    while len(stack) > 0:
        ((P, P_positions), (Q, Q_positions)) = stack.pop() #NOTE: P is the moved set of vertices, Q is the original positons
        if len(P)==len(Q)==1:
            dist = np.linalg.norm(P_positions[0] - Q_positions[0])
            if dist < upper_bound_closest_self_distance:
                return 0
            else:
                continue
        box_P = aabb(P_positions)
        box_Q = aabb(Q_positions)

        if aabbs_overlap(box_P, box_Q, upper_bound_closest_self_distance):
            if len(P) == 1: #split Q along longest axis
                Q_l, Q_r = subdivide_box_along_longest_edge(box_Q, (Q, Q_positions))    
                stack.append( ( (P, P_positions), Q_l))
                stack.append( ( (P, P_positions), Q_r))
            elif len(Q) == 1: #split P along longest axis
                P_l, P_r = subdivide_box_along_longest_edge(box_P, (P, P_positions))    
                stack.append( (P_l, (Q, Q_positions) ) )
                stack.append( (P_r, (Q, Q_positions) ) )
            elif max([(box_P[3] - box_P[0]), (box_P[4] - box_P[1]), (box_P[5] - box_P[2])]) > max([(box_Q[3] - box_Q[0]), (box_Q[4] - box_Q[1]), (box_Q[5] - box_Q[2])]): #split along P as box has longest edge
                P_l, P_r = subdivide_box_along_longest_edge(box_P, (P, P_positions))    
                stack.append( (P_l, (Q, Q_positions) ) )
                stack.append( (P_r, (Q, Q_positions) ) )
            else: #split along Q
                Q_l, Q_r = subdivide_box_along_longest_edge(box_Q, (Q, Q_positions))    
                stack.append( ( (P, P_positions), Q_l))
                stack.append( ( (P, P_positions), Q_r))
                
    return 1
