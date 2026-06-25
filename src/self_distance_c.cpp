/*
 * self_distance_c.cpp
 *
 * C++ implementation of the BVH overlap detection from self_distance.py.
 * The single entry point exposed to Python is:
 *
 *   int check_new_positions_do_not_cause_overlaps(
 *       int    total_points,       // total number of points across all strands
 *       int    num_strands,        // number of strands
 *       int   *strand_lengths,     // length of each strand
 *       double *all_data_x,        // x-coords of all data points (strand-major order)
 *       double *all_data_y,
 *       double *all_data_z,
 *       int    num_A,              // number of interior moved points (len(indices[1:-1:]))
 *       int   *A_strand,           // strand index for each point in A
 *       int   *A_j,                // vertex index for each point in A
 *       double *new_positions_x,   // new x-coords for each point in A
 *       double *new_positions_y,
 *       double *new_positions_z,
 *       int    A_first_strand,     // strand index of indices[0] (fixed endpoint)
 *       int    A_first_j,          // vertex index of indices[0]
 *       int    A_last_strand,      // strand index of indices[-1] (fixed endpoint)
 *       int    A_last_j,           // vertex index of indices[-1]
 *       int    config_closed,      // 1 = closed, 0 = open
 *       int    skippedInteger,     // arc-adjacency threshold
 *       double upper_bound         // overlap threshold
 *   );
 *
 * Returns 1 if no overlap detected, 0 if overlap detected.
 *
 * Build with:
 *   clang++ -O3 -Wall -dynamiclib self_distance_c.cpp -o libself_distance_c.so
 */

#include <cmath>
#include <vector>
#include <algorithm>
#include <cstring>

// --------------------------------------------------------------------------
// Point: a 3D coordinate
// --------------------------------------------------------------------------
struct Point3 {
    double x, y, z;
};

// --------------------------------------------------------------------------
// IndexedPoint: a point together with its (strand, vertex) identity
// --------------------------------------------------------------------------
struct IndexedPoint {
    int strand;
    int j;
    Point3 pos;
};

// --------------------------------------------------------------------------
// is_arc_adjacent
//
// Returns true if (strand_i, j) and (strand_l, k) are arc-adjacent.
// Two points are arc-adjacent only when they are on the same strand (i == l).
// For an OPEN strand:   abs(j - k) < skippedInteger
// For a CLOSED strand:  min(abs(j-k), n - abs(j-k)) < skippedInteger
// --------------------------------------------------------------------------
static inline bool is_arc_adjacent(int strand_i, int j,
                                   int strand_l, int k,
                                   int n,
                                   bool closed,
                                   int skippedInteger)
{
    if (strand_i != strand_l) return false;
    int d = std::abs(j - k);
    if (!closed) {
        return d < skippedInteger;
    } else {
        return std::min(d, n - d) < skippedInteger;
    }
}

// --------------------------------------------------------------------------
// AABB: axis-aligned bounding box
// --------------------------------------------------------------------------
struct AABB {
    double xmin, ymin, zmin;
    double xmax, ymax, zmax;
};

static AABB compute_aabb(const std::vector<IndexedPoint> &pts)
{
    AABB box;
    box.xmin = box.xmax = pts[0].pos.x;
    box.ymin = box.ymax = pts[0].pos.y;
    box.zmin = box.zmax = pts[0].pos.z;
    for (size_t i = 1; i < pts.size(); ++i) {
        if (pts[i].pos.x < box.xmin) box.xmin = pts[i].pos.x;
        if (pts[i].pos.x > box.xmax) box.xmax = pts[i].pos.x;
        if (pts[i].pos.y < box.ymin) box.ymin = pts[i].pos.y;
        if (pts[i].pos.y > box.ymax) box.ymax = pts[i].pos.y;
        if (pts[i].pos.z < box.zmin) box.zmin = pts[i].pos.z;
        if (pts[i].pos.z > box.zmax) box.zmax = pts[i].pos.z;
    }
    return box;
}

// aabbs_overlap: two AABBs (each expanded by upper_bound/2 notionally)
// overlap if the gap between them in every axis is < upper_bound.
// Matches Python: (box_p[0] - box_q[3] < upper_bound) etc.
static inline bool aabbs_overlap(const AABB &p, const AABB &q, double ub)
{
    return (p.xmin - q.xmax < ub) && (q.xmin - p.xmax < ub) &&
           (p.ymin - q.ymax < ub) && (q.ymin - p.ymax < ub) &&
           (p.zmin - q.zmax < ub) && (q.zmin - p.zmax < ub);
}

// --------------------------------------------------------------------------
// subdivide_box_along_longest_edge
//
// Split a set of IndexedPoints along the longest axis of their AABB,
// placing the lower half (by coordinate) in left and the upper half in right.
// Matches Python: order_along_axis, left = [:n//2], right = [n//2:].
// --------------------------------------------------------------------------
static void subdivide(const AABB &box,
                      const std::vector<IndexedPoint> &pts,
                      std::vector<IndexedPoint> &left,
                      std::vector<IndexedPoint> &right)
{
    double dx = box.xmax - box.xmin;
    double dy = box.ymax - box.ymin;
    double dz = box.zmax - box.zmin;

    int axis = 0;
    if (dy >= dx && dy >= dz) axis = 1;
    else if (dz >= dx && dz >= dy) axis = 2;

    // Sort indices by coordinate along the chosen axis
    std::vector<size_t> order(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        double ca = (axis == 0) ? pts[a].pos.x : (axis == 1) ? pts[a].pos.y : pts[a].pos.z;
        double cb = (axis == 0) ? pts[b].pos.x : (axis == 1) ? pts[b].pos.y : pts[b].pos.z;
        return ca < cb;
    });

    size_t n = pts.size();
    size_t half = n / 2;
    left.resize(half);
    right.resize(n - half);
    for (size_t i = 0; i < half; ++i)       left[i]        = pts[order[i]];
    for (size_t i = half; i < n; ++i)       right[i - half] = pts[order[i]];
}

// --------------------------------------------------------------------------
// BVH stack entry: a pair of point-sets (P, Q)
// P is the "moved" set, Q is the "original" set.
// --------------------------------------------------------------------------
struct StackEntry {
    std::vector<IndexedPoint> P;
    std::vector<IndexedPoint> Q;
};

// --------------------------------------------------------------------------
// Main entry point (C linkage for ctypes)
// --------------------------------------------------------------------------
extern "C" {

/*
 * check_new_positions_do_not_cause_overlaps
 *
 * Parameters match what self_distance_c.py passes via ctypes.
 *
 * data layout: all_data_x/y/z are flat arrays in strand-major order.
 * strand s starts at offset sum(strand_lengths[0..s-1]).
 * Point (s, j) is at offset[s] + j.
 */
int check_new_positions_do_not_cause_overlaps(
    int     num_strands,
    int    *strand_lengths,
    double *all_data_x,
    double *all_data_y,
    double *all_data_z,
    int     num_A,
    int    *A_strand,
    int    *A_j,
    double *new_positions_x,
    double *new_positions_y,
    double *new_positions_z,
    int     config_closed,
    int     skippedInteger,
    double  upper_bound
)
{
    bool closed = (config_closed != 0);

    // ------------------------------------------------------------------
    // Build offset table so we can look up data[strand][j] quickly.
    // ------------------------------------------------------------------
    std::vector<int> offset(num_strands + 1, 0);
    for (int s = 0; s < num_strands; ++s)
        offset[s + 1] = offset[s] + strand_lengths[s];

    // Helper: get original data position for (strand, j)
    auto data_pos = [&](int strand, int j) -> Point3 {
        int idx = offset[strand] + j;
        return { all_data_x[idx], all_data_y[idx], all_data_z[idx] };
    };

    // ------------------------------------------------------------------
    // Reconstruct A = indices[1:-1:] as a set of (strand, j) pairs.
    // We also need the two fixed endpoints: indices[0] and indices[-1].
    // ------------------------------------------------------------------
    // A is the interior moved points
    // The strand of A points — all must share the same strand (i_A).
    // (per the Python: i_A = A[0][0])
    // i_A: the strand that contains all points in A (per Python: i_A = A[0][0])
    int i_A = A_strand[0];
    int n_iA = strand_lengths[i_A];  // length of strand i_A

    // Build a quick lookup: is (strand, j) in A?
    // We use a flat boolean array over the i_A strand only
    // (A always lives on one strand per the Python code).
    std::vector<bool> in_A(n_iA, false);
    for (int s = 0; s < num_A; ++s) {
        if (A_strand[s] == i_A)
            in_A[A_j[s]] = true;
    }

    // ------------------------------------------------------------------
    // Build B: all points NOT in A.
    // Build B_prime (not arc-adjacent to either end of A) and
    //       B_complement (arc-adjacent to A[0] or A[-1]).
    // Matches Python initialise_stack exactly.
    // ------------------------------------------------------------------

    // The two endpoints of A for arc-adjacency checks
    int A0_strand = A_strand[0];
    int A0_j      = A_j[0];
    int A_last_s  = A_strand[num_A - 1];
    int A_last_jj = A_j[num_A - 1];

    std::vector<IndexedPoint> B_prime;
    std::vector<IndexedPoint> B_complement;

    for (int s = 0; s < num_strands; ++s) {
        for (int j = 0; j < strand_lengths[s]; ++j) {
            // Is (s, j) in A?
            bool is_in_A = false;
            if (s == i_A && in_A[j]) is_in_A = true;
            if (is_in_A) continue;

            // (s, j) is in B; check arc-adjacency with A[0] and A[-1]
            bool adj_first = is_arc_adjacent(s, j, A0_strand, A0_j, n_iA, closed, skippedInteger);
            bool adj_last  = is_arc_adjacent(s, j, A_last_s, A_last_jj, n_iA, closed, skippedInteger);

            IndexedPoint ip;
            ip.strand = s;
            ip.j      = j;
            ip.pos    = data_pos(s, j);

            if (adj_first || adj_last) {
                B_complement.push_back(ip);
            } else {
                B_prime.push_back(ip);
            }
        }
    }

    // ------------------------------------------------------------------
    // Build A as IndexedPoint list (with newPositions).
    // ------------------------------------------------------------------
    std::vector<IndexedPoint> A_pts(num_A);
    for (int s = 0; s < num_A; ++s) {
        A_pts[s].strand = A_strand[s];
        A_pts[s].j      = A_j[s];
        A_pts[s].pos    = { new_positions_x[s], new_positions_y[s], new_positions_z[s] };
    }

    // ------------------------------------------------------------------
    // Build the initial stack, matching Python initialise_stack.
    //
    // Python:
    //   for (i, j) in B_complement:
    //       A_b = [s for s in A if not arc_adjacent(i,j, s[0], s[1], n, ...)]
    //       if len(A_b) > 0:
    //           stack.append(((A_b, A_b_positions), ([(i,j)], [data[i][j]])))
    //   stack.append(((A, newPositions), (B_prime, B_prime_positions)))
    // ------------------------------------------------------------------
    std::vector<StackEntry> stack;

    for (const IndexedPoint &bp : B_complement) {
        // Build A_b: subset of A not arc-adjacent to (bp.strand, bp.j)
        std::vector<IndexedPoint> A_b;
        for (int s = 0; s < num_A; ++s) {
            if (!is_arc_adjacent(bp.strand, bp.j,
                                 A_pts[s].strand, A_pts[s].j,
                                 n_iA, closed, skippedInteger))
            {
                A_b.push_back(A_pts[s]);
            }
        }
        if (!A_b.empty()) {
            StackEntry entry;
            entry.P = A_b;
            entry.Q = { bp };
            stack.push_back(std::move(entry));
        }
    }

    // Final entry: A vs B_prime (if B_prime non-empty)
    if (!B_prime.empty()) {
        StackEntry entry;
        entry.P = A_pts;
        entry.Q = B_prime;
        stack.push_back(std::move(entry));
    }

    // ------------------------------------------------------------------
    // BVH traversal: pop entries, check distances, subdivide as needed.
    // Matches Python check_new_positions_do_not_cause_overlaps exactly.
    // ------------------------------------------------------------------
    while (!stack.empty()) {
        StackEntry cur = std::move(stack.back());
        stack.pop_back();

        std::vector<IndexedPoint> &P = cur.P;
        std::vector<IndexedPoint> &Q = cur.Q;

        if (P.size() == 1 && Q.size() == 1) {
            // Base case: compute exact distance
            double dx = P[0].pos.x - Q[0].pos.x;
            double dy = P[0].pos.y - Q[0].pos.y;
            double dz = P[0].pos.z - Q[0].pos.z;
            double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (dist < upper_bound) return 0;  // overlap detected
            continue;
        }

        AABB box_P = compute_aabb(P);
        AABB box_Q = compute_aabb(Q);

        if (!aabbs_overlap(box_P, box_Q, upper_bound)) continue;

        // Decide which set to split
        if (P.size() == 1) {
            // Split Q
            std::vector<IndexedPoint> Q_l, Q_r;
            subdivide(box_Q, Q, Q_l, Q_r);
            stack.push_back({ P, std::move(Q_l) });
            stack.push_back({ P, std::move(Q_r) });
        } else if (Q.size() == 1) {
            // Split P
            std::vector<IndexedPoint> P_l, P_r;
            subdivide(box_P, P, P_l, P_r);
            stack.push_back({ std::move(P_l), Q });
            stack.push_back({ std::move(P_r), Q });
        } else {
            // Split the set with the larger bounding box
            double P_max = std::max({box_P.xmax - box_P.xmin,
                                     box_P.ymax - box_P.ymin,
                                     box_P.zmax - box_P.zmin});
            double Q_max = std::max({box_Q.xmax - box_Q.xmin,
                                     box_Q.ymax - box_Q.ymin,
                                     box_Q.zmax - box_Q.zmin});
            if (P_max > Q_max) {
                std::vector<IndexedPoint> P_l, P_r;
                subdivide(box_P, P, P_l, P_r);
                stack.push_back({ std::move(P_l), Q });
                stack.push_back({ std::move(P_r), Q });
            } else {
                std::vector<IndexedPoint> Q_l, Q_r;
                subdivide(box_Q, Q, Q_l, Q_r);
                stack.push_back({ P, std::move(Q_l) });
                stack.push_back({ P, std::move(Q_r) });
            }
        }
    }

    return 1;  // no overlap
}

} // extern "C"
