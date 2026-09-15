// tsp_foundation.cpp
//
// Foundation solver for the TSP Approximation Competition: the textbook
// 2-approximation for metric TSP (minimum spanning tree, then a preorder walk
// of the tree). Students start from this file and make its tours SHORTER.
//
// Usage:
//     ./solver <instance.tsp> <output.tour> <time_limit_seconds>
//
// The third argument is the wall-clock budget the grader will enforce. This
// foundation ignores it (it always finishes well inside the budget); a better
// solver keeps improving its tour until the budget is nearly used up.
//
// File formats:
//
//   <instance.tsp>
//     N
//     x_0 y_0
//     ...
//     x_{N-1} y_{N-1}          (integer coordinates)
//
//   <output.tour>
//     p_0
//     ...
//     p_{N-1}                  (a permutation of 0..N-1; the tour returns
//                               from p_{N-1} to p_0)
//
// Distance between two cities is the Euclidean distance rounded to the
// nearest integer (TSPLIB "EUC_2D"):
//
//     d(i, j) = floor( sqrt((xi-xj)^2 + (yi-yj)^2) + 0.5 )
//
// The tour length is the sum of the N distances around the cycle. The grader
// computes it exactly this way, in 64-bit integers.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using std::vector;

static int32_t N = 0;
static vector<int64_t> X, Y;

static void read_instance(const char* path) {
    std::FILE* f = std::fopen(path, "r");
    if (!f) { std::fprintf(stderr, "cannot open instance: %s\n", path); std::exit(1); }
    if (std::fscanf(f, "%d", &N) != 1 || N <= 0) {
        std::fprintf(stderr, "bad instance header\n"); std::exit(1);
    }
    X.resize(N); Y.resize(N);
    for (int32_t i = 0; i < N; ++i) {
        long long x, y;
        if (std::fscanf(f, "%lld %lld", &x, &y) != 2) {
            std::fprintf(stderr, "bad point at line %d\n", i + 2); std::exit(1);
        }
        X[i] = x; Y[i] = y;
    }
    std::fclose(f);
}

// The official distance: Euclidean, rounded to the nearest integer.
static inline int64_t dist(int32_t i, int32_t j) {
    const double dx = double(X[i] - X[j]);
    const double dy = double(Y[i] - Y[j]);
    return (int64_t)std::floor(std::sqrt(dx * dx + dy * dy) + 0.5);
}

// The MST below uses SQUARED distances (no sqrt): a minimum spanning tree
// under squared distances is also one under the rounded Euclidean distance,
// because rounding and squaring are both monotone.

// Prim's algorithm on the complete graph, O(N^2) time, O(N) memory.
// Returns parent[] with parent[root] == -1.
//
// The vertices not yet in the tree are kept in compact arrays (coordinates,
// key, id) and swap-removed as they join, so the inner loop has no branch and
// no indirection and the compiler vectorises it. Single-precision floats hold
// the squared distances: coordinates are below 2^24 so they are exact, and a
// rounding error in the 7th digit of a squared distance only changes which of
// two nearly equal edges the tree picks. About 3 s for 100,000 cities and
// 12 s for 200,000; the textbook version with a done[] flag takes 17 s and
// 70 s.
static vector<int32_t> prim_mst(int32_t root) {
    vector<int32_t> parent(N, -1);
    int32_t m = N - 1;                      // vertices outside the tree
    vector<float> rx(m), ry(m);
    vector<int32_t> rid(m);
    // Key and parent packed into one 64-bit word: the squared distance's float
    // bit pattern in the high half (positive floats order like unsigned ints),
    // the parent in the low half. One unsigned min then updates both, which
    // is what lets the compiler vectorise the relaxation loop.
    vector<uint64_t> rkey(m, UINT64_MAX);
    for (int32_t v = 0, k = 0; v < N; ++v) {
        if (v == root) continue;
        rx[k] = float(X[v]); ry[k] = float(Y[v]); rid[k] = v; ++k;
    }
    float* __restrict px = rx.data();
    float* __restrict py = ry.data();
    uint64_t* __restrict pk = rkey.data();
    int32_t u = root;
    while (m > 0) {
        const float ux = float(X[u]), uy = float(Y[u]);
        const uint64_t ulow = (uint32_t)u;
        for (int32_t i = 0; i < m; ++i) {           // relax against u
            const float dx = ux - px[i], dy = uy - py[i];
            const float d2 = dx * dx + dy * dy;
            uint32_t bits;
            std::memcpy(&bits, &d2, sizeof bits);
            const uint64_t cand = ((uint64_t)bits << 32) | ulow;
            pk[i] = cand < pk[i] ? cand : pk[i];
        }
        uint64_t best = UINT64_MAX;                  // closest to the tree
        for (int32_t i = 0; i < m; ++i) best = pk[i] < best ? pk[i] : best;
        int32_t bi = 0;
        while (pk[bi] != best) ++bi;
        u = rid[bi];
        parent[u] = (int32_t)(uint32_t)best;
        --m;                                         // swap-remove bi
        px[bi] = px[m]; py[bi] = py[m]; pk[bi] = pk[m]; rid[bi] = rid[m];
    }
    return parent;
}

// Preorder walk of the tree, iteratively (a recursive DFS would blow the
// stack on a 100k-vertex path-like tree).
static vector<int32_t> preorder(const vector<int32_t>& parent, int32_t root) {
    vector<vector<int32_t>> children(N);
    for (int32_t v = 0; v < N; ++v)
        if (parent[v] >= 0) children[parent[v]].push_back(v);

    vector<int32_t> tour;
    tour.reserve(N);
    vector<int32_t> stack;
    stack.push_back(root);
    while (!stack.empty()) {
        int32_t u = stack.back();
        stack.pop_back();
        tour.push_back(u);
        // Push in reverse so the first child is visited first.
        for (auto it = children[u].rbegin(); it != children[u].rend(); ++it)
            stack.push_back(*it);
    }
    return tour;
}

static int64_t tour_length(const vector<int32_t>& tour) {
    int64_t total = 0;
    for (size_t i = 0; i < tour.size(); ++i)
        total += dist(tour[i], tour[(i + 1) % tour.size()]);
    return total;
}

static void write_tour(const char* path, const vector<int32_t>& tour) {
    std::FILE* f = std::fopen(path, "w");
    if (!f) { std::fprintf(stderr, "cannot open output: %s\n", path); std::exit(1); }
    for (int32_t v : tour) std::fprintf(f, "%d\n", v);
    std::fclose(f);
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s <instance.tsp> <output.tour> <time_limit_seconds>\n", argv[0]);
        return 1;
    }
    read_instance(argv[1]);

    // 1. MST of the complete Euclidean graph.
    // 2. Preorder walk of the MST, rooted at city 0.
    // 3. That order is the tour (Approximation Algorithms, slide 28).
    vector<int32_t> parent = prim_mst(0);
    vector<int32_t> tour = preorder(parent, 0);

    write_tour(argv[2], tour);
    std::fprintf(stderr, "tour length %lld\n", (long long)tour_length(tour));
    return 0;
}
