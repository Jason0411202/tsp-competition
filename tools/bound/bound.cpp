// bound.cpp -- a proven lower bound on the optimal tour length.
//
//   bound <instance.tsp> [--iters N] [--k K] [--ub UB] [--threads T] [--json]
//
// Held-Karp 1-tree bound with subgradient ascent (Held & Karp 1970/71,
// Volgenant & Jonker 1982 direction update).  The ascent itself runs on a
// sparse candidate graph (k nearest neighbours plus the exact Euclidean MST)
// because 300 iterations of an O(N^2) Prim on 100,000 cities would take an
// hour; the candidate graph only steers the penalties pi.  The bound that is
// REPORTED is then evaluated exactly on the complete graph with the final pi:
//
//     L(pi) = w_pi(MST over all N cities) + e2_pi(s) - 2 * sum(pi)
//
// where s is a leaf of that MST and e2_pi(s) its second-cheapest penalised
// edge.  For any pi and any leaf s this is a valid lower bound (the minimum
// 1-tree with special vertex s costs exactly this much, and every tour is a
// 1-tree).  Distances are the competition's EUC_2D integers; the optimum is
// therefore an integer and ceil(L) is also a bound.
//
// This is instructor tooling: it is multi-threaded on purpose, uses C++20,
// and is not bound by the competition's rules.

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

using std::vector;

static int N = 0;
static vector<int64_t> X, Y;

static inline int64_t dist(int i, int j) {
    const double dx = double(X[i] - X[j]), dy = double(Y[i] - Y[j]);
    return (int64_t)std::floor(std::sqrt(dx * dx + dy * dy) + 0.5);
}

static void read_instance(const char* path) {
    std::FILE* f = std::fopen(path, "r");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(1); }
    if (std::fscanf(f, "%d", &N) != 1 || N < 3) { std::fprintf(stderr, "bad header\n"); std::exit(1); }
    X.resize(N); Y.resize(N);
    for (int i = 0; i < N; ++i) {
        long long x, y;
        if (std::fscanf(f, "%lld %lld", &x, &y) != 2) { std::fprintf(stderr, "bad point %d\n", i); std::exit(1); }
        X[i] = x; Y[i] = y;
    }
    std::fclose(f);
}

// ----------------------------------------------------------- k nearest ----

struct Grid {
    int G; double cell; vector<int> start, items;
    void build() {
        G = std::max(1, (int)std::sqrt(N / 2.0));
        cell = 1000001.0 / G;
        vector<int> cnt(G * G + 1, 0);
        auto cid = [&](int i) {
            int cx = std::min(G - 1, (int)(X[i] / cell)), cy = std::min(G - 1, (int)(Y[i] / cell));
            return cy * G + cx;
        };
        for (int i = 0; i < N; ++i) cnt[cid(i) + 1]++;
        for (int c = 0; c < G * G; ++c) cnt[c + 1] += cnt[c];
        start = cnt; items.resize(N);
        vector<int> fill(start.begin(), start.end() - 1);
        for (int i = 0; i < N; ++i) items[fill[cid(i)]++] = i;
    }
    // k nearest neighbours of i (excluding i), by exact squared distance.
    // quad = -1: any direction; 0..3: only points in that quadrant relative to i
    // (bit 0: x >= xi, bit 1: y >= yi).
    void knn(int i, int k, vector<std::pair<int64_t,int>>& out, int quad = -1) const {
        out.clear();
        int cx = std::min(G - 1, (int)(X[i] / cell)), cy = std::min(G - 1, (int)(Y[i] / cell));
        for (int r = 0; ; ++r) {
            // Anything in ring r is at least (r-1)*cell away.
            if ((int)out.size() >= k) {
                double lim = (r - 1) * cell;
                if (lim > 0 && lim * lim > (double)out.front().first) break;
            }
            if (r > G) break;
            for (int dy = -r; dy <= r; ++dy) {
                int yy = cy + dy; if (yy < 0 || yy >= G) continue;
                for (int dx = -r; dx <= r; ++dx) {
                    if (std::abs(dx) != r && std::abs(dy) != r) continue;
                    int xx = cx + dx; if (xx < 0 || xx >= G) continue;
                    int c = yy * G + xx;
                    for (int p = start[c]; p < start[c + 1]; ++p) {
                        int j = items[p]; if (j == i) continue;
                        if (quad >= 0 && ((X[j] >= X[i]) != bool(quad & 1) || (Y[j] >= Y[i]) != bool(quad & 2))) continue;
                        int64_t ddx = X[i] - X[j], ddy = Y[i] - Y[j];
                        int64_t d2 = ddx * ddx + ddy * ddy;
                        if ((int)out.size() < k) {
                            out.push_back({d2, j}); std::push_heap(out.begin(), out.end());
                        } else if (d2 < out.front().first) {
                            std::pop_heap(out.begin(), out.end()); out.back() = {d2, j};
                            std::push_heap(out.begin(), out.end());
                        }
                    }
                }
            }
        }
    }
};

// -------------------------------------------------- exact parallel Prim ----

// Prim over the complete graph with weights d(u,v) + pi[u] + pi[v].
// Returns the tree weight (penalised) and fills parent[].  T threads share the
// O(N) relaxation of every step through a barrier; the 2N barrier phases cost
// a few seconds on 100k cities, the N^2/2 distance evaluations dominate.
static double exact_prim(const vector<double>& pi, vector<int>& parent, int T) {
    vector<double> key(N, 1e300);
    vector<char> done(N, 0);
    parent.assign(N, -1);
    std::atomic<int> cur{0};
    key[0] = 0;
    vector<std::pair<double,int>> best(T);
    std::barrier sync(T);
    double total = 0;
    auto worker = [&](int t) {
        const int lo = (int)((int64_t)N * t / T), hi = (int)((int64_t)N * (t + 1) / T);
        for (int step = 0; step < N; ++step) {
            const int u = cur.load(std::memory_order_acquire);
            const double pu = pi[u];
            const int64_t xu = X[u], yu = Y[u];
            double b = 1e300; int bi = -1;
            for (int v = lo; v < hi; ++v) {
                if (done[v]) continue;
                const double dx = double(xu - X[v]), dy = double(yu - Y[v]);
                const double w = std::floor(std::sqrt(dx * dx + dy * dy) + 0.5) + pu + pi[v];
                if (w < key[v]) { key[v] = w; parent[v] = u; }
                if (key[v] < b) { b = key[v]; bi = v; }
            }
            best[t] = {b, bi};
            sync.arrive_and_wait();
            if (t == 0) {
                double bb = 1e300; int bbi = -1;
                for (auto& p : best) if (p.second >= 0 && p.first < bb) { bb = p.first; bbi = p.second; }
                if (bbi >= 0) { done[bbi] = 1; total += bb; cur.store(bbi, std::memory_order_release); }
                else cur.store(-1, std::memory_order_release);
            }
            sync.arrive_and_wait();
            if (cur.load(std::memory_order_acquire) < 0) break;
        }
    };
    done[0] = 1;
    vector<std::thread> th;
    for (int t = 0; t < T; ++t) th.emplace_back(worker, t);
    for (auto& x : th) x.join();
    return total;
}

// Exact 1-tree bound for a given pi: L = w(MST_pi) + e2_pi(s) - 2*sum(pi),
// with s the MST leaf whose exact second-cheapest edge is largest among a
// sample of leaves (any leaf is valid; a good leaf is a tighter bound).
static double exact_one_tree(const vector<double>& pi, int T, double* mst_w_out, vector<int>* parent_out = nullptr) {
    vector<int> parent;
    double w = exact_prim(pi, parent, T);
    if (parent_out) *parent_out = parent;
    vector<int> deg(N, 0);
    for (int v = 0; v < N; ++v) if (parent[v] >= 0) { deg[v]++; deg[parent[v]]++; }
    vector<int> leaves;
    for (int v = 0; v < N; ++v) if (deg[v] == 1) leaves.push_back(v);
    // Exact second-cheapest edge at a leaf costs O(N); try up to 64 leaves.
    std::sort(leaves.begin(), leaves.end(), [&](int a, int b) {
        return dist(a, parent[a]) + pi[a] + pi[parent[a]] > dist(b, parent[b]) + pi[b] + pi[parent[b]]; });
    if (leaves.size() > 64) leaves.resize(64);
    double bestL = -1e300;
    for (int s : leaves) {
        double e1 = 1e300, e2 = 1e300;
        for (int v = 0; v < N; ++v) {
            if (v == s) continue;
            double d = dist(s, v) + pi[s] + pi[v];
            if (d < e1) { e2 = e1; e1 = d; } else if (d < e2) e2 = d;
        }
        // The leaf's tree edge is its cheapest edge (see the header comment),
        // so the 1-tree with special vertex s costs w + e2.
        bestL = std::max(bestL, w + e2);
    }
    double sp = 0; for (double p : pi) sp += p;
    if (mst_w_out) *mst_w_out = w;
    return bestL - 2 * sp;
}

// ------------------------------------------------------- sparse 1-tree ----

struct Edge { int u, v; int64_t d; };

struct DSU {
    vector<int> p;
    explicit DSU(int n) : p(n) { std::iota(p.begin(), p.end(), 0); }
    int f(int x) { while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; } return x; }
    bool join(int a, int b) { a = f(a); b = f(b); if (a == b) return false; p[a] = b; return true; }
};

// Minimum 1-tree on the candidate graph under pi; returns its penalised
// weight (minus 2*sum pi), fills deg[] (special vertex included).
static double sparse_one_tree(const vector<Edge>& E, const vector<double>& pi,
                              vector<int>& order, vector<double>& w, vector<int>& deg) {
    const int m = (int)E.size();
    for (int i = 0; i < m; ++i) w[i] = E[i].d + pi[E[i].u] + pi[E[i].v];
    std::sort(order.begin(), order.end(), [&](int a, int b) { return w[a] < w[b]; });
    DSU dsu(N);
    std::fill(deg.begin(), deg.end(), 0);
    double total = 0; int taken = 0;
    vector<char> inTree(m, 0);
    for (int idx : order) {
        const Edge& e = E[idx];
        if (dsu.join(e.u, e.v)) { total += w[idx]; deg[e.u]++; deg[e.v]++; inTree[idx] = 1; if (++taken == N - 1) break; }
    }
    if (taken != N - 1) { std::fprintf(stderr, "candidate graph disconnected\n"); std::exit(2); }
    // Special vertex: the leaf with the largest second-cheapest candidate edge.
    vector<double> e2(N, 1e300);
    {
        vector<double> e1(N, 1e300);
        for (int i = 0; i < m; ++i) {
            const Edge& e = E[i];
            for (int x : {e.u, e.v}) {
                if (w[i] < e1[x]) { e2[x] = e1[x]; e1[x] = w[i]; } else if (w[i] < e2[x]) e2[x] = w[i];
            }
        }
    }
    int s = -1; double bs = -1e300;
    for (int v = 0; v < N; ++v) if (deg[v] == 1 && e2[v] < 1e299 && e2[v] > bs) { bs = e2[v]; s = v; }
    if (s < 0) { for (int v = 0; v < N; ++v) if (deg[v] == 1) { s = v; break; } bs = 0; }
    // Find the other endpoint of that second edge to update degrees.
    int other = -1; double bo = 1e300;
    for (int i = 0; i < m; ++i) {
        const Edge& e = E[i];
        if (inTree[i]) continue;
        if (e.u == s || e.v == s) { if (w[i] < bo) { bo = w[i]; other = e.u == s ? e.v : e.u; } }
    }
    if (other >= 0) { deg[s]++; deg[other]++; total += bo; }
    double sp = 0; for (double p : pi) sp += p;
    return total - 2 * sp;
}

// --------------------------------------------------------------- main ----

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: bound <instance.tsp> [--iters N] [--k K] [--ub UB] [--threads T] [--json]\n"); return 1; }
    int iters = 6000, K = 10, T = (int)std::max(1u, std::thread::hardware_concurrency());
    int periodArg = 0, maxRounds = 12; double ub = 0; bool json = false; (void)ub;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (a == "--k" && i + 1 < argc) K = std::atoi(argv[++i]);
        else if (a == "--ub" && i + 1 < argc) ub = std::atof(argv[++i]);
        else if (a == "--period" && i + 1 < argc) periodArg = std::atoi(argv[++i]);
        else if (a == "--rounds" && i + 1 < argc) maxRounds = std::atoi(argv[++i]);
        else if (a == "--threads" && i + 1 < argc) T = std::atoi(argv[++i]);
        else if (a == "--json") json = true;
    }
    auto t0 = std::chrono::steady_clock::now();
    auto secs = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };

    read_instance(argv[1]);
    T = std::max(1, std::min(T, N / 2000 + 1));

    // 1. Exact Euclidean MST (pi = 0): candidate edges + the plain 1-tree bound.
    vector<double> pi(N, 0.0);
    double mst0;
    vector<int> parent0;
    double L0 = exact_one_tree(pi, T, &mst0, &parent0);
    std::fprintf(stderr, "[%6.1fs] exact MST %.0f, plain 1-tree bound %.1f\n", secs(), mst0, L0);

    // 2. Candidate graph: K nearest neighbours + MST edges.
    Grid grid; grid.build();
    vector<Edge> E;
    E.reserve((size_t)N * (K + 1));
    {
        vector<std::pair<int64_t,int>> nb;
        auto addAll = [&](int i) {
            for (auto& p : nb) if (i < p.second) E.push_back({i, p.second, dist(i, p.second)});
            else E.push_back({p.second, i, dist(i, p.second)});   // dedupe below
        };
        for (int i = 0; i < N; ++i) {
            grid.knn(i, K, nb); addAll(i);
            for (int quad = 0; quad < 4; ++quad) { grid.knn(i, 2, nb, quad); addAll(i); }
        }
        for (int v = 1; v < N; ++v) if (parent0[v] >= 0) {
            int a = std::min(v, parent0[v]), b = std::max(v, parent0[v]);
            E.push_back({a, b, dist(a, b)});
        }
        std::sort(E.begin(), E.end(), [](const Edge& a, const Edge& b) {
            return a.u != b.u ? a.u < b.u : a.v < b.v; });
        E.erase(std::unique(E.begin(), E.end(), [](const Edge& a, const Edge& b) {
            return a.u == b.u && a.v == b.v; }), E.end());
    }
    std::fprintf(stderr, "[%6.1fs] candidate graph: %zu edges\n", secs(), E.size());

    // 3. Subgradient ascent on the candidate graph -- the Held-Karp /
    //    Volgenant-Jonker schedule as implemented in LKH's Ascent(): a fixed
    //    step T for a period of iterations; in the initial phase T doubles on
    //    every improvement and the period doubles when the improvement came at
    //    its last iteration; once a period ends without progress, both halve.
    //    Direction is 0.7 * (deg - 2) + 0.3 * previous direction.
    //
    //    The candidate graph is then REPAIRED: the exact 1-tree at the best
    //    penalties is computed on the complete graph, and if it is cheaper than
    //    the sparse one (edges the ascent never saw became attractive under
    //    pi), its edges are added and the ascent is run again from the best
    //    penalties.  Without this the ascent on street-like instances happily
    //    climbs to a sparse value 25% above the true bound.
    vector<int> deg(N);
    vector<double> g(N), gprev(N), bestPi = pi;
    int done_iters = 0;
    double bestW = -1e300, L = L0, mstpi = mst0;
    const int initialPeriod = periodArg > 0 ? periodArg : std::max(20, std::min(N / 2, 400));

    auto ascend = [&](const vector<Edge>& Ecur, vector<double> start) {
        const int mm = (int)Ecur.size();
        vector<int> order(mm); std::iota(order.begin(), order.end(), 0);
        vector<double> w(mm);
        vector<double> cur = start, bp = start;
        std::fill(gprev.begin(), gprev.end(), 0.0);
        double bw = -1e300;
        int period = initialPeriod, its = 0;
        double T = 1.0;
        bool initialPhase = true, stop = false;
        for (; period > 0 && T > 1e-7 && !stop; period /= 2, T /= 2) {
            for (int P = 1; P <= period && !stop; ++P) {
                double W = sparse_one_tree(Ecur, cur, order, w, deg);
                ++its; ++done_iters;
                double norm = 0;
                for (int v = 0; v < N; ++v) { g[v] = 0.7 * (deg[v] - 2) + 0.3 * gprev[v]; norm += g[v] * g[v]; }
                if (W > bw + 1e-9) {
                    bw = W; bp = cur;
                    if (initialPhase) T *= 2;
                    if (P == period) period = std::min(period * 2, initialPeriod);
                } else if (initialPhase && P > period / 2) {
                    initialPhase = false; P = 0; T = 0.75 * T;
                }
                if (norm < 1e-12 || its >= iters) { stop = true; break; }
                for (int v = 0; v < N; ++v) { cur[v] += T * g[v]; gprev[v] = g[v]; }
                if (its % 500 == 0)
                    std::fprintf(stderr, "[%6.1fs]   iter %5d  W=%.1f  best=%.1f  T=%.4f period=%d\n", secs(), its, W, bw, T, period);
            }
        }
        return std::make_pair(bw, bp);
    };

    for (int round = 0; round < maxRounds; ++round) {
        auto [bw, bp] = ascend(E, bestPi);
        bestW = bw; bestPi = bp;
        vector<int> parentPi;
        L = exact_one_tree(bestPi, T, &mstpi, &parentPi);
        std::fprintf(stderr, "[%6.1fs] round %d: %d candidate edges, sparse best %.1f, exact %.1f\n",
                     secs(), round, (int)E.size(), bestW, L);
        if (bestW - L <= 2e-4 * std::max(1.0, L)) break;       // the candidate graph is adequate
        // Add the exact tree's edges and go again.
        size_t before = E.size();
        for (int v = 0; v < N; ++v) if (parentPi[v] >= 0) {
            int a = std::min(v, parentPi[v]), b = std::max(v, parentPi[v]);
            E.push_back({a, b, dist(a, b)});
        }
        std::sort(E.begin(), E.end(), [](const Edge& x, const Edge& y) { return x.u != y.u ? x.u < y.u : x.v < y.v; });
        E.erase(std::unique(E.begin(), E.end(), [](const Edge& x, const Edge& y) { return x.u == y.u && x.v == y.v; }), E.end());
        if (E.size() == before) break;                          // nothing new to learn
    }
    const int m = (int)E.size();
    double best = std::max(L, L0);
    std::fprintf(stderr, "[%6.1fs] exact Held-Karp bound %.3f (plain 1-tree %.1f)\n", secs(), best, L0);

    int64_t lb = (int64_t)std::ceil(best - 1e-6);
    if (json) {
        std::printf("{\"held_karp\": %.3f, \"lower_bound\": %lld, \"one_tree_plain\": %.3f, "
                    "\"mst\": %.0f, \"candidate_edges\": %d, \"k\": %d, \"iters\": %d, \"seconds\": %.1f}\n",
                    best, (long long)lb, L0, mst0, m, K, done_iters, secs());
    } else {
        std::printf("lower_bound %lld  (Held-Karp %.3f, MST %.0f)\n", (long long)lb, best, mst0);
    }
    return 0;
}
