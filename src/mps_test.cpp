#include <tamm/tamm.hpp>
#include <itensor/all.h>
#include <lapacke.h>
#include <vector>
#include <string>
#include <random>
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <algorithm>
#include <chrono>
#include <cstring>

using i64 = long long;

struct Args {
    i64 bond_dim = 128;
    i64 max_bond_dim = 128;
    i64 gates = 256;
    std::string gate = "cnot";
    unsigned long long seed = 42;
    i64 tilesz = 256;
};

struct TammTB {
    double t_allocate = 0.0;
    double t_fill_random = 0.0;
    double t_gate_fill = 0.0;
    double t_contract_ab = 0.0;
    double t_apply_gate = 0.0;
    double t_pack = 0.0;
    double t_svd = 0.0;
    double t_truncate = 0.0;
    double t_fill_uvt = 0.0;
    double t_deallocate = 0.0;
    double t_total = 0.0;
};

struct ITensorTB {
    double t_build = 0.0;
    double t_contract_ab = 0.0;
    double t_apply_gate = 0.0;
    double t_svd = 0.0;
    double t_reconstruct = 0.0;
    double t_total = 0.0;
};

static bool eqs(const char* a, const char* b) {
    return std::strcmp(a, b) == 0;
}

static Args parse_args(int argc, char** argv) {
    Args a;
    for(int i = 1; i < argc; i++) {
        if(eqs(argv[i], "--bond_dim") && i + 1 < argc) a.bond_dim = std::stoll(argv[++i]);
        else if(eqs(argv[i], "--max_bond_dim") && i + 1 < argc) a.max_bond_dim = std::stoll(argv[++i]);
        else if(eqs(argv[i], "--gates") && i + 1 < argc) a.gates = std::stoll(argv[++i]);
        else if(eqs(argv[i], "--gate") && i + 1 < argc) a.gate = std::string(argv[++i]);
        else if(eqs(argv[i], "--seed") && i + 1 < argc) a.seed = std::stoull(argv[++i]);
        else if(eqs(argv[i], "--tilesz") && i + 1 < argc) a.tilesz = std::stoll(argv[++i]);
    }
    return a;
}

static inline double now_s() {
    return std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

template<typename T>
static void make_gate(const std::string& kind, T* G16) {
    for(int i = 0; i < 16; i++) G16[i] = T(0);
    if(kind == "cnot") { G16[0]=1; G16[5]=1; G16[14]=1; G16[11]=1; return; }
    if(kind == "cz") { G16[0]=1; G16[5]=1; G16[10]=1; G16[15]=-1; return; }
    if(kind == "iswap") { G16[0]=1; G16[6]=1; G16[9]=1; G16[15]=1; return; }
    for(int i = 0; i < 4; i++) G16[i*4+i] = 1;
}

template<typename T>
static void fill_random(tamm::Tensor<T>& X, unsigned long long seed, i64 salt) {
    std::mt19937_64 gen(seed + static_cast<unsigned long long>(salt));
    std::uniform_real_distribution<T> dist(-1.0, 1.0);
    auto f = [&](const tamm::IndexVector& bid, tamm::span<T> buf) {
        for(size_t i = 0; i < buf.size(); i++) buf[i] = dist(gen);
    };
    tamm::update_tensor(X, f);
}

template<typename T>
static void fill_gate_tensor(tamm::Tensor<T>& G, const T* G16) {
    auto f = [&](const tamm::IndexVector& bid, tamm::span<T> buf) {
        auto offs = G.block_offsets(bid);
        int p1 = static_cast<int>(offs[0]);
        int p2 = static_cast<int>(offs[1]);
        int q1 = static_cast<int>(offs[2]);
        int q2 = static_cast<int>(offs[3]);
        int r = p1 * 8 + p2 * 4 + q1 * 2 + q2;
        buf[0] = G16[r];
    };
    tamm::update_tensor(G, f);
}

template<typename T>
static void pack_theta_to_matrix_colmajor(const tamm::Tensor<T>& Th, i64 D, std::vector<T>& A) {
    i64 m = 2 * D;
    auto f = [&](tamm::Tensor<T> t, const tamm::IndexVector& bid, tamm::span<T> buf) {
        auto dims = t.block_dims(bid);
        auto offs = t.block_offsets(bid);
        i64 lo = offs[0];
        i64 qo = offs[1];
        i64 ro = offs[2];
        i64 so = offs[3];
        i64 ld = dims[0];
        i64 qd = dims[1];
        i64 rd = dims[2];
        i64 sd = dims[3];
        i64 c = 0;
        for(i64 l = 0; l < ld; l++)
        for(i64 q1 = 0; q1 < qd; q1++)
        for(i64 q2 = 0; q2 < rd; q2++)
        for(i64 r = 0; r < sd; r++, c++) {
            i64 row = (lo + l) * 2 + (qo + q1);
            i64 col = (q2 + ro) * D + (so + r);
            A[static_cast<size_t>(row) + static_cast<size_t>(m) * static_cast<size_t>(col)] = buf[c];
        }
    };
    tamm::update_tensor_general(Th, f);
}

template<typename T>
static void fill_from_u_s_vt(tamm::Tensor<T>& A2, tamm::Tensor<T>& B2, i64 D, i64 chi, const std::vector<T>& U, const std::vector<T>& S, const std::vector<T>& VT) {
    i64 n = 2 * D;
    auto la = [&](const tamm::IndexVector& bid, tamm::span<T> buf) {
        auto dims = A2.block_dims(bid);
        auto offs = A2.block_offsets(bid);
        i64 lo = offs[0];
        i64 qo = offs[1];
        i64 so = offs[2];
        i64 ld = dims[0];
        i64 qd = dims[1];
        i64 sd = dims[2];
        i64 c = 0;
        for(i64 l = 0; l < ld; l++)
        for(i64 q1 = 0; q1 < qd; q1++)
        for(i64 s = 0; s < sd; s++, c++) {
            i64 row = (lo + l) * 2 + (qo + q1);
            i64 col = so + s;
            buf[c] = U[static_cast<size_t>(row) * static_cast<size_t>(chi) + static_cast<size_t>(col)];
        }
    };
    auto lb = [&](const tamm::IndexVector& bid, tamm::span<T> buf) {
        auto dims = B2.block_dims(bid);
        auto offs = B2.block_offsets(bid);
        i64 so = offs[0];
        i64 qo = offs[1];
        i64 ro = offs[2];
        i64 sd = dims[0];
        i64 qd = dims[1];
        i64 rd = dims[2];
        i64 c = 0;
        for(i64 s = 0; s < sd; s++)
        for(i64 q2 = 0; q2 < qd; q2++)
        for(i64 r = 0; r < rd; r++, c++) {
            i64 col = (qo + q2) * D + (ro + r);
            T v = VT[static_cast<size_t>(so + s) * static_cast<size_t>(n) + static_cast<size_t>(col)];
            buf[c] = S[static_cast<size_t>(so + s)] * v;
        }
    };
    tamm::update_tensor(A2, la);
    tamm::update_tensor(B2, lb);
}

static TammTB two_site_update_tamm_lapack(i64 D, i64 Dmax, const std::string& gate_kind, unsigned long long seed, tamm::ProcGroup self_pg) {
    TammTB tb;
    double t0 = now_s();
    tamm::ExecutionContext ec{self_pg, tamm::DistributionKind::dense, tamm::MemoryManagerKind::ga};
    tamm::Scheduler sch{ec};
    tamm::Tile bt = static_cast<tamm::Tile>(D);
    tamm::TiledIndexSpace bond{tamm::IndexSpace{tamm::range(D)}, bt};
    tamm::TiledIndexSpace phys{tamm::IndexSpace{tamm::range(2)}, 1};
    auto l = bond.label("all");
    auto b = bond.label("all");
    auto rr = bond.label("all");
    auto p1 = phys.label("all");
    auto p2 = phys.label("all");
    auto q1 = phys.label("all");
    auto q2 = phys.label("all");
    tamm::Tensor<double> A({bond, phys, bond});
    tamm::Tensor<double> B({bond, phys, bond});
    tamm::Tensor<double> Th({bond, phys, phys, bond});
    tamm::Tensor<double> Gt({phys, phys, phys, phys});
    A.set_dense();
    B.set_dense();
    Th.set_dense();
    Gt.set_dense();
    double ta0 = now_s();
    sch.allocate(A, B, Th, Gt).execute();
    tb.t_allocate += now_s() - ta0;
    double tf0 = now_s();
    fill_random(A, seed, 1);
    fill_random(B, seed, 2);
    tb.t_fill_random += now_s() - tf0;
    double G16[16];
    make_gate<double>(gate_kind, G16);
    double tg0 = now_s();
    fill_gate_tensor(Gt, G16);
    tb.t_gate_fill += now_s() - tg0;
    double tc1_0 = now_s();
    sch(Th(l, p1, p2, rr) = A(l, p1, b) * B(b, p2, rr)).execute();
    tb.t_contract_ab += now_s() - tc1_0;
    tamm::Tensor<double> Th2({bond, phys, phys, bond});
    Th2.set_dense();
    double ta1 = now_s();
    sch.allocate(Th2).execute();
    tb.t_allocate += now_s() - ta1;
    double tc2_0 = now_s();
    sch(Th2(l, q1, q2, rr) = Th(l, p1, p2, rr) * Gt(p1, p2, q1, q2)).execute();
    tb.t_apply_gate += now_s() - tc2_0;
    i64 m = 2 * D;
    i64 n = 2 * D;
    std::vector<double> Acol(static_cast<size_t>(m) * static_cast<size_t>(n));
    double tpack0 = now_s();
    pack_theta_to_matrix_colmajor(Th2, D, Acol);
    tb.t_pack += now_s() - tpack0;
    i64 k = std::min<i64>(m, n);
    std::vector<double> S(static_cast<size_t>(k));
    std::vector<double> U(static_cast<size_t>(m) * static_cast<size_t>(k));
    std::vector<double> VT(static_cast<size_t>(k) * static_cast<size_t>(n));
    double tsvd0 = now_s();
    LAPACKE_dgesdd(LAPACK_COL_MAJOR, 'S', (lapack_int)m, (lapack_int)n, Acol.data(), (lapack_int)m, S.data(), U.data(), (lapack_int)m, VT.data(), (lapack_int)k);
    tb.t_svd += now_s() - tsvd0;
    i64 chi = std::min<i64>(k, Dmax);
    std::vector<double> Uc(static_cast<size_t>(m) * static_cast<size_t>(chi));
    std::vector<double> VTc(static_cast<size_t>(chi) * static_cast<size_t>(n));
    double ttr0 = now_s();
    for(i64 i = 0; i < m; i++) {
        for(i64 j = 0; j < chi; j++) {
            Uc[static_cast<size_t>(i) * static_cast<size_t>(chi) + static_cast<size_t>(j)] =
                U[static_cast<size_t>(i) * static_cast<size_t>(k) + static_cast<size_t>(j)];
        }
    }
    for(i64 i = 0; i < chi; i++) {
        for(i64 j = 0; j < n; j++) {
            VTc[static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(j)] =
                VT[static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(j)];
        }
    }
    std::vector<double> Sx(static_cast<size_t>(chi));
    for(i64 i = 0; i < chi; i++) Sx[static_cast<size_t>(i)] = S[static_cast<size_t>(i)];
    tb.t_truncate += now_s() - ttr0;
    tamm::TiledIndexSpace chi_tis{tamm::IndexSpace{tamm::range(chi)}, static_cast<tamm::Tile>(chi)};
    auto s = chi_tis.label("all");
    tamm::Tensor<double> A2({bond, phys, chi_tis});
    tamm::Tensor<double> B2({chi_tis, phys, bond});
    A2.set_dense();
    B2.set_dense();
    double ta2 = now_s();
    sch.allocate(A2, B2).execute();
    tb.t_allocate += now_s() - ta2;
    double tfuv0 = now_s();
    fill_from_u_s_vt(A2, B2, D, chi, Uc, Sx, VTc);
    tb.t_fill_uvt += now_s() - tfuv0;
    double td0 = now_s();
    sch.deallocate(A, B, Th, Th2, Gt, A2, B2).execute();
    tb.t_deallocate += now_s() - td0;
    tb.t_total += now_s() - t0;
    return tb;
}

static ITensorTB two_site_update_itensor(i64 D, i64 Dmax, const std::string& gate_kind, unsigned long long seed) {
    using namespace itensor;
    ITensorTB tb;
    double t0 = now_s();
    Index l(int(D), "l");
    Index bb(int(D), "b");
    Index rr(int(D), "r");
    Index p1(2, "p1");
    Index p2(2, "p2");
    Index q1(2, "q1");
    Index q2(2, "q2");
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    ITensor A(l, p1, bb);
    ITensor B(bb, p2, rr);
    double tbuild0 = now_s();
    for(int il = 1; il <= int(D); ++il)
        for(int ip = 1; ip <= 2; ++ip)
            for(int ib = 1; ib <= int(D); ++ib)
                A.set(l=il, p1=ip, bb=ib, dist(gen));
    for(int ib = 1; ib <= int(D); ++ib)
        for(int ip = 1; ip <= 2; ++ip)
            for(int ir = 1; ir <= int(D); ++ir)
                B.set(bb=ib, p2=ip, rr=ir, dist(gen));
    ITensor G(p1, p2, q1, q2);
    double G16[16];
    make_gate<double>(gate_kind, G16);
    int idx = 0;
    for(int a = 1; a <= 2; ++a)
        for(int b2 = 1; b2 <= 2; ++b2)
            for(int c = 1; c <= 2; ++c)
                for(int d = 1; d <= 2; ++d)
                    G.set(p1=a, p2=b2, q1=c, q2=d, G16[idx++]);
    tb.t_build += now_s() - tbuild0;
    double tc1 = now_s();
    ITensor Th = A * B;
    tb.t_contract_ab += now_s() - tc1;
    double tg = now_s();
    ITensor Th2 = Th * G;
    tb.t_apply_gate += now_s() - tg;
    double tsvd = now_s();
    auto [U, S, V] = svd(Th2,
                         IndexSet(l, q1),
                         IndexSet(q2, rr),
                         itensor::Args("Cutoff", 0.0, "MaxDim", int(Dmax), "SVDMethod", "gesdd"));
    tb.t_svd += now_s() - tsvd;
    double trc = now_s();
    ITensor SV = S * V;
    volatile double sink = norm(SV);
    (void)sink;
    tb.t_reconstruct += now_s() - trc;
    tb.t_total += now_s() - t0;
    return tb;
}

static void print_input_and_global(const Args& args, int rank, int world_size, double total_max_time) {
    if(rank == 0) {
        std::cout << "ranks " << world_size << std::endl;
        std::cout << "config bond_dim " << args.bond_dim
                  << " max_bond_dim " << args.max_bond_dim
                  << " gates " << args.gates
                  << " gate " << args.gate
                  << " seed " << args.seed
                  << " tilesz " << args.tilesz << std::endl;
        std::cout << "time_total " << std::fixed << std::setprecision(6) << total_max_time << " s" << std::endl;
    }
}

static void print_rank_breakdown_tamm(int rank, const TammTB& tb, long long tasks) {
    std::cout << "rank " << rank
              << " tasks " << tasks
              << " TAMM+GESDD total " << std::fixed << std::setprecision(6) << tb.t_total
              << "s allocate " << tb.t_allocate
              << "s fill_random " << tb.t_fill_random
              << "s gate_fill " << tb.t_gate_fill
              << "s contract_ab " << tb.t_contract_ab
              << "s apply_gate " << tb.t_apply_gate
              << "s pack " << tb.t_pack
              << "s svd " << tb.t_svd
              << "s truncate " << tb.t_truncate
              << "s fill_uvt " << tb.t_fill_uvt
              << "s deallocate " << tb.t_deallocate
              << "s" << std::endl;
}

static void print_rank_breakdown_itensor(int rank, const ITensorTB& tb, long long tasks) {
    std::cout << "rank " << rank
              << " tasks " << tasks
              << " ITensor total " << std::fixed << std::setprecision(6) << tb.t_total
              << "s build " << tb.t_build
              << "s contract_ab " << tb.t_contract_ab
              << "s apply_gate " << tb.t_apply_gate
              << "s svd " << tb.t_svd
              << "s reconstruct " << tb.t_reconstruct
              << "s" << std::endl;
}

int main(int argc, char** argv) {
    tamm::initialize(argc, argv);
    Args args = parse_args(argc, argv);
    auto world_pg = tamm::ProcGroup::create_world_coll();
    auto self_pg = tamm::ProcGroup::create_self();
    int r = world_pg.rank().value();
    int p = world_pg.size().value();
    world_pg.barrier();
    tamm::AtomicCounterGA ac{world_pg, 1};
    ac.allocate(0);
    double t0 = now_s();
    TammTB acc_tamm;
    ITensorTB acc_it;
    long long tasks_done = 0;
    while(true) {
        long long idx = ac.fetch_add(0, 1);
        if(idx >= args.gates) break;
        TammTB tb1 = two_site_update_tamm_lapack(args.bond_dim, args.max_bond_dim, args.gate, args.seed + static_cast<unsigned long long>(idx), self_pg);
        ITensorTB tb2 = two_site_update_itensor(args.bond_dim, args.max_bond_dim, args.gate, args.seed + static_cast<unsigned long long>(idx));
        acc_tamm.t_allocate += tb1.t_allocate;
        acc_tamm.t_fill_random += tb1.t_fill_random;
        acc_tamm.t_gate_fill += tb1.t_gate_fill;
        acc_tamm.t_contract_ab += tb1.t_contract_ab;
        acc_tamm.t_apply_gate += tb1.t_apply_gate;
        acc_tamm.t_pack += tb1.t_pack;
        acc_tamm.t_svd += tb1.t_svd;
        acc_tamm.t_truncate += tb1.t_truncate;
        acc_tamm.t_fill_uvt += tb1.t_fill_uvt;
        acc_tamm.t_deallocate += tb1.t_deallocate;
        acc_tamm.t_total += tb1.t_total;
        acc_it.t_build += tb2.t_build;
        acc_it.t_contract_ab += tb2.t_contract_ab;
        acc_it.t_apply_gate += tb2.t_apply_gate;
        acc_it.t_svd += tb2.t_svd;
        acc_it.t_reconstruct += tb2.t_reconstruct;
        acc_it.t_total += tb2.t_total;
        tasks_done += 1;
    }
    world_pg.barrier();
    double t1 = now_s();
    double dt = t1 - t0;
    double mx = 0.0;
    world_pg.allreduce(&dt, &mx, 1, tamm::ReduceOp::max);
    print_input_and_global(args, r, p, mx);
    print_rank_breakdown_tamm(r, acc_tamm, tasks_done);
    print_rank_breakdown_itensor(r, acc_it, tasks_done);
    ac.deallocate();
    tamm::finalize();
    return 0;
}

