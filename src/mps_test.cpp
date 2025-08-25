#include <tamm/tamm.hpp>
#include <slate/slate.hh>
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

static inline void rprint(int rank, const std::string& s) {
    std::cout << "[Rank " << rank << "] " << s << std::endl;
}

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

template<typename T>
static void make_gate(const std::string& kind, T* G16, int rank) {
    rprint(rank, "make_gate enter kind=" + kind);
    for(int i = 0; i < 16; i++) G16[i] = T(0);
    if(kind == "cnot") { G16[0]=1; G16[5]=1; G16[14]=1; G16[11]=1; rprint(rank, "make_gate leave kind=cnot"); return; }
    if(kind == "cz") { G16[0]=1; G16[5]=1; G16[10]=1; G16[15]=-1; rprint(rank, "make_gate leave kind=cz"); return; }
    if(kind == "iswap") { G16[0]=1; G16[6]=1; G16[9]=1; G16[15]=1; rprint(rank, "make_gate leave kind=iswap"); return; }
    for(int i = 0; i < 4; i++) G16[i*4+i] = 1;
    rprint(rank, "make_gate leave kind=identity");
}

template<typename T>
static void fill_random(tamm::Tensor<T>& X, unsigned long long seed, i64 salt, const char* name, int rank) {
    rprint(rank, std::string("fill_random enter tensor=") + name + " seed=" + std::to_string(seed) + " salt=" + std::to_string(salt));
    std::mt19937_64 gen(seed + static_cast<unsigned long long>(salt));
    std::uniform_real_distribution<T> dist(-1.0, 1.0);
    auto f = [&](const tamm::IndexVector& bid, tamm::span<T> buf) {
        auto dims = X.block_dims(bid);
        auto offs = X.block_offsets(bid);
        rprint(rank, std::string("fill_random block tensor=") + name +
                      " off=[" + std::to_string(offs[0]) + "," + std::to_string(offs[1]) + "," + std::to_string(offs[2]) + "]" +
                      " dim=[" + std::to_string(dims[0]) + "," + std::to_string(dims[1]) + "," + std::to_string(dims[2]) + "]" +
                      " size=" + std::to_string(buf.size()));
        for(size_t i = 0; i < buf.size(); i++) buf[i] = dist(gen);
    };
    tamm::update_tensor(X, f);
    rprint(rank, std::string("fill_random leave tensor=") + name);
}

template<typename T>
static void fill_gate_tensor(tamm::Tensor<T>& G, const T* G16, const char* name, int rank) {
    rprint(rank, std::string("fill_gate_tensor enter tensor=") + name);
    auto f = [&](const tamm::IndexVector& bid, tamm::span<T> buf) {
        auto dims = G.block_dims(bid);
        auto offs = G.block_offsets(bid);
        rprint(rank, std::string("fill_gate_tensor block tensor=") + name +
                      " off=[" + std::to_string(offs[0]) + "," + std::to_string(offs[1]) + "," + std::to_string(offs[2]) + "," + std::to_string(offs[3]) + "]" +
                      " dim=[" + std::to_string(dims[0]) + "," + std::to_string(dims[1]) + "," + std::to_string(dims[2]) + "," + std::to_string(dims[3]) + "]" +
                      " size=" + std::to_string(buf.size()));
        for(size_t i = 0; i < buf.size(); i++) buf[i] = T(0);
        for(int p1 = 0; p1 < 2; p1++)
        for(int p2 = 0; p2 < 2; p2++)
        for(int q1 = 0; q1 < 2; q1++)
        for(int q2 = 0; q2 < 2; q2++) {
            int r = p1 * 8 + p2 * 4 + q1 * 2 + q2;
            buf[static_cast<size_t>(r)] = G16[r];
        }
    };
    tamm::update_tensor(G, f);
    rprint(rank, std::string("fill_gate_tensor leave tensor=") + name);
}

template<typename T>
static void pack_theta_to_matrix(const tamm::Tensor<T>& Th, i64 D, std::vector<T>& M, int rank) {
    i64 m = 2 * D;
    i64 n = 2 * D;
    rprint(rank, "pack_theta_to_matrix enter m=" + std::to_string(m) + " n=" + std::to_string(n) + " M.size=" + std::to_string(M.size()));
    auto f = [&](tamm::Tensor<T> t, const tamm::IndexVector& bid, tamm::span<T> buf) {
        auto dims = t.block_dims(bid);
        auto offs = t.block_offsets(bid);
        rprint(rank, std::string("pack block off=[") + std::to_string(offs[0]) + "," + std::to_string(offs[1]) + "," + std::to_string(offs[2]) + "," + std::to_string(offs[3]) + "]" +
                      " dim=[" + std::to_string(dims[0]) + "," + std::to_string(dims[1]) + "," + std::to_string(dims[2]) + "," + std::to_string(dims[3]) + "]" +
                      " size=" + std::to_string(buf.size()));
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
            M[static_cast<size_t>(row) * static_cast<size_t>(n) + static_cast<size_t>(col)] = buf[c];
        }
    };
    tamm::update_tensor_general(Th, f);
    rprint(rank, "pack_theta_to_matrix leave");
}

template<typename T>
static void slate_copy_in(slate::Matrix<T>& A, const std::vector<T>& src, int rank) {
    rprint(rank, "slate_copy_in enter mt=" + std::to_string(A.mt()) + " nt=" + std::to_string(A.nt()) + " m=" + std::to_string(A.m()) + " n=" + std::to_string(A.n()));
    for(int64_t j = 0; j < A.nt(); j++) {
        for(int64_t i = 0; i < A.mt(); i++) {
            if(!A.tileIsLocal(i, j)) continue;
            rprint(rank, "slate_copy_in tile i=" + std::to_string(i) + " j=" + std::to_string(j));
            A.tileGetForWriting(i, j, slate::LayoutConvert::ColMajor);
            auto Tt = A(i, j);
            T* a = Tt.data();
            int64_t lda = Tt.stride();
            int64_t mb = Tt.mb();
            int64_t nb = Tt.nb();
            int64_t roff = i * mb;
            int64_t coff = j * nb;
            for(int64_t jj = 0; jj < nb; jj++) {
                for(int64_t ii = 0; ii < mb; ii++) {
                    a[ii + lda * jj] = src[static_cast<size_t>(roff + ii) * static_cast<size_t>(A.n()) + static_cast<size_t>(coff + jj)];
                }
            }
        }
    }
    rprint(rank, "slate_copy_in leave");
}

template<typename T>
static void slate_copy_out(slate::Matrix<T>& A, std::vector<T>& dst, int rank) {
    rprint(rank, "slate_copy_out enter mt=" + std::to_string(A.mt()) + " nt=" + std::to_string(A.nt()) + " m=" + std::to_string(A.m()) + " n=" + std::to_string(A.n()));
    for(int64_t j = 0; j < A.nt(); j++) {
        for(int64_t i = 0; i < A.mt(); i++) {
            if(!A.tileIsLocal(i, j)) continue;
            rprint(rank, "slate_copy_out tile i=" + std::to_string(i) + " j=" + std::to_string(j));
            A.tileGetForReading(i, j, slate::LayoutConvert::ColMajor);
            auto Tt = A(i, j);
            const T* a = Tt.data();
            int64_t lda = Tt.stride();
            int64_t mb = Tt.mb();
            int64_t nb = Tt.nb();
            int64_t roff = i * mb;
            int64_t coff = j * nb;
            for(int64_t jj = 0; jj < nb; jj++) {
                for(int64_t ii = 0; ii < mb; ii++) {
                    dst[static_cast<size_t>(roff + ii) * static_cast<size_t>(A.n()) + static_cast<size_t>(coff + jj)] = a[ii + lda * jj];
                }
            }
        }
    }
    rprint(rank, "slate_copy_out leave");
}

template<typename T>
static void fill_from_u_s_vt(tamm::Tensor<T>& A2, tamm::Tensor<T>& B2, i64 D, i64 chi, const std::vector<T>& U, const std::vector<T>& S, const std::vector<T>& VT, int rank) {
    i64 n = 2 * D;
    rprint(rank, "fill_from_u_s_vt enter D=" + std::to_string(D) + " chi=" + std::to_string(chi) + " n=" + std::to_string(n));
    auto la = [&](const tamm::IndexVector& bid, tamm::span<T> buf) {
        auto dims = A2.block_dims(bid);
        auto offs = A2.block_offsets(bid);
        rprint(rank, std::string("fill A2 block off=[") + std::to_string(offs[0]) + "," + std::to_string(offs[1]) + "," + std::to_string(offs[2]) + "]" +
                      " dim=[" + std::to_string(dims[0]) + "," + std::to_string(dims[1]) + "," + std::to_string(dims[2]) + "]" +
                      " size=" + std::to_string(buf.size()));
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
        rprint(rank, std::string("fill B2 block off=[") + std::to_string(offs[0]) + "," + std::to_string(offs[1]) + "," + std::to_string(offs[2]) + "]" +
                      " dim=[" + std::to_string(dims[0]) + "," + std::to_string(dims[1]) + "," + std::to_string(dims[2]) + "]" +
                      " size=" + std::to_string(buf.size()));
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
    rprint(rank, "fill_from_u_s_vt leave");
}

template<typename T>
static void two_site_update(i64 D, i64 Dmax, i64 tilesz, const std::string& gate_kind, unsigned long long seed, tamm::ProcGroup self_pg, int rank) {
    rprint(rank, "two_site_update enter D=" + std::to_string(D) + " Dmax=" + std::to_string(Dmax) + " tilesz=" + std::to_string(tilesz) + " gate=" + gate_kind + " seed=" + std::to_string(seed));
    tamm::ExecutionContext ec{self_pg, tamm::DistributionKind::dense, tamm::MemoryManagerKind::ga};
    tamm::Scheduler sch{ec};
    tamm::Tile bt = static_cast<tamm::Tile>(D);
    tamm::TiledIndexSpace bond{tamm::IndexSpace{tamm::range(D)}, bt};
    tamm::TiledIndexSpace phys{tamm::IndexSpace{tamm::range(2)}, 1};
    auto l = bond.label("all");
    auto b = bond.label("all");
    auto r = bond.label("all");
    auto p1 = phys.label("all");
    auto p2 = phys.label("all");
    auto q1 = phys.label("all");
    auto q2 = phys.label("all");
    tamm::Tensor<T> A({bond, phys, bond});
    tamm::Tensor<T> B({bond, phys, bond});
    tamm::Tensor<T> Th({bond, phys, phys, bond});
    tamm::Tensor<T> Gt({phys, phys, phys, phys});
    A.set_dense();
    B.set_dense();
    Th.set_dense();
    Gt.set_dense();
    rprint(rank, "allocate A B Th Gt");
    sch.allocate(A, B, Th, Gt).execute();
    rprint(rank, "fill_random A B");
    fill_random(A, seed, 1, "A", rank);
    fill_random(B, seed, 2, "B", rank);
    T G16[16];
    make_gate<T>(gate_kind, G16, rank);
    fill_gate_tensor(Gt, G16, "Gt", rank);
    rprint(rank, "compute Th = A * B");
    sch(Th(l, p1, p2, r) = A(l, p1, b) * B(b, p2, r)).execute();
    rprint(rank, "allocate Th2 and apply gate");
    tamm::Tensor<T> Th2({bond, phys, phys, bond});
    Th2.set_dense();
    sch.allocate(Th2).execute();
    sch(Th2(l, q1, q2, r) = Th(l, p1, p2, r) * Gt(p1, p2, q1, q2)).execute();
    i64 m = 2 * D;
    i64 n = 2 * D;
    rprint(rank, "pack theta to matrix M m=" + std::to_string(m) + " n=" + std::to_string(n));
    std::vector<T> M(static_cast<size_t>(m) * static_cast<size_t>(n));
    pack_theta_to_matrix(Th2, D, M, rank);
    slate::Matrix<T> Mmat(m, n, tilesz, 1, 1, self_pg.comm());
    Mmat.insertLocalTiles();
    slate_copy_in(Mmat, M, rank);
    i64 k = std::min<i64>(m, n);
    rprint(rank, "prepare SVD k=" + std::to_string(k));
    std::vector<T> S(static_cast<size_t>(k));
    slate::Matrix<T> U(m, k, tilesz, 1, 1, self_pg.comm());
    slate::Matrix<T> VT(k, n, tilesz, 1, 1, self_pg.comm());
    U.insertLocalTiles();
    VT.insertLocalTiles();
    rprint(rank, "svd start");
    slate::svd(Mmat, S, U, VT, {{slate::Option::Target, slate::Target::Devices}});
    rprint(rank, "svd done");
    std::vector<T> Ubuf(static_cast<size_t>(m) * static_cast<size_t>(k));
    std::vector<T> VTbuf(static_cast<size_t>(k) * static_cast<size_t>(n));
    slate_copy_out(U, Ubuf, rank);
    slate_copy_out(VT, VTbuf, rank);
    i64 chi = std::min<i64>(k, Dmax);
    rprint(rank, "truncate chi=" + std::to_string(chi));
    std::vector<T> Uc(static_cast<size_t>(m) * static_cast<size_t>(chi));
    std::vector<T> VTc(static_cast<size_t>(chi) * static_cast<size_t>(n));
    for(i64 i = 0; i < m; i++) {
        for(i64 j = 0; j < chi; j++) {
            Uc[static_cast<size_t>(i) * static_cast<size_t>(chi) + static_cast<size_t>(j)] =
                Ubuf[static_cast<size_t>(i) * static_cast<size_t>(k) + static_cast<size_t>(j)];
        }
    }
    for(i64 i = 0; i < chi; i++) {
        for(i64 j = 0; j < n; j++) {
            VTc[static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(j)] =
                VTbuf[static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(j)];
        }
    }
    std::vector<T> Sx(static_cast<size_t>(chi));
    for(i64 i = 0; i < chi; i++) Sx[static_cast<size_t>(i)] = S[static_cast<size_t>(i)];
    tamm::TiledIndexSpace chi_tis{tamm::IndexSpace{tamm::range(chi)}, static_cast<tamm::Tile>(chi)};
    auto s = chi_tis.label("all");
    tamm::Tensor<T> A2({bond, phys, chi_tis});
    tamm::Tensor<T> B2({chi_tis, phys, bond});
    A2.set_dense();
    B2.set_dense();
    rprint(rank, "allocate A2 B2 and refill from U S VT");
    sch.allocate(A2, B2).execute();
    fill_from_u_s_vt(A2, B2, D, chi, Uc, Sx, VTc, rank);
    rprint(rank, "deallocate tensors");
    sch.deallocate(A, B, Th, Th2, Gt, A2, B2).execute();
    rprint(rank, "two_site_update leave");
}

int main(int argc, char** argv) {
    tamm::initialize(argc, argv);
    Args args = parse_args(argc, argv);
    auto world_pg = tamm::ProcGroup::create_world_coll();
    auto self_pg = tamm::ProcGroup::create_self();
    int r = world_pg.rank().value();
    int p = world_pg.size().value();
    if(r == 0) {
        std::cout << "[Rank 0] ranks " << p << std::endl;
        std::cout << "[Rank 0] config bond_dim " << args.bond_dim
                  << " max_bond_dim " << args.max_bond_dim
                  << " gates " << args.gates
                  << " gate " << args.gate
                  << " seed " << args.seed
                  << " tilesz " << args.tilesz << std::endl;
    }
    world_pg.barrier();
    rprint(r, "initialize atomic counter");
    tamm::AtomicCounterGA ac{world_pg, 1};
    ac.allocate(0);
    double t0 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    while(true) {
        long long idx = ac.fetch_add(0, 1);
        if(idx >= args.gates) break;
        rprint(r, "gate index " + std::to_string(idx));
        two_site_update<double>(args.bond_dim, args.max_bond_dim, args.tilesz, args.gate, args.seed + static_cast<unsigned long long>(idx), self_pg, r);
    }
    world_pg.barrier();
    double t1 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    double dt = t1 - t0;
    double mx = 0.0;
    world_pg.allreduce(&dt, &mx, 1, tamm::ReduceOp::max);
    if(r == 0) std::cout << "[Rank 0] time_total " << std::fixed << std::setprecision(6) << mx << " s" << std::endl;
    rprint(r, "finalize atomic counter");
    ac.deallocate();
    tamm::finalize();
    return 0;
}

