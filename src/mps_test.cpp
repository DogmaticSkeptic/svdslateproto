#include <tamm/tamm.hpp>
#include <itensor/all.h>
#include <vector>
#include <string>
#include <random>
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cuda_runtime.h>
#include <cusolverDn.h>

using i64 = long long;

struct Args {
    i64 bond_dim = 128;
    i64 max_bond_dim = 128;
    i64 gates = 256;
    std::string gate = "cnot";
    unsigned long long seed = 42;
    i64 tilesz = 256;
    double j_tol = 1e-14;
    int j_sweeps = 100;
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

struct CuCtx {
    cusolverDnHandle_t solver = nullptr;
    cusolverDnParams_t params = nullptr;
    cudaStream_t stream = nullptr;
    gesvdjInfo_t jp = nullptr;
    int lwork_jac = 0;
    double* d_work_jac = nullptr;
    void* d_work = nullptr;
    void* h_work = nullptr;
    size_t ws_dev = 0;
    size_t ws_host = 0;
    CuCtx() {
        cusolverDnCreate(&solver);
        cusolverDnCreateParams(&params);
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        cusolverDnSetStream(solver, stream);
        cusolverDnCreateGesvdjInfo(&jp);
    }
    ~CuCtx() {
        if(d_work_jac) cudaFree(d_work_jac);
        if(d_work) cudaFree(d_work);
        if(h_work) free(h_work);
        if(jp) cusolverDnDestroyGesvdjInfo(jp);
        if(params) cusolverDnDestroyParams(params);
        if(solver) cusolverDnDestroy(solver);
        if(stream) cudaStreamDestroy(stream);
    }
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
        else if(eqs(argv[i], "--j_tol") && i + 1 < argc) a.j_tol = std::atof(argv[++i]);
        else if(eqs(argv[i], "--j_sweeps") && i + 1 < argc) a.j_sweeps = std::atoi(argv[++i]);
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
static void pack_theta_to_matrix_colmajor_ptr(const tamm::Tensor<T>& Th, i64 D, T* A) {
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

static void rsvd_rankk_gpu(CuCtx& ctx, const double* A_h, int m, int n, int k, int p, int niters, std::vector<double>& S, std::vector<double>& U_row, std::vector<double>& VT_row) {
    int lda = m;
    int ldu = m;
    int ldv = n;
    double* d_A = nullptr;
    double* d_S = nullptr;
    double* d_U = nullptr;
    double* d_V = nullptr;
    int* d_info = nullptr;
    size_t Asz = size_t(lda) * size_t(n) * sizeof(double);
    cudaMalloc((void**)&d_A, Asz);
    cudaMemcpyAsync(d_A, A_h, Asz, cudaMemcpyHostToDevice, ctx.stream);
    cudaMalloc((void**)&d_S, size_t(k) * sizeof(double));
    cudaMalloc((void**)&d_U, size_t(ldu) * size_t(k) * sizeof(double));
    cudaMalloc((void**)&d_V, size_t(ldv) * size_t(k) * sizeof(double));
    cudaMalloc((void**)&d_info, sizeof(int));
    size_t ws_dev_req = 0, ws_host_req = 0;
    cusolverDnXgesvdr_bufferSize(ctx.solver, ctx.params, 'S', 'S', m, n, k, p, niters, CUDA_R_64F, d_A, lda, CUDA_R_64F, nullptr, CUDA_R_64F, nullptr, ldu, CUDA_R_64F, nullptr, ldv, CUDA_R_64F, &ws_dev_req, &ws_host_req);
    if(ws_dev_req > ctx.ws_dev) {
        if(ctx.d_work) cudaFree(ctx.d_work);
        ctx.ws_dev = ws_dev_req;
        cudaMalloc(&ctx.d_work, ctx.ws_dev);
    }
    if(ws_host_req > ctx.ws_host) {
        if(ctx.h_work) free(ctx.h_work);
        ctx.ws_host = ws_host_req;
        ctx.h_work = malloc(ctx.ws_host);
    }
    cusolverDnXgesvdr(ctx.solver, ctx.params, 'S', 'S', m, n, k, p, niters, CUDA_R_64F, d_A, lda, CUDA_R_64F, d_S, CUDA_R_64F, d_U, ldu, CUDA_R_64F, d_V, ldv, CUDA_R_64F, ctx.d_work, ctx.ws_dev, ctx.h_work, ctx.ws_host, d_info);
    cudaStreamSynchronize(ctx.stream);
    S.resize(size_t(k));
    std::vector<double> U_col(size_t(ldu) * size_t(k));
    std::vector<double> V_col(size_t(ldv) * size_t(k));
    cudaMemcpy(S.data(), d_S, sizeof(double) * size_t(k), cudaMemcpyDeviceToHost);
    cudaMemcpy(U_col.data(), d_U, sizeof(double) * size_t(ldu) * size_t(k), cudaMemcpyDeviceToHost);
    cudaMemcpy(V_col.data(), d_V, sizeof(double) * size_t(ldv) * size_t(k), cudaMemcpyDeviceToHost);
    U_row.assign(size_t(m) * size_t(k), 0.0);
    for(int i = 0; i < m; i++)
        for(int j = 0; j < k; j++)
            U_row[size_t(i) * size_t(k) + size_t(j)] = U_col[size_t(i) + size_t(ldu) * size_t(j)];
    VT_row.assign(size_t(k) * size_t(n), 0.0);
    for(int i = 0; i < k; i++)
        for(int j = 0; j < n; j++)
            VT_row[size_t(i) * size_t(n) + size_t(j)] = V_col[size_t(j) + size_t(ldv) * size_t(i)];
    cudaFree(d_info);
    cudaFree(d_V);
    cudaFree(d_U);
    cudaFree(d_S);
    cudaFree(d_A);
}

static void jacobi_full_gpu_ctx(CuCtx& ctx, const double* A_h, int m, int n, double tol, int sweeps, std::vector<double>& S, std::vector<double>& U_row, std::vector<double>& VT_row) {
    cusolverDnXgesvdjSetTolerance(ctx.jp, tol);
    cusolverDnXgesvdjSetMaxSweeps(ctx.jp, sweeps);
    int lda = m;
    int ldu = m;
    int ldv = n;
    int econ = 1;
    int k = std::min(m, n);
    double* d_A = nullptr;
    double* d_S = nullptr;
    double* d_U = nullptr;
    double* d_V = nullptr;
    int* d_info = nullptr;
    size_t Asz = size_t(lda) * size_t(n) * sizeof(double);
    cudaMalloc((void**)&d_A, Asz);
    cudaMemcpyAsync(d_A, A_h, Asz, cudaMemcpyHostToDevice, ctx.stream);
    cudaMalloc((void**)&d_S, size_t(k) * sizeof(double));
    cudaMalloc((void**)&d_U, size_t(ldu) * size_t(k) * sizeof(double));
    cudaMalloc((void**)&d_V, size_t(ldv) * size_t(k) * sizeof(double));
    cudaMalloc((void**)&d_info, sizeof(int));
    int lwork_req = 0;
    cusolverDnDgesvdj_bufferSize(ctx.solver, CUSOLVER_EIG_MODE_VECTOR, econ, m, n, d_A, lda, d_S, d_U, ldu, d_V, ldv, &lwork_req, ctx.jp);
    if(lwork_req > ctx.lwork_jac) {
        if(ctx.d_work_jac) cudaFree(ctx.d_work_jac);
        ctx.lwork_jac = lwork_req;
        cudaMalloc((void**)&ctx.d_work_jac, sizeof(double) * size_t(ctx.lwork_jac));
    }
    cusolverDnDgesvdj(ctx.solver, CUSOLVER_EIG_MODE_VECTOR, econ, m, n, d_A, lda, d_S, d_U, ldu, d_V, ldv, ctx.d_work_jac, ctx.lwork_jac, d_info, ctx.jp);
    cudaStreamSynchronize(ctx.stream);
    S.resize(size_t(k));
    std::vector<double> U_col(size_t(ldu) * size_t(k));
    std::vector<double> V_col(size_t(ldv) * size_t(k));
    cudaMemcpy(S.data(), d_S, sizeof(double) * size_t(k), cudaMemcpyDeviceToHost);
    cudaMemcpy(U_col.data(), d_U, sizeof(double) * size_t(ldu) * size_t(k), cudaMemcpyDeviceToHost);
    cudaMemcpy(V_col.data(), d_V, sizeof(double) * size_t(ldv) * size_t(k), cudaMemcpyDeviceToHost);
    U_row.assign(size_t(m) * size_t(k), 0.0);
    for(int i = 0; i < m; i++)
        for(int j = 0; j < k; j++)
            U_row[size_t(i) * size_t(k) + size_t(j)] = U_col[size_t(i) + size_t(ldu) * size_t(j)];
    VT_row.assign(size_t(k) * size_t(n), 0.0);
    for(int i = 0; i < k; i++)
        for(int j = 0; j < n; j++)
            VT_row[size_t(i) * size_t(n) + size_t(j)] = V_col[size_t(j) + size_t(ldv) * size_t(i)];
    cudaFree(d_info);
    cudaFree(d_V);
    cudaFree(d_U);
    cudaFree(d_S);
    cudaFree(d_A);
}

static TammTB two_site_update_tamm_gpu(i64 D, i64 Dmax, const std::string& gate_kind, unsigned long long seed, tamm::ProcGroup self_pg, const Args& args, CuCtx& ctx) {
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
    sch.allocate(A, B, Th, Gt).execute(tamm::ExecutionHW::GPU);
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
    sch(Th(l, p1, p2, rr) = A(l, p1, b) * B(b, p2, rr)).execute(tamm::ExecutionHW::GPU);
    tb.t_contract_ab += now_s() - tc1_0;
    tamm::Tensor<double> Th2({bond, phys, phys, bond});
    Th2.set_dense();
    double ta1 = now_s();
    sch.allocate(Th2).execute(tamm::ExecutionHW::GPU);
    tb.t_allocate += now_s() - ta1;
    double tc2_0 = now_s();
    sch(Th2(l, q1, q2, rr) = Th(l, p1, p2, rr) * Gt(p1, p2, q1, q2)).execute(tamm::ExecutionHW::GPU);
    tb.t_apply_gate += now_s() - tc2_0;
    i64 m = 2 * D;
    i64 n = 2 * D;
    double* A_pinned = nullptr;
    cudaHostAlloc((void**)&A_pinned, sizeof(double) * size_t(m) * size_t(n), cudaHostAllocDefault);
    double tpack0 = now_s();
    pack_theta_to_matrix_colmajor_ptr(Th2, D, A_pinned);
    tb.t_pack += now_s() - tpack0;
    i64 kfull = std::min<i64>(m, n);
    i64 chi = std::min<i64>(kfull, Dmax);
    std::vector<double> S;
    std::vector<double> U_row;
    std::vector<double> VT_row;
    double tsvd0 = now_s();
    if(chi < n) {
        int p_ov = std::min<int>(int(n - chi), std::max<int>(2 * int(chi), 64));
        int niters = 2;
        rsvd_rankk_gpu(ctx, A_pinned, int(m), int(n), int(chi), p_ov, niters, S, U_row, VT_row);
    } else {
        jacobi_full_gpu_ctx(ctx, A_pinned, int(m), int(n), args.j_tol, args.j_sweeps, S, U_row, VT_row);
    }
    tb.t_svd += now_s() - tsvd0;
    cudaFreeHost(A_pinned);
    i64 chi_used = chi;
    std::vector<double> Uc(static_cast<size_t>(m) * static_cast<size_t>(chi_used));
    std::vector<double> VTc(static_cast<size_t>(chi_used) * static_cast<size_t>(n));
    double ttr0 = now_s();
    for(i64 i = 0; i < m; i++)
        for(i64 j = 0; j < chi_used; j++)
            Uc[static_cast<size_t>(i) * static_cast<size_t>(chi_used) + static_cast<size_t>(j)] = U_row[static_cast<size_t>(i) * static_cast<size_t>(chi_used) + static_cast<size_t>(j)];
    for(i64 i = 0; i < chi_used; i++)
        for(i64 j = 0; j < n; j++)
            VTc[static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(j)] = VT_row[static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(j)];
    std::vector<double> Sx(static_cast<size_t>(chi_used));
    for(i64 i = 0; i < chi_used; i++) Sx[static_cast<size_t>(i)] = S[static_cast<size_t>(i)];
    tb.t_truncate += now_s() - ttr0;
    tamm::TiledIndexSpace chi_tis{tamm::IndexSpace{tamm::range(chi_used)}, static_cast<tamm::Tile>(chi_used)};
    auto s = chi_tis.label("all");
    tamm::Tensor<double> A2({bond, phys, chi_tis});
    tamm::Tensor<double> B2({chi_tis, phys, bond});
    A2.set_dense();
    B2.set_dense();
    double ta2 = now_s();
    sch.allocate(A2, B2).execute(tamm::ExecutionHW::GPU);
    tb.t_allocate += now_s() - ta2;
    double tfuv0 = now_s();
    fill_from_u_s_vt(A2, B2, D, chi_used, Uc, Sx, VTc);
    tb.t_fill_uvt += now_s() - tfuv0;
    double td0 = now_s();
    sch.deallocate(A, B, Th, Th2, Gt, A2, B2).execute(tamm::ExecutionHW::GPU);
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
    ITensor A(l, p1, bb);
    ITensor B(bb, p2, rr);
    double tbuild0 = now_s();
    A = randomITensor(l, p1, bb);
    B = randomITensor(bb, p2, rr);
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
    auto [U, S, V] = svd(Th2, IndexSet(l, q1), IndexSet(q2, rr), itensor::Args("Cutoff", 0.0, "MaxDim", int(Dmax), "SVDMethod", "gesdd"));
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
                  << " tilesz " << args.tilesz
                  << " j_tol " << args.j_tol
                  << " j_sweeps " << args.j_sweeps
                  << std::endl;
        std::cout << "time_total " << std::fixed << std::setprecision(6) << total_max_time << " s" << std::endl;
    }
}

static void print_rank_breakdown_tamm(int rank, const TammTB& tb, long long tasks) {
    std::cout << "rank " << rank
              << " tasks " << tasks
              << " TAMM+cuSOLVER total " << std::fixed << std::setprecision(6) << tb.t_total
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
    CuCtx ctx;
    double t0 = now_s();
    TammTB acc_tamm;
    ITensorTB acc_it;
    long long tasks_done = 0;
    while(true) {
        long long idx = ac.fetch_add(0, 1);
        if(idx >= args.gates) break;
        TammTB tb1 = two_site_update_tamm_gpu(args.bond_dim, args.max_bond_dim, args.gate, args.seed + static_cast<unsigned long long>(idx), self_pg, args, ctx);
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

