#include <mpi.h>
#include <tamm/tamm.hpp>
#ifdef I
#undef I
#endif
#include <slate/slate.hh>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <iostream>

static void fill_local_tiles(slate::Matrix<double>& A) {
    for (int64_t j = 0; j < A.nt(); ++j) {
        for (int64_t i = 0; i < A.mt(); ++i) {
            if (!A.tileIsLocal(i, j)) continue;
            auto T = A(i, j);
            double* a = T.data();
            int64_t lda = T.stride();
            int64_t mb = T.mb();
            int64_t nb = T.nb();
            for (int64_t jj = 0; jj < nb; ++jj) {
                for (int64_t ii = 0; ii < mb; ++ii) {
                    double x = double(i * mb + ii);
                    double y = double(j * nb + jj);
                    a[ii + lda * jj] = std::sin(0.001 * (x + 3.0 * y));
                }
            }
        }
    }
}

static double time_tamm_contractions(int64_t N, int n_contr, tamm::ProcGroup world_pg) {
    using T = double;
    tamm::ProcGroup self_pg = tamm::ProcGroup::create_subgroups(world_pg, 1);
    tamm::ExecutionContext ec{self_pg, tamm::DistributionKind::dense, tamm::MemoryManagerKind::ga};
    tamm::Scheduler sch{ec};
    tamm::AtomicCounterGA ac{world_pg, 1};
    ac.allocate(0);
    MPI_Barrier(world_pg.comm());
    double t0 = MPI_Wtime();
    while (true) {
        int64_t idx = ac.fetch_add(0, 1);
        if (idx >= n_contr) break;
        size_t M = static_cast<size_t>(N);
        auto bt = static_cast<tamm::Tile>(std::min(M, size_t(64)));
        tamm::TiledIndexSpace bond{tamm::IndexSpace{tamm::range(M)}, bt};
        tamm::TiledIndexSpace phys{tamm::IndexSpace{tamm::range(2)}, 1};
        auto [l, b, r] = bond.labels<3>("all");
        auto [p1, p2] = phys.labels<2>("all");
        tamm::Tensor<T> A({l, p1, b});
        tamm::Tensor<T> B({b, p2, r});
        tamm::Tensor<T> C({l, p1, p2, r});
        A.set_dense();
        B.set_dense();
        C.set_dense();
        sch.allocate(A, B, C).execute();
        sch(A() = T(1.0))(B() = T(1.0))(C() = T(0.0)).execute();
        sch(C(l, p1, p2, r) = A(l, p1, b) * B(b, p2, r)).execute(ec.exhw(), false);
        sch.deallocate(A, B, C).execute();
    }
    MPI_Barrier(world_pg.comm());
    double t1 = MPI_Wtime();
    ac.deallocate();
    double tloc = t1 - t0, tglob = 0.0;
    MPI_Reduce(&tloc, &tglob, 1, MPI_DOUBLE, MPI_MAX, 0, world_pg.comm());
    return tglob;
}

static double time_slate_svds(int64_t N, int n_svd, MPI_Comm world_comm) {
    tamm::ProcGroup world_pg = tamm::ProcGroup::create_coll(world_comm);
    tamm::AtomicCounterGA ac{world_pg, 1};
    ac.allocate(0);
    MPI_Barrier(world_comm);
    double t0 = MPI_Wtime();
    int64_t n = 2 * N;
    int64_t nb = 192;
    while (true) {
        int64_t idx = ac.fetch_add(0, 1);
        if (idx >= n_svd) break;
        slate::Matrix<double> A(n, n, nb, 1, 1, MPI_COMM_SELF);
        A.insertLocalTiles();
        fill_local_tiles(A);
        std::vector<double> S(static_cast<size_t>(n));
        slate::Matrix<double> U, VT;
        slate::svd(A, S, U, VT, { { slate::Option::Target, slate::Target::Devices } });
    }
    MPI_Barrier(world_comm);
    double t1 = MPI_Wtime();
    ac.deallocate();
    double tloc = t1 - t0, tglob = 0.0;
    MPI_Reduce(&tloc, &tglob, 1, MPI_DOUBLE, MPI_MAX, 0, world_comm);
    return tglob;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    tamm::initialize(argc, argv);
    int world_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if (argc < 7) {
        if (world_rank == 0) {
            std::fprintf(stderr, "usage: %s n_contr n_svd minN maxN step csv_filename\n", argv[0]);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int n_contr = std::stoi(argv[1]);
    int n_svd = std::stoi(argv[2]);
    int64_t minN = std::stoll(argv[3]);
    int64_t maxN = std::stoll(argv[4]);
    int64_t step = std::stoll(argv[5]);
    std::string csv_name = argv[6];
    tamm::ProcGroup world_pg = tamm::ProcGroup::create_world_coll();
    if (world_rank == 0) {
        std::ofstream ofs(csv_name, std::ios::out | std::ios::trunc);
        ofs << "bond_dim,t_contr,t_svd,t_total\n";
        ofs.close();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    for (int64_t N = minN; N <= maxN; N += step) {
        double t_contr = time_tamm_contractions(N, n_contr, world_pg);
        double t_svd = time_slate_svds(N, n_svd, MPI_COMM_WORLD);
        double t_total = t_contr + t_svd;
        if (world_rank == 0) {
            std::ofstream ofs(csv_name, std::ios::out | std::ios::app);
            ofs << N << ","
                << std::fixed << std::setprecision(6) << t_contr << ","
                << std::fixed << std::setprecision(6) << t_svd << ","
                << std::fixed << std::setprecision(6) << t_total << "\n";
            ofs.close();
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    tamm::finalize();
    MPI_Finalize();
    return 0;
}

