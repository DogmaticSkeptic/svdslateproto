#include <mpi.h>
#include <slate/slate.hh>
#include <tamm/tamm.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <cstdint>
#include <cmath>
#include <algorithm>

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
        tamm::Tile bt = static_cast<tamm::Tile>(std::min(M, size_t(64)));
        tamm::TiledIndexSpace bond{tamm::IndexSpace{tamm::range(M)}, bt};
        tamm::TiledIndexSpace phys{tamm::IndexSpace{tamm::range(2)}, 1};
        auto lbr = bond.labels<3>("all");
        auto p12 = phys.labels<2>("all");
        auto l = lbr[0];
        auto b = lbr[1];
        auto r = lbr[2];
        auto p1 = p12[0];
        auto p2 = p12[1];
        tamm::Tensor<T> A({l, p1, b}), B({b, p2, r}), C({l, p1, p2, r});
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
    double tloc = t1 - t0;
    double tglob = 0.0;
    MPI_Reduce(&tloc, &tglob, 1, MPI_DOUBLE, MPI_MAX, 0, world_pg.comm());
    return tglob;
}

static double time_slate_svds(int64_t N, int n_svd, MPI_Comm world_comm) {
    tamm::ProcGroup world_pg{world_comm};
    tamm::AtomicCounterGA ac{world_pg, 1};
    ac.allocate(0);
    MPI_Barrier(world_comm);
    double t0 = MPI_Wtime();
    int64_t n = 2 * N;
    int64_t nb = 192;
    while (true) {
        int64_t idx = ac.fetch_add(0, 1);
        if (idx >= n_svd) break;
        MPI_Comm self = MPI_COMM_SELF;
        slate::Matrix<double> A(n, n, nb, 1, 1, self);
        A.insertLocalTiles();
        fill_local_tiles(A);
        std::vector<double> S(static_cast<size_t>(n));
        slate::Matrix<double> U, VT;
        slate::svd(A, S, U, VT, { { slate::Option::Target, slate::Target::Devices } });
    }
    MPI_Barrier(world_comm);
    double t1 = MPI_Wtime();
    ac.deallocate();
    double tloc = t1 - t0;
    double tglob = 0.0;
    MPI_Reduce(&tloc, &tglob, 1, MPI_DOUBLE, MPI_MAX, 0, world_comm);
    return tglob;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    tamm::initialize(argc, argv);
    int world_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if (argc < 8) {
        if (world_rank == 0) {
            std::fprintf(stderr, "usage: %s n_contr n_svd minN maxN step csv_filename repetitions\n", argv[0]);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int n_contr = std::stoi(argv[1]);
    int n_svd = std::stoi(argv[2]);
    int64_t minN = std::stoll(argv[3]);
    int64_t maxN = std::stoll(argv[4]);
    int64_t step = std::stoll(argv[5]);
    std::string csv_name = argv[6];
    int reps = std::stoi(argv[7]);
    tamm::ProcGroup world_pg = tamm::ProcGroup::create_world_coll();
    if (world_rank == 0) {
        std::ofstream ofs(csv_name, std::ios::out | std::ios::trunc);
        ofs << "bond_dim,t_contr,t_svd,t_total\n";
        ofs.close();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    for (int64_t N = minN; N <= maxN; N += step) {
        double t_contr_acc = 0.0;
        double t_svd_acc = 0.0;
        for (int r = 0; r < reps; ++r) {
            double t_contr = time_tamm_contractions(N, n_contr, world_pg);
            double t_svd = time_slate_svds(N, n_svd, MPI_COMM_WORLD);
            t_contr_acc += t_contr;
            t_svd_acc += t_svd;
        }
        double t_contr_avg = t_contr_acc / double(reps);
        double t_svd_avg = t_svd_acc / double(reps);
        double t_total = t_contr_avg + t_svd_avg;
        if (world_rank == 0) {
            std::ofstream ofs(csv_name, std::ios::out | std::ios::app);
            ofs << N << ","
                << std::fixed << std::setprecision(6) << t_contr_avg << ","
                << std::fixed << std::setprecision(6) << t_svd_avg << ","
                << std::fixed << std::setprecision(6) << t_total << "\n";
            ofs.close();
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    tamm::finalize();
    MPI_Finalize();
    return 0;
}

