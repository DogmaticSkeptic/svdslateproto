#include <tamm/tamm.hpp>
#include <slate/slate.hh>
#ifdef I
#undef I
#endif
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <chrono>

using std::int64_t;

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

static double time_tamm_roundrobin(int64_t N, int n_tasks, int k, tamm::ProcGroup world_pg) {
    using T = double;
    int my = world_pg.rank().value();
    tamm::ProcGroup self_pg = tamm::ProcGroup::create_subgroups(world_pg, 1);
    tamm::ExecutionContext ec{self_pg, tamm::DistributionKind::dense, tamm::MemoryManagerKind::local};
    tamm::Scheduler sch{ec};
    world_pg.barrier();
    double t0 = 0.0;
    double t1 = 0.0;
    if (my == 0) {
        t0 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    }
    if (my < k) {
        for (int64_t idx = my; idx < n_tasks; idx += k) {
            size_t M = static_cast<size_t>(N);
            auto bt = static_cast<tamm::Tile>(std::min(M, size_t(164)));
            tamm::TiledIndexSpace bond{tamm::IndexSpace{tamm::range(M)}, bt};
            tamm::TiledIndexSpace phys{tamm::IndexSpace{tamm::range(2)}, 1};
            auto labels3 = bond.labels<3>("all");
            auto labels2 = phys.labels<2>("all");
            auto l = std::get<0>(labels3);
            auto b = std::get<1>(labels3);
            auto r = std::get<2>(labels3);
            auto p1 = std::get<0>(labels2);
            auto p2 = std::get<1>(labels2);
            tamm::Tensor<T> A({l, p1, b});
            tamm::Tensor<T> B({b, p2, r});
            tamm::Tensor<T> C({l, p1, p2, r});
            A.set_dense();
            B.set_dense();
            C.set_dense();
            sch.allocate(A, B, C);
            sch(A() = T(1.0));
            sch(B() = T(1.0));
            sch(C() = T(0.0));
            sch(C(l, p1, p2, r) = A(l, p1, b) * B(b, p2, r));
            sch.deallocate(A, B, C);
            sch.execute(ec.exhw(), false);
        }
    }
    world_pg.barrier();
    if (my == 0) {
        t1 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    }
    return (my == 0 ? (t1 - t0) : 0.0);
}

static double time_slate_roundrobin(int64_t N, int n_tasks, int k, tamm::ProcGroup world_pg) {
    int my = world_pg.rank().value();
    const int64_t n = 2 * N;
    const int64_t nb = 192;
    tamm::ProcGroup self_pg = tamm::ProcGroup::create_subgroups(world_pg, 1);
    world_pg.barrier();
    double t0 = 0.0;
    double t1 = 0.0;
    if (my == 0) {
        t0 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    }
    if (my < k) {
        for (int64_t idx = my; idx < n_tasks; idx += k) {
            slate::Matrix<double> A(n, n, nb, 1, 1, self_pg.comm());
            A.insertLocalTiles();
            fill_local_tiles(A);
            std::vector<double> S(static_cast<size_t>(n));
            slate::Matrix<double> U, VT;
            slate::svd(A, S, U, VT, {{slate::Option::Target, slate::Target::Devices}});
        }
    }
    world_pg.barrier();
    if (my == 0) {
        t1 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    }
    return (my == 0 ? (t1 - t0) : 0.0);
}

int main(int argc, char** argv) {
    tamm::initialize(argc, argv);
    tamm::ProcGroup world_pg = tamm::ProcGroup::create_world_coll();
    int rank = world_pg.rank().value();
    int size = world_pg.size().value();
    int n_tasks = 100;
    int64_t N = 256;
    std::string csv = "scaling.csv";
    int kmax = -1;
    if (argc >= 2) n_tasks = std::stoi(argv[1]);
    if (argc >= 3) N = std::stoll(argv[2]);
    if (argc >= 4) csv = std::string(argv[3]);
    if (argc >= 5) kmax = std::stoi(argv[4]);
    if (kmax <= 0 || kmax > size) kmax = size;
    if (rank == 0) {
        std::ofstream ofs(csv, std::ios::out | std::ios::trunc);
        ofs << "k,t_tamm,t_slate\n";
        ofs.close();
        std::cout << "world " << size << " tasks " << n_tasks << " N " << N << " csv " << csv << " kmax " << kmax << std::endl;
    }
    world_pg.barrier();
    for (int k = 1; k <= kmax; ++k) {
        if (rank == 0) std::cout << "k " << k << " tamm start" << std::endl;
        double t_tamm = time_tamm_roundrobin(N, n_tasks, k, world_pg);
        if (rank == 0) std::cout << "k " << k << " tamm done " << std::fixed << std::setprecision(6) << t_tamm << " s" << std::endl;
        if (rank == 0) std::cout << "k " << k << " slate start" << std::endl;
        double t_slate = time_slate_roundrobin(N, n_tasks, k, world_pg);
        if (rank == 0) std::cout << "k " << k << " slate done " << std::fixed << std::setprecision(6) << t_slate << " s" << std::endl;
        if (rank == 0) {
            std::ofstream ofs(csv, std::ios::out | std::ios::app);
            ofs << k << "," << std::fixed << std::setprecision(6) << t_tamm << "," << std::fixed << std::setprecision(6) << t_slate << "\n";
            ofs.close();
        }
        world_pg.barrier();
    }
    if (rank == 0) std::cout << "done" << std::endl;
    tamm::finalize();
    return 0;
}

