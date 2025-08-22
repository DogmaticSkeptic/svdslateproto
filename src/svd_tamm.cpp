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

static double time_tamm_contractions_queue(int64_t N, int n_contr, tamm::ProcGroup world_pg) {
    using T = double;
    tamm::ProcGroup self_pg = tamm::ProcGroup::create_subgroups(world_pg, 1);
    tamm::ExecutionContext ec{self_pg, tamm::DistributionKind::dense, tamm::MemoryManagerKind::ga};
    tamm::Scheduler sch{ec};
    tamm::AtomicCounterGA ac{world_pg, 1};
    ac.allocate(0);
    world_pg.barrier();
    double t0 = 0.0, t1 = 0.0;
    if (world_pg.rank().value() == 0) {
        t0 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    }
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
    world_pg.barrier();
    if (world_pg.rank().value() == 0) {
        t1 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    }
    ac.deallocate();
    return (world_pg.rank().value() == 0 ? (t1 - t0) : 0.0);
}

static double time_tamm_contractions_batch(int64_t N, int n_contr, tamm::ProcGroup world_pg) {
    using T = double;
    tamm::ExecutionContext ec{world_pg, tamm::DistributionKind::dense, tamm::MemoryManagerKind::ga};
    tamm::Scheduler sch{ec};
    size_t M = static_cast<size_t>(N);
    auto bt = static_cast<tamm::Tile>(std::min(M, size_t(64)));
    tamm::TiledIndexSpace bond{tamm::IndexSpace{tamm::range(M)}, bt};
    tamm::TiledIndexSpace phys{tamm::IndexSpace{tamm::range(2)}, 1};
    auto [l, b, r] = bond.labels<3>("all");
    auto [p1, p2] = phys.labels<2>("all");
    std::vector<tamm::Tensor<T>> A_list;
    std::vector<tamm::Tensor<T>> B_list;
    std::vector<tamm::Tensor<T>> C_list;
    A_list.reserve(static_cast<size_t>(n_contr));
    B_list.reserve(static_cast<size_t>(n_contr));
    C_list.reserve(static_cast<size_t>(n_contr));
    world_pg.barrier();
    double t0 = 0.0, t1 = 0.0;
    if (world_pg.rank().value() == 0) {
        t0 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    }
    for (int i = 0; i < n_contr; ++i) {
        A_list.emplace_back(std::initializer_list<tamm::TiledIndexLabel>{l, p1, b});
        B_list.emplace_back(std::initializer_list<tamm::TiledIndexLabel>{b, p2, r});
        C_list.emplace_back(std::initializer_list<tamm::TiledIndexLabel>{l, p1, p2, r});
        A_list.back().set_dense();
        B_list.back().set_dense();
        C_list.back().set_dense();
        sch.allocate(A_list.back(), B_list.back(), C_list.back());
        sch(A_list.back()() = T(1.0));
        sch(B_list.back()() = T(1.0));
        sch(C_list.back()() = T(0.0));
        sch(C_list.back()(l, p1, p2, r) = A_list.back()(l, p1, b) * B_list.back()(b, p2, r));
        sch.deallocate(A_list.back(), B_list.back(), C_list.back());
    }
    sch.execute(ec.exhw(), false);
    world_pg.barrier();
    if (world_pg.rank().value() == 0) {
        t1 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    }
    return (world_pg.rank().value() == 0 ? (t1 - t0) : 0.0);
}

static double time_slate_svds(int64_t N, int n_svd, tamm::ProcGroup world_pg) {
    tamm::AtomicCounterGA ac{world_pg, 1};
    ac.allocate(0);
    world_pg.barrier();
    double t0 = 0.0, t1 = 0.0;
    if (world_pg.rank().value() == 0) {
        t0 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    }
    const int64_t n = 2 * N;
    const int64_t nb = 192;
    tamm::ProcGroup self_pg = tamm::ProcGroup::create_subgroups(world_pg, 1);
    while (true) {
        int64_t idx = ac.fetch_add(0, 1);
        if (idx >= n_svd) break;
        slate::Matrix<double> A(n, n, nb, 1, 1, self_pg.comm());
        A.insertLocalTiles();
        fill_local_tiles(A);
        std::vector<double> S(static_cast<size_t>(n));
        slate::Matrix<double> U, VT;
        slate::svd(A, S, U, VT, { { slate::Option::Target, slate::Target::Devices } });
    }
    world_pg.barrier();
    if (world_pg.rank().value() == 0) {
        t1 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    }
    ac.deallocate();
    return (world_pg.rank().value() == 0 ? (t1 - t0) : 0.0);
}

int main(int argc, char** argv) {
    tamm::initialize(argc, argv);
    tamm::ProcGroup world_pg = tamm::ProcGroup::create_world_coll();
    int rank = world_pg.rank().value();
    int size = world_pg.size().value();

    int n_contr = 100;
    int n_svd = 100;
    int64_t minN = 128;
    int64_t maxN = 512;
    int64_t step = 128;
    std::string csv = "timings.csv";

    if (argc >= 2) n_contr = std::stoi(argv[1]);
    if (argc >= 3) n_svd = std::stoi(argv[2]);
    if (argc >= 4) minN = std::stoll(argv[3]);
    if (argc >= 5) maxN = std::stoll(argv[4]);
    if (argc >= 6) step = std::stoll(argv[5]);
    if (argc >= 7) csv = std::string(argv[6]);

    if (rank == 0) {
        std::cout << "ranks " << size << std::endl;
        std::cout << "config n_contr " << n_contr << " n_svd " << n_svd
                  << " minN " << minN << " maxN " << maxN
                  << " step " << step << " csv " << csv << std::endl;
        std::ofstream ofs(csv, std::ios::out | std::ios::trunc);
        ofs << "bond_dim,t_contr_queue,t_contr_batch,t_svd,t_total\n";
        ofs.close();
    }

    for (int64_t N = minN; N <= maxN; N += step) {
        if (rank == 0) std::cout << "N " << N << " contractions queue start" << std::endl;
        double t_contr_q = time_tamm_contractions_queue(N, n_contr, world_pg);
        if (rank == 0) std::cout << "N " << N << " contractions queue done " << std::fixed << std::setprecision(6) << t_contr_q << " s" << std::endl;

        if (rank == 0) std::cout << "N " << N << " contractions batch start" << std::endl;
        double t_contr_b = time_tamm_contractions_batch(N, n_contr, world_pg);
        if (rank == 0) std::cout << "N " << N << " contractions batch done " << std::fixed << std::setprecision(6) << t_contr_b << " s" << std::endl;

        if (rank == 0) std::cout << "N " << N << " svd start" << std::endl;
        double t_svd = time_slate_svds(N, n_svd, world_pg);
        if (rank == 0) std::cout << "N " << N << " svd done " << std::fixed << std::setprecision(6) << t_svd << " s" << std::endl;

        if (rank == 0) {
            double t_total = t_contr_q + t_svd;
            std::ofstream ofs(csv, std::ios::out | std::ios::app);
            ofs << N << ","
                << std::fixed << std::setprecision(6) << t_contr_q << ","
                << std::fixed << std::setprecision(6) << t_contr_b << ","
                << std::fixed << std::setprecision(6) << t_svd << ","
                << std::fixed << std::setprecision(6) << t_total << "\n";
            ofs.close();
            std::cout << "N " << N << " total " << std::fixed << std::setprecision(6) << t_total << " s" << std::endl;
        }
    }

    if (rank == 0) std::cout << "done" << std::endl;
    tamm::finalize();
    return 0;
}

