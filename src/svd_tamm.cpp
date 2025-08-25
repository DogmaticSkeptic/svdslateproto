#include <mpi.h>
#include <tamm/tamm.hpp>
#include <slate/slate.hh>
#ifdef I
#undef I
#endif
#include <Eigen/Dense>
#include <itensor/all.h>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <cstring>
#include <random>

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
        auto bt = static_cast<tamm::Tile>(M);//std::min(M, size_t(164)));
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
        sch.allocate(A, B, C);
        sch(A() = T(1.0));
        sch(B() = T(1.0));
        sch(C() = T(0.0));
        sch(C(l, p1, p2, r) = A(l, p1, b) * B(b, p2, r));
        sch.deallocate(A, B, C);
        sch.execute(tamm::ExecutionHW::GPU, false);
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
    auto bt = static_cast<tamm::Tile>(164);
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
    const int64_t n = 2 * N;
    const int64_t nb = 256;
    tamm::ProcGroup self_pg = tamm::ProcGroup::create_subgroups(world_pg, 1);
    slate::Matrix<double> A0(n, n, nb, 1, 1, self_pg.comm());
    A0.insertLocalTiles();
    std::mt19937_64 gen(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int64_t j = 0; j < A0.nt(); ++j) {
        for (int64_t i = 0; i < A0.mt(); ++i) {
            if (!A0.tileIsLocal(i, j)) continue;
            A0.tileGetForWriting(i, j, slate::LayoutConvert::ColMajor);
            auto T = A0(i, j);
            double* a = T.data();
            int64_t lda = T.stride();
            int64_t mb = T.mb();
            int64_t nbj = T.nb();
            for (int64_t jj = 0; jj < nbj; ++jj) {
                for (int64_t ii = 0; ii < mb; ++ii) {
                    a[ii + lda * jj] = dist(gen);
                }
            }
        }
    }
    world_pg.barrier();
    double total_svd_time = 0.0;
    while (true) {
        int64_t idx = ac.fetch_add(0, 1);
        if (idx >= n_svd) break;
        slate::Matrix<double> A(n, n, nb, 1, 1, self_pg.comm());
        A.insertLocalTiles();
        for (int64_t j = 0; j < A.nt(); ++j) {
            for (int64_t i = 0; i < A.mt(); ++i) {
                if (!A.tileIsLocal(i, j)) continue;
                A0.tileGetForReading(i, j, slate::LayoutConvert::ColMajor);
                A.tileGetForWriting(i, j, slate::LayoutConvert::ColMajor);
                auto Tsrc = A0(i, j);
                auto Tdst = A(i, j);
                const double* src = Tsrc.data();
                double* dst = Tdst.data();
                int64_t lda_src = Tsrc.stride();
                int64_t lda_dst = Tdst.stride();
                int64_t mb = Tsrc.mb();
                int64_t nbj = Tsrc.nb();
                for (int64_t jj = 0; jj < nbj; ++jj) {
                    std::memcpy(dst + lda_dst * jj, src + lda_src * jj, sizeof(double) * mb);
                }
            }
        }
        std::vector<double> S(static_cast<size_t>(n));
        slate::Matrix<double> U, VT;
        auto t0 = std::chrono::high_resolution_clock::now();
        slate::svd(A, S, U, VT, {{slate::Option::Target, slate::Target::Devices}});
        auto t1 = std::chrono::high_resolution_clock::now();
        total_svd_time += std::chrono::duration<double>(t1 - t0).count();
    }
    world_pg.barrier();
    ac.deallocate();
    return (world_pg.rank().value() == 0 ? total_svd_time : 0.0);
}

static double time_eigen_svds_host(int64_t N, int n_svd, tamm::ProcGroup world_pg) {
    world_pg.barrier();
    double t0 = 0.0, t1 = 0.0;
    if (world_pg.rank().value() == 0) {
        t0 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        const int64_t n = 2 * N;
        for (int i = 0; i < n_svd; ++i) {
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> A(n, n);
            for (int64_t j = 0; j < n; ++j) {
                for (int64_t k = 0; k < n; ++k) {
                    double x = double(k);
                    double y = double(j);
                    A(k, j) = std::sin(0.001 * (x + 3.0 * y));
                }
            }
            Eigen::BDCSVD<Eigen::MatrixXd> svd(A, 0);
        }
        t1 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    }
    world_pg.barrier();
    return (world_pg.rank().value() == 0 ? (t1 - t0) : 0.0);
}

static double time_itensor_contractions_host(int64_t N, int n_contr, tamm::ProcGroup world_pg) {
    world_pg.barrier();
    double t0 = 0.0, t1 = 0.0;
    if (world_pg.rank().value() == 0) {
        t0 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        for (int i = 0; i < n_contr; ++i) {
            size_t M = static_cast<size_t>(N);
            size_t bt = std::min(M, size_t(64));
            itensor::Index l(int(M), "l");
            itensor::Index p1(2, "p1");
            itensor::Index p2(2, "p2");
            itensor::Index b(int(M), "b");
            itensor::Index r(int(M), "r");
            itensor::ITensor A(l, p1, b), B(b, p2, r), C(l, p1, p2, r);
            A.fill(1.0);
            B.fill(1.0);
            C = A * B;
        }
        t1 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    }
    world_pg.barrier();
    return (world_pg.rank().value() == 0 ? (t1 - t0) : 0.0);
}

static double time_itensor_svds_host(int64_t N, int n_svd, tamm::ProcGroup world_pg) {
    world_pg.barrier();
    double t0 = 0.0, t1 = 0.0;
    if (world_pg.rank().value() == 0) {
        t0 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        int64_t n = 2 * N;
        for (int i = 0; i < n_svd; ++i) {
            itensor::Index x(int(n), "x");
            itensor::Index y(int(n), "y");
            itensor::ITensor A(x, y);
            for (int64_t jj = 1; jj <= n; ++jj) {
                for (int64_t ii = 1; ii <= n; ++ii) {
                    double xr = double(ii - 1);
                    double yr = double(jj - 1);
                    A.set(ii, jj, std::sin(0.001 * (xr + 3.0 * yr)));
                }
            }
            auto r = itensor::svd(A, {x}, {y});
            (void)r;
        }
        t1 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    }
    world_pg.barrier();
    return (world_pg.rank().value() == 0 ? (t1 - t0) : 0.0);
}

int main(int argc, char** argv) {
    tamm::initialize(argc, argv);
    tamm::ProcGroup world_pg = tamm::ProcGroup::create_world_coll();
    int rank = world_pg.rank().value();
    int size = world_pg.size().value();
    const double time_limit = 10.0;
    bool stop_contr_queue = false;
    bool stop_contr_batch = false;
    bool stop_svd_slate = false;
    bool stop_svd_eigen = false;
    bool stop_contr_itensor = false;
    bool stop_svd_itensor = false;
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
        std::cout << "config n_contr " << n_contr << " n_svd " << n_svd << " minN " << minN << " maxN " << maxN << " step " << step << " csv " << csv << std::endl;
        std::ofstream ofs(csv, std::ios::out | std::ios::trunc);
        ofs << "bond_dim,t_contr_queue,t_contr_batch,t_svd_slate,t_svd_eigen_host,t_contr_itensor_host,t_svd_itensor_host,t_total\n";
        ofs.close();
    }
    for (int64_t N = minN; N <= maxN; N += step) {
        if (rank == 0) std::cout << "N " << N << " contractions queue start" << std::endl;
        double t_contr_q = -1.0;
        if (!stop_contr_queue) t_contr_q = 0;//time_tamm_contractions_queue(N, n_contr, world_pg);
        int exceeded = 0;
        if (t_contr_q > time_limit) exceeded = 1;
        MPI_Bcast(&exceeded, 1, MPI_INT, 0, world_pg.comm());
        if (exceeded) stop_contr_queue = true;
        if (rank == 0) std::cout << "N " << N << " contractions queue done " << std::fixed << std::setprecision(6) << t_contr_q << " s" << std::endl;
        if (rank == 0) std::cout << "N " << N << " contractions batch start" << std::endl;
        double t_contr_b = -1.0;
        if (!stop_contr_batch) t_contr_b = 0;//time_tamm_contractions_batch(N, n_contr, world_pg);
        exceeded = 0;
        if (t_contr_b > time_limit) exceeded = 1;
        MPI_Bcast(&exceeded, 1, MPI_INT, 0, world_pg.comm());
        if (exceeded) stop_contr_batch = true;
        if (rank == 0) std::cout << "N " << N << " contractions batch done " << std::fixed << std::setprecision(6) << t_contr_b << " s" << std::endl;
        if (rank == 0) std::cout << "N " << N << " svd slate start" << std::endl;
        double t_slate = -1.0;
        if (!stop_svd_slate) t_slate = time_slate_svds(N, n_svd, world_pg);
        exceeded = 0;
        if (t_slate > time_limit) exceeded = 1;
        MPI_Bcast(&exceeded, 1, MPI_INT, 0, world_pg.comm());
        if (exceeded) stop_svd_slate = true;
        if (rank == 0) std::cout << "N " << N << " svd slate done " << std::fixed << std::setprecision(6) << t_slate << " s" << std::endl;
        if (rank == 0) std::cout << "N " << N << " svd eigen host start" << std::endl;
        double t_eigen = -1.0;
        if (!stop_svd_eigen) t_eigen = 0;//time_eigen_svds_host(N, n_svd, world_pg);
        exceeded = 0;
        if (t_eigen > time_limit) exceeded = 1;
        MPI_Bcast(&exceeded, 1, MPI_INT, 0, world_pg.comm());
        if (exceeded) stop_svd_eigen = true;
        if (rank == 0) std::cout << "N " << N << " svd eigen host done " << std::fixed << std::setprecision(6) << t_eigen << " s" << std::endl;
        if (rank == 0) std::cout << "N " << N << " itensor contractions host start" << std::endl;
        double t_it_contr = -1.0;
        if (!stop_contr_itensor) t_it_contr = 0;//time_itensor_contractions_host(N, n_contr, world_pg);
        exceeded = 0;
        if (t_it_contr > time_limit) exceeded = 1;
        MPI_Bcast(&exceeded, 1, MPI_INT, 0, world_pg.comm());
        if (exceeded) stop_contr_itensor = true;
        if (rank == 0) std::cout << "N " << N << " itensor contractions host done " << std::fixed << std::setprecision(6) << t_it_contr << " s" << std::endl;
        if (rank == 0) std::cout << "N " << N << " itensor svd host start" << std::endl;
        double t_it_svd = -1.0;
        if (!stop_svd_itensor) t_it_svd = time_itensor_svds_host(N, n_svd, world_pg);
        exceeded = 0;
        if (t_it_svd > time_limit) exceeded = 1;
        MPI_Bcast(&exceeded, 1, MPI_INT, 0, world_pg.comm());
        if (exceeded) stop_svd_itensor = true;
        if (rank == 0) std::cout << "N " << N << " itensor svd host done " << std::fixed << std::setprecision(6) << t_it_svd << " s" << std::endl;
        if (rank == 0) {
            double t_total = -1.0;
            if (t_contr_q >= 0.0 && t_slate >= 0.0) t_total = t_contr_q + t_slate;
            std::ofstream ofs(csv, std::ios::out | std::ios::app);
            ofs << N << ","
                << std::fixed << std::setprecision(6) << t_contr_q << ","
                << std::fixed << std::setprecision(6) << t_contr_b << ","
                << std::fixed << std::setprecision(6) << t_slate << ","
                << std::fixed << std::setprecision(6) << t_eigen << ","
                << std::fixed << std::setprecision(6) << t_it_contr << ","
                << std::fixed << std::setprecision(6) << t_it_svd << ","
                << std::fixed << std::setprecision(6) << t_total << "\n";
            ofs.close();
            std::cout << "N " << N << " total " << std::fixed << std::setprecision(6) << t_total << " s" << std::endl;
        }
        world_pg.barrier();
    }
    if (rank == 0) std::cout << "done" << std::endl;
    tamm::finalize();
    return 0;
}

