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

static double now_seconds() {
    return std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

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

static double time_tamm_queue_active(int64_t N, int n_tasks, int active_ranks, tamm::ProcGroup world_pg) {
    using T = double;
    int my = world_pg.rank().value();
    int sz = world_pg.size().value();
    std::cout.setf(std::ios::unitbuf);
    std::cout << "rank " << my << " tamm_enter N " << N << " n_tasks " << n_tasks << " active_ranks " << active_ranks << " world_size " << sz << std::endl;
    tamm::ProcGroup self_pg = tamm::ProcGroup::create_subgroups(world_pg, 1);
    std::cout << "rank " << my << " tamm_after_create_subgroups" << std::endl;
    tamm::ExecutionContext ec{self_pg, tamm::DistributionKind::dense, tamm::MemoryManagerKind::local};
    std::cout << "rank " << my << " tamm_after_exec_context" << std::endl;
    tamm::Scheduler sch{ec};
    std::cout << "rank " << my << " tamm_after_scheduler" << std::endl;
    tamm::AtomicCounterGA ac{world_pg, 1};
    std::cout << "rank " << my << " tamm_before_ac_allocate" << std::endl;
    ac.allocate(0);
    std::cout << "rank " << my << " tamm_after_ac_allocate" << std::endl;
    std::cout << "rank " << my << " tamm_barrier_pre" << std::endl;
    world_pg.barrier();
    std::cout << "rank " << my << " tamm_barrier_post" << std::endl;
    double t0 = 0.0;
    double t1 = 0.0;
    if (my == 0) t0 = now_seconds();
    if (world_pg.rank().value() < active_ranks) {
        std::cout << "rank " << my << " tamm_worker_begin" << std::endl;
        while (true) {
            int64_t idx = ac.fetch_add(0, 1);
            std::cout << "rank " << my << " tamm_fetch idx " << idx << std::endl;
            if (idx >= n_tasks) {
                std::cout << "rank " << my << " tamm_worker_done_no_more_tasks" << std::endl;
                break;
            }
            size_t M = static_cast<size_t>(N);
            auto bt = static_cast<tamm::Tile>(std::min(M, size_t(164)));
            std::cout << "rank " << my << " tamm_task_begin idx " << idx << " M " << M << " bt " << bt << std::endl;
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
            std::cout << "rank " << my << " tamm_set_dense_pre idx " << idx << std::endl;
            A.set_dense();
            B.set_dense();
            C.set_dense();
            std::cout << "rank " << my << " tamm_allocate_pre idx " << idx << std::endl;
            sch.allocate(A, B, C);
            std::cout << "rank " << my << " tamm_allocate_post idx " << idx << std::endl;
            sch(A() = T(1.0));
            sch(B() = T(1.0));
            sch(C() = T(0.0));
            std::cout << "rank " << my << " tamm_schedule_ops_post idx " << idx << std::endl;
            sch(C(l, p1, p2, r) = A(l, p1, b) * B(b, p2, r));
            std::cout << "rank " << my << " tamm_deallocate_pre idx " << idx << std::endl;
            sch.deallocate(A, B, C);
            std::cout << "rank " << my << " tamm_execute_pre idx " << idx << std::endl;
            sch.execute(ec.exhw(), false);
            std::cout << "rank " << my << " tamm_execute_post idx " << idx << std::endl;
        }
    } else {
        std::cout << "rank " << my << " tamm_nonworker_idle" << std::endl;
    }
    std::cout << "rank " << my << " tamm_barrier2_pre" << std::endl;
    world_pg.barrier();
    std::cout << "rank " << my << " tamm_barrier2_post" << std::endl;
    if (my == 0) t1 = now_seconds();
    std::cout << "rank " << my << " tamm_before_ac_deallocate" << std::endl;
    ac.deallocate();
    std::cout << "rank " << my << " tamm_after_ac_deallocate" << std::endl;
    double dt = (world_pg.rank().value() == 0 ? (t1 - t0) : 0.0);
    if (world_pg.rank().value() == 0) std::cout << "rank 0 tamm_time " << std::fixed << std::setprecision(6) << dt << " s" << std::endl;
    std::cout << "rank " << my << " tamm_exit" << std::endl;
    return dt;
}

static double time_slate_svd_queue_active(int64_t N, int n_tasks, int active_ranks, tamm::ProcGroup world_pg) {
    int my = world_pg.rank().value();
    int sz = world_pg.size().value();
    std::cout.setf(std::ios::unitbuf);
    std::cout << "rank " << my << " slate_enter N " << N << " n_tasks " << n_tasks << " active_ranks " << active_ranks << " world_size " << sz << std::endl;
    tamm::AtomicCounterGA ac{world_pg, 1};
    std::cout << "rank " << my << " slate_before_ac_allocate" << std::endl;
    ac.allocate(0);
    std::cout << "rank " << my << " slate_after_ac_allocate" << std::endl;
    std::cout << "rank " << my << " slate_barrier_pre" << std::endl;
    world_pg.barrier();
    std::cout << "rank " << my << " slate_barrier_post" << std::endl;
    double t0 = 0.0;
    double t1 = 0.0;
    if (world_pg.rank().value() == 0) t0 = now_seconds();
    if (world_pg.rank().value() < active_ranks) {
        const int64_t n = 2 * N;
        const int64_t nb = 192;
        tamm::ProcGroup self_pg = tamm::ProcGroup::create_subgroups(world_pg, 1);
        std::cout << "rank " << my << " slate_worker_begin n " << n << " nb " << nb << std::endl;
        while (true) {
            int64_t idx = ac.fetch_add(0, 1);
            std::cout << "rank " << my << " slate_fetch idx " << idx << std::endl;
            if (idx >= n_tasks) {
                std::cout << "rank " << my << " slate_worker_done_no_more_tasks" << std::endl;
                break;
            }
            std::cout << "rank " << my << " slate_task_begin idx " << idx << std::endl;
            slate::Matrix<double> A(n, n, nb, 1, 1, self_pg.comm());
            std::cout << "rank " << my << " slate_insert_tiles_pre idx " << idx << std::endl;
            A.insertLocalTiles();
            std::cout << "rank " << my << " slate_fill_tiles_pre idx " << idx << std::endl;
            fill_local_tiles(A);
            std::cout << "rank " << my << " slate_fill_tiles_post idx " << idx << std::endl;
            std::vector<double> S(static_cast<size_t>(n));
            slate::Matrix<double> U, VT;
            std::cout << "rank " << my << " slate_svd_pre idx " << idx << std::endl;
            slate::svd(A, S, U, VT, {{slate::Option::Target, slate::Target::Devices}});
            std::cout << "rank " << my << " slate_svd_post idx " << idx << std::endl;
        }
    } else {
        std::cout << "rank " << my << " slate_nonworker_idle" << std::endl;
    }
    std::cout << "rank " << my << " slate_barrier2_pre" << std::endl;
    world_pg.barrier();
    std::cout << "rank " << my << " slate_barrier2_post" << std::endl;
    if (world_pg.rank().value() == 0) t1 = now_seconds();
    std::cout << "rank " << my << " slate_before_ac_deallocate" << std::endl;
    ac.deallocate();
    std::cout << "rank " << my << " slate_after_ac_deallocate" << std::endl;
    double dt = (world_pg.rank().value() == 0 ? (t1 - t0) : 0.0);
    if (world_pg.rank().value() == 0) std::cout << "rank 0 slate_time " << std::fixed << std::setprecision(6) << dt << " s" << std::endl;
    std::cout << "rank " << my << " slate_exit" << std::endl;
    return dt;
}

int main(int argc, char** argv) {
    std::cout.setf(std::ios::unitbuf);
    tamm::initialize(argc, argv);
    tamm::ProcGroup world_pg = tamm::ProcGroup::create_world_coll();
    int rank = world_pg.rank().value();
    int size = world_pg.size().value();
    int n_tasks = 100;
    int64_t N = 256;
    std::string csv = "scaling.csv";
    if (argc >= 2) n_tasks = std::stoi(argv[1]);
    if (argc >= 3) N = std::stoll(argv[2]);
    if (argc >= 4) csv = std::string(argv[3]);
    if (rank == 0) {
        std::ofstream ofs(csv, std::ios::out | std::ios::trunc);
        ofs << "k,t_tamm,t_slate\n";
        ofs.close();
        std::cout << "world " << size << " tasks " << n_tasks << " N " << N << " csv " << csv << std::endl;
    }
    std::cout << "rank " << rank << " main_barrier0_pre" << std::endl;
    world_pg.barrier();
    std::cout << "rank " << rank << " main_barrier0_post" << std::endl;
    for (int k = 1; k <= size; ++k) {
        if (rank == 0) std::cout << "k " << k << " tamm start" << std::endl;
        std::cout << "rank " << rank << " main_before_tamm k " << k << std::endl;
        double t_tamm = time_tamm_queue_active(N, n_tasks, k, world_pg);
        std::cout << "rank " << rank << " main_after_tamm k " << k << std::endl;
        if (rank == 0) std::cout << "k " << k << " tamm done " << std::fixed << std::setprecision(6) << t_tamm << " s" << std::endl;
        if (rank == 0) std::cout << "k " << k << " slate start" << std::endl;
        std::cout << "rank " << rank << " main_before_slate k " << k << std::endl;
        double t_slate = time_slate_svd_queue_active(N, n_tasks, k, world_pg);
        std::cout << "rank " << rank << " main_after_slate k " << k << std::endl;
        if (rank == 0) std::cout << "k " << k << " slate done " << std::fixed << std::setprecision(6) << t_slate << " s" << std::endl;
        if (rank == 0) {
            std::ofstream ofs(csv, std::ios::out | std::ios::app);
            ofs << k << "," << std::fixed << std::setprecision(6) << t_tamm << "," << std::fixed << std::setprecision(6) << t_slate << "\n";
            ofs.close();
        }
        std::cout << "rank " << rank << " main_barrier_loop_pre k " << k << std::endl;
        world_pg.barrier();
        std::cout << "rank " << rank << " main_barrier_loop_post k " << k << std::endl;
    }
    if (rank == 0) std::cout << "done" << std::endl;
    std::cout << "rank " << rank << " finalize_pre" << std::endl;
    tamm::finalize();
    return 0;
}

