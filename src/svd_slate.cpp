#include <mpi.h>
#include <slate/slate.hh>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>

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

static void best_pq(int size, int& p, int& q) {
    p = 1;
    for (int f = 1; f * f <= size; ++f) if (size % f == 0) p = f;
    q = size / p;
}

static double run_slate_svd_sq(int64_t n, int64_t nb, MPI_Comm comm) {
    int p = 0, q = 0, comm_size = 0;
    MPI_Comm_size(comm, &comm_size);
    best_pq(comm_size, p, q);
    slate::Matrix<double> A(n, n, nb, p, q, comm);
    A.insertLocalTiles();
    fill_local_tiles(A);
    std::vector<double> S(n);
    slate::Matrix<double> U, VT;
    MPI_Barrier(comm);
    double t0 = MPI_Wtime();
    slate::svd(A, S, U, VT, { { slate::Option::Target, slate::Target::Devices } });
    MPI_Barrier(comm);
    double t1 = MPI_Wtime();
    double tloc = t1 - t0, tglob = 0.0;
    MPI_Reduce(&tloc, &tglob, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    return tglob;
}

static double run_eigen_svd_sq(int64_t n, MPI_Comm comm) {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    double t0 = 0.0, t1 = 0.0;
    if (rank == 0) {
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> A(n, n);
        for (int64_t j = 0; j < n; ++j) {
            for (int64_t i = 0; i < n; ++i) {
                double x = double(i);
                double y = double(j);
                A(i, j) = std::sin(0.001 * (x + 3.0 * y));
            }
        }
        t0 = MPI_Wtime();
        Eigen::BDCSVD<Eigen::MatrixXd> svd(A, 0);
        t1 = MPI_Wtime();
    }
    MPI_Barrier(comm);
    double tloc = t1 - t0, tglob = 0.0;
    MPI_Reduce(&tloc, &tglob, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    return tglob;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    int world_size = 0, world_rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    const int64_t nb = 192;

    std::vector<int64_t> sizes;
    for (int64_t n = 10; n <= 100; n += 10) sizes.push_back(n);
    for (int64_t n = 200; n <= 1000; n += 100) sizes.push_back(n);

    std::vector<int> ranks_to_test = {1, 2, 3, 4};

    std::string csv_name = "svd_compare_square.csv";
    if (world_rank == 0) {
        std::ofstream ofs(csv_name, std::ios::out | std::ios::trunc);
        ofs << "size,ranks,t_slate,t_eigen\n";
        ofs.close();
    }
    MPI_Barrier(MPI_COMM_WORLD);

    for (int r : ranks_to_test) {
        if (world_size < r) continue;
        int color = (world_rank < r) ? 0 : MPI_UNDEFINED;
        MPI_Comm sub = MPI_COMM_NULL;
        MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &sub);

        if (sub != MPI_COMM_NULL) {
            int sub_rank = 0;
            MPI_Comm_rank(sub, &sub_rank);
            for (int64_t n : sizes) {
                double t_slate = run_slate_svd_sq(n, nb, sub);
                double t_eigen = run_eigen_svd_sq(n, sub);
                if (sub_rank == 0 && world_rank == 0) {
                    std::ofstream ofs(csv_name, std::ios::out | std::ios::app);
                    ofs << n << "," << r << "," << t_slate << "," << t_eigen << "\n";
                    ofs.close();
                }
            }
            MPI_Comm_free(&sub);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}

