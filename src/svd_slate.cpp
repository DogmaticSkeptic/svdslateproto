#include <mpi.h>
#include <slate/slate.hh>
#include <vector>
#include <cmath>
#include <cstdint>
#include <iostream>
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

static void best_pq(int size, int& p, int& q) {
    p = 1;
    for (int f = 1; f * f <= size; ++f) if (size % f == 0) p = f;
    q = size / p;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int size = 0, rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const int64_t nb = 192;

    std::vector<std::pair<int64_t,int64_t>> cases;
    if ((argc - 1) >= 2 && ((argc - 1) % 2 == 0)) {
        for (int i = 1; i < argc; i += 2) {
            int64_t m = std::stoll(argv[i]);
            int64_t n = std::stoll(argv[i + 1]);
            cases.emplace_back(m, n);
        }
    } else {
        cases = {
            {4096, 4096},
            {8192, 4096},
            {8192, 8192},
            {12288, 6144},
            {16384, 8192}
        };
    }

    int p = 0, q = 0;
    best_pq(size, p, q);

    if (rank == 0) {
        std::cout << "m n nb ranks p q seconds smax" << std::endl;
    }

    for (auto [m, n] : cases) {
        slate::Matrix<double> A(m, n, nb, p, q, MPI_COMM_WORLD);
        A.insertLocalTiles();
        fill_local_tiles(A);

        int64_t k = std::min<int64_t>(m, n);
        std::vector<double> S(k);
        slate::Matrix<double> U, VT;

        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        slate::svd(A, S, U, VT, { { slate::Option::Target, slate::Target::Devices } });
        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime();

        double local = t1 - t0, global = 0.0;
        MPI_Reduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            double smax = 0.0;
            for (double s : S) if (s > smax) smax = s;
            std::cout << m << " " << n << " " << nb << " "
                      << size << " " << p << " " << q << " "
                      << global << " " << smax << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}

