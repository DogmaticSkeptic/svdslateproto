#include <mpi.h>
#include <slate/slate.hh>
#include <vector>
#include <cmath>
#include <cstdint>
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

    int64_t m = 8192;
    int64_t n = 4096;
    int64_t nb = 256;
    if (argc >= 3) {
        m = std::stoll(argv[1]);
        n = std::stoll(argv[2]);
    }
    if (argc >= 4) {
        nb = std::stoll(argv[3]);
    }

    int p = 0, q = 0;
    best_pq(size, p, q);

    slate::Matrix<double> A(m, n, nb, p, q, MPI_COMM_WORLD);
    A.insertLocalTiles();
    fill_local_tiles(A);

    int64_t k = std::min<int64_t>(m, n);
    std::vector<double> S(k);
    slate::Matrix<double> U, VT;

    slate::svd(A, S, U, VT, {
        { slate::Option::Target, slate::Target::Devices }
    });

    if (rank == 0) {
        double smax = 0.0;
        for (double s : S) if (s > smax) smax = s;
        std::cout << "m=" << m << " n=" << n << " nb=" << nb
                  << " ranks=" << size << " p=" << p << " q=" << q
                  << " smax=" << smax << std::endl;
    }

    MPI_Finalize();
    return 0;
}


