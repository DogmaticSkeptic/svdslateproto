#include <tamm/tamm.hpp>
#include <slate/slate.hh>
#include <chrono>
#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <cstring>

#ifdef I
#undef I
#endif
#include <Eigen/Dense>

// --- C BINDINGS for BLACS library ---
extern "C" {
    void Cblacs_get(int, int, int*);
    void Cblacs_gridinit(int*, const char*, int, int);
    void Cblacs_gridinfo(int, int*, int*, int*, int*);
    void Cblacs_gridexit(int);
}

using namespace tamm;
using T = double;

/**
 * @brief Computes the number of rows or columns a process owns for a
 *        ScaLAPACK-style block-cyclic distribution. This is a standard utility
 *        needed for wrapping raw buffers for SLATE/ScaLAPACK.
 */
inline int64_t numroc(int64_t n, int64_t nb, int iproc, int isrc, int64_t nprocs) {
    int64_t num_blocks = n / nb;
    int64_t remainder = n % nb;
    int64_t p_coord = (iproc - isrc + nprocs) % nprocs;
    int64_t num_blocks_p = num_blocks / nprocs;
    int64_t local_dim = num_blocks_p * nb;
    if (p_coord < (num_blocks % nprocs)) {
        local_dim += nb;
    } else if (p_coord == (num_blocks % nprocs)) {
        local_dim += remainder;
    }
    return local_dim;
}

/**
 * @brief Efficiently redistributes a 2D TAMM dense tensor to a SLATE BlockCyclic matrix.
 * 
 * @param tamm_tensor The source TAMM tensor (must be dense and 2D).
 * @param slate_matrix The destination SLATE matrix (must be pre-created with the correct dimensions).
 */
void tamm_to_slate(const Tensor<T>& tamm_tensor, slate::Matrix<T>& slate_matrix) {
    EXPECTS(tamm_tensor.num_modes() == 2);
    EXPECTS(tamm_tensor.kind() == TensorBase::TensorKind::dense);

    auto ec = tamm_tensor.execution_context();
    auto pg = ec->pg(); // Returns a copy, which is fine here
    pg.barrier();

    // Use const_cast if ga_handle() is not const in your TAMM version
    int ga_handle = const_cast<Tensor<T>&>(tamm_tensor).ga_handle();

    for (int64_t j = 0; j < slate_matrix.nt(); ++j) {
        for (int64_t i = 0; i < slate_matrix.mt(); ++i) {
            if (slate_matrix.tileIsLocal(i, j)) {
                slate_matrix.tileGetForWriting(i, j, slate::LayoutConvert::ColMajor);
                auto tile = slate_matrix(i, j);
                T* tile_buf = tile.data();
                
                int64_t global_i = i * slate_matrix.mb();
                int64_t global_j = j * slate_matrix.nb();
                
                std::vector<T> temp_row_buffer(tile.nb());
                for(int64_t row_idx = 0; row_idx < tile.mb(); ++row_idx) {
                    int64_t get_lo[] = {global_i + row_idx, global_j};
                    int64_t get_hi[] = {global_i + row_idx, global_j + tile.nb() - 1};
                    NGA_Get64(ga_handle, get_lo, get_hi, temp_row_buffer.data(), nullptr);
                    for(int64_t col_idx = 0; col_idx < tile.nb(); ++col_idx) {
                        tile_buf[row_idx + col_idx * tile.stride()] = temp_row_buffer[col_idx];
                    }
                }
            }
        }
    }
    pg.barrier();
}

/**
 * @brief Efficiently redistributes a SLATE BlockCyclic matrix to a TAMM dense tensor.
 */
void slate_to_tamm(slate::Matrix<T>& slate_matrix, Tensor<T>& tamm_tensor) {
    EXPECTS(tamm_tensor.num_modes() == 2);
    EXPECTS(tamm_tensor.kind() == TensorBase::TensorKind::dense);

    auto ec = tamm_tensor.execution_context();
    auto pg = ec->pg();
    pg.barrier();

    int ga_handle = tamm_tensor.ga_handle();

    for (int64_t j = 0; j < slate_matrix.nt(); ++j) {
        for (int64_t i = 0; i < slate_matrix.mt(); ++i) {
            if (slate_matrix.tileIsLocal(i, j)) {
                slate_matrix.tileGetForReading(i, j, slate::LayoutConvert::ColMajor);
                auto tile = slate_matrix(i, j);
                const T* tile_buf = tile.data();
                
                int64_t global_i = i * slate_matrix.mb();
                int64_t global_j = j * slate_matrix.nb();
                
                std::vector<T> temp_row_buffer(tile.nb());
                
                for (int64_t row_idx = 0; row_idx < tile.mb(); ++row_idx) {
                    for (int64_t col_idx = 0; col_idx < tile.nb(); ++col_idx) {
                        temp_row_buffer[col_idx] = tile_buf[row_idx + col_idx * tile.stride()];
                    }
                    int64_t put_lo[] = {global_i + row_idx, global_j};
                    int64_t put_hi[] = {global_i + row_idx, global_j + tile.nb() - 1};
                    NGA_Put64(ga_handle, put_lo, put_hi, temp_row_buffer.data(), nullptr);
                }
            }
        }
    }
    pg.barrier();
}


static double mps_gate_simulation_subgrouped(int64_t N, int n_gates, int n_subgroups, tamm::ProcGroup world_pg) {
    tamm::AtomicCounterGA ac{world_pg, 1};
    ac.allocate(0);
    world_pg.barrier();

    int world_size = world_pg.size().value();
    if (world_size % n_subgroups != 0) {
        if (world_pg.rank() == 0) {
            std::cerr << "Error: Total ranks (" << world_size 
                      << ") must be divisible by subgroups (" << n_subgroups << ")." << std::endl;
        }
        tamm::finalize(); exit(1);
    }
    int ranks_per_subgroup = world_size / n_subgroups;
    tamm::ProcGroup subgroup_pg = tamm::ProcGroup::create_subgroups(world_pg, ranks_per_subgroup);

    if (!subgroup_pg.is_valid()) {
        ac.deallocate();
        subgroup_pg.destroy_coll();
        return 0.0;
    }

    tamm::ExecutionContext ec{subgroup_pg, tamm::DistributionKind::dense, tamm::MemoryManagerKind::ga};
    tamm::Scheduler sch{ec};

    int npr_sub = std::max(1, (int)std::sqrt(ranks_per_subgroup));
    int npc_sub = ranks_per_subgroup / npr_sub;
    while(npr_sub * npc_sub != ranks_per_subgroup) { npr_sub--; npc_sub = ranks_per_subgroup / npr_sub; }
    
    int blacs_context;
    char order = 'R';
    Cblacs_get(-1, 0, &blacs_context);
    Cblacs_gridinit(&blacs_context, &order, npr_sub, npc_sub);
    int my_prow, my_pcol;
    Cblacs_gridinfo(blacs_context, &npr_sub, &npc_sub, &my_prow, &my_pcol);

    const size_t M = static_cast<size_t>(N);
    const tamm::Tile tile_size = static_cast<tamm::Tile>(std::min((size_t)128, M));
    tamm::TiledIndexSpace bond{tamm::IndexSpace{tamm::range(M)}, tile_size};
    tamm::TiledIndexSpace phys{tamm::IndexSpace{tamm::range(2)}, 2};
    auto [l, b, r] = bond.labels<3>("all");
    auto [p1, p2, p1_new, p2_new] = phys.labels<4>("all");

    world_pg.barrier();
    double t0 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    while (true) {
        int64_t task_idx = -1;
        if (subgroup_pg.rank() == 0) task_idx = ac.fetch_add(0, 1);
        subgroup_pg.broadcast(&task_idx, 0);
        if (task_idx >= n_gates) break;

        Tensor<T> M1({l, p1, b}), M2({b, p2, r}), Theta({l, p1, p2, r});
        Tensor<T> Gate({p1_new, p2_new, p1, p2}), Theta_prime({l, p1_new, p2_new, r});
        M1.set_dense(); M2.set_dense(); Theta.set_dense(); Gate.set_dense(); Theta_prime.set_dense();
        sch.allocate(M1, M2, Theta, Gate, Theta_prime);
        tamm::random_ip(M1); tamm::random_ip(M2);
        sch(Theta(l, p1, p2, r) = M1(l, p1, b) * M2(b, p2, r));
        sch(Gate() = 0.0).execute();
        
        if (ec.pg().rank() == 0) {
            Gate.put({0,0,0,0}, {1.0}); // |00> -> |00>
            Gate.put({0,1,0,1}, {1.0}); // |01> -> |01>
            Gate.put({1,1,1,0}, {1.0}); // |10> -> |11>
            Gate.put({1,0,1,1}, {1.0}); // |11> -> |10>
        }
        ec.pg().barrier();

        sch(Theta_prime(l, p1_new, p2_new, r) = Gate(p1_new, p2_new, p1, p2) * Theta(l, p1, p2, r));
        sch.execute();

        const int64_t matrix_rows = N * 2;
        const int64_t matrix_cols = 2 * N;
        const int64_t nb_slate = 128;

        Tensor<T> Theta_prime_bc;
        Theta_prime_bc.set_block_cyclic({npr_sub, npc_sub}, {nb_slate, nb_slate});
        sch.allocate(Theta_prime_bc).execute();
        tamm::to_block_cyclic_tensor(Theta_prime, Theta_prime_bc);

        int64_t local_rows = numroc(matrix_rows, nb_slate, my_prow, 0, npr_sub);
        slate::Matrix<T> Theta_slate = slate::Matrix<T>::fromScaLAPACK(
            matrix_rows, matrix_cols, Theta_prime_bc.access_local_buf(), local_rows, nb_slate,
            blacs_context, npr_sub, npc_sub
        );

        std::vector<T> S_vec(std::min(matrix_rows, matrix_cols));
        slate::Matrix<T> U, VT;
        slate::svd(Theta_slate, S_vec, U, VT);
        
        int64_t new_bond_dim = std::min((int64_t)N, (int64_t)S_vec.size());
        
        tamm::TiledIndexSpace new_bond{tamm::IndexSpace{tamm::range(new_bond_dim)}, new_bond_dim};
        auto [b_new] = new_bond.labels<1>("all");
        Tensor<T> M1_new({l, p1, b_new}), M2_new({b_new, p2, r});
        M1_new.set_dense(); M2_new.set_dense();
        sch.allocate(M1_new, M2_new).execute();

        slate_to_tamm(U, M1_new);
        for(int64_t i = 0; i < new_bond_dim; ++i) {
            auto tile = VT.sub(i, i, 0, VT.n()-1);
            slate::scale(S_vec[i], tile);
        }
        slate_to_tamm(VT, M2_new);
        
        sch.deallocate(M1, M2, Theta, Gate, Theta_prime, Theta_prime_bc, M1_new, M2_new).execute();
    }

    world_pg.barrier();
    double t1 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    double local_duration = t1 - t0;
    
    ac.deallocate();
    Cblacs_gridexit(blacs_context);
    subgroup_pg.destroy_coll();

    double max_duration = 0.0;
    world_pg.allreduce(&local_duration, &max_duration, 1, tamm::ReduceOp::max);
    
    return max_duration; 
}


int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <bond_dimension> <num_gates> <num_subgroups>" << std::endl;
        return 1;
    }
    
    int64_t N = 0; int n_gates = 0; int n_subgroups = 0;
    try {
        N = std::stoll(argv[1]);
        n_gates = std::stoi(argv[2]);
        n_subgroups = std::stoi(argv[3]);
    } catch (const std::exception& e) {
        std::cerr << "Error: Invalid arguments. Please provide integers." << std::endl;
        return 1;
    }

    tamm::initialize(argc, argv);
    tamm::ProcGroup world_pg = tamm::ProcGroup::create_world_coll();

    if (world_pg.rank().value() == 0) {
        std::cout << "-----------------------------------------------------" << std::endl;
        std::cout << " MPS 2-Site Gate Simulation Benchmark (Subgrouped)" << std::endl;
        std::cout << "-----------------------------------------------------" << std::endl;
        std::cout << " Total MPI Ranks:  " << world_pg.size() << std::endl;
        std::cout << " Subgroups:        " << n_subgroups << std::endl;
        std::cout << " Ranks/Subgroup:   " << world_pg.size().value() / n_subgroups << std::endl;
        std::cout << " Bond Dimension:   " << N << std::endl;
        std::cout << " Total Gates:      " << n_gates << std::endl;
        std::cout << "-----------------------------------------------------" << std::endl;
    }

    double total_time = mps_gate_simulation_subgrouped(N, n_gates, n_subgroups, world_pg);

    if (world_pg.rank().value() == 0) {
        std::cout << "-----------------------------------------------------" << std::endl;
        std::cout << "Total Wall-Clock Time: " << std::fixed << std::setprecision(6) << total_time << " s" << std::endl;
        std::cout << "Throughput:            " << std::fixed << std::setprecision(2) << n_gates / total_time << " gates/sec" << std::endl;
        std::cout << "-----------------------------------------------------" << std::endl;
    }

    tamm::finalize();
    return 0;
}
