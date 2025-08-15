# svdslateproto
Prototype of svdslate

Load the setuptamm bash script to start
# Load Environment
```bash
module load spack/0.22
spack env activate cuda
spack load slate lapackpp blaspp
cd ~/AaronForkNWQ-Sim/NWQ-Sim/
source environment/setup_tamm_perlmutter.sh
cd ~/svdslateproto
module load gcc
```


# Build

```bash
mkdir build
cd build

cmake -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=CC ..
make -j
```

# Run 

```bash
export LD_LIBRARY_PATH=/opt/cray/pe/gcc/12.2.0/snos/lib64:$LD_LIBRARY_PATH
export LD_PRELOAD=/opt/cray/pe/gcc/12.2.0/snos/lib64/libgcc_s.so.1:/opt/cray/pe/gcc/12.2.0/snos/lib64/libstdc++.so.6
srun -N 1 -C gpu -n 4 --ntasks-per-node=4 -c 32 --gpus-per-task=1 --cpu-bind=cores ./svd_slate
```
