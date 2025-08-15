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

cmake -DCMAKE_CXX_COMPILER=CC ..
make -j
```

# Run 

```bash
srun -N 1 -C gpu -n 4 --ntasks-per-node=4 -c 32 --gpus-per-task=1 --cpu-bind=cores ./svd_slate
```
