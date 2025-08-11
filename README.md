# svdslateproto
Prototype of svdslate


# Load Environment
```bash
module load cmake
module load spack/0.22
spack env activate cuda
spack load slate lapackpp blaspp
```


# Build

```bash
mkdir build
cd build

cmake -DCMAKE_CXX_COMPILER=CC ../source
make -j
```

# Run 

```bash
srun -N 1 -C gpu -n 4 --ntasks-per-node=4 -c 32 --gpus-per-task=1 --cpu-bind=cores ./svd_slate
```
