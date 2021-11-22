# WinterSchool2021

Winter school exercies


## Install EDDL

```
# Download source code
git clone https://github.com/deephealthproject/eddl.git
cd eddl/

# Install dependencies
conda env create -f environment.yml
conda activate eddl

# Build and install
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX

make install
```

**Documentation:** [here](https://deephealthproject.github.io/eddl/intro/installation.html)


## Install ECVL

```
# Download source code
git clone https://github.com/deephealthproject/ecvl.git

# Build and install
mkdir build
cd build
cmake ..
cmake --build . --config Release --parallel 4
cmake --build . --config Release --target install
```

**Documentation:** [here](https://deephealthproject.github.io/ecvl/)
