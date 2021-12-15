# PyECVL + PyEDDL installation steps

## 1. Install Anaconda
The installation process is based on anaconda to take care of all the dependencies. To install Anaconda enter [here](https://www.anaconda.com/products/individual)

## 2. Create an Anaconda environment

    conda create --name winter-school python=3.8
   
### REMEMBER to activate the conda enviroment before the next steps
    
    conda activate winter-school
    
## 3. Install PyECVL + PyEDDL
Prepare the conda channels to get the packages:

    conda config --add channels dhealth
    conda config --add channels bioconda
    conda config --add channels conda-forge
    conda config --set channel_priority strict
    
Install the libraries:

    conda install pyecvl-cudnn

Note: The pyecvl package also installs the pyeddl

If you don't have a GPU available you can install the CPU version with:

    conda install pyecvl-cpu
    
## 4. Test installation
Download the PyEDDL github repository:

    git clone https://github.com/deephealthproject/pyeddl.git
    
Go to the examples folder and execute one of them:

    cd pyeddl/examples/NN/1_MNIST
    python mnist_conv.py --gpu
    
## Sources
You can see more details about the installation and other installation methods like docker [here](https://deephealthproject.github.io/pyecvl/installation.html) 
