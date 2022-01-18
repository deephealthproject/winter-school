# PyECVL + PyEDDL installation steps

## 1. Install Anaconda

### 1.1 Winter School infrastructure (UNITO)

If you are going to perform the installation on the machines provided for the Winter School, install miniconda using the following commands:

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    chmod +x Miniconda3-latest-Linux-x86_64.sh
    ./Miniconda3-latest-Linux-x86_64.sh

### 1.2 Your personal computer

If you already have conda installed (Anaconda, miniconda,...), you can skip this step. If not, you have to install a version of conda. We suggest you try miniconda or Anaconda. For both cases, you can go [here](https://docs.conda.io/en/latest/miniconda.html) for miniconda or [here](https://www.anaconda.com/products/individual) for Anaconda and use the installer for your OS. You need to download it, execute it, and then accept the default settings of the installation.

**Note:** If you are going to install miniconda and you are using Linux, you can follow the steps for the Winter School infrastructure.

## 2. Create an Anaconda environment

    conda create --name winter-school python=3.8
   
### REMEMBER to activate the conda environment before the next steps
    
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
Download any of the examples from the lab0 related to the ECVL and EDDL environment:

* [MNIST example](https://github.com/deephealthproject/winter-school/blob/main/lab/00_ecvl_eddl_environment/scripts/eddl_mnist.py):

      wget https://raw.githubusercontent.com/deephealthproject/winter-school/main/lab/00_ecvl_eddl_environment/scripts/eddl_mnist.py
    
* [CIFAR-10 fine-grained training example](https://github.com/deephealthproject/winter-school/blob/main/lab/00_ecvl_eddl_environment/scripts/eddl_cifar_fine-grained-training.py)

      wget https://raw.githubusercontent.com/deephealthproject/winter-school/main/lab/00_ecvl_eddl_environment/scripts/eddl_cifar_fine-grained-training.py
      
* [IMDB recurrent net example](https://github.com/deephealthproject/winter-school/blob/main/lab/00_ecvl_eddl_environment/scripts/eddl_imdb.py)

      wget https://raw.githubusercontent.com/deephealthproject/winter-school/main/lab/00_ecvl_eddl_environment/scripts/eddl_imdb.py
    
To execute them you can run the script with the default settings (e.g. the MNIST example):

    python eddl_mnist.py

To see the available flags use the help command:

    python eddl_mnist.py -h

That command will output this:

    usage: eddl_mnist.py [-h] [--batch-size BATCH_SIZE] [--epochs EPOCHS] [--optimizer {Adam,SGD,RMSProp}] [--learning-rate LEARNING_RATE] [--cpu] [--onnx-output-path ONNX_OUTPUT_PATH]
                         [--from-ckpt FROM_CKPT]

    Example of a classification task using MNIST dataset

    optional arguments:
      -h, --help            show this help message and exit
      --batch-size BATCH_SIZE, -bs BATCH_SIZE
                            Size of the training batches of data (default: 100)
      --epochs EPOCHS, -e EPOCHS
                            Number of epochs to train (default: 10)
      --optimizer {Adam,SGD,RMSProp}, -opt {Adam,SGD,RMSProp}
                            Training optimizer (default: Adam)
      --learning-rate LEARNING_RATE, -lr LEARNING_RATE
                            Learning rate of the optimizer (default: 0.001)
      --cpu                 Sets CPU as the computing device (default: False)
      --onnx-output-path ONNX_OUTPUT_PATH
                            Filepath to store the best model in ONNX (default: best_model.onnx)
      --from-ckpt FROM_CKPT
                            Path to an ONNX file to use as starting point (default: )
    
## Sources
You can see more details about the installation and other installation methods like docker [here](https://deephealthproject.github.io/pyecvl/installation.html) 
