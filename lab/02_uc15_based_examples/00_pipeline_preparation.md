# PyECVL + PyEDDL pipeline preparation steps

## 1. Create a pipeline directory
We will use this directory to store the dataset and the pipeline code.

    mkdir uc15_lab
    cd uc15_lab

## 2. Download the dataset
Create a directory to store the data:

    # Inside the uc15_lab directory
    mkdir data

The dataset has two versions, one with the images with size 256x256 and other with size 512x512. Each version is in a separate zip
file that you can download [here](https://upvedues-my.sharepoint.com/:f:/g/personal/salcarpo_upv_edu_es/ErvSniya5ndOsXE-T3mffTEBsc0aaW4MjMGpGWqhT8VUwg?e=KLByo0).

Download the files **uc15_images-256x256.zip** (1.1GB) and **uc15_images-512x512.zip** (3.6GB) and extract them in the directory that we created **uc15_lab/data**

Inside the extracted folder of each dataset you will find the following:

- ***ecvl_256x256_normal-vs-covid_only-r0.yaml***: The YAML configuration file for the ECVL dataset. It contains the paths to the images, their corresponding labels and the definition of the dataset splits (training, validation and test)
- ***ecvl_256x256_normal-vs-covid.yaml***: Is a file of the same type as *ecvl_256x256_normal-vs-covid_only-r0.yaml*, but in this case the dataset was augmented by replicating each sample 15 times applying some data augmentation to increase the size of the dataset. The augmentations applied here are different from the ones used in the training pipeline
- ***data*** folder: This directory contains the original CSV files with the dataset partitioning (made by us) from which we created the ECVL YAML files
- ***images*** folder: It contains all the images in PNG format

**Note:** The file structure and dataset partitioning is the same in both datasets, the only difference is the size of the images (256x256 and 512x512).
    
## 3. Prepare the pipeline
Download the GitHub repository of the UC15 pipeline:

    # Inside the uc15_lab directory
    git clone https://github.com/deephealthproject/UC15_pipeline.git
    
### REMEMBER to activate the conda enviroment before the next step
If you haven't done the installation of the libraries (PyEDDL and PyECVL) you can
see how to install them [here](https://github.com/deephealthproject/winter-school/blob/main/lab/01_installation/README.md).
    
    conda activate winter-school

Install the additional dependencies of the pipeline:

    cd UC15_pipeline
    pip install -r requirements.txt
    
At the end you should have this directory structure:

    uc15_lab
    |
    +-- data
    |   |
    |   +-- 256x256
    |   +-- 512x512
    |
    +-- UC15_pipeline
