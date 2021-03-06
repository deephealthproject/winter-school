# PyECVL + PyEDDL pipeline preparation steps

With this guide, you will learn how to download and prepare the code and data to execute the UC15 pipeline.

**If you are using the Winter School infrastructure (UNITO), skip steps 1, 2 and 3**. Because you already have the code and data in your */mnt* directory. This directory is read-only, so to modify the code, we will copy the pipeline code to the home directory with the following commands:
    
    # Copy the code avoiding the data folder
    rsync -a --progress --exclude 'data/' /mnt/pipelines/uc15_lab/ ~/uc15_lab
    # Create a symlink to the data in the new folder
    ln -s /mnt/pipelines/uc15_lab/data ~/uc15_lab/data

If you are going to prepare the pipeline on your machine, follow just the following steps.

## 1. Create a pipeline directory
We will use this directory to store the dataset and the pipeline code.

    mkdir uc15_lab
    cd uc15_lab

## 2. Download the dataset
Create a directory to store the data:

    # Inside the uc15_lab directory
    mkdir data

The dataset has two versions, one with the images with size 256x256 and other with size 512x512. Each version is in a separate zip
file that you can download from
[a sharepoint folder](https://upvedues-my.sharepoint.com/:f:/g/personal/salcarpo_upv_edu_es/ErvSniya5ndOsXE-T3mffTEBsc0aaW4MjMGpGWqhT8VUwg?e=KLByo0)
or [from a Dropbox folder](https://www.dropbox.com/sh/vqdewy1ocqpkiu4/AADIRQh_nqdtEVWJHczBdKFKa)
or [from a local UPV server](https://clocalprog.dsic.upv.es/winter-school/data/),
_do not worry about the certificate, you can trust it, it was generated locally by the coordinator of this winter school (Jon Ander G&oacute;mez)_.

Download the files **uc15_images-256x256.zip** (1.1GB) and **uc15_images-512x512.zip** (3.6GB) and extract them in the directory that we created **uc15_lab/data**

Inside the extracted folder of each dataset you will find the following:

- ***ecvl_256x256_normal-vs-covid_only-r0.yaml***: The YAML configuration file for the ECVL dataset. It contains the paths to the images, their corresponding labels and the definition of the dataset splits (training, validation and test)
- ***ecvl_256x256_normal-vs-covid.yaml***: Is a file of the same type as *ecvl_256x256_normal-vs-covid_only-r0.yaml*, but in this case the dataset was augmented by replicating each sample 15 times applying some data augmentation to increase the size of the dataset. The augmentations applied here are different from the ones used in the training pipeline
- ***ecvl_256x256_normal-vs-covid-vs-infiltrates-vs-pneumonia_only-r0.yaml***: The same type of file as *ecvl_256x256_normal-vs-covid_only-r0.yaml*, but for 4 class multilabel classification
- ***ecvl_256x256_normal-vs-covid-vs-infiltrates-vs-pneumonia.yaml***: The same type of file as *ecvl_256x256_normal-vs-covid.yaml*, but for 4 class multilabel classification
- ***data*** folder: This directory contains the original CSV files with the dataset partitioning (made by us) from which we created the ECVL YAML files
- ***images*** folder: It contains all the images in PNG format

**Note:** The file structure and dataset partitioning is the same in both datasets, the only difference is the size of the images (256x256 and 512x512).
    
## 3. Prepare the pipeline code
Download the GitHub repository of the UC15 pipeline:

    # Inside the uc15_lab directory
    git clone https://github.com/deephealthproject/UC15_pipeline.git

## 4. Install additional dependencies

### REMEMBER to activate the conda environment before the next step
If you haven't done the installation of the libraries (PyEDDL and PyECVL) you can
see how to install them [here](https://github.com/deephealthproject/winter-school/blob/main/lab/01_installation/README.md).
    
    conda activate winter-school

Install the additional dependencies of the pipeline:

    # Inside uc15_lab/UC15_pipeline directory
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
