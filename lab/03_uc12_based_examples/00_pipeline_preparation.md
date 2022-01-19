# PyECVL + PyEDDL pipeline preparation steps

With this guide, you will learn how to download and prepare the code and data to execute the UC12 pipeline.

**If you are using the Winter School infrastructure (UNITO), skip steps 1 and 2**. Because you already have the code and data in your */mnt* directory. This directory is read-only, so to modify the code, we will copy the pipeline code to the home directory with the following commands:

    # Copy the code avoiding the data folder
    rsync -a --progress --exclude 'data/' /mnt/pipelines/uc12_pipeline/ ~/uc12_pipeline
    # Create a symlink to the data in the new folder
    ln -s /mnt/pipelines/uc12_pipeline/data ~/uc12_pipeline/data
    
If you are going to prepare the pipeline on your machine, follow just the following steps.

## 1. Prepare the pipeline
The Python code prepared for this lab session is in the winter school repo, to download it:

    # Inside the uc12_lab directory
    git clone https://github.com/deephealthproject/winter-school.git
    # Go to the pipeline directory
    cd winter-school/lab/03_uc12_based_examples/uc12_pipeline

## 2. Download the dataset
Create a directory to store the data:

    # Inside the winter-school/lab/03_uc12_based_examples/uc12_pipeline directory
    mkdir data

Download the dataset zip file from
[here](https://drive.google.com/uc?id=1RyYa32x9aqwd2kkQpCZ4Xa2h_VcgH3wI&export=download) (12GB)
or [from a Dropbox folder](https://www.dropbox.com/sh/vqdewy1ocqpkiu4/AADCArgKp97fLMZnV_0FZnjTa/isic_segmentation.zip?dl=0)
or [from a local UPV server](https://clocalprog.dsic.upv.es/winter-school/data/isic_segmentation.zip),
_do not worry about the certificate, you can trust it, it was generated locally by the coordinator of this winter school (Jon Ander G&oacute;mez)_.

Extract the downloaded zip file in the directory that we created **winter-school/lab/03_uc12_based_examples/uc12_pipeline/data**

Inside the extracted folder of each dataset you will find the following:

- ***isic_segmentation.yml***: The YAML configuration file for the ECVL dataset. It contains the paths to the images, their corresponding masks and the definition of the dataset splits (training, validation and test)
- ***ground_truth*** folder: Contains the masks in PNG format
- ***images_segmentation*** folder: Contains all the input images in PNG format

## 3. Install additional dependencies
### REMEMBER to activate the conda environment before the next step
If you haven't done the installation of the libraries (PyEDDL and PyECVL) you can
see how to install them [here](https://github.com/deephealthproject/winter-school/blob/main/lab/01_installation/README.md).
    
    conda activate winter-school

Install one additional dependency of the pipeline:

    pip install gdown
    
At the end you should have this directory structure inside the *uc12_pipeline* directory:

    uc12_pipeline
    |
    +-- data
    |   |
    |   +-- isic_segmentation
    |
    +-- lib
    |   |
    |   +-- models.py
    |   +-- utils.py
    |
    +-- skin_lesion_segmentation.py
