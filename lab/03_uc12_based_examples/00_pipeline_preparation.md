# PyECVL + PyEDDL pipeline preparation steps

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

Download the dataset zip file from [here](https://drive.google.com/uc?id=1RyYa32x9aqwd2kkQpCZ4Xa2h_VcgH3wI&export=download) (12GB).

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
