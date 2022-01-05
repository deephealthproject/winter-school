# Use Case 13 pipeline preparation for Winter School

## 1. Create a directory for the pipeline
```
mkdir uc13_lab
cd uc13_lab
```

## 2. Download and extract the data
You need to download the data from [here](https://upvedues-my.sharepoint.com/:u:/g/personal/salcarpo_upv_edu_es/EUByteF4uH1HorGCvqlat0QBSmT3wE2COc85i4mf80VzbA?e=yi4Yok).
It is a zip file (7.6 GB) which contains the signals prepared for this lab session. You need to download and extract it in the directory that was created before.
You can extract the file with the following command.
```
# Inside the lab directory
unzip -q clean_signals.zip
```

## 3. Download the code
In the same directory `uc13_lab/` you need to download the code. To do that, we will clone a branch of the Use Case 13 pipeline repository which has the required files.
```
git clone https://github.com/deephealthproject/UC13_pipeline --branch winter-school
```
It will generate a new directory called `UC13_pipeline` which will be the working directory for the experiments.

## 4. Activate environment
Last step is to activate the Anaconda environment (if it is not activated yet) and to install some dependencies.
```
conda activate winter-school

# Go inside the pipeline directory if you are not inside yet
cd uc13_lab/UC13_pipeline

# Install dependencies (it could be already installed by the other pipelines)
pip install -r requirements.txt
```
If you have not installed the environment, please follow the guide [here](https://github.com/deephealthproject/winter-school/blob/main/lab/01_installation/README.md).

At the end, you should have a directory structure like this:
```
uc13_lab/
|
+-- clean_signals/
|
+-- UC13_pipeline/
```


Finally, we are ready to run some experiments.
Continue with the main guide of the lab session [here](https://github.com/deephealthproject/winter-school/tree/main/lab/04_uc13_based_examples).
