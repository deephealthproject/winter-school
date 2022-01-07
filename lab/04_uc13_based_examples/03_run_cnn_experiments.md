# UC13: Run experiments with the Convolutional Neural Network approach
This is a guide on how to launch the experiments with the CNN approach.
First, we explain the parameters of the scripts that will be used.
After that, an example is provided for the attendees to replicate with their own modifications.

## 0. Working directory
To execute experiments in this Use Case, we need to work inside the pipeline repository.
```
cd uc13_lab/UC13_pipeline
```
Remember to activate the environment before launching experiments.
```
conda activate winter-school
```


## 1. The training script
The training script for the convolutional approach is `train_conv.py` (located in the python directory) and has the following flags:

### General Arguments

`--index` : Path to the index file of the training set.

`--index-val` : Path to the index file of the validation set.

`--id`: Identifier of the patient, e.g. "chb01"

`--model`: Model identifier: "conv1" or "conv2". Default: conv1

`--epochs`: Number of epochs to perform. Default: 10

`--batch-size`: Batch size. Default: 20

`--gpus`: Number of gpus to use. Usage: "--gpus 1 1" (for 2 gpus). Default: 1 (one gpu)

`--lr`: Initial Learning Rate. Default: 0.0001

`--opt`: Optimizer: "adam" or "sgd". Default: adam

### Data Loader Arguments

`--window-length`: Window length (in seconds) to use for generating samples. Default: 1

`--shift`: Window shift (in seconds) to use for generating samples. Default: 0.5

### Arguments to resume an experiment (optional)General Arguments

`--resume`: Path to the experiment directory for resuming an experiment. (OPTIONAL)

`--starting-epoch`: Number of epoch in which to resume the experiment. (OPTIONAL)


You can show the help about the different flags in the terminal by using the flag `--help`:
```
python python/train_conv.py --help
```


## 2. The test script
The test script for the convolutional approach is `test_conv.py` and has the following flags:

### General Arguments

`--index` : Path to the index file of the test set.

`--id`: Identifier of the patient, e.g. "chb01"

`--model`: Model identifier: "conv1" or "conv2". Default: conv1

`--dir`: Directory of the experiment to test.

`--batch-size`: Batch size. Default: 10

`--gpus`: Number of gpus to use. Usage: "--gpus 1 1" (for 2 gpus). Default: 1 (one gpu)

`--lr`: Initial learning rate. Default: 0.0001

`--opt`: Optimizer: "adam", "sgd" or "rmsprop". Default: adam

### Data Loader Arguments

`--window-length`: Window length (in seconds) to use for generating samples. Default: 1

`--shift`: Window shift (in seconds) to use for generating samples. Default: 0.5

### Post-Inference Arguments

`--inference-window`: Length of the sliding window (in timesteps) to use after inferencing the test set. Default: 20.

`--alpha-pos`: Minimum rate of positive predicted samples inside the sliding window for triggering a transition between normal state to ictal state. Default: 0.4

`--alpha-neg`: Maximum rate of positive predicted samples in the sliding window for triggering a transition between normal state to ictal state. Default: 0.4

`--detection-threshold`: Number of seconds from the seizure onset to take the detection of the seizure as valid. Default: 20



You can show the help about the different flags in the terminal by using the flag `--help`:
```
python python/test_conv.py --help
```


## 3. Run a complete experiment
Now, we are going to run a complete experiment.
We will train the convolutional neural network model **"conv1"** for patient `chb01` using a 10-second-long sliding
window shifted every 250 ms to generate samples with a shape equal to `(2560, 23)`,
i.e., a portion of 10 seconds of the raw signal including the 23 channels.

### Training
With the following command line example, the training process will iterate 5 epochs with the optimizer Adam and with an initial learning rate equal to 0.0001.
```
# Inside uc13_lab/UC13_pipeline/
python python/train_conv.py \
    --index indexes_detection/chb01/train.txt \
    --index-val indexes_detection/chb01/validation.txt \
    --id chb01 \
    --model conv1 \
    --batch-size 20 \
    --gpus 1 \
    --lr 0.0001 \
    --opt adam \
    --window-length 10 \
    --shift 0.5
```

By running the above script, the temporary results, i.e., the evolution of the learning process, is shown on the screen.
However, this script also creates a directory named `experiments/` where models and training results are stored.
Inside the `experiments/` directory, another directory is created for each experiment including
(i) the name of the experiment,
(ii) some of the configuration hyperparameters that were used, and
(iii) the date and time when the experiment was launched.
Inside the directory of a given experiment, we can find a text file with the training results and a directory
named `models` where the best models are stored.

### Test
Once the training process is complete, we can evaluate models using the test set in order to get the results.
To do that, we need
(i) the index file of the test set,
(ii) the model identifier,
(iii) the patient identifier and
(iv) the directory of the experiment (as explained in the previous paragraph).

Next example evaluates a pre-trained model with the following post processing parameters:
a post-inference window of 20 seconds,
an **alpha_pos** ratio of 0.4,
an **alpha_neg** ratio of 0.4, and
a detection threshold of 20 seconds.


Here the command line to evaluate a previously trained model:
```
python python/test_conv.py \
    --index indexes_detection/chb01/test.txt \
    --id chb01 \
    --model lstm \
    --dir PATH_TO_THE_EXPERIMENT_DIR \
    --batch-size 64 \
    --gpus 1 \
    --window-length 1 \
    --shift 0.5 \
    --timesteps 19 \
    --inference-window 20 \
    --alpha-pos 0.4 \
    --alpha-neg 0.4 \
    --detection-threshold 20
```


When this process is executed, in addition to inferring from the test samples,
the post-processing analysis explained in the slides is also performed to act
as a detector of epileptic seizures.
In this case, the test samples are processed in strict chronological order.

By performing all the steps explained above, we will have carried out a complete experiment in the Use Case 13 with the convolutional approach.

## Proposed Exercises
We propose the attendees to run similar experiments by modifying some of the hyperparameters.
You can modify any of the hyperparameters keeping in mind that some of them are related to each other.
