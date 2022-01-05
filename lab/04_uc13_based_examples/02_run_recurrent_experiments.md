# UC13: Run experiments with the Recurrent Neural Network approach
This is a guide on how to launch the experiments with the recurrent approach. First, we explain the parameters of the scripts that will be used. After that, an example is provided for the attendees to replicate with their own modifications.

## 0. Working directory
To execute experiments in this Use Case, we need to work inside the pipeline repository.
```
cd uc13_lab/UC13_pipeline
```
Remember to activate the environment before launching experiments.


## 1. The training script
The training script for the recurrent approach is `train_recurrent_detector.py` (located in the python directory) and has the next flags:

`--index` : Path to the index file of the training set.

`--index-val` : Path to the index file of the validation set.

`--id`: Identifier of the patient, e.g. "chb01"

`--model`: Model identifier: "lstm" or "gru". Default: lstm

`--epochs`: Number of epochs to perform. Default: 1

`--batch-size`: Batch size. Default: 10

`--gpus`: Number of gpus to use. Usage: "--gpus 1 1" (for 2 gpus). Default: 1 (one gpu)

`--lr`: Initial Learning Rate. Default: 0.0001

`--opt`: Optimizer: "adam" or "sgd". Default: adam

`--window-length`: Window length (in seconds) to use for generating samples. Default: 1

`--shift`: Window shift (in seconds) to use for generating samples. Default: 0.5

`--timesteps`: Number of time steps that will form a sequence. Default: 19

`--resume`: Path to the experiment directory for resuming an experiment. (OPTIONAL)

`--starting-epoch`: Number of epoch in which to resume the experiment. (OPTIONAL)


You can show the flags in the terminal by using the flag `--help`:
```
python python/train_recurrent_detector.py --help
```


## 2. The test script
The test script for the recurrent approach is `test_recurrent_detector.py` and has the next flags:

`--index` : Path to the index file of the test set.

`--id`: Identifier of the patient, e.g. "chb01"

`--model`: Model identifier: "lstm" or "gru". Default: lstm

`--dir`: Directory of the experiment to test.

`--batch-size`: Batch size. Default: 10

`--gpus`: Number of gpus to use. Usage: "--gpus 1 1" (for 2 gpus). Default: 1 (one gpu)

`--lr`: Initial learning rate. Default: 0.0001

`--opt`: Optimizer: "adam" or "sgd". Default: adam

`--window-length`: Window length (in seconds) to use for generating samples. Default: 1

`--shift`: Window shift (in seconds) to use for generating samples. Default: 0.5

`--timesteps`: Number of time steps that will form a sequence. Default: 19

`--inference-window`: Length of the sliding window (in timesteps) to use after inferencing the test set. Default: 20.

`--alpha-pos`: Minimum rate of positive predicted samples inside the sliding window for triggering a transition between normal state to ictal state. Default: 0.4

`--alpha-neg`: Maximum rate of positive predicted samples in the sliding window for triggering a transition between normal state to ictal state. Default: 0.4

`--detection-threshold`: Number of seconds from the seizure onset to take the detection of the seizure as valid. Default: 20


You can show the flags in the terminal by using the flag `--help`:
```
python python/test_recurrent_detector.py --help
```


## 3. Run a complete experiment
Now, we are going to run a complete experiment. We will train a recurrent neural network (LSTM) model for patient `ćhb01` using a window length of 1 second to generate samples, a shift of 0.5 seconds between the windows, and we will use a sequence of 19 timesteps to feed the network. This sequence will represent 10 seconds of the original signal, as we are overlapping the sliding window a 50%. We will run 5 epochs and the optimizer and initial learning rate will be Adam and 0.0001 respectively.

### Training
To run that training experiment, we will run the following command:
```
# Inside uc13_lab/UC13_pipeline/
python python/train_recurrent_detector.py \
    --index indexes_detection/chb01/train.txt \
    --index-val indexes_detection/chb01/validation.txt \
    --id chb01 \
    --model lstm \
    --batch-size 64 \
    --gpus 1 \
    --lr 0.0001 \
    --opt adam \
    --window-length 1 \
    --shift 0.5 \
    --timesteps 19
```

When running a training script, it will output the results while training.
It will also create a directory to store the best models and training results. This will be inside the `experiments/` directory. Inside, we can find a directory for each experiment we run, with some of the parameters in the name of the directory. We can also see the time and date when the experiment was launched.
Inside an experiment directory, we can find a text file with the training results and a `models` directory where the best models are stored.

### Test
After training the model, we will perform an inference of the test set in order to get the results. To do that, we will need the index file of the test set, the model identifier, the patient identifier and the experiment directory (which is the one we have just explaned).
We will test the model with the following post processing parameters: a sliding window of 20 seconds, an alpha_pos ratio of 0.4, an alpha_neg ratio of 0.4 and a detection threshold of 20 seconds.

To run the test of the experiment, we can run the following command:
```
python python/test_recurrent_detector.py \
    --index indexes_detection/chb01/test.txt \
    --id chb01 \
    --model lstm \
    --dir PATH_TO_THE_EXPERIMENT_DIR_HERE \
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

When running this script, it will perform the inference of the test samples through the trained neural network, and it will do a post processing analysis to act as a detector.

After this, we will have launched a complete experiment on the Use Case 13 with the recurrent approach.

## Proposed Exercises
We propose the attendees to run a similar experiment modifying some of the parameters. You can feel free to modify any of the parameters, taking into account that some of them are related to each other.