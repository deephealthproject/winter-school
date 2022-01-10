# Run inference with a pretrained model
 
## REMEMBER to activate the environment
```
conda activate winter-school
```
 
## 1. The test script
For running the inference we will use a pretrained model for the patient `chb01`. We will use a script (**test_recurrent.py**) that takes some arguments and performs the inference of the test set and prints the obtained results.

The script has many flags, that we will explore in the following guides. Just to see them as a first approach, we can do the following steps:
```
# Go to the pipeline directory (not the lab directory)
cd uc13_lab/UC13_pìpeline
# Show the available flags of the script
python python/test_recurrent_detector.py --help
```
This command will show the following output:
```
Script for testing recurrent models to detect epilepsy on UC13. 
This script loads the best model saved in the experiments directory specified and performs the inference, returning the obtained metrics.

optional arguments:
  -h, --help            show this help message and exit

General Arguments:
  --index INDEX         Index filename to use for test.
  --id ID               Id of the patient.
  --model MODEL         Model identifier. "lstm" "gru"
  --dir DIR             Directory of the experiment to test. Example: experiments/detection_recurrent_chb01_LSTM/
  --batch-size BATCH_SIZE
                        Batch size. Default -> 64
  --gpus GPUS [GPUS ...]
                        Sets the number of GPUs to use. Usage "--gpus 1 1" (two GPUs)

Data Loader Arguments:
  --window-length WINDOW_LENGTH
                        Window length  in seconds. Default -> 1
  --shift SHIFT         Window shift  in seconds. Default -> 0.5
  --timesteps TIMESTEPS
                        Timesteps to use as a  sequence. Default -> 19

Post Inference Process Arguments:
  --inference-window INFERENCE_WINDOW
                        Length of the sliding window to use for the post-inference process. Default -> 20
  --alpha-pos ALPHA_POS
                        Minimum rate of positive predicted samples in the sliding window for triggering a transition between normal state to ictal state during the post-inference process. Default -> 0.4
  --alpha-neg ALPHA_NEG
                        Maximum rate of positive predicted samples in the sliding window for triggering a transition between normal state to ictal state during the post-inference process. Default -> 0.4
  --detection-threshold DETECTION_THRESHOLD
                        Number of seconds from the seizure onset to allow the detection. Default -> 20

```


## 2. Run the inference with a pretrained model
Now, we will run the script with a pretrained model which we will have already in the pipeline.
```
python python/test_recurrent_detector.py \
    --index indexes_detection/chb01/test.txt \
    --id chb01 \
    --model lstm \
    --dir experiments/detection_recurrent_chb01_lstm_adam_0.0001
```
