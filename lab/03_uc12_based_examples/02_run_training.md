# Run training

### REMEMBER to activate the conda environment before running the scripts

    conda activate winter-school

## 1. The train script
For running inference on test data we are going to use the script **skin_lesion_segmentation.py**, this script is prepared to do the inference with a pretrained model
and also for training models. To do the inference we will need an ONNX file of the pretrained model and a dataset with a YAML file that defines the ECVL Dataset. This is
the dataset that we downloaded in the [UC12 pipeline preparaion guide](00_pipeline_preparation.md).

The script has some flags that are important in order to execute inference, to see them:

    # Show the available flags of the test script
    python skin_lesion_segmentation.py --help

That command will show the following output:

    UC12 Skin lesion Segmentation pipeline. Prepares a pipeline to train and/or test models for segmentation

    positional arguments:
      INPUT_DATASET

    optional arguments:
      -h, --help            show this help message and exit
      --ckpts CKPTS         Load an existing ONNX
      --model {Unet,SegNet}
                            Model to use for training from scratch
      --epochs INT
      --batch-size INT
      --learning-rate LEARNING_RATE
      --size INT            Size of input slices
      --gpu GPU [GPU ...]   `--gpu 1 1` to use two GPUs
      --out-dir DIR         if set, save images in this directory
      --weights DIR         save weights in this directory
      --train-val
      --no-train-val
      --test
      --no-test
      --datagen-workers DATAGEN_WORKERS
                            Number of worker threads to use for loading the batches
      --queue-ratio-size QUEUE_RATIO_SIZE
                            The producers-consumer queue of the data generator will have a maximum size equal to batch_size x queue_ratio_size x datagen_workers

For doing inference we will uses the flag *--no-train-val* to avoid the training phase and the *--test* for doing the inference with the test split of the dataset.

## 2. Run training
To run training we can do it running a command like this:

        # Inside uc12_pipeline folder
        python skin_lesion_segmentation.py \
            data/isic_segmentation/isic_segmentation.yml \
            --epochs 30 \
            --gpu 1 \
            --datagen-workers 4 \
            --queue-ratio-size 4 \
            --out-dir out_predictions

## 3. Training results
If we provide the *--out-dir* flag, the masks predicted during validation will be stored in the folder path provided.
