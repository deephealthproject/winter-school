# Run inference on test data with a pretrained model

### REMEMBER to activate the conda environment before running the scripts

    conda activate winter-school

## 1. The test script
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
      --model {Unet,SegNet,SegNetBN}
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

For doing inference we will uses the flad *--no-train-val* to avoid the training phase and the *--test* for doing the inference with the test split of the dataset.

## 2. Get a pretrained model in ONNX format
You can download a pretrained model based on the Unet architecture from
[here](https://drive.google.com/uc?id=16Xu_w1LJa1m2f7SIDxInmS5lv6PN_s7G&export=download)
or from a Dropbox folder _not ready yet_
or [from a local UPV server](https://clocalprog.dsic.upv.es/winter-school/data/UC12Segm_unet_224_bce.onnx),
_do not worry about the certificate, you can trust it, it was generated locally by the coordinator of this winter school (Jon Ander G&oacute;mez)_.

## 3. Run inference
Once we have the data and the model we can run inference with this command:

        # Inside uc12_pipeline folder
        python skin_lesion_segmentation.py \
            data/isic_segmentation/isic_segmentation.yml \
            --ckpts ~/Downloads/UC12Segm_unet_224_bce.onnx \
            --no-train-val \
            --test \
            --out-dir out_predictions \
            --gpu 1

The masks predicted during inference will be stored in the *out_predictions* folder path that we provided with the *--out-dir* flag.
