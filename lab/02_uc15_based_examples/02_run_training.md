# Run training

### REMEMBER to activate the conda environment before running the scripts

    conda activate winter-school

## 1. The train script
For running training we are going to use the **train.py** script of the **pyeddl_pipeline**. This script has flags
for configuring the training process, things like: selecting the model architecture, the optimizer, learning rate,
epochs, data augmentation, computing device, etc.

To see all the available flags we can execute the help command:

    # Go to the pyeddl version of the pipeline
    cd uc15_lab/UC15_pipeline/pyeddl_pipeline
    # Show the available flags of the test script
    python train.py --help

That command will show the following output:

     Script for training the classification models

    optional arguments:
      -h, --help            show this help message and exit
      --yaml-path YAML_PATH
                            Path to the YAML file to create the ECVL dataset
                            (default: ../../../datasets/BIMCV-COVID19-cIter_1_2/co
                            vid19_posi/ecvl_bimcv_covid19.yaml)
      --target-size HEIGHT WIDTH
                            Target size to resize the images, given by height and
                            width (default: [256, 256])
      --batch-size BATCH_SIZE, -bs BATCH_SIZE
                            Size of the training batches of data (default: 3)
      --epochs EPOCHS       Number of epochs to train (default: 10)
      --frozen-epochs FROZEN_EPOCHS
                            In case of using a pretrained model, this param sets
                            the number of epochs with the pretrained weights
                            frozen (default: 5)
      --augmentations {0.0,1.0,1.1,2.0}, -augs {0.0,1.0,1.1,2.0}
                            Version of data augmentation to use (default: 0.0)
      --rgb                 Load the images in RGB format instead of grayscale. If
                            the image is grayscale the single channel is
                            replicated two times (default: False)
      --model {model_1,model_2,model_3,model_4,ResNet18,ResNet34,ResNet50,ResNet101,ResNet152,Pretrained_ResNet18,Pretrained_ResNet34,Pretrained_ResNet50,Pretrained_ResNet101,Pretrained_ResNet152,VGG16,VGG16BN,VGG19,VGG19BN,Pretrained_VGG16,Pretrained_VGG19,Pretrained_VGG16BN,Pretrained_VGG19BN}
                            Model architecture to train (default: model_1)
      --multiclass          Prepares the pipeline for multiclass classification,
                            using sigmoid in the output layer and MSE loss
                            function (default: False)
      --binary-loss         Changes the pipeline to use binary cross entropy as
                            loss function enabling multiclass classification. The
                            loss is computed for each unit (class) of the output
                            layer (default: False)
      --normal-vs-classification, -normal-vs
                            Prepares the pipeline for binary classification where
                            one of the two classes provided in the dataset must be
                            'normal'. The model will use a single output neuron to
                            perform the binary classification using BCE loss and
                            binary accuracy metric (default: False)
      --regularization {l1,l2,l1l2}, -reg {l1,l2,l1l2}
                            Adds the selected regularization type to all the
                            layers of the model (default: None)
      --regularization-factor REGULARIZATION_FACTOR, -reg-f REGULARIZATION_FACTOR
                            Regularization factor to use (in case of using
                            --regularization) (default: 0.01)
      --optimizer {Adam,SGD}, -opt {Adam,SGD}
                            Training optimizer (default: Adam)
      --learning-rate LEARNING_RATE, -lr LEARNING_RATE
                            Learning rate of the optimizer (default: 0.001)
      --lr-decay LR_DECAY   Value to regulate the learning rate decay (default:
                            0.0)
      --cpu                 Sets CPU as the computing device (default: False)
      --gpus binary_flag [binary_flag ...]
                            Sets the number of GPUs to use. Usage "--gpus 1 1"
                            (two GPUs) (default: [1])
      --mem-level {full_mem,mid_mem,low_mem}, -mem {full_mem,mid_mem,low_mem}
                            Memory level for the computing device (default:
                            full_mem)
      --experiments-path EXPERIMENTS_PATH, -exp EXPERIMENTS_PATH
                            Path to the folder to store the results and
                            configuration of each experiment (default:
                            experiments)
      --seed SEED           Seed value for random operations (default: 1234)
      --model-ckpt MODEL_CKPT
                            An ONNX model checkpoint to start training from
                            (default: None)
      --pretrained-onnx PRETRAINED_ONNX
                            An ONNX file to use as a pretrained model to extract
                            the layers of interest (usually the conv block) and
                            then add a new densely connected block to classify.
                            IMPORTANT: Provide the corresponding model
                            architecture with the --model flag. (default: None)
      --datagen-workers DATAGEN_WORKERS
                            Number of worker threads to use for loading the
                            batches (default: 1)
      --queue-ratio-size QUEUE_RATIO_SIZE
                            The producers-consumer queue of the data generator
                            will have a maximum size equal to batch_size x
                            queue_ratio_size x datagen_workers (default: 1)

## 2. Run training
To execute the training script the main points to take in to account in order to avoid errors are:

- Provide the YAML file corresponding to the ECVL dataset configuration to use
- Select the input shape of the images (**--target-size** flag). You should change this depending on the dataset provided
- Use the *--rgb* flag if you are going to use one of the pretrained models (they need an input with 3 channels)

As an example, you could run an experiment with the following command:

        # Inside uc15_lab/UC15_pipeline/pyeddl_pipeline
        python train.py \
            --yaml-path ../../data/256x256/ecvl_256x256_normal-vs-covid_only-r0.yaml \
            --target-size 256 256 \
            --rgb \
            --augmentations 1.1 \
            --model Pretrained_ResNet50 \
            --epochs 50 \
            --frozen-epochs 20 \
            --optimizer Adam \
            --learning-rate 0.00001 \
            --regularization l2 \
            --regularization-factor 0.0002 \
            --datagen-workers 4

If you want to execute an experiment for the 4-class multilabel task, you should add at least the *--multiclass* flag and change the yaml file:

        # Inside uc15_lab/UC15_pipeline/pyeddl_pipeline
        python train.py \
            --yaml-path ../../data/512x512/ecvl_512x512_normal-vs-covid-vs-infiltrates-vs-pneumonia_only-r0.yaml \
            --target-size 512 512 \
            --rgb \
            --augmentations 1.1 \
            --model Pretrained_ResNet50 \
            --epochs 50 \
            --frozen-epochs 20 \
            --optimizer Adam \
            --learning-rate 0.00001 \
            --regularization l2 \
            --regularization-factor 0.0002 \
            --multiclass \
            --binary-loss \
            --datagen-workers 4
            
Note: We added the *--multiclass* flag to enable the multilabel classification, and we also used the *--binary-loss* flag to use the BCE instead of the MSE.

## 3. Training results
At the end of the training process the script will run the test with the best models by validation loss and accuracy. Those results will be
displayed and also saved in an experiment folder that will be created in a folder called *experiments* (by default). In that folder you will find
the results of the test and also: the training results for every epoch (CSV file), the experiment configuration (json file), the plots of
the training metrics and the ONNX checkpoint files.
