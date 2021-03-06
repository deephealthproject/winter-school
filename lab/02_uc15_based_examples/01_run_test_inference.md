# Run inference on test data with a pretrained model

### REMEMBER to activate the conda environment before running the scripts

    conda activate winter-school

## 1. The test script
For running inference on test data we are going to use a script (**test.py**) that takes a pretrained model (in ONNX format) and the YAML file
of an ECVL dataset to perform inference using the provided model and the test split of the dataset provided.

The script has some flags that are important in order to execute inference, to see them:

    # Go to the pyeddl version of the pipeline
    cd uc15_lab/UC15_pipeline/pyeddl_pipeline
    # Show the available flags of the test script
    python test.py --help

That command will show the following output:

    Script to perform test inference with one or more models

    optional arguments:
      -h, --help            show this help message and exit
      --yaml-path YAML_PATH
                            Path to the YAML file to create the ECVL dataset (default: ../../../datasets/BIMCV-COVID19-cIter_1_2/covid19_posi/ecvl_bimcv_covid19.yaml)
      --multiclass          Prepares the testing pipeline for multiclass classification (default: False)
      --binary-loss         Changes the pipeline to use binary cross entropy as loss function enabling multiclass classification. The loss and metrics are computed for each unit(class) of the
                            output layer (default: False)
      --normal-vs-classification, -normal-vs
                            Prepares the pipeline for binary classification where one of the two classes provided in the dataset must be 'normal'. The model will use a single output neuron to
                            perform the binary classification using BCE loss and binary accuracy metric (default: False)
      --onnx-files onnx_path [onnx_path ...]
                            A list of paths to the ONNX files to use for testing (default: [])
      --out-path OUT_PATH   Path of the folder to store the test results. If not provided, the tests are only shown in the standard output (default: )
      --target-size HEIGHT WIDTH
                            Target size to resize the images, given by height and width (default: [256, 256])
      --batch-size BATCH_SIZE, -bs BATCH_SIZE
                            Size of the training batches of data (default: 3)
      --rgb                 Load the images in RGB format instead of grayscale. If the image is grayscale the single channel is replicated two times (default: False)
      --cpu                 Sets CPU as the computing device (default: False)
      --gpus binary_flag [binary_flag ...]
                            Sets the number of GPUs to use. Usage "--gpus 1 1" (two GPUs) (default: [1])
      --mem-level {full_mem,mid_mem,low_mem}, -mem {full_mem,mid_mem,low_mem}
                            Memory level for the computing device (default: full_mem)
      --seed SEED           Seed value for random operations (default: 1234)
      --datagen-workers DATAGEN_WORKERS
                            Number of worker threads to use for loading the batches (default: 1)
      --queue-ratio-size QUEUE_RATIO_SIZE
                            The producers-consumer queue of the data generator will have a maximum size equal to batch_size x queue_ratio_size x datagen_workers (default: 1)

## 2. Get a pretrained model in ONNX format
In order to run the script we will need a pretrained model, you can take one of this:
- For the 256x256 dataset, with an accuracy on test of 88.57% (normal vs COVID 19, with BCE loss):
  [download uc15_256x256_normal-vs-covid_BCE-loss.onnx](https://upvedues-my.sharepoint.com/:u:/g/personal/salcarpo_upv_edu_es/ERjdPkf8_89Oh0wADBdC-jwB6mHbgzoztiwGdtefnlAsJw?e=3N1SpM)
  or [from a Dropbox folder](https://www.dropbox.com/sh/vqdewy1ocqpkiu4/AADklYzBxelzFXT2TApJNyFia/uc15_256x256_normal-vs-covid_BCE-loss.onnx?dl=0)
  or [from a local UPV server](https://clocalprog.dsic.upv.es/winter-school/data/uc15_256x256_normal-vs-covid_BCE-loss.onnx),
  _do not worry about the certificate, you can trust it, it was generated locally by the coordinator of this winter school (Jon Ander G&oacute;mez)_.
  
      # Download from the UPV server with wget
      wget https://clocalprog.dsic.upv.es/winter-school/data/uc15_256x256_normal-vs-covid_BCE-loss.onnx --no-check-certificate
  
- For the 512x512 dataset, with an accuracy on test of 88.57% (normal vs COVID 19, with CE loss): [download uc15_512x512_normal-vs-covid_CE-loss.onnx](https://upvedues-my.sharepoint.com/:u:/g/personal/salcarpo_upv_edu_es/EaLfcNGvMlFElO9Ml0-GI2UBxLxG5nOLVRBPgZe7F8S9rA?e=zMhyDj)
  or [from a Dropbox folder](https://www.dropbox.com/sh/vqdewy1ocqpkiu4/AACcx-9w5YB9X744BkzXqsKLa/uc15_512x512_normal-vs-covid_CE-loss.onnx?dl=0)
  or [from a local UPV server](https://clocalprog.dsic.upv.es/winter-school/data/uc15_512x512_normal-vs-covid_CE-loss.onnx).
 
      # Download from the UPV server with wget
      wget https://clocalprog.dsic.upv.es/winter-school/data/uc15_512x512_normal-vs-covid_CE-loss.onnx --no-check-certificate
  
- For the 512x512 dataset and multilabel classification (normal vs COVID 19 vs infiltrates vs pneumonia): [download uc15_512x512_normal-vs-covid-vs-infiltrates-vs-pneumonia.onnx](https://upvedues-my.sharepoint.com/:u:/g/personal/salcarpo_upv_edu_es/EWaFqI3auQlGuTIqhM-9lSEBkjq9_h0XFplSfakXBDX7fw?e=WDeAhb)
  or [from a Dropbox folder](https://www.dropbox.com/sh/vqdewy1ocqpkiu4/AACMbtb2pFLT6nXUSX01_mYNa/uc15_512x512_normal-vs-covid-vs-infiltrates-vs-pneumonia.onnx?dl=0)
  or [from a local UPV server](https://clocalprog.dsic.upv.es/winter-school/data/uc15_512x512_normal-vs-covid-vs-infiltrates-vs-pneumonia.onnx).
  
      # Download from the UPV server with wget
      wget https://clocalprog.dsic.upv.es/winter-school/data/uc15_512x512_normal-vs-covid-vs-infiltrates-vs-pneumonia.onnx --no-check-certificate

## 3. Run inference
Once we have the data and the model we can run inference with this command:
- For the 256x256 dataset and ONNX model (with BCE loss):

        # Inside uc15_lab/UC15_pipeline/pyeddl_pipeline
        python test.py \
            --yaml-path ../../data/256x256/ecvl_256x256_normal-vs-covid_only-r0.yaml \
            --onnx-files uc15_256x256_normal-vs-covid_BCE-loss.onnx \
            --target-size 256 256 \
            --normal-vs-classification \
            --rgb
        
- For the 512x512 dataset and ONNX model:

        # Inside uc15_lab/UC15_pipeline/pyeddl_pipeline
        python test.py \
            --yaml-path ../../data/512x512/ecvl_512x512_normal-vs-covid_only-r0.yaml \
            --onnx-files uc15_512x512_normal-vs-covid_CE-loss.onnx \
            --target-size 512 512 \
            --rgb

- For the 512x512 dataset for multilabel classification and ONNX model:

        # Inside uc15_lab/UC15_pipeline/pyeddl_pipeline
        python test.py \
            --yaml-path ../../data/512x512/ecvl_512x512_normal-vs-covid-vs-infiltrates-vs-pneumonia_only-r0.yaml \
            --onnx-files uc15_512x512_normal-vs-covid-vs-infiltrates-vs-pneumonia.onnx \
            --target-size 512 512 \
            --multiclass \
            --binary-loss \
            --rgb

***Remember*** to combine well the yaml file path, the onnx model and the target size depending on the version of the dataset used (256x256 or 512x512).

Note: The --rgb flag is used when the input of the model has 3 channels, even if the data is in grayscale, because in order to use some pretrained models
we had to convert the grayscale images to "RGB" by replicating the same channel to be able to fit the images in the model.
