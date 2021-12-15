# Run inference on test data with a pretrained model

### REMEMBER to activate the conda enviroment before running the scripts

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
      --binary-loss         Prepares the pipeline for models with as many output layers as classes (with one output neuron) (default: False)
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

## 2. Get a pretrained model in ONNX format
In order to run the script we will need a pretrained model, you can take one of this:
- For the 256x256 dataset, with an accuracy on test of 85.71% (normal vs COVID 19): [download uc15_256x256_ResNet101.onnx](https://upvedues-my.sharepoint.com/:u:/g/personal/salcarpo_upv_edu_es/EfX00LcDeINIuMvPnt1_aUABQlFO3zPletzohRtq5O9E3g?e=AbODWc)
- For the 512x512 dataset, with an accuracy on test of 88.57% (normal vs COVID 19): [download uc15_512x512_ResNet101.onnx](https://upvedues-my.sharepoint.com/:u:/g/personal/salcarpo_upv_edu_es/EaLfcNGvMlFElO9Ml0-GI2UBxLxG5nOLVRBPgZe7F8S9rA?e=2i1oD0)

## 3. Run inference
Once we have the data and the model we can run inference with this command:
- For the 256x256 dataset and ONNX model:

        # Inside uc15_lab/UC15_pipeline/pyeddl_pipeline
        python test.py \
            --yaml-path ../../data/256x256/ecvl_256x256_normal-vs-covid_only-r0.yaml \
            --onnx-files ~/Downloads/uc15_256x256_ResNet101.onnx \
            --target-size 256 256 \
            --rgb
        
- For the 512x512 dataset and ONNX model:

        # Inside uc15_lab/UC15_pipeline/pyeddl_pipeline
        python test.py \
            --yaml-path ../../data/512x512/ecvl_512x512_normal-vs-covid_only-r0.yaml \
            --onnx-files ~/Downloads/uc15_512x512_ResNet101.onnx \
            --target-size 512 512 \
            --rgb

***Remember*** to combine well the yaml file path, the onnx model and the target size depending on the version of the dataset used (256x256 or 512x512).

Note: The --rgb flag is used when the input of the model has 3 channels, even if the data is in grayscale, because in order to use some pretrained models
we had to convert the grayscale images to "RGB" by replicating the same channel to be able to fit the images in the model.
