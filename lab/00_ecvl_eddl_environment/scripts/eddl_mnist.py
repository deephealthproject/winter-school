'''
Example of a classification task using MNIST dataset
'''
import argparse

from pyeddl import eddl
from pyeddl.tensor import Tensor
from pyeddl.eddl import Input, ReLu, Conv, BatchNormalization, Reshape
from pyeddl.eddl import MaxPool, Dense, Flatten, Softmax, GlobalAveragePool


def get_optimizer(opt_name: str, learning_rate: float):
    if opt_name == 'Adam':
        return eddl.adam(lr=learning_rate)
    elif opt_name == 'SGD':
        return eddl.sgd(lr=learning_rate)
    elif opt_name == 'RMSProp':
        return eddl.rmsprop(lr=learning_rate)
    else:
        raise Exception(f'Invalid optimizer provided ("{opt_name}")')


def get_compserv(is_cpu: bool):
    return eddl.CS_CPU() if is_cpu else eddl.CS_GPU()


def main(args):
    # Download dataset
    eddl.download_mnist()
    # Load the dataset
    x_train = Tensor.load('mnist_trX.bin')
    y_train = Tensor.load('mnist_trY.bin')
    x_test = Tensor.load('mnist_tsX.bin')
    y_test = Tensor.load('mnist_tsY.bin')

    # Preprocess the images. From [0-255] to [0-1]
    x_train.div_(255.0)
    x_test.div_(255.0)

    # Show data shape
    print('Dataset shape:')
    print(f'Train split images: {x_train.shape}')
    print(f'Train split labels: {y_train.shape}')
    print(f'Test split images: {x_test.shape}')
    print(f'Test split labels: {y_test.shape}')

    if not args.from_ckpt:
        # Define the model topology
        num_classes = 10
        in_ = Input([784])
        layer = Reshape(in_, [1, 28, 28])
        layer = ReLu(BatchNormalization(Conv(layer, 32, [3, 3]), affine=True))
        layer = MaxPool(layer, [2, 2])
        layer = ReLu(BatchNormalization(Conv(layer, 64, [3, 3]), affine=True))
        layer = MaxPool(layer, [2, 2])
        layer = ReLu(BatchNormalization(Conv(layer, 128, [3, 3]), affine=True))
        layer = MaxPool(layer, [2, 2])
        layer = ReLu(BatchNormalization(Conv(layer, 256, [3, 3]), affine=True))
        layer = GlobalAveragePool(layer)
        layer = Flatten(layer)
        out_ = Softmax(Dense(layer, num_classes))

        # Create the model
        model = eddl.Model([in_], [out_])
    else:
        model = eddl.import_net_from_onnx_file(args.from_ckpt)

    # Select the optimizer
    opt = get_optimizer(args.optimizer, args.learning_rate)

    # Select the Computing Service
    cs = get_compserv(args.cpu)

    # Build the model to prepare it for training or inference
    eddl.build(model,
               opt,                            # Optimizer
               ['categorical_cross_entropy'],  # Losses
               ['accuracy'],                   # Metrics
               cs,                             # Computing Service
               args.from_ckpt == '')           # Weights initialization

    # Show the model layers
    eddl.summary(model)

    '''
    Train Phase
    '''
    best_loss = float('inf')  # To track the best model
    for e in range(1, args.epochs+1):
        print('#############')
        print(f'Epoch {e}/{args.epochs}')
        print('#############')

        eddl.fit(model, [x_train], [y_train], args.batch_size, 1)  # Train
        eddl.evaluate(model, [x_test], [y_test], args.batch_size)  # Validation

        # Check if we got a new best model
        val_loss = eddl.get_losses(model)[0]  # We only have one output layer
        if val_loss < best_loss:
            print(f'* New best model! loss: {best_loss:.4f} -> {val_loss:.4f}')
            eddl.save_net_to_onnx_file(model, args.onnx_output_path)
            best_loss = val_loss

    '''
    Test Phase
    '''
    print('\n##################################################')
    print('Test phase with the best model from training')
    print('##################################################\n')
    # Load the best model from the training phase
    best_model = eddl.import_net_from_onnx_file(args.onnx_output_path)

    # Create a new optimizer and comserv object
    opt = get_optimizer(args.optimizer, args.learning_rate)
    cs = get_compserv(args.cpu)

    eddl.build(best_model,
               opt,                            # Optimizer
               ['categorical_cross_entropy'],  # Losses
               ['accuracy'],                   # Metrics
               cs,                             # Computing Service
               init_weights=False)             # Avoid resetting the weights

    eddl.evaluate(best_model, [x_test], [y_test], args.batch_size)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument(
        '--batch-size', '-bs',
        help='Size of the training batches of data',
        type=int,
        default=100)

    arg_parser.add_argument(
        '--epochs', '-e',
        help='Number of epochs to train',
        default=10,
        type=int)

    arg_parser.add_argument(
        '--optimizer', '-opt',
        help='Training optimizer',
        default='Adam',
        choices=['Adam', 'SGD', 'RMSProp'],
        type=str)

    arg_parser.add_argument(
        '--learning-rate', '-lr',
        help='Learning rate of the optimizer',
        default=0.001,
        type=float)

    arg_parser.add_argument(
        '--cpu',
        help='Sets CPU as the computing device',
        action='store_true')

    arg_parser.add_argument(
        '--onnx-output-path',
        help='Filepath to store the best model in ONNX',
        default='best_model.onnx',
        type=str)

    arg_parser.add_argument(
        '--from-ckpt',
        help='Path to an ONNX file to use as staring point',
        default='',
        type=str)

    main(arg_parser.parse_args())
