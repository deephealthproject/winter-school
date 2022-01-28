# Copyright (c) 2021 CRS4 Università degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the 'Software'), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

'''
UC12 Skin lesion Segmentation pipeline.

Prepares a pipeline to train and/or test models for segmentation
'''

import argparse
import os

import numpy as np
import pyecvl.ecvl as ecvl
import pyeddl.eddl as eddl

from lib import utils
from lib.models import Unet, SegNet, SegNetBN


def inference(phase, dataset, epoch, args, num_batches, net, best_miou=0):
    current_path = os.path.join(args.out_dir, phase, f'Epoch_{epoch + 1}')
    evaluator = utils.Evaluator()
    out = eddl.getOut(net)[0]
    dataset.Start()
    out_string = f'- epoch [{epoch + 1}/{args.epochs}] ' if phase == "Validation" else ''

    for b in range(num_batches):
        print(f'{phase} {out_string}- batch [{b + 1}/{num_batches}]')
        samples, x, y = dataset.GetBatch()
        eddl.forward(net, [x])
        output = eddl.getOutput(out)
        current_bs = x.shape[0]
        for bs in range(current_bs):
            img = output.select([str(bs)])
            gt = y.select([str(bs)])
            img_np = np.array(img, copy=False)
            gt_np = np.array(gt, copy=False)
            iou = evaluator.BinaryIoU(img_np, gt_np, thresh=0.5)
            print(f' - IoU: {iou:.3f}', end='', flush=True)
            if args.out_dir:
                os.makedirs(current_path, exist_ok=True)
                # Original image
                orig_img = x.select([str(bs)])
                orig_img.mult_(255.)
                orig_img.normalize_(0., 255.)
                orig_img_t = ecvl.TensorToView(orig_img)
                orig_img_t.colortype_ = ecvl.ColorType.RGB
                orig_img_t.channels_ = 'xyc'
                # Draw the Predicted mask
                img_t = ecvl.TensorToView(img)
                img_t.colortype_ = ecvl.ColorType.GRAY
                img_t.channels_ = 'xyc'
                ecvl.Threshold(img_t, img_t, 0.5, 255)
                tmp, labels = ecvl.Image.empty(), ecvl.Image.empty()
                ecvl.ConvertTo(img_t, tmp, ecvl.DataType.uint8)
                ecvl.ConnectedComponentsLabeling(tmp, labels)
                ecvl.ConvertTo(labels, tmp, ecvl.DataType.uint8)
                contours = ecvl.FindContours(tmp)
                ecvl.ConvertTo(orig_img_t, tmp, ecvl.DataType.uint8)
                tmp_np = np.array(tmp, copy=False)
                for cseq in contours:
                    for c in cseq:
                        tmp_np[c[0], c[1], 0] = 255
                        tmp_np[c[0], c[1], 1] = 0
                        tmp_np[c[0], c[1], 2] = 0
                filename = samples[bs].location_[0]
                head, tail = os.path.splitext(os.path.basename(filename))
                bname = '%s.png' % head
                output_fn = os.path.join(current_path, bname)
                ecvl.ImWrite(output_fn, tmp)
                # Ground truth mask
                gt_t = ecvl.TensorToView(gt)
                gt_t.colortype_ = ecvl.ColorType.GRAY
                gt_t.channels_ = 'xyc'
                gt.mult_(255.)
                gt_filename = samples[bs].label_path_
                gt_fn = os.path.join(current_path,
                                     os.path.basename(gt_filename))
                ecvl.ImWrite(gt_fn, gt_t)
        print()

    dataset.Stop()  # Stop validation split generator

    last_miou = evaluator.MIoU()
    print(f'{phase} {out_string}- Total MIoU: {last_miou:.3f}')

    if phase == "Validation" and last_miou > best_miou:
        best_miou = last_miou
        filepath = os.path.join(args.weights, f'isic_segm_{args.model}_epoch_{epoch + 1}.onnx')
        eddl.save_net_to_onnx_file(net, filepath)
        print('Weights saved')

    return best_miou


def main(args):
    image_size = args.size, args.size

    if args.weights:
        os.makedirs(args.weights, exist_ok=True)

    training_augs = ecvl.SequentialAugmentationContainer([
        ecvl.AugResizeDim(image_size, ecvl.InterpolationType.cubic,
                          gt_interp=ecvl.InterpolationType.nearest),
        ecvl.AugMirror(.5),
        ecvl.AugFlip(.5),
        ecvl.AugRotate([-180, 180]),
        ecvl.AugAdditivePoissonNoise([0, 10]),
        ecvl.AugGammaContrast([0.5, 1.5]),
        ecvl.AugGaussianBlur([0, 0.8]),
        ecvl.AugCoarseDropout([0, 0.03], [0.02, 0.05], 0.25),
        ecvl.AugToFloat32(255, divisor_gt=255),
        ecvl.AugNormalize([0.6681, 0.5301, 0.5247],
                          [0.1337, 0.1480, 0.1595]),  # isic stats

    ])
    validation_test_augs = ecvl.SequentialAugmentationContainer([
        ecvl.AugResizeDim(image_size, ecvl.InterpolationType.cubic,
                          gt_interp=ecvl.InterpolationType.nearest),
        ecvl.AugToFloat32(255, divisor_gt=255),
        ecvl.AugNormalize([0.6681, 0.5301, 0.5247], [
                          0.1337, 0.1480, 0.1595]),  # isic stats

    ])
    dataset_augs = ecvl.DatasetAugmentations([training_augs, validation_test_augs, validation_test_augs])

    print('Reading dataset')
    d = ecvl.DLDataset(args.in_ds, args.batch_size,
                       dataset_augs, ctype=ecvl.ColorType.RGB,
                       num_workers=args.datagen_workers,
                       queue_ratio_size=args.queue_ratio_size)
    num_classes = d.n_channels_gt_
    size = d.n_channels_, args.size, args.size

    if args.ckpts:
        net = eddl.import_net_from_onnx_file(args.ckpts, size)
    else:
        in_ = eddl.Input(size)
        if args.model == 'SegNet':
            out = SegNet(in_, num_classes)
        elif args.model == 'SegNetBN':
            out = SegNetBN(in_, num_classes)
        elif args.model == 'Unet':
            out = Unet(in_, num_classes)
        else:
            raise Exception(f'Invalid model name provided ({args.model})')
        out_sigm = eddl.Sigmoid(out)
        net = eddl.Model([in_], [out_sigm])

    loss_name = 'binary_cross_entropy'
    metric_name = 'mean_squared_error'
    eddl.build(
        net,
        eddl.adam(args.learning_rate),
        [loss_name],
        [metric_name],
        eddl.CS_GPU(args.gpu, mem='low_mem') if args.gpu else eddl.CS_CPU(),
        False if args.ckpts else True  # Initialize weights only with new model
    )

    eddl.summary(net)
    os.makedirs('logs', exist_ok=True)
    eddl.setlogfile(net, 'logs/skin_lesion_segmentation')

    best_miou = 0.
    if args.train:
        num_batches_train = d.GetNumBatches("training")
        num_batches_val = d.GetNumBatches("validation")
        evaluator = utils.Evaluator()

        print('Starting training')
        for e in range(args.epochs):
            d.SetSplit(ecvl.SplitType.training)
            d.ResetBatch(ecvl.SplitType.training, shuffle=True)
            d.Start()
            eddl.reset_loss(net)
            for b in range(num_batches_train):
                _, x, y = d.GetBatch()
                eddl.train_batch(net, [x], [y])
                losses = eddl.get_losses(net)
                metrics = eddl.get_metrics(net)

                print(f'Train - epoch [{e + 1}/{args.epochs}] - batch [{b + 1}/{num_batches_train}]'
                      f' - {loss_name}={losses[0]:.3f} - {metric_name}={metrics[0]:.3f}', flush=True)
            d.Stop()

            d.SetSplit(ecvl.SplitType.validation)
            d.ResetBatch(ecvl.SplitType.validation, shuffle=False)
            evaluator.ResetEval()
            eddl.set_mode(net, 0)

            best_miou = inference("Validation", dataset=d, epoch=e, args=args, num_batches=num_batches_val, net=net,
                             best_miou=best_miou)

    if args.test:
        evaluator = utils.Evaluator()
        evaluator.ResetEval()

        d.SetSplit(ecvl.SplitType.test)
        num_batches_test = d.GetNumBatches("test")
        eddl.set_mode(net, 0)
        inference("Test", dataset=d, epoch=args.epochs, args=args, num_batches=num_batches_test, net=net)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('in_ds', metavar='INPUT_DATASET')
    parser.add_argument('--ckpts', type=str, help='Load an existing ONNX')
    parser.add_argument('--model', type=str, default='Unet',
                        choices=['Unet', 'SegNet', 'SegNetBN'],
                        help='Model to use for training from scratch')
    parser.add_argument('--epochs', type=int, metavar='INT', default=30)
    parser.add_argument('--batch-size', type=int, metavar='INT', default=24)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--size', type=int, metavar='INT', default=224,
                        help='Size of input slices')
    parser.add_argument('--gpu', nargs='+', type=int, required=False,
                        help='`--gpu 1 1` to use two GPUs')
    parser.add_argument('--out-dir', metavar='DIR', default="",
                        help='if set, save images in this directory')
    parser.add_argument('--weights', metavar='DIR', type=str, default='weights',
                        help='save weights in this directory')

    # Set the pipeline mode: training, testing or both
    parser.add_argument('--train-val', dest='train', action='store_true')
    parser.add_argument('--no-train-val', dest='train', action='store_false')
    parser.add_argument('--test', dest='test', action='store_true')
    parser.add_argument('--no-test', dest='test', action='store_false')
    parser.set_defaults(train=True)
    parser.set_defaults(test=False)

    # Data generator parallelization
    parser.add_argument('--datagen-workers', default=1, type=int,
                        help='Number of worker threads to use for loading the batches')
    parser.add_argument('--queue-ratio-size', default=1, type=int,
                        help=('The producers-consumer queue of the data '
                              'generator will have a maximum size equal to '
                              'batch_size x queue_ratio_size x datagen_workers'))
    main(parser.parse_args())
