import os

import numpy as np
import gdown

class Evaluator:
    def __init__(self):
        self.eps = 1e-06
        self.buf = []

    def ResetEval(self):
        self.buf = []

    def BinaryIoU(self, a, b, thresh=0.5):
        intersection = np.logical_and(a >= thresh, b >= thresh).sum()
        union = np.logical_or(a >= thresh, b >= thresh).sum()
        rval = (intersection + self.eps) / (union + self.eps)
        self.buf.append(rval)
        return rval

    def DiceCoefficient(self, a, b, thresh=0.5):
        a = Threshold(a, thresh)
        b = Threshold(b, thresh)
        intersection = np.logical_and(a, b).sum()
        rval = (2 * intersection + self.eps) / (a.sum() + b.sum() + self.eps)

        self.buf.append(rval)
        return rval

    def MIoU(self):
        if not self.buf:
            return 0
        return sum(self.buf) / len(self.buf)

    MeanMetric = MIoU


def Threshold(a, thresh=0.5):
    a[a >= thresh] = 1
    a[a < thresh] = 0
    return a


def ImageSqueeze(img):
    k = img.dims_.index(1)
    img.dims_ = [_ for i, _ in enumerate(img.dims_) if i != k]
    img.strides_ = [_ for i, _ in enumerate(img.strides_) if i != k]
    k = img.channels_.find("z")
    img.channels_ = "".join([_ for i, _ in enumerate(img.channels_) if i != k])


def DownloadModel(url: str, filename: str, dest_path: str) -> str:
    """
    Download a onnx from a URL and place it in a destination folder.
    @param url: The URL to follow for download the ONNX file
    @param filename: The filename of the downloaded file ('my_model.onxx')
    @param dest_path: The folder where to place the downloaded file
    @return: Return the full path to the ONNX model
    """
    if not os.path.exists(os.path.join(dest_path, filename)):
        os.makedirs(dest_path, exist_ok=True)
        gdown.download(url, os.path.join(dest_path, filename), quiet=False)
    return os.path.join(dest_path, filename)
