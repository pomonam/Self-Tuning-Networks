import numpy as np

import torch


class Cutout(object):
    # Contains a code from https://github.com/uoguelph-mlrg/Cutout
    """ Randomly mask out one or more patches from an image. """
    def __init__(self, n_holes, length):
        """ Initialize a class CutOut.
        :param n_holes: int
        :param length: int
        """
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        if self.length <= 0 or self.n_holes <= 0:
            return img

        h = img.size(1)
        w = img.size(2)
        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = int(np.clip(y - self.length / 2, 0, h))
            y2 = int(np.clip(y + self.length / 2, 0, h))
            x1 = int(np.clip(x - self.length / 2, 0, w))
            x2 = int(np.clip(x + self.length / 2, 0, w))

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
