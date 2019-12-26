from PIL import Image
import numpy as np
import cv2
import random
import torch
from torchvision.transforms import functional
from torchvision import transforms

class Transforms(object):
    def __init__(self, scale, crop, stride, gamma):
        self.scale = scale
        self.crop = crop
        self.stride = stride
        self.gamma = gamma

    def __call__(self, image, density):
        # random resize
        height, width = image.size[1], image.size[0]

        scale = random.uniform(self.scale[0], self.scale[1])
        height = round(height * scale)
        width = round(width * scale)
        image = image.resize((width, height), Image.BILINEAR)
        density = cv2.resize(density, (width, height), interpolation=cv2.INTER_LINEAR) / scale / scale

        # for case the image is too samll to crop
        short = min(height, width)

        if short < 512:
            scale = 512 / short
            height = round(height * scale)
            width = round(width * scale)
            image = image.resize((width, height), Image.BILINEAR)
            density = cv2.resize(density, (width, height), interpolation=cv2.INTER_LINEAR) / scale / scale

        # random crop
        h, w = self.crop[0], self.crop[1]
        dh = random.randint(0, height - h)
        dw = random.randint(0, width - w)
        image = image.crop((dw, dh, dw + w, dh + h))
        density = density[dh:dh + h, dw:dw + w]

        # random flip
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            density = density[:, ::-1]

        # random gamma
        if random.random() < 0.3:
            gamma = random.uniform(self.gamma[0], self.gamma[1])
            image = functional.adjust_gamma(image, gamma)

        image = functional.to_tensor(image)
        image = functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        density = cv2.resize(density, (density.shape[1] // self.stride, density.shape[0] // self.stride),
                             interpolation=cv2.INTER_LINEAR) * self.stride * self.stride

        density = np.reshape(density, [1, density.shape[0], density.shape[1]])

        return image, torch.from_numpy(density.copy()).float()