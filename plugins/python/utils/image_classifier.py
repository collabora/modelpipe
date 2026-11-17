# ImageClassifier
# Copyright (C) 2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import dataclasses

# torchvision imagenet weights expect inputs normalised with these
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def center_cropped_image(image, size, device):
    import numpy as np
    import torch
    from torch.nn import functional

    height, width = image.shape[:2]
    scale = size / min(height, width)
    resized_height, resized_width = round(height * scale), round(width * scale)
    resized = functional.interpolate(
        torch.from_numpy(np.ascontiguousarray(image))
        .to(device)
        .permute(2, 0, 1)[None]
        .float()
        / 255,
        size=(resized_height, resized_width),
        mode="bilinear",
        antialias=True,
    )
    top, left = (resized_height - size) // 2, (resized_width - size) // 2
    return resized[0, :, top : top + size, left : left + size]


def normalised_images(images):
    import torch

    mean = torch.tensor(IMAGENET_MEAN, device=images.device)[:, None, None]
    std = torch.tensor(IMAGENET_STD, device=images.device)[:, None, None]
    return (images - mean) / std


@dataclasses.dataclass(frozen=True)
class ImageClassifier:
    network: object
    classes: list
    image_size: int
    device: str

    def probabilities(self, rgb_images):
        import torch

        with torch.inference_mode():
            images = torch.stack(
                [
                    center_cropped_image(image, self.image_size, self.device)
                    for image in rgb_images
                ]
            )
            return torch.softmax(self.network(normalised_images(images)), dim=1)


def save_image_classifier(path, network, architecture, classes, image_size):
    import torch

    torch.save(
        {
            "architecture": architecture,
            "classes": classes,
            "image_size": image_size,
            "state_dict": network.state_dict(),
        },
        path,
    )


def load_image_classifier(path, device):
    import torch
    import torchvision

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    network = torchvision.models.get_model(
        checkpoint["architecture"], num_classes=len(checkpoint["classes"])
    )
    network.load_state_dict(checkpoint["state_dict"])
    return ImageClassifier(
        network.to(device).eval(),
        checkpoint["classes"],
        checkpoint["image_size"],
        device,
    )
