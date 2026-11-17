# ImageClassifierTraining
# Copyright (C) 2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import cv2
import numpy as np
import torch
import torchvision
from torchvision.transforms import v2

from utils.image_classifier import center_cropped_image, normalised_images

BATCH_SIZE = 64
# 64 images at 512 px exceed the 11 GB of an rtx 2080 ti
PIXELS_PER_BATCH = BATCH_SIZE * 256 * 256
PREDICT_BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WORKERS = 8
DEVICE = "cuda:0"


def read_rgb(path):
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


class AugmentedImages(torch.utils.data.Dataset):
    def __init__(self, samples, image_size):
        self.samples = samples
        self.image_size = image_size
        # hue shifts turn one number cloth colour into another
        self.augment = v2.Compose(
            [
                v2.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.8, 1.25)),
                v2.RandomHorizontalFlip(),
                v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, class_index = self.samples[index]
        image = center_cropped_image(read_rgb(path), self.image_size, "cpu")
        return self.augment(image), class_index


def pretrained_network(architecture, class_count):
    pretrained = torchvision.models.get_model(
        architecture, weights="DEFAULT"
    ).state_dict()
    network = torchvision.models.get_model(architecture, num_classes=class_count)
    shapes = {name: tensor.shape for name, tensor in network.state_dict().items()}
    network.load_state_dict(
        {
            name: tensor
            for name, tensor in pretrained.items()
            if shapes[name] == tensor.shape
        },
        strict=False,
    )
    return network


def train_image_classifier(samples, class_count, architecture, image_size, epochs):
    loader = torch.utils.data.DataLoader(
        AugmentedImages(samples, image_size),
        batch_size=min(BATCH_SIZE, PIXELS_PER_BATCH // image_size**2),
        shuffle=True,
        num_workers=WORKERS,
        persistent_workers=True,
    )
    network = pretrained_network(architecture, class_count).to(DEVICE).train()
    optimizer = torch.optim.AdamW(network.parameters(), lr=LEARNING_RATE)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, LEARNING_RATE, total_steps=epochs * len(loader)
    )
    scaler = torch.amp.GradScaler()
    for epoch in range(epochs):
        total_loss = 0.0
        for images, classes in loader:
            images, classes = images.to(DEVICE), classes.to(DEVICE)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = torch.nn.functional.cross_entropy(
                    network(normalised_images(images)), classes
                )
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            schedule.step()
            total_loss += loss.item() * len(classes)
        print(f"epoch {epoch + 1} loss {total_loss / len(samples):.3f}", flush=True)
    return network.eval()


def path_probabilities(classifier, paths):
    return np.concatenate(
        [
            classifier.probabilities(
                [read_rgb(path) for path in paths[start : start + PREDICT_BATCH_SIZE]]
            )
            .cpu()
            .numpy()
            for start in range(0, len(paths), PREDICT_BATCH_SIZE)
        ]
    )
