# FinishLineModel
# Copyright (C) 2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import dataclasses

import numpy as np
import torch
import torchvision

from utils.image_classifier import normalised_images

HEAD_CHANNELS = 256
LINE_ENDS = 2
HEATMAP_SIGMA = 2
PEAK_MASS_RADIUS = 2 * HEATMAP_SIGMA


class ResnetFeatures(torch.nn.Sequential):
    def __init__(self, architecture, weights):
        resnet = torchvision.models.get_model(architecture, weights=weights)
        super().__init__(*list(resnet.children())[:-2])
        self.channels = resnet.fc.in_features
        self.size_multiple = 32
        self.upsampling_layers = 3


class Dinov2Features(torch.nn.Module):
    def __init__(self, architecture, weights):
        super().__init__()
        from transformers import Dinov2Config, Dinov2Model

        if weights is None:
            self.model = Dinov2Model(Dinov2Config.from_pretrained(architecture))
        else:
            self.model = Dinov2Model.from_pretrained(architecture)
        self.channels = self.model.config.hidden_size
        self.size_multiple = self.model.config.patch_size
        self.upsampling_layers = 2

    def forward(self, images):
        rows, columns = (
            images.shape[2] // self.size_multiple,
            images.shape[3] // self.size_multiple,
        )
        # dinov2 puts its class token before the patch tokens
        patches = self.model(pixel_values=images).last_hidden_state[:, 1:]
        return patches.transpose(1, 2).reshape(
            len(images), self.channels, rows, columns
        )


def backbone_features(architecture, weights):
    if "dinov2" in architecture:
        return Dinov2Features(architecture, weights)
    return ResnetFeatures(architecture, weights)


class FinishLineNetwork(torch.nn.Module):
    def __init__(self, architecture, weights):
        super().__init__()
        self.backbone = backbone_features(architecture, weights)
        channels = self.backbone.channels
        layers = []
        for _ in range(self.backbone.upsampling_layers):
            layers += [
                torch.nn.ConvTranspose2d(
                    channels, HEAD_CHANNELS, 4, stride=2, padding=1, bias=False
                ),
                torch.nn.BatchNorm2d(HEAD_CHANNELS),
                torch.nn.ReLU(inplace=True),
            ]
            channels = HEAD_CHANNELS
        self.upsampling = torch.nn.Sequential(*layers)
        self.end_heatmaps = torch.nn.Conv2d(HEAD_CHANNELS, LINE_ENDS, 1)
        self.visible = torch.nn.Linear(self.backbone.channels, 1)

    def forward(self, images):
        features = self.backbone(images)
        heatmaps = self.end_heatmaps(self.upsampling(features))
        return heatmaps, self.visible(features.mean((2, 3)))[:, 0]


def resized_image(rgb, input_width, input_height, device):
    image = torch.from_numpy(np.ascontiguousarray(rgb)).to(device).permute(2, 0, 1)
    return torch.nn.functional.interpolate(
        image[None].float() / 255,
        size=(input_height, input_width),
        mode="bilinear",
        antialias=True,
    )[0]


def vertex_offset(before, centre, after, inside):
    # the log of a gaussian is a parabola, so its vertex is the peak even where the border cuts the gaussian
    curvature = before - 2 * centre + after
    return torch.where(
        inside & (curvature < 0), (before - after) / (2 * curvature), 0.0
    )


def heatmap_peaks(heatmaps):
    count, ends, height, width = heatmaps.shape
    log_probabilities = torch.log_softmax(heatmaps.flatten(2).float(), dim=2)
    top = log_probabilities.argmax(dim=2)
    maps = log_probabilities.view(count, ends, height, width)
    images = torch.arange(count, device=heatmaps.device)[:, None]
    channels = torch.arange(ends, device=heatmaps.device)[None, :]
    row, column = top // width, top % width

    def value(rows, columns):
        return maps[
            images, channels, rows.clamp(0, height - 1), columns.clamp(0, width - 1)
        ]

    centre = value(row, column)
    x = column + vertex_offset(
        value(row, column - 1),
        centre,
        value(row, column + 1),
        (column > 0) & (column < width - 1),
    )
    y = row + vertex_offset(
        value(row - 1, column),
        centre,
        value(row + 1, column),
        (row > 0) & (row < height - 1),
    )
    return torch.stack([x, y], dim=2)


def peak_masses(heatmaps):
    count, ends, height, width = heatmaps.shape
    probabilities = torch.softmax(heatmaps.flatten(2).float(), dim=2).view(
        count, ends, height, width
    )
    side = 2 * PEAK_MASS_RADIUS + 1
    window_sums = (
        torch.nn.functional.avg_pool2d(
            probabilities, side, stride=1, padding=PEAK_MASS_RADIUS
        )
        * side**2
    )
    return window_sums.flatten(2).max(dim=2).values


@dataclasses.dataclass(frozen=True)
class FinishLinePrediction:
    confidence: float
    # far end first, in pixels of the image passed to predict
    ends: np.ndarray


@dataclasses.dataclass(frozen=True)
class FinishLineModel:
    network: FinishLineNetwork
    input_width: int
    input_height: int
    device: str

    def predict(self, rgb):
        height, width = rgb.shape[:2]
        with torch.inference_mode():
            image = resized_image(rgb, self.input_width, self.input_height, self.device)
            heatmaps, visible = self.network(normalised_images(image[None]))
            ends = heatmap_peaks(heatmaps)[0].cpu().numpy()
            confidence = torch.sigmoid(visible[0]) * peak_masses(heatmaps)[0].min()
        heatmap_height, heatmap_width = heatmaps.shape[2:]
        scale = np.array([width / heatmap_width, height / heatmap_height])
        return FinishLinePrediction(float(confidence), ends * scale)


def save_finish_line_model(path, network, architecture, input_width, input_height):
    torch.save(
        {
            "architecture": architecture,
            "input_width": input_width,
            "input_height": input_height,
            "state_dict": network.state_dict(),
        },
        path,
    )


def load_finish_line_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    network = FinishLineNetwork(checkpoint["architecture"], weights=None)
    network.load_state_dict(checkpoint["state_dict"])
    return FinishLineModel(
        network.to(device).eval(),
        checkpoint["input_width"],
        checkpoint["input_height"],
        device,
    )
