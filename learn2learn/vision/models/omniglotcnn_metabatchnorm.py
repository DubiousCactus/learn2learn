#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.


"""
OmniglotCNN using a MetaBatchNorm layer allowing to accumulate per-step running statistics and use
per-step bias and variance parameters.
"""

import learn2learn as l2l
import torch

from learn2learn.nn.metabatchnorm import MetaBatchNorm
from learn2learn.vision.models.cnn4 import maml_init_, fc_init_


class LinearBlock_MetaBatchNorm(torch.nn.Module):
    def __init__(self, input_size, output_size, adaptation_steps):
        super(LinearBlock_MetaBatchNorm, self).__init__()
        self.relu = torch.nn.ReLU()
        self.normalize = MetaBatchNorm(
            output_size,
            adaptation_steps,
            affine=True,
            momentum=0.999,
            eps=1e-3,
        )
        self.linear = torch.nn.Linear(input_size, output_size)
        fc_init_(self.linear)

    def forward(self, x, inference=False):
        x = self.linear(x)
        x = self.normalize(x, inference=inference)
        x = self.relu(x)
        return x


class ConvBlock_MetaBatchNorm(torch.nn.Module):
    def __init__(
        self,
        adaptation_steps,
        in_channels,
        out_channels,
        kernel_size,
        max_pool=True,
        max_pool_factor=1.0,
    ):
        super(ConvBlock_MetaBatchNorm, self).__init__()
        stride = (int(2 * max_pool_factor), int(2 * max_pool_factor))
        if max_pool:
            self.max_pool = torch.nn.MaxPool2d(
                kernel_size=stride,
                stride=stride,
                ceil_mode=False,
            )
            stride = (1, 1)
        else:
            self.max_pool = lambda x: x
        self.normalize = MetaBatchNorm(
            out_channels,
            adaptation_steps,
            affine=True,
            # eps=1e-3,
            # momentum=0.999,
        )
        torch.nn.init.uniform_(self.normalize.weight)
        self.relu = torch.nn.ReLU()

        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=1,
            bias=True,
        )
        maml_init_(self.conv)

    def forward(self, x, inference=False):
        x = self.conv(x)
        x = self.normalize(x, inference=inference)
        x = self.relu(x)
        x = self.max_pool(x)
        return x


class ConvBase_MetaBatchNorm(torch.nn.Sequential):

    # NOTE:
    #     Omniglot: hidden=64, channels=1, no max_pool
    #     MiniImagenet: hidden=32, channels=3, max_pool

    def __init__(
        self,
        adaptation_steps,
        hidden=64,
        channels=1,
        max_pool=False,
        layers=4,
        max_pool_factor=1.0,
    ):
        core = [
            ConvBlock_MetaBatchNorm(
                adaptation_steps,
                channels,
                hidden,
                (3, 3),
                max_pool=max_pool,
                max_pool_factor=max_pool_factor,
            ),
        ]
        for _ in range(layers - 1):
            core.append(
                ConvBlock_MetaBatchNorm(
                    adaptation_steps,
                    hidden,
                    hidden,
                    kernel_size=(3, 3),
                    max_pool=max_pool,
                    max_pool_factor=max_pool_factor,
                )
            )
        super(ConvBase_MetaBatchNorm, self).__init__(*core)

    def forward(self, x, inference=False):
        for module in self:
            x = module(x, inference=inference)
        return x


class OmniglotFC_MetaBatchNorm(torch.nn.Module):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/models/cnn4.py)

    **Description**

    The fully-connected network used for Omniglot experiments, as described in Santoro et al, 2016.

    **References**

    1. Santoro et al. 2016. “Meta-Learning with Memory-Augmented Neural Networks.” ICML.

    **Arguments**

    * **input_size** (int) - The dimensionality of the input.
    * **output_size** (int) - The dimensionality of the output.
    * **sizes** (list, *optional*, default=None) - A list of hidden layer sizes.

    **Example**
    ~~~python
    net = OmniglotFC(input_size=28**2,
                     output_size=10,
                     sizes=[64, 64, 64])
    ~~~
    """

    def __init__(self, input_size, output_size, adaptation_steps, sizes=None):
        super(OmniglotFC_MetaBatchNorm, self).__init__()
        if sizes is None:
            sizes = [256, 128, 64, 64]
        layers = [
            LinearBlock_MetaBatchNorm(input_size, sizes[0], adaptation_steps),
        ]
        for s_i, s_o in zip(sizes[:-1], sizes[1:]):
            layers.append(LinearBlock_MetaBatchNorm(s_i, s_o, adaptation_steps))
        layers = torch.nn.Sequential(*layers)
        self.features = torch.nn.Sequential(
            l2l.nn.Flatten(),
            layers,
        )
        self.classifier = fc_init_(torch.nn.Linear(sizes[-1], output_size))
        self.input_size = input_size

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class OmniglotCNN_MetaBatchNorm(torch.nn.Module):
    """

    [Source](https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/models/cnn4.py)

    **Description**

    The convolutional network commonly used for Omniglot, as described by Finn et al, 2017.

    This network assumes inputs of shapes (1, 28, 28).

    **References**

    1. Finn et al. 2017. “Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks.” ICML.

    **Arguments**

    * **output_size** (int) - The dimensionality of the network's output.
    * **hidden_size** (int, *optional*, default=64) - The dimensionality of the hidden representation.
    * **layers** (int, *optional*, default=4) - The number of convolutional layers.

    **Example**
    ~~~python
    model = OmniglotCNN(output_size=20, hidden_size=128, layers=3)
    ~~~

    """

    def __init__(self, adaptation_steps, output_size=5, hidden_size=64, layers=4):
        super(OmniglotCNN_MetaBatchNorm, self).__init__()
        self.hidden_size = hidden_size
        self.base = ConvBase_MetaBatchNorm(
            adaptation_steps,
            hidden=hidden_size,
            channels=1,
            max_pool=False,
            layers=layers,
        )
        self.features = torch.nn.Sequential(
            l2l.nn.Lambda(lambda x: x.view(-1, 1, 28, 28)),
            self.base,
            l2l.nn.Lambda(lambda x: x.mean(dim=[2, 3])),
            l2l.nn.Flatten(),
        )
        self.classifier = torch.nn.Linear(hidden_size, output_size, bias=True)
        self.classifier.weight.data.normal_()
        self.classifier.bias.data.mul_(0.0)

    def backup_stats(self):
        """
        Backup stored batch statistics before running a validation epoch.
        """
        for layer in self.features.modules():
            if type(layer) is MetaBatchNorm:
                layer.backup_stats()

    def restore_backup_stats(self):
        """
        Reset stored batch statistics from the stored backup.
        """
        for layer in self.features.modules():
            if type(layer) is MetaBatchNorm:
                layer.restore_backup_stats()


    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
