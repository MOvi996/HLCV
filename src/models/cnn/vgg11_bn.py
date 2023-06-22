import torchvision
import torch
import torch.nn as nn

from ..base_model import BaseModel


class VGG11_bn(BaseModel):
    def __init__(
        self,
        layer_config,
        num_classes,
        activation,
        norm_layer,
        fine_tune,
        pretrained=True,
    ):
        super(VGG11_bn, self).__init__()

        # TODO: Initialize the different model parameters from the config file  #
        # for activation and norm_layer refer to cnn/model.py
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self._pretrained = pretrained
        self._build_model(num_classes, activation, norm_layer, fine_tune)

    def _build_model(self, num_classes, activation, norm_layer, fine_tune):
        #################################################################################
        # TODO: Build the classification network described in Q4 using the              #
        # models.vgg11_bn network from torchvision model zoo as the feature extraction  #
        # layers and two linear layers on top for classification. You can load the      #
        # pretrained ImageNet weights based on the weights variable. Set it to None if  #
        # you want to train from scratch, 'DEFAULT'/'IMAGENET1K_V1' if you want to use  #
        # pretrained weights. You can either write it here manually or in config file   #
        # You can enable and disable training the feature extraction layers based on    #
        # the fine_tune flag.                                                           #
        #################################################################################
        self.vgg11_bn = torchvision.models.vgg11_bn(pretrained=self._pretrained)

        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self.vgg11_bn.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            getattr(nn, norm_layer["type"])(4096),
            getattr(nn, activation["type"])(**activation["args"]),
            nn.Linear(4096, num_classes),
        )

        for param in self.vgg11_bn.features.parameters():
            if not fine_tune:
                param.requires_grad = False
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def forward(self, x):
        #################################################################################
        # TODO: Implement the forward pass computations                                 #
        #################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        x = self.vgg11_bn(x)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return x
