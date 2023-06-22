import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

from ..base_model import BaseModel


class ConvNet(BaseModel):
    def __init__(
        self,
        input_size,
        hidden_layers,
        num_classes,
        activation,
        norm_layer,
        drop_prob=0.0,
    ):
        super(ConvNet, self).__init__()

        ######################################################################################################
        # TODO: Initialize the different model parameters from the config file                               #
        # You can use the arguments given in the constructor. For activation and norm_layer                  #
        # to make it easier, you can use the following two lines                                             #
        #   self._activation = getattr(nn, activation["type"])(**activation["args"])                         #
        #   self._norm_layer = getattr(nn, norm_layer["type"])                                               #
        # Or you can just hard-code using nn.Batchnorm2d and nn.ReLU as they remain fixed for this exercise. #
        ######################################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self._build_model(
            input_size, hidden_layers, num_classes, norm_layer, drop_prob, activation
        )

    def _build_model(
        self, input_size, hidden_layers, num_classes, norm_layer, drop_prob, activation
    ):
        #################################################################################
        # TODO: Initialize the modules required to implement the convolutional layer    #
        # described in the exercise.                                                    #
        # For Q1.a make use of conv2d and relu layers from the torch.nn module.         #
        # For Q2.a make use of BatchNorm2d layer from the torch.nn module.              #
        # For Q3.b Use Dropout layer from the torch.nn module.                          #
        #################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(
                    input_size,
                    hidden_layers[0],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="zeros",
                ),
                getattr(nn, norm_layer["type"])(hidden_layers[0]),
                nn.MaxPool2d(kernel_size=2, stride=2),
                getattr(nn, activation["type"])(**activation["args"]),
                nn.Dropout(drop_prob),
                nn.Conv2d(
                    hidden_layers[0],
                    hidden_layers[1],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                getattr(nn, norm_layer["type"])(hidden_layers[1]),
                nn.MaxPool2d(kernel_size=2, stride=2),
                getattr(nn, activation["type"])(**activation["args"]),
                nn.Dropout(drop_prob),
                nn.Conv2d(
                    hidden_layers[1],
                    hidden_layers[2],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                getattr(nn, norm_layer["type"])(hidden_layers[2]),
                nn.MaxPool2d(kernel_size=2, stride=2),
                getattr(nn, activation["type"])(**activation["args"]),
                nn.Dropout(drop_prob),
                nn.Conv2d(
                    hidden_layers[2],
                    hidden_layers[3],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                getattr(nn, norm_layer["type"])(hidden_layers[3]),
                nn.MaxPool2d(kernel_size=2, stride=2),
                getattr(nn, activation["type"])(**activation["args"]),
                nn.Dropout(drop_prob),
                nn.Conv2d(
                    hidden_layers[3],
                    hidden_layers[4],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                getattr(nn, norm_layer["type"])(hidden_layers[4]),
                nn.MaxPool2d(kernel_size=2, stride=2),
                getattr(nn, activation["type"])(**activation["args"]),
                nn.Dropout(drop_prob),
                nn.Flatten(),
                nn.Linear(hidden_layers[5], num_classes),
            ]
        )

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def _normalize(self, img):
        """
        Helper method to be used for VisualizeFilter.
        This is not given to be used for Forward pass! The normalization of Input for forward pass
        must be done in the transform presets.
        """
        max = np.max(img)
        min = np.min(img)
        return (img - min) / (max - min)

    def VisualizeFilter(self):
        ################################################################################
        # TODO: Implement the functiont to visualize the weights in the first conv layer#
        # in the model. Visualize them as a single image fo stacked filters.            #
        # You can use matlplotlib.imshow to visualize an image in python                #
        #################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        filters = self.layers[0].weight.data.cpu().numpy()
        fig, ax = plt.subplots(nrows=8, ncols=16, figsize=(10, 10))
        for i in range(8):
            for j in range(16):
                ax[i, j].imshow(self._normalize(filters[i * 8 + j]), cmap="cividis")
                ax[i, j].axis("off")

        fig.set_facecolor("black")
        fig.tight_layout()
        plt.show()
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def forward(self, x):
        #################################################################################
        # TODO: Implement the forward pass computations                                 #
        #################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        for i, l in enumerate(self.layers):
            x = self.layers[i](x)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return x
