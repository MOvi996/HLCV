import torch.nn as nn
import numpy as np
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    Base class for all models
    """

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """

        ret_str = super().__str__()

        #################################################################################
        # TODO: Q1.b) Print the number of trainable parameters for each layer and total number of trainable parameters
        # Simply update the ret_str by adding new lines to it.
        #################################################################################
        total_p = 0
        for name, params in self.named_parameters():
            layer_p = 1
            for si in list(params.size()):
                layer_p = layer_p * si
            ret_str += f"\n{name} : {layer_p}"
            total_p += layer_p

        ret_str += f"\nModel params: {total_p}\n"
        return ret_str
