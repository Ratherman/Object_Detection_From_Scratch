import torch
import torch.nn as nn

architecture_config = [
    # ================================================ #
    # Spec: (kernel_size, num_filter, stride, padding) #
    # ================================================ #

    (7, 64, 2, 3),  # -------------------------------------------------------   # Conv. Layer: 7x7x64-s-2
    "M", # ------------------------------------------------------------------   # Maxp. Layer: 2x2-s-2
    (3, 192, 1, 1), # -------------------------------------------------------   # Conv. Layer: 3x3x192
    "M", # ------------------------------------------------------------------   # Maxp. Layer: 2x2-s-2
    (1, 128, 1, 0), (3, 256, 1, 1), (1, 256, 1, 0), (3, 512, 1, 1), # -------   # Conv. Layer: ...
    "M", # ------------------------------------------------------------------   # Maxp. Layer: 2x2-s-2
    [(1, 256, 1, 0), (3, 512, 1, 1), 4], (1, 512, 1, 0), (3, 1024, 1, 1), # -   # Conv. Layer: ...
    "M", # ------------------------------------------------------------------   # Maxp. Layer: 2x2-s-2
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2], (3, 1024, 1, 1), (3, 1024, 2, 1),     # Conv. Layer: ...
    (3, 1024, 1, 1), (3, 1024, 1, 1) # --------------------------------------   # Conv. Layer: ...

    # =================================================== #
    # Note: This architecture doesn't include Conn. Layer #
    # =================================================== #
]

class CNNBlock(nn.Module):
    
    # **kwargs will make assigned parameters become dictionary
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        # bias=False due to using BatchNorm, BatchNorm will center the data
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))

class YOLOv1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(YOLOv1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)
    
    def forward(self, x):
        x = self.darknet(x)
        # start_dim=1 is because the 0-th dim means "num_of_examples"
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers.append(CNNBlock(in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3]))
                in_channels = x[1]
            elif type(x) == str: # i.e., "M"
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif type(x) == list:
                conv1 = x[0] # The 1st tuple in the list
                conv2 = x[1] # the 2nd tuple in the list
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers.append(CNNBlock(in_channels, conv1[1], kernel_size=conv1[0], stride=conv1[2], padding=conv1[3]))
                    layers.append(CNNBlock(conv1[1], conv2[1], kernel_size=conv2[0], stride=conv2[2], padding=conv2[3]))
                    in_channels = conv2[1]

            # *layers: this only works for single-input layers
        return nn.Sequential(*layers)

    def _create_fcs(self, split_size, num_boxes, num_classes):
        pass