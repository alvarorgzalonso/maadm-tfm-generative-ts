import torch
from torch import nn


class InceptionModule(nn.Module):
    """
    InceptionModule is a module that implements the Inception module architecture.
    
    Args:
        in_channels (int): Number of input channels.
        n_filters (int): Number of filters for each convolutional layer.
        kernel_sizes (list): List of kernel sizes for the convolutional layers.
        bottleneck_channels (int): Number of channels in the bottleneck layer.
    """
    def __init__(self, in_channels, n_filters, kernel_sizes, bottleneck_channels):
        super(InceptionModule, self).__init__()
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, 
                                    kernel_size=1, bias=False)
        
        self.convs = nn.ModuleList()
        for kernel_size in kernel_sizes:
            self.convs.append(nn.Conv1d(bottleneck_channels, n_filters, 
                                        kernel_size, padding='same', bias=False))
        
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.maxpool_conv = nn.Conv1d(bottleneck_channels, n_filters, 
                                      kernel_size=1, bias=False)
        
        self.batchnorm = nn.BatchNorm1d(n_filters * (len(kernel_sizes) + 1))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.bottleneck(x)
    
        conv_outputs = [conv(x) for conv in self.convs]
        maxpool_output = self.maxpool(x)
        maxpool_output = self.maxpool_conv(maxpool_output)
        
        outputs = torch.cat(conv_outputs + [maxpool_output], 1)
        outputs = self.batchnorm(outputs)
        return self.relu(outputs)
    

class InceptionTime(nn.Module):
    def __init__(self, n_classes, in_channels, kszs=[10, 20, 40]):
        super(InceptionTime, self).__init__()
        self.inception_block1 = InceptionModule(in_channels, n_filters=32, 
                                                kernel_sizes=kszs, 
                                                bottleneck_channels=32)
        # The output channels are the number of filters times the number of paths (3 convs + 1 maxpool)
        out_channels = 32 * (len(kszs) + 1)
        
        # Subsequent Inception modules have out_channels as their in_channels
        self.inception_block2 = InceptionModule(out_channels, n_filters=32, 
                                                kernel_sizes=kszs, 
                                                bottleneck_channels=32)
        self.inception_block3 = InceptionModule(out_channels, n_filters=32, 
                                                kernel_sizes=kszs, 
                                                bottleneck_channels=32)
        self.inception_block4 = InceptionModule(out_channels, n_filters=32, 
                                                kernel_sizes=kszs, 
                                                bottleneck_channels=32)
        self.inception_block5 = InceptionModule(out_channels, n_filters=32, 
                                                kernel_sizes=kszs, 
                                                bottleneck_channels=32)
        self.inception_block6 = InceptionModule(out_channels, n_filters=32, 
                                                kernel_sizes=kszs, 
                                                bottleneck_channels=32)
        
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 
                                       kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(out_channels, n_classes)

    def forward(self, x):
        res = x # Save residual
        # Apply blocks
        for i, inception_block in enumerate([self.inception_block1,
                                             self.inception_block2, 
                                             self.inception_block3, 
                                             self.inception_block4, 
                                             self.inception_block5,
                                             self.inception_block6]):
            # Residuals after block 3 nd block 6
            x = inception_block(x)
            if (i == 2):
                x = x + self.act(self.bn(self.residual_conv(res)))
                res = x
            if (i == 5):
                x = x + self.act(self.bn(res))
        
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)