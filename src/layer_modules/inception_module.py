import torch.nn as nn
import torch

class InceptionModule(nn.Module):
    """
    InceptionModule is a module that implements the Inception module architecture.
    
    Args:
        in_channels (int): Number of input channels.
        n_filters (int): Number of filters for each convolutional layer.
        kernel_sizes (list): List of kernel sizes for the convolutional layers.
        bottleneck_channels (int): Number of channels in the bottleneck layer.
    """
    def __init__(self, in_channels, n_filters, kernel_sizes, bottleneck_channels=2, nb_filters=32, use_residual=True):
        super(InceptionModule, self).__init__()

        self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, 
                                    kernel_size=1, bias=False)
        kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]
        self.convs = nn.ModuleList()
        for kernel_size in kernel_size_s:
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
    

class CustomInceptionModule(nn.Module):
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