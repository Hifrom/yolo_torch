import torch

class yolov4(torch.nn.Module):
    def __init__(self):
        super(yolov4, self).__init__()
        self.input_layer = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.output_layer = torch.nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.output_layer(x)
        return x