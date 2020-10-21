import sys
import time
from typing import Dict
import numpy as np
from torch import nn
from torch import Tensor, nn, optim
from torchvision.models import resnet18


class GRUDecoder(nn.Module):
    def __init__(self, device, batch=32, in_dim=512, out_dim=100, hidden_size=2048):
        super().__init__()
        self.batch = batch
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.h0 = torch.zeros(2, batch, hidden_size).to(device)
            
        self.decoder1 = nn.GRU(in_dim, hidden_size, batch_first=True, bidirectional=True).to(device)
        self.linear1 = nn.Linear(2 * hidden_size, out_dim + 1).to(device)
        
        self.decoder2 = nn.GRU(out_dim + 1, hidden_size, batch_first=True, bidirectional=True).to(device)
        self.linear2 = nn.Linear(2 * hidden_size, out_dim + 1).to(device)
        
        self.decoder3 = nn.GRU(out_dim + 1, hidden_size, batch_first=True, bidirectional=True).to(device)
        self.linear2 = nn.Linear(2* hidden_size, out_dim + 1).to(device)
        
        self.softmax = nn.Softmax(dim=1).to(device)
            
    def forward(self, x):
        x1, h = self.decoder1(x.view(self.batch, 1, self.in_dim), self.h0)
        x1 = self.linear1(x1)
        coord1, conf1 = torch.split(x1.view(self.batch, self.out_dim + 1), self.out_dim, dim=1)
        
        x2, h = self.decoder2(x1, h)
        x2 = self.linear2(x2)
        coord2, conf2 = torch.split(x2.view(self.batch, self.out_dim + 1), self.out_dim, dim=1)
        
        x3, h = self.decoder3(x2, h)
        x3 = self.linear1(x3)
        coord3, conf3 = torch.split(x3.view(self.batch, self.out_dim + 1), self.out_dim, dim=1)
        
        conf = self.softmax(torch.cat([conf1, conf2, conf3], dim=1))
        output = torch.cat([coord1, coord2, coord3], dim=1).view(self.batch, 3, 50, 2)
        return conf, output


class Resnet18GRU(nn.Module):
    """Multi Mode Baseline
    """

    def __init__(self, cfg: Dict, device, num_modes=3):
        """Init Mode Instance

        Args:
            cfg (Dict): Configuration Dict
            num_modes (int, optional): number of trajectories. Defaults to 3.
            device: needed to move GRU to cuda
        """
        super().__init__()
        # TODO: support other than resnet18?
        backbone = resnet18(pretrained=True, progress=True)
        self.backbone = backbone
        num_history_channels = (
            cfg["model_params"]["history_num_frames"] + 1) * 2
        num_in_channels = 3 + num_history_channels
        self.backbone.conv1 = nn.Conv2d(
            num_in_channels,
            self.backbone.conv1.out_channels,
            kernel_size=self.backbone.conv1.kernel_size,
            stride=self.backbone.conv1.stride,
            padding=self.backbone.conv1.padding,
            bias=False,
        )
        # This is 512 for resnet18 and resnet34;
        # And it is 2048 for the other resnets
        backbone_out_features = 512
        # X, Y coords for the future positions (output shape: Bx50x2)
        self.future_len = cfg["model_params"]["future_num_frames"]
        num_targets = 2 * self.future_len
        self.num_preds = num_targets * num_modes
        self.num_modes = num_modes
        self.gru_decoder = GRUDecoder(
            device,
            batch=self.batch_size, 
            in_dim=backbone_out_features
            )

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        confidences, pred = self.gru_decoder(x)
        assert confidences.shape == (self.batch_size, self.num_modes)
        return pred, confidences
