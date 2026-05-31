import torch.nn as nn
import torch.nn.functional as F
from src.config import *

def create_conv(in_channels, out_channels, kernel_size, stride, padding_mode='same', dim=1):
    if dim == 1:
        conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding_mode)
    elif dim == 2:
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding_mode)
    else:
        raise ValueError("dim must be 1 or 2")

    nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)

    return conv


class student(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.to(DEVICE)

        self.conv = nn.Sequential(
            create_conv(in_channels, 32, (3, 3), 1, padding_mode='same', dim=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            create_conv(32, 128, (3, 3), 1, padding_mode='same', dim=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x, return_feature=True):
        feature = self.conv(x)
        feature_flat = feature.view(feature.size(0), -1)
        fc = self.fc(feature_flat)
        if return_feature:
            return feature_flat, fc
        else:
            return fc

    def extract_features(self, x):
        return self.conv(x)


class teacher(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.to(DEVICE)

        self.conv = nn.Sequential(
            create_conv(in_channels, 32, (3, 3), 1, padding_mode='same', dim=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            create_conv(32, 64, (3, 3), 1, padding_mode='same', dim=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            create_conv(64, 128, (3, 3), 1, padding_mode='same', dim=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 10),
        )

    def forward(self, x, return_feature=True):
        feature = self.conv(x)
        feature_flat = feature.view(feature.size(0), -1)
        fc = self.fc(feature_flat)
        if return_feature:
            return feature_flat, fc
        else:
            return fc

    def extract_features(self, x):
        return self.conv(x)