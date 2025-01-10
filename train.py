import json
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset

from ultralytics.nn.tasks import SegmentationModel


class YOLOSegmentation(nn.Module):
    """Single-class implementation of YOLOv8 Segmentation model"""

    def __init__(self, num_classes=80, proto_channels=32):
        super().__init__()
        self.num_classes = num_classes
        self.proto_channels = proto_channels

        # Create all layers
        self._create_backbone()
        self._create_neck()
        self._create_head()

    def _conv_block(self, in_channels, out_channels, kernel_size=3, stride=1):
        """Creates a convolution block with BatchNorm and SiLU activation"""
        padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03),
            nn.SiLU(inplace=True)
        )

    def _c2f_block(self, in_channels, out_channels, num_bottlenecks=1):
        """Creates a C2f feature extraction block"""
        layers = []
        # Initial 1x1 conv
        layers.append(self._conv_block(in_channels, out_channels, kernel_size=1))

        # Bottleneck layers
        self.bottlenecks = nn.ModuleList([
            nn.Sequential(
                self._conv_block(out_channels // 2, out_channels // 2),
                self._conv_block(out_channels // 2, out_channels // 2)
            ) for _ in range(num_bottlenecks)
        ])

        # Final 1x1 conv
        layers.append(self._conv_block(out_channels * 2, out_channels, kernel_size=1))

        return nn.ModuleList(layers)

    def _create_backbone(self):
        """Creates the backbone layers"""
        self.backbone = nn.ModuleDict({
            'conv1': self._conv_block(3, 16, stride=2),
            'conv2': self._conv_block(16, 32, stride=2),
            'c2f1': self._c2f_block(32, 32, 1),
            'conv3': self._conv_block(32, 64, stride=2),
            'c2f2': self._c2f_block(64, 64, 2),
            'conv4': self._conv_block(64, 128, stride=2),
            'c2f3': self._c2f_block(128, 128, 2),
            'conv5': self._conv_block(128, 256, stride=2),
            'c2f4': self._c2f_block(256, 256, 1),
            'sppf': self._create_sppf(256, 256)
        })

    def _create_neck(self):
        """Creates the neck layers"""
        self.neck = nn.ModuleDict({
            'upsample': nn.Upsample(scale_factor=2, mode='nearest'),
            'c2f5': self._c2f_block(384, 128, 1),  # 256 + 128 concat
            'c2f6': self._c2f_block(192, 64, 1),  # 128 + 64 concat
            'conv6': self._conv_block(64, 64, stride=2),
            'c2f7': self._c2f_block(192, 128, 1),  # 128 + 64 concat
            'conv7': self._conv_block(128, 128, stride=2),
            'c2f8': self._c2f_block(384, 256, 1)  # 256 + 128 concat
        })

    def _create_sppf(self, in_channels, out_channels):
        """Creates Spatial Pyramid Pooling - Fast layer"""
        mid_channels = in_channels // 2
        return nn.ModuleDict({
            'conv1': self._conv_block(in_channels, mid_channels, kernel_size=1),
            'maxpool': nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
            'conv2': self._conv_block(mid_channels * 4, out_channels, kernel_size=1)
        })

    def _create_proto_layer(self, in_channels):
        """Creates prototype mask generation layer"""
        return nn.ModuleDict({
            'conv1': self._conv_block(in_channels, in_channels),
            'upsample': nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
            'conv2': self._conv_block(in_channels, in_channels),
            'conv3': self._conv_block(in_channels, self.proto_channels, kernel_size=1)
        })

    def _create_head(self):
        """Creates the segmentation head"""
        channels_list = [64, 128, 256]

        self.proto = self._create_proto_layer(channels_list[0])

        # Detection and mask branches for each scale
        self.detect_branches = nn.ModuleList([
            nn.Sequential(
                self._conv_block(c, c),
                self._conv_block(c, c),
                nn.Conv2d(c, self.num_classes + 4, kernel_size=1)
            ) for c in channels_list
        ])

        self.mask_branches = nn.ModuleList([
            nn.Sequential(
                self._conv_block(c, c),
                self._conv_block(c, c),
                nn.Conv2d(c, self.proto_channels, kernel_size=1)
            ) for c in channels_list
        ])

    def _forward_backbone(self, x):
        """Forward pass through backbone"""
        features = {}
        x = features['x1'] = self.backbone['conv1'](x)
        x = features['x2'] = self.backbone['conv2'](x)
        x = features['x3'] = self._forward_c2f(self.backbone['c2f1'], x)
        x = features['x4'] = self.backbone['conv3'](x)
        x = features['x5'] = self._forward_c2f(self.backbone['c2f2'], x)
        x = features['x6'] = self.backbone['conv4'](x)
        x = features['x7'] = self._forward_c2f(self.backbone['c2f3'], x)
        x = features['x8'] = self.backbone['conv5'](x)
        x = features['x9'] = self._forward_c2f(self.backbone['c2f4'], x)
        x = features['x10'] = self._forward_sppf(x)
        return features

    def _forward_c2f(self, c2f_block, x):
        """Forward pass through C2f block"""
        x = c2f_block[0](x)  # First conv
        y = torch.chunk(x, 2, 1)
        bottleneck_outputs = [y[1]]
        z = y[0]
        for bottleneck in self.bottlenecks:
            z = bottleneck(z)
            bottleneck_outputs.append(z)
        return c2f_block[1](torch.cat([y[0]] + bottleneck_outputs, 1))  # Final conv

    def _forward_sppf(self, x):
        """Forward pass through SPPF layer"""
        x = self.backbone['sppf']['conv1'](x)
        y1 = self.backbone['sppf']['maxpool'](x)
        y2 = self.backbone['sppf']['maxpool'](y1)
        y3 = self.backbone['sppf']['maxpool'](y2)
        return self.backbone['sppf']['conv2'](torch.cat([x, y1, y2, y3], 1))

    def _forward_proto(self, x):
        """Forward pass through prototype layer"""
        x = self.proto['conv1'](x)
        x = self.proto['upsample'](x)
        x = self.proto['conv2'](x)
        return self.proto['conv3'](x)

    def forward(self, x):
        """Forward pass through the entire model"""
        # Backbone
        features = self._forward_backbone(x)

        # Neck
        fpn1 = self.neck['upsample'](features['x10'])
        fpn1 = torch.cat([fpn1, features['x7']], 1)
        fpn1 = self._forward_c2f(self.neck['c2f5'], fpn1)

        fpn2 = self.neck['upsample'](fpn1)
        fpn2 = torch.cat([fpn2, features['x5']], 1)
        fpn2 = self._forward_c2f(self.neck['c2f6'], fpn2)

        pan1 = self.neck['conv6'](fpn2)
        pan1 = torch.cat([pan1, fpn1], 1)
        pan1 = self._forward_c2f(self.neck['c2f7'], pan1)

        pan2 = self.neck['conv7'](pan1)
        pan2 = torch.cat([pan2, features['x10']], 1)
        pan2 = self._forward_c2f(self.neck['c2f8'], pan2)

        # Head
        proto_masks = self._forward_proto(fpn2)

        outputs = []
        for feat, detect_branch, mask_branch in zip([fpn2, pan1, pan2],
                                                    self.detect_branches,
                                                    self.mask_branches):
            det = detect_branch(feat)
            mask_coeffs = mask_branch(feat)
            outputs.append((det, mask_coeffs))

        return outputs, proto_masks

class AnyLabelingDataset(Dataset):
    def __init__(self, path: Path, transform=None):
        self.path = path
        self.transform = transform
        self.labels = []

        for path in path.glob('*.json'):
            old_data = path.read_text()
            old_data = json.loads(old_data)

            new_data = {
                'polygons': [],
                'source_file': self.path / old_data['imagePath'],
            }

            for shape in old_data['shapes']:
                label = shape['label']
                points = [[int(x), int(y)] for x, y in shape['points']]

                new_data['polygons'].append({
                    'label': label,
                    'points': points
                })

            self.labels.append(new_data)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label_data = self.labels[idx]

        image_path = self.path / label_data['source_file']
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label_data['polygons']


# Print state dict
model = YOLOSegmentation()
