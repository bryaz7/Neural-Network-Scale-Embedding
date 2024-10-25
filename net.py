
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from config import Config

import matplotlib.pyplot as plt

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, infeat, outfeat, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(infeat, outfeat[0], kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(outfeat[0])
        self.conv2 = nn.Conv2d(outfeat[0], outfeat[1], kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(outfeat[1])
        self.conv3 = nn.Conv2d(outfeat[1], outfeat[2], kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(outfeat[2])
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module, Config):
    def __init__(self, freeze=True):
        super(ResNet, self).__init__()
        Config.__init__(self) # TODO: why need this redundancy here?

        self.freeze = freeze
        # NOTE: using name to load tf chkpts easier
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_res_block(  64, [ 64,  64,  256], 3, stride=1)
        self.layer2 = self._make_res_block( 256, [128, 128,  512], 4, stride=2)
        self.layer3 = self._make_res_block( 512, [256, 256, 1024], 6, stride=2)
        self.layer4 = self._make_res_block(1024, [512, 512, 2048], 3, stride=2)

        self.gap = nn.AdaptiveAvgPool2d((1, 1)) # Global Average Pooling
        self.gmp = nn.AdaptiveMaxPool2d((1, 1)) # Global Max Pooling

        if self.exp_mode == 'baseline':
            self.classifier = nn.Linear(2048, self.nr_classes)
        elif self.exp_mode == 'scale_add':
            self.fc_scale = nn.Linear(2048, 1024)
            self.classifier = nn.Linear(1024, self.nr_classes)
        elif self.exp_mode == 'scale_conv':
            self.conv_scale = nn.Conv2d(2048, 512, kernel_size=1, bias=False)
            self.classifier = nn.Linear(512, self.nr_classes)
        elif self.exp_mode == 'scale_concat':
            self.fc_scale = nn.Linear(2048 * len(self.scale_list), 1024)
            self.classifier = nn.Linear(1024, self.nr_classes)
        else:
            self.classifier = nn.Linear(1024, self.nr_classes)

            self.conv_u = nn.ModuleList()
            self.conv_v = nn.ModuleList()

            for level in self.down_sample_level_list:  
                in_ch = 64 * (2**level)      
                self.conv_u.append(
                    nn.Sequential(
                        nn.Conv2d(in_ch, 512, 1, stride=1, padding=0, bias=False),
                        nn.Conv2d( 512, 512, 3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(512, eps=1e-5, momentum=0.9),
                        nn.ReLU(inplace=True),
                    )
                )

                self.conv_v.append(
                    nn.Sequential(
                        nn.Conv2d(in_ch, 512, 1, stride=1, padding=0, bias=True),
                        nn.Conv2d( 512, 512, 3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(512, eps=1e-5, momentum=0.9),
                        nn.ReLU(inplace=True),
                    )
                )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)

    def _make_res_block(self, infeat, outfeat, nr_blocks, stride=1):
        downsample = None
        if stride != 1 or infeat != outfeat[-1]:
            downsample = nn.Sequential(
                nn.Conv2d(infeat, outfeat[-1], kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outfeat[-1]),
            )

        layers = []
        layers.append(Bottleneck(infeat, outfeat, stride, downsample))
        for _ in range(1, nr_blocks):
            layers.append(Bottleneck(outfeat[-1], outfeat))

        return nn.Sequential(*layers)

    def forward(self, imgs):
        def scale_to(x, size):
            return nn.functional.interpolate(x, size=tuple(size), 
                            mode='bilinear', align_corners=True)
        def extract_feat(imgs):
            with torch.no_grad():
                d1 = self.relu(self.bn1(self.conv1(imgs)))
                d2 = self.maxpool(d1)
                d2 = self.layer1(d2)
                d3 = self.layer2(d2)           
            if not self.freeze:
                d4 = self.layer3(d3)
                d5 = self.layer4(d4)
            else:
                with torch.no_grad():
                    d4 = self.layer3(d3)
                    d5 = self.layer4(d4)
            return [d1, d2, d3, d4, d5]

        # feature extractor only
        if self.exp_mode == 'baseline':
            with torch.no_grad():
                feat = extract_feat(imgs)[-1]
                feat = self.gap(feat) # NOTE: Global Average Pool
        elif self.exp_mode == 'scale_embedding':
            scale_unit = np.array(self.input_size) / max(self.scale_list)
            aligned_scale_size = scale_unit * self.align_to_scale
            aligned_downsample_size = [(aligned_scale_size / 2**i) for i in range(1,6)]
            aligned_downsample_size = [list(size.astype('int32')) for  
                                                        size in aligned_downsample_size]
            levels_feat_list = [[] for i in range(len(self.down_sample_level_list))] 
            for scale_idx in self.scale_list:
                scale_imgs = imgs
                scale_size = (scale_unit * scale_idx)
                scale_size = list(scale_size.astype('int32'))
                if scale_size != self.input_size:
                    scale_imgs = scale_to(scale_imgs, scale_size)
                scale_feat = extract_feat(scale_imgs)
                for idx, level in enumerate(self.down_sample_level_list):
                    aligned_size = aligned_downsample_size[level-1]
                    aligned_feat = scale_to(scale_feat[level-1], aligned_size)
                    levels_feat_list[idx].append(aligned_feat)

            level_scale_embedding_feat_list = []
            for idx in range(len(self.down_sample_level_list)):
                # scale embedding with equivariant scale convolution
                scale_feat = torch.stack(levels_feat_list[idx], dim=4)
                mag_scale, pos_scale = torch.max(scale_feat, 4, keepdim=False)
                ang_scale = (10.0 * (pos_scale + 1)).float()
                u = self.conv_u[idx](mag_scale * torch.cos(ang_scale))
                v = self.conv_v[idx](mag_scale * torch.sin(ang_scale))
                feat = torch.sqrt(u * u + v * v)
                feat = self.gmp(feat) # NOTE: Global Max Pool
                level_scale_embedding_feat_list.append(feat)
            feat = torch.cat(level_scale_embedding_feat_list, dim=1)
        else:
            scale_unit = np.array(self.input_size) / max(self.scale_list)

            scale_feat_list = [] 
            for scale_idx in self.scale_list:
                scale_imgs = imgs
                scale_size = (scale_unit * scale_idx).astype('int32')
                if list(scale_size) != self.input_size:
                    scale_imgs = scale_to(scale_imgs, scale_size)
                # only use features at the last downsampling level
                scale_feat = extract_feat(scale_imgs)[-1]
                scale_feat = self.gmp(scale_feat)
                scale_feat = torch.squeeze(scale_feat)
                scale_feat_list.append(scale_feat)
            scale_feat = torch.stack(scale_feat_list, dim=-1)
            
            if self.exp_mode == 'scale_add':
                # sum across scale, element wise
                feat = torch.sum(scale_feat, -1)
                feat = self.fc_scale(feat)
            elif self.exp_mode == 'scale_concat':
                # flatten all features
                feat = scale_feat.view(scale_feat.size(0), -1)
                feat = self.fc_scale(feat)
            elif self.exp_mode == 'scale_conv':
                feat = torch.unsqueeze(scale_feat,3)
                feat = self.conv_scale(feat)
                feat = self.gmp(feat)

        out = feat.view(feat.size(0), -1)
        out = self.classifier(out)
        return out
