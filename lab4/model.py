import torch
import torch.nn as nn
import torch.nn.functional as F

class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, bias=True):
        super(_BNReluConv, self).__init__()
        self.append(nn.BatchNorm2d(num_maps_in))
        self.append(nn.ReLU(True))
        self.append(nn.Conv2d(num_maps_in, num_maps_out, k, padding=k // 2, bias=bias))

class SimpleMetricEmbedding(nn.Module):
    def __init__(self, input_channels, emb_size=32):
        super().__init__()
        self.emb_size = emb_size
        self.bn_relu_conv1 = _BNReluConv(input_channels, self.emb_size, k=3)
        self.bn_relu_conv2 = _BNReluConv(self.emb_size, self.emb_size, k=3)
        self.bn_relu_conv3 = _BNReluConv(self.emb_size, self.emb_size, k=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

    def get_features(self, img):
        x = self.bn_relu_conv1(img)
        x = self.maxpool(x)
        x = self.bn_relu_conv2(x)
        x = self.maxpool(x)
        x = self.bn_relu_conv3(x)
        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def loss(self, anchor, positive, negative):
        a_x = self.get_features(anchor)
        p_x = self.get_features(positive)
        n_x = self.get_features(negative)
        
        pos_dist = torch.norm(a_x - p_x, p=2, dim=1)
        neg_dist = torch.norm(a_x - n_x, p=2, dim=1)
        return F.relu(1 + pos_dist - neg_dist).mean()
    
class IdentityModel(nn.Module):
    def __init__(self):
        super(IdentityModel, self).__init__()

    def get_features(self, img):
        feats = img.view(img.size(0), -1)
        return feats