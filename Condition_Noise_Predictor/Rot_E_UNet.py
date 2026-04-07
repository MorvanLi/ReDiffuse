# -*- coding: utf-8 -*-

import torch
from torch import nn
import numpy as np
import random
import torch.nn.functional as F
from abc import abstractmethod
from . import B_Conv as fn  # Rot-E
from . import e_linear as en
# import B_Conv as fn  # Rot-E
# import e_linear as en

import time
import json

from .__init__ import time_embedding
from .__init__ import Upsample
from .__init__ import Downsample
from .__init__ import EquivariantDownsample
from .__init__ import EquivariantUpsample

# use GN for norm layer
# def group_norm(channels, tranNum=4):
#     # return fn.F_GN(channels // tranNum, tranNum=tranNum)
#     return fn.F_BN(channels // tranNum, tranNum=tranNum)
#     # return nn.GroupNorm(32, channels)

def group_norm(channels, tranNum=4):
    return fn.F_GN(channels // tranNum, tranNum=tranNum, num_groups=16)



# ���� time_embedding �� block
class TimeBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        """
        
        """


class TimeSequential(nn.Sequential, TimeBlock):
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimeBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x



class ResBlock(TimeBlock):
    def __init__(self, in_channels, out_channels, time_channels, dropout, tranNum=4, iniScale=0.1, Smooth=False):
        super().__init__()
        self.conv1 = nn.Sequential(
            group_norm(in_channels),
            nn.SiLU(),
            fn.Fconv_PCA(sizeP=3, inNum=in_channels // tranNum, outNum=out_channels // tranNum, padding=1, inP=3, tranNum=tranNum, ifIni=0, Smooth=Smooth, iniScale=iniScale)
        )

        # pojection for time step embedding   
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels)
        )

        self.conv2 = nn.Sequential(
            group_norm(out_channels),
            nn.SiLU(),
            fn.F_Dropout(zero_prob=dropout, tranNum=tranNum),

            fn.Fconv_PCA(sizeP=3, inNum=out_channels // tranNum, outNum=out_channels // tranNum, tranNum=tranNum,
                         padding=1, inP=3, ifIni=0, Smooth=Smooth, iniScale=iniScale)
        )

        if in_channels != out_channels:
            self.shortcut = fn.Fconv_1X1(inNum=in_channels // tranNum, outNum=out_channels // tranNum, tranNum=tranNum, ifIni=0, bias=True, Smooth=False, iniScale=iniScale)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t):
        """
        `x` has shape `[batch_size, in_dim, height, width]`
        `t` has shape `[batch_size, time_dim]`
        """
        h = self.conv1(x)
        # Add time step embeddings
        h += self.time_emb(t)[:, :, None, None]
        h = self.conv2(h)
        return h + self.shortcut(x)


class NoisePred(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 model_channels,
                 num_res_blocks,
                 dropout,
                 time_embed_dim_mult,
                 down_sample_mult,
                 tranNum=4,
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout
        self.down_sample_mult = down_sample_mult

        ##  Rot-E Parametes
        Smooth = False
        inP = 3
        iniScale = 0.1
        kernel_rot = 3
        tranNum = tranNum

        # time embedding
        time_embed_dim = model_channels * time_embed_dim_mult
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

       
        down_channels = [model_channels * i for i in down_sample_mult]
        up_channels = down_channels[::-1]

        
        downBlock_chanNum = [num_res_blocks + 1] * (len(down_sample_mult) - 1)
        downBlock_chanNum.append(num_res_blocks)
        upBlock_chanNum = downBlock_chanNum[::-1]
        self.downBlock_chanNum_cumsum = np.cumsum(downBlock_chanNum)
        self.upBlock_chanNum_cumsum = np.cumsum(upBlock_chanNum)[:-1]

        
        self.inBlock = fn.Fconv_PCA(sizeP=kernel_rot, inNum=in_channels, outNum=down_channels[0] // tranNum, tranNum=tranNum,  inP=inP, padding=(kernel_rot-1)//2, ifIni=1, Smooth=Smooth, iniScale=iniScale)
        self.downBlock = nn.ModuleList()

        down_init_channel = model_channels
        for level, channel in enumerate(down_channels):
            for _ in range(num_res_blocks):
                layer1 = ResBlock(in_channels=down_init_channel,
                                  out_channels=channel,
                                  time_channels=time_embed_dim,
                                  dropout=dropout)
                down_init_channel = channel
                self.downBlock.append(TimeSequential(layer1))
            
            if level != len(down_sample_mult) - 1:
                down_layer = EquivariantDownsample(inNum=channel//tranNum, outNum=channel//tranNum, tranNum=tranNum)
                self.downBlock.append(TimeSequential(down_layer))

        # middle block
        self.middleBlock = nn.ModuleList()
        for _ in range(num_res_blocks):
            layer2 = ResBlock(in_channels=down_channels[-1],
                              out_channels=down_channels[-1],
                              time_channels=time_embed_dim,
                              dropout=dropout)
            self.middleBlock.append(TimeSequential(layer2))

        
        self.upBlock = nn.ModuleList()
        up_init_channel = down_channels[-1]
        for level, channel in enumerate(up_channels):
            if level == len(up_channels) - 1:
                out_channel = model_channels
            else:
                out_channel = channel // 2
            for _ in range(num_res_blocks):
                layer3 = ResBlock(in_channels=up_init_channel,
                                  out_channels=out_channel,
                                  time_channels=time_embed_dim,
                                  dropout=dropout)
                up_init_channel = out_channel
                self.upBlock.append(TimeSequential(layer3))
            if level > 0:
                up_layer = EquivariantUpsample(inNum=out_channel//tranNum, outNum=out_channel//tranNum, tranNum=tranNum)
                self.upBlock.append(TimeSequential(up_layer))

        # out block
        self.outBlock = nn.Sequential(
            group_norm(model_channels),
            nn.SiLU(),
            fn.Fconv_PCA_out(sizeP=3, inNum=model_channels//tranNum, outNum=out_channels, tranNum=tranNum, inP=inP, padding=1, ifIni=0, Smooth=Smooth, iniScale=iniScale),
        )

    def forward(self, x, timesteps):
        embedding = time_embedding(timesteps, self.model_channels)

        time_emb = self.time_embed(embedding)


        res = []

        # in stage
        x = self.inBlock(x)

        # down stage
        h = x
        num_down = 1
        for down_block in self.downBlock:
            h = down_block(h, time_emb)
            if num_down in self.downBlock_chanNum_cumsum:
                res.append(h)
            num_down += 1

        # middle stage
        for middle_block in self.middleBlock:
            h = middle_block(h, time_emb)
        h = h + res.pop()
        assert len(res) == len(self.upBlock_chanNum_cumsum)

        num_up = 1
        for up_block in self.upBlock:
            if num_up in self.upBlock_chanNum_cumsum:  # [2,5,8]
                h = up_block(h, time_emb)
                h_crop = h[:, :, :res[-1].shape[2], :res[-1].shape[3]]
                h = h_crop + res.pop()
            else:
                h = up_block(h, time_emb)
            num_up += 1
        assert len(res) == 0


        # out stage
        out = self.outBlock(h)

        return out



def train(config_path):
    timestr = time.strftime('%Y%m%d_%H%M%S')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    # Condition Noise Predictor
    in_channels = config["Condition_Noise_Predictor"]["UNet"]["in_channels"]
    out_channels = config["Condition_Noise_Predictor"]["UNet"]["out_channels"]
    model_channels = config["Condition_Noise_Predictor"]["UNet"]["model_channels"]
    num_res_blocks = config["Condition_Noise_Predictor"]["UNet"]["num_res_blocks"]
    dropout = config["Condition_Noise_Predictor"]["UNet"]["dropout"]
    time_embed_dim_mult = config["Condition_Noise_Predictor"]["UNet"]["time_embed_dim_mult"]
    down_sample_mult = config["Condition_Noise_Predictor"]["UNet"]["down_sample_mult"]
    model = NoisePred(in_channels, out_channels, model_channels, num_res_blocks, dropout, time_embed_dim_mult,
                      down_sample_mult, tranNum=4)
    return model

# ------------------------------
# 1. discrete rotation operator
# ------------------------------
def rotate_tensor_90(x, k=1):
    # x: [B, C, H, W]
    return torch.rot90(x, k, dims=[2, 3])


# ------------------------------
# 2. rotate back output channels (robust)
#    If channels are organized as (field × tranNum),
#    we must roll rotation group dimension.
#    Otherwise just return spatially rotated tensor.
# ------------------------------
def rotate_group_channels(x, tranNum, k):
    """
    x: [B, C, H, W]
    tranNum: number of rotation group elements (e.g. 4 or 8)
    k: number of 90-degree rotations (k=1 => 90deg)
    """
    B, C, H, W = x.shape

    # if channels are divisible by tranNum, treat as group representation
    if C % tranNum == 0 and C > 0:
        fields = C // tranNum
        # reshape into [B, fields, tranNum, H, W], roll group dim
        x = x.view(B, fields, tranNum, H, W)
        x = torch.roll(x, shifts=k, dims=2)
        return x.view(B, C, H, W)
    else:
        # channels are not group-structured (e.g. RGB) -> do nothing on channels
        return x


# ------------------------------
# 3. test function (robust)
# ------------------------------
@torch.no_grad()
def test_equivariance(model, tranNum=4, device="cuda" if torch.cuda.is_available() else "cpu"):
    model = model.to(device)
    model.eval()

    B, C_in, H, W = 1, model.in_channels, 256, 256
    x = torch.randn(B, C_in, H, W, device=device)
    t = torch.randint(0, 1000, (B,), device=device).long()

    # ----- baseline -----
    y = model(x, t)                     # model output (could be image or group rep)

    # ----- rotated input -----
    k = 1   # rotate 90 degrees
    x_rot = rotate_tensor_90(x, k)
    y_rot_in = model(x_rot, t)

    # ----- rotate original output spatially -----
    # rotate spatial dims first
    y_rot_out = rotate_tensor_90(y, k)

    # if output has group channels, also roll group channels
    y_rot_out = rotate_group_channels(y_rot_out, tranNum, k)

    # ----- compute error -----
    # if either tensor shapes differ, raise informative error
    if y_rot_in.shape != y_rot_out.shape:
        raise RuntimeError(f"Shape mismatch: y_rot_in.shape={y_rot_in.shape}, y_rot_out.shape={y_rot_out.shape}")

    diff = (y_rot_in - y_rot_out).abs().max().item()
    rel = diff / (y_rot_out.abs().max().item() + 1e-12)

    print("===========================================")
    print("Rotation Equivariance Test (C{})".format(tranNum))
    print("-------------------------------------------")
    print("Max Absolute Error:  {:.6e}".format(diff))
    print("Relative Error:      {:.6e}".format(rel))

    if diff < 1e-3:
        print("✓ Model is ROTATION EQUIVARIANT (within tol)")
    else:
        print("✗ Model BREAKS equivariance (error > tol)")
    print("===========================================")

    return diff, rel


def disable_time_emb(model):
    # find all TimeScalarToGroup instances and replace their net with zero output
    for name, m in model.named_modules():
        if m.__class__.__name__ == "TimeScalarToGroup":
            # monkeypatch its forward
            def zero_forward(self, t):
                B = t.shape[0]
                return torch.zeros(B, self.fields * self.tranNum, device=t.device)
            m.forward = zero_forward.__get__(m, m.__class__)  # bind

def patched_upsample_forward(self, x):
    # x: [B, C, H, W], C = fields * tranNum
    B, C, H, W = x.shape
    # nearest-equivalent by repeating pixels
    x_up = x.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,2,2).reshape(B, C, H*2, W*2)
    # apply small 1x1 group conv to match original projection
    x_up = self.proj(x_up)
    return x_up

# apply monkeypatch to all Rot_E_Upsample instances
def patch_upsamples(model):
    for name, m in model.named_modules():
        if m.__class__.__name__ == "Rot_E_Upsample":
            m.forward = patched_upsample_forward.__get__(m, m.__class__)


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    config_path = "../config.json"
    model = train(config_path)
    model.eval()
    input = torch.rand(2, 9, 256, 256)
    t = torch.randint(0, 1000, (2,), device="cpu").long()
    predicted_noise = model(input, t)
    print(predicted_noise.shape)
    test_equivariance(model, tranNum=4)








