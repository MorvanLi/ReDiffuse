
# -*- coding: utf-8 -*-
import torch
from torch import nn
import math

from . import B_Conv as fn
# import B_Conv as fn
import torch.nn.functional as F
tranNum = 4

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

def time_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [batch_size x dim] Tensor of positional embeddings.
    """

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class EquivariantDownsample(nn.Module):

    def __init__(self, inNum, outNum, sizeP=3,
                 tranNum=2, inP=None,
                 padding=1, stride=2,
                 ifIni=0, bias=True,
                 Smooth=False, iniScale=1.0):

        super(EquivariantDownsample, self).__init__()

        if inP is None:
            inP = sizeP

        self.tranNum = tranNum
        self.outNum  = outNum
        self.inNum   = inNum
        self.sizeP   = sizeP
        self.stride  = stride

        Basis, Rank, weight = fn.GetBasis_PCA(sizeP, tranNum, inP, Smooth=Smooth)
        self.register_buffer("Basis", Basis)

        self.ifbias = bias

        if ifIni:
            expand = 1
        else:
            expand = tranNum
        self.expand = expand

        # [outNum, inNum, expand, Rank]
        self.weights = nn.Parameter(
            torch.Tensor(outNum, inNum, expand, Basis.size(3)),
            requires_grad=True
        )

        if padding is None:
            self.padding = 0
        else:
            self.padding = padding

        if bias:
            self.c = nn.Parameter(torch.Tensor(1, outNum, 1, 1))
        else:
            self.register_parameter('c', None)

        self.reset_parameters()

    def forward(self, input):

        if self.training:
            tranNum = self.tranNum
            outNum  = self.outNum
            inNum   = self.inNum
            expand  = self.expand

            # Basis: [i,j,o,k], weights: [m,n,a,k]
            tempW = torch.einsum('ijok,mnak->monaij', self.Basis, self.weights)

            Num = tranNum // expand
            tempWList = [
                torch.cat(
                    [
                        tempW[:, i*Num:(i+1)*Num, :, -i:, :, :],
                        tempW[:, i*Num:(i+1)*Num, :, :-i, :, :]
                    ],
                    dim=3
                )
                for i in range(expand)
            ]
            tempW = torch.cat(tempWList, dim=1)

            _filter = tempW.reshape(
                [outNum * tranNum,
                 inNum * self.expand,
                 self.sizeP,
                 self.sizeP]
            )

            if self.ifbias:
                _bias = self.c.repeat([1, 1, tranNum, 1])\
                             .reshape([1, outNum * tranNum, 1, 1])
                # 这里跟 Fconv_PCA 一样，训练时不缓存 filter，只缓存 bias（你原代码就是这样）
        else:
            _filter = self.filter
            if self.ifbias:
                _bias = self.bias

        output = F.conv2d(
            input, _filter,
            padding=self.padding,
            stride=self.stride,  # ★ 下采样关键在这里
            # dilation=1,
            # groups=1
        )

        if self.ifbias:
            output = output + _bias

        return output

    def train(self, mode=True):
        if mode:
            # 切回 train 模式时，丢掉缓存的 filter/bias
            if hasattr(self, "filter"):
                del self.filter
                if self.ifbias and hasattr(self, "bias"):
                    del self.bias
        elif self.training:
            # 第一次切到 eval() 时预计算好 filter / bias
            tranNum = self.tranNum
            outNum  = self.outNum
            inNum   = self.inNum
            expand  = self.expand

            tempW = torch.einsum('ijok,mnak->monaij', self.Basis, self.weights)
            Num = tranNum // expand
            tempWList = [
                torch.cat(
                    [
                        tempW[:, i*Num:(i+1)*Num, :, -i:, :, :],
                        tempW[:, i*Num:(i+1)*Num, :, :-i, :, :]
                    ],
                    dim=3
                )
                for i in range(expand)
            ]
            tempW = torch.cat(tempWList, dim=1)

            _filter = tempW.reshape(
                [outNum * tranNum,
                 inNum * self.expand,
                 self.sizeP,
                 self.sizeP]
            )
            self.register_buffer("filter", _filter)

            if self.ifbias:
                _bias = self.c.repeat([1, 1, tranNum, 1])\
                             .reshape([1, outNum * tranNum, 1, 1])
                self.register_buffer("bias", _bias)

        return super(EquivariantDownsample, self).train(mode)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.c is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.c, -bound, bound)






class EquivariantUpsample(nn.Module):

    def __init__(self, inNum, outNum, sizeP=3,
                 tranNum=2, inP=None,
                 padding=1, stride=2, output_padding=1,
                 ifIni=0, bias=True, Smooth=False, iniScale=1.0):

        super(EquivariantUpsample, self).__init__()

        if inP is None:
            inP = sizeP

        self.tranNum = tranNum
        self.outNum  = outNum
        self.inNum   = inNum
        self.sizeP   = sizeP
        self.stride  = stride
        self.output_padding = output_padding

        Basis, Rank, weight = fn.GetBasis_PCA(sizeP, tranNum, inP, Smooth=Smooth)
        self.register_buffer("Basis", Basis)

        self.ifbias = bias

        # 和 Fconv_PCA 一样：ifIni=1 时 expand=1，否则 expand=tranNum
        if ifIni:
            expand = 1
        else:
            expand = tranNum
        self.expand = expand

        # 权重形状和 Fconv_PCA 完全一致
        # [outNum, inNum, expand, Rank]
        self.weights = nn.Parameter(
            torch.Tensor(outNum, inNum, expand, Basis.size(3)),
            requires_grad=True
        )

        if padding is None:
            self.padding = 0
        else:
            self.padding = padding

        if bias:
            self.c = nn.Parameter(torch.Tensor(1, outNum, 1, 1))
        else:
            self.register_parameter('c', None)

        self.reset_parameters()

    def forward(self, input):

        if self.training:
            tranNum = self.tranNum
            outNum  = self.outNum
            inNum   = self.inNum
            expand  = self.expand

            # Basis: [i,j,o,k], weights: [m,n,a,k] → tempW: [m,o,n,a,i,j]
            tempW = torch.einsum('ijok,mnak->monaij', self.Basis, self.weights)

            Num = tranNum // expand
            tempWList = [
                torch.cat(
                    [
                        tempW[:, i*Num:(i+1)*Num, :, -i:, :, :],
                        tempW[:, i*Num:(i+1)*Num, :, :-i, :, :]
                    ],
                    dim=3
                )
                for i in range(expand)
            ]
            tempW = torch.cat(tempWList, dim=1)

            # ★ 和 Fconv_PCA 唯一不同：这里是反卷积，所以需要 swap IO 维
            # conv2d:       [outNum*tranNum, inNum*expand, k, k]
            # conv_transpose: [inNum*expand, outNum*tranNum, k, k]
            _filter = tempW.reshape(
                [inNum * self.expand,
                 outNum * tranNum,
                 self.sizeP,
                 self.sizeP]
            )

            if self.ifbias:
                _bias = self.c.repeat([1, 1, tranNum, 1])\
                             .reshape([1, outNum * tranNum, 1, 1])
        else:
            _filter = self.filter
            if self.ifbias:
                _bias = self.bias

        output = F.conv_transpose2d(
            input, _filter,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=1
        )

        if self.ifbias:
            output = output + _bias

        return output

    def train(self, mode=True):
        if mode:
            # 切回 train 时，删掉缓存
            if hasattr(self, "filter"):
                del self.filter
                if self.ifbias and hasattr(self, "bias"):
                    del self.bias
        elif self.training:
            # 第一次 .eval() 时，预先算好 filter / bias 缓存
            tranNum = self.tranNum
            outNum  = self.outNum
            inNum   = self.inNum
            expand  = self.expand

            tempW = torch.einsum('ijok,mnak->monaij', self.Basis, self.weights)
            Num = tranNum // expand
            tempWList = [
                torch.cat(
                    [
                        tempW[:, i*Num:(i+1)*Num, :, -i:, :, :],
                        tempW[:, i*Num:(i+1)*Num, :, :-i, :, :]
                    ],
                    dim=3
                )
                for i in range(expand)
            ]
            tempW = torch.cat(tempWList, dim=1)

            _filter = tempW.reshape(
                [inNum * self.expand,
                 outNum * tranNum,
                 self.sizeP,
                 self.sizeP]
            )
            self.register_buffer("filter", _filter)

            if self.ifbias:
                _bias = self.c.repeat([1, 1, tranNum, 1])\
                             .reshape([1, outNum * tranNum, 1, 1])
                self.register_buffer("bias", _bias)

        return super(EquivariantUpsample, self).train(mode)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.c is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.c, -bound, bound)




