import torch
from torch.autograd import Variable
import torch.nn as nn
import utils


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


# class BasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#
#         return out


class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Downsample(nn.Sequential):

    def __init__(self, inplanes, outplanes, stride=2, padding=1):
        super(Downsample, self).__init__(
            nn.Conv2d(inplanes, outplanes*1, kernel_size=3, stride=stride, bias=False, padding=padding),
            nn.BatchNorm2d(outplanes),
        )



class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut, ngpu):
        super(BidirectionalLSTM, self).__init__()
        self.ngpu = ngpu

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = utils.data_parallel(
            self.rnn, input, self.ngpu)  # [T, b, h * 2]

        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = utils.data_parallel(
            self.embedding, t_rec, self.ngpu)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, ngpu=1, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        self.ngpu = ngpu
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 2, 2, 1, 2, 2, 2]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()
        # input: 1x32x128
        cnn.add_module('conv', conv3x3(1, 64))  # 64x64x32*32
        # cnn.add_module('res0',Bottleneck(64,64,1,None))# 64x64x32*32
        cnn.add_module('res1',Bottleneck(64,64,1,None))# 64x64x32*32
        cnn.add_module('res2',Bottleneck(64,128,2,Downsample(64, 128)))# 64x128x16*16
        # cnn.add_module('res4',Bottleneck(128,128,1,None))# 64x128x16*16

        cnn.add_module('res5',Bottleneck(128,256,2,Downsample(128, 256)))# 64x256x8*8
        # cnn.add_module('res7', Bottleneck(256, 256, 1, None))# 64x256x8*8

        cnn.add_module('res8', Bottleneck(256, 512, (2,1), Downsample(256, 512, stride=(2, 1), padding=(1, 1))))# 64x512x4*4

        # cnn.add_module('res13', Bottleneck(512, 1024, 1, None))
        cnn.add_module('res9', Bottleneck(512, 1024, (2,1), Downsample(512, 1024, stride=(2, 1), padding=(1, 1))))# 64x1024x2*4
        cnn.add_module('res10', Bottleneck(1024, 1024, (2,1), Downsample(1024,1024, stride=(2, 1), padding=(1, 1))))# 64x1024x1*4

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(1024, nh, nh, ngpu),
            BidirectionalLSTM(nh, nh, nclass, ngpu)
        )

    def forward(self, input):
        # for i in range(len(self.cnn)):
        #     input = self.cnn[i](input)
        #     print(input.size())
        #     print('===========')
        #
        # return input

        conv = utils.data_parallel(self.cnn, input, self.ngpu)

        # conv=self.cnn(input)
        b, c, h, w = conv.size()  # batchsize,channel,image hight,image width
        # print('conv.size():', conv.size())
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        # print('conv.size():', conv.size())

        # rnn features
        output = utils.data_parallel(self.rnn, conv, self.ngpu)

        return output