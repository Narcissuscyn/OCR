# #coding: utf-8
# import torch
# from torch.autograd import Variable
# import utils
# import dataset
# import os
# from PIL import Image
# import keys
# import models.crnn as crnn
#
# #os.environ["CUDA_VISIBLE_DEVICES"] ="1"
# model_path = './expr_cmp1/netCRNN_69_2713.pth'
# img_path = './data/0000002_1.jpg'
#
# alphabet=keys.alphabet
# nclass = len(alphabet)+1
# pre_model = torch.load(model_path)
# model = crnn.CRNN(32, 1, nclass, 512).cuda()
#
# mymodel=model.state_dict()
# for k,v in mymodel.items():
#     print(k,len(v))
#
# print('loading pretrained model from %s' % model_path)
# mymodel={}
# for k,v in pre_model.items():
#     mymodel[k[7:]]=v
#     print(k,len(v))
#
# model.load_state_dict(mymodel)
#
# converter = utils.strLabelConverter(alphabet)
#
# transformer = dataset.resizeNormalize((150, 32))
# image = Image.open(img_path).convert('L')
# image = transformer(image).cuda()
# image = image.view(1, *image.size())
# image = Variable(image)
#
# model.eval()
# preds = model(image)
#
# _, preds = preds.max(2)
# # preds = preds.squeeze(2)
# preds = preds.transpose(1, 0).contiguous().view(-1)
#
# preds_size = Variable(torch.IntTensor([preds.size(0)]))
# raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
# sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
# print('%-20s => %-20s' % (raw_pred, sim_pred))
#
#coding: utf-8


import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import os
import utils
import dataset
from PIL import Image
import models.crnn_origin as crnn
import keys
alphabet = keys.alphabet
nclass=len(alphabet)+1
# raw_input('\ninput:')
converter = utils.strLabelConverter(alphabet)
model = crnn.CRNN(32, 1, nclass, 512,1).cuda()
path = './expr1/netCRNN_79_2679.pth'
mymodel={}
pre_model = torch.load(path)

for k,v in pre_model.items():
    mymodel[k[7:]]=v
    print(k,len(v))

model.load_state_dict(mymodel)
print(model)


while 1:
    # im_name = raw_input("\nplease input file name:")
    im_name='./data/0000002_1.jpg'
    im_path =  './data/0000001_0.jpg'
    image = Image.open(im_path).convert('L')
    scale = image.size[1]*1.0 / 32
    w = image.size[0] / scale
    w = int(w)
    # print(w)

    transformer = dataset.resizeNormalize((w, 32))
    image = transformer(image).cuda()
    image = image.view(1, *image.size())
    image = Variable(image)
    model.eval()
    preds = model(image)
    _, preds = preds.max(2)
    # preds = preds.squeeze(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    print('%-20s => %-20s' % (raw_pred, sim_pred))