#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc     :data preprocess
# @Time    : 18-4-23 下午7:06
# @Author  : Narcissus
# @File    : my_util.py
# @Contact : ***

import os
import cv2
import shutil
import numpy as np
def rename_image(dst_dir='',sou_dir='',img_nam_sta=188858,img_max_idx=47544):
    # img_nam_sta=188858
    index=1
    while index<=img_max_idx:
        res = cv2.imread(sou_dir + str(index).zfill(7)+'.jpg')
        if res is not None:
            filename=dst_dir + str(img_nam_sta).zfill(7) + '.jpg'
            cv2.imwrite(filename, res)
            img_nam_sta += 1
        index+=1



def resize_image(imgH,img_dir,dst_dir):
    """re size an imagemv 01

    given hight, resize an image with equal proportion

    Args:
        :param imgH: the hight that an image should be resized to

    Returns:
        :return: no return

    """

    for root, dirs, files in os.walk(img_dir, topdown=False):
        for index, file in enumerate(files):
            image=cv2.imread(img_dir+file)
            if image is not None:
                h = image.shape[0]
                w=image.shape[1]
                ratio = float(h) / float(imgH)
                imgW = int(np.ceil(w /ratio))
                image= cv2.resize(image,(imgW,imgH), interpolation=cv2.INTER_CUBIC)
                if (image.shape)[0]!=32:
                    print('error!')
                cv2.imwrite(dst_dir+file,image)
            else:
                print('image '+file+' is NoneType')

def get_ver_img(data_root='', vert_root='', hori_root='', img_max_idx=0, label_pth='', text_vert='', text_hroi=''):
    """
    get the images with verticle texts
    :return:None
    """
    text_vert = open(text_vert, 'w+')
    text_hroi = open(text_hroi, 'w+')

    label_pth = open(label_pth, 'r')
    label = label_pth.readlines()
    label_pth.close()


    index = 1
    label_idx=0
    vert_num=1

    hori_num=1
    while index <=img_max_idx:
        img_pth = data_root + str(index).zfill(7) + '.jpg'
        # img_pth = data_root + 'line_'+str(index)+ '.jpg'
        if not os.path.exists(img_pth) :
            print('path:', img_pth, ' is not exist!')
            index += 1
            continue


        sou_img = cv2.imread(img_pth)

        if sou_img is None:
            print('image',img_pth,' can not open!')
            index += 1
            continue
        width = sou_img.shape[1]
        height = sou_img.shape[0]

        if height / width >= 2.0:  # vertical image
            cv2.imwrite(vert_root + str(vert_num).zfill(7) + '.jpg', sou_img)
            # cv2.imwrite(vert_root +'line_'+ str(index) + '.jpg', sou_img)
            text_vert.write(label[label_idx])
            label_idx+=1
            vert_num+=1

        else:
            cv2.imwrite(hori_root + str(hori_num).zfill(7) + '.jpg', sou_img)
            # cv2.imwrite(hori_root +'line_'+ str(index) + '.jpg', sou_img)
            text_hroi.write(label[label_idx])
            label_idx+=1
            hori_num+=1
        index+=1
    text_hroi.close()
    text_vert.close()


def get_data_cmp():
    """
    merge data:train+test+icdar
    :return:
    """
    crop_root = '/home/new/File/DataSet/crop_root/test_rename/'
    file_path = '/home/new/File/DataSet/crop_root/test1'
    txt_num = 173602
    for index in range(15256):
        res = cv2.imread(file_path + '/' + str(index + 1).zfill(7) + '.jpg')
        if res is not None:
            filename = crop_root + str(txt_num).zfill(7) + '.jpg'
            cv2.imwrite(filename, res)
            txt_num += 1


def save_txt_sole(txt_pth='',save_root=''):
    """
    save each sentence in a file individually
    :return:None
    """
    # txt_pth = '/home/new/File/DataSet/crop_root/data_cmp_hori.txt'
    # save_root = '/home/new/File/DataSet/crop_root/data_hori_txt/'
    f = open(txt_pth, 'r')

    txts = f.readlines()
    length = txts.__len__()
    for i in range(length):
        file = open(save_root + str(i + 1).zfill(7) + '.txt', 'w+')
        file.write(txts[i])
        file.close()


def merge_img_txt(img_dir, txt_dir, tar_dir, file_num, tar_txt_name):
    """
    1.move images in img_dir to tar_img_pth

    2.concate individual sentence into one file

    :param img_dir:
    :param txt_dir:
    :param tar_dir:
    :param file_num:
    :param tar_txt_name:
    :return:None
    """
    out = open(tar_dir + tar_txt_name, 'w+')
    i = 1
    index = 1
    while index <= file_num:
        img_pth = img_dir + str(i).zfill(7) + '.jpg'
        txt_pth = txt_dir + str(i).zfill(7) + '.txt'
        tar_img_pth = tar_dir + str(index).zfill(7) + '.jpg'
        if os.path.exists(img_pth):
            shutil.move(img_pth, tar_img_pth)

            f = open(txt_pth, 'r')
            line = f.readline()
            f.close()

            out.write(line)
            index += 1
        i += 1
    out.close()


def cut_train_val(sou_dir='', val_dir='', label_dir='',val_num=20000):
    """
    cut the cmplete data to train and test
    :return:None
    """
    # tar_root = '/home/new/File/DataSet/crop_root/my_test/'
    # sou_img = '/home/new/File/DataSet/crop_root/data_hori/'
    i = 0
    while i <val_num :
        n = np.random.randint(1, 236401)
        # print(n)
        file_name = str(n).zfill(7)

        if os.path.exists(sou_dir + file_name + '.jpg'):
            shutil.move(sou_dir + file_name + '.jpg', val_dir + file_name + '.jpg')

            shutil.move(label_dir + file_name + '.txt', val_dir + file_name + '.txt')
            i += 1


def get_img_nam_sub_lst(img_dir):
    """
    generate an image name list,the name with a subcribe according to the dir
    :return:

    """
    img_nam_lst=[]
    out=open('img_nam_lst.txt','w+')
    for root, dirs, files in os.walk(img_dir, topdown=False):
        temp=[x[:8]+chr(int(x[8:-4])+65) for x in files ]
        temp=sorted(temp)
        for index, file in enumerate(temp):
            print(file)#sorted(a,key=lambda x: int(x[8:-4]))
            file=file[:8]+str((ord(file[8:])-65))+'\n'
            out.write(file)
    out.close()

'''
rename images by increasing number
'''

def ren_syn_img(txt_num=0, dst='', src=''):
    i=0
    sub_scr=0
    num=1
    while num<=txt_num:

        imagePath = src + str(i).zfill(5) + '_' + str(sub_scr) + '.png'
        sub_scr += 1
        if not os.path.exists(imagePath):
            # print('%s does not exist' % imagePath)
            sub_scr = 0
            i += 1
            continue
        im=cv2.imread(imagePath)
        dst_pth=dst+str(num).zfill(7)+'.jpg'
        cv2.imwrite(dst_pth,im)
        num+=1

def get_name_list(dir,dst):
    """
       generate an image name list,the name with a subcribe according to the dir
       :return:

       """
    out = open(dst, 'w+')
    for root, dirs, files in os.walk(dir, topdown=False):
        temp =sorted(files,key = lambda x:int(x[5:-4]))
        for index, file in enumerate(temp):
            print(file)  # sorted(a,key=lambda x: int(x[8:-4]))
            file =file+ '\n'
            out.write(file)
    out.close()



def sort_results(sou_lst='', dst_lst=''):
    '''
    to sort test results in increasing number
    :param sou_lst:
    :param dst_lst:
    :return:
    '''
    sou_lst=open(sou_lst,'r')

    dst_lst=open(dst_lst,'w+')

    labels=sou_lst.readlines()
    labels[0]=labels[0][1:]
    sou_lst.close()

    temp = sorted(labels, key=lambda x: int(x.split(' ')[0][5:-4]))

    for label in temp:
        dst_lst.write(label)

    dst_lst.close()


###########################################################################



# img_dir='/home/new/File/DataSet/crop_root/my_test/'
# txt_dir='/home/new/File/DataSet/crop_root/my_test_txt/'
# tar_dir='/home/new/File/DataSet/crop_root/my_test1/'
# file_num=10000
# tar_txt_name='my_test.txt'
# merge_img_txt(img_dir,txt_dir,tar_dir,file_num,tar_txt_name)
############################################################################
# #resize test images
# resize_image(32,'/home/new/File/DataSet/crop_root/my_test/','/home/new/File/DataSet/crop_root/my_test_dst/')
#
# #resize validate images
# resize_image(32,'/home/narcissus/file/dataset/ocr/validate/','/home/narcissus/file/dataset/ocr/validate_res/')
#
# #resize train images
# resize_image(32,'/home/narcissus/file/dataset/ocr/train/','/home/narcissus/file/dataset/ocr/train_res/')

#resize pretrain data:
# resize_image(32,'/home/narcissus/file/recent_job/ocr/syn_data/dataset/train/','/home/narcissus/file/recent_job/ocr/syn_data/dataset/syn_cha_res/')

#resize fina_test data:
#
# resize_image(32,'/home/narcissus/file/dataset/ocr/test_hori_cut/','/home/narcissus/file/dataset/ocr/test_hori_cut_res/')
#
# resize_image(32,'/home/narcissus/file/dataset/ocr/test_vert_cut/','/home/narcissus/file/dataset/ocr/test_vert_cut_res/')


#resize all train data:icpr2018+icdar hua ke + synthesis train
# resize_image(32,'/home/narcissus/file/dataset/ocr/train/','/home/narcissus/file/dataset/ocr/train/')



############################################################################
#get image name list:
# get_img_nam_lst('/home/new/File/DataSet/crop_root/my_test_dst')

# get_name_list('/home/narcissus/file/dataset/ocr/test_line_image','/home/narcissus/file/dataset/ocr/test_image_list1.txt')

###########################################################################

##rename the synthetic image
# 47544
# ren_syn_img(47544,'/home/narcissus/file/recent_job/ocr/syn_data/dataset/syn_train/','/home/narcissus/file/recent_job/ocr/syn_data/dataset/syn_cha_res/')

###########################################################################
#get vertical image and horizental image
# get_ver_img('/home/narcissus/file/dataset/ocr/test_line_image/','/home/narcissus/file/dataset/ocr/test_vert/','/home/narcissus/file/dataset/ocr/test_hori/',148738,'/home/narcissus/file/dataset/ocr/test_nam_lst.txt','/home/narcissus/file/dataset/ocr/test_vert_nam_lst.txt','/home/narcissus/file/dataset/ocr/test_hori_nam_lst.txt')

# get_ver_img('/home/narcissus/file/dataset/ocr/data_cmp/','/home/narcissus/file/dataset/ocr/train_vert/','/home/narcissus/file/dataset/ocr/train_hori/',188857,'/home/narcissus/file/dataset/ocr/data_cmp.txt','/home/narcissus/file/dataset/ocr/train_vert_label.txt','/home/narcissus/file/dataset/ocr/train_hori_label.txt')
#
# get_ver_img('/home/narcissus/file/dataset/ocr/validate/','/home/narcissus/file/dataset/ocr/val_vert/','/home/narcissus/file/dataset/ocr/val_hori/',30000,'/home/narcissus/file/dataset/ocr/validate_label.txt','/home/narcissus/file/dataset/ocr/val_vert_label.txt','/home/narcissus/file/dataset/ocr/val_hori_label.txt')

###########################################################################
# rename synthesis images ,in order to add it to previous trainning data
# rename_image(dst_dir='/home/narcissus/file/dataset/ocr/syn_train1/',sou_dir='/home/narcissus/file/dataset/ocr/syn_train/',img_nam_sta=188858,img_max_idx=47544)


###########################################################################

# devide train set and validate set:
# save_txt_sole('/home/narcissus/file/dataset/ocr/train_hori_label.txt','/home/narcissus/file/dataset/ocr/train_hori_sole/')
# cut_train_val('/home/narcissus/file/dataset/ocr/train_hori/','/home/narcissus/file/dataset/ocr/validate/','/home/narcissus/file/dataset/ocr/train_hori_sole/')

# merge_img_txt('/home/narcissus/file/dataset/ocr/validate/', '/home/narcissus/file/dataset/ocr/validate_sole/', '/home/narcissus/file/dataset/ocr/validate1/', 20000, 'validate_label.txt')

# merge_img_txt('/home/narcissus/file/dataset/ocr/train_hori/', '/home/narcissus/file/dataset/ocr/train_hori_sole/', '/home/narcissus/file/dataset/ocr/train/', 159082, 'train_label.txt')

###########################################################################



sort_results('/home/narcissus/file/dataset/ocr/test_result4/results.txt','/home/narcissus/file/dataset/ocr/test_result4/result.txt')

###########################################################################
