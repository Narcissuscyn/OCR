import cv2

# # 1、读取图像，并把图像转换为灰度图像并显示
# img = cv2.imread("./origin/1.jpg")  # 读取图片
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # 转换了灰度化
# cv2.imshow('gray', img_gray)  # 显示图片
# cv2.waitKey(0)
#
# # 2、将灰度图像二值化，设定阈值是100
# # img_thre = img_gray
# # cv2.threshold(img_gray, 120, 255, cv2.THRESH_BINARY_INV, img_thre)
# # cv2.imshow('threshold', img_thre)
# # cv2.waitKey(0)

import os
crop_root='../data'
file_root='../data'
import cv2
import numpy as np
from matplotlib import pyplot as plt
CROP_THRESH_HOLD=3

label_file='/home/narcissus/file/dataset/ocr/test_hori_cut_nam_lst1.txt'

label_file=open(label_file,'w+')

for root, dirs, files in os.walk(file_root,topdown=False):

    # files=sorted(files,key = lambda x:int(x[5:-4]))

    for index,file in enumerate(files):
        # print(file)
        img = cv2.imread(file_root+'/'+file)
        img=np.rot90(img)

        img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #
        kernel_3x3 = np.ones((3, 3), np.float32) /9
        # We apply the filter and display the image
        img_gray = cv2.filter2D(img_gray, -1, kernel_3x3)
        # kernel_sharpening = np.array([[-1, -1, -1],
        #                               [-1, 9, -1],
        #                               [-1, -1, -1]])
        #
        # img_gray = cv2.filter2D(img_gray, -1, kernel_sharpening)
        kernel = np.ones((3, 3), np.uint8)
        img_gray = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel)
        kernel = np.ones((1, 1), np.uint8)

        img_gray = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel)

        ret3, img_thre = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


        # kernel = np.ones((1, 1), np.uint8)
        #
        #
        # 3、保存黑白图片
        # cv2.imwrite(crop_root+'/'+file, img_thre)

        # 4、分割字符

        white = []  # 记录每一行的白色像素总和
        black = []  # ..........黑色.......

        height = img_thre.shape[0]
        width = img_thre.shape[1]
        white_max = 0
        black_max = 0
        val_fir_col=0
        val_lst_col=0

        for j in range(height-1):
            if img_thre[j][0]==255:
                val_fir_col+=0
            else:
                val_fir_col += 1
        for j in range(height - 1):
            if img_thre[j][width-1] == 255:
                val_lst_col += 0
            else:
                val_lst_col += 1


        # 计算每一行的黑白色像素总和
        for i in range(height):
            s = 0  # 这一行白色总数
            t = 0  # 这一行黑色总数
            for j in range(width):
                if img_thre[i][j] == 255:
                    s += 1
                if img_thre[i][j] == 0:
                    t += 1
            white_max = max(white_max, s)
            black_max = max(black_max, t)
            white.append(s)
            black.append(t)

        # fig = plt.figure()
        # plt.bar(np.arange(black.__len__()),black,alpha=0.9, width = 0.35, facecolor = 'lightskyblue', edgecolor = 'white', label='one', lw=1)
        # fig.savefig(crop_root + '/'+file[:-4]+'bar.jpg')
        # fig.clf()
        arg = False  # False表示白底黑字；True表示黑底白字


        if val_fir_col>height/2 and val_lst_col>height/2:
            arg=True


        print(white_max,' ',arg)
        # 分割图像

        def find_mid(bins):
            idxes=[]
            idxes.append(0)
            i=0
            flag=False
            while i<len(bins):
                # print('----------')
                if bins[i]==1:
                    flag=True
                if flag:
                    if bins[i]==0:
                        idx=i
                        while idx<len(bins)-1 and bins[idx]!=1:
                            idx+=1
                        if idx<len(bins)-1:
                            idxes.append((i+idx)/2)
                        i= idx
                i+=1
            idxes.append(len(bins)-1)
            return  idxes
        def find_end1(start_):
            my_black=[]
            my_white=[]

            if arg:#黑底白字

                for idx,item in enumerate(white):
                    if item<CROP_THRESH_HOLD:
                        my_white.append(0)
                    else:
                        my_white.append(1)
                # plt.plot(np.arange(my_white.__len__()), my_white)
                # fig.savefig(crop_root + '/'+file[:-4]+'bin.jpg' )
                # fig.clf()
                return  find_mid(my_white)

            else:
                for idx,item in enumerate(black):
                    if item<CROP_THRESH_HOLD:
                        my_black.append(0)
                    else:
                        my_black.append(1)
                # plt.plot(np.arange(my_black.__len__()), my_black)
                # fig.savefig(crop_root + '/' + file[:-4] + 'bin.jpg')
                # fig.clf()
                return find_mid(my_black)



        n = 1
        start = 1
        end = 2
        print(height)
        # s =0

        results=find_end1(start)
        i=0
        sub=0
        write_flag=False
        while i <= len(results) - 2:
            if results[i+1]-int(results[i])>width/2:
                write_flag=True
                cj = img[int(results[i]):int(results[i+1]), 1:width]

                cj=np.rot90(cj)
                cj=np.rot90(cj)
                cj=np.rot90(cj)

                cv2.imwrite(crop_root + '/' + file[:-4] + '_' + str(sub) + '.jpg', cj)
                # label_file.write(file[:-4] + '_' + str(sub) + '.jpg\n')
                sub+=1
            i+=1
        if not write_flag:
            img = np.rot90(img)
            img = np.rot90(img)
            img = np.rot90(img)

            cv2.imwrite(crop_root + '/' + file[:-4] + '_' + str(0) + '.jpg', img)
            # label_file.write(file[:-4] + '_' + str(0) + '.jpg\n')



label_file.close()
