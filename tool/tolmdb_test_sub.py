import os
import lmdb  # install lmdb by "pip install lmdb"
import cv2
# import re
# import Image
import codecs
import numpy as np
import imghdr


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    try:
        imageBuf = np.fromstring(imageBin, dtype=np.uint8)
        img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
        imgH, imgW = img.shape[0], img.shape[1]
    except:
        return False
    else:
        if imgH * imgW == 0:
            return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            # print(k, v)
            txn.put(k, v)


def createDataset(outputPath,img_num, labelList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert (img_num == len(labelList))
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 0
    sub_scr=0
    i=1
    while i <=7445:
        # imagePath = './recognition/'+''.join(imagePathList[i]).split()[0].replace('\n','').replace('\r\n','')
        imagePath = '/home/new/File/DataSet/crop_root/my_test_dst/' + str(i).zfill(7) +'_'+str(sub_scr)+ '.jpg'
        sub_scr+=1
        label = ''.join(labelList[i])
        label = bytes(label, encoding="utf8")
        if not os.path.exists(imagePath):
            # print('%s does not exist' % imagePath)
            sub_scr=0
            i+=1
            continue

        with open(imagePath, 'rb') as f:
            imageBin = f.read()

        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue
        imageKey = b'image-%09d' % cnt
        labelKey = b'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label
        if lexiconList:
            lexiconKey = b'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            # print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    cache[b'num-samples'] = bytes(str(cnt), encoding="utf8")
    writeCache(env, cache)
    print('Created dataset with %d samples' % cnt)


if __name__ == '__main__':
    outputPath = "./final_test_sub"
    # imgdata = open("/home/new/File/DataSet/crop_root/test1/test1.txt")
    # imagePathList = list(imgdata)
    img_num=27375
    labelList = []
    for line in range(img_num):
        # word = line.split()[1]
        word ='1'

        labelList.append(word)
    createDataset(outputPath, img_num,labelList)
    # pass