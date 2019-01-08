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
            print(k, v)
            txn.put(k, v)


def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert (len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    print(nSamples)

    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in range(nSamples):
        # imagePath = './recognition/'+''.join(imagePathList[i]).split()[0].replace('\n','').replace('\r\n','')
        # imagePath = '/home/narcissus/file/dataset/ocr/final_test/'+'line_' + str(i + 1) + '.jpg'
        # imagePath = '/home/narcissus/file/recent_job/ocr/syn_data/dataset/train/' + imagePathList[i][:-1] + '.jpg'
        imagePath = '/home/narcissus/file/dataset/ocr/train_res/' + str(i+1).zfill(7) + '.jpg'

        label = ''.join(labelList[i])
        label = bytes(label, encoding="utf8")
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
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
        # print(cnt)
    nSamples = cnt - 1
    cache[b'num-samples'] = bytes(str(nSamples), encoding="utf8")
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    outputPath = "./train"
    imgdata = open("/home/narcissus/file/dataset/ocr/train_label.txt",encoding='utf8')
    imagePathList = list(imgdata)
    labelList = []
    for line in imagePathList:
        # word = line.split()[1]
        word = line[:-1]

        labelList.append(word)
    createDataset(outputPath, imagePathList, labelList)
    # pass