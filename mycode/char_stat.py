
import os
import matplotlib.pyplot as plt
import numpy as np

txt_root='/home/new/File/DataSet/crop_root/char_text'

dct={}

for root, dirs, files in os.walk(txt_root,topdown=False):
    for file in files:
        txt_path=txt_root+'/'+file
        print(txt_path)
        mytxt=open(txt_path,'r',encoding='utf8').readlines()
        print(mytxt.__len__())
        for txt in mytxt:
            for char in txt:
                if char != '\n':

                    if char not in dct:
                        dct[char] = 1
                    else:
                        dct[char] += 1

print(dct.items())

a=sorted(dct.items(),key=lambda item:item[1])
b=sorted(dct.values())


'''''''#analysise how many words appear with number i'''
#
# bins=np.bincount(np.array(b))
#
# print(bins)
#

'''get the characters that the occurrance is more than 2'''
keys=[]
for item in a:
    if item[1]>=2:
        keys.append(item[0])
print(len(keys))
''''# analysise the number of appearance of each character'''
# num_char=[]
# for item in a:
#     if item[1] in num_char:
#         num_char[item[1]]+=1
#     else:
#         num_char[item[1]] = 1
#
#
#     bins = np.arange(-100, 100, 5)  # fixed bin size
#
#     plt.xlim([min(data) - 5, max(data) + 5])
#
#     plt.hist(data, bins=bins, alpha=0.5)
#     plt.title('Random Gaussian data (fixed bin size)')
#     plt.xlabel('variable X (bin size = 5)')
#     plt.ylabel('count')

    # plt.show()
# f=open('/home/new/File/DataSet/crop_root/char_text/icdar2017rctw.txt','r',encoding='utf8')
# o=open('/home/new/File/DataSet/crop_root/char_text/icdar.txt','w+',encoding='utf8')
# lines=f.readlines()
# for line in lines:
#     o.write(line[1:-2]+'\n')

