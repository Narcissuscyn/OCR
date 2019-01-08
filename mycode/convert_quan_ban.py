
def strQ2B( ustring):
    ss = []
    for s in ustring:
        rstring = ""
        for uchar in s:
            inside_code = ord(uchar)
            if inside_code == 12288:  # 全角空格直接转换
                inside_code = 32
            elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
                inside_code -= 65248
            rstring += chr(inside_code)
        ss.append(rstring)
    return ss

def is_quan_other(uchar):
    if (ord(uchar) >= 65281 and ord(uchar) <= 65295)or(ord(uchar) >= 65306 and ord(uchar) <= 65312)or(ord(uchar) >= 65339 and ord(uchar) <= 65344)or(ord(uchar) >= 65371 and ord(uchar) <= 65374):
        return True
    else:
        return False

def is_quan_number(uchar):
    if ord(uchar) >= 65296 and ord(uchar) <= 65305:
        return True
    else:
        return False


def is_quan_alphabet(uchar):
    if (ord(uchar) >= 65313 and ord(uchar) <= 65338) or (ord(uchar) >= 65345 and ord(uchar) <= 65370):
        return True
    else:
        return False

def is_ban_other(uchar):
    if (ord(uchar) >= 33 and ord(uchar) <= 47)or(ord(uchar) >= 58 and ord(uchar) <= 64)or(ord(uchar) >= 91 and ord(uchar) <= 96)or(ord(uchar) >= 123 and ord(uchar) <= 126):
        return True
    else:
        return False

def is_ban_number(uchar):
    if ord(uchar) >= 48 and ord(uchar) <= 57:
        return True
    else:
        return False


def is_ban_alphabet(uchar):
    if (ord(uchar) >= 65 and ord(uchar) <= 90) or (ord(uchar) >= 97 and ord(uchar) <= 122):
        return True
    else:
        return False


line=[]
with open('./mycode/n.txt','r',encoding='utf8') as f1:
    line=f1.readlines()
f=open('./mycode/n1.txt','w+',encoding='utf8')
for char in line[0]:

    if is_quan_alphabet(char) or is_quan_number(char):
        print(char)
    else:
        f.write(char)
f.close()

