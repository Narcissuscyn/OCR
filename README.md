partly forked from YoungMiao/crnn https://github.com/YoungMiao/crnn

## 说明

**网络图像的文本识别**

ICPR （International Conference on Pattern Recognition）2018 是图像识别与机器视觉领域的国际学术顶会。在互联网世界中，图片是传递信息的重要媒介。特别是电子商务，社交，搜索等领域，每天都有数以亿兆级别的图像在流动传播。图片中的文字识别（OCR）在商业领域有重要的应用价值，同时也是学术界的研究热点。然而，研究领域尚没有基于网络图片的、以中文为主的OCR数据集。

阿里巴巴“图像和美”团队联合华南理工大学共同举办ICPR MTWI 2018 挑战赛，本竞赛将公开基于网络图片的中英数据集，该数据集数据量充分，涵盖数十种字体，几个到几百像素字号，多种版式，较多干扰背景。期待学术界可以在本数据集上作深入的研究，工业界可以藉此发展在图片管控，搜索，信息录入等AI领域的工作。

**解决方案**

采用crnn模型进行调优,这个库改自 YoungMiao/crnn,用来处理中文字符识别,环境配置与之一致;修改的地方有如下方面:

- 对网络的改进 

	1. 将VGG 修改为resnet的版本:./crnn_resnet.py
	
	2.调整网络模型大小和参数
	
- 对于wild中文数据的处理(mycode文件夹下的代码)

	1.将文字区域抠出,需要处理坐标点标注方向不同的问题,对于标注方向不同,采用法向量的方式解决;抠出文字时,需要做投影变换,因为标注框不一定和坐标轴平行.
	2.判断横排和竖排文字,并分别预处理横排文字和竖排文字
		- 对于横排文字,可以直接输入到网络中进行预测;
		- 对与竖排文字,需要进行文字分割,接着预测每个分割的部分,最后将分割的部分进行组合,形成完整的预测
	3. 处理全角字符和半角字符
	4.统计字符,将字符次数出现小于2的字符去除掉,以减小分类数目
		
		

- 添加编辑距离的计算 : ./val_edit_dist.py

**以下内容来自 YoungMiao/crnn**

## crnn实现细节(pytorch)
### 1.环境搭建
#### 1.1 基础环境
* Ubuntu14.04 + CUDA
* opencv2.4 + pytorch + lmdb +wrap_ctc

安装lmdb `apt-get install lmdb`
#### 1.2 安装pytorch
pip,linux,cuda8.0,python2.7:pip install http://download.pytorch.org/whl/cu80/torch-0.1.12.post2-cp27-none-linux_x86_64.whl
参考：http://pytorch.org/
#### 1.3 安装wrap_ctc
    git clone https://github.com/baidu-research/warp-ctc.git`
    cd warp-ctc
    mkdir build; cd build
    cmake ..
    make

GPU版在环境变量中添加
    export CUDA_HOME="/usr/local/cuda"

    cd pytorch_binding
    python setup.py install
    
参考：https://github.com/SeanNaren/warp-ctc/tree/pytorch_bindings/pytorch_binding
#### 1.4 注意问题
1. 缺少cffi库文件 使用`pip install cffi`安装
2. 安装pytorch_binding前,确认设置CUDA_HOME,虽然编译安装不会报错,但是在调用gpu时，会出现wrap_ctc没有gpu属性的错误

   # 设置模型参数 图片高度imgH=32, nc, 分类数目nclass=len(alphabet)+1 一个预留位, LSTM设置隐藏层数nh=128, 使用GPU个数ngpu=1
    model = crnn.CRNN(32, 1, 22, 128, 1).cuda()

替换模型时，注意模型分类的类别数目
## crnn 训练(以21类中英文为例)
1. 数据预处理

运行`/contrib/crnn/tool/tolmdb.py`

    # 生成的lmdb输出路径
    outputPath = "./train_lmdb"
    # 图片及对应的label
    imgdata = open("./train.txt")

2. 训练模型

运行`/contrib/crnn/crnn_main.py`

    python crnn_main.py [--param val]
    --trainroot        训练集路径
    --valroot          验证集路径
    --workers          CPU工作核数, default=2
    --batchSize        设置batchSize大小, default=64
    --imgH             图片高度, default=32
    --nh               LSTM隐藏层数, default=256
    --niter            训练回合数, default=25
    --lr               学习率, default=0.01
    --beta1             
    --cuda             使用GPU, action='store_true'
    --ngpu             使用GPU的个数, default=1
    --crnn             选择预训练模型
    --alphabet         设置分类
    --Diters            
    --experiment        模型保存目录
    --displayInterval   设置多少次迭代显示一次, default=500
    --n_test_disp        每次验证显示的个数, default=10
    --valInterval        设置多少次迭代验证一次, default=500
    --saveInterval       设置多少次迭代保存一次模型, default=500
    --adam               使用adma优化器, action='store_true'
    --adadelta           使用adadelta优化器, action='store_true'
    --keep_ratio         设置图片保持横纵比缩放, action='store_true'
    --random_sample      是否使用随机采样器对数据集进行采样, action='store_true'
    
示例:python /contrib/crnn/crnn_main.py --tainroot [训练集路径] --valroot [验证集路径] --nh 128 --cuda --crnn [预训练模型路径] 

修改`/contrib/crnn/keys.py`中`alphabet = 'ACIMRey万下依口哺摄次状璐癌草血运重'`增加或者减少类别

3. 注意事项

训练和预测采用的类别数和LSTM隐藏层数需保持一致



