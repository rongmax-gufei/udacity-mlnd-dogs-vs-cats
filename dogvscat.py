#!/usr/bin/env python
# coding: utf-8

# # 机器学习纳米学位毕业项目

# ## 项目背景
# 项目来源于 kaggle 在 2013 年组织的一场比赛，它使用25000张（约543M）猫狗图片作为训练集，12500张(约271M)图片作为测试集，数据都是分辨率400x400左右的小图片，目标是识别测试集中的图片是猫还是狗。赛题网址：https://www.kaggle.com/c/dogs-vs-cats。
# 
# 目前 Leaderboard 上展示了 1314 支队伍的成绩，排名第一的 score 是 0.03302，Top2% 的成绩是 0.04357。本项目的最低要求是 kaggle Public Leaderboard 前 10%，即 0.06149。

# ## 问题陈述
# 深度学习中最突出的问题之一是图像分类。图像分类的目的是根据潜在的类别对特定的图像进行分类。图像分类的一个经典示例是在一组图像中识别猫和狗。
# 
# 本文将介绍如何在图像分类问题中实施迁移学习解决方案。主要是使用"监督学习”实现一个图像分类器，来识别一张图片是猫还是狗。
# 
# 对于图像识别，在数据量足量大的情况下，一般使用深度学习中的卷积神经网络（Convolutional Neural Networks, CNN），而本文将从迁移学习的角度，看看如何应用现有的深度学习模型（ResNet50、InceptionV3 和 Xception），从图片中提取特征，供分类器使用。使用此方法，即无需大量学习和训练模型的时间成本，又能解决图片识别相关的大多数问题。
# 
# 本项目需要对测试样本进行分类，然后基于 CNN 的模型进行训练，并将结果上传至 kaggle 进行评分。

# ## 数据集
# 此数据集可以从 kaggle 上下载。[Dogs vs. Cats Redux: Kernels Edition](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data)
# 下载 kaggle 猫狗数据集解压后分为 3 个文件 train.zip、 test.zip 和 sample_submission.csv。
# 
# train 训练集包含了 25000 张猫狗的图片，猫狗各一半，每张图片包含图片本身和图片名。命名规则根据 “type.num.jpg” 方式命名。
# 
# test 测试集包含了 12500 张猫狗的图片，没有标定是猫还是狗，每张图片命名规则根据 “num.jpg”，需要注意的是测试集编号从 1 开始，而训练集的编号从 0 开始。
# 
# sample_submission.csv 需要将最终测试集的测试结果写入.csv 文件中，上传至 kaggle 进行打分。

# ## 导入库

# In[48]:


import os
import shutil
import h5py
import numpy as np
import math
from math import ceil
import pandas as pd
from sklearn.utils import shuffle
from PIL import Image
from IPython.display import SVG
from collections import Counter
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from keras.utils.vis_utils import model_to_dot, plot_model
from keras import Input, Model
from keras.applications import ResNet50, InceptionV3, inception_v3, Xception, xception
from keras.layers import Lambda, GlobalAveragePooling2D
from keras.layers import Dropout
from keras.layers import Dense
from keras.preprocessing.image import *
from keras.models import *


# In[2]:


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# ## 数据探索
# 在做计算机视觉的任务时，第一步很重要的事情是看看要做的是什么样的数据集，是非常干净的？还是存在各种遮挡的？猫和狗是大是小？图片清晰度一般怎么样？会不会数据集中有标注错误的数据，比例是多少？散点分布情况？

# In[11]:


imagelist = os.listdir('train/')

total_width = 0
total_height = 0
avg_width = 0
avg_height = 0
count = 0
min_width = 10000
min_height = 10000
max_width = 0
max_height = 0
min_product = 100000001
min_product_width = 0
min_product_height = 0

for x in imagelist:
    image_name = 'train/' + x
    image = Image.open(image_name)
    width = image.size[0]
    height = image.size[1]
    total_width += width
    total_height += height
    if min_width > width:
        min_width = width
    if min_height > height:
        min_height = height
    if max_width < width:
        max_width = width
    if max_height < height:
        max_height = height
    if min_product > width * height:
        min_product = width * height
        min_product_width = width
        min_product_height = height
    count += 1
print(count)
avg_width = total_width / count
avg_height = total_height / count
print("avg_width={}\navg_height={}\nThe total number of image is {}".format(avg_width, avg_height, count))
print("The min width is {}\nThe max width is {}".format(min_width, max_width))
print("The min height is {}\nThe max height is {}".format(min_height, max_height))
print("The min image size is {}*{}".format(min_product_width, min_product_height))    


# train 训练集包含了 25000 张猫狗的图片，平均宽=404px，平均高=360px，最小的宽=42px，最大宽=1050px，最小高=32px，最大高=768px；可以发现很多分辨率低图片，我们需要清理掉这些的图片。

# #### 绘制训练集中所有图片的大小散点图分布情况：

# In[14]:


train_image_list = os.listdir('train/')
height_array = []
width_array = []
for name in train_image_list[1:]:
    image = load_img('train/' + name)
    x = img_to_array(image)
    height_array.append(x.shape[0])
    width_array.append(x.shape[1])

x = np.array(width_array)
y = np.array(height_array)
area = np.pi * (15 * 0.05) ** 2

plt.scatter(x, y, s = area, alpha = 0.5,  marker='x')
plt.show()


# #### 找出训练集中的所有长 or 宽 < 70px 的图片，分析其清晰度

# In[17]:


train_image_list = os.listdir('train/')
bad_pictures = []
for name in train_image_list[1:]:
    image = load_img('train/' + name)
    x = img_to_array(image)
    if x.shape[0] < 70 or x.shape[1] < 70:
        bad_pictures.append(name) 
print(bad_pictures)


# In[18]:


for name in bad_pictures[:]:
    image = load_img('train/' + name)
    x = img_to_array(image)
    plt.title(name)
    plt.imshow(image)
    plt.show()


# #### 去除离群点

# In[19]:


plt.style.use('seaborn-white')

train_image_list = os.listdir('train/')
ratio_list = []

for name in train_image_list[1:]:
    image = Image.open('train/' + name)
    x = image.histogram(mask = None)
    count = Counter(x)
    ratio = float(len(count)) / len(x)
    ratio_list.append(ratio)

# np.percentile获取百分位数
q99, q01 = np.percentile(a=ratio_list, q=[99, 1])
print(q99, q01)


# In[20]:


# 将异常图片输出
plt.style.use('seaborn-white')

outlier_images = []
train_image_list = os.listdir('train/')
for name in train_image_list[:]:
    image = Image.open('train/' + name)
    x = image.histogram(mask = None)
    count = Counter(x)
    ratio = float(len(count)) / len(x)
    if ratio< q01:
        outlier_images.append(name)
        img = load_img('train/' + name)     
        x = img_to_array(img)
        plt.title(name)
        plt.imshow(img)
        plt.show()
        
print(outlier_images)


# In[21]:


# 定义创建目标路径方法
def mkdir(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.mkdir(dirname)


# In[22]:


# 创建异常图片集文件夹
mkdir('outlier')


# In[23]:


# 挑出所有低分率的图片
outlier_list = ['cat.10107.jpg', 'cat.10277.jpg', 'cat.10392.jpg', 'cat.10893.jpg', 'cat.11091.jpg', 'cat.11184.jpg', 
                'cat.2433.jpg', 'cat.2939.jpg', 'cat.3216.jpg', 'cat.4821.jpg', 'cat.4833.jpg', 'cat.5534.jpg', 
                'cat.6402.jpg', 'cat.6699.jpg', 'cat.7703.jpg', 'cat.7968.jpg', 'cat.8138.jpg', 'cat.8456.jpg', 
               'cat.8470.jpg', 'cat.8504.jpg', 'cat.9171.jpg', 'dog.10190.jpg', 'dog.10654.jpg', 'dog.10747.jpg',
               'dog.11248.jpg', 'dog.11465.jpg', 'dog.11686.jpg', 'dog.1174.jpg', 'dog.12331.jpg', 'dog.1308.jpg',
               'dog.1381.jpg', 'dog.1895.jpg', 'dog.3074.jpg', 'dog.4367.jpg', 'dog.4507.jpg', 'dog.5604.jpg',
               'dog.630.jpg', 'dog.6685.jpg', 'dog.7772.jpg', 'dog.8450.jpg', 'dog.8736.jpg', 'dog.9188.jpg', 
               'dog.9246.jpg', 'dog.9517.jpg', 'dog.9705.jpg']


# In[ ]:


# 将这些异常图片从训练集中删除
plt.figure(figsize=(12, 20))
outlier_image_size = len(outlier_list)
for i in range(0, outlier_image_size):
    plt.subplot(ceil(outlier_image_size / 6), 6, i+1)
    img = load_img('train/'+ outlier_list[i])
    x = img_to_array(img)
    plt.title(outlier_list[i])
    plt.axis('off')
    plt.tight_layout()
    plt.imshow(img, interpolation="nearest")
    shutil.move('train/' + outlier_list[i], 'outlier/' + outlier_list[i])


# ## 数据预处理
# 由于我们的数据集的文件名是以 type.num.jpg 这样的方式命名，如 cat.0.jpg，但是使用 Keras 的 ImageDataGenerator 需要将不同种类的图片分在不同的文件夹中，因此我们需要对数据集进行预处理。这里我们采取的思路是借鉴[杨培文](https://www.sohu.com/a/130598226_473283)的创建符号链接(symbol link)，优点是不用复制一遍图片，占用不必要的空间。

# In[24]:


train_list = os.listdir('train')
# 找出所有的猫
train_cat = filter(lambda x:x[:3] == 'cat', train_list)
# 找出所有的狗
train_dog = filter(lambda x:x[:3] == 'dog', train_list)


# In[25]:


# 定义 训练集的 symlink 文件夹名字
train_symlink_path = 'train-symlink'

# 创建 训练集的 symlink 文件夹
mkdir(train_symlink_path)

# 分类猫图片到 train-symlink/cat 下
os.mkdir(train_symlink_path + '/cat')
for filename in train_cat:
    os.symlink('../../train/' + filename, train_symlink_path + '/cat/' + filename)

# 分类狗图片到  train-symlink/dog 下
os.mkdir(train_symlink_path + '/dog')
for filename in train_dog:
    os.symlink('../../train/' + filename, train_symlink_path + '/dog/' + filename)


# In[26]:


# 定义 测试集的 symlink 文件夹名字 
test_symlink_path = 'test-symlink'

# 创建 测试集的 symlink 文件夹
mkdir(test_symlink_path)

# 分类测试集图片到  test-symlink/test 下
os.symlink('../test/', test_symlink_path + '/test')


# ## 导出特征向量
# 
# 为了提高模型的表现，本项目决定使用预训练网络，最终选择了ResNet50, Xception, Inception V3 这三个模型，由于在笔记本上跑的，三个模型导出的时间耗了一天时间，时常有中途下载失败的情况。 这三个模型都是在 ImageNet 上面预训练过的，由此我们实际的预测训练会带来极高的初始精度。我们可以将多个不同的网络输出的特征向量先保存下来，后续即使是在普通笔记本上也能轻松训练。

# In[28]:


"""
定义通用函数
入参：模型、输入图片的大小、预处理函数
"""
def write_gap(MODEL, image_size, lambda_func=None):
    input_tensor = Input(shape=(image_size[0], image_size[1], 3))
    x = input_tensor
    if lambda_func:
        x = Lambda(lambda_func)(x)
         
    base_model = MODEL( include_top=False, weights='imagenet', input_tensor=x)
    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))

    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    # 用于测试的增强配置
    test_datagen = ImageDataGenerator(rescale=1./255)

    # 定义了两个 generator
    # 读取在train中找到的图片，并无限期地生成大量的增强图像数据
    train_generator = train_datagen.flow_from_directory(
                                              "train-symlink", 
                                              target_size = image_size, 
                                              shuffle=False, 
                                              batch_size=32,
                                              class_mode='binary')
    
    test_generator = test_datagen.flow_from_directory(
                                             "train-symlink", 
                                             target_size = image_size, 
                                             shuffle=False, 
                                             batch_size=32, 
                                             class_mode='binary')

    # 导出特征向量
    train = model.predict_generator(train_generator, 
                                    math.ceil(train_generator.samples*1.0/train_generator.batch_size), 
                                    verbose=1)
    test = model.predict_generator(test_generator, 
                                   math.ceil(test_generator.samples*1.0/test_generator.batch_size), 
                                   verbose=1)
    
    # 此项目选择了：ResNet50, Xception, InceptionV3 这三个模型
    if MODEL == ResNet50:
        model_name = "gap_ResNet50.h5"
    elif MODEL == Xception:
        model_name = "gap_Xception.h5"
    elif MODEL == InceptionV3:
        model_name = "gap_InceptionV3.h5"
        
    with h5py.File(model_name) as h:
        h.create_dataset("train", data=train)
        h.create_dataset("test", data=test)
        h.create_dataset("label", data=train_generator.classes)


# In[ ]:


# write_gap(ResNet50, (224, 224))


# In[ ]:


# write_gap(Xception, (299, 299), xception.preprocess_input)


# In[ ]:


# write_gap(InceptionV3, (299, 299), inception_v3.preprocess_input)


# ## 载入特征向量
# 
# 现在我们获得了3个特征向量文件：
# 
# - gap_ResNet50.h5
# - gap_Xception.h5
# - gap_InceptionV3.h5

# 把三个模型合并在一起，每个图片就有2048*3个权重值

# In[29]:


np.random.seed(2019)

X_train = []
X_test = []

for filename in ["gap_ResNet50.h5", "gap_Xception.h5", "gap_InceptionV3.h5"]:
    with h5py.File(filename, 'r') as h:
        X_train.append(np.array(h['train']))
        X_test.append(np.array(h['test']))
        y_train = np.array(h['label'])

X_train = np.concatenate(X_train, axis=1)
X_test = np.concatenate(X_test, axis=1)

X_train, y_train = shuffle(X_train, y_train)


# In[59]:


## 单模型测试
# np.random.seed(2019)

# X_train = []
# X_test = []
# # gap_ResNet50.h5
# # gap_Inception.h5
# for filename in ["gap_InceptionV3.h5"]:
#     with h5py.File(filename, 'r') as h:
#         X_train.append(np.array(h['train']))
#         X_test.append(np.array(h['test']))
#         y_train = np.array(h['label'])

# X_train = np.concatenate(X_train, axis=1)
# X_test = np.concatenate(X_test, axis=1)

# X_train, y_train = shuffle(X_train, y_train)


# 我们基于这些权重值建立一个全连接
# 
# 训练深度神经网络的时候，总是会遇到两大缺点：
# 
# （1）容易过拟合
# 
# （2）费时
# 
# Dropout可以比较有效的缓解过拟合的发生，在一定程度上达到正则化的效果。

# In[60]:


inputs = Input(X_train.shape[1:])
x = inputs
x = Dropout(0.25)(inputs)
x = Dense(1, activation='sigmoid')(x)
model = Model(inputs, x)

model.compile(optimizer='adadelta', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])


# 模型可视化

# In[ ]:


# plot_model(model, to_file='model.png')
# SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))


# ## 模型训练

# In[32]:


model.summary()


# In[61]:


history = model.fit(X_train, y_train, batch_size=128, epochs=8, validation_split=0.2, verbose=1)


# 8 次 epochs，训练完不到1分钟，第一次达到了97%，后面7次均达到了99%

# 保存模型

# In[34]:


model.save('model.h5')


# #### 训练过中的 accuracy 表现

# In[43]:


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Training and validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# #### 训练过中的 loss 表现

# In[44]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training and validation loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# ## 预测测试集

# In[45]:


y_pred = model.predict(X_test, verbose=1)
y_pred = y_pred.clip(min=0.002, max=0.998)


# 测试集结果保存至sample_submission.csv，以供提交至Kaggle上查看成绩

# In[50]:


df = pd.read_csv("sample_submission.csv")

path = 'test-symlink/'

# 原图
data_generator = image.ImageDataGenerator()
generator_data = data_generator.flow_from_directory(
                                                                            path, 
                                                                            batch_size=32, 
                                                                            shuffle=False, 
                                                                            class_mode='binary', 
                                                                            target_size=(224, 224))

for i, x in enumerate(generator_data.filenames):
    index = int(x[x.rfind('/')+1:x.rfind('.')]) - 1
    df.set_value(index, 'label', y_pred[i])

df.to_csv('pred_result.csv', index=None)

df.head(10)


# ## Kaggle 评分

# ![](score.png)
# Kaggle 得分：0.04106，leaderboard 18 / 1314。

# ## 参考文献

# [1] [基于Theano的深度学习(Deep Learning)框架Keras学习随笔-05-模型](https://blog.csdn.net/niuwei22007/article/details/49207187)
# 
# [2][keras-model-visualization](https://keras.io/visualization/)
# 
# [3][手把手教你如何在Kaggle猫狗大战冲到Top2%](https://zhuanlan.zhihu.com/p/25978105?utm_source=weibo) 
# 
# [4][image_classification_using_very_little_data](https://keras-cn-docs.readthedocs.io/zh_CN/latest/blog/image_classification_using_very_little_data)
# 
# [5][pooling_layer](https://keras-cn.readthedocs.io/en/latest/layers/pooling_layer)
# 
# [6][利用resnet 做kaggle猫狗大战图像识别，秒上98准确率](https://blog.csdn.net/shizhengxin123/article/details/72473245)
# 
# [7][plt.Scatter函数解析](https://blog.csdn.net/tefuirnever/article/details/88944438)
# 
# [8][numpy : percentile使用](https://blog.csdn.net/u011630575/article/details/79451357)
# 
# [9] Xie S, Girshick R, Dollár P, et al. Aggregated residual transformations for deep neural networks[J]. arXiv preprint arXiv:1611.05431, 2016.
# 
# [10] Donahue J, Jia Y, Vinyals O, et al. DeCAF: A Deep Convolutional Activation Feature for Generic Visual Recognition[C]//Icml. 2014, 32: 647-655.
# 
# [11] Ruder S. An overview of gradient descent optimization algorithms[J]. arXiv preprint arXiv:1609.04747, 2016.
# 
# [12] He K, Zhang X, Ren S, et al. Deep residual learning for image recognition[C]. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016: 770-778.
# 
# [13] Chollet, François, [Keras](https://github.com/fchollet/keras).

# In[ ]:




