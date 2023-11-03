import random
import numpy as np

##定义一个函数 get_all_samples，它接受一个参数 conjunction。该函数的功能是根据传入的 conjunction 数组生成样本集。
##数据集1有12269个负样本，605个正样本。共12874个样本。
##数据集2中有15381个负样本，1529个正样本，共16910个样本。
def get_all_samples(conjunction):
    ##创建空列表，用于存储正负样本。
    pos = []
    neg = []
   ##index：表示当前迭代的行索引。
    ##range(conjunction.shape[0])：迭代范围是 conjunction 矩阵的行数，conjunction.shape[0] 返回矩阵的行数。
    ##ange(conjunction.shape[1])：迭代范围是 conjunction 矩阵的列数，conjunction.shape[1] 返回矩阵的列数
    ##conjunction：一个矩阵或二维数组，包含要遍历的元素
    ##conjunction[index, col]：表示矩阵 conjunction 中第 index 行、第 col 列的元素值
    ##pos：一个列表，用于存储值为 1 的元素的索引和值。
    ##neg：一个列表，用于存储值为 0 的元素的索引和值
    for index in range(conjunction.shape[0]):
        for col in range(conjunction.shape[1]):
            if conjunction[index, col] == 1:
                pos.append([index, col, 1])
            else:
                neg.append([index, col, 0])
    ##获取正样本的数量 pos_len，然后从负样本中随机选择与正样本数量相同的样本，存储在new_neg 中
    pos_len = len(pos)
    new_neg = random.sample(neg, pos_len)
    ##将正样本和随机选择的负样本合并为一个样本集 samples，然后对样本集进行随机重排
    samples = pos + new_neg
    samples = random.sample(samples, len(samples))
    ##将样本集转换为 NumPy 数组
    samples = np.array(samples)
    return samples

##定义一个函数 generate_f，它接受两个参数 samples 和 features。根据传入的样本集和特征生成特征向量和标签。
##一行为一个样本。特征（属性）：反映事件或对象在某方面的表现或性质的事
def generate_f(samples: np.ndarray, features: list):
    # 创建一个长度为 features 列表长度的零向量
    vec_lens = np.zeros(len(features), dtype=int)

    for index in range(len(features)):
        # 计算每个特征的列数，存储在 vec_lens 中
        vec_lens[index] = features[index].shape[1]
        # 计算特征向量总长度
    vec_len = np.sum(vec_lens)
    # 样本数量
    num = samples.shape[0]
    # 创建一个零矩阵，用于存储特征
    feature = np.zeros([num, vec_len])
    # 创建一个零向量，用于存储标签
    label = np.zeros([num])

    for i in range(num):
        tail = 0
        for index in range(len(features)):
            head = tail
            tail += vec_lens[index]
            # 将每个特征的数据按顺序拼接到 feature 矩阵的每一行
            feature[i, head: tail] = features[index][samples[i, 1 if index % 2 else 0], :]
            # 将每个样本的标签存储到 label 向量中
        label[i] = samples[i, 2]
        # 返回生成的特征矩阵和标签向量
    return feature, label