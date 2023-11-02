import fasttext
import datetime
import numpy as np
import re
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='train_seg.txt', required=False, help='input file')
parser.add_argument('--lr', type=float, default=0.25, required=False, help='learning rate')
parser.add_argument('--dim', type=int, default=200, required=False, help='dimension')
parser.add_argument('--ngrams', type=int, default=3, required=False, help='ngrams')
parser.add_argument('--epoch', type=int, default=60, required=False, help='epoch')
parser.add_argument('--loss', type=str, default='ova', required=False, help='loss function')
parser.add_argument('--threshold', type=float, default=0.5, required=False, help='threshold')
parser.add_argument('--pretrainedVectors', type=str, default=None, required=False, help='pretrained vectors: crawl-300d-2M-subword.vec, wiki-news-300d-1M-subword.vec')
args = parser.parse_args()

print(args)

"""
  训练一个监督模型, 返回一个模型对象

  @param input:           训练数据文件路径
  @param lr:              学习率
  @param dim:             向量维度
  @param ws:              cbow模型时使用
  @param epoch:           次数
  @param minCount:        词频阈值, 小于该值在初始化时会过滤掉
  @param minCountLabel:   类别阈值，类别小于该值初始化时会过滤掉
  @param minn:            构造subword时最小char个数
  @param maxn:            构造subword时最大char个数
  @param neg:             负采样
  @param wordNgrams:      n-gram个数
  @param loss:            损失函数类型, softmax, ns: 负采样, hs: 分层softmax
  @param bucket:          词扩充大小, [A, B]: A语料中包含的词向量, B不在语料中的词向量
  @param thread:          线程个数, 每个线程处理输入数据的一段, 0号线程负责loss输出
  @param lrUpdateRate:    学习率更新
  @param t:               负采样阈值
  @param label:           类别前缀
  @param verbose:         ??
  @param pretrainedVectors: 预训练的词向量文件路径, 如果word出现在文件夹中初始化不再随机
  @return model object
"""

if args.pretrainedVectors:
    model = fasttext.train_supervised(input='train_seg.txt', 
                                  dim=args.dim, 
                                  epoch=args.epoch,
                                  lr=args.lr,
                                  wordNgrams=args.ngrams, 
                                  loss=args.loss,
                                  pretrainedVectors=args.pretrainedVectors)
else:
    model = fasttext.train_supervised(input='train_seg.txt', 
                                  dim=args.dim, 
                                  epoch=args.epoch,
                                  lr=args.lr,
                                  wordNgrams=args.ngrams, 
                                  loss=args.loss)


current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
model_path = f'../../checkpoints/fasttext/toxic_{current_time}.model'
print('Save path:', model_path)
model.save_model(model_path)
# classifier = fasttext.load_model('../../checkpoints/fasttext/classifier.model')

# 训练集准确率
result = model.test('train_seg.txt')

print('P@1:', result[1])
print('R@1:', result[2])
print('Number of examples:', result[0])

# 获取标签列表
labels = model.get_labels()


y_true = []
y_pred = []

with open('train_seg.txt', 'r', encoding='utf-8') as file:
    for line in file:
        line = line.strip()
        pattern = "|".join(labels)
        true_labels = re.findall(pattern, line)
        line = re.sub(pattern, "", line)
        predictions = model.predict(line, k=7)

        predict_labels = [predictions[0][i] for i in range(len(predictions[0])) if predictions[1][i] > args.threshold]

        if not predict_labels:
            max_prob_index = np.argmax(predictions[1])
            predict_labels = [predictions[0][max_prob_index]]

        y_true.append(true_labels)
        y_pred.append(predict_labels)

# 将多标签列表转换为二进制数组形式
mlb = MultiLabelBinarizer()
y_true_binary = mlb.fit_transform(y_true)
y_pred_binary = mlb.transform(y_pred)

# 计算每个标签的精确率和召回率
precision = precision_score(y_true_binary, y_pred_binary, average=None)
recall = recall_score(y_true_binary, y_pred_binary, average=None)

# 打印每个标签的精确率和召回率
for label, p, r in zip(mlb.classes_, precision, recall):
    print(f"Label: {label}")
    print(f"Precision: {p}")
    print(f"Recall: {r}")
    print()


print("Valuate.")

result = model.test('test_seg.txt')

print('P@1:', result[1])
print('R@1:', result[2])
print('Number of examples:', result[0])

# 获取标签列表
labels = model.get_labels()


y_true = []
y_pred = []

with open('test_seg.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        pattern = "|".join(labels)
        true_labels = re.findall(pattern, line)
        line = re.sub(pattern, "", line)
        predictions = model.predict(line, k=7)

        predict_labels = [predictions[0][i] for i in range(len(predictions[0])) if predictions[1][i] > args.threshold]

        if not predict_labels:
            max_prob_index = np.argmax(predictions[1])
            predict_labels = [predictions[0][max_prob_index]]

        y_true.append(true_labels)
        y_pred.append(predict_labels)

# 将多标签列表转换为二进制数组形式
mlb = MultiLabelBinarizer()
y_true_binary = mlb.fit_transform(y_true)
y_pred_binary = mlb.transform(y_pred)

# 计算每个标签的精确率和召回率
precision = precision_score(y_true_binary, y_pred_binary, average=None)
recall = recall_score(y_true_binary, y_pred_binary, average=None)

# 打印每个标签的精确率和召回率
for label, p, r in zip(mlb.classes_, precision, recall):
    print(f"Label: {label}")
    print(f"Precision: {p}")
    print(f"Recall: {r}")
    print()

# 计算总的精确率和召回率
total_precision = precision_score(y_true_binary, y_pred_binary, average='micro')
total_recall = recall_score(y_true_binary, y_pred_binary, average='micro')

# 计算准确率和F1分数
accuracy = accuracy_score(y_true_binary, y_pred_binary)
f1 = f1_score(y_true_binary, y_pred_binary, average='micro')

# 打印总的精确率和召回率、准确率和F1分数
print("Total Precision:", total_precision)
print("Total Recall:", total_recall)
print("Accuracy:", accuracy)
print("F1 Score:", f1)
