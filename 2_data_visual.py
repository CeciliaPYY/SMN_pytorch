"""
1. 对清洗并分词之后的数据，分别从句子的长度，对话的轮数等方面，进行预先的可视化，如基本统计量（最大值、最小值、平均值）；
2. 按照词频，生成词云；
"""

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import imageio
import numpy as np


def get_basic_statistics(length):
    max_val = np.max(length)
    min_val = np.min(length)
    mean_val = np.mean(length)
    std_val = np.std(length)
    
    print("Max Length is : {}".format(max_val))
    print("Min Length is : {}".format(min_val))
    print("Mean Length is : {}".format(mean_val))
    print("Std Length is : {}".format(std_val))

    def plot_histogram(data, title, x_label, y_label, **kwargs):
        plt.style.use('seaborn')
        n, bins, patches = plt.hist(data, 500, facecolor='green', alpha=0.75, **kwargs)

        plt.xlabel(x_label)
        plt.ylabel(y_label)
    #     plt.ylim((0, 100000))
        plt.title(title)
        plt.grid(True)


        plt.show()


if __name__ == "__main__":
    with open('../data/train_.txt', "r", encoding='utf-8') as f:
        train_data = f.readlines()
    
    # 词云生成
    bg_pic = imageio.imread("../img/last_qie.jpeg")

    wordcloud = WordCloud(mask = bg_pic, background_color="white", scale=2.0 ).generate('\t'.join(train_data))
    image_colors = ImageColorGenerator(bg_pic)

    plt.figure(figsize=(16,8))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()
    
    # 基本统计量
    count = 0
    utterance_length = []
    response_length = []
    sent_length = []

    for t_data in train_data:
        label = t_data.split('\t')[0]
        count += 1
        if label == '1':
            utterance = t_data.split('\t')[1:-1]
            response = t_data.split('\t')[-1]

            utterance_length.append(len(utterance))
            sent_length.extend([len(sent.split(' ')) for sent in utterance])
            response_length.append(len(response.split(' ')))
        else:
            continue
            
    # 1. 计算一下 utterances 的最值，即 最大对话轮数 、 最小对话轮数 、平均对话轮数等
    get_basic_statistics(utterance_length)
    
    # 2. 计算一下 responses 的最值，
    get_basic_statistics(response_length)
    
    # 3. 计算一下 sentence 的最值，
    # sent_length.extend(response_length)
    get_basic_statistics(sent_length)
    
    

    # 1. 对话长度的分布
#     new_utterance_length = list(filter(lambda x: x != 19, new_utterance_length)) + [19]*6700
    plot_histogram(new_utterance_length, "Train Utterances", "Number of utterances", "Count")
    
    # 2. 对话句子长度的分布
    plot_histogram(sent_length, "Train Utterances Length", "Length of utterances", "Count")
    
    # 3. 回复句子长度的分布
    plot_histogram(response_length, "Train Responses Length", "Length of responses", "Count")
