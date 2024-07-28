from collections import defaultdict
import math


# 代码说明：
# train_bigram函数：
# 输入：词表。
# 输出：unigram_count和bigram_count，分别是单个词和相邻词对的计数。

# calculate_bigram_probability函数：
# 输入：bigram_count和unigram_count，以及两个词word1和word2。
# 输出：计算Bigram条件概率，使用Add-Delta平滑处理并取对数。

# segment_with_bigram函数：
# 输入：待分词的句子和词表。
# 输出：根据Bigram模型计算出的最优分词序列。


def train_bigram(word_list):
    unigram_count = defaultdict(int)
    bigram_count = defaultdict(int)

    for word in word_list:
        unigram_count[word] += 1

    for i in range(len(word_list) - 1):
        bigram_count[(word_list[i], word_list[i + 1])] += 1

    return unigram_count, bigram_count


def calculate_bigram_probability(bigram_count, unigram_count, word1, word2, delta=1):
    V = len(unigram_count)  # 词汇表大小
    numerator = bigram_count[(word1, word2)] + delta
    denominator = unigram_count[word1] + V * delta
    return math.log(numerator / denominator)


def segment_with_bigram(sentence, word_list):
    unigram_count, bigram_count = train_bigram(word_list)
    n = len(sentence)

    # dp[i]表示从0到i最优的分词概率和路径
    dp = [(-float('inf'), '')] * (n + 1)
    dp[0] = (0, '')

    for i in range(1, n + 1):
        for j in range(max(0, i - 10), i):
            word = sentence[j:i]
            if word in word_list:
                if j == 0:
                    prob = calculate_bigram_probability(bigram_count, unigram_count, '<s>', word)
                else:
                    prev_word = sentence[j - 1:i - 1]
                    prob = calculate_bigram_probability(bigram_count, unigram_count, prev_word, word)
                new_prob = dp[j][0] + prob
                if new_prob > dp[i][0]:
                    dp[i] = (new_prob, word)

    # 回溯找出最优的分词结果
    segmented_words = []
    i = n
    while i > 0:
        word = dp[i][1]
        segmented_words.append(word)
        i -= len(word)

    segmented_words.reverse()
    return segmented_words


# 给定词表和句子
word_list = ["他", "的", "瞒", "确", "实", "在", "理", "由", "不", "相", "其", "理由", "确实", "实在", "的确", "在理"]
sentence = "他说的确实在理"

# 使用Bigram模型进行分词
segmented_sentence = segment_with_bigram(sentence, word_list)
print(segmented_sentence)
