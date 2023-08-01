import sys
import json
import numpy as np
from bert4keras.backend import keras, K, batch_gather
from bert4keras.layers import Loss
from bert4keras.layers import LayerNormalization, MultiHeadAttention, PositionEmbedding
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_exponential_moving_average
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open, to_array
from keras.optimizers import Nadam
from keras.layers import *
from keras import initializers, activations
from keras.models import Model, Sequential
from tqdm.notebook import tqdm
import os
import tensorflow as tf

os.environ['TF_KERAS'] = '1'
strategy = tf.distribute.MirroredStrategy()

maxlen = 256
batch_size = 8
learning_rate_=1e-5
epochs_=60

#config_path = '../model/chinese_L-12_H-768_A-12/bert_config.json'
#checkpoint_path = '../model/chinese_L-12_H-768_A-12/bert_model.ckpt'
#dict_path = '../model/chinese_L-12_H-768_A-12/vocab.txt'
config_path = '../model/FinBERT_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../model/FinBERT_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../model/FinBERT_L-12_H-768_A-12/vocab.txt'

def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)
            for line in l:
                D.append({
                    'text': line['text'],
                    'spo_list': [(spo[0], spo[1], spo[2])
                                 for spo in line['spo_list']]
                })
    return D


print('loading data...')

predicate2id, id2predicate = {}, {}

train_data = load_data('../datasets/finance/train_data.json')
valid_data = load_data('../datasets/finance/dev_data.json')
test_data = load_data('../datasets/finance/dev_data.json')

#train_data = load_data('../datasets/coop/train_coo_data.json')
#valid_data = load_data('../datasets/coop/dev_coo_data.json')
#test_data = load_data('../datasets/coop/dev_coo_data.json')

#with open('../datasets/coop/all_schemas',encoding="utf-8") as f:
with open('../datasets/finance/all_5_schemas',encoding="utf-8") as f:
    for l in f:
        l = json.loads(l)
        if l['predicate'] not in predicate2id:
            id2predicate[len(predicate2id)] = l['predicate']
            predicate2id[l['predicate']] = len(predicate2id)       
            

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        batch_subject_labels, batch_subject_ids, batch_object_labels = [], [], []

        for is_end, d in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(d['text'], maxlen=maxlen)
            # 整理三元组 {s: [(o, p)]}
            spoes = {}
            for s, p, o in d['spo_list']:
                s = tokenizer.encode(s)[0][1:-1]
                p = predicate2id[p]
                o = tokenizer.encode(o)[0][1:-1]
                s_idx = search(s, token_ids)
                o_idx = search(o, token_ids)
                if s_idx != -1 and o_idx != -1:
                    s = (s_idx, s_idx + len(s) - 1)
                    o = (o_idx, o_idx + len(o) - 1, p)
                    if s not in spoes:
                        spoes[s] = []
                    spoes[s].append(o)
            if spoes:
                # subject标签
                subject_labels = np.zeros((len(token_ids), 2))
                for s in spoes:
                    subject_labels[s[0], 0] = 1
                    subject_labels[s[1], 1] = 1
                # 随机选一个subject
                start, end = np.array(list(spoes.keys())).T
                rindex = np.random.choice(len(start))
                start = start[rindex]
                end = end[rindex]
                #                 end = np.random.choice(end)
                subject_ids = (start, end)
                # 对应的object标签
                object_labels = np.zeros((len(token_ids), len(predicate2id), 2))
                for o in spoes.get(subject_ids, []):
                    object_labels[o[0], o[2], 0] = 1
                    object_labels[o[1], o[2], 1] = 1
                # 构建batch
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_subject_labels.append(subject_labels)
                batch_subject_ids.append(subject_ids)
                batch_object_labels.append(object_labels)
                if len(batch_token_ids) == self.batch_size or is_end:
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_segment_ids = sequence_padding(batch_segment_ids)
                    batch_subject_labels = sequence_padding(
                        batch_subject_labels
                    )
                    batch_subject_ids = np.array(batch_subject_ids)
                    batch_object_labels = sequence_padding(batch_object_labels)
                    yield [
                              batch_token_ids, batch_segment_ids,
                              batch_subject_labels, batch_subject_ids,
                              batch_object_labels
                          ], None
                    batch_token_ids, batch_segment_ids = [], []
                    batch_subject_labels, batch_subject_ids, batch_object_labels = [], [], []


def extract_subject(inputs):
    """根据subject_ids从output中取出subject的向量表征
    """
    output, subject_ids = inputs
    start = batch_gather(output, subject_ids[:, :1])
    end = batch_gather(output, subject_ids[:, 1:])
    subject = K.concatenate([start, end], 2)
    return subject[:, 0]


# 补充输入
subject_labels = Input(shape=(None, 2), name='Subject-Labels')
subject_ids = Input(shape=(2,), name='Subject-Ids')
object_labels = Input(shape=(None, len(predicate2id), 2), name='Object-Labels')


class Self_Attention(Layer):

    def __init__(self, output_dim, kernal_initializer='uniform', **kwargs):
        self.output_dim = output_dim
        self.kernal_initializer = initializers.get(kernal_initializer)
        super(Self_Attention, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        # inputs.shape = (batch_size, time_steps, seq_len)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3, input_shape[2], self.output_dim),
                                      initializer=self.kernal_initializer,
                                      trainable=True)

        super(Self_Attention, self).build(input_shape)  # 一定要在最后调用它

    def call(self, x):
        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(x, self.kernel[2])

        QK = K.batch_dot(WQ, K.permute_dimensions(WK, [0, 2, 1]))

        QK = QK / (64 ** 0.5)

        QK = K.softmax(QK)

        V = K.batch_dot(QK, WV)

        return V

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)


print('loading bert...')
# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model='bert',
    return_keras_model=False,
)
output = LayerNormalization(conditional=False)(bert.model.output)
# 预测subject
output = Dense(
    units=2, activation='sigmoid', kernel_initializer=bert.initializer
)(output)
subject_preds = Lambda(lambda x: x ** 2)(output)

subject_model = Model(bert.model.inputs, subject_preds)

# 传入subject，预测object
# 通过Conditional Layer Normalization将subject融入到object的预测中
output = bert.model.layers[-2].get_output_at(-1)
subject = Lambda(extract_subject)([output, subject_ids])

# 位置编码
output = PositionEmbedding(
     input_dim=maxlen,
     output_dim=768,
     merge_mode='mul',
     embeddings_initializer=bert.initializer
)(output)

#output = Self_Attention(768)(output)
output = LayerNormalization(conditional=True)([output, subject])

#output = Self_Attention(896)(output)
print(output.shape)
output = Dense(
    units=len(predicate2id) * 2,
    activation='sigmoid',
    kernel_initializer=bert.initializer
)(output)
output = Lambda(lambda x: x ** 4)(output)
object_preds = Reshape((-1, len(predicate2id), 2))(output)

object_model = Model(bert.model.inputs + [subject_ids], object_preds)


class TotalLoss(Loss):
    """subject_loss与object_loss之和，都是二分类交叉熵
    """

    def compute_loss(self, inputs, mask=None):
        subject_labels, object_labels = inputs[:2]
        subject_preds, object_preds, _ = inputs[2:]
        if mask[4] is None:
            mask = 1.0
        else:
            mask = K.cast(mask[4], K.floatx())
        # sujuect部分loss
        subject_loss = K.binary_crossentropy(subject_labels, subject_preds)
        subject_loss = K.mean(subject_loss, 2)
        subject_loss = K.sum(subject_loss * mask) / K.sum(mask)
        # object部分loss
        object_loss = K.binary_crossentropy(object_labels, object_preds)
        object_loss = K.sum(K.mean(object_loss, 3), 2)
        object_loss = K.sum(object_loss * mask) / K.sum(mask)
        # 总的loss
        return subject_loss + object_loss


subject_preds, object_preds = TotalLoss([2, 3])([
    subject_labels, object_labels, subject_preds, object_preds,
    bert.model.output
])

# AdamEMA = extend_with_exponential_moving_average(Adam, name='AdamEMA')
# optimizer = AdamEMA(learning_rate=1e-5)
# 训练模型
optimizer = Adam(learning_rate=learning_rate_)

print('start training...')

with strategy.scope():
    train_model = Model(
        bert.model.inputs + [subject_labels, subject_ids, object_labels],
        [subject_preds, object_preds]
    )
    train_model.compile(optimizer=optimizer, metrics=['acc'])


def extract_spoes(text):
    """抽取输入text所包含的三元组
    """
    tokens = tokenizer.tokenize(text, maxlen=maxlen)
    mapping = tokenizer.rematch(text, tokens)
    token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
    token_ids, segment_ids = to_array([token_ids], [segment_ids])
    # 抽取subject
    subject_preds = subject_model.predict([token_ids, segment_ids])
    start = np.where(subject_preds[0, :, 0] > 0.6)[0]
    end = np.where(subject_preds[0, :, 1] > 0.5)[0]
    subjects = []
    for i in start:
        j = end[end >= i]
        if len(j) > 0:
            j = j[0]
            subjects.append((i, j))
    if subjects:
        spoes = []
        token_ids = np.repeat(token_ids, len(subjects), 0)
        segment_ids = np.repeat(segment_ids, len(subjects), 0)
        subjects = np.array(subjects)
        # 传入subject，抽取object和predicate
        object_preds = object_model.predict([token_ids, segment_ids, subjects])
        for subject, object_pred in zip(subjects, object_preds):
            start = np.where(object_pred[:, :, 0] > 0.6)
            end = np.where(object_pred[:, :, 1] > 0.5)
            for _start, predicate1 in zip(*start):
                for _end, predicate2 in zip(*end):
                    if _start <= _end and predicate1 == predicate2:
                        spoes.append(
                            ((mapping[subject[0]][0],
                              mapping[subject[1]][-1]), predicate1,
                             (mapping[_start][0], mapping[_end][-1]))
                        )
                        break
        return [(text[s[0]:s[1] + 1], id2predicate[p], text[o[0]:o[1] + 1])
                for s, p, o, in spoes]
    else:
        return []


class SPO(tuple):
    """用来存三元组的类
    表现跟tuple基本一致，只是重写了 __hash__ 和 __eq__ 方法，
    使得在判断两个三元组是否等价时容错性更好。
    """

    def __init__(self, spo):
        self.spox = (
            tuple(tokenizer.tokenize(spo[0])),
            spo[1],
            tuple(tokenizer.tokenize(spo[2])),
        )

    def __hash__(self):
        return self.spox.__hash__()

    def __eq__(self, spo):
        return self.spox == spo.spox


f1_process = []
ac_process = []
rc_process = []


def evaluate(data):
    """评估函数，计算f1、precision、recall
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    f = open('dev_pred.json', 'w', encoding='utf-8')
    #     pbar = tqdm(data)
    count_muti=0
    count_muti_false=0
    for d in data:
        R = set([SPO(spo) for spo in extract_spoes(d['text'])])
        T = set([SPO(spo) for spo in d['spo_list']])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        s = json.dumps({
            'text': d['text'],
            'spo_list': list(T),
            'spo_list_pred': list(R),
            'new': list(R - T),
            'lack': list(T - R),
        },
            ensure_ascii=False,
            indent=4)
        if len(T)>1:
            count_muti+=1
            if max(len(R-T),len(T-R))>0:
                f.write(s + '\n')
                count_muti_false+=1
    print("多关系准确率: %.5f"%(1-count_muti_false/count_muti))
    print(count_muti)
            
    #     pbar.close()
    f1_process.append(f1)
    ac_process.append(precision)
    rc_process.append(recall)
    f.close()
    return f1, precision, recall


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """

    def __init__(self):
        self.best_val_f1 = 0.

    def on_epoch_end(self, epoch, logs=None):
        #         optimizer.apply_ema_weights()
        f1, precision, recall = evaluate(valid_data)
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            train_model.save_weights('best_model.weights')
        #         optimizer.reset_old_weights()
        filename="%d_%f.txt"%(batch_size,learning_rate_)
        with open("./output/epoch/"+filename,"a") as f:
            f.write(
                '%.5f, %.5f, %.5f\n' %
                (f1,precision,recall)
            )
        print(
            'f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1))


def predict(text):
    R = set([SPO(spo) for spo in extract_spoes(text)])
    print(list(R))

    # %%capture capt

filename="%d_%f.txt"%(batch_size,learning_rate_)
with open("./output/epoch/" + filename, "w") as f:
    pass
    
if len(sys.argv) == 1 or sys.argv[1]=="train":

    train_generator = data_generator(train_data, batch_size)
    evaluator = Evaluator()

    train_model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs_,
        callbacks=[evaluator]
    )
    train_model.load_weights('best_model.weights')
    f1, precision, recall = evaluate(test_data)
    filename = "%d_%f_test.txt" % (batch_size, learning_rate_)
    with open("./output/" + filename, "w") as f:
        f.write(
            '%.5f, %.5f, %.5f\n' %
            (f1, precision, recall)
        )
    print(
        'f1: %.5f, precision: %.5f, recall: %.5f\n' %
        (f1, precision, recall))

elif sys.argv[1]=="test":
    train_model.load_weights('best_model.weights')
    f1, precision, recall = evaluate(test_data)
    filename = "%d_%d_%f_test.txt" % (maxlen, batch_size, learning_rate_)
    with open("./output/" + filename, "w") as f:
        f.write(
            '%.5f, %.5f, %.5f\n' %
            (f1, precision, recall)
        )
    print(
        'f1: %.5f, precision: %.5f, recall: %.5f\n' %
        (f1, precision, recall))
