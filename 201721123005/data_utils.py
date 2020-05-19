
from queue import Queue
from threading import Thread
import random


# 序列截断与补齐，保持一样的长度
def padding_seq(seq):
    results = []
    max_len = 0
    # 找到最长的长度max_len
    for s in seq:
        if max_len < len(s):
            max_len = len(s)
    # 如果句子长度小于最大长度，补零填充到最大长度
    for i in range(0, len(seq)):
        l = max_len - len(seq[i])
        results.append(seq[i] + [0 for j in range(l)])
    return results

#根据词表里的词与id对应关系决定

#把文本序列映射为下标id序列
def encode_text(words, vocab_indices):
    return [vocab_indices[word] for word in words if word in vocab_indices]

#把输出的下标id序列映射回文本序列 ，‘ </s>’是词表里表示句子结尾的符号。
def decode_text(labels, vocabs, end_token = '</s>'):
    results = []
    for idx in labels:
        word = vocabs[idx]
        if word == end_token:
            return ' '.join(results)
        results.append(word)
    return ' '.join(results)

#加载词表
#’<s>’、’</s>’分别表示句子的开头和结尾，但是用零填充句子后，id值为0的数字没有标识符与之对应，所以要添加一个’<pad>’标识符与之对应
def read_vocab(vocab_file):
     f = open(vocab_file, 'rb')
     vocabs = [line.decode('utf8')[:-1] for line in f]
     vocabs.insert(0,'<pad>')
     f.close()
     return vocabs


class SeqReader():
    def __init__(self, input_file, target_file, vocab_file, batch_size,
            queue_size = 2048, worker_size = 2, end_token = '</s>',
            padding = True, max_len = 50):
        self.input_file = input_file
        self.target_file = target_file
        self.end_token = end_token
        self.batch_size = batch_size
        self.padding = padding
        self.max_len = max_len
        # self.vocabs = read_vocab(vocab_file) + [end_token]
        self.vocabs = read_vocab(vocab_file)
        self.vocab_indices = dict((c, i) for i, c in enumerate(self.vocabs))
        self.data_queue = Queue(queue_size)
        self.worker_size = worker_size
        with open(self.input_file) as f:
            for i, l in enumerate(f):
                pass
            f.close()
            self.single_lines = i+1
        self.data_size = int(self.single_lines / batch_size)
        self.data_pos = 0
        self._init_reader()


    def start(self):
        return
    '''
        for i in range(self.worker_size):
            t = Thread(target=self._init_reader())
            t.daemon = True
            t.start()
    '''


    def read_single_data(self):
        if self.data_pos >= len(self.data):
            random.shuffle(self.data)
            self.data_pos = 0
        result = self.data[self.data_pos]
        self.data_pos += 1
        return result


    def read(self):
        while True:
            batch = {'in_seq': [],
                    'in_seq_len': [],
                    'target_seq': [],
                    'target_seq_len': []}
            for i in range(0, self.batch_size):
                item = self.read_single_data()
                batch['in_seq'].append(item['in_seq'])
                batch['in_seq_len'].append(item['in_seq_len'])
                batch['target_seq'].append(item['target_seq'])
                batch['target_seq_len'].append(item['target_seq_len'])
            if self.padding:
                batch['in_seq'] = padding_seq(batch['in_seq'])
                batch['target_seq'] = padding_seq(batch['target_seq'])
            yield batch


    def _init_reader(self):
        self.data = []
        input_f = open(self.input_file, 'rb')
        target_f = open(self.target_file, 'rb')
        for input_line in input_f:
            input_line = input_line.decode('utf-8')[:-1]
            target_line = target_f.readline().decode('utf-8')[:-1]
            input_words = [x for x in input_line.split(' ') if x != '']
            if len(input_words) >= self.max_len:
                input_words = input_words[:self.max_len-1]
            input_words.append(self.end_token)
            target_words = [x for x in target_line.split(' ') if x != '']
            if len(target_words) >= self.max_len:
                target_words = target_words[:self.max_len-1]
            target_words = ['<s>',] + target_words
            target_words.append(self.end_token)
            in_seq = encode_text(input_words, self.vocab_indices)
            target_seq = encode_text(target_words, self.vocab_indices)
            self.data.append({
                'in_seq': in_seq,
                'in_seq_len': len(in_seq),
                'target_seq': target_seq,
                'target_seq_len': len(target_seq) - 1
            })
        input_f.close()
        target_f.close()
        self.data_pos = len(self.data)
