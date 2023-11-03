import json
import re
import pickle
import itertools
from functools import reduce


class InformalHelper:
    def __init__(self, vocab, verb_handler, mapper):
        self.vocab = vocab
        self.mapper = mapper
        self.verb_handler = verb_handler
        self.adjectives = [word for word,info in self.vocab.items() if 'AJ' in info['tags'].split(',')]
        self.pronouns = ['من', 'تو', 'او', 'ما', 'شما', 'آن‌ها', 'ان‌ها', 'آنان', 'انان', 'آن‌ها', 'ایشان']
        self.op_separete = {'م': 'من',  'ت': 'تو', 'ش' : 'آن', 'تان': 'شما', 'تون': 'شما', 'شون': 'آنان', 'شان': 'آنان', 'مان': 'ما'}
        self.op = ['مون', 'شون','تون', 'مان','تان', 'شان','م','ت' ,'ش']
        self.verb_ends = [ 'یم', 'ین', 'ید', 'ند', 'ن', 'م', 'ی', 'ه']
        self.not_connect_chars = ['ا', 'د', 'ذ', 'ر', 'ز', 'ژ', 'و']

    def get_verb_ends(self):
        return self.verb_ends

    def is_in_vocab(self, word):
        return word in self.vocab

    def is_verb(self, word):
        return self.verb_handler.is_verb(word)


    def is_formal_verb(self, word):
        return self.verb_handler.is_formal_verb(word)

    def is_adj(self, word):
        return word in self.adjectives

    def is_pronoun(self, word):
        return word in self.pronouns

    def is_in_mapper(self, word):
        return word in self.mapper

    def get_op_separete(self, word):
        if word in self.op_separete:
            return self.op_separete[word]

    def get_objective_prouns(self):
        return self.op

    def is_objective_p(self, word):
        return word in self.op

    def is_valid_word_by_po(self, word):
        op_pattern = '(' + '|'.join(self.get_objective_prouns()) + ')'
        op_match = re.match('(\w+)' + op_pattern + '$', word)
        if op_match is not None and self.is_in_vocab(op_match.group(1)):
            return True
        return False

    def is_formal_prefixed(self, word):
        nim_fasele = '‌'
        m1 = re.match('(.+)های(م|ت|ش|مان|تان|شان)?$', word)
        m2 = re.match('(.+[ا|و|ی])ی(م|ت|ش|مان|تان|شان)$', word)
        m3 = re.match('(.+[^ا^و^ی])(م|ت|ش|مان|تان|شان)$', word)
        m4 = re.match('(.+)(ها)$', word)
        m5 = re.match('(.+[ه|ی]‌)(اش|ام|ات)$', word)
        if m3 or m2:
            prefix_word = list(filter(lambda m: m is not None, [m3, m2]))[0].group(1)
            if self.is_in_vocab(prefix_word):
                return True
        m_fired = list(filter(lambda m: m is not None, [m1, m4, m5]))
        if len(m_fired) > 0:
            # print(word, m_fired[0].groups())
            prefix_word = m_fired[0].group(1)
            if prefix_word[-1] != nim_fasele and prefix_word[-1] not in self.not_connect_chars:
                return False
            if prefix_word[-1] == nim_fasele and not self.is_in_vocab(prefix_word[:-1]):
                return False
            if prefix_word[-1] != nim_fasele and not self.is_in_vocab(prefix_word):
                return False
            return True
        return False




def get_objective_prouns():
    return ['م', 'ت', 'ش', 'مون', 'تون', 'شون', 'شان' '', 'تان', 'مان']

def get_mosavet_list():
    return ['ا', 'ی', 'و']

def load_var(load_path):
    file = open(load_path, 'rb')
    variable = pickle.load(file)
    file.close()
    return variable

def mix_str(s_list1, s_list2):
    return []

def extract(nested_list, res):

    """
  [[['میز, 'است'], ['میز']]]
  for item in neste_list:
    if all(type(w) == list for w in item):
        return ' '.join(item)
    else:
    return mix_str(non_str)
    abcdf
    abcef

    :param nested_list:
    :param res:
    :return:
    """
    if type(nested_list) != list:
        return nested_list
    for item in nested_list:
        if type(item) != list:
            res.append(item)
        else:
            extract(item, res)


def mix_str(list_s1, list_s2):
    if len(list_s1) == 0:
        return list_s2
    if len(list_s2) == 0:
        return list_s1
    return list(itertools.product(list_s1, list_s2))

def nested_join(l):
    def expand_tuple(t):
        if type(t) == str:
            return t
        if type(t) == list:
            return nested_join(t)
        out = []
        for item in t:
            expnded_item = expand_tuple(item)
            if type(expnded_item) == tuple:
                for element in expnded_item:
                    out.append(element)
            else:
                out.append(expnded_item)
        return tuple(x for x in out)

    if type(l) == str:
        return l
    if type(l) == list and all(type(item) == str for item in l):
        return ' '.join(l)
    if type(l) == tuple and all(type(item) == str for item in l):
        return l
    if type(l) == tuple:
        l = expand_tuple(l)
        if all(type(x) == str for x in l):
            return l
    if type(l) == tuple:
        return list(l)
    if type(l) == list:
        l = [nested_join(item) for item in l]
        out = []
        #['a', ['b','c'], 'd']
        swap = ['X' for _ in range(len(l))]
        should_product = []
        for i, item in enumerate(l):
            if type(item) != tuple:
                swap[i] = item
            else:
                should_product.append(item)
        p = list(itertools.product(*[x for x in should_product]))
        j = 0

        for item in p:
            s = ''
            j = 0
            for i, x in enumerate(swap):
                if x == 'X':
                    if type(item[j]) == list:
                        s += ' '.join(item[j])
                    else:
                        s += item[j]
                    j+=1
                else:
                    if type(x) == list:
                        s += ' '.join(x)
                    else:
                        s += x
                s+= ' '

            out.append(s.strip())
        return tuple(out)

def is_sequence(item):
    return  all(type(x) == list and len(x) == 1 for x in item)

def unpack_nested(nested_list):

    if type(nested_list) == str:
        return nested_list
    if type(nested_list) != str and len(nested_list) == 1:
        return nested_list[0]
    #tuple
    if type(nested_list) == tuple:
        out = tuple(unpack_nested(item) for item in nested_list)
        return nested_join(out)
    elif type(nested_list) == list:
        return nested_join(nested_list)
    # while True:
    #     if type(nested_list) == list and len(nested_list) == 1 and type(nested_list[0]) == list:
    #         nested_list =  nested_list[0]
    #     else:
    #         break
    # if type(nested_list) == str:
    #     return nested_list
    # if len(nested_list) == 1 and type(nested_list[0]) == str:
    #     return nested_list[0]
    # if type(nested_list) == list:
    #     out = [unpack_nested(item) for item in nested_list]
    # else:
    #     out = tuple(unpack_nested(item) for item in nested_list)
    # return  nested_join(out)

def is_number(word):
    pattern = '[1234567890.۲۱۳۴۵۶۷۸۹]+$'
    if re.match(pattern, word):
        return True
    else:
        return False

def is_punc(word):
    pattern = '[?!@#$%^&*()_?؟+]+'
    if re.match(pattern, word):
        return True
    else:
        return False

def get_set_of_modified_word(word, swaped_chars):
    #swaped_chars = ['0 : [a,b], ,'1': '', '2': [c,d]]
    should_product = [list(c_set) for c_set in swaped_chars if len(c_set)>0]
    p = list(itertools.product(*should_product))
    swaped_indx = [i for i,item in enumerate(swaped_chars) if len(item) > 0]
    out = []
    for item in p:
        s_indx = 0
        s = []
        for i, ch in enumerate(word):
            if i in swaped_indx:
                s += item[s_indx]
                s_indx += 1
            else:
                s += word[i]
        out.append(''.join(s))
    return out

def get_possible_repeated_word(word):
    current_ch = word[0]
    l = 0
    chars = []
    for i, ch in enumerate(word):
        if ch == current_ch:
            l+=1
            continue
        else:
            chars.append((current_ch, l))
            l = 1
            current_ch = ch
    chars.append((current_ch, l))
    repeat_chars = [c[0] for c in chars if c[1] > 1]
    repeat_count = len(repeat_chars)
    bin_combination = list(itertools.product([0, 1], repeat=repeat_count))
    all_words = []
    for item in bin_combination:
        must_uniqed_chars = [ch for i,ch in enumerate(repeat_chars) if item[i] == 0]
        new_word = word
        for ch in must_uniqed_chars:
            patt = ch + '{2,}'
            new_word = re.sub(patt, ch, new_word)
        all_words.append(new_word)
    return all_words

def spelling_similairty(word):
    all_possible = []
    possible_repeated = get_possible_repeated_word(word)
    all_possible = possible_repeated
    # for word in possible_repeated:
    #     ingroup_chars = [{'ا'},
    #                           {'ت', 'ط'},
    #                           {'ئ', 'ی'},
    #                           {'ث', 'س', 'ص'},
    #                           {'ح', 'ه'},
    #                           {'ذ', 'ز', 'ض', 'ظ'},
    #                           {'ق', 'غ'},
    #                           {'س', 'ش'},
    #                           {'ز', 'ر'},
    #                           {'ب', 'ی'},
    #                           {'گ', 'ک'},
    #                           {'چ', 'ج'}]
    #     #remove repeated chars
    #     sim_sets = [set() for _ in range(len(word))]
    #     for i,c in enumerate(word):
    #         for c_set in ingroup_chars:
    #             if c in c_set:
    #                 sim_sets[i].update(c_set)
    #
    #     modified_set = get_set_of_modified_word(word, swaped_chars = sim_sets)
    #     all_possible += modified_set
    if word in all_possible:
        all_possible.remove(word)
    return all_possible

def add_nim_alef_hat_dictionary(vocab):
    word_with_hat = filter(lambda w: 'آ' in w, vocab)
    word_with_nim = filter(lambda w: '‌' in w, vocab)
    mapper1 = {w.replace('آ', 'ا').replace('‌', ''): w for w in word_with_hat}
    mapper2 = {w.replace('‌', ''): w for w in word_with_nim}
    mapper1.update(mapper2)
    return mapper1

def generate_spell_mapper(vocab):
    hat = 'آ'
    tanvin =  'اً'
    nim =  '‌'
    hamzeh = 'أ'
    hamzeh_y = 'ئ'
    sp_mapper = {hamzeh_y: ['ی'], hat: ['ا'], tanvin: ['ن', 'ا'], nim:[''], hamzeh:['ا', '']}
    special_chars = [hat, tanvin, nim, hamzeh]
    out = {}
    for word in vocab:
        p_words = [word.replace(sp, sp_alt) for sp in special_chars for sp_alt in sp_mapper[sp]]
        spell_errors = []
        for p in p_words:
            spell_errors += spelling_similairty(p)
        p_words += spell_errors
        p_words = list(set(p_words) - set([word]))
        for pw in p_words:
            if pw in out:
                out[pw].add(word)
            else:
                out[pw] = {word}
    out = {w: list(out[w]) for w in out}
    with open('spell_dict.json', 'w+', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

def create_bigram_freq(corpus, normalizer, tokenizer):
    out = {}
    for item in corpus:
        item = normalizer.normalize(item)
        tokens = tokenizer.tokenize(item)
        bigrams = [(t1, t2) for t1, t2 in zip(tokens, tokens[1:])]
        for bi in bigrams:
            if bi in out:
                out[bi] += 1
            else:
                out[bi] = 1
    return out


def powerset(lst):
    return reduce(lambda result, x: result + [subset + [x] for subset in result],
                  lst, [[]])

def create_onegram(corpus, normalizer, tokenizer, freq_min):
    out = {}
    for item in corpus:
        item = normalizer.normalize(item)
        tokens = tokenizer.tokenize(item)
        # bigrams = [(t1, t2) for t1, t2 in zip(tokens, tokens[1:])]
        for tok in tokens:
            if tok in out:
                out[tok] += 1
            else:
                out[tok] = 1
    out = {tok:freq for tok,freq in out.items() if freq > freq_min}
    return out
if __name__ == '__main__':
    with open('resources/spell/mybigram_lm.pckl', 'rb') as f:
        data = pickle.load(f)
        print('')
#
#     vocab_addr='resources/words.dat'
#     vocab = {}
#     with open(vocab_addr, 'r', encoding='utf-8') as f:
#         for line in f:
#             try:
#                 word, freq, p_tags = line.strip().split('\t')
#                 # word = normalizer.normalize(word)
#                 vocab[word] = {'freq': freq, 'tags': p_tags}
#             except:
#                 word = line.strip()
#                 vocab[word] = {'freq': 1, 'tags':'NUM'}
#
#         vocab = [word for word in vocab]
#         generate_spell_mapper(vocab)
# #
# #     with open('formal_mapper', 'w+') as f:
# #         for word1, word2 in mappers.items():
# #             f.write('{},{}'.format(word2, word1))
# #             f.write('\n')
# import pandas as pd
# mapper = pd.read_csv('resources/mapper.csv', names=['formal', 'informal'], header=None, delimiter=',')
# print('اگه' in mapper['formal'])
# exit()
# with open('formal_mapper') as f:
#     f_mapper = {line.split(',')[1] : line.split(',')[0] for line in f.read().splitlines() if line.split(',')[1]  not in mapper['informal']}
# mapper_dict = {str(w1):str(w2) for _, (w1,w2) in mapper.iterrows()}
# mapper_dict.update(f_mapper)
# with open('resources/word_mapper.txt', 'w+') as f:
#     f.write('\n'.join(['{},{}'.format(for_word, inf_word) for inf_word, for_word in mapper_dict.items() if for_word not in vocab]))
# # print(spelling_similairty('ممممنونمممممممممم'))


# sample = [('x', ['yz', 'yt']), ('p', ('q', 'r'))]
# sample2 = (('AB', 'CD'), (('E', 'F'), 'G', 'H'))
# print(nested_join(sample))
# print(expand_tuple(sample2))

#
# objective_p = ['مان', 'تان', 'شان', 'مون', 'تون', 'شون', 'م', 'ت', 'ش']
# p = r'(می|نمی)\s+(\w+)\b'
# verb_prefix_pattern = re.compile(p)
# verb_prefix_repl = r'\1\2'
# informal_pattern = re.compile('(\w+)\s*(ها)?\s*(' + '|'.join(objective_p) + ')(و)?\\b')
# informal_pattern_alef = re.compile('(\w+)\s+(ا)(' + '|'.join(objective_p) + ')(و)?\\b')
# informal_repl = r'\1\2\3\4'
# informal_alef_repl = r'\1‌\2\3\4'
# text = informal_pattern.sub(informal_repl, text)
# text = informal_pattern_alef.sub(informal_alef_repl, text)
# text = verb_prefix_pattern.sub(verb_prefix_repl, text)


# def get_words_with_ch(dictionary, ch):
# 	return filter(lambda w: ch in w, dictionary)
# vocab_addr='resources/words.dat'
# vocab = {}
# with open(vocab_addr, 'r', encoding='utf-8') as f:
#     for line in f:
#         try:
#             word, freq, p_tags = line.strip().split('\t')
#             # word = normalizer.normalize(word)
#             vocab[word] = {'freq': freq, 'tags': p_tags}
#         except:
#             word = line.strip()
#             vocab[word] = {'freq': 1, 'tags':'NUM'}
#
#     vocab = [word for word in vocab]
# prefixs = []
# postfixs = []
# for v in vocab:
#     m_prefix = re.match('(\w+)‌.*', v)
#     m_postfix = re.match('.*‌(\w+)', v)
#     if m_prefix :
#         prefixs.append(m_prefix.group(1))
#     if m_postfix:
#         postfixs.append(m_postfix.group(1))
# prfx = set(filter(lambda w: prefixs.count(w) > 20, prefixs))
# psfix = set(filter(lambda w: postfixs.count(w) > 20, postfixs))
# with open('resources/prefixs.txt', 'w+') as f:
#     f.write('\n'.join(prfx))
# with open('resources/postfixs.txt', 'w+') as f:
#     f.write('\n'.join(psfix))
# print('')
#     #
# with_hat = get_words_with_ch(vocab, 'آ')
# with_tanvin = get_words_with_ch(vocab, 'ً')
# with_nim = get_words_with_ch(vocab, '‌')
# expctions = [w for w in with_hat if w.replace('آ', 'ا') in vocab]
# expctions1 = [w for w in with_tanvin if w.replace('ً', '') in vocab]
# expctions2 = [w for w in with_nim if w.replace('‌', '') in vocab]
# # expctions3 = [w for w in with_nim if w.replace('‌', ' ') in vocab]
# print(expctions2)
# with open('resources/words_with_hat.txt', 'w+') as f:
# 	f.write('\n'.join(with_hat))
# with open('resources/words_with_tanvin.txt', 'w+') as f:
# 	f.write('\n'.join(with_tanvin))
# with open('resources/words_with_nim.txt', 'w+') as f:
# 	f.write('\n'.join(with_nim))
# # #
# def get_words_with_tanvin(dictionary):
# 	return filter(lambda w: 'ً' in w, dictionary)

