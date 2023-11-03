import itertools
import json
import os
import pickle
import re
import string
from random import shuffle
import utils
from LMGP2 import GP2LM
from NER import NER
from OneShotTransformer import OneShotTransformer
from VerbHandler import VerbHandler
import pandas as pd
from hazm import WordTokenizer, Normalizer
from postagger import Postagger
from spellCheker import SpellChecker
from tokenizer import InformalTokenizer
from flask import Flask, request, Response
from flask_restful import Api, Resource, reqparse
from flask_cors import CORS
from flask import jsonify

from utils import extract_non_convertable_words, create_bigram, create_ongram, create_vocab_file

base_addr = os.path.dirname(__file__)
H_MALFOOZ_ADDR = './files/h_malfouz_words500.txt'
MAPPER_ADDR = './files/mapper.csv'
HAZM_VOCAB_ADDR = './files/words.dat'
wiki50_vocab_addr = './files/wiki50_vocab.dat'
UPC_VOCAB_ADDR = './files/word_pos_UPC.txt'
POSTFIX_MAPPER_ADDR = './files/tokenizer/postfix_mapper.csv'
ISOLATED_WORDS_ADDR = './files/isolate_words'
POSTFIX_FREQ_ADDR = './files/tokenizer/informal_postfixs_freq'
POSTFIXS_ADDR = './files/tokenizer/informal_postfixs.txt'
VERBS_CSV_ADDR = './files/verbs.csv'
INFORMAL_IN_FORMAL_WORDS_ADDR = './files/informal_words_in_formal_corpus.txt'
ENDS_WITH_TANVIN_WORDS_ADDR = './files/spell/end_with_tanvin.txt'
SPELL_CHECKING_MAPPER_ADDR = './files/spell/spell_checker_mapper.json'
IRREGULAR_VERBS_MAPPRER_ADDR = './files/irregular_verb_mapper.csv'
FLEXIOCN_VOCAB_ADDR = os.path.join(base_addr, 'files/flexicon_vocab.txt')


class Informal2Formal:
    def __init__(self):
        # self.onegrams = self.load_onegram(ONEGRAM_ADDR)
        self.vocab = self.load_vocab([FLEXIOCN_VOCAB_ADDR, HAZM_VOCAB_ADDR,  UPC_VOCAB_ADDR], remove_informals=False)
        # bigrams = load_bigram()
        self.ner = NER()
        self.word_ends_tanvin = open(ENDS_WITH_TANVIN_WORDS_ADDR).read().splitlines()
        self.non_hidden_h_words = self.get_non_hidden_h_words()
        self.isolated_words = open(ISOLATED_WORDS_ADDR).read().splitlines() + self.word_ends_tanvin
        self.ignore_words_ferq = open(POSTFIX_FREQ_ADDR).read().splitlines()
        self.ignore_words = [x[0] for x in [l.split() for l in self.ignore_words_ferq]]
        # create objects
        self.normalizer = Normalizer()
        self.tokenizer = WordTokenizer(separate_emoji=True)
        self.informal_tokenizer = InformalTokenizer(self.vocab, POSTFIXS_ADDR)
        self.verb_handler = VerbHandler(csv_verb_addr=VERBS_CSV_ADDR, csv_irregular_verbs_mapper=IRREGULAR_VERBS_MAPPRER_ADDR)
        self.mapper = self.load_mapper(MAPPER_ADDR, True)
        self.postfix_mapper = self.load_mapper(POSTFIX_MAPPER_ADDR, False)
        self.oneshot_transformer = OneShotTransformer(self.vocab, self.mapper, self.verb_handler.informal_to_formal,
                                                      ignore_words=self.ignore_words,
                                                      postfix_mapper=self.postfix_mapper,
                                                      isolated_words=self.isolated_words,
                                                      non_hidden_h_words=self.non_hidden_h_words)
        self.spell_checker = SpellChecker(using_symspell=False, check_informal_valid=self.oneshot_transformer.transform,
                                          vocab=self.vocab,
                                          spell_checker_mapper_addr=SPELL_CHECKING_MAPPER_ADDR)
        # self.pos_tagger = Postagger()
        self.lm = GP2LM()

    def get_non_hidden_h_words(self):
        with open(H_MALFOOZ_ADDR, 'r') as f:
            return f.read().splitlines()

    def load_vocab(self, all_vocab_addrs, remove_informals):
        vocab = {}
        for vocab_addr in all_vocab_addrs:
            with open(vocab_addr, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        word, freq, p_tags = line.strip().split('\t')
                        vocab[word] = {'freq': freq, 'tags': p_tags}
                    except:
                        word = line.strip()
                        vocab[word] = {'freq': 1, 'tags': 'NUM'}

        if remove_informals:
            with open(INFORMAL_IN_FORMAL_WORDS_ADDR) as f:
                inf_formal_words = f.read().splitlines()
            inf_formal_words = set(inf_formal_words)
            filtered_vocab = {}
            for v in vocab:
                if v not in inf_formal_words:
                    filtered_vocab[v] = vocab[v]
            return filtered_vocab
        return vocab

    def load_mapper(self, mapper_addr, is_list):
        df = pd.read_csv(mapper_addr)
        if is_list:
            mapper = {informal: [f.strip() for f in formal.split('-')] for _, (informal, formal) in df.iterrows()}
            print([x for x in mapper if len(mapper[x]) > 1])
        else:
            mapper = {informal: formal for _, (informal, formal) in df.iterrows()}
        return mapper

    def load_onegram(self, addr):
        with open(addr, 'rb') as f:
            data = pickle.load(f)
        return data

    def load_bigram(self, addr):
        with open(addr, 'rb') as f:
            data = pickle.load(f)
        return data

    def should_filtered_by_one_bigram(self, lemma, word, original_word):
        NIM_FASELE = '‌'
        if original_word in self.vocab and (len(word.split()) > 1 or NIM_FASELE in word):
            return True
        # ONEGRAM_LIMIT = 10
        # # BIGRAM_LIMIT = 5
        # toks = word.split()
        # if original_word in self.onegrams:
        #     original_freq = self.onegrams[original_word]
        # else:
        #     return False
        # for t in toks:
        #     if (original_word in self.vocab) and (
        #             (t not in self.onegrams) or (self.onegrams[t] < 500 and original_freq / self.onegrams[t] > 100)):
        #         return True

        return False

    def repalce_for_gpt2(self, word_repr):
        if word_repr in self.word_ends_tanvin:
            return word_repr[:-2] + 'ا'
        return word_repr

    def seprate_emoji_string(self, txt):
        try:
            # Wide UCS-4 build
            oRes = re.compile(u'(['
                              u'\U0001F300-\U0001F64F'
                              u'\U0001F680-\U0001F6FF'
                              u'\u2600-\u26FF\u2700-\u27BF]+)',
                              re.UNICODE)
        except re.error:
            # Narrow UCS-2 build
            oRes = re.compile(u'(('
                              u'\ud83c[\udf00-\udfff]|'
                              u'\ud83d[\udc00-\ude4f\ude80-\udeff]|'
                              u'[\u2600-\u26FF\u2700-\u27BF])+)',
                              re.UNICODE)

        return oRes.sub(r'  \1  ', txt)


def split_by_verbs(txt):
    tokens = model.tokenizer.tokenize(txt)
    short_sents = []
    current_s = ''
    for t in tokens:
        if model.verb_handler.informal_to_formal(t) and t[1] != 'ن' and len(current_s.split()) > 4:
            current_s = current_s + ' ' + t
            short_sents.append(current_s)
            current_s = ''
        else:
            current_s = current_s + ' ' + t
    if current_s != '':
        short_sents.append(current_s)
    return short_sents


def clean_text_for_lm(txt):
    ignore_chars = '.1234567890!@#$%^&*()_+۱۲۳۴۵۶۷۸۹÷؟×−+?><}،,{":' + string.ascii_lowercase + string.ascii_uppercase
    tokens = txt.split()
    clean_tokens = [t for t in tokens if not (any(ic in t for ic in ignore_chars) or utils.if_emoji(t))]
    return ' '.join(clean_tokens)


def translate_short_sent(sent):
    # print(sent)
    out_dict = {}
    txt = sent.strip()
    txt = re.sub('\s+', ' ', txt)
    txt = re.sub('\u200f', '', txt)
    #remove extra nim_fasele
    txt = re.sub('‌+', '‌', txt)
    #remove nim_fasele before space
    txt = re.sub('‌ ', ' ', txt)
    txt = re.sub(' ‌', ' ', txt)
    txt = model.normalizer.normalize(txt)
    txt = model.seprate_emoji_string(txt)
    txt = ' '.join(model.tokenizer.tokenize(txt))
    pn_words = model.ner.get_n_words(txt)
    is_valid = lambda w: w in pn_words or model.oneshot_transformer.transform(w, None)
    cnd_tokens = model.informal_tokenizer.tokenize(txt, is_valid)

    min_ppl_score = 1000000
    best = ''
    for tokens in cnd_tokens:
        tokens = [t for t in tokens if t != '']
        new_tokens = []
        for t in tokens:
            new_tokens.extend(t.split())
        txt = ' '.join(new_tokens)
        txt = model.spell_checker.spell_checking(txt, pn_words)
        tokens = txt.split()
        # poses = model.pos_tagger.tag(txt)
        candidates = []
        for index in range(len(tokens)):
            # try:
            #     pos = poses[index + 1]
            # except:
            pos = None
            # pos = None
            tok = tokens[index]
            # print(tok, tok in vocab)
            cnd = set()
            # print(pos , tok)
            if model.verb_handler.informal_to_formal(tok):
                pos = 'VERB'
            elif pos == 'VERB':
                pos = None
            if tok in pn_words:
                f_words_lemma = [(tok, tok)]
            else:
                f_words_lemma = model.oneshot_transformer.transform(tok, pos)
                f_words_lemma = list(f_words_lemma)
            ### only lemma of nim_fasele words
            NIM_FASELE = '‌'
            # print(f_words_lemma)
            for index, (word, lemma) in enumerate(f_words_lemma):
                if pos != 'VERB' and tok not in model.mapper and model.should_filtered_by_one_bigram(lemma, word, tok):
                    f_words_lemma[index] = (tok, tok)
                else:
                    word_toks = word.split()
                    word_repr = ''
                    # if not model.lm.is_exist_lemmas_in_tokenizer(lemma):
                    #     continue
                    for t in word_toks:
                        if model.lm.is_exist_lemmas_in_tokenizer(t):
                            word_repr += ' ' + t
                        else:
                            word_repr += ' ' + t
                    word_repr = word_repr.strip()
                    word_repr = model.repalce_for_gpt2(word_repr)
                    f_words_lemma[index] = (word, word_repr)
            # lemmas = [fl[1] for fl in f_words_lemma]
            if f_words_lemma:
                # if type(rule_engine_formal) is str:
                #     rule_engine_formal = {rule_engine_formal}
                cnd.update(f_words_lemma)
            else:
                cnd = {(tok, tok)}
            # outputs.append( (tok, cnd))
            candidates.append(cnd)
        # print(candidates)
        all_combinations = itertools.product(*candidates)
        all_combinations_list = list(all_combinations)
        for id, cnd in enumerate(all_combinations_list):
            normal_seq = ' '.join([c[0] for c in cnd])
            lemma_seq = ' '.join([c[1] for c in cnd])
            lemma_seq = clean_text_for_lm(lemma_seq)
            out_dict[id] = (normal_seq, lemma_seq)
        lemma_seqs = {id: out_dict[id][1] for id in out_dict}
        # print(lemma_seqs)
        current_best_id, current_ppl_score = model.lm.get_best_candidates(lemma_seqs)
        if current_ppl_score < min_ppl_score:
            best_id = current_best_id
            min_ppl_score = current_ppl_score
            best = out_dict[best_id][0]
        return best


def translate(txts):
    outputs = {}
    for txt_index, origin_txt in enumerate(txts):
        try:
            short_sents = split_by_verbs(origin_txt)
            origin_best = ''
            for s_s in short_sents:
                formal_s_s = translate_short_sent(s_s)
                origin_best += (' ' + formal_s_s)
            outputs[txt_index] = origin_best
            print(origin_best)
        except :
            outputs[txt_index] = origin_txt
    return outputs


if __name__ == '__main__':
    """development!"""
    global model
    # results = []
    model = Informal2Formal()
    # print('اینا' in model.vocab)
    # print('همینه' in model.vocab)
    # print('وصله' in model.vocab)
    # print('دیگه‌ای' in model.vocab)
    # print('دیگه' in model.vocab)
    # print('اینو' in model.vocab)
    # print('بعنوان' in model.vocab)
    # print('نادره' in model.vocab)
    # print('بخوايين' in model.vocab)
    # print('میخوایی' in model.vocab)
    # print('وایسا' in model.vocab)
    # print('نوبت' in model.vocab)
    # print('منه' in model.vocab)
    # df = pd.read_excel('/home/mohammadkrb/Desktop/Ahd/InformalToFormal/InformalToFormal/temp/mohavere_analysis.xlsx', names=['informal', 'outv1', 'details'])
    # txts = df['informal'].tolist()
    # output = translate(txts)
    # df['outv2'] = output.values()
    # df.to_excel('mohavere_analysis_new_version.xlsx')
    # print(model.oneshot_transformer.transform('پاهام', None))
    # print(model.oneshot_transformer.transform('دیروزه', None))
    # print(model.oneshot_transformer.transform('دیروزه', None))
    # print(model.oneshot_transformer.transform('خوشبخت', None))
    # data = [
    #
    #     ' . زره ‏ ش رو در بيارين',
    #     'رو من ننداز اونو',
    #     'اون رو بردار از اونجا',
    #     'رو در رو باید صحبت کنیم',
    #     'رو تو نمیشه اصلا حساب کرد',
    #     'کارام رو کردم میام پیشت'
    #     ]
    # print(translate(data))
    # print(model.verb_handler.informal_to_formal('نمیومدن'))
    # print(model.verb_handler.informal_to_formal('ننداز'))
    # print(model.verb_handler.informal_to_formal('وامیسته'))
    # print(model.verb_handler.informal_to_formal('میندازه'))
    # print(model.verb_handler.informal_to_formal('میاره'))
    # print(model.verb_handler.informal_to_formal('نیاره'))
    # print(model.verb_handler.informal_to_formal('نمیاریم'))
    # print(model.verb_handler.informal_to_formal('نیفتن'))
    # print(model.verb_handler.informal_to_formal('نیومدین'))
    # print(model.verb_handler.informal_to_formal('امادن'))
    # print(model.verb_handler.informal_to_formal('میاورن'))
    # print(model.verb_handler.informal_to_formal('میارن'))
    # print(model.verb_handler.informal_to_formal('میای'))
    # print(model.verb_handler.informal_to_formal('میایی'))
    # print(model.verb_handler.informal_to_formal('میایی'))
    # print(model.verb_handler.informal_to_formal('بیا'))
    # print(model.verb_handler.informal_to_formal('بیای'))
    # print(model.verb_handler.informal_to_formal('بیایی'))
    # print('مخ' in model.vocab)
    # ps = Postagger()
    # cant_translate = {}
    # with open('temp/onegram28m.pkl', 'rb') as f:
    #     onegrams = pickle.load(f)
    # # addr = '/home/mohammadkrb/Desktop/Ahd/Seq/short_sentences'
    # # addr = '/home/mohammadkrb/Desktop/Ahd/persianCorpus/informal_corpus_twitter_websites_subtitles_28M'
    # for word, freq in onegrams.items():
    #     if freq > 1000 and not model.oneshot_transformer.transform(word, None):
    #         print(word)
    #         if word not in cant_translate:
    #             cant_translate[word] = freq

    # f = open(addr)
    # for index, txt in enumerate(f):
    #     if len (txt.split()) > 15:
    #         continue
    #     print(index)
    #     if index > 20000:
    #         break
        # translated_txt = translate_short_sent(txt)
        # tokens = translated_txt.split()
        # for t in tokens:
        #     if t not in onegrams:
        #         if t not in bad_translated:
        #             bad_translated[t] = 1
        #         else:
        #             bad_translated[t] += 1
        #         print(txt, ' ******** ', t)
        # txt = txt.replace('\n', '')
        # tokens = txt.split()
        # try:
        #     poses = ps.tag(txt)
        # except :
        #     print(txt)
        # candidates = []
        # for index in range(len(tokens)):
        #     try:
        #         pos = poses[index]
        #     except:
        #         pos = None
        #     tok = tokens[index]
        #     if model.verb_handler.informal_to_formal(tok) and pos and pos != 'VERB' and pos!= 'AUX':
        #         if tok in semi_verbs:
        #             semi_verbs[tok] += 1
        #         else:
        #             semi_verbs[tok] = 1
    #     #         print(tok, pos)
    # cant_translate = sorted([(verb, freq) for verb, freq in cant_translate.items()] , key=lambda x: x[1], reverse=True)
    # with open('temp/cant_translate', 'w+') as f:
    #     f.write('\n'.join([f'{verb}\t{str(freq)}' for verb, freq in cant_translate]))
        # with open('/home/mohammadkrb/Desktop/Ahd/persianCorpus/infromalData/subts_norm_cmts.json') as f:
    #     data = json.load(f)
    # data = data[:100000]
    # shuffle(data)

    # # print(model.verb_handler.informal_to_fo # for i, item in enumerate(data):
    #     #     print(i)
    #     # #     if i > 100:
    #     # #         break
    #     # #     print(item)
    #     #     formal = translate([item])
    #     #     # results.append([item, formal[0]])
    #     # # # print(results)rmal('برداریم'))
    # # exit()
    # # data = [d['content'].replace('\t', '') for d in data if 20>len(d['content'].replace('\t', '') .split()) > 7]

    # df = pd.DataFrame(results, columns=['informal', 'formal'])
    # df.to_csv('sample_subtitles_translated.csv')
