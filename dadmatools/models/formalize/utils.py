import itertools
import json
import re
import pickle
import string
import numpy as np
import pandas as pd

def add_to_mapper(mapping_list):
    print(len(mapping_list))
    df = pd.read_csv('resources/mapper.csv', delimiter=',', index_col=None)
    print(df.columns)
    for item in mapping_list:
        df = df.append({'formal': item[1], 'informal': item[0]}, ignore_index=True)
    df.to_csv('resources/mapper.csv', index=False)

def select_best_candidate(word, prev_word, candidate_list, next_candidates):
    best_candidate = None
    best_operation = None
    best_score = -1000

    candidate_list, operation_list = candidate_list
    if next_candidates is not None:
        next_candidate_list = next_candidates
    else:
        next_candidate_list = [None]

    for i, candidate in enumerate(candidate_list):

            onegram_score = self.get_word_probability(candidate)
            bigram_score_with_prev = self.bigram_markov_factor(prev_word, candidate)

            bigram_score_next = -1000
            for j, next_word in enumerate(next_candidate_list):

                tmp_score = self.bigram_markov_factor(candidate, next_word)
                if tmp_score > bigram_score_next:
                    bigram_score_next = tmp_score


            score = 1 * onegram_score + 0.7 * bigram_score_with_prev + 0.7 * bigram_score_next
            if score > best_score:
                best_candidate = candidate_list[i]
                best_score = score

    return best_candidate, best_operation

def create_vocab_file(pkl_dictionary, output_addr, limit_size):
    with open(pkl_dictionary, 'rb') as f:
        words_freq = pickle.load(f)
    f_out = open(output_addr, 'a+')
    for i, (w, f) in enumerate(words_freq.items()):
        print(i)
        if f < limit_size:
            continue
        if w in ['\t', '\n', ' ']:
            continue
        w = w.replace('\t', '')
        tag = 'NN'
        line = f'{w}\t{f}\t{tag}\n'
        f_out.write(line)

def create_ongram(corpus_addr, normalizer, tokenizer, output_addr, validator=None):
    onegram_dict = {}
    valid_checked = {}
    i = 0
    with open(corpus_addr) as f:
        for line in f:
            i += 1
            print(i)
            line = normalizer.normalize(line)
            words = tokenizer.tokenize(line)
            for w in words:
                if w in onegram_dict:
                    onegram_dict[w] += 1
                else:
                    # if w not in valid_checked:
                    #     is_valid = validator(w, None) != []
                    #     valid_checked[w] = is_valid
                    # if valid_checked[w]:
                    onegram_dict[w] = 1
    print('count words: ', len(onegram_dict))
    with open(output_addr, 'wb+') as f:
        pickle.dump(onegram_dict, f)


def create_bigram(corpus_addr, normalizer, tokenizer, validator):
    bigrams_dict = {}
    i = 0
    valid_checked = {}
    with open(corpus_addr) as f:
        for line in f:
            i += 1
            print(i)
            line = normalizer.normalize(line)
            words = tokenizer.tokenize(line)
            bigram = list(zip(words, words[1:]))
            for bi in bigram:
                if bi in bigrams_dict:
                    bigrams_dict[bi] += 1
                else:
                    # for b in bi:
                    #    if b not in valid_checked:
                    #         is_valid = validator(b, None) != []
                    #         valid_checked[b] = is_valid
                    # if valid_checked[bi[0]] and valid_checked[bi[1]]:
                        bigrams_dict[bi] = 1
    
    with open('bigram1m_pkl', 'wb+') as f:
        pickle.dump(bigrams_dict, f)
    return bigrams_dict


def extract_non_convertable_words(corpus_addr, tokenizer, normalizer, transformer, output_addr, vocab):
    f = open(corpus_addr)
    non_convertables = {}
    seen_words = set()
    nim_fasele = '‌'
    for i, line in enumerate(f):
        print(i)
        # if i > 500:
        #     break
        line = normalizer.normalize(line)
        tokens = tokenizer.tokenize(line)
        for t in tokens:
        #     if nim_fasele in t:
        #         print(t)
            if t in seen_words:
                if t in non_convertables:
                    non_convertables[t] += 1
            else:
                candidates = transformer.transform(t, None)
                # if not candidates and any(t.startswith(pre) for pre in ['از', 'در', 'چند', 'هر', 'هیچ', 'هم', 'با', 'بی', 'تا', 'و']):
                #     print(t)
                if not candidates:
                    non_convertables[t] = 1
                seen_words.add(t)
    words_count = sorted([(word, count) for word, count in non_convertables.items()], key=lambda item: item[1], reverse=True)
    words_count = [str(word) + ' ########### ' + str(count) for (word, count) in words_count]
    with open(output_addr, 'w+') as f:
        f.write('\n'.join(words_count))


def generate_irrgular_informal_verbs():
    """
    برمیگرده میوفته برمیداره برمیگردونه درمیاره ایستادن نمیومد وامیسته

    اومد
    نیومد
    اومدی
    نیومدی
    میومدی
    نیومده
    یومد
    میومده
    """

    mapping_verbs = []
    past_ends = ['م', 'ی', 'ه', 'یم', 'ین', 'ید', 'ند', '', 'ن']
    neg = ['ن', '']
    pre = ['می', 'ب']
    pre_verbs = [('بر', 'دار'), ('در', 'یار'), ('وا', 'ست'), ('بر', 'گرد'), ('ور', 'دار'), ('بر', 'گشت')]
    extras = ['ن', 'نمی', 'می']
    mapper = {'ه':'د', 'ن': 'ند', 'ین': 'ید', 'ور': 'بر', 'ست':'ایست', 'وا':'', 'یار':'آور'}
    for item in pre_verbs:
        for pe in past_ends:
            for ex in extras:
                p_end = pe
                item0 = item[0]
                item1 = item[1]
                inf = item0 + ex + item1 + p_end
                inf = inf.replace('یی', 'ی')
                if item0 in mapper:
                    item0 = mapper[item0]
                if item1 in mapper:
                    item1 = mapper[item1]
                if p_end in mapper:
                    p_end = mapper[p_end]
                formal = item0 + ex + item1 + p_end
                formal = formal.replace('می', 'می‌')
                formal = formal.replace('نآ', 'نیا')
                mapping_verbs.append([formal, inf])
    bons = ['یومد', 'یوفت']
    v_mapper = {'یومد': 'یامد', 'یوفت': 'افت'}
    verbs = itertools.product(neg, pre, bons, past_ends)
    for v in verbs:
        if v[0] == 'ن' and v[1] == 'ب' or (v[2] == 'یومد' and v[1] == 'ب'):
            continue
        inf = v[0] + v[1] + v[2] + v[3]
        inf = inf.replace('یی', 'ی')
        pe = v[3]
        if pe in mapper:
            pe = mapper[pe]
        formal = v[0] + v[1]  +  '‌' + v_mapper[v[2]] + pe
        formal = formal.replace('ی‌ی', 'ی')
        formal = formal.replace('یا', 'ی‌آ')
        formal = formal.replace('دد', 'ده')
        formal = formal.replace('ب‌ا', 'بی')
        mapping_verbs.append([formal, inf])
    add_to_mapper(mapping_verbs)



def load_vocab(vocab_addr='resources/words.dat'):
    vocab = {}
    with open(vocab_addr, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                word, freq, p_tags = line.strip().split('\t')
                vocab[word] = {'freq': freq, 'tags': p_tags}
            except:
                word = line.strip()
                vocab[word] = {'freq': 1, 'tags': 'NUM'}
    return vocab

def if_connect(word1, word2):
    not_connect_chars = ['ا', 'د', 'ذ', 'ر', 'ز', 'ژ', 'و']
    if any(w =='' for w in [word1, word2]) or word1[-1] in not_connect_chars:
        return True
    return False
def split_conj_words(word, conjs):
    candidates = set()
    sorted_conjs = sorted(conjs, key=lambda x: len(x), reverse=True)
    for c in sorted_conjs:
        indx = word.find(c)
        if indx != -1 and indx in [0, len(word)-1]:
            pre_w = word[:indx]
            next_w = word[indx+len(c) :]
            if if_connect(pre_w, c) and if_connect(c, next_w):
                cnd = ' '.join([pre_w, c, next_w])
                cnd = cnd.strip()
                candidates.add(cnd)
    return list(candidates)


def is_formal_prefixed(word, vocab):
    not_connect_chars = ['ا', 'د', 'ذ', 'ر', 'ز', 'ژ', 'و']
    nim_fasele = '‌'
    m1 = re.match('(.+)های(م|ت|ش|مان|تان|شان)?$', word)
    m2 = re.match('(.+[ا|و|ی])ی(م|ت|ش|مان|تان|شان)$', word)
    m3 = re.match('(.+[^ا^و^ی])(م|ت|ش|مان|تان|شان)$', word)
    m4 = re.match('(.+)(ها)$', word)
    m5 = re.match('(.+[ه|ی]‌)(اش|ام|ات)$', word)
    if m3 or m2:
        prefix_word = list(filter(lambda m: m is not None, [m3, m2]))[0].group(1)
        if prefix_word in vocab:
            return True
    m_fired = list(filter(lambda m: m is not None, [m1, m4, m5]))
    if len(m_fired) > 0:
        # print(word, m_fired[0].groups())
        prefix_word = m_fired[0].group(1)
        if prefix_word[-1] != nim_fasele and prefix_word[-1] not in not_connect_chars:
            return False
        if prefix_word[-1] == nim_fasele and not (prefix_word[:-1] in vocab):
            return False
        if prefix_word[-1] != nim_fasele and not (prefix_word in vocab):
            return False
        return True
    return False


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
    sp_mapper = {hamzeh_y: ['ی'], hat: ['ا'], tanvin: ['ن', 'ا'], nim:['', ' '], hamzeh:['ا', '']}
    special_chars = [hat, tanvin, nim, hamzeh]
    out = {}
    for word in vocab:
        p_words = [word.replace(sp, sp_alt) for sp in special_chars for sp_alt in sp_mapper[sp]]
        spell_errors = []
        p_words = list(set(p_words) - set([word]))
        for pw in p_words:
            if pw in out:
                out[pw].add(word)
            else:
                out[pw] = {word}
    out = {w: list(out[w]) for w in out}
    with open('spell_checker_mapper.json', 'w+', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)



def create_mapper_tanvin_hamze_hat_nim_fasele():
    mapper = {}
    hats_word = open('resources/spell/words_with_hat.txt').read().splitlines()
    nim_words = open('resources/spell/words_with_nim.txt').read().splitlines()
    tanvin_words = open('resources/spell/words_with_tanvin.txt').read().splitlines()
    hat_ch = 'آ'
    nim_fasele = '‌'
    for w in hats_word:
        w_without_h = w.replace(hat_ch, 'ا')
        mapper[w_without_h] = w
    for w in nim_words:
        w_without_nim = w.remove(nim_fasele)
        mapper[w_without_nim] = w
        w_space_instead_nim = w.replace(nim_fasele, ' ')
        mapper[w_space_instead_nim] = w

def extract_lemma_nim_fasele_words(word, vocab):
        prefixs = ['اون']
        postfixs = {'ست': 'است', 'هام':'هایم', 'ام':'ام', 'ها':'ها', 'هامون':'هایمان', 'ترین': 'ترین', 'هایشان':'هایشان'}
        tokens = word.split('‌')
        index = 0
        for i in range(len(tokens)):
            index = i
            if tokens[i] not in prefixs:
                break

        for i in range(len(tokens), 0, -1):
            current_tok = '‌'.join(tokens[index:i])
            if current_tok in vocab or  tokens[i-1] not in postfixs:
                return current_tok
        # tokens_without_pres = [t for t in tokens if t not in prefixs]
        # w_without_pres = '‌'.join(tokens_without_pres)
        # if w_without_pres in vocab:
        #     return w_without_pres
        # tokens_without_pres_posts = [t for t in tokens_without_pres if t not in postfixs]
        # w_without_pres_posts = '‌'.join(tokens_without_pres_posts)
        # return w_without_pres_posts


def if_emoji(text):
    # Wide UCS-4 build
    try:
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

    return oRes.findall(text)

def flexicon2vocab():
    addr = '/home/mohammadkrb/Desktop/Ahd/persianCorpus/Flexicon.xlsx'
    df = pd.read_excel(addr, usecols=[2])
    df.columns = ['word']
    df['freq'] = -1
    df['tag'] = 'None'
    f = open('flexicon_vocab.txt', 'w+', encoding='utf-8')
    for index, row in df.iterrows():
        if index < 116:
            continue
        row = list(map(str, row))
        f.write('\t'.join(row))
        f.write('\n')

if __name__ == '__main__':
    pass
    # addr = '/home/mohammadkrb/Desktop/Ahd/persianCorpus/tmp/hmBlogs-37274-clean-random-posts.txt'
    # with open(addr) as f:
    #     for line in f:
    #         if len(line.split()) <3:
    #              print(line)
    # word_pos = {}
    # with open('resources/UPC-2016.txt') as f:
    #     for line in f:
    #         try:
    #             w, p = line.split('\t')
    #             p = p.replace('\n', '')
    #             if w in word_pos:
    #                 if p in word_pos[w]:
    #                     word_pos[w][p] += 1
    #                 else:
    #                     word_pos[w][p] = 1
    #             else:
    #                 word_pos[w] = {}
    #                 word_pos[w][p] = 1
    #
    #         except :
    #             continue
    # with open('resources/word_pos_UPC.json', 'r', encoding='utf-8') as f:
    #     word_pos = json.load(f)
    #
    #
    #
    #
    # # with open('resources/word_pos_UPC.txt', 'a+', encoding='utf-8') as f:
    # #     for word, ps in word_pos.items():
    # #         freq = sum([f for p, f in ps.items()])
    # #         p_tags_str = ' '.join([p for p, f in ps.items()])
    # #         line = f'{word}\t{freq}\t{p_tags_str}\n'
    # #         f.write(line)
    #

