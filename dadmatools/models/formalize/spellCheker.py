# import itertools
import json
import re

# from symspellpy.symspellpy import SymSpell
# import flask

# app = flask.Flask(__name__)

# def load_spell():
#     global ss
#     ss = SymSpell(max_dictionary_edit_distance=2)
#     ss.load_dictionary(corpus='resources/persian_informal_onegram_100_corpus.txt', separator='\t', encoding='utf-8',
#                        term_index=0, count_index=1)
#     ss.load_bigram_dictionary(corpus='resources/persian_informal_bigram_corpus.txt', term_index=0, count_index=1,
#                               separator='\t', encoding='utf-8')
#
#
# @app.route('/spellchecker', methods=['POST'])
# def spell_checking():
#     data = flask.request.json
#     sent = data['sent']
#     out = ss.word_segmentation(sent)
#     print(sent)
#     print(out)
#     return out.corrected_string

class SpellChecker():
    def __init__(self, using_symspell, check_informal_valid, vocab, spell_checker_mapper_addr):
        self.using_symspell = using_symspell
        self.check_informal_valid = check_informal_valid
        self.vocab = vocab
        self.minimal_mapper = json.load(open(spell_checker_mapper_addr))
        if using_symspell:
            raise NotImplementedError("sym spell cant work without bigram and onegrams!")
            # self.ss = SymSpell(max_dictionary_edit_distance=2)
            # self.ss.load_dictionary(corpus=onegram_addr, separator='\t', encoding='utf-8',
            #                    term_index=0, count_index=1)
            # self.ss.load_bigram_dictionary(corpus=bigram_addr, term_index=0, count_index=1,
            #                           separator='\t', encoding='utf-8')

    def remove_repeated_chars(self, w):
        w_one_ch = re.sub(r'(.)\1+', r'\1', w)
        # w_two_ch = re.sub(r'(.)\1+', r'\1\1', w)
        # if w_one_ch != w_two_ch and self.check_informal_valid(w_two_ch, None):
        #     new_w = w_two_ch
        # else:
        #     new_w = w_one_ch
        return w_one_ch
        # current_ch = word[0]
        # l = 0
        # chars = []
        # for i, ch in enumerate(word):
        #     if ch == current_ch:
        #         l += 1
        #         continue
        #     else:
        #         chars.append((current_ch, l))
        #         l = 1
        #         current_ch = ch
        # chars.append((current_ch, l))
        # repeat_chars = [c[0] for c in chars if c[1] > 1]
        # repeat_count = len(repeat_chars)
        # bin_combination = list(itertools.product([0, 1], repeat=repeat_count))
        # all_words = []
        # for item in bin_combination:
        #     must_uniqed_chars = [ch for i, ch in enumerate(repeat_chars) if item[i] == 0]
        #     new_word = word
        #     for ch in must_uniqed_chars:
        #         patt = ch + '{2,}'
        #         new_word = re.sub(patt, ch, new_word)
        #     all_words.append(new_word)
        # return all_words



    def minimal_spell_checking(self, w):
        # excluedes = ['اگه', 'ابرو', 'از']
        if w not in self.vocab and w in self.minimal_mapper:
            w = self.minimal_mapper[w][0]
        return w

    def tiny_spell_checking(self, word):
        # hat
        if word.startswith('ا') and len(word) > 1:
            new_w = 'آ' + word[1:]
            if self.check_informal_valid(new_w, None):
                return new_w
        if word.endswith('ا') and len(word) > 1:
            new_w = word[:-1] + 'اً'
            if self.check_informal_valid(new_w, None):
                return new_w
        if 'ی' in word:
            new_w = word[:word.index('ی')] + 'ئ' + word[word.index('ی')+1:]
            if self.check_informal_valid(new_w, None):
                return new_w

        return word

    def spell_checking(self, txt, pn_words):
        new_tokens = []
        words = txt.split()
        is_valid = lambda w: w in pn_words or  self.check_informal_valid(w, None)
        for w in words:
            if not is_valid(w):
                w = self.remove_repeated_chars(w)
                w = self.minimal_spell_checking(w)
                # w = self.tiny_spell_checking(w)
            new_tokens.append(w)

        corrected_str = ' '.join(new_tokens)
        if self.using_symspell:
            out = self.ss.word_segmentation(txt, is_valid=is_valid)
            corrected_str =  out.corrected_string
        return corrected_str