from transformers import pipeline


class NER:
    def __init__(self):
        self.nlp = pipeline('ner', model="../common/bert-base-parsbert-ner-uncased",
                       tokenizer="../common/bert-base-parsbert-ner-uncased")

    def get_n_words(self, txt):
        results = self.nlp(txt)
        results =  [item['word'] for item in results]
        ref_index = 0
        for i, r in enumerate(results):
            if r.startswith('##'):
                results[ref_index] += r.replace('#', '')
            else:
                ref_index = i

        return [w for w in results if not w.startswith('##')]



if __name__ == '__main__':
    ner = NER()
    res = ner.get_n_words('کدوم یکی از شما خانم ویلکسه؟')
    print(res)