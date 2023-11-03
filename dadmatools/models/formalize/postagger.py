import stanza
import flask
import json


# app = flask.Flask(__name__)

# c
# def load_model():
#     global nlp
#     nlp = stanza.Pipeline('fa') # initialize persian neural pipeline
#
# # @app.route('/postagging', methods=['POST'])
# def pos_tagging():
#     # data = flask.request.json
#     # sent = data['sent']
#     sent = 'اونام آمادن واسه همچین روزایی'
#     doc = nlp(sent) # run annotation over a sentence
#     print(*[f'word: {word.text}\tupos: {word.upos}\txpos: {word.xpos}\tfeats: {word.feats if word.feats else "_"}' for
#             sent in doc.sentences for word in sent.words], sep='\n')
#
#
# if __name__ == '__main__':
#     load_model()
#     d = pos_tagging()
#     print(d)
#     # app.run(port=5003)

class Postagger:
    def __init__(self):
        try:
            self.nlp = stanza.Pipeline('fa')
        except:
            stanza.download('fa')
            self.nlp = stanza.Pipeline('fa')
    def tag(self, txt):
        doc = self.nlp(txt)  # run annotation over a sentence

        poses = {}
        index = 0
        for sent in doc.sentences:
            for word in sent.words:
                poses[index] = word.upos
                index += 1
        return poses


if __name__ == '__main__':
    ps = Postagger()
    tokens = txt.split()
    poses = ps.tag(txt)
    candidates = []
    for index in range(len(tokens)):
        try:
            pos = poses[index + 1]
        except:
            pos = None
        tok = tokens[index]
        if is_verb(tok) and pos != 'VERB':
            print(tok)