"""
    brief: runs stanza POS-tagger / NER on text
"""

import stanza
import argparse
import io

def tagit(oneline,nlp, analysis):
    doc = nlp(oneline)
    for sentence in doc.sentences: 
        if analysis == 'pos':
            yield [word.text for word in sentence.words], [word.pos for word in sentence.words]
        elif analysis == 'ner':
            yield [word.text for word in sentence.words], [word.ner for word in sentence.tokens]



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',help='input text file',required=True)
    parser.add_argument('--output',help='output for tokens',required=True)
    parser.add_argument('--labels',help='output for label tags', required=True)
    parser.add_argument('--do', help='what analysis to do: [pos|ner]', default='pos')
    opts = parser.parse_args()

    analysis = opts.do
    if analysis == 'pos':
        nlpModel = stanza.Pipeline('en',processors='tokenize,pos', tokenize_batch_size=1000, pos_batch_size=1000)
    elif analysis == 'ner':
        nlpModel = stanza.Pipeline('en',processors='tokenize,ner')
    else:
        raise ValueError('analysis {} unknown'.format(analysis))

    with io.open(opts.input,mode='r',encoding='utf-8') as fin:
        with io.open(opts.output,mode='w', encoding='utf-8') as ftokens:
            with io.open(opts.labels, mode='w', encoding='utf-8') as flabels:
                for l in fin:
                    l = l.strip()
                    if l=='':continue
                    for tokens, labels in tagit(l,nlpModel, analysis):
                        assert len(tokens) == len(labels)
                        print(' '.join(tokens),file=ftokens)
                        print(' '.join(labels), file=flabels)