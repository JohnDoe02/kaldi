#!/usr/bin/env python3
#
# This file is part of kaldi-active-grammar.
# (c) Copyright 2019 by David Zurow
# Licensed under the AGPL-3.0; see LICENSE.txt file.
#

import os, re, shutil
from io import open

from six import PY2, text_type
import requests

try:
    # g2p_en==2.0.0
    import g2p_en
except ImportError:
    g2p_en = None

class Lexicon(object):
    def __init__(self, phones):
        """ phones: list of strings, each being a phone """
        self.phone_set = set(self.make_position_independent(phones))

    # XSAMPA phones are 1-letter each, so 2-letter below represent 2 separate phones.
    CMU_to_XSAMPA_dict = {
        "'"   : "'",
        'AA'  : 'A',
        'AE'  : '{',
        'AH'  : 'V',  ##
        'AO'  : 'O',  ##
        'AW'  : 'aU',
        'AY'  : 'aI',
        'B'   : 'b',
        'CH'  : 'tS',
        'D'   : 'd',
        'DH'  : 'D',
        'EH'  : 'E',
        'ER'  : '3',
        'EY'  : 'eI',
        'F'   : 'f',
        'G'   : 'g',
        'HH'  : 'h',
        'IH'  : 'I',
        'IY'  : 'i',
        'JH'  : 'dZ',
        'K'   : 'k',
        'L'   : 'l',
        'M'   : 'm',
        'NG'  : 'N',
        'N'   : 'n',
        'OW'  : 'oU',
        'OY'  : 'OI', ##
        'P'   : 'p',
        'R'   : 'r',
        'SH'  : 'S',
        'S'   : 's',
        'TH'  : 'T',
        'T'   : 't',
        'UH'  : 'U',
        'UW'  : 'u',
        'V'   : 'v',
        'W'   : 'w',
        'Y'   : 'j',
        'ZH'  : 'Z',
        'Z'   : 'z',
    }
    CMU_to_XSAMPA_dict.update({'AX': '@'})
    del CMU_to_XSAMPA_dict["'"]
    XSAMPA_to_CMU_dict = { v: k for k,v in CMU_to_XSAMPA_dict.items() }  # FIXME: handle double-entries

    @classmethod
    def cmu_to_xsampa_generic(cls, phones, lexicon_phones=None):
        new_phones = []
        for phone in phones:
            stress = False
            if phone.endswith('1'):
                phone = phone[:-1]
                stress = True
            elif phone.endswith(('0', '2')):
                phone = phone[:-1]
            phone = cls.CMU_to_XSAMPA_dict[phone]
            assert 1 <= len(phone) <= 2

            new_phone = ("'" if stress else '') + phone
            if (lexicon_phones is not None) and (new_phone in lexicon_phones):
                # Add entire possibly-2-letter phone
                new_phones.append(new_phone)
            else:
                # Add each individual 1-letter phone
                for match in re.finditer(r"('?).", new_phone):
                    new_phones.append(match.group(0))

        return new_phones

    def cmu_to_xsampa(self, phones):
        return self.cmu_to_xsampa_generic(phones, self.phone_set)

    @classmethod
    def make_position_dependent(cls, phones):
        if len(phones) == 0: return []
        elif len(phones) == 1: return [phones[0]+'_S']
        else: return [phones[0]+'_B'] + [phone+'_I' for phone in phones[1:-1]] + [phones[-1]+'_E']

    @classmethod
    def make_position_independent(cls, phones):
        return [re.sub(r'_[SBIE]', '', phone) for phone in phones]

    g2p_en = None

    @classmethod
    def generate_pronunciations(cls, word):
        """returns CMU/arpabet phones"""
        if g2p_en:
            try:
                if not cls.g2p_en:
                    cls.g2p_en = g2p_en.G2p()
                phones = cls.g2p_en(word)
            #    print("generated pronunciation with g2p_en for %r: %r" % (word, phones))
                return phones
            except Exception as e:
                print("generate_pronunciations exception using g2p_en")

        if True:
            try:
                files = {'wordfile': ('wordfile', word)}
                req = requests.post('http://www.speech.cs.cmu.edu/cgi-bin/tools/logios/lextool.pl', files=files)
                req.raise_for_status()
                # FIXME: handle network failures
                match = re.search(r'<!-- DICT (.*)  -->', req.text)
                if match:
                    url = match.group(1)
                    req = requests.get(url)
                    req.raise_for_status()
                    entries = req.text.strip().split('\n')
                    pronunciations = []
                    for entry in entries:
                        tokens = entry.strip().split()
                        assert re.match(word + r'(\(\d\))?', tokens[0], re.I)  # 'SEMI-COLON' or 'SEMI-COLON(2)'
                        phones = tokens[1:]
                        print("generated pronunciation with cloud-cmudict for %r: CMU phones are %r" % (word, phones))
                        pronunciations.append(phones)
                    return pronunciations[0]
            except Exception as e:
                print("generate_pronunciations exception accessing www.speech.cs.cmu.edu")

        raise 

def readLexicon(filename):
    words = set()
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            words.add(line.split(" ")[0])

    return words

def readCorpus(corpus):
    words = set()
    with open(corpus, "r") as f:
        lines = f.readlines()
        for line in lines:
            words.update(line.split())

    return words
    

words_lexicon = readLexicon("kaldi_model/lexicon.txt")
words_corpus = readCorpus("data/train/corpus.txt")
diff = words_corpus.difference(words_lexicon)

lexicon = Lexicon([])
for word in diff:
    print(word, " ".join(lexicon.cmu_to_xsampa(lexicon.generate_pronunciations(word.replace("'", "")))))

