# Copyright (c) 2017-present, Swisscom (Schweiz) AG.
# All rights reserved.
#
#Authors: Kamil Bennani-Smires, Yann Savary

"""Contain method that return list of candidate"""

import re
import nltk
import spacy

nlp = spacy.load("en_core_web_sm", entity=False)

GRAMMAR_EN = """  NP:
        {<PROPN|NOUN|ADJ>*<PROPN|NOUN>+<ADJ>*}  
        """

GRAMMAR_DE = """
NBAR:
        {<JJ|CARD>*<NN.*>+}  # [Adjective(s) or Article(s) or Posessive pronoun](optional) + Noun(s)
        {<NN>+<PPOSAT><JJ|CARD>*<NN.*>+}
        {<NOUN.*|ADJ>*<NOUN.*>+<ADJ>*}
        {<NOUN.*|ADJ>*<NOUN.*>+<ADJ>*}
        

NP:
{<NBAR><APPR|APPRART><ART>*<NBAR>}# Above, connected with APPR and APPART (beim vom)
{<NBAR>+}

        {<NN.*|JJ>*<NN.*>}  # Adjective(s)(optional) + Noun(s)        
        """

GRAMMAR_FR = """  NP:
         # Adjective(s)(optional) + Noun(s) + Adjective(s)(optional)
        {<NN.*|JJ>*<NN.*>+<JJ>*} 
        {<NOUN.*|ADJ>*<NOUN.*>+<ADJ>*}
        {<NOUN><PROPN><NOUN>}
        {<NOUN><ADP><NOUN>}
        {"<NOUN>"}
        {<NOUN><ADJ>}
        {<NOUN|PROPN><NOUN|PROPN><NOUN|PROPN>}
        {<NOUN.*|ADJ>*<NOUN.*>} 
        {<NOUN>+<ADJ>*<PREP>*)?<NOUN>+<ADJ>*}
        {<NN.*|JJ>*<NN.*>+<JJ>*}
        {<NOUN.*|ADJ>*<NOUN.*>+<ADJ>*} 
        {<NN.*|JJ>*<NN.*>+<JJ>*} 
        {(<NOUN>+<ADJ>*<PREP>*)?<NOUN>+<ADJ>*}
 # Adjective(s)(optional) + Noun(s)
         """

GRAMMAR_PT = """  NP:
    {(<NOUN>+ <ADJ>* <PREP>*)? <NOUN>+ <ADJ>*}
    """

def get_grammar(lang):
    if lang == 'en':
        grammar = GRAMMAR_EN
    elif lang == 'de':
        grammar = GRAMMAR_DE
    elif lang == 'pt':
        grammar = GRAMMAR_PT
    elif lang == 'fr':
        grammar = GRAMMAR_FR
    else:
        raise ValueError('Language not handled')
    return grammar

def extract_candidates(text_obj, no_subset=False):
    """
    Based on part of speech return a list of candidate phrases
    :param text_obj: Input text Representation see @InputTextObj
    :param no_subset: if true won't put a candidate which is the subset of an other candidate
    :param lang: language (currently en, fr and de are supported)
    :return: list of candidate phrases (string)
    """
    keyphrase_candidate = set()

    np_parser = nltk.RegexpParser(get_grammar(text_obj.lang))  # Noun phrase parser
    trees = np_parser.parse_sents(text_obj.pos_tagged)  # Generator with one tree per sentence       
    for tree in trees:
        for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):  # For each nounphrase
            # Concatenate the token with a space
            keyphrase_candidate.add(' '.join(word for word, tag in subtree.leaves()))
    doc = nlp(text_obj.rawtext)
    labes = ['PERSON','NORP','FAC','ORG','GPE','LOC','PRODUCT','EVENT','WORK_OF_ART','LAW']
    for ent in doc.ents:
        if ent.label_ in labes:
            keyphrase_candidate.add(ent.text.replace('\n',' '))
    keyphrase_candidate = {kp for kp in keyphrase_candidate if len(kp.split()) <= 10}
    if no_subset: keyphrase_candidate = unique_ngram_candidates(keyphrase_candidate)
    else: keyphrase_candidate = list(keyphrase_candidate)
    return keyphrase_candidate

def extract_sent_candidates(text_obj):
    """
    :param text_obj: input Text Representation see @InputTextObj
    :return: list of tokenized sentence (string) , each token is separated by a space in the string
    """
    return [(' '.join(word for word, tag in sent)) for sent in text_obj.pos_tagged]

def unique_ngram_candidates(strings):
    """
    ['machine learning', 'machine', 'backward induction', 'induction', 'start'] ->
    ['backward induction', 'start', 'machine learning']
    :param strings: List of string
    :return: List of string where no string is fully contained inside another string
    """
    results = []
    for s in sorted(set(strings), key=len, reverse=True):
        if not any(re.search(r'\b{}\b'.format(re.escape(s)), r) for r in results): results.append(s)
    return results
