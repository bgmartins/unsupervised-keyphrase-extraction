# Copyright (c) 2017-present, Swisscom (Schweiz) AG.
# All rights reserved.
#
#Authors: Kamil Bennani-Smires, Yann Savary

import numpy as np
import torch
from swisscom_ai.research_keyphrase.embeddings.emb_distrib_interface import EmbeddingDistributor
import sent2vec
from sentence_transformers import SentenceTransformer
import logging
import torch
from torch import nn
import logging
from transformers import AutoModel, AutoConfig, AutoModelForMaskedLM, AutoTokenizer
from nltk.tokenize import sent_tokenize
from swisscom_ai.research_keyphrase.embeddings.utils import aggregate_embedding
#from swisscom_ai.research_keyphrase.embeddings.laser_embed import SentenceEncoder

class EmbeddingDistributorLocal(EmbeddingDistributor):

    """
    Concrete class of @EmbeddingDistributor using a local installation of sent2vec
    https://github.com/epfml/sent2vec    
    """
    def __init__(self, fasttext_model):
        # LASER embeddingss
        #self.model = SentenceEncoder("/home/bgmartins/keyphrases/LASER/models/bilstm.93langs.2018-12-26.pt")
        #self.model_type = 3
        
        # Original implementation with sent2vec
        #self.model = sent2vec.Sent2vecModel()
        #self.model.load_model(fasttext_model)
        #self.model_type = 2
            
        # Any Transformer model can be used here. Examples include:
        #
        # sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens
        # johngiorgi/declutr-small or johngiorgi/declutr-base
        # sentence-transformers/bert-large-nli-mean-tokens , sentence-transformers/roberta-large-nli-stsb-mean-tokens , sentence-transformers/bert-base-nli-stsb-mean-tokens
        # bert-base-multilingual-cased
        # xlm-mlm-100-1280
        # xlm-roberta-large
        # SpanBERT/spanbert-large-cased"
        # albert-xxlarge-v2
        #
        self.model = "sentence-transformers/bert-base-nli-stsb-mean-tokens"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model)
        self.model_type = 1
        
    def seq_in_seq(self, subseq, seq):
        p = 0
        l = [ ]
        seq = list(seq)
        subseq = list(subseq)
        while subseq[0] in seq:
            index = seq.index(subseq[0])
            if subseq == seq[index:index + len(subseq)]: 
                l.append(index + p)
                seq = seq[index + len(subseq):]
                p = p + index + len(subseq)
            else: 
                seq = seq[index + 1:]
                p = p + index + 1
        else: return l

    def get_tokenized_sents_embeddings(self, sents, doc=None):
        """
        @see EmbeddingDistributor
        """        
        saida = [ ]
        sents = [ sent.replace('\n',' ').replace(" 's ", "'s ").replace(" 'll ", "'ll ").replace(" n't ", "n't ").replace(" , ", ", ").replace("  +", " ").strip().lower() for sent in sents ]
        if self.model_type == 2: return self.model.embed_sentences(sents)
        if self.model_type == 3: return self.model.encode_sentences(sents)
        if not(doc is None):
            doc = doc.replace('\n',' ').replace(" 's ", "'s ").replace(" 'll ", "'ll ").replace(" n't ", "n't ").replace(" , ", ", ").replace("  +", " ").strip()
            doc = sent_tokenize(doc)
        for pos, w in enumerate(sents):
            inputs = self.tokenizer( [w] , padding=True, truncation=True, return_tensors="pt", max_length=512)
            with torch.no_grad(): sequence_output, _ = self.model(**inputs, output_hidden_states=True)
            embeddings = (torch.sum( sequence_output * inputs["attention_mask"].unsqueeze(-1), dim=1 ) / torch.clamp(torch.sum(inputs["attention_mask"], dim=1, keepdims=True), min=1e-9)).detach().numpy().copy()
            #with torch.no_grad(): sequence_output = self.model(**inputs, output_hidden_states=True)
            #embeddings = aggregate_embedding(masks=inputs["attention_mask"].detach().numpy()).dissecting(torch.stack(sequence_output[1]).permute(1, 0, 2, 3).numpy()).copy()
            tmp = [ embeddings[0] ]
            if not(doc is None):
                w2 = " \"" + w + "\" "
                for s in doc:
                    if not(w in s): continue
                    s = s.replace(w,w2)
                    inputs = self.tokenizer( [s] , padding=True, truncation=True, return_tensors="pt", max_length=512)
                    inputs_aux = self.tokenizer( [w2] , padding=True, truncation=True, return_tensors="pt", max_length=512).input_ids.detach().numpy()[0][1:-1]
                    inputs_aux2 = inputs.input_ids.detach().numpy()[0]
                    last = 1
                    for aux in self.seq_in_seq( inputs_aux , inputs_aux2 ):
                        for i in range(last,aux): inputs["attention_mask"][0][i] = 0
                        last = aux+len(inputs_aux)
                    for i in range(last, len(inputs_aux2)): inputs["attention_mask"][0][i] = 0
                    with torch.no_grad(): sequence_output, _ = self.model(**inputs, output_hidden_states=True)
                    embeddings = (torch.sum( sequence_output * inputs["attention_mask"].unsqueeze(-1), dim=1 ) / torch.clamp(torch.sum(inputs["attention_mask"], dim=1, keepdims=True), min=1e-9))[0].detach().numpy()
                    #with torch.no_grad(): sequence_output = self.model(**inputs, output_hidden_states=True)
                    #embeddings = aggregate_embedding(masks=inputs["attention_mask"].detach().numpy()).dissecting(torch.stack(sequence_output[1]).permute(1, 0, 2, 3).numpy())[0].copy()
                    if not( np.isnan(embeddings).any() ):
                        tmp.append( embeddings.copy() )
                        tmp.append( tmp[0].copy() )
            saida.append( np.mean(np.array(tmp), axis=0) )
        return saida
