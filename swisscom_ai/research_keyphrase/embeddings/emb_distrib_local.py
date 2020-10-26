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
from transformers import BertConfig, BertForPreTraining, load_tf_weights_in_bert, RobertaConfig
from transformers import RobertaTokenizer
from .bertkpe import networks, config_class
from transformers import AutoModel, AutoConfig, AutoModelWithLMHead, AutoTokenizer
from nltk.tokenize import sent_tokenize

class EmbeddingDistributorLocal(EmbeddingDistributor):

    """
    Concrete class of @EmbeddingDistributor using a local installation of sent2vec
    https://github.com/epfml/sent2vec
    
    """
    def __init__(self, fasttext_model):
        #self.models = sent2vec.Sent2vecModel()
        #self.models.load_model(fasttext_model)
        #self.model_type = 2
                
        #self.model = AutoModel.from_pretrained("johngiorgi/declutr-small")               
        #self.tokenizer = AutoTokenizer.from_pretrained("johngiorgi/declutr-small")
        #self.model_type = 1
        
        #self.model = AutoModel.from_pretrained("johngiorgi/declutr-base")               
        #self.tokenizer = AutoTokenizer.from_pretrained("johngiorgi/declutr-base")
        #self.model_type = 1
        
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/bert-base-nli-stsb-mean-tokens")
        self.model = AutoModel.from_pretrained("sentence-transformers/bert-base-nli-stsb-mean-tokens")
        self.model_type = 1
        
        #self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/bert-large-nli-mean-tokens")
        #self.model = AutoModel.from_pretrained("sentence-transformers/bert-large-nli-mean-tokens")
        #self.model_type = 1
        
        #self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/roberta-large-nli-stsb-mean-tokens")
        #self.model = AutoModel.from_pretrained("sentence-transformers/roberta-large-nli-stsb-mean-tokens")
        #self.model_type = 1
        
        #self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens")
        #self.model = AutoModel.from_pretrained("sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens")
        #self.model_type = 1

        #self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
        #self.model = AutoModel.from_pretrained("xlm-roberta-large")
        #self.model_type = 1

        #self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        #self.model = AutoModel.from_pretrained("bert-base-multilingual-cased")
        #self.model_type = 1
        
        #self.tokenizer = AutoTokenizer.from_pretrained("xlm-mlm-100-1280")
        #self.model = AutoModel.from_pretrained("xlm-mlm-100-1280")
        #self.model_type = 1        
                
        #self.tokenizer = AutoTokenizer.from_pretrained("albert-xxlarge-v2")
        #self.model = AutoModel.from_pretrained("albert-xxlarge-v2")
        #self.model_type = 1
        
        #self.tokenizer = AutoTokenizer.from_pretrained("SpanBERT/spanbert-large-cased")
        #self.model = AutoModel.from_pretrained("SpanBERT/spanbert-large-cased")
        #self.model_type = 1

        #self.tokenizer = AutoTokenizer.from_pretrained("mrm8488/spanbert-large-finetuned-tacred")
        #self.model = AutoModel.from_pretrained("mrm8488/spanbert-large-finetuned-tacred")
        #self.model_type = 1
        
        #self.model = AutoModel.from_pretrained("/home/bgmartins/keyphrases/burt-base-model/0_BERT")
        #self.tokenizer = AutoTokenizer.from_pretrained("/home/bgmartins/keyphrases/burt-base-model/0_BERT")
        #self.model_type = 1
        
        #tf_checkpoint_path = '/home/bgmartins/keyphrases/bert2joint.openkp/bert2joint.openkp.roberta.checkpoint'
        #bert_config_file = '/home/bgmartins/keyphrases/bert2joint.openkp/config.json'
        #pytorch_dump_output = '/home/bgmartins/keyphrases/bert2joint.openkp/pytorch_model2.bin'
        #filename = tf_checkpoint_path
        #saved_params = torch.load(filename, map_location=lambda storage, loc:storage)
        #args = saved_params['args']
        #epoch = saved_params['epoch']
        #state_dict = saved_params['state_dict']
        #args.model_class = 'bert2joint'
        #network = networks.get_class(args)
        #args.num_labels = 2
        #args.cache_dir = '/home/bgmartins/keyphrases/bert2joint.openkp'
        #model_config = config_class[args.pretrain_model_type].from_pretrained(args.cache_dir, num_labels=args.num_labels)
        #network = network.from_pretrained(args.cache_dir, config=model_config)
        #if state_dict is not None: network.load_state_dict(state_dict)
        #self.model = network.roberta
        #self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        #self.model_type = 1
        
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
        sents = [ sent.replace('\n',' ').replace(" 's ", "'s ").replace(" 'll ", "'ll ").replace(" n't ", "n't ").replace(" , ", ", ").replace("  +", " ").strip() for sent in sents ]
        if self.model_type == 2: return self.models.embed_sentences(sents)
        if not(doc is None):
            doc = doc.replace('\n',' ').replace(" 's ", "'s ").replace(" 'll ", "'ll ").replace(" n't ", "n't ").replace(" , ", ", ").replace("  +", " ").strip()
            doc = sent_tokenize(doc)
        for pos, w in enumerate(sents):
            inputs = self.tokenizer( [w] , padding=True, truncation=True, return_tensors="pt", max_length=512)
            with torch.no_grad(): sequence_output, _ = self.model(**inputs, output_hidden_states=False)
            embeddings = torch.sum( sequence_output * inputs["attention_mask"].unsqueeze(-1), dim=1 ) / torch.clamp(torch.sum(inputs["attention_mask"], dim=1, keepdims=True), min=1e-9)
            tmp = [ embeddings[0].detach().numpy().copy() ]
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
                    with torch.no_grad(): sequence_output, _ = self.model(**inputs, output_hidden_states=False)
                    embeddings = (torch.sum( sequence_output * inputs["attention_mask"].unsqueeze(-1), dim=1 ) / torch.clamp(torch.sum(inputs["attention_mask"], dim=1, keepdims=True), min=1e-9))[0].detach().numpy()
                    if not( np.isnan(embeddings).any() ): 
                        tmp.append( embeddings.copy() )
                        tmp.append( tmp[0].copy() )
                        tmp.append( tmp[0].copy() )
            saida.append( np.mean(np.array(tmp), axis=0) )
        return saida
