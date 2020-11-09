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
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoConfig, AutoModelForMaskedLM, AutoTokenizer
from nltk.tokenize import sent_tokenize
from torch.utils.data.dataset import Dataset
from typing import Dict

class LineByLineTextDataset(Dataset):

    def __init__(self, tokenizer, data, block_size):
        lines = [line for line in data if (len(line) > 0 and not line.isspace())]
        batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=block_size)
        self.examples = batch_encoding["input_ids"]
        self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]
        
class EmbeddingDistributorLocal(EmbeddingDistributor):

    """
    Concrete class of @EmbeddingDistributor using a local installation of sent2vec
    https://github.com/epfml/sent2vec    
    """
    def __init__(self, fasttext_model):        
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
        self.model = AutoModel.from_pretrained(self.model)
        self.model_type = 1
    
    def finetune_model(self, doc, keyphrases ):
        dataset = LineByLineTextDataset( tokenizer = self.tokenizer, data = sent_tokenize(doc) + keyphrases, block_size = 16 )
        data_collator = transformers.DataCollatorForLanguageModeling( tokenizer = tokenizer, mlm = True, mlm_probability = 0.15 )
        training_args = transformers.TrainingArguments( num_train_epochs = 5, per_device_train_batch_size = 2, do_train = True )
        trainer = transformers.Trainer( model = self.model, args = training_args, data_collator = data_collator, train_dataset = dataset, prediction_loss_only = True )
        trainer.train()

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

    def check_within(self, w, s):
        if s.startswith(w + " "): return True
        if s.startswith(w + "."): return True
        if s.startswith(w + "!"): return True
        if s.startswith(w + "?"): return True
        if s.startswith(w + ","): return True
        if s.startswith(w + ";"): return True
        if s.endswith(" " + w + "."): return True
        if s.endswith(" " + w + "!"): return True
        if s.endswith(" " + w + "?"): return True
        if s.endswith(" " + w + ","): return True
        if s.endswith(" " + w + ";"): return True
        if s.endswith(" " + w): return True
        if " " + w + " " in s: return True
        if " " + w + "," in s: return True
        if " " + w + ";" in s: return True
        if " " + w + "." in s: return True
        if " " + w + "!" in s: return True
        if " " + w + "?" in s: return True
        return False
        
    def get_tokenized_sents_embeddings(self, sents, doc=None):
        """
        @see EmbeddingDistributor
        """        
        sents = [ sent.replace('\n',' ').replace(" 's ", "'s ").replace(" 'll ", "'ll ").replace(" n't ", "n't ").replace(" , ", ", ").replace("  +", " ").strip().lower() for sent in sents ]
        if self.model_type == 2: return self.model.embed_sentences(sents)
        doc_embedd = None
        saida = [ ]
        if not(doc is None):
            doc_embedd = self.get_tokenized_sents_embeddings( [doc] )
            doc = doc.replace('\n',' ').replace(" 's ", "'s ").replace(" 'll ", "'ll ").replace(" n't ", "n't ").replace(" , ", ", ").replace("  +", " ").strip()
            doc = sent_tokenize(doc)
        for pos, w in enumerate(sents):
            inputs = self.tokenizer( [w] , padding=True, truncation=True, return_tensors="pt", max_length=512)
            with torch.no_grad(): sequence_output, _ = self.model(**inputs, output_hidden_states=False)
            embeddings = (torch.sum( sequence_output * inputs["attention_mask"].unsqueeze(-1), dim=1 ) / torch.clamp(torch.sum(inputs["attention_mask"], dim=1, keepdims=True), min=1e-9)).detach().numpy()
            tmp = [ embeddings[0].copy() ]
            tmp_weights = [ 1.0 ]
            if doc is None:
                w2 = sent_tokenize(w)
                if len(w) > 2:
                    for pos2, s in enumerate(w2):
                        inputs = self.tokenizer( [s] , padding=True, truncation=True, return_tensors="pt", max_length=512)
                        with torch.no_grad(): sequence_output, _ = self.model(**inputs, output_hidden_states=False)
                        embeddings = (torch.sum( sequence_output * inputs["attention_mask"].unsqueeze(-1), dim=1 ) / torch.clamp(torch.sum(inputs["attention_mask"], dim=1, keepdims=True), min=1e-9)).detach().numpy()
                        tmp.append( embeddings[0].copy() )
                        tmp_weights.append( 1.0 / ( 2 + pos2 ) )
            else:
                w2 = " '" + w + "' "    
                s2 = [ s for s in doc if self.check_within( w , s ) ]
                for pos2, s in enumerate(s2):
                    s = s.replace(w,w2)
                    inputs = self.tokenizer( [s] , padding=True, truncation=True, return_tensors="pt", max_length=512)
                    inputs_aux = self.tokenizer( [w2] , padding=True, truncation=True, return_tensors="pt", max_length=512).input_ids.detach().numpy()[0][1:-1]
                    inputs_aux2 = inputs.input_ids.detach().numpy()[0]
                    with torch.no_grad(): sequence_output, _ = self.model(**inputs, output_hidden_states=False)
                    embeddings_doc = (torch.sum( sequence_output * inputs["attention_mask"].unsqueeze(-1), dim=1 ) / torch.clamp(torch.sum(inputs["attention_mask"], dim=1, keepdims=True), min=1e-9)).detach().numpy()
                    last = 1
                    for aux in self.seq_in_seq( inputs_aux , inputs_aux2 ):
                        for i in range(last,aux): inputs["attention_mask"][0][i] = 0
                        last = aux+len(inputs_aux)
                    for i in range(last, len(inputs_aux2)): inputs["attention_mask"][0][i] = 0
                    embeddings = (torch.sum( sequence_output * inputs["attention_mask"].unsqueeze(-1), dim=1 ) / torch.clamp(torch.sum(inputs["attention_mask"], dim=1, keepdims=True), min=1e-9)).detach().numpy()
                    if not( np.isnan(embeddings[0]).any() ): 
                        tmp.append( embeddings[0].copy() )
                        weight = cosine_similarity(embeddings_doc, doc_embedd)
                        tmp_weights.append( weight[0][0] * ( 0.5 / len(s2)) )
            saida.append( np.average(np.array(tmp), weights=tmp_weights, axis=0) )
        return saida
