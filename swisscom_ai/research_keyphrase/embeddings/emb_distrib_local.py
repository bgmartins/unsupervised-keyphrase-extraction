# Copyright (c) 2017-present, Swisscom (Schweiz) AG.
# All rights reserved.
#
#Authors: Kamil Bennani-Smires, Yann Savary

import numpy as np
import tempfile
import re
import torch
import sent2vec
import logging
import torch
import logging
import transformers
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoConfig, AutoTokenizer, AutoModelForMaskedLM
from nltk.tokenize import sent_tokenize
from torch.utils.data.dataset import Dataset
from typing import Dict

class LineByLineTextDataset(Dataset):

    def __init__(self, tokenizer, data, block_size):
        lines = [ line.strip() for line in data if len(line.strip()) > 0 ]
        batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=block_size)
        self.examples = batch_encoding["input_ids"]
        self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]
        
class EmbeddingDistributorLocal:

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
        self.model_name = "sentence-transformers/bert-base-nli-stsb-mean-tokens"
        #self.model_name = "sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model_type = 1
    
    def finetune_model(self, doc, keyphrases ):
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        dataset = sent_tokenize(doc)
        for s in keyphrases: dataset.append(str(s))
        dataset = [ sent.replace('\n',' ').replace(" 's ", "'s ").replace(" 'll ", "'ll ").replace(" n't ", "n't ").replace(" , ", ", ").replace("  +", " ").strip() for sent in dataset ]
        directory = tempfile.TemporaryDirectory(suffix="transformer")
        dataset = LineByLineTextDataset( tokenizer = self.tokenizer, data = dataset, block_size = 16 )
        data_collator = transformers.DataCollatorForLanguageModeling( tokenizer = self.tokenizer, mlm = True, mlm_probability = 0.15 )
        training_args = transformers.TrainingArguments( num_train_epochs = 2, per_device_train_batch_size = 2, do_train = True, output_dir = directory.name, prediction_loss_only = True )
        trainer = transformers.Trainer( model = self.model, args = training_args, data_collator = data_collator, train_dataset = dataset )
        trainer.train()
        trainer.save_model()
        self.model = AutoModel(directory.name)
        directory.cleanup()

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
        
    def get_doc_masked_embedding(self, sents, doc):
        sents = [ sent.replace('\n',' ').replace(" 's ", "'s ").replace(" 'll ", "'ll ").replace(" n't ", "n't ").replace(" , ", ", ").replace("  +", " ").strip() for sent in sents ]
        doc = doc.replace('\n',' ').replace(" 's ", "'s ").replace(" 'll ", "'ll ").replace(" n't ", "n't ").replace(" , ", ", ").replace("  +", " ").strip()
        for s in sents:
            doc = doc.replace(s,"[MASK]")
            #pattern = re.compile(s, re.IGNORECASE)
            #doc = pattern.sub("[MASK]",doc)
        inputs = self.tokenizer( [doc] , padding=True, truncation=True, return_tensors="pt", max_length=512)
        with torch.no_grad(): sequence_output = self.model(**inputs, output_hidden_states=True)
        sequence_output = torch.cat((sequence_output[2][-1], sequence_output[2][-3]), -1)
        embeddings = (torch.sum( sequence_output * inputs["attention_mask"].unsqueeze(-1), dim=1 ) / torch.clamp(torch.sum(inputs["attention_mask"], dim=1, keepdims=True), min=1e-9)).detach().numpy()
        tmp = [ embeddings[0].copy() ]
        tmp_weights = [ 1.0 ]
        w2 = sent_tokenize(doc)
        for pos2, s in enumerate(w2):
            inputs = self.tokenizer( [s] , padding=True, truncation=True, return_tensors="pt", max_length=512)
            with torch.no_grad(): sequence_output = self.model(**inputs, output_hidden_states=True)
            sequence_output = torch.cat((sequence_output[2][-1], sequence_output[2][-3]), -1)
            embeddings = (torch.sum( sequence_output * inputs["attention_mask"].unsqueeze(-1), dim=1 ) / torch.clamp(torch.sum(inputs["attention_mask"], dim=1, keepdims=True), min=1e-9)).detach().numpy()
            tmp.append( embeddings[0].copy() )
            tmp_weights.append( 1.0 / ( 2 + pos2 ) )
        return [ np.average(np.array(tmp), weights=tmp_weights, axis=0) ]
    
    def get_tokenized_sents_embeddings(self, sents, doc=None):
        """
        @see EmbeddingDistributor
        """        
        sents = [ sent.replace('\n',' ').replace(" 's ", "'s ").replace(" 'll ", "'ll ").replace(" n't ", "n't ").replace(" , ", ", ").replace("  +", " ").strip() for sent in sents ]
        if self.model_type == 2: 
            sents = [ sent.lower() for sent in sents ]
            return self.model.embed_sentences(sents)
        doc_embedd = None
        saida = [ ]
        if not(doc is None):
            doc_embedd = self.get_tokenized_sents_embeddings( [doc] )
            doc = doc.replace('\n',' ').replace(" 's ", "'s ").replace(" 'll ", "'ll ").replace(" n't ", "n't ").replace(" , ", ", ").replace("  +", " ").strip()
            doc = sent_tokenize(doc)
                    
        for pos, w in enumerate(sents):
            inputs = self.tokenizer( [w] , padding=True, truncation=True, return_tensors="pt", max_length=512)
            with torch.no_grad(): sequence_output = self.model(**inputs, output_hidden_states=True)
            #sequence_output = sequence_output[2][-1]
            sequence_output = torch.cat((sequence_output[2][-1], sequence_output[2][-3]), -1)            
            embeddings = (torch.sum( sequence_output * inputs["attention_mask"].unsqueeze(-1), dim=1 ) / torch.clamp(torch.sum(inputs["attention_mask"], dim=1, keepdims=True), min=1e-9)).detach().numpy()
            tmp = [ embeddings[0].copy() ]
            tmp_weights = [ 1.0 ]

            if doc is None:
                for pos2, s in enumerate(sent_tokenize(w)):
                    inputs = self.tokenizer( [s] , padding=True, truncation=True, return_tensors="pt", max_length=512)
                    with torch.no_grad(): sequence_output = self.model(**inputs, output_hidden_states=True)
                    #sequence_output = sequence_output[2][-1]
                    sequence_output = torch.cat((sequence_output[2][-1], sequence_output[2][-3]), -1)
                    embeddings = (torch.sum( sequence_output * inputs["attention_mask"].unsqueeze(-1), dim=1 ) / torch.clamp(torch.sum(inputs["attention_mask"], dim=1, keepdims=True), min=1e-9)).detach().numpy()
                    tmp.append( embeddings[0].copy() )
                    tmp_weights.append( 1.0 / ( 2 + pos2 ) )
            else:
                w2 = " \"" + w + "\" "
                s2 = [ s for s in doc if self.check_within( w , s ) ]
                for pos2, s in enumerate(s2):
                    inputs = self.tokenizer( [s] , padding=True, truncation=True, return_tensors="pt", max_length=512)
                    with torch.no_grad(): sequence_output = self.model(**inputs, output_hidden_states=True)
                    #sequence_output = sequence_output[2][-1]
                    sequence_output = torch.cat((sequence_output[2][-1], sequence_output[2][-3]), -1)
                    embeddings_sent = (torch.sum( sequence_output * inputs["attention_mask"].unsqueeze(-1), dim=1 ) / torch.clamp(torch.sum(inputs["attention_mask"], dim=1, keepdims=True), min=1e-9)).detach().numpy().copy()
                    s = s.replace(w, w2)
                    inputs = self.tokenizer( [s] , padding=True, truncation=True, return_tensors="pt", max_length=512)
                    inputs_aux = self.tokenizer( [w2] , padding=True, truncation=True, return_tensors="pt", max_length=512).input_ids.detach().numpy()[0][1:-1]
                    with torch.no_grad(): sequence_output = self.model(**inputs, output_hidden_states=True)
                    last = 1
                    for aux in self.seq_in_seq( inputs_aux , inputs.input_ids.detach().numpy()[0] ):
                        for i in range(last,aux): inputs["attention_mask"][0][i] = 0
                        last = aux + len(inputs_aux)
                    for i in range(last, len(inputs.input_ids.detach().numpy()[0])): inputs["attention_mask"][0][i] = 0
                    #sequence_output = sequence_output[2][-1]
                    sequence_output = torch.cat((sequence_output[2][-1], sequence_output[2][-3]), -1)
                    embeddings = (torch.sum( sequence_output * inputs["attention_mask"].unsqueeze(-1), dim=1 ) / torch.clamp(torch.sum(inputs["attention_mask"], dim=1, keepdims=True), min=1e-9)).detach().numpy()
                    if not( np.isnan(embeddings[0]).any() ): 
                        tmp.append( embeddings[0].copy() )
                        weight = cosine_similarity(embeddings_sent, doc_embedd)
                        tmp_weights.append( weight[0][0] * ( 0.5 / len(s2)) )
            saida.append( np.average(np.array(tmp), weights=tmp_weights, axis=0) )
        return saida
