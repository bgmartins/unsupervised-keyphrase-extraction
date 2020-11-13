#!/usr/bin/env python
# -*- coding: utf-8 -*-

import launch
import os
import codecs
import numpy as np
from nltk.stem import PorterStemmer
from swisscom_ai.research_keyphrase.util.evaluation import eval_metrics
import string

#set de variaveis de entrada -tipo do dataset e lingagem
tipo_dataset = "DUC"
lang = 'en'

#pasta onde estao localizados os arquivos
dataset = './datasets/' + tipo_dataset + '/dataset.txt'

#gerar listas para comparacao na rotina de indicadores
keyphases_verdadeiras = {}
keyphases_extraídas = {}

#load dos pretrained de embeding e postagging
embedding_distributor = launch.load_local_embedding_distributor()
pos_tagger = launch.load_local_corenlp_pos_tagger()

#caso datase possua blacklist de arquivos indevidos os mesmos devem ser inseridos aqui
blacklist = ['https://www.goodtherapy.org/blog/married-with-undiagnosed-autism-why-women-who-leave-lose-twice-0420164','https://www.history.navy.mil/research/library/online-reading-room/title-list-alphabetically/w/war-damage-reports/uss-enterprise-cv6-war-history-1941-1945.html','https://www.iwillteachyoutoberich.com/blog/how-to-start-an-online-business/','https://www.judiciaryreport.com/bvc','https://www.judiciaryreport.com/']

#leitura do arquivo
with open(dataset, 'r') as arquivo: raw_text=arquivo.readlines()
	
#contagem dos arquivos para gerar porcentagem de processamento no print de tela
cont_total = len(raw_text)
cont_cresc = 1

for line in raw_text:
#	data = line.split("|")
	data = line.split("℗")
	id_url = data[0]
	print ("Processing", id_url)
	print ("Document number", cont_cresc,"/",cont_total)
	if id_url in blacklist:
		print ("Document", id_url, "is blacklisted!" )
		cont_cresc = cont_cresc +1
	else:
		new_raw = data[1]
		print ("Document size is",sum([i.strip(string.punctuation).isalpha() for i in new_raw.split()]))
		kp1 = launch.extract_keyphrases(embedding_distributor, pos_tagger, new_raw, 10000, lang, beta=1.0, alias_threshold=0.9)
		if not kp1[0]:
			print ("Document", id_url, "without any keyphrases!")
			cont_cresc = cont_cresc +1
		else:
			keyphases_extraídas[id_url] =  kp1[0]
			keyphases_verdadeiras[id_url] = data[2].split("!")
			keyphases_verdadeiras[id_url] = [w.replace('\n','') for w in keyphases_verdadeiras[id_url]]
			cont_cresc = cont_cresc +1
		for w in keyphases_verdadeiras[id_url]:
			if w not in new_raw.lower():
				print ("Candidate keyphrase not found in the text:", w)
		print ("-----------------")
print ("-----------------")
eval_metrics(keyphases_verdadeiras, keyphases_extraídas, lang=lang)
