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
	#arquivo do dataset ja parametrizado com: nome_do_arquivo|raw_text|keywords_verdadeiras
	#divisao do arquivo do dataset de entrada para acessar os valores
#	data = line.split("|")
	data = line.split("℗")
	id_url = data[0]
	print ("Processando ", id_url)
	print ("Arquivo ",cont_cresc,"/",cont_total)
	#conferenia da blacklist
	if id_url in blacklist:
		print ("URL", id_url, "na Blacklist!" )
		cont_cresc = cont_cresc +1
	else:
		new_raw = data[1]
		print ("TAMANHO DO RAW: ",sum([i.strip(string.punctuation).isalpha() for i in new_raw.split()]))
		#funcao que gera as kp
		kp1 = launch.extract_keyphrases(embedding_distributor, pos_tagger, new_raw, 10000, lang, beta=1.0, alias_threshold=0.9)
		if not kp1[0]:
			#se arquivo nao conseguir fazer a extracao nao sera inserido no dicionario
			print ("url ", id_url, "sem extração.")
			cont_cresc = cont_cresc +1
			print ("-----------------")
		else:
			#salvar KP nos dicionarios criados
			keyphases_extraídas[id_url] =  kp1[0]
			keyphases_verdadeiras[id_url] = data[2].split("!")
			keyphases_verdadeiras[id_url] = [w.replace('\n','') for w in keyphases_verdadeiras[id_url]]
			cont_cresc = cont_cresc +1
			print ("-----------------")
			
		for w in keyphases_verdadeiras[id_url]:
			if w not in new_raw.lower():
				print ("Keyword nao encontrada: ", w)
				print ('\n')
			
#acionamento da funcao de indicadores
print ("-----------------")
eval_metrics(keyphases_verdadeiras, keyphases_extraídas, lang=lang)
