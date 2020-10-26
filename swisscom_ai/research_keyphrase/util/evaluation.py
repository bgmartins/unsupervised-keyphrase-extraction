import numpy as np

from nltk.stem import PorterStemmer

def eval_metrics(keyphases_verdadeiras, keyphases_extraídas, lang="en"):
	lowercase = True
	stemming = True
	porter = PorterStemmer()
	resultados = { }
	cont_semcop = 0	
	for doc, keyphase_extraída in keyphases_extraídas.items():
		keyphase_verdadeira = keyphases_verdadeiras[doc]
		if stemming:
			keyphase_extraída = [ " ".join([ porter.stem(k) for k in ks.split(" ") ]) for ks in keyphase_extraída ]
			keyphase_verdadeira = [ " ".join([ porter.stem(k) for k in ks.split(" ") ]) for ks in keyphase_verdadeira ]
		if lowercase:
			keyphase_extraída = [ k.lower() for k in keyphase_extraída ]
			keyphase_verdadeira = [ k.lower() for k in keyphase_verdadeira ]
			
		if len( [ k for k in keyphase_extraída if k in keyphase_verdadeira ]) == 0: cont_semcop += 1
		
		if len( keyphase_extraída ) > 0: 
			p = len( [ k for k in keyphase_extraída if k in keyphase_verdadeira ] ) / float( len( keyphase_extraída ) )
			p1 = len( [ k for k in keyphase_extraída[:1] if k in keyphase_verdadeira ] ) / float( len( keyphase_extraída[:1] ) )
			p5 = len( [ k for k in keyphase_extraída[:5] if k in keyphase_verdadeira ] ) / float( len( keyphase_extraída[:5] ) )
			p10 = len( [ k for k in keyphase_extraída[:10] if k in keyphase_verdadeira ] ) / float( len( keyphase_extraída[:10] ) )
			p15 = len( [ k for k in keyphase_extraída[:15] if k in keyphase_verdadeira ] ) / float( len( keyphase_extraída[:15] ) )
		else: p = p1 = p5 = p10 = p15 = 0.0
		if len( keyphase_verdadeira ) > 0: 
			r = len( [ k for k in keyphase_extraída if k in keyphase_verdadeira ] ) / float( len( keyphase_verdadeira ) )
			r1 = len( [ k for k in keyphase_extraída[:1] if k in keyphase_verdadeira ] ) / float( len( keyphase_verdadeira ) )
			r5 = len( [ k for k in keyphase_extraída[:5] if k in keyphase_verdadeira ] ) / float( len( keyphase_verdadeira ) ) 
			r10 = len( [ k for k in keyphase_extraída[:10] if k in keyphase_verdadeira ] ) / float( len( keyphase_verdadeira ) )
			r15 = len( [ k for k in keyphase_extraída[:15] if k in keyphase_verdadeira ] ) / float( len( keyphase_verdadeira ) )
		else: r = r1 = r5 = r10 = r15 = 0.0
		if p>0 or r >0: f = ( 2.0 * p * r ) / ( p + r )
		else: f = 0.0
		if p1>0 or r1>0: f1 = ( 2.0 * p1 * r1 ) / ( p1 + r1 )
		else: f1=0
		if p5>0 or r5 >0: f5 = ( 2.0 * p5 * r5 ) / ( p5 + r5 )
		else: f5=0
		if p10>0 or r10>0: f10 = ( 2.0 * p10 * r10 ) / ( p10 + r10 )
		else: f10=0
		if p15>0 or r15>0: f15 = ( 2.0 * p15 * r15 ) / ( p15 + r15 )	
		else: f15=0
		j = len( [ k for k in keyphase_extraída if k in keyphase_verdadeira ] ) / float( len( set ( keyphase_extraída + keyphase_verdadeira ) ) )			
		if len( keyphase_extraída ) > 0:
			rp = len( [ k for k in keyphase_extraída[:len(keyphase_verdadeira)] if k in keyphase_verdadeira ] ) / float( len( keyphase_extraída[:len(keyphase_verdadeira)] ) )
			ap = [ len( [ k for k in keyphase_extraída[:p] if k in keyphase_verdadeira ] ) / float( p ) for p in range(1,len(keyphase_extraída) + 1) if keyphase_extraída[p - 1] in keyphase_verdadeira ]
			ap = np.sum(ap) / float( len( keyphase_verdadeira ) )
			ndcg = np.sum( [ 1.0 / np.log2(p + 1) for p in range(1,len(keyphase_extraída) + 1) if keyphase_extraída[p - 1] in keyphase_verdadeira ] )
			ndcg = ndcg / np.sum( [ 1.0 / np.log2(p + 1) for p in range(1,len(keyphase_verdadeira) + 1) ] )	
		else: rp = ap = ndcg = 0
		resultados[doc] = [ p , r, f, j , p1 , p5 , p10 , p15 , r1 , r5 , r10 , r15 , f1 , f5 , f10 , f15 , rp , ap, ndcg ]	
	print('-------------------------------------')
	print("Precision Candidates - " + repr( np.mean( [ v[0] for k, v in resultados.items() ] ) ) )
	print("Recall Candidates - " + repr( np.mean( [ v[1] for k, v in resultados.items() ] ) ) )
	print("F1-Score Candidates - " + repr( np.mean( [ v[2] for k, v in resultados.items() ] ) ) )
	print("Jaccard - " + repr( np.mean( [ v[3] for k, v in resultados.items() ] ) ) )
	print("Precision@1 - " + repr( np.mean( [ v[4] for k, v in resultados.items() ] ) ) )
	print("Precision@5 - " + repr( np.mean( [ v[5] for k, v in resultados.items() ] ) ) )
	print("Precision@10 - " + repr( np.mean( [ v[6] for k, v in resultados.items() ] ) ) )
	print("Precision@15 - " + repr( np.mean( [ v[7] for k, v in resultados.items() ] ) ) )
	print("Recall@1 - " + repr( np.mean( [ v[8] for k, v in resultados.items() ] ) ) )
	print("Recall@5 - " + repr( np.mean( [ v[9] for k, v in resultados.items() ] ) ) )
	print("Recall@10 - " + repr( np.mean( [ v[10] for k, v in resultados.items() ] ) ) )
	print("Recall@15 - " + repr( np.mean( [ v[11] for k, v in resultados.items() ] ) ) )
	print("F1-Score@1 - " + repr( np.mean( [ v[12] for k, v in resultados.items() ] ) ) )
	print("F1-Score@5 - " + repr( np.mean( [ v[13] for k, v in resultados.items() ] ) ) )
	print("F1-Score@10 - " + repr( np.mean( [ v[14] for k, v in resultados.items() ] ) ) )
	print("F1-Score@15 - " + repr( np.mean( [ v[15] for k, v in resultados.items() ] ) ) )
	print("R-Precision - " + repr( np.mean( [ v[16] for k, v in resultados.items() ] ) ) )
	print("Mean Average Precision - " + repr( np.mean( [ v[17] for k, v in resultados.items() ] ) ) )
	print("Normalized Discounted Cumulative Gain - " + repr( np.mean( [ v[18] for k, v in resultados.items() ] ) ) )
	print('Documents without any correct key-phrase extracted: ', cont_semcop)
	print('-------------------------------------')
