# Copyright (c) 2017-present, Swisscom (Schweiz) AG.
# All rights reserved.
#
#Authors: Kamil Bennani-Smires, Yann Savary

import warnings

import numpy as np
import networkx as nx
import sklearn
import torch
from sklearn.cluster import OPTICS
from sklearn.decomposition import FastICA, PCA, TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from sklearn.preprocessing import normalize
from swisscom_ai.research_keyphrase.model.methods_embeddings import extract_candidates_embedding_for_doc, extract_doc_embedding, extract_sent_candidates_embedding_for_doc, get_doc_mask_embedding

def smooth_l1_distances(X, Y=None, threshold=0.5):
    aux = manhattan_distances(X,Y)
    flag = aux > threshold
    aux = (~flag) * (0.5 * aux ** 2) - (flag) * threshold * (0.5 * threshold - aux)
    return aux

def _MMR(embdistrib, text_obj, candidates, X, beta, N, use_filtered, alias_threshold, alpha=0.0, smoothl1=False):
    """
    Core method using Maximal Marginal Relevance in charge to return the top-N candidates

    :param embdistrib: embdistrib: embedding distributor see @EmbeddingDistributor
    :param text_obj: Input text representation see @InputTextObj
    :param candidates: list of candidates (string)
    :param X: numpy array with the embedding of each candidate in each row
    :param beta: hyperparameter beta for MMR (control tradeoff between informativeness and diversity)
    :param N: number of candidates to extract
    :param use_filtered: if true filter the text by keeping only candidate word before computing the doc embedding
    :return: A tuple with 3 elements :
    1)list of the top-N candidates (or less if there are not enough candidates) (list of string)
    2)list of associated relevance scores (list of float)
    3)list containing for each keyphrase a list of alias (list of list of string)
    """
    N = min(N, len(candidates))
    doc_embedd = extract_doc_embedding(embdistrib, text_obj, use_filtered)  # Extract doc embedding    
    doc_embedd_mask = np.zeros((1,1536))
    doc_embedd_mask = get_doc_mask_embedding(embdistrib, text_obj)
    if smoothl1: doc_sim = ( 1.0 / ( 1.0 + smooth_l1_distances(X, doc_embedd) ) ) - (0.5 * ( 1.0 / ( 1.0 + smooth_l1_distances(X, doc_embedd_mask) ) ))
    else: doc_sim = cosine_similarity(X, doc_embedd) - (0.5 * cosine_similarity(X, doc_embedd_mask))
    doc_sim_norm = doc_sim / np.max(doc_sim)
    doc_sim_norm = 0.5 + (doc_sim_norm - np.average(doc_sim_norm)) / np.std(doc_sim_norm)    
    if smoothl1: sim_between = 1.0 / ( 1.0 + smooth_l1_distances(X) )
    else: sim_between = cosine_similarity(X)
    np.fill_diagonal(sim_between, np.NaN)
    sim_between_norm = sim_between / np.nanmax(sim_between, axis=0)
    sim_between_norm = 0.5 + (sim_between_norm - np.nanmean(sim_between_norm, axis=0)) / np.nanstd(sim_between_norm, axis=0)
    
    clusters = OPTICS(min_samples=np.min((5,len(X)))).fit_predict(X)
        
    # Post-processing with TextRank
    if alpha > 0.0:
        graph = nx.DiGraph()
        aux = []
        for pos1, w in enumerate(candidates):
            for pos2, y in enumerate(candidates):
                if pos1 != pos2 and clusters[pos1] == clusters[pos2] and sim_between_norm[pos1,pos2] > 0.0: 
                    aux.append((w,y, 1.0 + sim_between_norm[pos1,pos2]))
                    aux.append((y,w, 1.0 + sim_between_norm[pos1,pos2]))
            aux.append((w,"<DOC>", 1.0 + doc_sim_norm[pos1]))
            aux.append(("<DOC>",w, 1.0 + doc_sim_norm[pos1]))
        graph.add_weighted_edges_from(aux)
        aux = { "<DOC>" : 0.0 }
        for pos, w in enumerate(candidates): aux[w] = doc_sim_norm[pos]
        try:
            pr = nx.pagerank(graph, personalization=aux, alpha=alpha, max_iter=500, tol=1e-06)
            for pos, w in enumerate(candidates):
                doc_sim[pos] = pr[w]
                doc_sim_norm[pos] = pr[w]
        except: doc_sim = doc_sim

    selected_candidates = []
    unselected_candidates = [c for c in range(len(candidates))]    
    j = np.argmax(doc_sim)
    selected_candidates.append(j)
    unselected_candidates.remove(j)
    for _ in range(N - 1):
        selec_array = np.array(selected_candidates)
        unselec_array = np.array(unselected_candidates)
        distance_to_doc = doc_sim_norm[unselec_array, :]
        dist_between = sim_between_norm[unselec_array][:, selec_array]
        if dist_between.ndim == 1: dist_between = dist_between[:, np.newaxis]
        j = np.argmax(beta * distance_to_doc - (1 - beta) * np.max(dist_between, axis=1).reshape(-1, 1))
        item_idx = unselected_candidates[j]
        selected_candidates.append(item_idx)
        unselected_candidates.remove(item_idx)
    # Not using normalized version of doc_sim for computing relevance
    relevance_list = max_normalization(doc_sim[selected_candidates]).tolist()
    aliases_list = get_aliases(sim_between[selected_candidates, :], candidates, alias_threshold)
    return candidates[selected_candidates].tolist(), relevance_list, aliases_list, clusters[selected_candidates].tolist()

def MMRPhrase(embdistrib, text_obj, beta=0.65, N=20, use_filtered=False, alias_threshold=0.8):
    """
    Extract N keyphrases

    :param embdistrib: embedding distributor see @EmbeddingDistributor
    :param text_obj: Input text representation see @InputTextObj
    :param beta: hyperparameter beta for MMR (control tradeoff between informativeness and diversity)
    :param N: number of keyphrases to extract
    :param use_filtered: if true filter the text by keeping only candidate word before computing the doc embedding
    :return: A tuple with 3 elements :
    1)list of the top-N candidates (or less if there are not enough candidates) (list of string)
    2)list of associated relevance scores (list of float)
    3)list containing for each keyphrase a list of alias (list of list of string)
    """
    candidates, X = extract_candidates_embedding_for_doc(embdistrib, text_obj)
    if len(candidates) == 0:
        warnings.warn('No keyphrase extracted for this document')
        return None, None, None
    return _MMR(embdistrib, text_obj, candidates, X, beta, N, use_filtered, alias_threshold)


def MMRSent(embdistrib, text_obj, beta=0.5, N=10, use_filtered=True):
    """

    Extract N key sentences

    :param embdistrib: embedding distributor see @EmbeddingDistributor
    :param text_obj: Input text representation see @InputTextObj
    :param beta: hyperparameter beta for MMR (control tradeoff between informativeness and diversity)
    :param N: number of key sentences to extract
    :param use_filtered: if true filter the text by keeping only candidate word before computing the doc embedding
    :return: list of N key sentences (or less if there are not enough candidates)
    """
    candidates, X = extract_sent_candidates_embedding_for_doc(embdistrib, text_obj)
    if len(candidates) == 0:
        warnings.warn('No keysentence extracted for this document')
        return []
    return _MMR(embdistrib, text_obj, candidates, X, beta, N, use_filtered)


def max_normalization(array):
    """
    Compute maximum normalization (max is set to 1) of the array
    :param array: 1-d array
    :return: 1-d array max- normalized : each value is multiplied by 1/max value
    """
    return 1/np.max(array) * array.squeeze(axis=1)

def get_aliases(kp_sim_between, candidates, threshold):
    """
    Find candidates which are very similar to the keyphrases (aliases)
    :param kp_sim_between: ndarray of shape (nb_kp , nb candidates) containing the similarity
    of each kp with all the candidates. Note that the similarity between the keyphrase and itself should be set to
    NaN or 0
    :param candidates: array of candidates (array of string)
    :return: list containing for each keyphrase a list that contain candidates which are aliases
    (very similar) (list of list of string)
    """
    kp_sim_between = np.nan_to_num(kp_sim_between, 0)
    idx_sorted = np.flip(np.argsort(kp_sim_between), 1)
    aliases = []
    for kp_idx, item in enumerate(idx_sorted):
        alias_for_item = []
        for i in item:
            if kp_sim_between[kp_idx, i] >= threshold:
                alias_for_item.append(candidates[i])
            else:
                break
        aliases.append(alias_for_item)
    return aliases
