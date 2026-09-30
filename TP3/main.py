#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
=============================================================================
TP3 : Construction d'un moteur de recherche vectoriel avec TF-IDF
Filière : FIGL2 - Indexation et recherche d'information
Enseignante : Dr. F. Chaabane
Auteur : Diya135
=============================================================================

Ce script exécute successivement et de manière automatisée l'intégralité des 
exercices du TP3 (Exercices 1 à 12), incluant :
- Le chargement et l'analyse du corpus (documents.tsv)
- Les étapes de prétraitement avec et sans racinage
- L'indexation inversée enrichie (TF, DF, postings)
- Le calcul de l'IDF et des poids TF-IDF logarithmiques
- La vectorisation des documents et requêtes
- Le calcul de la similarité cosinus sans bibliothèque de ranking
- Le ranking Top-k pour 5 requêtes variées avec analyse critique
- L'expérience morphologique 'model' vs 'models'
- La comparaison Booléen AND vs Vectoriel TF-IDF
- La réponse argumentée à la question de synthèse (BM25 vs TF-IDF)
- Le lancement interactif du FIGL Search Engine V2 via argument --interactive
"""

import sys
import os
import json
from typing import Dict, List, Tuple

from search_engine import (
    load_documents_from_tsv,
    preprocess,
    trace_preprocessing_steps,
    construire_index_enrichi,
    format_index_display,
    calculer_idf,
    get_df_idf_stats,
    poids_tfidf,
    document_terms_tfidf_ranking,
    document_vector,
    query_vector,
    cosine_similarity,
    search,
    boolean_and_search,
    FIGLSearchEngineV2
)


def print_section(title: str):
    print("\n" + "=" * 80)
    print(f" {title.upper()} ")
    print("=" * 80)


def main():
    tsv_path = os.path.join(os.path.dirname(__file__), "documents.tsv")
    if not os.path.exists(tsv_path):
        print(f"Erreur : Le fichier {tsv_path} est introuvable.")
        sys.exit(1)

    print("*" * 80)
    print("  FIGL2 - TP3 : CONSTRUCTION D'UN MOTEUR DE RECHERCHE VECTORIEL TF-IDF  ")
    print("*" * 80)

    # =========================================================================
    # EXERCICE 1 — Constitution d'un corpus de test
    # =========================================================================
    print_section("Exercice 1 — Constitution du corpus de test")
    documents = load_documents_from_tsv(tsv_path)
    N = len(documents)
    print(f"[OK] Nombre de documents chargés depuis documents.tsv : N = {N}")
    print("\nÉchantillon des 5 premiers documents :")
    for doc_id in list(documents.keys())[:5]:
        print(f"  {doc_id:<4} : {documents[doc_id]}")

    print("\n[QUESTION THEORIQUE]")
    print("Pourquoi un corpus composé de documents proches thématiquement est-il plus")
    print("intéressant pour tester un algorithme de ranking ?")
    print("-" * 80)
    print("REPONSE :")
    print("  Dans un corpus hétérogène (ex: cuisine, sport, astrophysique), retrouver")
    print("  le bon document est trivial : la simple présence d'un mot-clé suffit à")
    print("  discriminer le sujet sans tester la finesse du modèle de ranking.")
    print("  À l'inverse, dans un corpus thématiquement homogène (ici la recherche")
    print("  d'information et le NLP), de nombreux documents partagent les mêmes termes")
    print("  généraux ('search', 'retrieval', 'index', 'vector'). L'algorithme de ranking")
    print("  doit donc impérativement discriminer la pertinence fine grâce à la pondération")
    print("  TF-IDF (fréquence locale vs spécificité globale) et à la similarité cosinus.")

    # =========================================================================
    # EXERCICE 2 — Pipeline de prétraitement
    # =========================================================================
    print_section("Exercice 2 — Pipeline de prétraitement")
    sample_docs = ["D1", "D2", "D7"]
    for doc_id in sample_docs:
        raw_text = documents[doc_id]
        trace = trace_preprocessing_steps(raw_text, use_stemming=True)
        print(f"\nDocument {doc_id} :")
        print(f"  1. Texte original                 : {trace['original']}")
        print(f"  2. Tokens (minuscules + regex)    : {trace['tokens']}")
        print(f"  3. Tokens après suppression stopw : {trace['tokens_without_stopwords']}")
        print(f"  4. Termes indexés (avec stemming) : {trace['final_terms']}")

    print("\n[QUESTION THEORIQUE]")
    print("Pourquoi faut-il appliquer exactement le même pipeline de prétraitement")
    print("aux documents et aux requêtes ?")
    print("-" * 80)
    print("REPONSE :")
    print("  L'espace vectoriel repose sur une correspondance stricte terme-à-terme entre")
    print("  les dimensions du document et celles de la requête. Si les pipelines diffèrent")
    print("  (ex: mise en minuscules sur la requête mais pas sur les documents, ou stemming")
    print("  appliqué d'un seul côté), des termes sémantiquement identiques comme 'Models'")
    print("  et 'model' ne partageront pas la même dimension dans le vecteur. Le produit")
    print("  scalaire sera nul, provoquant un silence documentaire injustifié (chute du rappel).")

    # =========================================================================
    # EXERCICE 3 — Construction d'un index inversé enrichi
    # =========================================================================
    print_section("Exercice 3 — Construction de l'index inversé enrichi")
    # Pour les exercices 3 à 9, nous utilisons le prétraitement standard sans stemming pour observer les termes bruts
    index_inverse = construire_index_enrichi(documents, use_stemming=False)
    vocabulaire = sorted(index_inverse.keys())
    print(f"[OK] Taille du vocabulaire |V| : {len(vocabulaire)} termes uniques.")

    # Sauvegarde de l'index enrichi en JSON
    json_path = os.path.join(os.path.dirname(__file__), "index_inverse.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(index_inverse, f, indent=2, ensure_ascii=False)
    print(f"[OK] Index inversé enrichi sauvegardé dans '{json_path}'.")

    print("\nAffichage lisible d'un extrait de l'index inversé enrichi :")
    sample_terms = ["retrieval", "vector", "boolean", "information", "embeddings", "indexes"]
    sample_index = {t: index_inverse[t] for t in sample_terms if t in index_inverse}
    print(format_index_display(sample_index))

    print("\n[QUESTION THEORIQUE]")
    print("Quelle est la différence entre TF(t,d) et DF(t) ?")
    print("-" * 80)
    print("REPONSE :")
    print("  - TF(t,d) (Term Frequency) est une mesure LOCALE intra-document : elle compte")
    print("    le nombre de fois où le terme t apparaît dans un document spécifique d.")
    print("    Un TF élevé indique que le terme est central dans ce document particulier.")
    print("  - DF(t) (Document Frequency) est une mesure GLOBALE inter-documents : elle compte")
    print("    le nombre total de documents de la collection contenant le terme t au moins une fois.")
    print("    Un DF élevé indique un terme banal/omniprésent dans la collection, donc peu discriminant.")

    # =========================================================================
    # EXERCICE 4 — Calcul de IDF
    # =========================================================================
    print_section("Exercice 4 — Calcul de l'IDF")
    max_df_term, min_df_term, max_idf_term, df_idf_table = get_df_idf_stats(index_inverse, N)

    print(f"{'Terme':<20} {'DF':<8} {'IDF (log10(N/DF))':<18}")
    print("-" * 48)
    sample_display_terms = ["retrieval", "information", "vector", "search", "boolean", "bm25", "embeddings", "databases"]
    for t in sample_display_terms:
        if t in df_idf_table:
            df, idf = df_idf_table[t]
            print(f"{t:<20} {df:<8} {idf:.4f}")

    print("\n[Identification automatique des extrêmes]")
    print(f"  - Terme de DF maximal  : '{max_df_term}' (DF = {df_idf_table[max_df_term][0]}, IDF = {df_idf_table[max_df_term][1]:.4f})")
    print(f"  - Terme de DF minimal  : '{min_df_term}' (DF = {df_idf_table[min_df_term][0]}, IDF = {df_idf_table[min_df_term][1]:.4f})")
    print(f"  - Terme de plus grand IDF : '{max_idf_term}' (IDF = {df_idf_table[max_idf_term][1]:.4f}, DF = {df_idf_table[max_idf_term][0]})")

    print("\n[QUESTION THEORIQUE]")
    print("Pourquoi un terme rare dans la collection reçoit-il généralement un IDF plus élevé ?")
    print("-" * 80)
    print("REPONSE :")
    print("  Mathématiquement, IDF(t) = log10(N / DF(t)). Lorsque DF(t) est petit, le rapport")
    print("  N / DF(t) est grand, donc son logarithme est élevé. Sur le plan de la théorie de")
    print("  l'information (Shannon), un terme rare apporte une très grande quantité d'information")
    print("  et un fort pouvoir discriminant : s'il apparaît dans une requête, seuls les rares")
    print("  documents qui le mentionnent sont hautement pertinents.")

    # =========================================================================
    # EXERCICE 5 — Calcul des poids TF-IDF
    # =========================================================================
    print_section("Exercice 5 — Calcul des poids TF-IDF")
    chosen_doc = "D8"
    ranked_terms_d8 = document_terms_tfidf_ranking(chosen_doc, index_inverse, N)

    print(f"Document {chosen_doc} : \"{documents[chosen_doc]}\"")
    print(f"{'TERM':<18} {'TF':<6} {'DF':<6} {'IDF':<10} {'TF-IDF':<10}")
    print("-" * 52)
    for row in ranked_terms_d8:
        print(f"{row['term']:<18} {row['tf']:<6} {row['df']:<6} {row['idf']:<10.4f} {row['tfidf']:<10.4f}")

    print("\n[QUESTION THEORIQUE]")
    print("Les termes ayant le plus grand TF sont-ils toujours ceux qui ont le plus grand")
    print("TF-IDF ? Justifier à partir de vos résultats.")
    print("-" * 80)
    print("REPONSE :")
    print("  NON, absolument pas. Le poids TF-IDF résulte du produit d'une composante locale")
    print("  (TF) et d'une composante globale (IDF). Un terme peut avoir un TF très élevé dans")
    print("  un document, mais s'il est présent dans presque tous les documents de la collection")
    print("  (DF élevé), son IDF sera proche de 0, réduisant son poids TF-IDF à une valeur minime.")
    print("  À l'inverse, un terme avec TF = 1 mais très spécifique (ex: 'embeddings' ou 'dense')")
    print("  aura un IDF élevé et surpassera un terme générique récurrent.")

    # =========================================================================
    # EXERCICE 6 — Construction des vecteurs documents
    # =========================================================================
    print_section("Exercice 6 — Construction des vecteurs documents")
    v_D1 = document_vector("D1", vocabulaire, index_inverse, N)
    v_D2 = document_vector("D2", vocabulaire, index_inverse, N)
    v_D3 = document_vector("D3", vocabulaire, index_inverse, N)

    zero_D1 = sum(1 for x in v_D1 if x == 0.0)
    zero_D2 = sum(1 for x in v_D2 if x == 0.0)
    zero_D3 = sum(1 for x in v_D3 if x == 0.0)

    print(f"Dimension totale du vocabulaire |V| : {len(vocabulaire)}")
    print(f"Vecteur D1 : dimension = {len(v_D1)}, composantes nulles = {zero_D1}/{len(v_D1)} ({zero_D1/len(v_D1)*100:.1f}%)")
    print(f"  Termes non nuls de D1 : {[(vocabulaire[i], round(v_D1[i], 4)) for i in range(len(v_D1)) if v_D1[i] > 0]}")
    print(f"Vecteur D2 : dimension = {len(v_D2)}, composantes nulles = {zero_D2}/{len(v_D2)} ({zero_D2/len(v_D2)*100:.1f}%)")
    print(f"  Termes non nuls de D2 : {[(vocabulaire[i], round(v_D2[i], 4)) for i in range(len(v_D2)) if v_D2[i] > 0]}")
    print(f"Vecteur D3 : dimension = {len(v_D3)}, composantes nulles = {zero_D3}/{len(v_D3)} ({zero_D3/len(v_D3)*100:.1f}%)")
    print(f"  Termes non nuls de D3 : {[(vocabulaire[i], round(v_D3[i], 4)) for i in range(len(v_D3)) if v_D3[i] > 0]}")

    print("\n[QUESTION THEORIQUE]")
    print("Pourquoi les vecteurs TF-IDF de documents sont-ils généralement creux (sparse) ?")
    print("-" * 80)
    print("REPONSE :")
    print("  Le vocabulaire global |V| englobe l'ensemble des termes distincts de TOUTE")
    print("  la collection (souvent des milliers ou millions de mots dans des systèmes réels).")
    print("  Or, un document individuel ne contient typiquement que quelques dizaines ou")
    print("  centaines de mots uniques. Par conséquent, plus de 90% à 99% des dimensions du")
    print("  vecteur reçoivent un poids TF=0 (donc TF-IDF=0), rendant la matrice creuse.")

    # =========================================================================
    # EXERCICE 7 — Représentation vectorielle d'une requête
    # =========================================================================
    print_section("Exercice 7 — Représentation vectorielle d'une requête")
    q_initial = "information retrieval model"
    v_q = query_vector(q_initial, vocabulaire, index_inverse, N, use_stemming=False)

    print(f"Requête initiale : \"{q_initial}\"")
    print("Termes non nuls de la requête et leurs poids :")
    for i, w in enumerate(v_q):
        if w > 0:
            print(f"  Terme : '{vocabulaire[i]:<15}' -> Poids w(t,q) = {w:.4f} (IDF = {calculer_idf(vocabulaire[i], index_inverse, N):.4f})")

    print("\n[QUESTION THEORIQUE]")
    print("Que doit faire le programme lorsqu'un terme de la requête n'appartient pas au vocabulaire de la collection ?")
    print("-" * 80)
    print("REPONSE :")
    print("  Un terme hors-vocabulaire (Out-Of-Vocabulary / OOV) possède un DF=0, ce qui")
    print("  signifie qu'aucun document de la collection ne le contient. Le programme doit")
    print("  l'ignorer lors de la projection vectorielle (ou lui affecter un poids 0),")
    print("  car ce terme ne peut apporter aucune intersection avec les postings des documents.")

    # =========================================================================
    # EXERCICE 8 — Similarité cosinus
    # =========================================================================
    print_section("Exercice 8 — Similarité cosinus (sans sklearn)")
    print("Calcul de similarité cosinus entre la requête et chaque document...")
    sims = []
    for doc_id, text in documents.items():
        v_d = document_vector(doc_id, vocabulaire, index_inverse, N)
        score = cosine_similarity(v_q, v_d)
        sims.append((doc_id, score, text))

    sims.sort(key=lambda x: x[1], reverse=True)
    print(f"Nombre de documents ayant une similarité non nulle : {sum(1 for s in sims if s[1] > 0)} / {N}")

    print("\n[QUESTION THEORIQUE]")
    print("Que signifient intuitivement des scores proches de 1 et proches de 0 ?")
    print("-" * 80)
    print("REPONSE :")
    print("  - Un score proche de 1 signifie que l'angle entre le vecteur requête et le vecteur")
    print("    document est très faible (cos(0)=1). Le document partage les mêmes termes clés")
    print("    dans des proportions très similaires à la requête : il est hautement pertinent.")
    print("  - Un score égal à 0 (vecteurs orthogonaux) indique qu'aucun terme commun n'existe.")
    print("  - Un score proche de 0 indique une très faible superposition sémantique (seul un")
    print("    mot banal ou marginal est partagé avec un faible poids).")

    # =========================================================================
    # EXERCICE 9 — Ranking et Top-k
    # =========================================================================
    print_section("Exercice 9 — Ranking et Top-k (5 requêtes de test)")
    test_queries = [
        "information retrieval model",
        "inverted indexes for search",
        "vector embeddings and similarity",
        "boolean retrieval operators",
        "ranking function bm25"
    ]

    for q in test_queries:
        print(f"\nQuery > {q}")
        print("TOP-5 RESULTS")
        print(f"{'Rank':<6} {'DocID':<8} {'Score':<10} {'Document'}")
        print("-" * 75)
        top5 = search(q, documents, index_inverse, vocabulaire, N, k=5, use_stemming=False)
        for rank, (doc_id, score, text) in enumerate(top5, start=1):
            print(f"{rank:<6} {doc_id:<8} {score:<10.4f} {text}")

    print("\n[ANALYSE ET COMMENTAIRE DES RESULTATS DES REQUETES]")
    print("-" * 80)
    print("  - Requête 1 ('information retrieval model') : D2 arrive en tête car il contient")
    print("    'model' et 'retrieval'/'information'. D1 et D6 suivent grâce à 'information retrieval'.")
    print("  - Requête 2 ('inverted indexes for search') : D1, D10 et D11 dominent nettement car")
    print("    ils traitent directement de l'indexation inversée et des moteurs de recherche.")
    print("  - Requête 3 ('vector embeddings and similarity') : D8, D9 et D18 dominent le classement")
    print("    grâce à la forte concordance des termes discriminants 'embeddings' et 'vector'.")
    print("  - Requête 4 ('boolean retrieval operators') : D3 est classé 1er avec un score très élevé")
    print("    (contient 'Boolean retrieval uses AND OR and NOT operators'), suivi de D20.")
    print("  - Requête 5 ('ranking function bm25') : D6 et D15 ressortent en top position,")
    print("    reflétant fidèlement la spécificité des termes 'BM25' et 'ranking function'.")

    # =========================================================================
    # EXERCICE 10 — Expérience : model vs models
    # =========================================================================
    print_section("Exercice 10 — Expérience : model vs models")
    q_exp = "information retrieval model"

    # Expérience A : sans normalisation morphologique
    results_no_stem = search(q_exp, documents, index_inverse, vocabulaire, N, k=N, use_stemming=False)
    rank_dict_no_stem = {doc_id: (rank, score) for rank, (doc_id, score, _) in enumerate(results_no_stem, start=1)}

    # Expérience B : avec normalisation morphologique
    index_stem = construire_index_enrichi(documents, use_stemming=True)
    vocab_stem = sorted(index_stem.keys())
    results_stem = search(q_exp, documents, index_stem, vocab_stem, N, k=N, use_stemming=True)
    rank_dict_stem = {doc_id: (rank, score) for rank, (doc_id, score, _) in enumerate(results_stem, start=1)}

    # Documents clés : D2 (contient 'model') et D7 (contient 'models' et 'information retrieval')
    d2_no_stem = rank_dict_no_stem.get("D2", (">N", 0.0))
    d7_no_stem = rank_dict_no_stem.get("D7", (">N", 0.0))
    d2_stem = rank_dict_stem.get("D2", (">N", 0.0))
    d7_stem = rank_dict_stem.get("D7", (">N", 0.0))

    print(f"Requête testée : \"{q_exp}\"")
    print(f"Document D2 (avec 'model')  : \"{documents['D2']}\"")
    print(f"Document D7 (avec 'models') : \"{documents['D7']}\"")
    print()
    print(f"{'Mesure':<38} {'Sans normalisation':<22} {'Avec normalisation'}")
    print("-" * 80)
    print(f"{'Score du document D2 (model)':<38} {d2_no_stem[1]:<22.4f} {d2_stem[1]:.4f}")
    print(f"{'Score du document D7 (models)':<38} {d7_no_stem[1]:<22.4f} {d7_stem[1]:.4f}")
    print(f"{'Rang du document D2 (model)':<38} {str(d2_no_stem[0]):<22} {str(d2_stem[0])}")
    print(f"{'Rang du document D7 (models)':<38} {str(d7_no_stem[0]):<22} {str(d7_stem[0])}")

    print("\n[QUESTION THEORIQUE]")
    print("Le prétraitement peut-il modifier le classement final ? Montrer l'effet observé à partir des scores et des rangs.")
    print("-" * 80)
    print("REPONSE :")
    print(f"  OUI, le prétraitement modifie considérablement le classement final !")
    print(f"  - Sans normalisation morphologique, la requête contient 'model' au singulier.")
    print(f"    Le mot 'models' dans D7 n'est pas apparié avec 'model', donc D7 ne bénéficie")
    print(f"    que de 'information retrieval' et son score est de {d7_no_stem[1]:.4f} (rang {d7_no_stem[0]}).")
    print(f"  - Avec normalisation morphologique (PorterStemmer), 'model' et 'models' sont tous")
    print(f"    deux ramenés au radical commun 'model'. D7 capture maintenant la totalité des 3 termes")
    print(f"    de la requête ('inform', 'retriev', 'model') ! Son score monte à {d7_stem[1]:.4f}")
    print(f"    et il grimpe à la première place (rang {d7_stem[0]}), illustrant l'impact majeur du stemming.")

    # =========================================================================
    # EXERCICE 11 — Comparaison Booléen AND vs Vectoriel TF-IDF
    # =========================================================================
    print_section("Exercice 11 — Comparaison Booléen AND vs Vectoriel TF-IDF")
    q_comp = "inverted indexes search"
    print(f"Requête testée (3 termes) : \"{q_comp}\"\n")

    bool_res = boolean_and_search(q_comp, index_inverse, use_stemming=False)
    vec_res = search(q_comp, documents, index_inverse, vocabulaire, N, k=10, use_stemming=False)

    print(f"Résultats Modèle Booléen AND (intersection stricte) : {bool_res}")
    print(f"Résultats Modèle Vectoriel TF-IDF (Top-5) :")
    for r, (did, sc, txt) in enumerate(vec_res[:5], 1):
        print(f"  {r}. {did} (score={sc:.4f}) : {txt}")

    print("\nTableau comparatif des deux modèles :")
    print(f"{'Critère':<30} {'Booléen AND':<25} {'Vectoriel TF-IDF'}")
    print("-" * 80)
    print(f"{'Document retrouvé/non retrouvé':<30} {'Binaire (Strict 0 ou 1)':<25} {'Continu (Graduel)'}")
    print(f"{'Correspondance partielle':<30} {'Non (Rejet total)':<25} {'Oui (Scores partiels > 0)'}")
    print(f"{'Score de pertinence':<30} {'Aucun score calculé':<25} {'Similarité cosinus [0, 1]'}")
    print(f"{'Classement des résultats':<30} {'Impossible (Ensemble plat)':<25} {'Ordonné par pertinence'}")

    print("\n[QUESTION THEORIQUE]")
    print("Un document contenant seulement deux termes importants d'une requête à trois")
    print("termes peut-il être retrouvé par les deux modèles de la même manière ? Expliquer.")
    print("-" * 80)
    print("REPONSE :")
    print("  NON :")
    print("  - Dans le modèle Booléen AND, la condition logique impose la présence simultanée")
    print("    des 3 termes (T1 AND T2 AND T3). Un document contenant 2 termes sur 3 est")
    print("    strictement éliminé (résultat vide / faux négatif).")
    print("  - Dans le modèle Vectoriel TF-IDF, les deux termes présents contribuent positivement")
    print("    au produit scalaire (q . d). Le document reçoit un score proportionnel à l'importance")
    print("    des termes partagés et apparaît dans le classement, évitant le silence documentaire.")

    # =========================================================================
    # QUESTION DE SYNTHESE (Page 7)
    # =========================================================================
    print_section("Question de synthèse : Limites de TF-IDF et apport de BM25")
    print("L'index inversé permet de retrouver les documents.")
    print("TF-IDF et la similarité cosinus permettent maintenant de les classer.")
    print("Quelles limites de TF-IDF pourraient justifier l'utilisation d'un modèle")
    print("de ranking plus avancé comme BM25 ?")
    print("-" * 80)
    print("REPONSE SYNTHETIQUE :")
    print("  1. Saturation non linéaire du Term Frequency :")
    print("     Dans TF-IDF classique (même logarithmique), la répétition d'un terme continue")
    print("     d'augmenter le score de façon non bornée. BM25 introduit un paramètre k1 (généralement")
    print("     entre 1.2 et 2.0) qui borne asymptotiquement la contribution du TF : après quelques")
    print("     occurrences, un mot n'augmente plus artificiellement la pertinence.")
    print("  2. Normalisation par la longueur du document (Document Length Normalization) :")
    print("     TF-IDF favorise souvent les documents longs car ils contiennent mécaniquement plus de")
    print("     mots. Bien que la similarité cosinus normalise par la norme euclidienne, cette normalisation")
    print("     est géométrique et ne tient pas compte de la longueur moyenne des documents de la collection.")
    print("     BM25 pénalise explicitement la verbosité via le paramètre b (souvent 0.75) et le ratio")
    print("     |D| / avgdl (longueur du document divisée par la longueur moyenne de la collection).")
    print("  3. Fondement probabiliste :")
    print("     BM25 découle du cadre théorique probabiliste 2-Poisson / Robertson-Spärck Jones,")
    print("     offrant une estimation formelle de la probabilité de pertinence P(R=1 | d, q).")

    print("\n" + "=" * 80)
    print("Tous les exercices 1 à 11 et la question de synthèse ont été exécutés avec succès !")
    print("Pour lancer le moteur interactif FIGL Search Engine V2, utilisez : python main.py --interactive")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    if "--interactive" in sys.argv or "-i" in sys.argv:
        tsv_path = os.path.join(os.path.dirname(__file__), "documents.tsv")
        engine = FIGLSearchEngineV2(tsv_path, use_stemming=True)
        engine.interactive_cli()
    else:
        main()
