import math
import re
import os
import json

# =============================================================================
# EXERCICE 1 — Chargement du corpus
# =============================================================================

def charger_documents(fichier="documents.tsv"):
    chemin = os.path.join(os.path.dirname(__file__), fichier) if "__file__" in globals() else fichier
    documents = {}
    with open(chemin, "r", encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if ligne:
                parties = ligne.split("\t", 1)
                if len(parties) == 2:
                    documents[parties[0].strip()] = parties[1].strip()
    return documents

# =============================================================================
# EXERCICE 2 — Pipeline de prétraitement
# =============================================================================

stopwords = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "in", "into", "is", "it", "its", "of", "on", "or", "that",
    "the", "to", "was", "were", "with", "over"
}

def stem(mot):
    if mot.endswith("ies") and len(mot) > 4:
        return mot[:-3] + "y"
    if mot.endswith("es") and len(mot) > 3:
        return mot[:-2]
    if mot.endswith("s") and not mot.endswith("ss") and len(mot) > 3:
        return mot[:-1]
    return mot

def preprocess(text, stemming=False):
    tokens = re.findall(r"\b\w+\b", text.lower())
    tokens = [t for t in tokens if t not in stopwords]
    if stemming:
        tokens = [stem(t) for t in tokens]
    return tokens

# =============================================================================
# EXERCICE 3 — Construction de l'index inversé enrichi
# =============================================================================

def construire_index_enrichi(documents, stemming=False):
    index_inverse = {}
    for doc_id, texte in documents.items():
        termes = preprocess(texte, stemming=stemming)
        tf_doc = {}
        for t in termes:
            tf_doc[t] = tf_doc.get(t, 0) + 1
        
        for terme, tf in tf_doc.items():
            if terme not in index_inverse:
                index_inverse[terme] = {"df": 0, "postings": {}}
            index_inverse[terme]["postings"][doc_id] = {"tf": tf}
            
    for terme in index_inverse:
        index_inverse[terme]["df"] = len(index_inverse[terme]["postings"])
        
    return index_inverse

def afficher_index(index_inverse, nb=10):
    print(f"{'TERM':<15} {'DF':<5} {'POSTINGS'}")
    for terme in sorted(index_inverse.keys())[:nb]:
        df = index_inverse[terme]["df"]
        postings = ", ".join(f"{doc}(tf={info['tf']})" for doc, info in index_inverse[terme]["postings"].items())
        print(f"{terme:<15} {df:<5} {postings}")

# =============================================================================
# EXERCICE 4 — Calcul de IDF
# =============================================================================

def calculer_idf(terme, index_inverse, N):
    if terme in index_inverse and index_inverse[terme]["df"] > 0:
        return math.log10(N / index_inverse[terme]["df"])
    return 0.0

# =============================================================================
# EXERCICE 5 — Calcul des poids TF-IDF
# =============================================================================

def poids_tfidf(terme, doc_id, index_inverse, N):
    if terme in index_inverse and doc_id in index_inverse[terme]["postings"]:
        tf = index_inverse[terme]["postings"][doc_id]["tf"]
        idf = calculer_idf(terme, index_inverse, N)
        return (1 + math.log10(tf)) * idf
    return 0.0

# =============================================================================
# EXERCICE 6 — Représentation vectorielle des documents
# =============================================================================

def document_vector(doc_id, vocabulary, index_inverse, N):
    return [poids_tfidf(terme, doc_id, index_inverse, N) for terme in vocabulary]

# =============================================================================
# EXERCICE 7 — Représentation vectorielle d'une requête
# =============================================================================

def query_vector(query, vocabulary, index_inverse, N, stemming=False):
    tokens = preprocess(query, stemming=stemming)
    tf_q = {}
    for t in tokens:
        tf_q[t] = tf_q.get(t, 0) + 1
        
    vector_q = []
    for terme in vocabulary:
        if terme in tf_q:
            idf = calculer_idf(terme, index_inverse, N)
            vector_q.append((1 + math.log10(tf_q[terme])) * idf)
        else:
            vector_q.append(0.0)
    return vector_q

# =============================================================================
# EXERCICE 8 — Similarité cosinus
# =============================================================================

def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

# =============================================================================
# EXERCICE 9 — Ranking et Top-k
# =============================================================================

def search(query, documents, index_inverse, vocabulary, N, k=5, stemming=False):
    vq = query_vector(query, vocabulary, index_inverse, N, stemming=stemming)
    scores = []
    for doc_id, texte in documents.items():
        vd = document_vector(doc_id, vocabulary, index_inverse, N)
        score = cosine_similarity(vq, vd)
        if score > 0:
            scores.append((doc_id, score, texte))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]

def afficher_topk(query, resultats):
    print(f"\nQuery > {query}")
    print("TOP-5 RESULTS")
    print(f"{'Rank':<5} {'DocID':<6} {'Score':<8} {'Document'}")
    print("-" * 65)
    for rang, (doc_id, score, texte) in enumerate(resultats, 1):
        print(f"{rang:<5} {doc_id:<6} {score:<8.4f} {texte}")

# =============================================================================
# EXERCICE 11 — Recherche Booléenne AND
# =============================================================================

def recherche_booleenne_and(query, index_inverse, stemming=False):
    termes = preprocess(query, stemming=stemming)
    if not termes:
        return []
    postings_lists = [set(index_inverse.get(t, {}).get("postings", {}).keys()) for t in termes]
    resultats = postings_lists[0]
    for p in postings_lists[1:]:
        resultats = resultats.intersection(p)
    return sorted(list(resultats))


# =============================================================================
# EXECUTION DES EXERCICES ET AFFICHAGE DES RESULTATS
# =============================================================================

if __name__ == "__main__":
    # --- Exercice 1 ---
    print("=" * 65)
    print("EXERCICE 1 : Corpus de test")
    print("=" * 65)
    documents = charger_documents("documents.tsv")
    N = len(documents)
    print(f"Nombre de documents chargés : {N}")
    for doc_id in list(documents.keys())[:3]:
        print(f"  {doc_id} : {documents[doc_id]}")
    print("\nQ: Pourquoi un corpus thématiquement proche est-il plus intéressant ?")
    print("R: Car plusieurs documents partagent les mêmes termes généraux, ce qui oblige")
    print("   l'algorithme de ranking à discriminer la pertinence fine via TF-IDF.")

    # --- Exercice 2 ---
    print("\n" + "=" * 65)
    print("EXERCICE 2 : Prétraitement")
    print("=" * 65)
    for doc_id in ["D1", "D2", "D3"]:
        texte = documents[doc_id]
        toks = re.findall(r"\b\w+\b", texte.lower())
        sans_sw = [t for t in toks if t not in stopwords]
        final = preprocess(texte)
        print(f"\nDocument {doc_id} :")
        print("  Texte original   :", texte)
        print("  Tokens           :", toks)
        print("  Sans stopwords   :", sans_sw)
        print("  Termes indexés   :", final)
    print("\nQ: Pourquoi appliquer le même prétraitement aux documents et aux requêtes ?")
    print("R: Pour garantir la cohérence lexicale (casse, stopwords) lors de l'appariement.")

    # --- Exercice 3 ---
    print("\n" + "=" * 65)
    print("EXERCICE 3 : Index inversé enrichi")
    print("=" * 65)
    index_inverse = construire_index_enrichi(documents)
    vocabulary = sorted(index_inverse.keys())
    afficher_index(index_inverse, nb=8)
    
    # Sauvegarde de l'index au format JSON
    json_path = os.path.join(os.path.dirname(__file__), "index_inverse.json") if "__file__" in globals() else "index_inverse.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(index_inverse, f, indent=2)
    print(f"\nIndex inversé sauvegardé dans {json_path}")
    print("\nQ: Différence entre TF(t,d) et DF(t) ?")
    print("R: TF mesure la fréquence locale dans un doc, DF le nombre global de docs contenant t.")

    # --- Exercice 4 ---
    print("\n" + "=" * 65)
    print("EXERCICE 4 : Calcul IDF")
    print("=" * 65)
    print(f"{'Terme':<15} {'DF':<5} {'IDF'}")
    print("-" * 30)
    for t in ["information", "retrieval", "vector", "boolean"]:
        df = index_inverse.get(t, {}).get("df", 0)
        idf = calculer_idf(t, index_inverse, N)
        print(f"{t:<15} {df:<5} {idf:.4f}")

    terme_df_max = max(index_inverse.keys(), key=lambda t: index_inverse[t]["df"])
    terme_df_min = min(index_inverse.keys(), key=lambda t: index_inverse[t]["df"])
    terme_idf_max = max(index_inverse.keys(), key=lambda t: calculer_idf(t, index_inverse, N))
    print(f"\nDF maximal : '{terme_df_max}' (DF={index_inverse[terme_df_max]['df']})")
    print(f"DF minimal : '{terme_df_min}' (DF={index_inverse[terme_df_min]['df']})")
    print(f"IDF le plus grand : '{terme_idf_max}' (IDF={calculer_idf(terme_idf_max, index_inverse, N):.4f})")
    print("\nQ: Pourquoi un terme rare a-t-il un IDF plus élevé ?")
    print("R: Parce que DF est petit, donc N/DF est grand ; un mot rare est plus informatif.")

    # --- Exercice 5 ---
    print("\n" + "=" * 65)
    print("EXERCICE 5 : Poids TF-IDF")
    print("=" * 65)
    doc_choisi = "D8"
    termes_d8 = set(preprocess(documents[doc_choisi]))
    stats_d8 = []
    for t in termes_d8:
        tf = index_inverse[t]["postings"][doc_choisi]["tf"]
        df = index_inverse[t]["df"]
        idf = calculer_idf(t, index_inverse, N)
        w = poids_tfidf(t, doc_choisi, index_inverse, N)
        stats_d8.append((t, tf, df, idf, w))
    stats_d8.sort(key=lambda x: x[4], reverse=True)

    print(f"Document {doc_choisi} : {documents[doc_choisi]}")
    print(f"{'TERM':<15} {'TF':<5} {'DF':<5} {'IDF':<8} {'TF-IDF'}")
    print("-" * 45)
    for t, tf, df, idf, w in stats_d8:
        print(f"{t:<15} {tf:<5} {df:<5} {idf:<8.4f} {w:.4f}")
    print("\nQ: Les termes ayant le plus grand TF ont-ils toujours le plus grand TF-IDF ?")
    print("R: Non, un terme fréquent dans le document mais présent partout a un IDF faible,")
    print("   ce qui réduit son poids TF-IDF final.")

    # --- Exercice 6 ---
    print("\n" + "=" * 65)
    print("EXERCICE 6 : Représentation vectorielle des documents")
    print("=" * 65)
    for doc_id in ["D1", "D2", "D3"]:
        vec = document_vector(doc_id, vocabulary, index_inverse, N)
        nb_zeros = vec.count(0.0)
        print(f"Vecteur {doc_id} : {len(vec)} dimensions, {nb_zeros} composantes nulles ({nb_zeros/len(vec)*100:.1f}%)")
    print("\nQ: Pourquoi les vecteurs sont-ils creux (sparse) ?")
    print("R: Parce que chaque document n'utilise qu'un sous-ensemble très réduit du vocabulaire global.")

    # --- Exercice 7 ---
    print("\n" + "=" * 65)
    print("EXERCICE 7 : Vecteur requête")
    print("=" * 65)
    q = "information retrieval model"
    vq = query_vector(q, vocabulary, index_inverse, N)
    print(f"Requête : '{q}'")
    print("Termes non nuls et leurs poids :")
    for t, w in zip(vocabulary, vq):
        if w > 0:
            print(f"  {t:<15} : {w:.4f}")
    print("\nQ: Que faire si un terme de requête n'est pas dans le vocabulaire ?")
    print("R: Il est ignoré (poids 0) car il n'apporte aucune information discriminante.")

    # --- Exercice 8 ---
    print("\n" + "=" * 65)
    print("EXERCICE 8 : Similarité cosinus")
    print("=" * 65)
    print(f"Similarité entre q='{q}' et les 5 premiers documents :")
    for doc_id in list(documents.keys())[:5]:
        vd = document_vector(doc_id, vocabulary, index_inverse, N)
        score = cosine_similarity(vq, vd)
        print(f"  {doc_id} : {score:.4f}")
    print("\nQ: Signification des scores proches de 1 et 0 ?")
    print("R: 1 indique une forte similarité vectorielle, 0 indique l'orthogonalité (aucun terme commun).")

    # --- Exercice 9 ---
    print("\n" + "=" * 65)
    print("EXERCICE 9 : Ranking et Top-k")
    print("=" * 65)
    requetes_test = [
        "information retrieval model",
        "inverted indexes search",
        "vector database embeddings",
        "bm25 ranking relevance",
        "boolean retrieval search"
    ]
    for rq in requetes_test:
        res = search(rq, documents, index_inverse, vocabulary, N, k=5)
        afficher_topk(rq, res)

    # --- Exercice 10 ---
    print("\n" + "=" * 65)
    print("EXERCICE 10 : Expérience model vs models")
    print("=" * 65)
    q_exp = "information retrieval model"
    
    # Sans normalisation
    res_sans = search(q_exp, documents, index_inverse, vocabulary, N, k=len(documents), stemming=False)
    score_d2_sans = next((s for d, s, _ in res_sans if d == "D2"), 0.0)
    rang_d2_sans = next((i + 1 for i, (d, _, _) in enumerate(res_sans) if d == "D2"), "-")
    score_d7_sans = next((s for d, s, _ in res_sans if d == "D7"), 0.0)
    rang_d7_sans = next((i + 1 for i, (d, _, _) in enumerate(res_sans) if d == "D7"), "-")

    # Avec normalisation morphologique
    index_stem = construire_index_enrichi(documents, stemming=True)
    vocab_stem = sorted(index_stem.keys())
    res_avec = search(q_exp, documents, index_stem, vocab_stem, N, k=len(documents), stemming=True)
    score_d2_avec = next((s for d, s, _ in res_avec if d == "D2"), 0.0)
    rang_d2_avec = next((i + 1 for i, (d, _, _) in enumerate(res_avec) if d == "D2"), "-")
    score_d7_avec = next((s for d, s, _ in res_avec if d == "D7"), 0.0)
    rang_d7_avec = next((i + 1 for i, (d, _, _) in enumerate(res_avec) if d == "D7"), "-")

    print(f"Requête : '{q_exp}'")
    print(f"{'Mesure':<35} {'Sans normalisation':<20} {'Avec normalisation'}")
    print("-" * 75)
    print(f"{'Score D2 (contient model)':<35} {score_d2_sans:<20.4f} {score_d2_avec:.4f}")
    print(f"{'Rang D2':<35} {str(rang_d2_sans):<20} {str(rang_d2_avec)}")
    print(f"{'Score D7 (contient models)':<35} {score_d7_sans:<20.4f} {score_d7_avec:.4f}")
    print(f"{'Rang D7':<35} {str(rang_d7_sans):<20} {str(rang_d7_avec)}")

    # --- Exercice 11 ---
    print("\n" + "=" * 65)
    print("EXERCICE 11 : Comparaison Booléen AND vs Vectoriel TF-IDF")
    print("=" * 65)
    q_comp = "information retrieval model"
    docs_bool = recherche_booleenne_and(q_comp, index_inverse)
    docs_vect = [d for d, s, _ in search(q_comp, documents, index_inverse, vocabulary, N, k=5)]
    print(f"Requête : '{q_comp}'")
    print(f"Résultats Booléen AND  : {docs_bool}")
    print(f"Résultats Vectoriel (Top-5) : {docs_vect}")

    print("\nTableau comparatif :")
    print(f"{'Critère':<30} {'Booléen AND':<25} {'Vectoriel TF-IDF'}")
    print("-" * 75)
    print(f"{'Documents retrouvés':<30} {'Tous les termes exigés':<25} {'Correspondance partielle'}")
    print(f"{'Score de pertinence':<30} {'Binaire (0 ou 1)':<25} {'Gradué (0 à 1)'}")
    print(f"{'Classement des résultats':<30} {'Aucun':<25} {'Ordonné par score décroissant'}")

    # --- Question de synthèse ---
    print("\n" + "=" * 65)
    print("QUESTION DE SYNTHESE : Limites de TF-IDF justifiant BM25")
    print("=" * 65)
    print("1. Saturation du TF : TF-IDF fait croître le score indéfiniment avec log(TF),")
    print("   alors que BM25 sature via le paramètre k1 pour éviter la sur-pondération.")
    print("2. Longueur des documents : BM25 normalise les scores par la longueur du document")
    print("   via le paramètre b par rapport à la moyenne de la collection.")

    # --- Exercice 12 : Démonstration interactive ---
    print("\n" + "=" * 65)
    print("       FIGL SEARCH ENGINE V2")
    print("=" * 65)
    print(f"Documents indexed : {N}")
    print(f"Vocabulary size   : {len(vocabulary)}")

    # Boucle de recherche
    while True:
        try:
            req = input("\nQuery (ou Entrée pour quitter) > ").strip()
            if not req or req.lower() in ["exit", "quit", "q"]:
                break
            k_input = input("Top-k (3, 5, 10) [5] : ").strip()
            k = int(k_input) if k_input in ["3", "5", "10"] else 5
            res = search(req, documents, index_inverse, vocabulary, N, k=k)
            if not res:
                print("Aucun document pertinent trouvé.")
            else:
                for rang, (doc_id, score, texte) in enumerate(res, 1):
                    print(f"\n{rang}. {doc_id}  score = {score:.4f}")
                    print(f"   {texte}")
        except (EOFError, KeyboardInterrupt):
            break
