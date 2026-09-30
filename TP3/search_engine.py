"""
TP3 : Construction d'un moteur de recherche vectoriel avec TF-IDF
Indexation et recherche d'information - FIGL2
Enseignante : Dr. F. Chaabane

Ce module implémente toutes les fonctions fondamentales d'un moteur de recherche vectoriel :
- Prétraitement de texte (normalisation, tokenisation, stopwords, stemming)
- Construction d'un index inversé enrichi (TF, DF, postings)
- Calcul de l'IDF et des poids TF-IDF
- Représentation vectorielle des documents et des requêtes
- Calcul de la similarité cosinus (sans bibliothèques externes de ranking)
- Recherche vectorielle et booléenne AND
- Comparaisons expérimentales et moteur interactif FIGL Search Engine V2
"""

import math
import re
from typing import Dict, List, Tuple, Set, Optional, Any

# Liste standard et complète de stopwords anglais pour assurer l'autonomie du module
STOPWORDS_EN = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}

# Initialisation du PorterStemmer de NLTK avec gestion d'exception si indisponible
try:
    from nltk.stem import PorterStemmer
    _STEMMER = PorterStemmer()
except ImportError:
    _STEMMER = None


def load_documents_from_tsv(file_path: str) -> Dict[str, str]:
    """
    Charge les documents depuis un fichier TSV (DocID \\t Texte).
    Retourne un dictionnaire {doc_id: raw_text}.
    """
    documents = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                doc_id, text = parts
                documents[doc_id.strip()] = text.strip()
    return documents


# ==============================================================================
# EXERCICE 2 — Pipeline de prétraitement
# ==============================================================================

def preprocess(text: str, use_stemming: bool = False) -> List[str]:
    """
    Applique le pipeline complet de prétraitement sur une chaîne de texte :
    1. Passage en minuscules
    2. Tokenisation (mots alphanumériques)
    3. Suppression des mots vides (stopwords)
    4. Option : stemming / lemmatisation (via PorterStemmer)

    Args:
        text (str): Texte brut à prétraiter.
        use_stemming (bool): Active ou désactive le racinage morphologique.

    Returns:
        List[str]: Liste des termes prétraités et normalisés.
    """
    # 1. Passage en minuscules
    text_lower = text.lower()

    # 2. Tokenisation par expressions régulières (mots alphanumériques)
    tokens = re.findall(r"\b[a-zA-Z0-9]+\b", text_lower)

    # 3. Suppression des stopwords
    tokens_without_sw = [tok for tok in tokens if tok not in STOPWORDS_EN]

    # 4. Option : stemming / lemmatisation
    if use_stemming and _STEMMER is not None:
        tokens_final = [_STEMMER.stem(tok) for tok in tokens_without_sw]
    else:
        tokens_final = tokens_without_sw

    return tokens_final


def trace_preprocessing_steps(text: str, use_stemming: bool = True) -> Dict[str, Any]:
    """
    Exécute le prétraitement en conservant les étapes intermédiaires
    pour l'affichage pédagogique requis par l'Exercice 2.
    """
    text_lower = text.lower()
    tokens = re.findall(r"\b[a-zA-Z0-9]+\b", text_lower)
    tokens_no_sw = [t for t in tokens if t not in STOPWORDS_EN]
    if use_stemming and _STEMMER is not None:
        tokens_stemmed = [_STEMMER.stem(t) for t in tokens_no_sw]
    else:
        tokens_stemmed = list(tokens_no_sw)

    return {
        "original": text,
        "tokens": tokens,
        "tokens_without_stopwords": tokens_no_sw,
        "final_terms": tokens_stemmed
    }


# ==============================================================================
# EXERCICE 3 — Construction d'un index inversé enrichi
# ==============================================================================

def construire_index_enrichi(documents: Dict[str, Any], use_stemming: bool = False) -> Dict[str, Any]:
    """
    Construit un index inversé enrichi mémorisant les statistiques utiles au calcul des scores.

    Structure attendue :
    index_inverse = {
        "terme": {
            "df": nombre_de_documents,
            "postings": {
                "D1": {"tf": nombre_occurrences},
                ...
            }
        },
        ...
    }

    Args:
        documents: Dictionnaire {doc_id: raw_text} ou {doc_id: tokens_list}
        use_stemming: Si True, applique le stemming lors du prétraitement.

    Returns:
        Dict[str, Any]: L'index inversé enrichi.
    """
    index_inverse: Dict[str, Dict[str, Any]] = {}

    for doc_id, content in documents.items():
        if isinstance(content, str):
            tokens = preprocess(content, use_stemming=use_stemming)
        else:
            tokens = content

        # Calcul des fréquences de termes locales (TF) pour ce document
        tf_doc: Dict[str, int] = {}
        for token in tokens:
            tf_doc[token] = tf_doc.get(token, 0) + 1

        # Mise à jour des entrées de l'index inversé
        for token, count in tf_doc.items():
            if token not in index_inverse:
                index_inverse[token] = {
                    "df": 0,
                    "postings": {}
                }
            index_inverse[token]["postings"][doc_id] = {"tf": count}

    # Calcul du Document Frequency (DF) pour chaque terme
    for term, data in index_inverse.items():
        data["df"] = len(data["postings"])

    return index_inverse


def format_index_display(index_inverse: Dict[str, Any], max_terms: Optional[int] = None) -> str:
    """
    Formate l'index inversé enrichi sous une forme textuelle lisible.
    Exemple : TERM  DF  POSTINGS
              retrieval  3   D1(tf=2), D3(tf=1), D7(tf=3)
    """
    lines = [f"{'TERM':<20} {'DF':<6} {'POSTINGS'}"]
    lines.append("-" * 65)

    sorted_terms = sorted(index_inverse.keys())
    if max_terms is not None:
        sorted_terms = sorted_terms[:max_terms]

    for term in sorted_terms:
        df = index_inverse[term]["df"]
        postings = index_inverse[term]["postings"]
        postings_str = ", ".join([f"{doc_id}(tf={info['tf']})" for doc_id, info in sorted(postings.items())])
        lines.append(f"{term:<20} {df:<6} {postings_str}")

    return "\n".join(lines)


# ==============================================================================
# EXERCICE 4 — Calcul de IDF
# ==============================================================================

def calculer_idf(terme: str, index_inverse: Dict[str, Any], N: int) -> float:
    """
    Calcule l'IDF (Inverse Document Frequency) pour un terme t donné :
    IDF(t) = log10(N / DF(t))

    Args:
        terme (str): Le terme recherché.
        index_inverse (dict): L'index inversé enrichi.
        N (int): Nombre total de documents dans la collection.

    Returns:
        float: Valeur de l'IDF (0.0 si absent de la collection).
    """
    if terme not in index_inverse:
        return 0.0
    df = index_inverse[terme]["df"]
    if df == 0:
        return 0.0
    return math.log10(N / df)


def get_df_idf_stats(index_inverse: Dict[str, Any], N: int) -> Tuple[str, str, str, Dict[str, Tuple[int, float]]]:
    """
    Calcule l'IDF pour chaque terme du vocabulaire et identifie :
    - le terme de DF maximal
    - le terme de DF minimal
    - le terme de plus grand IDF
    """
    table: Dict[str, Tuple[int, float]] = {}
    max_df_val = -1
    min_df_val = float("inf")
    max_idf_val = -1.0

    max_df_term = ""
    min_df_term = ""
    max_idf_term = ""

    for term in sorted(index_inverse.keys()):
        df = index_inverse[term]["df"]
        idf = calculer_idf(term, index_inverse, N)
        table[term] = (df, idf)

        if df > max_df_val:
            max_df_val = df
            max_df_term = term

        if df < min_df_val:
            min_df_val = df
            min_df_term = term

        if idf > max_idf_val:
            max_idf_val = idf
            max_idf_term = term

    return max_df_term, min_df_term, max_idf_term, table


# ==============================================================================
# EXERCICE 5 — Calcul des poids TF-IDF
# ==============================================================================

def poids_tfidf(terme: str, doc_id: str, index_inverse: Dict[str, Any], N: int) -> float:
    """
    Calcule le poids TF-IDF d'un terme dans un document selon la pondération logarithmique imposée :
    w(t,d) = (1 + log10(TF(t,d))) * IDF(t)  si TF(t,d) > 0 sinon 0

    Args:
        terme (str): Le terme.
        doc_id (str): L'identifiant du document.
        index_inverse (dict): L'index inversé enrichi.
        N (int): Nombre total de documents.

    Returns:
        float: Le poids w(t,d).
    """
    if terme not in index_inverse:
        return 0.0

    postings = index_inverse[terme]["postings"]
    if doc_id not in postings:
        return 0.0

    tf = postings[doc_id]["tf"]
    if tf > 0:
        idf = calculer_idf(terme, index_inverse, N)
        return (1.0 + math.log10(tf)) * idf

    return 0.0


def document_terms_tfidf_ranking(doc_id: str, index_inverse: Dict[str, Any], N: int) -> List[Dict[str, Any]]:
    """
    Retourne la liste des termes présents dans un document doc_id,
    ordonnés par poids TF-IDF décroissant.
    """
    terms_info = []
    for term, data in index_inverse.items():
        if doc_id in data["postings"]:
            tf = data["postings"][doc_id]["tf"]
            df = data["df"]
            idf = calculer_idf(term, index_inverse, N)
            tfidf = poids_tfidf(term, doc_id, index_inverse, N)
            terms_info.append({
                "term": term,
                "tf": tf,
                "df": df,
                "idf": idf,
                "tfidf": tfidf
            })

    # Tri décroissant par TF-IDF, puis par TF décroissant
    terms_info.sort(key=lambda x: (x["tfidf"], x["tf"]), reverse=True)
    return terms_info


# ==============================================================================
# EXERCICE 6 — Construction des vecteurs documents
# ==============================================================================

def document_vector(doc_id: str, vocabulary: List[str], index_inverse: Dict[str, Any], N: int) -> List[float]:
    """
    Construit le vecteur TF-IDF d'un document dans l'espace vectoriel R^|V|
    déterminé par l'ordre lexicographique du vocabulaire global :
    Di = (w1,i, w2,i, ..., w|V|,i)

    Args:
        doc_id (str): L'identifiant du document.
        vocabulary (List[str]): Vocabulaire trié de la collection.
        index_inverse (dict): Index inversé enrichi.
        N (int): Nombre total de documents.

    Returns:
        List[float]: Vecteur des poids TF-IDF.
    """
    vector = []
    for term in vocabulary:
        weight = poids_tfidf(terme=term, doc_id=doc_id, index_inverse=index_inverse, N=N)
        vector.append(weight)
    return vector


# ==============================================================================
# EXERCICE 7 — Représentation vectorielle d'une requête
# ==============================================================================

def query_vector(query: str, vocabulary: List[str], index_inverse: Dict[str, Any], N: int, use_stemming: bool = False) -> List[float]:
    """
    Construit la représentation vectorielle d'une requête dans le même espace vectoriel
    que les documents.
    Applique le même prétraitement, calcule TF(t, q) et le poids w(t, q) = (1 + log10(TF(t, q))) * IDF(t).
    Les termes de la requête n'appartenant pas au vocabulaire de la collection sont ignorés (poids 0).

    Args:
        query (str): Texte de la requête utilisateur.
        vocabulary (List[str]): Vocabulaire trié de la collection.
        index_inverse (dict): Index inversé enrichi.
        N (int): Nombre total de documents.
        use_stemming (bool): Active ou non le racinage.

    Returns:
        List[float]: Vecteur TF-IDF de la requête.
    """
    q_tokens = preprocess(query, use_stemming=use_stemming)

    # Calcul du TF dans la requête
    tf_q: Dict[str, int] = {}
    for tok in q_tokens:
        tf_q[tok] = tf_q.get(tok, 0) + 1

    vector_q = []
    for term in vocabulary:
        if term in tf_q and tf_q[term] > 0:
            idf = calculer_idf(term, index_inverse, N)
            w_tq = (1.0 + math.log10(tf_q[term])) * idf
            vector_q.append(w_tq)
        else:
            vector_q.append(0.0)

    return vector_q


# ==============================================================================
# EXERCICE 8 — Similarité cosinus
# ==============================================================================

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """
    Implémente manuellement la similarité cosinus entre deux vecteurs sans sklearn :
    sim(v1, v2) = (v1 . v2) / (||v1|| * ||v2||)

    Prévoit explicitement le cas d'un vecteur de norme nulle (évite la division par zéro en retournant 0.0).

    Args:
        v1 (List[float]): Premier vecteur numérique.
        v2 (List[float]): Deuxième vecteur numérique.

    Returns:
        float: Score de similarité compris entre 0.0 et 1.0 (pour des vecteurs positifs).
    """
    if len(v1) != len(v2):
        raise ValueError("Les vecteurs doivent être de même dimension.")

    # 1. Produit scalaire
    dot_product = sum(a * b for a, b in zip(v1, v2))

    # 2. Norme de v1
    norm_v1 = math.sqrt(sum(a * a for a in v1))

    # 3. Norme de v2
    norm_v2 = math.sqrt(sum(b * b for b in v2))

    # Cas d'un vecteur de norme nulle
    if norm_v1 == 0.0 or norm_v2 == 0.0:
        return 0.0

    # 4. Cosinus
    cosinus = dot_product / (norm_v1 * norm_v2)
    return cosinus


# ==============================================================================
# EXERCICE 9 — Ranking et Top-k
# ==============================================================================

def search(query: str,
           documents: Dict[str, str],
           index_inverse: Dict[str, Any],
           vocabulary: List[str],
           N: int,
           k: int = 5,
           use_stemming: bool = False,
           precomputed_doc_vectors: Optional[Dict[str, List[float]]] = None) -> List[Tuple[str, float, str]]:
    """
    Effectue une recherche vectorielle complète avec ranking Top-k :
    1. Prétraite la requête
    2. Construit le vecteur requête
    3. Calcule sim(q, Di) pour chaque document
    4. Élimine les scores nuls
    5. Trie les résultats par score décroissant
    6. Retourne les k premiers résultats

    Args:
        query (str): Requête utilisateur.
        documents (dict): Dictionnaire {doc_id: raw_text}.
        index_inverse (dict): Index inversé enrichi.
        vocabulary (list): Liste ordonnée des termes du vocabulaire.
        N (int): Nombre total de documents.
        k (int): Nombre de résultats à retourner.
        use_stemming (bool): Option de stemming.
        precomputed_doc_vectors (dict, optional): Vecteurs documents précalculés pour optimiser.

    Returns:
        List[Tuple[str, float, str]]: Liste de tuples (doc_id, score, document_text).
    """
    # Construction du vecteur requête
    v_q = query_vector(query, vocabulary, index_inverse, N, use_stemming=use_stemming)

    # Calcul des scores de similarité pour tous les documents
    scored_results = []
    for doc_id, text in documents.items():
        if precomputed_doc_vectors is not None and doc_id in precomputed_doc_vectors:
            v_d = precomputed_doc_vectors[doc_id]
        else:
            v_d = document_vector(doc_id, vocabulary, index_inverse, N)

        score = cosine_similarity(v_q, v_d)
        if score > 0.0:
            scored_results.append((doc_id, score, text))

    # Tri par score décroissant
    scored_results.sort(key=lambda item: item[1], reverse=True)

    # Retourner les k premiers résultats
    return scored_results[:k]


# ==============================================================================
# EXERCICE 11 — Recherche Booléenne AND
# ==============================================================================

def boolean_and_search(query: str, index_inverse: Dict[str, Any], use_stemming: bool = False) -> List[str]:
    """
    Implémente la recherche booléenne AND stricte par intersection des postings lists.

    Args:
        query (str): Requête textuelle (ex: "inverted index search").
        index_inverse (dict): Index inversé enrichi.
        use_stemming (bool): Option de stemming.

    Returns:
        List[str]: Liste des DocIDs contenant TOUS les termes de la requête.
    """
    q_tokens = preprocess(query, use_stemming=use_stemming)
    if not q_tokens:
        return []

    # Vérification si un des termes est hors-vocabulaire
    matching_sets = []
    for term in q_tokens:
        if term not in index_inverse:
            return []  # Intersection vide car un terme obligatoire est absent de la collection
        matching_sets.append(set(index_inverse[term]["postings"].keys()))

    # Intersection successive de tous les ensembles de documents
    result_docs = set.intersection(*matching_sets)
    return sorted(list(result_docs))


# ==============================================================================
# EXERCICE 12 — Challenge : FIGL Search Engine V2 (Interactive CLI)
# ==============================================================================

class FIGLSearchEngineV2:
    """
    Moteur de recherche interactif FIGL Search Engine V2.
    L'index et les vecteurs documents sont construits une seule fois au lancement.
    Plusieurs requêtes consécutives peuvent ensuite être exécutées sans réindexer.
    """

    def __init__(self, tsv_file_path: str, use_stemming: bool = True):
        self.tsv_file_path = tsv_file_path
        self.use_stemming = use_stemming

        # 1. Chargement du corpus
        self.documents = load_documents_from_tsv(tsv_file_path)
        self.N = len(self.documents)

        # 2. Construction de l'index enrichi (une seule fois)
        self.index_inverse = construire_index_enrichi(self.documents, use_stemming=self.use_stemming)
        self.vocabulary = sorted(self.index_inverse.keys())

        # 3. Précalcul des vecteurs documents pour des requêtes instantanées
        self.doc_vectors = {
            doc_id: document_vector(doc_id, self.vocabulary, self.index_inverse, self.N)
            for doc_id in self.documents
        }

    def run_query(self, query: str, k: int = 5) -> List[Tuple[str, float, str]]:
        return search(
            query=query,
            documents=self.documents,
            index_inverse=self.index_inverse,
            vocabulary=self.vocabulary,
            N=self.N,
            k=k,
            use_stemming=self.use_stemming,
            precomputed_doc_vectors=self.doc_vectors
        )

    def interactive_cli(self):
        """
        Lance la console interactive conformément à la maquette demandée.
        """
        print("========================================")
        print("         FIGL SEARCH ENGINE V2          ")
        print("========================================")
        print(f"Documents indexed : {self.N}")
        print(f"Vocabulary size   : {len(self.vocabulary)}")
        print(f"Stemming enabled  : {self.use_stemming}")
        print("Tapez 'exit' ou 'quit' pour quitter.")
        print("Vous pouvez spécifier k (ex: /k 3, /k 5, /k 10). Par défaut: k=5.")
        print("----------------------------------------\n")

        current_k = 5

        while True:
            try:
                user_input = input("Query > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nArrêt du moteur FIGL Search Engine V2. Au revoir !")
                break

            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "q"):
                print("Au revoir !")
                break

            if user_input.startswith("/k"):
                parts = user_input.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    current_k = int(parts[1])
                    print(f"[Configuration] Paramètre k mis à jour : Top-{current_k}\n")
                else:
                    print("[Erreur] Format attendu : /k 3 ou /k 5 ou /k 10\n")
                continue

            results = self.run_query(user_input, k=current_k)

            if not results:
                print(f"Aucun résultat trouvé pour la requête : '{user_input}'.\n")
                continue

            print(f"\nTOP-{len(results)} RESULTS for '{user_input}':")
            for rank, (doc_id, score, text) in enumerate(results, start=1):
                print(f"{rank}. {doc_id}  score = {score:.4f}")
                print(f"   {text}")
            print("-" * 40 + "\n")
