# TP3 : Construction d'un Moteur de Recherche Vectoriel avec TF-IDF

**Filière :** FIGL2 — Indexation et Recherche d'Information  
**Enseignante :** Dr. F. Chaabane  
**Auteur :** Diya135  

---

## 📖 Sommaire
1. [Présentation générale](#-présentation-générale)
2. [Structure du projet](#-structure-du-projet)
3. [Installation et exécution](#-installation-et-exécution)
4. [Détail des exercices & Réponses théoriques](#-détail-des-exercices--réponses-théoriques)
   - [Exercice 1 — Constitution du corpus](#exercice-1--constitution-du-corpus)
   - [Exercice 2 — Pipeline de prétraitement](#exercice-2--pipeline-de-prétraitement)
   - [Exercice 3 — Index inversé enrichi](#exercice-3--index-inversé-enrichi)
   - [Exercice 4 — Calcul de l'IDF](#exercice-4--calcul-de-lidf)
   - [Exercice 5 — Calcul des poids TF-IDF](#exercice-5--calcul-des-poids-tf-idf)
   - [Exercice 6 — Vecteurs documents & Sparsité](#exercice-6--vecteurs-documents--sparsité)
   - [Exercice 7 — Représentation vectorielle d'une requête](#exercice-7--représentation-vectorielle-dune-requête)
   - [Exercice 8 — Similarité cosinus](#exercice-8--similarité-cosinus)
   - [Exercice 9 — Ranking Top-k & 5 requêtes de test](#exercice-9--ranking-top-k--5-requêtes-de-test)
   - [Exercice 10 — Expérience : model vs models](#exercice-10--expérience--model-vs-models)
   - [Exercice 11 — Comparaison Booléen AND vs Vectoriel TF-IDF](#exercice-11--comparaison-booléen-and-vs-vectoriel-tf-idf)
   - [Exercice 12 — Challenge : FIGL Search Engine V2](#exercice-12--challenge--figl-search-engine-v2)
5. [Question de synthèse : Limites de TF-IDF et apport de BM25](#-question-de-synthèse--limites-de-tf-idf-et-apport-de-bm25)

---

## 🎯 Présentation générale

Ce projet implémente un système complet de **recherche d'information vectorielle (Vector Space Model)** basé sur la pondération **TF-IDF logarithmique** et la mesure de **similarité cosinus**, développé entièrement en Python pur sans recourir à des fonctions toutes faites de calcul de similarité ou de ranking (telles que celles de `scikit-learn`).

Le système permet de :
- Indexer automatiquement une collection de documents textuels sous forme d'**index inversé enrichi** (mémorisant les fréquences locales $TF$ et documentaires $DF$).
- Pondérer chaque terme selon la formule :
  $$w(t, d) = (1 + \log_{10}(TF(t, d))) \times \log_{10}\left(\frac{N}{DF(t)}\right)$$
- Projeter documents et requêtes dans l'espace vectoriel $\mathbb{R}^{|V|}$.
- Classer les documents par similarité cosinus avec gestion stricte des cas limites (vecteurs nuls, mots hors vocabulaire).
- Comparer rigoureusement les approches avec et sans racinage morphologique (*stemming*), ainsi que le modèle booléen strict versus le modèle vectoriel graduel.

---

## 📂 Structure du projet

```bash
TP3/
├── documents.tsv                        # Corpus de test (25 documents au format DocID \t Texte)
├── search_engine.py                    # Module cœur contenant toutes les fonctions algorithmiques
├── main.py                             # Script d'exécution complet des exercices et CLI interactive
├── TP3_Moteur_Recherche_Vectoriel.ipynb # Notebook Jupyter riche et pré-exécuté
├── index_inverse.json                  # Index inversé enrichi généré automatiquement
└── README.md                           # Rapport exhaustif du TP et documentation technique
```

---

## 🚀 Installation et exécution

### Prérequis
- Python 3.10+
- Bibliothèques recommandées : `nltk`, `pandas`

```bash
pip install nltk pandas
```

### Exécution du script principal
Pour exécuter l'ensemble des exercices 1 à 11 avec les calculs statistiques et les affichages formatés :
```bash
python main.py
```

### Lancement du moteur de recherche interactif (Challenge V2)
Pour tester des requêtes arbitraires en temps réel :
```bash
python main.py --interactive
# ou
python main.py -i
```

Commandes dans le mode interactif :
- `Query > <votre requête>` : lance la recherche avec similarité cosinus.
- `/k 3`, `/k 5`, `/k 10` : ajuste le nombre de résultats affichés dans le Top-$k$.
- `exit` ou `quit` : quitte la console.

---

## 📝 Détail des exercices & Réponses théoriques

### Exercice 1 — Constitution du corpus
Le corpus est constitué de 25 documents thématiquement ciblés sur le domaine de la recherche d'information, des bases vectorielles, des modèles booléens, de TF-IDF, de BM25 et de la recherche dense.

#### Question théorique :
> **Pourquoi un corpus composé de documents proches thématiquement est-il plus intéressant pour tester un algorithme de ranking ?**
- **Réponse :**  
  Dans un corpus hétérogène (mélangeant cuisine, sport et finance), la discrimination documentaire est quasi-triviale : la simple présence d'un mot-clé rare sépare immédiatement les catégories sans tester la finesse du modèle de ranking.  
  À l'inverse, dans un corpus **thématiquement homogène**, la plupart des documents partagent le même vocabulaire général (*search, retrieval, document, query, vector*). L'algorithme est donc forcé d'exploiter la subtilité des pondérations TF-IDF (rareté globale $IDF$ et intensité locale $TF$) et l'orientation angulaire de la similarité cosinus pour établir un classement pertinent.

---

### Exercice 2 — Pipeline de prétraitement
La fonction `preprocess(text, use_stemming=False)` réalise les 4 étapes requises :
1. Mise en minuscules (`text.lower()`).
2. Tokenisation par expression régulière alphanumérique (`\b[a-zA-Z0-9]+\b`).
3. Suppression des mots vides anglais (*stopwords*).
4. Normalisation morphologique optionnelle (Porter Stemmer).

#### Question théorique :
> **Pourquoi faut-il appliquer exactement le même pipeline de prétraitement aux documents et aux requêtes ?**
- **Réponse :**  
  Le modèle vectoriel repose sur une correspondance terme-à-terme dans une base canonique $\mathbb{R}^{|V|}$. Si la requête et les documents subissent des prétraitements distincts (ex: casse différente, ou racinage appliqué sur la requête mais absent des documents), des mots sémantiquement identiques comme `model` et `models` ne partageront pas la même dimension vectorielle. Le produit scalaire $\vec{q} \cdot \vec{d}$ sera alors nul sur ces termes, provoquant un silence documentaire injustifié (*chute critique du rappel*).

---

### Exercice 3 — Index inversé enrichi
L'index inversé est structuré selon le schéma demandé :
```python
index_inverse = {
    "information": {
        "df": 6,
        "postings": {
            "D1": {"tf": 1},
            "D6": {"tf": 1},
            "D7": {"tf": 1},
            "D19": {"tf": 1},
            "D22": {"tf": 1}
        }
    },
    ...
}
```

Extrait de l'affichage lisible :
```
TERM                 DF     POSTINGS
-----------------------------------------------------------------
boolean              2      D3(tf=1), D20(tf=1)
embeddings           3      D8(tf=1), D9(tf=1), D18(tf=1)
indexes              4      D1(tf=1), D10(tf=1), D11(tf=1), D25(tf=1)
information          6      D1(tf=1), D6(tf=1), D7(tf=1), D19(tf=1), D22(tf=1)
retrieval            8      D1(tf=1), D3(tf=1), D6(tf=1), D7(tf=1), D8(tf=1), D10(tf=1), D16(tf=2), D19(tf=1), D22(tf=1)
vector               5      D2(tf=2), D9(tf=1), D12(tf=1), D18(tf=2), D23(tf=1)
```

#### Question théorique :
> **Quelle est la différence entre $TF(t,d)$ et $DF(t)$ ?**
- **Réponse :**  
  - **$TF(t,d)$ (Term Frequency)** : Mesure **locale** intra-document. Elle compte le nombre de fois où le terme $t$ apparaît dans un document spécifique $d$. Un $TF$ élevé indique que le terme est un sujet central de ce document.
  - **$DF(t)$ (Document Frequency)** : Mesure **globale** inter-documents. Elle compte le nombre de documents de toute la collection contenant le terme $t$ au moins une fois. Un $DF$ élevé révèle un terme omniprésent et peu spécifique, donc peu informatif.

---

### Exercice 4 — Calcul de l'IDF
Formule :
$$IDF(t) = \log_{10}\left(\frac{N}{DF(t)}\right)$$

Exemples de valeurs calculées ($N = 25$) :
| Terme | DF | IDF ($\log_{10}(N/DF)$) |
| :--- | :--- | :--- |
| `retrieval` | 8 | 0.4949 |
| `information` | 6 | 0.6198 |
| `search` | 6 | 0.6198 |
| `vector` | 5 | 0.6990 |
| `indexes` | 4 | 0.7959 |
| `embeddings` | 3 | 0.9208 |
| `boolean` | 2 | 1.0969 |
| `bm25` | 2 | 1.0969 |
| `databases` | 2 | 1.0969 |

**Extrêmes identifiés automatiquement :**
- Terme de $DF$ maximal : `retrieval` ($DF = 8$, $IDF = 0.4949$)
- Termes de $DF$ minimal : Termes apparaissant dans 1 seul document ($DF = 1$, $IDF = 1.3979$)
- Termes de plus grand $IDF$ : $IDF_{\max} = \log_{10}(25/1) \approx 1.3979$

#### Question théorique :
> **Pourquoi un terme rare dans la collection reçoit-il généralement un IDF plus élevé ?**
- **Réponse :**  
  Mathématiquement, la fonction $\log_{10}(N/DF)$ est strictement décroissante par rapport à $DF$. Lorsque $DF$ est petit, le rapport $N/DF$ est maximal.  
  Sur le plan de la théorie de l'information (Spärck Jones), un terme rare possède un pouvoir discriminant très élevé : s'il figure dans la requête de l'utilisateur, les documents qui le contiennent ont une probabilité bien plus forte d'être ciblés et pertinents que s'il s'agissait d'un mot courant.

---

### Exercice 5 — Calcul des poids TF-IDF
Formule :
$$w(t,d) = (1 + \log_{10}(TF(t,d))) \times IDF(t) \quad \text{si } TF(t,d) > 0 \text{ sinon } 0$$

Pour le document **D8** (*"Dense retrieval represents queries and documents using embeddings"*) :
| TERM | TF | DF | IDF | TF-IDF |
| :--- | :--- | :--- | :--- | :--- |
| `dense` | 1 | 2 | 1.0969 | **1.0969** |
| `embeddings` | 1 | 3 | 0.9208 | **0.9208** |
| `queries` | 1 | 3 | 0.9208 | **0.9208** |
| `represents` | 1 | 2 | 1.0969 | **1.0969** |
| `documents` | 1 | 6 | 0.6198 | **0.6198** |
| `retrieval` | 1 | 8 | 0.4949 | **0.4949** |

#### Question théorique :
> **Les termes ayant le plus grand TF sont-ils toujours ceux qui ont le plus grand TF-IDF ? Justifier à partir de vos résultats.**
- **Réponse :**  
  **NON.** Le score TF-IDF est le produit de la fréquence locale ($TF$) et de la rareté globale ($IDF$). Même si un terme possède un $TF$ élevé dans un document, si ce terme est générique à toute la collection ($DF$ grand), son $IDF$ sera très faible, annihilant le poids final.  
  Dans le document D8, tous les mots ont $TF = 1$, mais `dense` obtient $w = 1.0969$ tandis que `retrieval` n'obtient que $w = 0.4949$. Si un terme courant avait $TF = 2$, son poids serait $(1 + \log_{10}(2)) \times 0.4949 \approx 0.64$, ce qui reste nettement inférieur au poids de `dense` ($1.0969$) avec un $TF$ de seulement 1 !

---

### Exercice 6 — Vecteurs documents & Sparsité
Chaque document $D_i$ est projeté dans l'espace $\mathbb{R}^{|V|}$ ($|V| = 91$ dimensions sans stemming) :
- **D1** : 86 composantes nulles sur 91 (**94.5% de sparsité**)
- **D2** : 85 composantes nulles sur 91 (**93.4% de sparsité**)
- **D3** : 87 composantes nulles sur 91 (**95.6% de sparsité**)

#### Question théorique :
> **Pourquoi les vecteurs TF-IDF de documents sont-ils généralement creux (*sparse*) ?**
- **Réponse :**  
  L'espace vectoriel possède une dimension égale au nombre total de mots distincts de toute la collection ($|V|$). Un document individuel n'est qu'une courte séquence textuelle (une phrase ou un paragraphe) qui n'emploie qu'une fraction infime du vocabulaire global. Toutes les autres dimensions reçoivent un $TF = 0$, d'où une prépondérance écrasante de zéros.

---

### Exercice 7 — Représentation vectorielle d'une requête
Pour la requête : $q = \text{"information retrieval model"}$  
Termes non nuls et poids dans $\mathbb{R}^{|V|}$ :
- `information` : $w(t,q) = 0.6198$ ($IDF = 0.6198$)
- `retrieval` : $w(t,q) = 0.4949$ ($IDF = 0.4949$)
- `model` : $w(t,q) = 1.0969$ ($IDF = 1.0969$)

#### Question théorique :
> **Que doit faire le programme lorsqu'un terme de la requête n'appartient pas au vocabulaire de la collection ?**
- **Réponse :**  
  Un terme absent de la collection possède un $DF = 0$ (terme OOV - *Out Of Vocabulary*). Aucun document ne le contient. Le programme doit l'ignorer lors de la projection vectorielle (poids affecté = 0) car il ne peut contribuer à aucune intersection de produit scalaire avec les documents existants.

---

### Exercice 8 — Similarité cosinus
Formule implémentée sans `sklearn` :
$$\text{sim}(\vec{q}, \vec{d}) = \frac{\sum_{i=1}^{|V|} q_i \cdot d_i}{\sqrt{\sum_{i=1}^{|V|} q_i^2} \times \sqrt{\sum_{i=1}^{|V|} d_i^2}}$$

Cas limite : si $\|\vec{q}\| = 0$ ou $\|\vec{d}\| = 0$, la fonction retourne explicitement `0.0` afin d'éviter la division par zéro.

#### Question théorique :
> **Que signifient intuitivement des scores proches de 1 et proches de 0 ?**
- **Réponse :**  
  - **Score proche de 1** : L'angle entre $\vec{q}$ et $\vec{d}$ est presque nul ($\cos(0)=1$). Le document contient les termes clés recherchés dans des proportions très similaires à celles de la requête : pertinence maximale.
  - **Score = 0** : Les vecteurs sont orthogonaux (aucun mot en commun).
  - **Score proche de 0** : Intersection minime (seul un mot partagé, souvent de faible IDF ou dilué dans un document long).

---

### Exercice 9 — Ranking Top-k & 5 requêtes de test

#### 1. Query : `information retrieval model`
```
Rank   DocID    Score      Document
---------------------------------------------------------------------------
1      D2       0.4508     The vector space model represents documents and queries as vectors
2      D1       0.3340     Information retrieval systems use inverted indexes for efficient search
3      D7       0.1674     Neural information retrieval uses deep learning models
4      D6       0.1652     BM25 is a ranking function used in information retrieval
5      D19      0.1610     Information retrieval evaluation relies on precision recall and mean average precision
```

#### 2. Query : `inverted indexes for search`
```
Rank   DocID    Score      Document
---------------------------------------------------------------------------
1      D1       0.4723     Information retrieval systems use inverted indexes for efficient search
2      D10      0.4704     Search engines use inverted indexes to retrieve documents efficiently
3      D11      0.3204     Inverted indexes store postings lists mapping terms to document identifiers
4      D25      0.1698     Inverted index compression reduces memory footprint in large scale search engines
5      D5       0.0657     Search engines rank documents according to their relevance
```

#### 3. Query : `vector embeddings and similarity`
```
Rank   DocID    Score      Document
---------------------------------------------------------------------------
1      D9       0.6406     Vector databases support similarity search over embeddings
2      D8       0.4687     Dense retrieval represents queries and documents using embeddings
3      D18      0.3957     Vector embeddings map semantic meanings of queries and documents into continuous vector space
4      D12      0.3541     Cosine similarity calculates the cosine of the angle between two vectors
5      D23      0.2818     Approximate nearest neighbor algorithms speed up similarity search in large vector databases
```

#### 4. Query : `boolean retrieval operators`
```
Rank   DocID    Score      Document
---------------------------------------------------------------------------
1      D3       0.7831     Boolean retrieval uses AND OR and NOT operators
2      D20      0.2476     Boolean search models evaluate queries through exact logical expressions without ranking
3      D1       0.0653     Information retrieval systems use inverted indexes for efficient search
4      D10      0.0650     Search engines use inverted indexes to retrieve documents efficiently
5      D8       0.0566     Dense retrieval represents queries and documents using embeddings
```

#### 5. Query : `ranking function bm25`
```
Rank   DocID    Score      Document
---------------------------------------------------------------------------
1      D6       0.7285     BM25 is a ranking function used in information retrieval
2      D15      0.4801     BM25 ranking function incorporates document length normalization and term frequency saturation
3      D17      0.1086     Modern search engines rank documents using machine learning and deep ranking models
4      D20      0.1002     Boolean search models evaluate queries through exact logical expressions without ranking
```

---

### Exercice 10 — Expérience : model vs models

Requête : `information retrieval model`  
- **D2** : *"The vector space model represents documents and queries as vectors"* (contient le terme exact `model`)
- **D7** : *"Neural information retrieval uses deep learning models"* (contient `models`, `information` et `retrieval`)

| Mesure | Sans normalisation morphologique | Avec normalisation morphologique |
| :--- | :--- | :--- |
| **Score du document D2 (`model`)** | **0.4508** | 0.2605 |
| **Score du document D7 (`models`)** | 0.1674 | **0.4694** |
| **Rang du document D2 (`model`)** | **1** | 3 |
| **Rang du document D7 (`models`)** | 3 | **1** |

#### Question théorique :
> **Le prétraitement peut-il modifier le classement final ? Montrer l'effet observé à partir des scores et des rangs.**
- **Réponse :**  
  **OUI, et l'impact est spectaculaire.**  
  Sans stemming, `models` (dans D7) ne correspond pas à `model` (dans la requête). D7 n'est donc matché que sur `information retrieval`, son score est de **0.1674** et il est classé au **rang 3**, tandis que D2 prend la première place.  
  Dès que la normalisation morphologique (PorterStemmer) est activée, `model` et `models` sont tous deux ramenés à la racine commune `model`. D7 valide alors **l'ensemble des 3 termes de la requête** (`inform`, `retriev`, `model`) : son score s'envole à **0.4694** et il s'empare de la **1ère place du classement** !

---

### Exercice 11 — Comparaison Booléen AND vs Vectoriel TF-IDF

Requête testée (3 termes) : `inverted indexes search`

| Critère | Booléen AND | Vectoriel TF-IDF |
| :--- | :--- | :--- |
| **Document retrouvé / non retrouvé** | Binaire (Strict 0 ou 1) | Continu (Graduel) |
| **Correspondance partielle** | Non (Rejet si 1 seul terme manque) | Oui (Score calculé sur les termes partagés) |
| **Score de pertinence** | Aucun score (Ensemble plat) | Similarité cosinus continue $[0, 1]$ |
| **Classement des résultats** | Impossible | Tri ordonné par pertinence |

Résultats observés :
- **Booléen AND** : Retourne uniquement `['D1', 'D10']` (les deux seuls documents possédant les 3 termes simultanément).
- **Vectoriel TF-IDF** :
  1. `D1` (0.4723) — Possède les 3 termes.
  2. `D10` (0.4704) — Possède les 3 termes.
  3. `D11` (0.3204) — *"Inverted indexes store postings lists..."* : ne possède que 2 termes (`inverted` et `indexes`), mais est récupéré avec un score élevé au 3e rang.

#### Question théorique :
> **Un document contenant seulement deux termes importants d'une requête à trois termes peut-il être retrouvé par les deux modèles de la même manière ? Expliquer.**
- **Réponse :**  
  **NON.** Le modèle booléen AND exige strictement $T_1 \land T_2 \land T_3$. Le document D11 contenant 2 termes sur 3 est impitoyablement exclu (faux négatif / silence documentaire). Le modèle vectoriel, quant à lui, évalue l'intersection géométrique : chaque terme commun apporte une composante positive au produit scalaire, permettant à D11 d'être classé 3e.

---

### Exercice 12 — Challenge : FIGL Search Engine V2

Le moteur `FIGLSearchEngineV2` initialise et précalcule l'index enrichi et les représentations vectorielles une seule fois en mémoire au démarrage. Les requêtes utilisateur successives sont traitées instantanément en $O(N \cdot |V_{\text{active}}|)$ sans réindexation.

---

## 🔬 Question de synthèse : Limites de TF-IDF et apport de BM25

> *L'index inversé permet de retrouver les documents. TF-IDF et la similarité cosinus permettent maintenant de les classer.  
> **Quelles limites de TF-IDF pourraient justifier l'utilisation d'un modèle de ranking plus avancé comme BM25 ?***

Bien que TF-IDF avec similarité cosinus constitue le fondement historique des moteurs de recherche vectoriels, il souffre de trois limites structurelles majeures qui expliquent la supériorité systématique de **BM25** (Best Matching 25) dans les benchmarks d'information retrieval :

### 1. Saturation asymptotique du Term Frequency ($TF$)
- **Limite de TF-IDF :** Dans TF-IDF (même sous forme $\log(TF)$), la valeur continue d'augmenter indéfiniment avec le nombre d'occurrences. Un document artificiellement suroptimisé (*keyword stuffing*) qui répète 100 fois un mot-clé verra son score croître anormalement.
- **Solution BM25 :** BM25 introduit une borne supérieure asymptotique contrôlée par le paramètre $k_1$ ($1.2 \le k_1 \le 2.0$) :
  $$\text{Contribution } TF = \frac{TF(t,d) \cdot (k_1 + 1)}{TF(t,d) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
  Au fur et à mesure que $TF$ augmente, le gain marginal décroît rapidement pour saturer à $k_1 + 1$. Répéter un mot au-delà de 3 à 5 fois n'apporte quasiment plus de bonus de pertinence.

### 2. Normalisation réaliste de la longueur des documents
- **Limite de TF-IDF :** La similarité cosinus normalise par la norme euclidienne $\|d\| = \sqrt{\sum w(t,d)^2}$. Cette pénalisation purement géométrique pénalise excessivement les documents longs de qualité (qui abordent plusieurs sous-thèmes) par rapport aux documents très courts.
- **Solution BM25 :** BM25 compare la longueur du document $|D|$ à la **longueur moyenne de tous les documents de la collection** ($\text{avgdl}$), modulée par le paramètre $b$ ($0 \le b \le 1$, typiquement $0.75$). Si un document est long simplement parce qu'il est verbeux, il est pénalisé ; si la collection entière est constituée de documents longs, aucune pénalité injustifiée n'est appliquée.

### 3. Fondement probabiliste
- **Limite de TF-IDF :** TF-IDF est un modèle heuristique et géométrique sans justification statistique formelle sur la pertinence perçue par un utilisateur.
- **Solution BM25 :** BM25 est formellement dérivé du modèle probabiliste de Robertson-Spärck Jones (modèle 2-Poisson), estimant formellement l'espérance de la pertinence $P(R=1 \mid d, q)$.
