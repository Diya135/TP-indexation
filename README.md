# Travaux Pratiques — Indexation et Recherche d'Information

**Filière :** FIGL2  
**Enseignante :** Dr. F. Chaabane  
**Étudiant :** Diya135  

Ce dépôt regroupe l'ensemble des travaux pratiques (TPs) réalisés dans le cadre du module **Indexation et Recherche d'Information**.

---

## 📁 Structure du Dépôt

```bash
TP-indexation/
│
├── TP3/                                      # TP3 : Construction d'un moteur de recherche vectoriel avec TF-IDF
│   ├── documents.tsv                         # Corpus thématique de test (25 documents)
│   ├── search_engine.py                     # Module d'indexation, TF-IDF et similarité cosinus
│   ├── main.py                              # Script d'exécution des exercices et CLI interactive
│   ├── TP3_Moteur_Recherche_Vectoriel.ipynb  # Notebook Jupyter complet et pré-exécuté
│   ├── index_inverse.json                   # Index inversé enrichi sauvegardé
│   └── README.md                            # Documentation détaillée et réponses aux questions théoriques
│
└── README.md                                 # Vue d'ensemble du dépôt
```

---

## 📚 Liste des TPs

### [TP3 : Construction d'un moteur de recherche vectoriel avec TF-IDF](./TP3/)
- **Objectifs :**
  - Enrichissement de l'index inversé ($TF$, $DF$, listes de postings).
  - Calcul automatique des poids TF-IDF logarithmiques et normalisation de requête.
  - Implémentation manuelle de la similarité cosinus (sans bibliothèques externes de ranking).
  - Ranking Top-$k$ de documents et filtrage des scores nuls.
  - Analyse comparative de la normalisation morphologique (*Stemming* : `model` vs `models`).
  - Comparaison entre recherche booléenne AND stricte et recherche vectorielle graduée.
  - Développement de l'interface console interactive `FIGL Search Engine V2`.
  - Analyse critique et théorique des limites de TF-IDF face à **BM25**.

Consultez le dossier [`TP3/`](./TP3/) pour accéder au code source, au notebook Jupyter et au rapport technique complet.
