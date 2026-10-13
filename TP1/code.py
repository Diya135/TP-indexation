import re

stopwords = {
"le", "la", "les", "un", "une",
"de", "des", "du", "et", "dans",
"par", "pour", "d", "l"
}

def charger_docs(chemain):
    
  documents = {}

  with open(chemain, "r", encoding="utf-8") as f:
      for ligne in f:
          ligne = ligne.strip()
          if ligne:
              parties = ligne.split(maxsplit=1)
              doc_id = parties[0]
              texte = parties[1]
              documents[doc_id] = texte
  return documents


# Implémentation d’une recherche naïve
def recherche_naive(terme, documents):
    resultats = []
    for doc_id, texte in documents.items():
        if terme.lower() in texte.lower():
            resultats.append(doc_id)
    return resultats

########Prétraitement des documents###########
def tokeniser(texte):
 return re.findall(r"\b\w+\b", texte.lower())

def supprimer_stopwords(tokens):
  result=[]
  for i in tokens:
    if i not in stopwords:
        result.append(i)
  return result
#########################################

def preprocess(texte):
 tokens = tokeniser(texte)
 tokens = supprimer_stopwords(tokens)
 return tokens




documents=charger_docs("documents.tsv")

#print("Nombre de documents :", len(documents))
#print(documents)

#termes = input("Entrez un terme : ")

#print(recherche_naive(terme, documents))
#listTermes=tokeniser(terme)
#print(f'la nouvelle terme {listTermes}')
#print(f'la nouvelle list est : {supprimer_stopwords(listTermes)}')

#print(preprocess(termes))

################ Creation et affichage d indexe inverse###########
def construire_index(documents):
    index_inverse = {}
    for doc_id, texte in documents.items():
        termes = preprocess(texte)
        for terme in set(termes):
            if terme not in index_inverse:
                index_inverse[terme] = []
            index_inverse[terme].append(doc_id)
    return index_inverse


def afficher_index(index_inverse):
    for terme in sorted(index_inverse):
        print(terme, "->", index_inverse[terme])

index_inv=construire_index(documents)

afficher_index(index_inv)
##############################################################


############### Recherche à l’aide de l’index inversé ######################
def rechercher(terme, index_inverse):
 return index_inverse.get(terme.lower(), [])

#print(rechercher(termes,index_inv))

##########################################################################



###########Requête à deux termes : intersection ################
def intersection(list1, list2): 
    i = 0 
    j = 0 
    resultat = [] 
 
    while i < len(list1) and j < len(list2): 
 
        if list1[i] == list2[j]: 
            resultat.append(list1[i]) 
            i += 1 
            j += 1 
 
        elif list1[i] < list2[j]: 
            i += 1 
 
        else: 
            j += 1 
 
    return resultat
#################################################################

terme1 = input("Entrez le premier terme : ").lower() 
terme2 = input("Entrez le deuxième terme : ").lower() 
 
docs1 = index_inv.get(terme1, []) 
docs2 = index_inv.get(terme2, []) 
 
print("Intersection :", intersection(docs1, docs2))



