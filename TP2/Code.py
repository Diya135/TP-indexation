


import math


index_inverse = {
    "maison": [1, 2, 4],
    "jardin": [3, 4],
    "belle": [2, 4],
    "chat": [1],
    "chien": [3, 5, 7],
    "terrasse": [4, 6, 7]
}



# la boucle while peut parcourir tous les deux listes a travers i et j donc la complexite est O(M+N) avec M et N les longueurs des listes 
def intersection(list1, list2):
    i, j = 0, 0
    result = []
    
    while i < len(list1) and j < len(list2):
        if list1[i] == list2[j]:
            result.append(list1[i])
            i += 1
            j += 1
        elif list1[i] < list2[j]:
            i += 1
        else:
            j += 1
            
    return result


terme1 = input("Entrez le premier terme (ex: maison): ")
terme2 = input("Entrez le deuxième terme (ex: belle): ")



def intersection_skips(list1, list2):
    k1 = int(math.sqrt(len(list1)))
    k2 = int(math.sqrt(len(list2)))
    
    i, j = 0, 0
    result = []

    while i < len(list1) and j < len(list2):
        if list1[i] == list2[j]:
            result.append(list1[i])
            i += 1
            j += 1
        elif list1[i] < list2[j]:
          
            if i + k1 < len(list1) and list1[i + k1] <= list2[j]:
                i += k1
            else:
                i += 1
        else:
            if j + k2 < len(list2) and list2[j + k2] <= list1[i]:
                j += k2
            else:
                j += 1

    return result

l1 = index_inverse.get(terme1, [])
l2 = index_inverse.get(terme2, [])

print(f"Intersection : {intersection(l1, l2)}")

print("Résultat de l'intersection :", intersection_skips(l1, l2))