
def probabilityForPrediction(listOfDictionaries, label):
    dic = {}

    for i, k in enumerate(listOfDictionaries):
        dic[i] = k

    unique_values_feature = []
    for i in dic:
        if dic[i][label] not in unique_values_feature:
            unique_values_feature.append(dic[i][label])

    init = {i: 0 for i in unique_values_feature}

    unique_classes = []

    for i in dic:
        if dic[i]['Class'] not in unique_classes:
            unique_classes.append(dic[i]['Class'])

    for i in init:
        init[i] = {t: 0 for t in unique_classes}

    for i in dic:
        for k in init:
            for f in init[k]:
                if dic[i][label] == k:
                    if dic[i]['Class'] == f:
                        init[k][f] += 1

    d = {i: {} for i in init}

    for i in init:
        su = sum(init[i].values())
        for k in init[i]:
            if init[i][k] != 0:
                d[i][k] = init[i][k] / su
            else:
                d[i][k] = 0

    return d