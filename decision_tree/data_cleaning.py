import global_variables


def cleanUpData(listOfDict):
    # we use this funciton to remove the "?" from the data file, and we just drop the whole dictionary (row)
    pureData = []
    for myDict in listOfDict:
        newDict = {}
        addDictionary = True
        for key, value in myDict.items():
            if value == global_variables.missingValue:
                addDictionary = False
            else:
                newDict[key] = value
        if addDictionary:
            pureData.append(newDict)

    return pureData


def mostProbableValue(data):
    counts = {}

    for i in data:
        for feature in i:
            if feature != 'Class':
                counts[feature] = {}

    for i in data:
        for feature in i:
            if feature != "Class":
                for key in i[feature]:
                    if key not in counts[feature]:
                        counts[feature][key] = 0
                    if key in counts[feature]:
                        counts[feature][key] += 1

    most_probable = {}

    for feature in counts:
        ma = 0
        for i, k in counts[feature].items():
            current = k
            if current > ma:
                ma = current
        for i in counts[feature]:
            if counts[feature][i] == ma:
                most_probable[feature] = i

    for i in data:
        for feature in i:
            if i[feature] == '?':
                i[feature] = most_probable[feature]

    return data
