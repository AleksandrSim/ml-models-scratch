import global_variables


def dictionaryUniqueValues(listOfDict, keyName):
    # this function accepts a dictionary and a key and will return the unique values of the key in a list
    uniqueValues = []
    # listOfDict is a list of dicitonaries
    for myDictionary in listOfDict:
        for key, value in myDictionary.items():
            if key == keyName:
                # we found the key we are looking for, now check its value and if it not in the
                # unique dictionary add it there
                if value not in uniqueValues:
                    uniqueValues.append(value)
    return uniqueValues


def dictionaryKeyCounter(listOfDict, keyName):
    counter = 0
    for myDictionary in listOfDict:
        for key, value in myDictionary.items():
            if key == keyName:
                counter = counter + 1
    return counter


def dictionaryValueCounter(listOfDict, keyName, targetValue):
    counter = 0
    for myDictionary in listOfDict:
        for key, value in myDictionary.items():
            if key == keyName:
                if value == targetValue:
                    counter = counter + 1
    return counter


def dictionaryClassLabelComp(listofDict, attr, targetValue, attrValue):
    counter = 0
    for myDictionary in listofDict:
        found = False
        for key, value in myDictionary.items():
            if key == attr:
                if value == targetValue:
                    found = True
            if key == global_variables.target_attribute and found:
                if value == attrValue:
                    counter = counter + 1
    return counter


def listAllAttributes(ld):
    # this function accepts a list of dictionaries and returns all the attributes
    uniqueAttributes = []
    # listOfDict is a list of dicitonaries
    for myDictionary in ld:
        for key, value in myDictionary.items():
            if key not in uniqueAttributes:
                uniqueAttributes.append(key)
    return uniqueAttributes


def dictionary_key_max_value(dict):
    dict_values = list(dict.values())
    dict_keys = list(dict.keys())
    return dict_keys[dict_values.index(max(dict_values))]


def removeAttributeFromList(listOfDict, keyOfDict, valueOfDict):
    newListToReturn = []

    for myDictionary in listOfDict:
        newDictionary = {}
        addDictionary = True
        for key, value in myDictionary.items():
            if key == keyOfDict and value != valueOfDict:
                addDictionary = False
            else:
                newDictionary[key] = value
        if addDictionary:
            # print(keyOfDict, newDictionary)
            newListToReturn.append(newDictionary)

    return newListToReturn


def removeColumnFromDict(listOfDict, column):
    for myDictionary in listOfDict:
        del myDictionary[column]
    return listOfDict
