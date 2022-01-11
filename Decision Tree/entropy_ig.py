import list_dictionary_functions
import math
import global_variables


def calculateMaxInformationGain(ld):
    # 4) In this step we need to calculate the entropy for the complete dataset
    overallEntropy = datasetEntropyCalculator(ld)
    # 5) In the next step we need to calculate the entropy for the whole dataset, for each attribute
    # and store each entropy in a dictionary
    entropy = {}
    # 6) Next step is for us to find the information gain for each one of the attributes based on the calculated
    # entropy
    iG = {}
    attributes = list_dictionary_functions.listAllAttributes(ld)
    # Remove the target variable if it is included in the list
    if global_variables.target_attribute in attributes:
        attributes.remove(global_variables.target_attribute)

    for attr in attributes:
        entropy[attr] = entropyCalculator(ld, attr)
        # entropy[attr] = conditional_entropy(ld, attr, globalTargetAttribute)
        # The pseudocode for IG is: Information Gain of Attribute = entropy of whole dataset - entropy of attribute
        iG[attr] = overallEntropy - entropy[attr]
    # print some statements to check validity
    '''
    print("\n")
    print("The entropy for the whole dataset is: ", overallEntropy)
    print("The entropy for every attribute is: ", entropy)
    print("The Information Gain for every attribute is: ", iG)
    print("The attribute with Max Information Gain is: ", max(iG, key=iG.get))
    print("\n")
    '''

    # 7) Now we need to find the max value in the values of the Information gain dictionary that we formed
    # which will give us the name of the attribute wit the max value that should become the root once
    # building our tree
    return max(iG, key=iG.get), max(iG.values())


def datasetEntropyCalculator(ld):
    # Calculates the entropy for the whole dataset
    wholeDatasetEntropy = 0
    values = list_dictionary_functions.dictionaryUniqueValues(ld, global_variables.target_attribute)
    for value in values:
        fraction = list_dictionary_functions.dictionaryValueCounter(ld, global_variables.target_attribute,
                                                                    value) / list_dictionary_functions.dictionaryKeyCounter(
            ld, global_variables.target_attribute)
        wholeDatasetEntropy += -fraction * math.log2(fraction)
    return wholeDatasetEntropy


# define the entropy method which will take as an input the dataframe Pandas and the columnName
def entropyCalculator(ld, targetAttr):
    target_variables = list_dictionary_functions.dictionaryUniqueValues(ld, global_variables.target_attribute)
    variables = list_dictionary_functions.dictionaryUniqueValues(ld, targetAttr)
    overallEntropy = 0

    for variable in variables:
        attributeEntropy = 0
        for target_variable in target_variables:
            # Numerator, function returns count where xy
            numerator = list_dictionary_functions.dictionaryClassLabelComp(ld, targetAttr, variable, target_variable)  # numerator
            denominator = list_dictionary_functions.dictionaryValueCounter(ld, targetAttr, variable)  # denominator
            numerator += global_variables.epsilon
            denominator += global_variables.epsilon
            fraction = numerator / denominator  # pi
            attributeEntropy += -fraction * math.log2(fraction)  # This calculates entropy
        fraction2 = denominator / len(ld)
        overallEntropy += -fraction2 * attributeEntropy  # Sums up all the entropy

    return abs(overallEntropy)
