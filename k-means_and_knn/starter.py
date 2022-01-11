import data_transformation
import data_cleaning
import knn
import timer
import kmeans
import global_variables
import confusion_matrix
import softkmeans


def main():
    # We first have to read the data from the file
    # Gathering the training data first
    """************************** Reading the Data ********************************"""
    training_data = data_cleaning.read_data("train.csv")
    validation_data = data_cleaning.read_data("valid.csv")
    testing_data = data_cleaning.read_data("test.csv")
    sets = [training_data, validation_data, testing_data]

    print(
        "************************** Transforming the Data ********************************"
    )
    if global_variables.perform_dimensionality_reduction == "zero":
        # provide the training set to the zero_values_index_locator function to return the indexes
        zero_index_list = data_transformation.zero_values_index_locator(sets[0])
        sets = data_transformation.dimensionality_reduction(zero_index_list, sets)
        # Uncomment this line to use binary reduction too
        # sets = data_transformation.binary_conversion(sets)

    elif global_variables.perform_dimensionality_reduction == "variance":
        boring = data_transformation.theyre_boring(sets[0])
        sets = data_transformation.dimensional_smash(boring, sets)

    # final_predictions, labels = knn.knn(sets[0], sets[2], 'euclidean')
    # print(confusion_matrix(final_predictions, labels))

    print("************************** Calling KNN ********************************")
    if global_variables.call_KNN:

        for metric in global_variables.metrics:
            start_time = timer.timer()
            final_predictions, labels, accuracy = knn.knn(sets[0], sets[2], metric)
            stop_time = timer.timer()
            timer.print_timer(start_time, stop_time, "KNN", metric, accuracy)
            print("\n\n**************************")
            confusion_matrix_result = confusion_matrix.confusion_matrix(
                final_predictions, labels
            )
            # Pretty printing the confusion matrix
            confusion_matrix.pretty_print(confusion_matrix_result)
            # print("Accuracy of KNN: ", dictionary_of_accuracies, ", using: ", metric, " as the metric.")
            print("**************************\n")

    print("************************** Calling K MEANS ********************************")
    if global_variables.call_KMeans:
        predicted_labels, org_labels = kmeans.kmeans(
            sets[0], sets[2], global_variables.metrics[1]
        )

        confused_matrix = confusion_matrix.confusion_matrix(
            org_labels, predicted_labels
        )
        confusion_matrix.pretty_print(confused_matrix)

    if global_variables.call_SoftKMeans:
        predicted_labels, org_labels = softkmeans.soft_kmeans(
            sets[0], sets[2], global_variables.metrics[0]
        )
        # print(labels)
        confused_matrix = confusion_matrix.confusion_matrix(
            org_labels, predicted_labels
        )
        confusion_matrix.pretty_print(confused_matrix)


if __name__ == "__main__":
    main()
