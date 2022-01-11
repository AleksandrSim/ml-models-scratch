import cosine_similarity
import euclidean_distance
from euclidean_distance import euclidean
from cosine_similarity import cosim


# returns a list of labels for the query dataset based upon labeled observations in the train dataset.
# metric is a string specifying either "euclidean" or "cosim".
# All hyper-parameters should be hard-coded in the algorithm.
def knn(train, query, metric):
    is_euclidean_activated = metric == "euclidean"
    # Hard-coded into the algorithm
    number_of_neighbors = 5
    labels = []
    final_predictions = []
    counter = 0
    for row_query in query:
        # Just a simple counter if statement
        counter += 1
        if counter % 26 != 0:
            print(counter, sep=" ", end=" ", flush=True)
        else:
            print("\n")
            print(counter, sep=" ", end=" ", flush=True)

        labels.append(row_query[0])
        class_distances = []

        for row in train:
            if is_euclidean_activated:
                distance_temp = euclidean_distance.euclidean(row[1], row_query[1])
            else:
                distance_temp = cosine_similarity.cosim(row[1], row_query[1])

            class_distances.append([row[0], distance_temp])

        if is_euclidean_activated:
            # Next step is to sort the values of each dictionary key ( Sort the values which are distances)
            class_distances.sort(key=lambda x: x[1])
        else:
            # Here we have to check if we are using euclidean distance or the cosine similarity for our metric
            class_distances.sort(key=lambda x: x[1], reverse=True)

        generated_neighbors = []
        # print(class_distances)
        for i in range(1, number_of_neighbors):
            generated_neighbors.append(class_distances[i][0])

        label_counts = {}
        for label in generated_neighbors:
            if label not in label_counts:
                label_counts[label] = 1
            else:
                label_counts[label] += 1

        # Now we have to make classification predictions based on the generated neighbors
        prediction = max(label_counts, key=label_counts.get)
        final_predictions.append(prediction)

    number_of_examples = len(final_predictions)
    correct = 0

    for i in range(number_of_examples):
        if final_predictions[i] == labels[i]:
            correct += 1

    accuracy = (correct / number_of_examples) * 100

    return final_predictions, labels, accuracy


def knn_best_number_of_neighbours_test(train, query, metric):
    is_euclidean_activated = metric == "euclidean"
    # Hard-coded into the algorithm
    number_of_neighbors = [2, 3, 4, 5]
    accuracies = {}
    for number_of_neighbor in number_of_neighbors:
        labels = []
        final_predictions = []
        counter = 0
        for row_query in query:
            counter += 1
            print(counter)

            labels.append(row_query[0])
            class_distances = []

            for row in train:
                if is_euclidean_activated:
                    distance_temp = euclidean_distance.euclidean(row[1], row_query[1])
                else:
                    distance_temp = cosine_similarity.cosim(row[1], row_query[1])

                class_distances.append([row[0], distance_temp])

            if is_euclidean_activated:
                # Next step is to sort the values of each dictionary key ( Sort the values which are distances)
                class_distances.sort(key=lambda x: x[1])
            else:
                # Here we have to check if we are using euclidean distance or the cosine similarity for our metric
                class_distances.sort(key=lambda x: x[1], reverse=True)

            generated_neighbors = []
            # print(class_distances)
            for i in range(1, number_of_neighbor):
                generated_neighbors.append(class_distances[i][0])

            label_counts = {}
            for label in generated_neighbors:
                if label not in label_counts:
                    label_counts[label] = 1
                else:
                    label_counts[label] += 1

            # Now we have to make classification predictions based on the generated neighbors
            prediction = max(label_counts, key=label_counts.get)
            final_predictions.append(prediction)

        number_of_examples = len(final_predictions)
        correct = 0

        for i in range(number_of_examples):
            if final_predictions[i] == labels[i]:
                correct += 1

        accuracies[number_of_neighbor] = (correct / number_of_examples)
        best_number_of_neighbors = max(accuracies, key=accuracies.get)

    return best_number_of_neighbors
