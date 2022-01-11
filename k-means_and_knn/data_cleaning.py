def read_data(file_name):
    data_set = []
    with open(file_name, "rt") as f:
        for line in f:
            line = line.replace("\n", "")
            tokens = line.split(",")
            label = int(tokens[0])
            attribs = []
            for i in range(784):
                attribs.append(float(tokens[i + 1]))
            data_set.append([label, attribs])
    return data_set


def show(file_name, mode):
    data_set = read_data(file_name)
    for obs in range(len(data_set)):
        for idx in range(784):
            if mode == "pixels":
                if data_set[obs][1][idx] == 0:
                    print(" ", end="")
                else:
                    print("*", end="")
            else:
                print("%4s " % data_set[obs][1][idx], end="")
            if (idx % 28) == 27:
                print(" ")
        print("LABEL: %s" % data_set[obs][0], end="")
        print(" ")


def show_on_existing_data(data_set, mode):

    for obs in range(len(data_set)):
        for idx in range(784):
            if mode == "pixels":
                if data_set[obs][1][idx] == 0:
                    print(" ", end="")
                else:
                    print("*", end="")
            else:
                print("%4s " % data_set[obs][1][idx], end="")
            if (idx % 28) == 27:
                print(" ")
        print("LABEL: %s" % data_set[obs][0], end="")
        print(" ")
