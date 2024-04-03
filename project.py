# Implements a simple two-layer neural network. Input layer 𝑎[0] will have 784 units corresponding to the 784 pixels in each 28x28 input image. 
# A hidden layer 𝑎[1] will have 200 units with ReLU activation, and finally our output layer 𝑎[2] will have 10 units corresponding to the ten digit
# classes with softmax activation.
# Video: https://www.youtube.com/watch?v=w8yWXqWQYmU
# Blog post: https://www.samsonzhang.com/2020/11/24/understanding-the-math-behind-neural-networks-by-building-one-from-scratch-no-tf-keras-just-numpy

# This code uses Gradient Descent. The basic idea of gradient descent is to figure out what direction each parameter can go in to decrease error
# by the greatest amount, then nudge each parameter in its corresponding direction over and over again until the parameters for minimum error and
# highest accuracy are found. In a neural network, gradient descent is carried out via a process called backward propagation. We take a prediction,
# calculate an error of how off it was from the actual value, then run this error backwards through the network to find out how much each weight
# and bias parameter contributed to this error. Once we have these error derivative terms, we can nudge our weights and biases accordingly to improve
# our model. Do it enough times, and we'll have a neural network that can recognize handwritten digits accurately.

# To run:
#  - Go to Desktop > Programming > Instrumented-Neural-Network
#  - Type "python Simple-Neural-Network.py"
# 
# To commit changes:
#  - Edit with Visual Studio
#  - git add *
#  - git commit -m "message"
#  - git push

# TODO:
#  - See if there's a way to add X, Z1, A1, Z2, and A2 to the serialized working data.
#  - Add the back propagation data to the serialized working data.
#  - Refactor the code so the number of layers is not hardcoded.
#  - See if it makes sense to serialize the matrices directly, and shift the processing of them to the Javascript code
#  - Add other activation functions.
#  - Consider variations of gradient descent that improve training efficiency: gradient descent with momentum, RMSProp, and Adam optimization. 
#  - Create a package for the refactored neural network code.

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker
import seaborn as sn
import os, shutil
import json


num_input_nodes = 784
num_hidden_layers = 1
num_hidden_nodes = 180
num_output_nodes = 10
# The accuracy does keep improving after 100 epochs, but the rate of improvement is quite slow.
num_iterations = 100
# Sigmoid provides a smoother accuracy line. Whereas, ReLU gives an accuracy line that goes back-and-forth fairly wildly.
# But ReLU provides the greater overall accuracy.
activation_fn = "ReLU"
# When the learning rate (alpha) is higher, the accuracy line becomes like a saw tooth. 
alpha_value = 0.15
loss_fn = "Subtract a one hot encoding of the label from the probabilities"

# Lists to hold the values for Training and Validation Accuracy graph and the Training and Validation Loss graph.
training_accuracy = []
validation_accuracy = []
training_loss = []
validation_loss = []


def clear_directory(directory):
    """
    Clear the contents of a directory.
    """
    try:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    except Exception as e:
        print('Error clearing directory {}: {}'.format(directory, e))


def load_data(file_path):
    """
    Load data from a CSV file.
    """
    try:
        data = pd.read_csv(file_path)
        return np.array(data)
    except FileNotFoundError:
        print("File not found:", file_path)
        return None
    except Exception as e:
        print("Error loading data:", e)
        return None


def write_json_data(data, file_path):
    """
    Write JSON data to a file.
    """
    try:
        with open(file_path, "w") as outfile:
            json.dump(data, outfile, indent=4)
        print("JSON data written to", file_path)
    except IOError as e:
        print("Error writing JSON data to file:", e)


def init_params():
    """
    Create the initial weights and biases for the neural network. Start with a normal probability distribution centered around zero 
    with a standard deviation that is related to the number of incoming links to a node.
    Note: it's best practice to initialize your weights/biases close to 0, otherwise your gradients get really small really quickly:
    https://stackoverflow.com/questions/47240308/differences-between-numpy-random-rand-vs-numpy-random-randn-in-python
    """
    W1 = np.random.normal(0.0, pow(num_input_nodes, -0.5), (num_hidden_nodes, num_input_nodes)) 
    b1 = np.random.normal(size=(num_hidden_nodes, 1)) * 0.05 
    W2 = np.random.normal(0.0, pow(num_hidden_nodes, -0.5), (num_output_nodes, num_hidden_nodes))
    b2 = np.random.normal(size=(num_output_nodes, 1)) * 0.05 
    return W1, b1, W2, b2


def ReLU(Z):
    """
    Implement the Rectified Linear Unit (ReLU) function, which can be used as an activation function for nodes. That is, a simple 
    linear function that returns:
        x if x > 0
        0 if x <= 0
    """
    return np.maximum(Z, 0)


def ReLU_deriv(Z):
    """
    Implement the derivative of the ReLU activation function (i. the ReLU function), which can be used during back propagation. 
    Note that the slope of the ReLU function when X is less than zero is 0, and the slope of the ReLU function when X is greater 
    than zero is 1. When booleans convert to numbers, true converts to 1 and false converts to 0. Z > 0 is true when any one 
    element of Z is greater than 0 (ie. the function returns 1). Z > 0 is false when no element of Z is greater than 0 (i.e. 
    the function returns 0)
    """
    return Z > 0


def Sigmoid(Z):
    """
    Implement the Sigmoid function, which can be used as an activation function for nodes. 
    """
    return 1 / (1 + np.exp(-Z))


def Sigmoid_deriv(Z):
    """
    Implement the derivative of the Sigmoid function, which can be used during back propagation.
    """
    return Sigmoid(Z) * (1 - Sigmoid(Z))


def softmax(Z):
    """
    Implement the softmax function. That is, it translates the values to probabilities, between 0 and 1, that all add up to 1.
    Softmax takes a column of data at a time, subtracts the max value for numerical stability, and then takes each element in 
    the column and outputting the exponential of that element divided by the sum of the exponentials of each of the elements in 
    the input column.
    """
    Z -= np.max(Z, axis=0)
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    return A


def forward_prop(W1, b1, W2, b2, X):
    """
    Perform forward propagation for the hidden and ouput layers.
        𝑍[1] = 𝑊[1]𝑋+𝑏[1]
        𝐴[1] = 𝑔ReLU(𝑍[1]))
        𝑍[2] = 𝑊[2]𝐴[1]+𝑏[2]
        𝐴[2] = 𝑔softmax(𝑍[2])
    """
    # Calculate the node values for layer 1 (the hiden layer). Remember W1 is a numpy array, so we can use .dot for matrix operations.
    Z1 = W1.dot(X) + b1
    # Apply the activation function. We are using the Rectified Linear Unit (ReLU) function.
    A1 = ReLU(Z1)
    # Calculate the node values for layer 2 (the output layer).
    Z2 = W2.dot(A1) + b2
    # Apply the softmax function. The softmax function turns the output values into probabilities.
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def one_hot(Y):
    """
    Implement "one hot" encoding for the labels in the training data. That is, create a matrix for all images, where each column 
    represents an image label. Put 1 in the position of the label, and 0's in all other positions.
    """
    # Create an m x 10 matrix.  Y.size is m.  Y.max() is 9 (i.e. the biggest value when working with the digits 0-9 is 9).
    # Initialize the matrix to have zeros in all positions.
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    # For each row identified by np.arange(Y.size), change the value in column Y to 1.
    one_hot_Y[np.arange(Y.size), Y] = 1
    # Transpose the matrix, so each column represents an image label. That is, return a 10 x m matrix.
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    """
    Perform back propagation through the neural network. 
    Here are the calcuations for the weights and biases for layer 2 (i.e. the output layer)
        𝑑𝑍[2]=𝐴[2]−𝑌                        To determine the error for the output layer during training (i.e. dZ2),  
                                            subtract a "one hot encoding" of the label from the probabilities.
        𝑑𝑊[2]=1/𝑚 . 𝑑𝑍[2]𝐴[1]𝑇             That is, the average of the error values.
        𝑑𝐵[2]=1/𝑚 . Σ𝑑𝑍[2]        
    Here are the calcuations for the weights and biases for layer 1 (i.e. the hidden layer)
        𝑑𝑍[1]=𝑊[2]𝑇 . 𝑑𝑍[2].∗𝑔[1]′(𝑧[1])    Taking error from layer 2 (i.e. dZ2), and applying weights to it in reverse 
                                            (i.e. transpose of W2). g' is the drivative of the activation function.
        𝑑𝑊[1]=1/𝑚 . 𝑑𝑍[1]𝐴[0]𝑇
        𝑑𝐵[1]=1/𝑚 . Σ𝑑𝑍[1]
    Note that one commenter wrote that... I believe dZ[2] should be 2(A[2]−Y) because the error/cost at the final output 
    layer should be (A[2]−Y)^2. 
    """
    m = Y.size
    one_hot_Y = one_hot(Y)
    # The closer the prediction probability is to 1, the closer the loss is to 0. By minimizing the cost function, we improve 
    # the accuracy of our model. We do so by substracting the derivative of the loss function with respect to each parameter 
    # from that parameter over many rounds of graident descent.
    dZ2 = 2 * (A2 - one_hot_Y)
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    # Calculate the loss value
    loss_array = (A2 - one_hot_Y) ** 2
    _, num_items = loss_array.shape
    loss_value = (1/num_items) * np.sum(loss_array)
    return dW1, db1, dW2, db2, loss_value


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    """
    Update our parameters as follows:
        𝑊[2]:=𝑊[2]−𝛼𝑑𝑊[2]
        𝑏[2]:=𝑏[2]−𝛼𝑑𝑏[2]
        𝑊[1]:=𝑊[1]−𝛼𝑑𝑊[1]
        𝑏[1]:=𝑏[1]−𝛼𝑑𝑏[1]
    Alpha is the learning rate. Alpha is a hyper parameter (i.e. it is not trained by the model).
    """
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2


def get_predictions(A2):
    """
    Takes A2, which is the output of the neural network, and returns the index of the maximum value in column 0.
    np.argmax returns the indices of the max element of the array in a particular axis.
    """
    return np.argmax(A2, 0)


def get_accuracy(predictions, Y):
    """
    Get the accuracy between the predictions (i.e. A2) and Y (i.e. the labels).
    """
    return np.sum(predictions == Y) / Y.size


def gradient_descent(X, Y, alpha, iterations):
    """
    This pulls everything together. It initializes the parameters, performs the forward propagation, the backward 
    propagation, and updates the parameters. It does this iteration times, and it prints out an update every 10 
    iterations.
    """
    try:        
        W1, b1, W2, b2 = init_params()
        for i in range(iterations):        
            # Create the data structures for storing the details of the neural network working data.
            # For each iteration, we will have one working data file. The name of the file will indicate the iteration.
            # The working_data dictionary will have three lists: meta_data, node_data, and connections_data.
            #
            # Metadata:
            #  - Number of hidden layers
            #  - Number of nodes in each layer
            #  - Number of iterations
            #  - Iteration
            #  - Direction (i.e. forward or backward)
            #  - Activation function (i.e. descriptive text)
            #  - Alpha
            #  - Prediction
            #  - Label (i.e. the actual value)
            #  - Loss function (i.e. descriptive text)
            #
            # Neurons:
            #  - Layer #
            #  - Node #
            #  - ID # (which is used for creating the links)
            #  - Bias (db if backward step)
            # 
            # Connections:
            #  - Source neuron node #
            #  - Target neuron node #
            #  - Weight (dW if backward step)
            #
            # Note: I don't see a practcal way to include the X (training data), Z1, A1, Z2, or A2 values. During each
            # iteration, we process 41,000 images. That means X is a 784x41000 matrix. In other words, during each iteration, 
            # we process 41,000 values through each node in the network. This also means 41,000 values of Z1, A1, Z2, and A2 
            # for each iteration. I'm not sure how to gracefully show this. For now, I will not include this information in 
            # the JSON. Maybe she can show this informatn for the "inference" phase, rather than the training phase.
            working_data = {}
            meta_data = []
            nodes_data = []
            connections_data = []

            # Creating the data structure that stores the meta data for the working data
            temp_metadata = {}
            temp_metadata["num_input_nodes"] = num_input_nodes
            temp_metadata["num_hidden_layers"] = num_hidden_layers
            temp_metadata["num_hidden_nodes"] = num_hidden_nodes
            temp_metadata["num_output_nodes"] = num_output_nodes
            temp_metadata["num_iterations"] = num_iterations
            temp_metadata["iteration_number"] = i
            temp_metadata["direction"] = "forward"
            temp_metadata["activation_fn"] = activation_fn
            temp_metadata["alpha_value"] = alpha_value
            temp_metadata["prediction"] = ""
            temp_metadata["actual_value"] = ""
            temp_metadata["loss_fn"] = loss_fn
            meta_data.append(temp_metadata)

            # Creating the data structure that stores the working data for the connections between nodes in the input layer and the hidden layer
            for temp_i in range(1, num_input_nodes):
                for temp_j in range(1, num_hidden_nodes):
                    temp_connection = {}
                    temp_connection["source"] = 10000 + temp_i       # To make the node IDs unique I am adding a number indicating the layer
                    temp_connection["target"] = 20000 + temp_j       # To make the node IDs unique I am adding a number indicating the layer
                    temp_connection["weight"] = W1[temp_j,temp_i]
                    connections_data.append(temp_connection)
            # Creating the data structure that stores the working data for the connections between nodes in the hidden layer and the output layer
            for temp_k in range(1, num_hidden_nodes):
                for temp_l in range(1, num_output_nodes):
                    temp_connection = {}
                    temp_connection["source"] = 20000 + temp_k       # To make the node IDs unique I am adding a number indicating the layer
                    temp_connection["target"] = 30000 + temp_l       # To make the node IDs unique I am adding a number indicating the layer
                    temp_connection["weight"] = W2[temp_l,temp_k]
                    connections_data.append(temp_connection)

            for temp_m in range(1, num_input_nodes):
                temp_node = {}
                temp_node["layer"] = 0
                temp_node["node"] = temp_m
                temp_node["id"] = 10000 + temp_m
                temp_node["bias"] = 0                               # There is no bias for the input nodes
                nodes_data.append(temp_node)
            for temp_n in range(1, num_hidden_nodes):
                temp_node = {}
                temp_node["layer"] = 1
                temp_node["node"] = temp_n
                temp_node["id"] = 20000 + temp_n
                temp_node["bias"] = b1[temp_n, 0]
                nodes_data.append(temp_node)
            for temp_o in range(1, num_output_nodes):
                temp_node = {}
                temp_node["layer"] = 2
                temp_node["node"] = temp_o
                temp_node["id"] = 30000 + temp_o
                temp_node["bias"] = b2[temp_o, 0]
                nodes_data.append(temp_node)

            working_data["metadata"] = meta_data
            working_data["nodes"] = nodes_data
            working_data["connections"] = connections_data

            # Serializing the JSON data
            json_object = json.dumps(working_data, indent=4)
    
            # Writing JSON data to file
            file_name =  directory_name + "/working-data-" + str(i)
            write_json_data(json_object, file_name)

            Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
            dW1, db1, dW2, db2, training_loss_value = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)

            # Store the Training Accuracy data for this epoch, so we can graph it later.
            # A2 are the predictions that come out the other end of forward propagation.
            # Y are the image labels.
            predictions = get_predictions(A2)
            training_accuracy.append(get_accuracy(predictions, Y))
            # Store the Validation Accuracy data for this epoch, so we can graph it later.
            dev_predictions, validation_loss_value = make_predictions(X_dev, W1, b1, W2, b2)
            validation_accuracy.append(get_accuracy(dev_predictions, Y_dev))

            training_loss.append(training_loss_value)
            validation_loss.append(validation_loss_value)

            # TODO: Add the back propagation data to the serialized working data.

            W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
            if i % 10 == 0:
                print("Iteration: ", i)
                print("Accuracy: ", get_accuracy(predictions, Y))
        return W1, b1, W2, b2
    except Exception as e:
        print("Error in gradient descent:", e)
        return None, None, None, None


def make_predictions(X, W1, b1, W2, b2):
    """
    Returns predictions for the given data set (X). It uses the weights (W1 and W2) and biases (b1 and b2) and
    calls forward_prop to get the outputs of the neural network. It then gets the predictions from these outputs.
    This function also returns the "loss"
    """
    # The first part of the function gets the predictions...
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    # The second part of this function calculates the loss...
    one_hot_Y = one_hot(Y_dev)
    loss_array = (A2 - one_hot_Y) ** 2
    _, num_items = loss_array.shape
    loss_value = (1/num_items) * np.sum(loss_array)
    return predictions, loss_value


def test_prediction(index, W1, b1, W2, b2):
    """
    Test the neural network's prediction for the image at the "index" parameter.
    """
    current_image = X_train[:, index, None]
    prediction, _ = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


def main():
    # We will place our files in the "Neural-Network-Parameters" directory. If the directory does not exist, create it.
    # In case there are files in there, clear its contents. In the directory, we will have one JSON file for each 
    # iteration (epoch). We will also store images for the test and validation error rates.
    directory_name = "Neural-Network-Parameters"
    if not os.path.isdir(directory_name):
        os.makedirs(directory_name)
    clear_directory(directory_name)    

    # We are using the MNIST digit recognizer dataset. MNIST ("Modified National Institute of Standards and Technology") is the de facto “hello world”
    # dataset of computer vision. Since its release in 1999, this classic dataset of handwritten images has served as the basis for benchmarking
    # classification algorithms. Use pandas to read the CSV file with the data.
    data = load_data('train.csv')
    # Get the dimensions of the array. There are m rows (i.e. images). Each image has n (i.e. 785 values; one for the label and 784 for the pixels)
    m, n = data.shape

    # Shuffle the data before splitting into dev and training sets.
    np.random.shuffle(data)

    # Create the dev data (i.e. validation data) from the first 1,000 images.
    # Remember to transpose the matrix, so each column (rather than row) is an image.
    data_dev = data[0:1000].T
    # Now, Y_dev (i.e. the image label) will just be the first row.
    Y_dev = data_dev[0]
    # And X_dev will be the image pixels.
    X_dev = data_dev[1:n]
    # The pixel valueas (0-255) are transformed into decimal values (0-1).
    X_dev = X_dev / 255.

    # Create the training data from the remaining images. There are something like 41,000 of them.
    # Again, remember to transpose the matrix so each column is an image.
    data_train = data[1000:m].T
    Y_train = data_train[0]
    X_train = data_train[1:n]
    # The pixel valueas (0-255) are transformed into decimal values (0-1).
    X_train = X_train / 255.

    # Run the neural network on the training set.
    W1, b1, W2, b2 = gradient_descent(X_train, Y_train, alpha_value, num_iterations)
    if W1 is None:
        exit()

    # This code generates the Confusion Matrix for the last Validation run. It shows the accuracy between the 
    # predictions (i.e. A2) and Y (i.e. the labels). It also generates the HTML table that shows all the predictions 
    # that were wrong in the last Validation run.
    html_string = "<table><tbody>"
    val_predictions, _ = make_predictions(X_dev, W1, b1, W2, b2)
    confusion_matrix = np.zeros((10, 10))

    # Transpose the Validation run image matrix, so the images are in the rows. 
    image_matrix = X_dev.T
    # Multiply the image matrix values by 255, to get them back to the orignial greyscale values.
    image_matrix = image_matrix * 255
    # A counter that keeps track of the current image as we go through the Validation run.
    count = 0
    # Go through the predicted values, one-by-one.
    for predicted_value in val_predictions:
        # Get the label for the corresponding image.
        actual_value = Y_dev[count]
        # Increment the appropriate cell in the Confusion Matrix.
        confusion_matrix[(actual_value, predicted_value)] += 1
        # If the prediction does not match the label, then generate a table entry for it.
        if actual_value != predicted_value:
            # Get the values for the pixels in the corresponding image.
            image_vector = image_matrix[count]
            # Reshape the pixel values into a 28x28 array.
            image_array = image_vector.reshape((28,28)).T
            # Start the HTML code to create an SVG image for the current image.
            svg_string = """<svg width="28" height="28" style="background-color:white">"""
            # Iterate through the pixels in the 28x28 array, adding a 1-pixel rectangle to the SVG image when the pixel has a value.
            for i in range(image_array.shape[0]):
                for j in range(image_array.shape[1]):
                    if image_array[i, j] > 0:
                        # Convert the greyscale value (0-255) into a hex value (#aabbcc).
                        hex_color = hex(int(255 - image_array[i, j]))[2:].zfill(2)
                        hex_color_string = "#" + hex_color * 3
                        svg_string += """<rect x="{0}" y="{1}" width="1" height="1" fill="{2}"/>""".format(i, j, hex_color_string)
            svg_string += """</svg>"""
            html_string += "<tr><td>" + svg_string + "</td><td>" + str(actual_value) + "</td><td>" + str(predicted_value) + "</td></tr>"
        count += 1
    # Put the closing tags in the HTML string.
    html_string += "</tbody></table>"
    # Generate the Confusion Matrix.
    sn.heatmap(data=confusion_matrix, annot=True, fmt="g", cmap="tab20", center=0)
    plt.xlabel("Prediction")
    plt.ylabel("Actual Value")
    # If the file for the Confusion Matrix exists, remove it, before creating it again with the new Confusion Matrix plot.
    if os.path.isdir("static/Confusion.png"):
        os.remove("static/Confusion.png")
    plt.savefig("static/Confusion.png")
    plt.close()
    # If the file for the incorrect predictions HTML exists, remove it, before creating it again with the newly generated HTML string.
    if os.path.exists("static/BadPredictions.html"):
        os.remove("static/BadPredictions.html")
    file = open("static/BadPredictions.html", "w")
    file.write(html_string)
    file.close()

    # The following code generates the graph showing the Training Accuracy and the Validation accuracy.
    iteration_array = np.arange(0, num_iterations)
    training_accuracy_array = np.array(training_accuracy)
    validation_accuracy_array = np.array(validation_accuracy)
    plt.plot(iteration_array, training_accuracy_array, color='r', label="Training Accuracy")
    plt.plot(iteration_array, validation_accuracy_array, color='g', label="Validation Accuracy")
    plt.title("Accuracy at each Epoch", fontweight='bold')
    plt.xlabel("Epoch Number")
    plt.ylabel("Accuracy")
    plt.legend(loc='best', frameon=False)
    formatter = matplotlib.ticker.PercentFormatter(xmax=1)
    plt.gca().yaxis.set_major_formatter(formatter)
    if os.path.isdir("static/Accuracy.png"):
        os.remove("static/Accuracy.png")
    plt.savefig("static/Accuracy.png")
    plt.close()

    # The following code generates the graph showing the Training Accuracy and the Validation accuracy.
    iteration_array = np.arange(0, num_iterations)
    training_loss_array = np.array(training_loss)
    validation_loss_array = np.array(validation_loss)
    plt.plot(iteration_array, training_loss_array, color='r', label="Training Loss")
    plt.plot(iteration_array, validation_loss_array, color='g', label="Validation Loss")
    plt.title("Loss at each Epoch", fontweight='bold')
    plt.xlabel("Epoch Number")
    plt.ylabel("Loss")
    plt.legend(loc='best', frameon=False)
    if os.path.isdir("static/Loss.png"):
        os.remove("static/Loss.png")
    plt.savefig("static/Loss.png")
    plt.close()

    # Test the neural network's prediction for the images at indexes 0, 1, 2, and 3.
    #test_prediction(0, W1, b1, W2, b2)
    #test_prediction(1, W1, b1, W2, b2)
    #test_prediction(2, W1, b1, W2, b2)
    #test_prediction(3, W1, b1, W2, b2)



if __name__ == "__main__":
    main()
