# This code uses flask to create a web page that shows the operation of a neural network. It displays the entire neural 
# network, while allowing you to zoom in on certain neurons. It uses D3 to display the network graph.

# To run:
#  - Go to Desktop > Programming > Instrumented-Neural-Network
#  - Type "export FLASK_APP=Show-Neural-Network.py"
#  - Type "flask run"
#  - To view, load the following page in a browser: http://127.0.0.1:5000/
# 
# To commit changes:
#  - Edit with Visual Studio
#  - git add *
#  - git commit -m "message"
#  - git push

# TODO:
#  - Switch it so we write to a TempFiles directory, and then ovwerwrite the contents of the directory on the next run.
#  - Is it possible to make this more efficient: read JSON --> create output_str --> create SVG

from flask import Flask, render_template, abort
import json


def load_working_data(file_name):
    """
    Returns the JSON data from the specified file.
    """
    try:
        with open(file_name) as file_object:
            return json.load(file_object)
    except FileNotFoundError:
        abort(404)


def generate_output_str(working_data):
    """
    Takes the JSON data and returns HTML code that represents the data.
    """
    meta_data = working_data["metadata"]
    nodes_data = working_data["nodes"]
    connections_data = working_data["connections"]
    output_str = "var newGraph = { 'metadata': [], 'nodes': [], 'connections':[] }; \n"
    for item in working_data["metadata"][0].keys():
        new_str = f"var {item} = '{meta_data[0][item]}'; \n"
        output_str += new_str
    for node in nodes_data:
        new_str = f"var tempNode = {node}; newGraph.nodes.push(tempNode); \n"
        output_str += new_str
    for link in connections_data:
        new_str = f"var tempLink = {link}; newGraph.connections.push(tempLink); \n"
        output_str += new_str
    return output_str


app = Flask(__name__)
directory_name = "Neural-Network-Parameters/"
file_name_base = "working-data-"
file_name_number = "0"
working_data = load_working_data(directory_name + file_name_base + "0")
meta_data = working_data["metadata"]
num_iterations = meta_data[0]["num_iterations"]              # Number of iterations (epochs)
iteration_number = meta_data[0]["iteration_number"]          # Current iteration (epoch))


@app.route("/")
def index():
    output_str = generate_output_str(working_data)
    return render_template("index.html", network_graph=output_str)

@app.route('/iteration/<int:iteration_number>')
def iteration(iteration_number):
    global file_name_number
    global num_iterations
    global working_data
    if iteration_number < 0:
        file_name_number = 0
    elif iteration_number >= num_iterations:
        file_name_number = num_iterations - 1
    else:
        file_name_number = iteration_number
    file_name = directory_name + file_name_base + str(file_name_number)
    working_data = load_working_data(file_name)
    output_str = generate_output_str(working_data)
    return render_template("index.html", network_graph=output_str)

@app.route('/first')
def first_iteration():
    return iteration(0)

@app.route('/previous')
def previous_iteration():
    return iteration(int(file_name_number) - 1)

@app.route('/next')
def next_iteration():
    return iteration(int(file_name_number) + 1)

@app.route('/last')
def last_iteration():
    return iteration(num_iterations - 1)
