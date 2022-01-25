# QACNNnet
This repository contains a project whose goal is to solve a very interesting and challenging NLP task: Question Answering.


## Description
QACNNet is a neural network mainly composed by CNNs along with attentions.

The relation that comes along with the project contains all the details that are related to the network that has been created, from the architecture specifications to the results obtained after a lot of the testing.

For convenience, it is reported here the architecture overview, and how to execute the evaluation script for the model.

### Architecture Overview
The following picture depicts an overview of the implemented model architecture.


### Evaluation Script
For evaluating the model, it is necessary to execute the two following scripts.

* **compute_answers.py**: this script is responsible of creating the predictions.json file, which is necessary for the following script. As argument it requires the path of the dataset that must be tested. Usage example:

        python compute_answers.py data/dev_set.json
* **evaluate.py**: after executing the previous script it is possible to proceed with the actual evaluation of the model. The script requires two arguments: the path of the dataset that must be tested (the same used in the previous script), and the predictions.json file path. This file is automatically created in the *predictions* project folder. Usage example:

        python evaluate.py data/dev_set.json predictions/predictions.json
     
 


