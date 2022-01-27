# QACNNnet
This repository contains a project whose goal is to solve a very interesting and challenging NLP task: Question Answering.


## Description
QACNNet is the neural network that has been created for solving the question answering task, and it is mainly composed by CNNs along with attentions.

The relation that comes along with the project contains all the details that are related to the network, from the architecture specifications to the various results obtained during our testing.

For convenience, it is reported here the architecture overview, and how to execute the evaluation script for the model.

### Architecture Overview
The following picture depicts an overview of the implemented model architecture.

![Alt text](docs/QACNNet_Architecture.jpg?raw=true "QACNNet")

### The data folder
It is **mandatory** to copy the content of the data folder in the project files, and it can be downloaded from the following link: https://drive.google.com/drive/folders/1HAOte5VlHcmSgkMv4-G6JlXyFRLHqGME?usp=sharing.

The final result that you should see inside the project data folder is the following one:

![Alt text](docs/data_folder.png?raw=true "data folder")

### Evaluation Script
For evaluating the model, it is necessary to execute the two following scripts.

* **compute_answers.py**: this script is responsible of creating the predictions.json file, which is necessary for the following script. As argument it requires the path of the dataset that must be tested. Usage example:

        python compute_answers.py data/dev_set.json
    It is important that model weights are correcly loaded before it proceeds to compute the answers. If everything is ok (e.g., the **data** folder has been correctly imported in the project), then you should read the following logs:
    
        ....
        Loading model's weights...
        Model's weights successfully loaded!
        Model succesfully built!  
        ...
* **evaluate.py**: after executing the previous script it is possible to proceed with the actual evaluation of the model. The script requires two arguments: the path of the dataset that must be tested (the same used in the previous script), and the predictions.json file path. This file is automatically created in the *predictions* project folder. Usage example:

        python evaluate.py data/dev_set.json predictions/predictions.json
      

## Contributors
* Gaetano Signorelli
* Daniele Sirocchi
     
 


