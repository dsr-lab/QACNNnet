import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from metrics_extractor import extract_metrics
from preprocessing.preprocess import preprocess_text
import config

#This module executes a statistical analysis on the model by examining its predictions.

F1_ERROR_THRESHOLD = 0.5 #Define the minimum F1 score to consider a question partially solved
QUANTILE = 0.99
ARTICLES = ["the","a","an"] #Articles, to be removed as requested by the evaluation script
QUESTION_TOKENS = ["why","who","what","where","when","which","how"]

PREPROCESSING_OPTIONS = config.PREPROCESSING_OPTIONS.copy()
PREPROCESSING_OPTIONS["replace"]=True #Force punctuation removal, as requested by the evaluation script

def extract_parameters(args):

    '''
    Parse the arguments in input: only paths for
    test set and predictions are required.
    '''

    if len(args)==3:
        true_ans_path = args[1]
        pred_ans_path = args[2]
        if not os.path.exists(true_ans_path):
            print("Invalid argument: {} does not exists".format(true_ans_path))
            return None,None
        elif not os.path.exists(pred_ans_path):
            print("Invalid argument: {} does not exists".format(pred_ans_path))
            return None,None
        else:
            return true_ans_path, pred_ans_path

    elif len(args)<3:
        print("Missing one or more required argument: 'test set path' and 'predictions path' required")
        return None,None

    else:
        print("Too many arguments, two are expected: 'test set path' and 'predictions path'")
        return None,None

def remove_articles(words):

    '''
    Apply the removal of articles from a text
    '''

    filtered_words = [word for word in words if word not in ARTICLES]
    return filtered_words

def classify_questions(scores, compare_ans_lengths=True):

    '''
    Split questions into one of the following classes:
    -Solved (EM=1)
    -Partialy solved (F1>threshold)
    -Unsolved (F1<threshold and EM=0)
    '''

    solved = []
    partially_solved = []
    unsolved = []

    ans_lenghts_solved = []
    ans_lenghts_parsolved = []
    ans_lenghts_unsolved = []

    for question, val  in scores.items():
        exact = val["EM"]
        f1 = val["F1"]
        answer_length = val["Answer length"]
        if exact==1.0:
            solved.append(remove_articles(preprocess_text(question, PREPROCESSING_OPTIONS)))
            ans_lenghts_solved.append(answer_length)
        elif f1>F1_ERROR_THRESHOLD:
            partially_solved.append(remove_articles(preprocess_text(question, PREPROCESSING_OPTIONS)))
            ans_lenghts_parsolved.append(answer_length)
        else:
            unsolved.append(remove_articles(preprocess_text(question, PREPROCESSING_OPTIONS)))
            ans_lenghts_unsolved.append(answer_length)

    #Compare the distributions of answers length and plot results
    if compare_ans_lengths:
        compare_distributions([ans_lenghts_solved,ans_lenghts_parsolved, ans_lenghts_unsolved],
                            ["Solved","Partially solved", "Unsolved"])

    return solved, partially_solved, unsolved

def get_tokens_distribution(texts):

    '''
    Get an ordered distribution (in form of a dictionary) of the number of
    tokens inside of a group of tokenized texts.
    '''

    distribution = {}
    for text in texts:
        for token in text:
            if token in distribution:
                distribution[token]+=1
            else:
                distribution[token]=1

    ordered_distribution = {k: v for k, v in sorted(distribution.items(), key=lambda item: item[1], reverse=True)}

    return ordered_distribution

def normalize_distribution(distribution):

    '''
    Normalize a distribution into a [0,1] interval, so that all the values
    sum up to 1.
    '''

    total = sum(list(distribution.values()))
    for token in distribution:
        distribution[token]=distribution[token]/total

def get_statistics(distribution):

    '''
    Compute all the main statistics (mean, variance and quantile) on
    the given distribution.
    '''

    statistics = {}

    values = list(distribution.values())
    statistics["Mean"] = np.mean(values)
    statistics["Variance"] = np.var(values)
    statistics["Quantile"] = np.quantile(values, QUANTILE)

    return statistics

def get_most_frequent_tokens(distribution, statistics):

    '''
    Return the tokens with the highest occurrences, using a quantile-cut.
    '''

    quantile = statistics["Quantile"]
    frequent_tokens = [token for token,value in distribution.items() if value>quantile]

    return frequent_tokens

def get_question_tokens_distribution(distribution):

    '''
    Compute the distribution (in the form of a dictionary) of the
    questions-related tokens.
    '''

    question_tokens_distribution = {token:value for token, value in distribution.items() if token in QUESTION_TOKENS}
    normalize_distribution(question_tokens_distribution)

    return question_tokens_distribution

def show_distribution(distribution):

    '''
    Plot an histogram showing the given distribution with labels and different
    colours.
    '''

    #Define a different colour for each element
    color_map = plt.cm.get_cmap("hsv", len(distribution)+1)

    #Add bars to histogram
    bars = []
    for i, token in enumerate(QUESTION_TOKENS):
        value = distribution[token]
        bars.append(plt.bar(i, value, color=color_map(i), alpha=0.5)[0])

    plt.gca().set(title='Frequency histogram of question tokens', ylabel='Frequency')
    plt.legend(bars, QUESTION_TOKENS)
    plt.show()


def compare_distributions(distributions, names):

    '''
    Show into a single plot histograms coming from different
    distributions.
    '''

    assert len(distributions)==len(names)

    #Define a different colour for each distribution
    color_map = plt.cm.get_cmap("hsv", len(distributions)+1)
    #Common plot arguments
    kwargs = {"alpha":0.2, "bins":100}

    for i, distribution in enumerate(distributions):
        name = names[i]
        plt.hist(distribution, **kwargs, color=color_map(i), label=name)

    plt.gca().set(title='Answers length histogram', ylabel='Frequency')
    plt.xlim(0,60)
    plt.legend()
    plt.show()

def run_full_analysis(category, name, show_frequent=True):

    '''
    Get, print and plot all the statistical analysis on a set of data.
    '''

    distribution = get_tokens_distribution(category)
    statistics = get_statistics(distribution)
    frequent_tokens = get_most_frequent_tokens(distribution, statistics)

    print()
    print("Analysis of " + name + ":")
    print()

    print("Mean in the number of occurrences of tokens: {}".format(statistics["Mean"]))
    #print("Variance in the number of occurrences of tokens: {}".format(statistics["Variance"]))

    if show_frequent:
        print("Most frequent words are:")
        for frequent_token in frequent_tokens:
            print("{}: {}".format(frequent_token, distribution[frequent_token]))

    question_tokens_distribution = get_question_tokens_distribution(distribution)
    show_distribution(question_tokens_distribution)

#Main
args = sys.argv
true_ans_path, pred_ans_path = extract_parameters(args)
if true_ans_path is not None and pred_ans_path is not None:
    #Get scores from predictions
    scores = extract_metrics(true_ans_path, pred_ans_path)
    #Split into categories
    solved, partially_solved, unsolved = classify_questions(scores)

    #Error analysis on all the three classes
    run_full_analysis(unsolved, "Unsolved")
    run_full_analysis(partially_solved, "Partially solved")
    run_full_analysis(solved, "Solved")
