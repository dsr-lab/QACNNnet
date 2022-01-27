import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from metrics_extractor import extract_metrics
from preprocessing.preprocess import preprocess_text
import config

F1_ERROR_THRESHOLD = 0.5 #Define the minimum F1 score to be considered partially solved
QUANTILE = 0.99
ARTICLES = ["the","a","an"]
QUESTION_TOKENS = ["why","who","what","where","when","how"]

def extract_parameters(args):

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

    filtered_words = [word for word in words if word not in ARTICLES]
    return filtered_words

def classify_questions(scores):

    solved = []
    partially_solved = []
    unsolved = []

    for question, val  in scores.items():
        exact = val["EM"]
        f1 = val["F1"]
        if exact==1.0:
            solved.append(remove_articles(preprocess_text(question, config.PREPROCESSING_OPTIONS)))
        elif f1>F1_ERROR_THRESHOLD:
            partially_solved.append(remove_articles(preprocess_text(question, config.PREPROCESSING_OPTIONS)))
        else:
            unsolved.append(remove_articles(preprocess_text(question, config.PREPROCESSING_OPTIONS)))

    return solved, partially_solved, unsolved

def get_tokens_distribution(texts):

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

    total = sum(list(distribution.values()))
    for token in distribution:
        distribution[token]=distribution[token]/total

def get_statistics(distribution):

    statistics = {}

    values = list(distribution.values())
    statistics["Mean"] = np.mean(values)
    statistics["Variance"] = np.var(values)
    statistics["Quantile"] = np.quantile(values, QUANTILE)

    return statistics

def get_most_frequent_tokens(distribution, statistics):

    quantile = statistics["Quantile"]
    frequent_tokens = [token for token,value in distribution.items() if value>quantile]

    return frequent_tokens

def get_question_tokens_distribution(distribution):

    question_tokens_distribution = {token:value for token, value in distribution.items() if token in QUESTION_TOKENS}
    normalize_distribution(question_tokens_distribution)

    return question_tokens_distribution

def show_distribution(distribution):

    color_map = plt.cm.get_cmap("hsv", len(distribution)+1)

    bars = []
    for i, token in enumerate(QUESTION_TOKENS):
        value = distribution[token]
        bars.append(plt.bar(i, value, color=color_map(i), alpha=0.5)[0])

    plt.gca().set(title='Frequency histogram of question tokens', ylabel='Frequency')
    plt.legend(bars, QUESTION_TOKENS)
    plt.show()

def run_full_analysis(category, name, show_frequent=True):

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
    scores = extract_metrics(true_ans_path, pred_ans_path)
    solved, partially_solved, unsolved = classify_questions(scores)

    #Error analysis
    run_full_analysis(unsolved, "Unsolved")
    run_full_analysis(partially_solved, "Partially solved")
    run_full_analysis(solved, "Solved")
