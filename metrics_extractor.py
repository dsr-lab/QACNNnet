import json
from evaluate import normalize_answer, compute_exact, compute_f1


# This module computes the metrics (separately) related to model predictions using the evaluation script

def read_data(true_ans_path, pred_ans_path):
    '''
    Open and read json data coming from the dataset
    containing the ground truth and model's predictions.
    '''

    with open(true_ans_path) as f:
        dataset_json = json.load(f)
        dataset = dataset_json['data']
    with open(pred_ans_path) as f:
        preds = json.load(f)

    return dataset, preds


def get_scores(dataset, preds):
    '''
    Compute and return scores (EM and F1) and answers lengths
    for each single predictions. Functions coming from the
    official evaluation script are used to compute scores.
    '''

    exact_scores = {}
    f1_scores = {}
    ans_lenghts = {}

    # Iterate on dataset
    for article in dataset:
        for p in article['paragraphs']:
            for qa in p['qas']:
                qid = qa['id']
                q_text = qa['question']
                gold_answers = [a['text'] for a in qa['answers']
                                if normalize_answer(a['text'])]

                if not gold_answers:
                    # For unanswerable questions, the only correct answer is empty string
                    gold_answers = ['']

                if qid not in preds:
                    print('Missing prediction for %s' % qid)
                    continue

                a_pred = preds[qid]
                # Take max over all gold answers
                exact_scores[q_text] = max(compute_exact(a, a_pred) for a in gold_answers)
                f1_scores[q_text] = max(compute_f1(a, a_pred) for a in gold_answers)
                # Take answer of minimum length
                ans_lenghts[q_text] = min(len(a) for a in gold_answers)

    return exact_scores, f1_scores, ans_lenghts


def merge_scores(exact, f1, ans_lenghts):
    '''
    Merge the scores (EM and F1) and answers lengths
    into a single dictionary.
    '''

    assert len(exact) == len(f1)
    assert len(exact) == len(ans_lenghts)

    scores = {}
    for question in exact.keys():
        scores[question] = {"EM": exact[question], "F1": f1[question], "Answer length": ans_lenghts[question]}

    return scores


def extract_metrics(true_ans_path, pred_ans_path):
    '''
    Extract the metrics for each predicition and return
    a dictionary associating them to questions'texts.
    '''

    dataset, preds = read_data(true_ans_path, pred_ans_path)
    exact, f1, ans_lenghts = get_scores(dataset, preds)
    scores = merge_scores(exact, f1, ans_lenghts)

    return scores
