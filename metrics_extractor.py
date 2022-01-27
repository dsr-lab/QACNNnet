import json
from evaluate import normalize_answer, compute_exact, compute_f1

def read_data(true_ans_path, pred_ans_path):

    with open(true_ans_path) as f:
        dataset_json = json.load(f)
        dataset = dataset_json['data']
    with open(pred_ans_path) as f:
        preds = json.load(f)

    return dataset, preds

def get_scores(dataset, preds):
  exact_scores = {}
  f1_scores = {}
  ans_lenghts = {}
  for article in dataset:
    for p in article['paragraphs']:
      for qa in p['qas']:
        qid = qa['id']
        q_text = qa['question']
        gold_answers = [a['text'] for a in qa['answers']
                        if normalize_answer(a['text'])]
        if not gold_answers:
          # For unanswerable questions, only correct answer is empty string
          gold_answers = ['']
        if qid not in preds:
          print('Missing prediction for %s' % qid)
          continue
        a_pred = preds[qid]
        # Take max over all gold answers
        exact_scores[q_text] = max(compute_exact(a, a_pred) for a in gold_answers)
        f1_scores[q_text] = max(compute_f1(a, a_pred) for a in gold_answers)
        ans_lenghts[q_text] = min(len(a) for a in gold_answers)

  return exact_scores, f1_scores, ans_lenghts

def merge_scores(exact, f1, ans_lenghts):

    assert len(exact)==len(f1)
    assert len(exact)==len(ans_lenghts)

    scores = {}
    for question in exact.keys():
        scores[question]={"EM": exact[question],"F1": f1[question], "Answer length":ans_lenghts[question]}

    return scores

def extract_metrics (true_ans_path, pred_ans_path):
    dataset, preds = read_data(true_ans_path, pred_ans_path)
    exact, f1, ans_lenghts = get_scores(dataset, preds)
    scores = merge_scores(exact, f1, ans_lenghts)

    return scores
