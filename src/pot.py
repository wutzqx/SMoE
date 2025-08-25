import numpy as np

from src.spot import SPOT
from src.constants import *
from sklearn.metrics import *

def calc_point2point(predict, actual):
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    try:
        roc_auc = roc_auc_score(actual, predict)
    except:
        roc_auc = 0
    return f1, precision, recall, TP, TN, FP, FN, roc_auc


# the below function is taken from OmniAnomaly code base directly
def adjust_predicts(score, label,
                    threshold=None,
                    pred=None,
                    calc_latency=False):

    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)

    # avg=average_length_of_ones(label)
    latency = 0
    if pred is None:
        predict = score > threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
                anomaly_state = True
                anomaly_count += 1
                for j in range(i, 0, -1):
                    if not actual[j]:
                        break
                    else:
                        if not predict[j]:
                            predict[j] = True
                            latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict

def predicts(score, label,
                    threshold=None,
                    pred=None,
                    calc_latency=False):

    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    threshold = np.mean(score) + np.std(score)

    latency = 0
    if pred is None:
        predict = score > threshold
    else:
        predict = pred
    return predict
def adjust_window(score, label,
                    threshold=None,
                    pred=None,
                    calc_latency=False):

    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    avg=average_length_of_ones(label)
    print(avg)
    latency = 0
    if pred is None:
        predict = score > threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    wa = 0
    WA = False
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
            anomaly_state = True
            anomaly_count += 1
            for j in range(i, 0, -1):
                if not actual[j]:
                    break
                else:
                    if not predict[j]:
                        predict[j] = True
                        latency += 1
        elif not actual[i] and predict[i] and not anomaly_state:
            WA = True
            anomaly_state = True
        if anomaly_state:
            predict[i] = True
            if WA:
                wa += 1
            if (not actual[i] and not WA) or wa == avg:
                anomaly_state = False
                WA = False
                wa=0
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict

def average_length_of_ones(sequence):

    diff = np.diff(np.concatenate(([0], sequence, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]


    lengths = ends - starts


    if len(lengths) == 0:
        return 0


    return np.mean(lengths)

def calc_seq(score, label, threshold, calc_latency=False):

    if calc_latency:
        predict, latency = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        t = list(calc_point2point(predict, label))
        t.append(latency)
        return t
    else:
        predict = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        return calc_point2point(predict, label)


def bf_search(score, label, start, end=None, step_num=1, display_freq=1, verbose=True):

    if step_num is None or end is None:
        end = start
        step_num = 1
    search_step, search_range, search_lower_bound = step_num, end - start, start
    if verbose:
        print("search range: ", search_lower_bound, search_lower_bound + search_range)
    threshold = search_lower_bound
    m = (-1., -1., -1.)
    m_t = 0.0
    for i in range(search_step):
        threshold += search_range / float(search_step)
        target = calc_seq(score, label, threshold, calc_latency=True)
        if target[0] > m[0]:
            m_t = threshold
            m = target
        if verbose and i % display_freq == 0:
            print("cur thr: ", threshold, target, m, m_t)
    print(m, m_t)
    return m, m_t

def early_warning_f1_score(labels, pred, t=4):


    anomaly_segments = []
    in_anomaly = False
    start = 0

    for i in range(len(labels)):
        if labels[i] == 1 and not in_anomaly:
            in_anomaly = True
            start = i
        elif labels[i] == 0 and in_anomaly:
            in_anomaly = False
            anomaly_segments.append((start, i - 1))

    if in_anomaly:
        anomaly_segments.append((start, len(labels) - 1))

    adjusted_pred = np.zeros_like(pred)

    for seg_start, seg_end in anomaly_segments:

        end = seg_end
        seg_end = min(seg_end, seg_start + t)
        pred_in_window = any(pred[seg_start: seg_end + 1] == 1)
        pos = (pred[seg_start: seg_end + 1] != 0).argmax(axis=0)
        if pred_in_window:

            adjusted_pred[seg_start: end + 1] = 1
            print(f'{t}:{seg_start}')

    non_anomaly_mask = labels == 0
    adjusted_pred[non_anomaly_mask] = pred[non_anomaly_mask]
    return f1_score(labels, adjusted_pred)

def early_warning_precision(labels, pred):
    t = [0.1, 5, 10, 20, 30, 60]
    t = [int(i * 60) for i in t]
    precision = []
    for t_ in t:
        precision.extend([early_warning_f1_score(labels, pred, t=t_)])
    return precision

def pot_eval(init_score, score, label, q=1e-5, level=0.02):
    lms = lm[0]
    while True:
        try:
            s = SPOT(q)
            s.fit(init_score, score)
            s.initialize(level=lms, min_extrema=False, verbose=False)
        except: lms = lms * 0.999
        else: break

    ret = s.run(dynamic=False)
    pot_th = np.mean(ret['thresholds']) * lm[1]
    if max(label) == 1:
        roc_auc = roc_auc_score(label, score)
        roc_pr = average_precision_score(label, score)
    else:
        roc_auc = 0
        roc_pr = 0
    pred, p_latency = adjust_predicts(score, label, pot_th, calc_latency=True)
    pred2, p_latency2 = adjust_window(score, label, pot_th, calc_latency=True)
    pred3 = predicts(score, label, pot_th, calc_latency=True)
    p_t = calc_point2point(pred, label)
    p_t2 = calc_point2point(pred2, label)
    p_t3 = calc_point2point(pred3, label)
    p_t4 = early_warning_precision(label, pred3)

    return {
        'f1': p_t[0],
        'f12':p_t2[0],
        'f13':p_t3[0],
        'f14':p_t4,
        'precision': p_t[1],
        'recall': p_t[2],
        'TP': p_t[3],
        'TN': p_t[4],
        'FP': p_t[5],
        'FN': p_t[6],
        'ROC/AUC': roc_auc,
        'PRC/AUC': roc_pr,
        'threshold': pot_th,
    }, np.array(pred)
