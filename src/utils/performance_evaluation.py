import numpy as np
import matplotlib.pyplot as plt

from .object_identification import find_best_match


def plot_rpc(D, plot_color):
    
    """
    Compute and plot the recall/precision curve
    D - square matrix, D(i, j) = distance between model image i, and query image j
    Note: assume that query and model images are in the same order, i.e. correct answer for i-th query image is the i-th model image
    """

    recall = []
    precision = []
    total_imgs = D.shape[1]

    num_images = D.shape[0]
    assert(D.shape[0] == D.shape[1])

    labels = np.diag([1]*num_images)

    d = D.reshape(D.size)
    l = labels.reshape(labels.size)

    sortidx = np.argsort(d, axis=0)
    d = d[sortidx]
    l = l[sortidx]
    tp = 0
    # for idx in range(len(d)):
    #     tp = tp + l[idx]
    # 
    # print(np.min(d), np,max(d))
    
    ## Compute precision and recall values and append them to "recall" and "precision" vectors
    ## Your code here
    tau = np.arange(0,1,0.001)
    
    for t in tau:
        tp, fp, tn, fn = 0, 0, 0, 0
        for idx in range(len(d)):
            if l[idx] == 1 and d[idx] <= t:
                tp += 1
            if l[idx] != 1 and d[idx] <= t:
                fp+=1
            if l[idx] != 1 and d[idx] >= t:
                tn+=1
            if l[idx] == 1 and d[idx] >= t:
                fn+=1
        if tp+fp != 0 and tp+fn != 0:
            precision.append(tp/(tp+fp))
            recall.append(tp/(tp+fn))
    
    
    plt.plot([1-precision[i] for i in range(len(precision))], recall, plot_color+'-')
    return precision, recall

def compare_dist_rpc(model_images, query_images, dist_types, hist_type, num_bins, plot_colors):
    precision_all = []
    recall_all = []
    assert len(plot_colors) == len(dist_types)
    for idx in range(len(dist_types)):
        [_, D] = find_best_match(model_images, query_images, dist_types[idx], hist_type, num_bins)
        precision, recall = plot_rpc(D, plot_colors[idx])
        precision_all.append(precision)
        recall_all.append(recall)
        plt.axis([0, 1, 0, 1]);
        plt.xlabel('1 - precision');
        plt.ylabel('recall');
        # legend(dist_types, 'Location', 'Best')
        plt.legend(dist_types, loc='best')
    return precision_all, recall_all
