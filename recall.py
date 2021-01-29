import numpy as np

def recall3(labels_list, scores_list, k):
    num_imgs = np.sum(labels_list)
    image_scores = scores_list[labels_list.astype(bool)]

    # Get top 3 labels (index) from images that contain label k                                                                        
    if len(image_scores.shape)==1:
        top3_labels = image_scores.argsort()[-3:]
    else:
        top3_labels = image_scores.argsort()[:, -3:]
    num_intop3 = np.sum(top3_labels==k)

    # Recall@3 is num_intop3/num_imgs
    recall = num_intop3/num_imgs if num_imgs!=0 else 0

    return recall
