import numpy as np

def get_IoU(pred_bbox, gt_bbox):
    '''@pred_bbox & gt_bbox : np.ndarray [xmin, ymin, xmax, ymax]
    '''
    assert pred_bbox[2] - pred_bbox[0] > 0 and pred_bbox[3] - pred_bbox[1] > 0
    assert gt_bbox[2] - gt_bbox[0] > 0 and gt_bbox[3] - gt_bbox[1] > 0

    # get coordinates of inters
    ixmin = max(pred_bbox[0], gt_bbox[0])
    iymin = max(pred_bbox[1], gt_bbox[1])
    ixmax = min(pred_bbox[2], gt_bbox[2])
    iymax = min(pred_bbox[3], gt_bbox[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)

    # intersection
    inter = iw * ih

    # union, union = S1 + S2 - inter
    S1 = (pred_bbox[2] - pred_bbox[0] + 1.) * (pred_bbox[3] - pred_bbox[1] + 1.)
    S2 = (gt_bbox[2] - gt_bbox[0] + 1.) * (gt_bbox[3] - gt_bbox[1] + 1.)
    union = S1 + S2 - inter

    # overlap
    overlap = inter / union
    return overlap


if __name__ == "__main__":
    pred_bbox = np.array([0, 0, 10, 10])
    gt_bbox = np.array([0, 0, 10, 10])
    print(get_IoU(pred_bbox, gt_bbox))
