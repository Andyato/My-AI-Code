import numpy as np

def nms_cpu(detects, thres):
    '''@params
            detects: np.ndarray [[xmin,ymin,xmax,ymax, score]]
            thres: IoU threshold(float)
       @return
            keep: keeped detections' index 
    '''
    xmins  = detects[:, 0]
    ymins  = detects[:, 1]
    xmaxs  = detects[:, 2]
    ymaxs  = detects[:, 3]
    scores = detects[:, 4]

    areas = (xmaxs - xmins + 1.) * (ymaxs - ymins + 1.)
    order = scores.argsort()[::-1] 

    keep = []
    while order.size > 0:
        keep.append(order[0])
        ixmins = np.maximum(xmins[order[0]], xmins[order[1:]])
        iymins = np.maximum(ymins[order[0]], ymins[order[1:]])
        ixmaxs = np.minimum(xmaxs[order[0]], xmaxs[order[1:]])
        iymaxs = np.minimum(ymaxs[order[0]], ymaxs[order[1:]])

        iws = np.maximum(ixmaxs - ixmins + 1., 0.)
        ihs = np.maximum(iymaxs - iymins + 1., 0.)

        inters = iws * ihs
        unions = areas[order[0]] + areas[order[1:]] - inters

        overlaps = inters / unions
        idxes = np.where(overlaps <= thres)[0]
        order = order[idxes + 1] 
    return keep

if __name__ == "__main__":
    detects = np.array([
        [0, 0, 10, 10, 0.9],
        [1, 1, 11, 11, 0.8],
        [2, 2, 12, 12, 0.5],
        [11, 11, 21, 21, 0.9],
        [12, 12, 22, 22, 0.7],
    ])

    keep = nms_cpu(detects, 0.5)
    print(detects[keep])
