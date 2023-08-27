import torch
import numpy as np
import torch.backends.cudnn as cudnn


def choose_gpu(i_gpu):
    """choose current CUDA device"""
    torch.cuda.device(i_gpu).__enter__()
    cudnn.benchmark = True


def feed_random_seed(seed=59495):
    """feed random seed"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def compute_result(dataloader, net):
    codes, labels = [], []
    net.eval()
    for i, (image, label) in enumerate(dataloader):
        with torch.no_grad():
            d, t = image.cuda(), label.cuda()
            d = net(d)
            labels.append(t)
            codes.append(d)
    return (torch.cat(codes)).sign().cpu().data.numpy(), (torch.cat(labels)).cpu().data.numpy()


def compute_ratio(dataloader):
    pos_ratio = []
    neg_ratio = []
    for i, (image, label) in enumerate(dataloader):
        label = label.cuda()
        pos_mask = (torch.mm(label.float(), label.float().T) > 0).float()
        neg_mask = 1 - pos_mask
        pos_mask2 = label
        neg_mask2 = 1 - label
        pos_ratio.append(pos_mask.sum(1)/pos_mask2.sum(1))
        neg_ratio.append(neg_mask.sum(1)/neg_mask2.sum(1))
    return torch.cat(pos_ratio), torch.cat(neg_ratio)


def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH


def calc_r_2(qB, rB, queryL, retrievalL):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # queryL: {0,1}^{mxl}
    # retrievalL: {0,1}^{nxl}
    num_query = queryL.shape[0]
    r_2 = 0.0

    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.int64)
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hammingDist(qB[iter, :], rB)
        true_in_r_2 = (hamm <= 2)
        if np.sum(true_in_r_2) != 0:
            r_2_ = np.sum(true_in_r_2 * gnd) / np.sum(true_in_r_2)
        else:
            r_2_ = 0.0
        r_2 = r_2 + r_2_
    r_2 = r_2 / num_query

    return r_2


def calc_topMap(qB, rB, queryL, retrievalL, topk):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # queryL: {0,1}^{mxl}
    # retrievalL: {0,1}^{nxl}
    num_query = queryL.shape[0]
    topkmap = 0
    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = calc_hammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = int(np.sum(tgnd))
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap

