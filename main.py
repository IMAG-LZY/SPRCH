import argparse
import torch.optim as optim
import scipy.io as io
import torchvision
from utils.evaluate import *
from utils.dataloader import *
from losses import *
import time
import os


def train(dataloader, net, optimizer, criterion, epoch, opt, emb):
    accum_loss = 0
    net.train()
    for _, (img, label) in enumerate(dataloader):
        features = net(img.cuda())
        features = torch.tanh(features)
        label = label.cuda()
        prototypes = emb(torch.eye(opt.data_class).cuda())
        prototypes = torch.tanh(prototypes)
        loss = criterion(features, prototypes, label, epoch, opt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        accum_loss += loss.item()
    return accum_loss / len(dataloader)


def str2bool(str):
    return True if str.lower() == 'true' else False


def main():
    parser = argparse.ArgumentParser(description='SPRCH')
    # parser.add_argument('--data_path', default='/home/lzy/dataset', help='path to dataset')
    parser.add_argument('--data_path', default='/opt/data/private/dataset', help='path to dataset')
    parser.add_argument('--data_name', type=str, default='coco', help='cifar or coco...')
    parser.add_argument('--data_class', type=int, default=10, help='the number of dataset classes')
    parser.add_argument('--outf', default='save', help='folder to output model checkpoints')
    parser.add_argument('--checkpoint', type=int, default=20, help='checkpointing after batches')
    parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
    parser.add_argument('--binary_bits', type=int, default=16, help='length of hashing binary')
    parser.add_argument('--temp', type=float, default=0.3, help='temperature')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--k', type=int, default=5000, help='mAP@k')
    parser.add_argument('--loss', type=str, default='p2p', help='different kinds of loss')
    parser.add_argument('--weighting',  type=str2bool, default='True', help='--balance two kinds of pairs')
    parser.add_argument('--self_paced',  type=str2bool, default='True', help='--self_paced learning schedule')
    opt = parser.parse_args()
    print(opt)
    print('data_name:', opt.data_name, 'hash_bit:', opt.binary_bits, 'batchSize:', opt.batchSize)
    print('loss:', opt.loss, 'self_paced:', opt.self_paced)
    outf = opt.outf + f'/{opt.loss}_{opt.weighting}_{opt.self_paced}' + '/SPRCH/' + opt.data_name + '/' + str(
        opt.binary_bits)
    os.makedirs(outf, exist_ok=True)
    feed_random_seed()

    # setup dataset
    train_list = 'data/' + opt.data_name + '/train.txt'
    database_list = 'data/' + opt.data_name + '/database.txt'
    test_list = 'data/' + opt.data_name + '/test.txt'
    train_loader, database_loader, test_loader = init_dataloader(opt.data_path, opt.data_name, train_list,
                                                                 database_list, test_list, opt.batchSize)

    # setup net
    net = torchvision.models.resnet18(pretrained=True)
    net.fc = nn.Linear(512, opt.binary_bits)
    net.cuda()

    class Embedding(torch.nn.Module):
        def __init__(self):
            super(Embedding, self).__init__()
            self.Embedding = nn.Linear(opt.data_class, opt.binary_bits)

        def forward(self, x):
            output = self.Embedding(x)
            return output
    emb = Embedding().cuda()

    # setup loss
    criterion = SupConLoss(loss=opt.loss, temperature=opt.temp,  data_class=opt.data_class).cuda()

    # setup optimizer
    hash_id = list(map(id, net.fc.parameters()))
    feature_params = filter(lambda p: id(p) not in hash_id, net.parameters())
    optimizer = optim.Adam([{'params': feature_params, 'lr': opt.lr},
                            {'params': emb.parameters(), 'lr': 100*opt.lr},
                            {'params': net.fc.parameters(), 'lr': 10*opt.lr}]
                           )

    # calculate pos_ratio and neg_ratio
    pos_ratio, neg_ratio = compute_ratio(train_loader)
    print(f'mean ratio for postive pairs:{pos_ratio.mean():.2f} ',
          f'mean ratio for negative pairs:{neg_ratio.mean():.2f}')

    # training process
    BestmAP = 0
    for epoch in range(1, opt.epochs+1):
        train_loss = train(train_loader, net, optimizer, criterion, epoch, opt, emb)
        # if epoch % opt.checkpoint != 0:
        # print(f'[{epoch + 1}] train_loss:{train_loss:.4f}')
        if epoch % opt.checkpoint == 0:
            start_time = time.time()
            rB, rL = compute_result(database_loader, net)
            qB, qL = compute_result(test_loader, net)
            retrieval_time = time.time()
            mAP = calc_topMap(qB, rB, qL, rL, opt.k)
            r_2 = calc_r_2(qB, rB, qL, rL)
            if BestmAP < mAP:
                BestmAP = mAP
                # save hash codes and model
                io.savemat(os.path.join(outf, 'save.mat'),
                           {'train_code': rB,
                            'L_tr': rL,
                            'test_code': qB,
                            'L_te': qL})
                torch.save(net.state_dict(),
                           os.path.join(outf, 'save.pth'))
            print(time.strftime("%Y/%m/%d %H:%M:%S"),
                  f'[{epoch}] train_loss:{train_loss:.4f} '
                  f'retrieval_time_1:{(retrieval_time - start_time) // 60:.0f}m{(retrieval_time - start_time) % 60:.0f}s '
                  f'retrieval_mAP: {mAP:.3f} '
                  f'retrieval_r_2: {r_2:.3f} '
                  f'BestmAP: {BestmAP:.3f}'
                  )


if __name__ == '__main__':
    print(time.strftime("%Y/%m/%d %H:%M:%S"))
    main()
    print('\n')

