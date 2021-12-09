import sys
from models.softmax_model import *
from models.fc_model import *
from models.conv_model import *
from models.utils import *
import torch.utils.data
from opts import opts
from torchvision import *
import torch.optim as optim
from tqdm import tqdm
import math
# import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

def test_LR(net, X, Y):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    pi = torch.zeros_like(Y)
    for i in range(len(X)):
        pi[i] = net(X)
        if pi[i] < 0.5 and abs(Y[i]) < 1e-3:
            tn += 1
        elif pi[i] >= 0.5 and abs(Y[i]) < 1e-3:
            fp += 1
        elif pi[i] < 0.5 and Y[i] > 1e-3:
            fn += 1
        elif pi[i] >= 0.5 and Y[i] > 1e-3:
            tp += 1
        else:
            print(pi, Y[i])
            print('eeeerror')
    result = {}
    result['recall'] = tp / (tp + fn)
    result['precision'] = tp / (tp + fp)
    result['fallout'] = fp / (tp + fp)
    result['sensitivity'] = result['recall']
    result['specificity'] = tn / (tn + fp)
    result['f1'] = 2 * result['precision'] * result['recall'] \
                   / (result['precision'] + result['recall'])
    result['acc'] = (tp + tn) / (tp + tn + fp + fn)
    result['ber'] = 0.5 * (fp / (fp + tn) + fn / (fn + tp))
    result['mcc'] = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (fp + tn) * (tn + fn) * (fn + tp))
    result['auPRC'] = average_precision_score(Y, pi)
    result['auROC'] = roc_auc_score(Y, pi)

    return result

def main(opt):
    opt = opts().init()
    print(opt)

    print('Loading data...')
    trans_mnist = transform = transforms.Compose(
        [
            transforms.ToTensor()
        ]
    )

    print('Creating model...')
    train_set = datasets.MNIST(root='../data/MNIST',train=True,transform=trans_mnist,download=True)
    val_set = datasets.MNIST(root='../data/MNIST',train=False,transform=trans_mnist,download=True)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=1,
        shuffle=True,
        drop_last=True
    )

    if opt.trainer == 'softmax':
        fc_layer = [784, 10]
        network = softmax_model(fc_layer)
    elif opt.trainer == 'fc':
        fc_layer = [784, 520, 320, 10, ]
        network = fc_model(fc_layer)
    elif opt.trainer == 'conv':
        network = conv_model()
    else:
        print('Unspecified trainer!')
        sys.exit()

    loss = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters(), lr=opt.lr)

    print('Start training...')
    best = 1e-10
    marker = 'best'
    for epoch in range(opt.num_epochs):
        train_l = 0
        for X, y in tqdm(train_loader):
            _y = network(X)
            l = loss(_y, y)
            l.backward()
            train_l = train_l + l
            optimizer.step()
        train_l = train_l / len(train_loader)
        # Calculate accracy
        if epoch % opt.val_intervals == 0 and opt.metric=='loss':
            optimizer.zero_grad()
            ac = 0
            test_l = 0
            for X_test, y_test in tqdm(val_loader):
                _y = network(X_test)
                ac += sum(_y.argmax(dim=1) == y_test)
                test_l = test_l + loss(_y, y_test)
            test_l = test_l / len(val_loader)
            ac = ac / len(val_loader)
            print('epoch %d, acc %.3f, train loss %.5f, test loss %.5f' % (epoch, ac, train_l, test_l))
            if (ac > best):
                best = ac
                save_model(os.path.join(opt.save_dir, 'model_{}_{}_{}.pth'.format(marker, opt.trainer, epoch)),
                       epoch, network, optimizer)
                print('save model {} ...'.format(epoch))

            # for k, v in log_dict_val.items():
            #     logger.scalar_summary('val_{}'.format(k), v, epoch)
            #     logger.write('{} {:8f} | '.format(k, v))
            # if log_dict_val[opt.metric] < best:
            #     best = log_dict_val[opt.metric]
            #     save_model(os.path.join(opt.save_dir, 'model_best.pth'),
            #                epoch, model)
            #     print('save model 2 ...')
        # else:
        #     save_model(os.path.join(opt.save_dir, 'model_last.pth'),
        #                epoch, model, optimizer)
        #     print('save model 3 ...')
    #     logger.write('\n')
    #     if epoch in opt.lr_step:
    #         save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
    #                    epoch, model, optimizer)
    #         print('save model 4 ...')
    #         lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
    #         print('Drop LR to', lr)
    #         for param_group in optimizer.param_groups:
    #             param_group['lr'] = lr
    # logger.close()



if __name__ == '__main__':
    opt = opts().parse()
    main(opt)