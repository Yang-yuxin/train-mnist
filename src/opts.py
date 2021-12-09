import argparse
import os


class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # basic experiment setting
        self.parser.add_argument('--trainer', default='softmax',
                                 help='softmax | fc | conv, specify trainer of the model')
        self.parser.add_argument('--metric', default='loss',
                                 help='main metric to save best model')
        # train
        self.parser.add_argument('--lr', type=float, default=1.5e-4,
                                 help='learning rate')
        self.parser.add_argument('--num_epochs', type=int, default=100,
                                 help='total training epochs')
        self.parser.add_argument('--batch_size', type=int, default=32,
                                 help='batch size')
        self.parser.add_argument('--val_intervals', type=int, default=1,
                                 help='number of epochs to run validation')

        # save
        self.parser.add_argument('--save_dir', default='..\data\model_saved',
                                 help='models saving directory')


    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)
        return opt

    def init(self, args=''):
        opt = self.parse(args)
        return opt

