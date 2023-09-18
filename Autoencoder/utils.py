import csv
import pdb
import torch
from torch import nn
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.lines as mlines
from sklearn.calibration import calibration_curve
# import matplotlib


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class WriteLogger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'a+')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def calculate_accuracy(outputs, targets):
    # pdb.set_trace()

    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    # n_correct_elems = correct.float().sum().data[0]
    n_correct_elems = correct.float().sum().data

    return n_correct_elems / batch_size, correct


def calculate_accuracy_binary(outputs, targets):
    batch_size = targets.size(0)
    outputs_sigma = nn.Sigmoid()(outputs)
    pred = torch.where(outputs_sigma>0.5,torch.tensor(1).cuda(),torch.tensor(0).cuda())
    pred = pred.t()
    correct = pred.eq(targets.view(1,-1))
    n_correct_elems = correct.float().sum().data

    return n_correct_elems / batch_size, pred


def calculate_accuracy_binary_multilabel(outputs, targets):
    acc = torch.zeros(outputs.shape[1])
    pred = torch.zeros(outputs.shape)

    for i in range(outputs.shape[1]):
        acc[i], pred[:, i] = calculate_accuracy_binary(outputs[:, i], targets[:, i])

    return acc, pred


def ComputeProbCurve(predprob,tarlabel):

    # this function is used to compute probability 
    # curve for binary classification
    # pdb.set_trace()
    ind = np.argsort(predprob)
    sortedtarlabel = tarlabel[ind]

    # Set the font dictionaries (for plot title and axis titles)
    font = {'fontname':'Arial', 'size':'20', 'color':'black', 'weight':'bold',
            'verticalalignment':'bottom','horizontalalignment':'top'} # Bottom vertical alignment for more space

    matplotlib.rc('xtick',labelsize=20)
    matplotlib.rc('ytick',labelsize=20)

    plt.figure()
    # plt.title('Probability plot')
    plt.xlim([0.0,len(predprob)])
    plt.ylim([0.0,1.0])
    # plt.ylabel('probability')
    # plt.xlabel('Sample Index')

    flag0 = 0
    flag1 = 0

    for i in range(0,len(ind)):
        if sortedtarlabel[i] == 0 and flag0 == 0:
            plt.plot(i, predprob[ind][i],'ob', label = 'NO') # 0
            flag0 = flag0+1 
        elif sortedtarlabel[i] == 0:
            plt.plot(i, predprob[ind][i],'ob') # 0
        elif sortedtarlabel[i] == 1 and flag1 == 0:
            plt.plot(i,predprob[ind][i],'or', label = 'YES') # 1
            flag1 = flag1+1
        elif sortedtarlabel[i] == 1:
            plt.plot(i,predprob[ind][i],'or') # 1

    ax = plt.gca()
    
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)

    plt.legend(loc='upper left')
    plt.show()


def MakeCalibrationPlot(target_label,prediction_prob_forpos,n_bins):
    # this function is used to draw the calibration plot
    # the target_label is the binary labels for the data points
    # the prediction_prob_forpos is the predicted probablity 
    # pdb.set_trace()
    # cc_y_2,cc_x_2= calibration_curve(target_label,prediction_prob_forpos,n_bins=2)
    # cc_y_3,cc_x_3= calibration_curve(target_label,prediction_prob_forpos,n_bins=3)
    # cc_y_4,cc_x_4= calibration_curve(target_label,prediction_prob_forpos,n_bins=4)
    cc_y_5,cc_x_5 = calibration_curve(target_label,prediction_prob_forpos,n_bins=5)
    # cc_y_10,cc_x_10= calibration_curve(target_label,prediction_prob_forpos,n_bins=10)
    cci_y_5,cci_x_5,cci_bincount_5 = calibration_curve_morefeature(target_label,prediction_prob_forpos, n_bins, mode='EI')

    ccm_y_5,ccm_x_5,ccm_bincount_5 = calibration_curve_morefeature(target_label,prediction_prob_forpos, n_bins, mode='EN')

    # Set the font dictionaries (for plot title and axis titles)
    font = {'fontname':'Arial', 'size':'24', 'color':'black', 'weight':'bold',
            'verticalalignment':'bottom'} # Bottom vertical alignment for more space

    fig,ax = plt.subplots()
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    # plt.plot(cc_x_2,cc_y_2, marker='o', linewidth=1,label='2 bins')
    # plt.plot(cc_x_3,cc_y_3, marker='o', linewidth=1,label='3 bins')
    # plt.plot(cc_x_4,cc_y_4, marker='o', linewidth=1,label='4 bins')
    # plt.plot(cc_x_5,cc_y_5, marker='o', linewidth=1,label='5 bins')
    # plt.plot(cc_x_10,cc_y_10, marker='o', linewidth=1,label='10 bins')

    plt.plot(cc_x_5,cc_y_5, marker='o', linewidth=4)
    line = mlines.Line2D([0,1],[0,1],color='black')
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)
    # fig.suptitle('Calibration plot for outcome prediction--resnet')
    # ax.set_xlabel('Predicted Probability')
    # ax.set_ylabel('True probability in each bin')
    # plt.legend()
    plt.show()

    fig,ax = plt.subplots()
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.plot(ccm_x_5,ccm_y_5, marker='s', linewidth=4)

    line = mlines.Line2D([0,1],[0,1],color='black')
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)
    # fig.suptitle('Calibration plot for outcome prediction--resnet')
    # ax.set_xlabel('Predicted Probability')
    # ax.set_ylabel('True probability in each bin')
    ax = plt.gca()
    
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)

    # plt.legend()
    plt.show()


def bin_total(prediction_prob_forpos, n_bins):
    pdb.set_trace()
    bins = np.linspace(0., 1. + 1e-8, n_bins + 1)

    # In sklearn.calibration.calibration_curve,
    # the last value in the array is always 0.
    binids = np.digitize(prediction_prob_forpos, bins) - 1
    return np.bincount(binids, minlength=len(bins))


def calibration_curve_morefeature(target_label,prediction_prob_forpos, n_bins, mode='EI'):
    
    # pdb.set_trace()
    if mode == 'EI': 
       # equal intervals for splitting the bins
       bins = np.linspace(0., 1. + 1e-8, n_bins + 1)

    elif mode == 'EN':
       # equal number of data points per bin
       sorted_prob = np.sort(prediction_prob_forpos)
       num_perbin = round(len(prediction_prob_forpos)/n_bins)
       bins = [sorted_prob[num_perbin*i] for i in range(0,n_bins)]
       bins.append(sorted_prob[-1]+1e-8)
       bins = np.asarray(bins)

    binids = np.digitize(prediction_prob_forpos, bins) - 1

    bin_sums = np.bincount(binids, weights=prediction_prob_forpos, minlength=len(bins))
    bin_true = np.bincount(binids, weights=target_label, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    prob_true = (bin_true[nonzero] / bin_total[nonzero])
    prob_pred = (bin_sums[nonzero] / bin_total[nonzero])

    return prob_true, prob_pred, np.bincount(binids, minlength=len(bins))