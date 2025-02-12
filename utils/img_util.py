from pathlib import Path

import itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')


def drawLossF(params, result_folder, is_trainer=False):
    loss = []
    val_loss = []

    history_dict = params['history']
    total_epoch = params['total_epoch']

    if not is_trainer:
        loss = history_dict['loss']
        val_loss = history_dict['val_loss']
    else:
        for i in history_dict:
            for j in i['loss']:
                loss.append(j)
            for j in i['val_loss']:
                val_loss.append(j)

    loss = loss[0:total_epoch]
    val_loss = val_loss[0:total_epoch]

    # "bo" is for "blue dot"
    # b is for "solid blue line"
    lines = [{'label': 'Training loss', 'datas': loss[0:total_epoch], 'color': 'bo'},
             {'label': 'Validation loss', 'datas': val_loss[0:total_epoch], 'color': 'b'}]

    fn = Path().joinpath(result_folder, 'LossF.png')
    drawEpochLineGraph(lines, 'Training and validation loss', 'Loss', fn)


def drawAccF(params, result_folder, is_trainer=False):
    acc = []
    val_acc = []

    total_epoch = params['total_epoch']
    history_dict = params['history']
    if not is_trainer:
        acc = history_dict['acc']
        val_acc = history_dict['val_acc']
    else:
        for i in history_dict:
            for j in i['acc']:
                acc.append(j)
            for j in i['val_acc']:
                val_acc.append(j)

    acc = acc[0:total_epoch]
    val_acc = val_acc[0:total_epoch]

    # "bo" is for "blue dot"
    # b is for "solid blue line"
    lines = [{'label': 'Training acc', 'datas': acc[0:total_epoch], 'color': 'bo'},
             {'label': 'Validation acc', 'datas': val_acc[0:total_epoch], 'color': 'b'}]
    fn = Path().joinpath(result_folder, 'AccF.png')
    drawEpochLineGraph(lines, 'Training and validation accuracy', 'Accuracy', fn)

def drawLossF_FL_server(params, result_folder, is_trainer=False):
    test_loss = []

    history_dict = params['history']
    total_epoch = params['total_epoch']

    if not is_trainer:
        test_loss = history_dict['test_loss']
    else:
        for i in history_dict:
            test_loss.append(i['test_loss'])

    test_loss = test_loss[0:total_epoch]

    # "bo" is for "blue dot"
    # b is for "solid blue line"
    lines = [{'label': 'test loss', 'datas': test_loss[0:total_epoch], 'color': 'b'}]

    fn = Path().joinpath(result_folder, 'TestLossF.png')
    drawEpochLineGraph(lines, 'Test loss', 'Loss', fn)

def drawAccF_FL_server(params, result_folder, is_trainer=False):
    test_acc = []

    total_epoch = params['total_epoch']
    history_dict = params['history']
    if not is_trainer:
        test_acc = history_dict['test_acc']
    else:
        for i in history_dict:
            test_acc.append(i['test_acc'])

    test_acc = test_acc[0:total_epoch]

    # "bo" is for "blue dot"
    # b is for "solid blue line"
    lines = [{'label': 'test acc', 'datas': test_acc[0:total_epoch], 'color': 'b'}]
    fn = Path().joinpath(result_folder, 'TestAccF.png')
    drawEpochLineGraph(lines, 'Test accuracy', 'Accuracy', fn)


def drawEpochLineGraph(line_list, title, ylabel, fn: Path):

    plt.figure()

    for line in line_list:
        epochs = range(1, len(line['datas']) + 1)
        plt.plot(epochs, line['datas'], line['color'], label=line['label'])
    plt.title(title, fontsize=20)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend()

    plt.savefig(fn, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, suptitle = '',result_folder=''):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)
    plt.figure()

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.suptitle(suptitle, y=-0.02, fontsize=30)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", 
                 fontsize=14)

    plt.ylabel('True label',fontsize=14)
    plt.xlabel('Predicted label',fontsize=14)
    plt.tight_layout()

    fn = Path().joinpath(result_folder, 'CM.png')
    
    plt.savefig(fn, bbox_inches='tight')
    plt.close()

