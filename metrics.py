import sklearn
import numpy as np
import scipy as sci
import config as cfg
import matplotlib.pyplot as plt
import itertools


class model_eval():
    def __init__(self):
        self.truth = []
        self.predicted = []
        self.y_score = []
        self.report_dir = 'classfication_report.txt'

    def read_predicted(self, file_dir):

        with open(file_dir, 'r') as f:
            f.seek(0)
            self.predicted = f.readlines()

    def read_truth(self, file_dir):

        with open(file_dir, 'r') as f:
            f.seek(0)
            self.truth = f.readlines()

    def plot_confusion_matrix(self,
                              y_pred,
                              y_truth,
                              classes=None,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blue):
        cm = sklearn.metrics.confusion_matrix(y_true=y_truth, y_pred=y_pred)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    def plot_roc_curve(self, y_true, y_score):
        fpr, tpr = sklearn.metrics.roc_curve(y_true=y_true, y_score=y_score)
        plt.plot(fpr, tpr)
        pass

    def model_summary(self,
                      y_pred,
                      y_truth,
                      classes_name: list,
                      ):
        text = sklearn.metrics.classification_report(y_true=y_truth,
                                                     y_pred=y_pred,
                                                     target_names=classes_name)
        with open(self.report_dir, 'w+') as f:
            f.writelines(text)

        pass


if __name__ == '__main__':
    eval_solution = model_eval()
