import sklearn
import sklearn.metrics as metrics
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
        self.probabilities=[]
        # self.report_dir = 'classfication_report.txt'

    def read_predicted(self, file_dir):

        with open(file_dir, 'r') as f:
            f.seek(0)
            temp = f.readlines()
            self.predicted = list(map(int, temp))

    def read_truth(self, file_dir):

        with open(file_dir, 'r') as f:
            f.seek(0)
            temp = f.readlines()
            self.truth = list(map(int, temp))
    def read_probabilities(self,file_dir):
        with open(file_dir,'r') as f:
            f.seek(0)
            lines=f.readlines()
            for i in range(1,len(lines),2):
                temp=lines[i]+lines[i+1]
                num_list=temp.replace("\n",'').replace('[','').replace(']','').split(' ')
                num_list=list(map(float,num_list))
                self.probabilities.append(num_list)
            # print("shit")


    def plot_confusion_matrix(self,
                              y_pred,
                              y_truth,
                              classes=None,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        cm = metrics.confusion_matrix(y_true=y_truth, y_pred=y_pred)
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
        plt.savefig(title + '.png')

    def plot_roc_curve(self, y_true, y_score):
        fpr, tpr = sklearn.metrics.roc_curve(y_true=y_true, y_score=y_score)
        plt.plot(fpr, tpr)
        pass

    def model_summary(self,
                      y_pred,
                      y_truth,
                      classes_name: list,
                      report_file='classfication_report.txt'
                      ):
        if classes_name == None:
            text = sklearn.metrics.classification_report(y_true=y_truth,
                                                         y_pred=y_pred,
                                                         digits=4
                                                         )
        else:
            text = sklearn.metrics.classification_report(y_true=y_truth,
                                                         y_pred=y_pred,
                                                         target_names=classes_name,
                                                         digits=4
                                                         )
        print(text)
        with open(report_file, 'w+') as f:
            f.writelines(text)

    def dcase_result_output(self, input_file, output_file):
        results = []
        with open(input_file, 'r') as f:
            f.seek(0)
            results = f.readlines()
        results = list(map(int, results))
        with open(output_file, 'w+') as f:
            for i in range(len(results)):
                result_string = "audio/" + str(i + 1) + '.wav ' + str(cfg.class_name[results[i]]) + "\n"
                print(result_string)
                f.write(result_string)


if __name__ == '__main__':
    a = model_eval()


    # a.read_predicted('crnn_acr_stft_perdiction.txt')
    # a.read_truth('Ground_truth.txt')
    # # a.read_probabilities('crnn_angular_probabilities.txt')
    # a.model_summary(a.predicted,a.truth,cfg.class_name,'crnn_acr_stft_report.txt')
    # a.plot_confusion_matrix(a.predicted, a.truth, cfg.class_name,title='crnn acr_stft confusion matrix',normalize=True)

    # name_list=['acr_stft','bump','morse','mel','angular']
    name_list = ['acr_stft','bump', 'morse', 'mel', 'angular','gfcc']
    for var in name_list:
        a.read_truth('Ground_truth.txt')

        a.read_predicted('cnn_{}_perdiction.txt'.format(var))
        a.model_summary(a.predicted, a.truth, cfg.class_name, 'cnn_{}_report.txt'.format(var))
        a.plot_confusion_matrix(a.predicted, a.truth, cfg.class_name, title='cnn {} confusion matrix'.format(var), normalize=True)

        a.read_predicted('crnn_{}_perdiction.txt'.format(var))
        a.model_summary(a.predicted, a.truth, cfg.class_name, 'crnn_{}_report.txt'.format(var))
        a.plot_confusion_matrix(a.predicted, a.truth, cfg.class_name, title='crnn {} confusion matrix'.format(var), normalize=True)

    print('shit')
