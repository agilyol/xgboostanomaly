import numpy as np
import time
import math
import functools
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, accuracy_score, auc
from utils import *
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier


class KubAnomalyExtension():

    def __init__(self, paths, name):
        self.train_X = []
        self.train_Y = []
        for path in paths:
            X, Y = dataset(path, normalize=True, normalize_method='l2')
            self.train_X.append(X)
            self.train_Y.append(Y)
        self.train_X = np.concatenate(self.train_X)
        self.train_Y = np.concatenate(self.train_Y)
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.train_X, 
            self.train_Y, 
            test_size=0.2, random_state=1, stratify=self.train_Y)
        self.final = {}
        self.name = name

    @staticmethod
    def truncate_number(number, decimal=4):
        factor = 10.0 ** decimal
        return math.trunc(number * factor) / factor

    def time_wrapper(func):
        s_time = time.time()
        @functools.wraps(func)
        def wrap(self, *args, **kwargs):
            print("inside wrap")
            return func(self, *args, **kwargs)
        e_time = time.time()
        print('Took {} seconds'.format(e_time - s_time))
        return wrap

    def plot_chart(self):
        model_name = list(self.final.keys())
        auc_value = list(self.final.values())
        x_pos = np.arange(len(model_name))
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(x_pos, auc_value,align='center', alpha=0.7,color=['turquoise', 'r', 'orange','green','brown','blue','pink'], capsize=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_name)
        ax.yaxis.grid(False)
        plt.ylabel('AUC Score (%)')
        ax.margins(x=0.05)
        for index,data1 in enumerate(auc_value):
            plt.text(x=index , y =data1+0.01, s=f"{data1}" , fontdict=dict(fontsize=10), horizontalalignment='center')
        plt.tight_layout()
        plt.savefig(self.name, dpi=400)

    @time_wrapper
    def decisiton_tree_classifier(self):
        clf = tree.DecisionTreeClassifier()
        clf.fit(self.X_train, self.Y_train)
        score_testing = clf.score(self.X_test, self.Y_test)
        y_pred = clf.predict(self.X_test)
        fpr, tpr, _ = roc_curve(self.Y_test, y_pred)
        auc_score = auc(fpr, tpr)
        acc = accuracy_score(self.Y_test, y_pred)
        auc_score = 100 * self.truncate_number(auc_score)
        result = {'DTC': auc_score}
        self.final.update(result)
        print("DECISION TREE SCORE", score_testing)
        print("DECISION TREE FPR " + str(fpr))
        print("DECISION TREE TPR " + str(tpr))
        print("DECISION TREE ACC " + str(acc))
        print("DECISION TREE AUC " + str(auc_score))

    @time_wrapper
    def random_forest_classifier(self):
        forest = RandomForestClassifier(criterion='gini', n_estimators=5,random_state=1, n_jobs=2)
        forest.fit(self.X_train, self.Y_train)
        forest.fit(self.X_train, self.Y_train)
        score_testing = forest.score(self.X_test, self.Y_test)
        y_pred = forest.predict(self.X_test)
        fpr, tpr, _ = roc_curve(self.Y_test, y_pred)
        auc_score = auc(fpr, tpr)
        acc = accuracy_score(self.Y_test, y_pred)
        auc_score = 100 * self.truncate_number(auc_score)
        result = {'RFC': auc_score}
        self.final.update(result)
        print("RANDOM FOREST TREE", score_testing)
        print("RANDOM FOREST TREE FPR " + str(fpr))
        print("RANDOM FOREST TREE TPR " + str(tpr))
        print("RANDOM FOREST TREE ACC " + str(acc))
        print("RANDOM FOREST TREE AUC " + str(auc_score))

    @time_wrapper
    def xgboost(self):
        xg_boost = xgb.XGBClassifier(silent=False, 
                      scale_pos_weight=1,
                      booster='gbtree',
                      learning_rate=0.41,  
                      colsample_bytree = 0.4,
                      subsample = 0.8,
                      objective='binary:logistic', 
                      reg_alpha = 0.1,
                      max_depth=8, 
                      gamma=5,
                      reg_lambda = 2,
                      random_state=50)
        xg_boost.fit(self.X_train, self.Y_train)
        score_testing = xg_boost.score(self.X_test, self.Y_test)
        y_pred = xg_boost.predict(self.X_test)
        fpr, tpr, _ = roc_curve(self.Y_test, y_pred)
        auc_score = auc(fpr, tpr)
        auc_score = 100 * self.truncate_number(auc_score)
        acc = accuracy_score(self.Y_test, y_pred)
        result = {'XGBoost': auc_score}
        self.final.update(result)
        print("XGBOOST SCORE", score_testing)
        print("XGBOOST FPR " + str(fpr))
        print("XGBOOST TPR " + str(tpr))
        print("XGBOOST ACC " + str(acc))
        print("XGBOOST AUC " + str(auc_score))

    def run(self):
        self.xgboost()
        self.decisiton_tree_classifier()
        self.random_forest_classifier()
        self.plot_chart()


if __name__ == '__main__':
    simple_data_path = "<path to simple dataset>"
    complex_data_path = "<path to complex dataset>"
    data_sets = {'simple': simple_data_path, 'complex': complex_data_path}
    for name, data in data_sets.items():
        kube_anomaly_simple = KubAnomalyExtension(data, name)
        kube_anomaly_simple.run()

