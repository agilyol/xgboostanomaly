import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, Normalizer


def dataset(path, normalize=False, normalize_method='std'):
    x = []
    y = []
    with open(path, 'rt') as csvfile:
        csv_reader = csv.reader(csvfile)
        for idx, row in enumerate(csv_reader):
            if idx == 0:
                continue
            x.append([float(v) for v in row[1:37]])
            y.append([int(v) for v in row[37:]].index(1))
    x = np.array(x)
    y = np.array(y)
    if normalize:
        if normalize_method == 'std':
            normalizer = StandardScaler()
        if normalize_method == 'l2':
            normalizer = Normalizer()
        x = normalizer.fit_transform(x)
    return x, y

