import csv
import json
import operator

import numpy as np
import pandas


# extract valid_accuracy during training across cross-validation fold

nr_fold = 5
log_path = '/media/bryan/Data_2/bryan/output/NUCLEI-ATTENTION/colon/v1.0.0.0_base1_aug1_xy_only/'

fold_stat = []
for fold_idx in range(0, nr_fold):
    stat_file = '%s/%02d/stats.json' % (log_path, fold_idx)
    with open(stat_file) as f:
        info = json.load(f)

    best_value = 0
    for epoch in info:
        epoch_stats = info[epoch]
        epoch_value = epoch_stats['valid-acc']
        if epoch_value > best_value:
            best_value = epoch_value
    fold_stat.append(best_value)
print(fold_stat)

