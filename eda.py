import numpy as np
import pandas as pd

from viz_utils import num_to_text

def get_percentage(sequence_key, train):
  anno_seq = num_to_text(train['sequences'][sequence_key]['annotations'])
  vocabulary = train['vocabulary']
  counts = {k: np.mean(np.array(anno_seq) == k) for k in vocabulary}
  return counts

def get_analysis(train):
    anno_percentages = {k: get_percentage(k, train) for k in train['sequences']}
    anno_perc_df = pd.DataFrame(anno_percentages).T

    all_annotations = []
    for sk in train['sequences']:
      anno = train['sequences'][sk]['annotations']
      all_annotations.extend(list(anno))
    all_annotations = num_to_text(all_annotations)
    classes, counts = np.unique(all_annotations, return_counts=True)
    class_distr_df = pd.DataFrame({"Behavior": classes,
                  "Percentage Frames": counts/len(all_annotations)})
    return anno_perc_df, class_distr_df
