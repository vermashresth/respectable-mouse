import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from viz_utils import num_to_text

def normalize_data(orig_pose_dictionary):
  for key in orig_pose_dictionary:
    X = orig_pose_dictionary[key]['keypoints']
    X = X.transpose((0,1,3,2)) #last axis is x, y coordinates
    X[..., 0] = X[..., 0]/1024
    X[..., 1] = X[..., 1]/570
    orig_pose_dictionary[key]['keypoints'] = X
  return orig_pose_dictionary

def split_validation(orig_pose_dictionary, vocabulary, seed=2021,
                       test_size=0.5, split_videos=False):
  if split_videos:
    pose_dictionary = {}
    for key in orig_pose_dictionary:
      key_pt1 = key + '_part1'
      key_pt2 = key + '_part2'
      anno_len = len(orig_pose_dictionary[key]['annotations'])
      split_idx = anno_len//2
      pose_dictionary[key_pt1] = {
          'annotations': orig_pose_dictionary[key]['annotations'][:split_idx],
          'keypoints': orig_pose_dictionary[key]['keypoints'][:split_idx]}
      pose_dictionary[key_pt2] = {
          'annotations': orig_pose_dictionary[key]['annotations'][split_idx:],
          'keypoints': orig_pose_dictionary[key]['keypoints'][split_idx:]}
  else:
    pose_dictionary = orig_pose_dictionary

  def get_percentage(sequence_key):
    anno_seq = num_to_text(pose_dictionary[sequence_key]['annotations'])
    counts = {k: np.mean(np.array(anno_seq) == k) for k in vocabulary}
    return counts

  anno_percentages = {k: get_percentage(k) for k in pose_dictionary}

  anno_perc_df = pd.DataFrame(anno_percentages).T

  rng_state = np.random.RandomState(seed)
  try:
    idx_train, idx_val = train_test_split(anno_perc_df.index,
                                      stratify=anno_perc_df['attack'] > 0,
                                      test_size=test_size,
                                      random_state=rng_state)
  except:
    idx_train, idx_val = train_test_split(anno_perc_df.index,
                                      test_size=test_size,
                                      random_state=rng_state)

  train_data = {k : pose_dictionary[k] for k in idx_train}
  val_data = {k : pose_dictionary[k] for k in idx_val}
  return train_data, val_data, anno_perc_df
