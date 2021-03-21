import numpy as np
import os
import tensorflow as tf
from copy import deepcopy

from trainer import Trainer
from preprocess import normalize_data, split_validation

def seed_everything(seed):
  np.random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  tf.random.set_seed(seed)

def run_task1(results_dir, dataset, vocabulary, test_data,
              augment=False, epochs=15, skip_test_prediction=False, seed=2021):
  HPARAMS = {}
  val_size = HPARAMS["val_size"] = 0.2
  normalize = HPARAMS["normalize"] = True
  HPARAMS["seed"] = seed
  seed_everything(seed)
  split_videos = HPARAMS["split_videos"] = False

  if normalize:
    dataset = normalize_data(deepcopy(dataset))
    if not skip_test_prediction:
      test_data = normalize_data(deepcopy(test_data))
    else:
      test_data = None

  train_data, val_data, anno_perc_df = split_validation(dataset,
                                                        seed=seed,
                                                        vocabulary=vocabulary,
                                                        test_size=val_size,
                                                        split_videos=split_videos)
  num_classes = len(anno_perc_df.keys())
  feature_dim = HPARAMS["feature_dim"] = (2,7,2)

  # Generator parameters
  past_frames = HPARAMS["past_frames"] = 50
  future_frames = HPARAMS["future_frames"] = 50
  frame_gap = HPARAMS["frame_gap"] = 1
  use_conv = HPARAMS["use_conv"] = True
  batch_size = HPARAMS["batch_size"] = 128

  # Model parameters
  dropout_rate = HPARAMS["dropout_rate"] = 0.5
  learning_rate = HPARAMS["learning_rate"] = 5e-4
  layer_channels = HPARAMS["layer_channels"] = (128, 64, 32)
  conv_size = HPARAMS["conv_size"] = 5
  augment = HPARAMS["augment"] = augment
  class_to_number = HPARAMS['class_to_number'] = vocabulary
  epochs = HPARAMS["epochs"] = epochs

  trainer = Trainer(train_data=train_data,
                    val_data=val_data,
                    test_data=test_data,
                    feature_dim=feature_dim,
                    batch_size=batch_size,
                    num_classes=num_classes,
                    augment=augment,
                    class_to_number=class_to_number,
                    past_frames=past_frames,
                    future_frames=future_frames,
                    frame_gap=frame_gap,
                    use_conv=use_conv)

  trainer.initialize_model(layer_channels=layer_channels,
                          dropout_rate=dropout_rate,
                          learning_rate=learning_rate,
                          conv_size=conv_size)

  trainer.train(epochs=epochs)
  augment_str = '_augmented' if augment else ''
  trainer.model.save(f'{results_dir}/task1{augment_str}.h5')
  np.save(f"{results_dir}/task1{augment_str}_hparams", HPARAMS)

  val_metrics = trainer.get_validation_metrics()
  val_metrics.to_csv(f"{results_dir}/task1_metrics_val.csv", index=False)

  if not skip_test_prediction:
    test_results = trainer.get_test_predictions()
    np.save(f"{results_dir}/test_results", test_results)
  else:
    test_results = {}

  del trainer # clear ram as the test dataset is large
  gc.collect()
  return test_results
