import numpy as np

import sklearn
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
import tensorflow_addons as tfa

from batch_gen import MABe_Generator

class Trainer:
  def __init__(self, *,
               train_data,
               val_data,
               test_data,
               feature_dim,
               batch_size,
               num_classes,
               augment=False,
               class_to_number=None,
               past_frames=0,
               future_frames=0,
               frame_gap=1,
               use_conv=False):
    flat_dim = np.prod(feature_dim)
    if use_conv:
      input_dim = ((past_frames + future_frames + 1), flat_dim,)
    else:
      input_dim = (flat_dim * (past_frames + future_frames + 1),)

    self.input_dim = input_dim
    self.use_conv=use_conv
    self.num_classes=num_classes

    c2n = {'other': 0,'investigation': 1,
                'attack' : 2, 'mount' : 3}
    self.class_to_number = class_to_number or c2n

    self.train_generator = MABe_Generator(train_data,
                                      batch_size=batch_size,
                                      dim=input_dim,
                                      num_classes=num_classes,
                                      past_frames=past_frames,
                                      future_frames=future_frames,
                                      class_to_number=self.class_to_number,
                                      use_conv=use_conv,
                                      frame_gap=frame_gap,
                                      augment=augment,
                                      shuffle=True,
                                      mode='fit')

    self.val_generator = MABe_Generator(val_data,
                                        batch_size=batch_size,
                                        dim=input_dim,
                                        num_classes=num_classes,
                                        past_frames=past_frames,
                                        future_frames=future_frames,
                                        use_conv=use_conv,
                                        class_to_number=self.class_to_number,
                                        frame_gap=frame_gap,
                                        augment=False,
                                        shuffle=False,
                                        mode='fit')

    if test_data is not None:
      self.test_generator = MABe_Generator(test_data,
                                        batch_size=8192,
                                        dim=input_dim,
                                        num_classes=num_classes,
                                        past_frames=past_frames,
                                        future_frames=future_frames,
                                        use_conv=use_conv,
                                        class_to_number=self.class_to_number,
                                        frame_gap=frame_gap,
                                        augment=False,
                                        shuffle=False,
                                        mode='predict')

  def delete_model(self):
    self.model = None

  def initialize_model(self, layer_channels=(512, 256), dropout_rate=0.,
                       learning_rate=1e-3, conv_size=5):

    def add_dense_bn_activate(model, out_dim, activation='relu', drop=0.):
      model.add(layers.Dense(out_dim))
      model.add(layers.BatchNormalization())
      model.add(layers.Activation('relu'))
      if drop > 0:
        model.add(layers.Dropout(rate=drop))
      return model

    def add_conv_bn_activate(model, out_dim, activation='relu', conv_size=3, drop=0.):
      model.add(layers.Conv1D(out_dim, conv_size))
      model.add(layers.BatchNormalization())
      model.add(layers.Activation('relu'))
      model.add(layers.MaxPooling1D(2, 2))
      if drop > 0:
        model.add(layers.Dropout(rate=drop))
      return model

    model = Sequential()
    model.add(layers.Input(self.input_dim))
    model.add(layers.BatchNormalization())
    for ch in layer_channels:
      if self.use_conv:
        model = add_conv_bn_activate(model, ch, conv_size=conv_size,
                                     drop=dropout_rate)
      else:
        model = add_dense_bn_activate(model, ch, drop=dropout_rate)
    model.add(layers.Flatten())
    model.add(layers.Dense(self.num_classes, activation='softmax'))

    metrics = [tfa.metrics.F1Score(num_classes=self.num_classes)]
    optimizer = keras.optimizers.Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=metrics)

    self.model = model

  def _set_model(self, model):
      """ Set an external, provide initialized and compiled keras model """
      self.model = model

  def train(self, epochs=20, class_weight=None):
    if self.model is None:
      print("Please Call trainer.initialize_model first")
      return
    self.model.fit(self.train_generator,
          validation_data=self.val_generator,
          epochs=epochs,
          class_weight=class_weight)

  def get_validation_labels(self, on_test_set=False):
    y_val = []
    for _, y in self.val_generator:
      y_val.extend(list(y))
    y_val = np.argmax(np.array(y_val), axis=-1)
    return y_val

  def get_validation_predictions(self):
    y_val_pred = self.model.predict(self.val_generator)
    y_val_pred = np.argmax(y_val_pred, axis=-1)
    return y_val_pred

  def get_validation_metrics(self):
    y_val = self.get_validation_labels()
    y_val_pred = self.get_validation_predictions()

    f1_scores = sklearn.metrics.f1_score(y_val, y_val_pred,average=None)
    rec_scores = sklearn.metrics.precision_score(y_val, y_val_pred,average=None)
    prec_scores = sklearn.metrics.recall_score(y_val, y_val_pred,average=None)
    classes = list(self.class_to_number.keys())
    metrics = pd.DataFrame({"Class": classes, "F1": f1_scores, "Precision": prec_scores, "Recall": rec_scores})
    return metrics

  def get_test_predictions(self):
    all_test_preds = {}
    for vkey in self.test_generator.video_keys:
      nframes = self.test_generator.seq_lengths[vkey]
      all_test_preds[vkey] = np.zeros(nframes, dtype=np.int32)

    for X, vkey_fi_list in tqdm.tqdm(self.test_generator):
      test_pred = self.model.predict(X)
      test_pred = np.argmax(test_pred, axis=-1)

      for p, (vkey, fi) in zip(test_pred, vkey_fi_list):
        all_test_preds[vkey][fi] = p
    return all_test_preds
