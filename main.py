import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras import initializers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


def plot_correlation(Xs, y):
    fig, ax = plt.subplots(figsize=(15, 10))
    dataframe = Xs.join(y)
    sns.heatmap(dataframe.corr(), annot=True, annot_kws={"size": 20})
    plt.show()


def plot_rotation_feed_velocity(rotation, feed, velocity):
    plt.rcParams["figure.figsize"] = (15, 8)
    plt.rcParams['lines.linewidth'] = 1
    plt.rcParams['lines.color'] = 'r'
    plt.rcParams['axes.grid'] = True

    plt.subplot(311)
    plt.plot(rotation)
    plt.xlabel('time')
    plt.ylabel('rotation')
    plt.subplot(312)
    plt.plot(feed)
    plt.xlabel('time')
    plt.ylabel('feed')
    plt.subplot(313)
    plt.plot(velocity)
    plt.xlabel('time')
    plt.ylabel('velocity')
    plt.tight_layout()
    plt.show()


class Normalizer:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def min_max_scaling(self):
        min_max_scaler = MinMaxScaler()
        min_max_scaler.fit(self.dataframe)
        scaled_data_frame = min_max_scaler.transform(self.dataframe)

        return pd.DataFrame(scaled_data_frame, columns=self.dataframe.columns, index=list(self.dataframe.index.values))

    def standardization(self):
        standard_scaler = StandardScaler()
        standard_scaler.fit(self.dataframe)
        scaled_data_frame = standard_scaler.transform(self.dataframe)

        return pd.DataFrame(scaled_data_frame, columns=self.dataframe.columns, index=list(self.dataframe.index.values))

    def robust_scaling(self):
        robust_scaler = RobustScaler()
        robust_scaler.fit(self.dataframe)
        scaled_data_frame = robust_scaler.transform(self.dataframe)

        return pd.DataFrame(scaled_data_frame, columns=self.dataframe.columns, index=list(self.dataframe.index.values))


def make_dataset(data, label, window_size=20):
    feature_list = []
    label_list = []
    for i in range(len(data) - window_size):
        feature_list.append(np.array(data.iloc[i:i + window_size]))
        label_list.append(np.array(label.iloc[i + window_size]))

    return np.array(feature_list), np.array(label_list)


if __name__ == '__main__':
    # Read data
    df = pd.read_csv('./data/youngdeok_data.csv', usecols=['rotation', 'feed', 'velocity', 'grade'])
    df.columns = ['rotation', 'feed', 'velocity', 'label']
    X = df.loc[:, 'rotation':'velocity']
    y = df.loc[:, ['label']]

    # Plot data
    plot_correlation(X, y)

    plot_rotation_feed_velocity(X['rotation'], X['feed'], X['velocity'])

    # Preprocessing, split train, valid and test dataset
    X = Normalizer(X).min_max_scaling()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=False)

    X_train, y_train = make_dataset(X_train, y_train, 20)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=20, shuffle=False)

    X_test, y_test = make_dataset(X_test, y_test, 20)

    # Create model
    model = Sequential()
    model.add(LSTM(16,
                   input_shape=(X_train.shape[1], X_train.shape[2]),
                   kernel_initializer=initializers.GlorotNormal(seed=12),
                   bias_initializer='zeros',
                   return_sequences=False)
              )
    model.add(Dense(1, kernel_initializer=initializers.GlorotNormal(seed=12),
                    bias_initializer='zeros'))

    # Learning
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=5)
    filename = os.path.join('./model', 'tmp_checkpoint.h5')
    checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

    history = model.fit(X_train, y_train,
                        epochs=200,
                        batch_size=16,
                        validation_data=(X_valid, y_valid),
                        callbacks=[early_stop, checkpoint])

    model.load_weights(filename)
    pred = model.predict(X_test)

    # Predict and plot
    plt.figure(figsize=(15, 3))
    plt.plot(y_test, label='actual')
    plt.plot(pred, label='prediction')
    plt.legend()
    plt.show()
