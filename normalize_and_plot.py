import pandas as pd

from main import plot_rotation_feed_velocity, Normalizer

if __name__ == '__main__':
    df = pd.read_csv('./data/youngdeok_data.csv', usecols=['rotation', 'feed', 'velocity', 'grade'])
    df.columns = ['rotation', 'feed', 'velocity', 'label']
    X = df.loc[:, 'rotation':'velocity']
    y = df.loc[:, ['label']]

    plot_rotation_feed_velocity(X['rotation'], X['feed'], X['velocity'])

    scaled_X = Normalizer(X).min_max_scaling()
    plot_rotation_feed_velocity(scaled_X['rotation'], scaled_X['feed'], scaled_X['velocity'])

    scaled_X = Normalizer(X).standardization()
    plot_rotation_feed_velocity(scaled_X['rotation'], scaled_X['feed'], scaled_X['velocity'])

    scaled_X = Normalizer(X).robust_scaling()
    plot_rotation_feed_velocity(scaled_X['rotation'], scaled_X['feed'], scaled_X['velocity'])