import pandas as pd

from main import interpolate_y_data, plot_rotation_feed_velocity, Normalizer

if __name__ == '__main__':
    df = pd.read_csv('./data/0506-0_v2.csv', usecols=[' CH 5 ', ' CH 6 ', ' CH 9 ', ' CH 10 '])
    df.columns = ['rotation', 'feed', 'velocity', 'label']
    X = df.loc[:, 'rotation':'velocity']
    y = df.loc[:, ['label']]
    y = pd.DataFrame(interpolate_y_data(y.values), y.index, y.columns)

    plot_rotation_feed_velocity(X['rotation'], X['feed'], X['velocity'])

    scaled_X = Normalizer(X).min_max_scaling()
    plot_rotation_feed_velocity(scaled_X['rotation'], scaled_X['feed'], scaled_X['velocity'])

    scaled_X = Normalizer(X).standardization()
    plot_rotation_feed_velocity(scaled_X['rotation'], scaled_X['feed'], scaled_X['velocity'])

    scaled_X = Normalizer(X).robust_scaling()
    plot_rotation_feed_velocity(scaled_X['rotation'], scaled_X['feed'], scaled_X['velocity'])