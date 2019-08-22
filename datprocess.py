# -*- coding: utf-8 -*-
# Origin resource from MovieLens: http://grouplens.org/datasets/movielens/1m
import sys
import pandas as pd


class Channel:
    """
    simple processing for *.dat to *.csv
    """

    def __init__(self, inpath, outpath):
        self.input_path = inpath + '/{}'
        self.output_path = outpath + '/{}'

    def _process_user_data(self, file='users.dat'):
        f = pd.read_csv(self.input_path.format(file), sep='::', engine='python',
                          names=['userID', 'Gender', 'Age', 'Occupation', 'Zip-code'])
        f.to_csv(self.output_path.format('users.csv'), index=False)

    def _process_rating_data(self, file='ratings.dat'):
        f = pd.read_csv(self.input_path.format(file), sep='::', engine='python',
                          names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
        f.to_csv(self.output_path.format('ratings.csv'), index=False)

    def _process_movies_date(self, file='movies.dat'):
        f = pd.read_csv(self.input_path.format(file), sep='::', engine='python',
                          names=['MovieID', 'Title', 'Genres'])
        f.to_csv(self.output_path.format('movies.csv'), index=False)

    def dat_process(self):
        print('Process user data...')
        self._process_user_data()
        print('Process movies data...')
        self._process_movies_date()
        print('Process rating data...')
        self._process_rating_data()
        print('End.')


if __name__ == '__main__':
    argv = sys.argv[1:]

    if len(argv) < 2:
        print("Error: not enough argument supplied:")
        print("datprocess.py <input path> <output path>")
        exit(0)
    else:
        input_path = argv[0]
        output_path = argv[1]
        Channel(input_path, output_path).dat_process()
