# coding: utf-8 -*-
import sys
import pickle
import pandas as pd


class Corpus:
    
    def __init__(self, inpath, outpath):
        self.input_path = inpath + '/ratings.csv'
        self.output_path = outpath + '/rbm_items.dict'

    def _get_pos_neg_item(self, user_id):
        """
        Define the pos and neg item for user.
        pos_item mean items that user have rating, and neg_item can be items
        that user never seen before.
        Simple down sample method to solve unbalance sample.
        """
        print('Process: {}'.format(user_id))
        pos_item_ids = set(self.frame[self.frame['UserID'] == user_id]['MovieID'])
        neg_item_ids = self.item_ids ^ pos_item_ids
        # neg_item_ids = [(item_id, len(self.frame[self.frame['MovieID'] == item_id]['UserID'])) for item_id in neg_item_ids]
        # neg_item_ids = sorted(neg_item_ids, key=lambda x: x[1], reverse=True)
        # neg_item_ids = list(neg_item_ids)[:len(pos_item_ids)]
        item_dict = {}
        for item in pos_item_ids: item_dict[item] = 1
        for item in neg_item_ids: item_dict[item] = 0
        return item_dict

    def save(self):
        f = open(self.output_path, 'wb')
        pickle.dump(self.items_dict, f)
        f.close()

    def csv_process(self):
        self.frame = pd.read_csv(self.input_path)
        self.user_ids = set(self.frame['UserID'].values)
        self.item_ids = set(self.frame['MovieID'].values)
        self.items_dict = {user_id: self._get_pos_neg_item(user_id) for user_id in list(self.user_ids)}
        self.save()


if __name__ == '__main__':
    argv = sys.argv[1:]

    if len(argv) < 2:
        print("Error: not enough argument supplied:")
        print("csvprocess.py <input path> <output path>")
        exit(0)
    else:
        input_path = argv[0]
        output_path = argv[1]
        Corpus(input_path, output_path).csv_process()