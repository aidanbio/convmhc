import pandas as pd
from sklearn import preprocessing
from commons import Const

class AASubstScorer(object):
    def scores(self, aa):
        raise NotImplementedError()

    def score_table(self):
        raise NotImplementedError()

    def n_scores(self):
        return self.score_table().shape[1]

# % run 'commons.ipynb'

class BLOSUMAASubstScorer(AASubstScorer):
    def __init__(self, log_odds_mat='datasets/blosum/blosum62.blast.new',
                 target_probs_mat='datasets/blosum/target_prob_blosum62.txt',
                 scaler=preprocessing.MinMaxScaler()):
        # Load log odds score tab
        tab = pd.read_table(log_odds_mat, header=6, index_col=0, sep=' +')
        tab = tab.loc[list(Const.AMINO_ACIDS), list(Const.AMINO_ACIDS)]

        if scaler is not None:
            tab = pd.DataFrame(scaler.fit_transform(tab), index=tab.index, columns=tab.columns)
        self._lodds_score_tab = tab.transpose()

        # Load target probs tab
        tab = pd.read_table(target_probs_mat, header=3, sep=' +')
        tab.index = tab.columns
        for i in range(20):
            for j in range(i, 20):
                tab.iloc[i, j] = tab.iloc[j, i]
        tab = tab.loc[list(Const.AMINO_ACIDS), list(Const.AMINO_ACIDS)]
        if scaler is not None:
            tab = pd.DataFrame(scaler.fit_transform(tab), index=tab.index, columns=tab.columns)

        self._target_probs_tab = tab.transpose()

    def log_odds_scores(self, aa):
        return self._lodds_score_tab.loc[aa, :]

    def target_probs_scores(self, aa):
        return self._target_probs_tab.loc[aa, :]

    def scores(self, aa):
        return self.log_odds_scores(aa)

    def score_table(self):
        return self._lodds_score_tab

    def aas(self):
        return self._lodds_score_tab.columns.values

####################################################
# aascorer = BLOSUMAASubstScorer()
# print aascorer.scores('Y')
# print aascorer.n_scores()
# cols = aascorer.aas()
# for aa in Const.AMINO_ACIDS:
#     scores = aascorer.scores(aa)
# #     print scores.index
#     assert all(cols == scores.index)
#     assert all(1-scores[scores.index == aa] < Const.EPS)
#     assert all(1-scores[scores.index != aa] > Const.EPS)

