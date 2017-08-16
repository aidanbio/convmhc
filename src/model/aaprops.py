import pandas as pd
from commons import Const, StatUtils
import aaindex
from sklearn import preprocessing

class AAPropScorer(object):
    def __init__(self, corr_cutoff=0.8, data_transformer=None, aa_subst_scorer=None):
        self._corr_cutoff = corr_cutoff
        self._data_transformer = data_transformer
        self._aa_subst_scorer = aa_subst_scorer
        self._scoretab = None

    def load_score_tab(self, **kwargs):
        self._load_score_tab(**kwargs)

        # Dealing with missing data
        self._scoretab = self._scoretab.dropna(axis=1)

        # Preprocessing with data transformer
        if self._data_transformer is not None:
            self._scoretab = pd.DataFrame(self._data_transformer.fit_transform(self._scoretab.values),
                                          index=self._scoretab.index,
                                          columns=self._scoretab.columns)

        # Remove highly correlated features
        if self._corr_cutoff is not None:
            hcorr = StatUtils.find_corr(self._scoretab, self._corr_cutoff)
            self._scoretab = self._scoretab.drop(hcorr, 1)

        # Extend scores by the substitution scorer
        if self._aa_subst_scorer is not None:
            subst_scoretab = self._aa_subst_scorer.score_table()
            #             Tracer()()
            colnames = ['%s_%s' % (c1, c2) for c1 in self._scoretab.columns for c2 in subst_scoretab.columns]
            scoretab = pd.DataFrame(index=self._scoretab.index, columns=colnames)
            for aa in scoretab.index:
                vals = [p * s for p in self._scoretab.loc[aa] for s in subst_scoretab.loc[aa]]
                scoretab.loc[aa, :] = vals
            self._scoretab = scoretab

    def scores(self, aa):
        return self._scoretab.loc[aa]

    def score_table(self):
        return self._scoretab

    def n_scores(self):
        return self._scoretab.shape[1]

    def feature_names(self):
        return self._scoretab.columns

    def _load_score_tab(self, **kwargs):
        raise NotImplementedError()


class AAIndex1AAPropScorer(AAPropScorer):
    def __init__(self, corr_cutoff=0.8, data_transformer=None):
        super(AAIndex1AAPropScorer, self).__init__(corr_cutoff, data_transformer)

    def _load_score_tab(self, **kwargs):
        aaindex_dir = kwargs['aaindex_dir'] if kwargs.has_key('aaindex_dir') else 'datasets/aaindex/'
        aaindex.init(aaindex_dir, '1')
        scoretab = pd.DataFrame(index=list(Const.AMINO_ACIDS), )
        for key, rec in aaindex._aaindex.iteritems():
            # rec = aaindex.get(key)
            vals = [rec.get(aa) for aa in Const.AMINO_ACIDS]
            scoretab[key] = vals
        # print('%s==>%s' % (key, rec.desc))
        self._scoretab = scoretab

# % run
# 'commons.ipynb'
# % run
# 'aasubst.ipynb'


class WenLiuAAPropScorer(AAPropScorer):
    def __init__(self, corr_cutoff=0.85, data_transformer=None, aa_subst_scorer=None):
        super(WenLiuAAPropScorer, self).__init__(corr_cutoff, data_transformer, aa_subst_scorer)

    def _load_score_tab(self, **kwargs):
        fn = kwargs['file'] if kwargs.has_key('file') else 'datasets/aaprops_Wen_Liu_ext.csv'
        scoretab = pd.read_csv(fn, header=0)
        scoretab = scoretab.loc[list(Const.AMINO_ACIDS), :]
        self._scoretab = scoretab
