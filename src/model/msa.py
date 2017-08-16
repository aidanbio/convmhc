import pandas as pd
import collections
import numpy as np
import re
from commons import Const, HLAAlleleUtils

from IPython.core.debugger import Tracer


def aa_freqs(seq, aa_unknown='*'):
    target = seq
    if not isinstance(seq, list):
        target = list(seq)
    n_unknown = np.float64(target.count(aa_unknown)) / 20
    return [(np.float64(target.count(aa)) + n_unknown) if target.count(aa) > 0 else n_unknown for aa in
            Const.AMINO_ACIDS]


def aa_props(seq, aa_unknown='*'):
    freqs = aa_freqs(seq, aa_unknown)
    return np.divide(freqs, np.float64(np.sum(freqs)))


class IMGTMultipleSequenceAlignment:
    def __init__(self, _msa_tab):
        self._msa_tab = _msa_tab
        self.shape = self._msa_tab.shape
        self._aases = [''.join(np.unique(filter(lambda x: x in list(Const.AMINO_ACIDS), self._msa_tab.iloc[:, pos])))
                       for pos in range(self.shape[1])]
        self._allele_seq_map = {}

    def rep_aa_at(self, pos):
        return self._msa_tab.iloc[0, pos]

    def aas_at(self, pos):
        return self._aases[pos]

    def aa_freqs_at(self, pos):
        rep_aa = self.rep_aa_at(pos)
        seq = self.aas_at(pos)
        seq = seq.replace('-', rep_aa)
        return aa_freqs(list(seq), aa_unknown='*')

    def aa_probs_at(self, pos):
        rep_aa = self.rep_aa_at(pos)
        seq = self.aas_at(pos)
        seq = seq.replace('-', rep_aa)
        return aa_props(list(seq), aa_unknown='*')

    def rep_seq(self):
        return self._msa_tab.iloc[0, :]

    def rep_allele_seq(self, allele):  # Nonsynonymous mutation allele, eg. A*02:01
        #         Tracer()()
        seq = None
        # Get from local cache
        if self._allele_seq_map.has_key(allele):
            seq = self._allele_seq_map[allele]
        else:
            mtab = self._msa_tab
            seq = mtab.iloc[map(lambda ix: HLAAlleleUtils.equal_names(ix, allele), mtab.index.values).index(True), :]
            seq = ''.join(filter(lambda ch: (ch is not None and ch != '.'), seq.values))
            self._allele_seq_map[allele] = seq
        return seq

    def has_allele(self, allele):
        mtab = self._msa_tab
        return any(map(lambda ix: HLAAlleleUtils.equal_names(ix, allele), mtab.index.values))

    def alleles(self):
        return self._msa_tab.index.values

    @classmethod
    def from_file(cls, filename, offset=None):

        f = open(filename)
        od = collections.OrderedDict()

        try:
            for line in f:
                tokens = line.split()
                if len(tokens) > 0 and tokens[0].find('*') >= 0:
                    allele = tokens[0]
                    seq = ''.join(tokens[1:])
                    #                     print '%s\t%s' % (allele, seq)

                    if len(seq) > 0:
                        if seq[len(seq) - 1] == 'X':
                            seq = seq[:len(seq) - 1]

                        if od.has_key(allele):
                            od[allele].extend(list(seq))
                        else:
                            od[allele] = list(seq)

            # Set full sequences for reference alleles
            rep_seq = None
            naa = len(Const.AMINO_ACIDS)
            for allele, seq in od.iteritems():
                #                 print '%s===>%s' % (allele, seq)
                if rep_seq is None:
                    rep_seq = seq
                else:
                    if (allele.count(':') == 1) or (
                        re.search(':0[12][NLSAQ]?$', allele) is not None):  # Nonsynonymous mutation allele
                        #                         print 'Nonsynonymous allele===>%s' % allele
                        new_seq = seq
                        for i in range(len(seq)):
                            aa = seq[i]
                            if aa == '-':
                                new_seq[i] = rep_seq[i]
                            elif aa == '*':
                                new_seq[i] = Const.AMINO_ACIDS[np.random.choice(naa, 1)[0]]

                                #                         new_seq = filter(lambda ch: (ch is not None and ch != '.'), new_seq)
                        od[allele] = new_seq
                        #                         print 'Converted==>%s' % od[allele]
            tab = pd.DataFrame.from_dict(od, orient='index')
            if offset > 0:
                tab = tab.iloc[:, offset:]
                tab.columns = range(tab.columns.shape[0])
            return cls(tab)

        finally:
            f.close()

    @classmethod
    def from_url(cls, url):
        pass

####################################################

import unittest
import numpy as np


class IMGTMultipleSequenceAlignmentTest(unittest.TestCase):
    def setUp(self):
        self.msa_hlaA = IMGTMultipleSequenceAlignment.from_file(Const.FN_IMGT_HLA_CLASSI_A_PROT)
        self.allele_exist = ['HLA-A*01:01', 'HLA-A*02:01', 'HLA-A*02:02', 'HLA-A*02:03',
                             'HLA-A*02:04', 'HLA-A*02:05', 'HLA-A*02:06', 'HLA-A*02:07',
                             'HLA-A*02:10', 'HLA-A*02:11', 'HLA-A*02:12', 'HLA-A*02:16',
                             'HLA-A*02:17', 'HLA-A*02:19', 'HLA-A*02:50', 'HLA-A*03:01',
                             'HLA-A*03:02', 'HLA-A*03:19', 'HLA-A*11:01', 'HLA-A*11:02',
                             'HLA-A*23:01', 'HLA-A*24:02', 'HLA-A*24:03', 'HLA-A*25:01',
                             'HLA-A*26:01', 'HLA-A*26:02', 'HLA-A*26:03', 'HLA-A*29:02',
                             'HLA-A*30:01', 'HLA-A*30:02', 'HLA-A*31:01', 'HLA-A*32:01',
                             'HLA-A*32:07', 'HLA-A*32:15', 'HLA-A*33:01', 'HLA-A*66:01',
                             'HLA-A*68:01', 'HLA-A*68:02', 'HLA-A*68:23', 'HLA-A*69:01',
                             'HLA-A*74:01', 'HLA-A*80:01']

        self.allele_not_exist = ['HLA-A1', 'HLA-A11', 'HLA-A2', 'HLA-A24', 'HLA-A26', 'HLA-A3', 'HLA-A3/11']

    def test_rep_allele_seq(self):

        for allele in self.allele_exist:
            print allele
            seq = self.msa_hlaA.rep_allele_seq(allele)
            print '%s, %s' % (seq, len(seq))
            for aa in seq:
                self.assertTrue(aa in Const.AMINO_ACIDS)

    def test_has_allele(self):
        for al in self.allele_exist:
            self.assertTrue(self.msa_hlaA.has_allele(al))
        for al in self.allele_not_exist:
            self.assertFalse(self.msa_hlaA.has_allele(al))

if __name__ == '__main__':
    unittest.main()
