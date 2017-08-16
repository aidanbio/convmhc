# % run
# 'commons.ipynb'
# % run
# 'msa.ipynb'

import numpy as np
from commons import Const, StatUtils, HLAAlleleUtils, PrintUtils
from msa import IMGTMultipleSequenceAlignment
from sklearn.preprocessing import MinMaxScaler
from aaprops import WenLiuAAPropScorer


# @singleton
class PanMHCIBindingDomain(object):
    __imgt_contact_sites_8 = [(0, 58), (0, 61), (0, 62), (0, 65), (0, 162), (0, 166), (0, 170),
                              (1, 6), (1, 23), (1, 44), (1, 98),
                              (2, 98), (2, 113), (2, 152), (2, 155), (2, 156), (2, 159),
                              (4, 6), (4, 8), (4, 21), (4, 69), (4, 73), (4, 96), (4, 98), (4, 113), (4, 115),
                              (5, 148), (5, 150), (5, 152), (5, 155),
                              (6, 72), (6, 75), (6, 76),
                              (7, 76), (7, 79), (7, 80), (7, 83), (7, 94), (7, 115), (7, 122), (7, 123), (7, 144),
                              (7, 148)]

    __imgt_contact_sites_9 = [(0, 4), (0, 58), (0, 61), (0, 62), (0, 65), (0, 162), (0, 166), (0, 170),
                              (1, 6), (1, 8), (1, 21), (1, 23), (1, 33), (1, 44), (1, 62), (1, 65), (1, 66), (1, 69),
                              (2, 96), (2, 98), (2, 113), (2, 155), (2, 156), (2, 159),
                              (3, 64), (3, 65), (3, 155),
                              (4, 69), (4, 72), (4, 73), (4, 96), (4, 115), (4, 155), (4, 156),
                              (5, 65), (5, 68), (5, 69), (5, 72), (5, 73), (5, 96), (5, 113), (5, 151), (5, 155),
                              (6, 96), (6, 113), (6, 148), (6, 150), (6, 152), (6, 155),
                              (7, 71), (7, 72), (7, 75), (7, 79), (7, 147),
                              (8, 76), (8, 79), (8, 80), (8, 83), (8, 94), (8, 115), (8, 122), (8, 123), (8, 144),
                              (8, 148)]

    __epiccapo_contact_sites_9 = [(0, 4), (0, 6), (0, 8), (0, 44), (0, 57), (0, 58), (0, 61), (0, 62), (0, 65), (0, 66),
                                  (0, 162), (0, 163), (0, 166), (0, 170),
                                  (1, 6), (1, 8), (1, 21), (1, 23), (1, 33), (1, 44), (1, 62), (1, 65), (1, 66),
                                  (1, 69), (1, 98), (1, 158),
                                  (2, 8), (2, 65), (2, 66), (2, 69), (2, 96), (2, 98), (2, 151), (2, 154), (2, 155),
                                  (2, 158), (2, 159),
                                  (3, 61), (3, 64), (3, 65), (3, 154), (3, 157),
                                  (4, 64), (4, 68), (4, 69), (4, 71), (4, 72), (4, 73), (4, 96), (4, 113), (4, 115),
                                  (4, 146), (4, 149), (4, 150), (4, 151), (4, 154), (4, 155),
                                  (5, 64), (5, 65), (5, 68), (5, 69), (5, 72), (5, 73), (5, 96), (5, 98), (5, 113),
                                  (5, 146), (5, 150), (5, 151), (5, 154), (5, 155),
                                  (6, 58), (6, 62), (6, 96), (6, 113), (6, 115), (6, 132), (6, 145), (6, 146), (6, 149),
                                  (6, 151), (6, 154),
                                  (7, 71), (7, 72), (7, 75), (7, 76), (7, 79), (7, 145), (7, 146),
                                  (8, 25), (8, 32), (8, 54), (8, 57), (8, 76), (8, 79), (8, 80), (8, 83), (8, 94),
                                  (8, 96), (8, 115), (8, 122), (8, 123), (8, 141), (8, 142), (8, 145), (8, 146)]

    __netmhcpan_contact_sites_9 = [(0, 6), (0, 58), (0, 61), (0, 62), (0, 65), (0, 158), (0, 162), (0, 166), (0, 170),
                                   (1, 6), (1, 8), (1, 23), (1, 44), (1, 61), (1, 62), (1, 65), (1, 66), (1, 69),
                                   (1, 98), (1, 158),
                                   (2, 69), (2, 96), (2, 98), (2, 113), (2, 155), (2, 158),
                                   (3, 65), (3, 157), (3, 158), (3, 162),
                                   (4, 68), (4, 69), (4, 157),
                                   (5, 68), (5, 69), (5, 72), (5, 73), (5, 96), (5, 155),
                                   (6, 68), (6, 72), (6, 96), (6, 113), (6, 146), (6, 149), (6, 151), (6, 155),
                                   (7, 72), (7, 75), (7, 76), (7, 146),
                                   (8, 73), (8, 76), (8, 79), (8, 80), (8, 83), (8, 94), (8, 96), (8, 115), (8, 117),
                                   (8, 142), (8, 146)]

    def __init__(self, add_all_contact_sites=True):
        self._peplen_cs_map = {}
        if add_all_contact_sites:
            self.add_contact_sites(8, self.__imgt_contact_sites_8)
            self.add_contact_sites(9, self.__imgt_contact_sites_9)
            self.add_contact_sites(9, self.__netmhcpan_contact_sites_9)
            self.add_contact_sites(9, self.__epiccapo_contact_sites_9)

        self._hlacls_msa_map = {}
        self.set_msa(Const.HLA_CLASSI_A,
                     IMGTMultipleSequenceAlignment.from_file(filename=Const.FN_IMGT_HLA_CLASSI_A_PROT,
                                                             offset=Const.BD_G_ALPHA1_POS[0]))
        self.set_msa(Const.HLA_CLASSI_B,
                     IMGTMultipleSequenceAlignment.from_file(filename=Const.FN_IMGT_HLA_CLASSI_B_PROT,
                                                             offset=Const.BD_G_ALPHA1_POS[0]))
        self.set_msa(Const.HLA_CLASSI_C,
                     IMGTMultipleSequenceAlignment.from_file(filename=Const.FN_IMGT_HLA_CLASSI_C_PROT,
                                                             offset=Const.BD_G_ALPHA1_POS[0]))

    def add_contact_sites(self, peptide_length, contact_sites):
        if not self._peplen_cs_map.has_key(peptide_length):
            self._peplen_cs_map[peptide_length] = []

        target_cs = self._peplen_cs_map[peptide_length]
        for cs in contact_sites:
            if cs not in target_cs:
                target_cs.append(cs)

        target_cs.sort(cmp=lambda x, y: cmp(x[0], y[0]))

    def remove_contact_sites(self, peptide_length, contact_sites):
        target_cs = self._peplen_cs_map[peptide_length]
        for cs in contact_sites:
            if cs in target_cs:
                target_cs.remove(cs)

    def set_contact_sites(self, peptide_length, contact_sites):
        self._peplen_cs_map[peptide_length] = contact_sites

    def set_msa(self, hlacls, msa):
        self._hlacls_msa_map[hlacls] = msa

    def contact_sites(self, pep_len, p_margin=0, h_margin=0, pseudo=False):
        #         Tracer()()
        old_css = self._peplen_cs_map[pep_len]
        css = None
        # Extend contact sites by p_margin(peptide sites) and h_margin(hla sites)
        if p_margin > 0 or h_margin > 0:
            css = []
            hla_len = max([ cs[1] for cs in old_css ])
            for cs in old_css:
                p_begin = max(0, cs[0] - p_margin)
                p_end = min(pep_len, cs[0] + p_margin + 1)
                h_begin = max(0, cs[1] - h_margin)
                h_end = min(hla_len, cs[1] + h_margin + 1)
                for cs_i in range(p_begin, p_end):
                    for cs_j in range(h_begin, h_end):
                        new_cs = (cs_i, cs_j)
                        if new_cs not in css:
                            css.append(new_cs)
        elif p_margin < 0 or h_margin < 0:
            css = []
            h_sites = sorted(np.unique([cs[1] for cs in old_css]))
            for cs_i in range(pep_len):
                for cs_j in h_sites:
                    css.append((cs_i, cs_j))
        else:
            css = old_css

        if pseudo:
            pd_sites = sorted(np.unique([cs[1] for cs in css]))
            css = [(cs[0], pd_sites.index(cs[1])) for cs in css]
        return css

    def contact_site_seq(self, pep_len, allele, margin=0):
        #         Tracer()()
        seq = self.rep_allele_seq(allele)
        #         css = None
        #         if margin > 0:
        #             css = []
        #             for cs in self.contact_sites(pep_len):
        #                 begin = max(0, cs[1] - margin)
        #                 end = min(len(seq), cs[1] + margin + 1)
        #                 for si in range(begin, end):
        #                     new_cs = (cs[0], si)
        #                     if new_cs not in css:
        #                         css.append(new_cs)
        #         else:
        #             css = self.contact_sites(pep_len)

        css = self.contact_sites(pep_len, h_margin=margin)
        sites = sorted(np.unique([cs[1] for cs in css]))
        return ''.join([seq[i] for i in sites])

    def aa_weights_at(self, hlacls, pos):
        msa = self._hlacls_msa_map[hlacls]
        return msa.aa_probs_at(pos)

    def rep_aa_at(self, hlacls, pos):
        #         weights = self.aa_weights_at(hlacls, pos)
        #         return commons.AMINO_ACIDS[np.argmax(weights)]
        msa = self._hlacls_msa_map[hlacls]
        return msa.rep_aa_at(pos)

    def aas_at(self, hlacls, pos):
        msa = self._hlacls_msa_map[hlacls]
        return msa.aas_at(pos)

    def shape(self, hlacls):
        msa = self._hlacls_msa_map[hlacls]
        return msa.shape

    def target_hla_cls(self):
        return self._hlacls_msa_map.keys()

    def has_hla_cls(self, hla_cls):
        return self.target_hla_cls().count(hla_cls) > 0

    def target_peptide_len(self):
        return self._peplen_cs_map.keys()

    def has_peptide_len(self, pep_len):
        return self.target_peptide_len().count(pep_len) > 0

    def rep_seq(self, hlacls):
        msa = self._hlacls_msa_map[hlacls]
        return msa.rep_seq().values

    def rep_allele_seq(self, allele):
        hlacls = HLAAlleleUtils.hla_class(allele)
        msa = self._hlacls_msa_map[hlacls]
        return msa.rep_allele_seq(allele)

    def has_allele(self, hlacls, allele):
        if hlacls in self._hlacls_msa_map:
            msa = self._hlacls_msa_map[hlacls]
            return msa.has_allele(allele)
        return False

    def alleles(self, hla_cls):
        msa = self._hlacls_msa_map[hla_cls]
        return msa.alleles()

    def domain_range(self, peplen):
        begin, end = StatUtils.minmax([cs[1] for cs in self.contact_sites(peplen)])
        return begin, end + 1

    def domain_seq(self, peplen, hlacls, allele, margin=0):

        msa = self._hlacls_msa_map[hlacls]
        seq = msa.rep_allele_seq(allele)
        seqlen = len(seq)
        begin, end = tuple(np.subtract(self.domain_range(peplen), (margin, -margin)))
        if begin < 0:
            begin = 0
        if end > seqlen:
            end = seqlen
        return seq[begin:end]

    def get_msa(self, hlacls):
        return self._hlacls_msa_map[hlacls]

    def binding_image(self, allele, pep_seq, p_margin=0, h_margin=0, p_aa_scorer=None, h_aa_scorer=None, aai_scorer=None):
        #         Tracer()()
        #         hla_cls = HLAAlleleUtils.hla_class(allele)
        hla_seq = self.rep_allele_seq(allele)
        hla_len = len(hla_seq)
        pep_len = len(pep_seq)
        css = self.contact_sites(pep_len, p_margin=p_margin, h_margin=h_margin)

        p_sites = sorted(np.unique([cs[0] for cs in css]))
        h_sites = sorted(np.unique([cs[1] for cs in css]))
        n_scores = 0
        if p_aa_scorer is not None:
            n_scores += p_aa_scorer.n_scores()
        if h_aa_scorer is not None:
            n_scores += h_aa_scorer.n_scores()
        if aai_scorer is not None:
            n_scores += aai_scorer.n_scores()
        # n_scores = (0 if aa_scorer is None else aa_scorer.n_scores()*2) + (0 if aai_scorer is None else aai_scorer.n_scores())
        mat = np.zeros((n_scores, len(p_sites), len(h_sites)))
        for cs in css:
            pa = pep_seq[cs[0]]
            ha = hla_seq[cs[1]]
            pi = p_sites.index(cs[0])
            hi = h_sites.index(cs[1])
            scores = np.asarray([])
            if p_aa_scorer is not None:
                scores = np.append(scores, p_aa_scorer.scores(pa).values)
            if h_aa_scorer is not None:
                scores = np.append(scores, h_aa_scorer.scores(ha).values)
            if aai_scorer is not None:
                scores = np.append(scores, aai_scorer.scores(pa, ha).values)

            for si in range(scores.shape[0]):
                mat[si, pi, hi] = scores[si]

        return mat

###############################################
import pandas as pd
import unittest

class PanMHCIBindingDomainTest(unittest.TestCase):
    def setUp(self):
        self.bdomain = PanMHCIBindingDomain()
        self.target_peplen = 9

    #         self.target_alleles = ['HLA-A*01:01', 'HLA-A*02:01', 'HLA-A*02:02', 'HLA-A*02:03',
    #                                'HLA-A*02:04', 'HLA-A*02:05', 'HLA-A*02:06', 'HLA-A*02:07',
    #                                'HLA-A*02:10', 'HLA-A*02:11', 'HLA-A*02:12', 'HLA-A*02:16',
    #                                'HLA-A*02:17', 'HLA-A*02:19', 'HLA-A*02:50', 'HLA-A*03:01',
    #                                'HLA-A*03:02', 'HLA-A*03:19', 'HLA-A*11:01', 'HLA-A*11:02',
    #                                'HLA-A*23:01', 'HLA-A*24:02', 'HLA-A*24:03', 'HLA-A*25:01',
    #                                'HLA-A*26:01', 'HLA-A*26:02', 'HLA-A*26:03', 'HLA-A*29:02',
    #                                'HLA-A*30:01', 'HLA-A*30:02', 'HLA-A*31:01', 'HLA-A*32:01',
    #                                'HLA-A*32:07', 'HLA-A*32:15', 'HLA-A*33:01', 'HLA-A*66:01',
    #                                'HLA-A*68:01', 'HLA-A*68:02', 'HLA-A*68:23', 'HLA-A*69:01',
    #                                'HLA-A*74:01', 'HLA-A*80:01']

    #     def is_all_amino_acids(self, seq):
    #         return all([(aa in Const.AMINO_ACIDS) for aa in seq])

    #     def test_rep_allele_seq(self):
    #         for allele in self.target_alleles:
    #             seq = self.bdomain.rep_allele_seq(self.target_hlacls, allele)
    #             self.assertTrue(self.is_all_amino_acids(seq))

    #     def test_contact_site_seq(self):
    #         contact_sites = np.unique([cs[1] for cs in self.bdomain.contact_sites(self.target_peplen)])
    #         n_css = len(contact_sites)
    #         for allele in self.target_alleles:
    #             seq = self.bdomain.contact_site_seq(self.target_hlacls, self.target_peplen, allele)
    #             self.assertEquals(n_css, len(seq))
    #             self.assertTrue(self.is_all_amino_acids(seq))

    #     def test_pseudo_contact_sites(self):
    #         css1 = self.bdomain.contact_sites(self.target_peplen)
    #         css2 = self.bdomain.contact_sites(self.target_peplen, pseudo=True)

    #         self.assertEquals(len(css1), len(css2))

    def test_binding_image(self):
        scaler = MinMaxScaler()
        aa_scorer = WenLiuAAPropScorer(corr_cutoff=0.8, data_transformer=scaler)
        aa_scorer.load_score_tab()

        allele = 'HLA-A*02:01'
        pep_seq = 'SSKGLACYR'
        n_channels = aa_scorer.n_scores()*2
        pep_len = len(pep_seq)
        hla_seq = self.bdomain.rep_allele_seq(allele)
        hla_len = len(hla_seq)

        mat = self.bdomain.binding_image(allele= allele,
                                         pep_seq= pep_seq,
                                         p_aa_scorer= aa_scorer,
                                         h_aa_scorer= aa_scorer)

        self.assertEquals((n_channels, pep_len, hla_len), mat.shape)

    def test_contact_sites(self):
        css1 = self.bdomain.contact_sites(9)
        p_sites = sorted(np.unique([cs[0] for cs in css1]))
        h_sites = sorted(np.unique([cs[1] for cs in css1]))
        css2 = self.bdomain.contact_sites(9, p_margin=-1)
        self.assertEquals(len(p_sites)*len(h_sites), len(css2))
        print css2


    def test_netmhcpan_contact_site_seq(self):
        old_css = self.bdomain.contact_sites(9)
        self.bdomain.set_contact_sites(9, self.bdomain._PanMHCIBindingDomain__netmhcpan_contact_sites_9)
        seqtab = pd.read_table('datasets/netmhcpan_HLA_pseudo.txt', sep=' ')
        for allele, seq in zip(seqtab.allele, seqtab.seq):
            hlacls = HLAAlleleUtils.hla_class(allele)
            print 'Allele:', allele
            print 'HLA class:', hlacls
            print 'NetMHCPan seq:', seq
            if hlacls is not None:
                if self.bdomain.has_allele(hlacls, allele):
                    seq1 = self.bdomain.contact_site_seq(9, allele)
                    print 'Contact seq:', seq1
                    self.assertEquals(seq, seq1)
        self.bdomain.set_contact_sites(9, old_css)


if __name__ == '__main__':
    unittest.main()
