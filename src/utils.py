import unittest
import re
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from model.bdomain import PanMHCIBindingDomain

class Utils(object):
    AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
    @staticmethod
    def is_valid_aaseq(seq):
        for aa in seq:
            if aa not in Utils.AMINO_ACIDS:
                return False
        return True

    @staticmethod
    def split_fasta(fasta, seq_len=None):
        seqs = None
        seqs = re.split("^>.*\\n", fasta.strip(' \n\t'), flags=re.MULTILINE)
        seqs = [ s.replace('\n', '').strip(' \n\t') for s in seqs[1:] ]
        # filter empty seqs
        seqs = filter(lambda seq: len(seq) > 0, seqs)
        if len(seqs) <= 0:
            # print 'empty seqs'
            raise SyntaxError('No sequence with valid FASTA format')

        for seq in seqs:
            if not Utils.is_valid_aaseq(seq):
                # print aa, 'is not amino acid'
                raise SyntaxError('Invalid amino acid sequence:%s' % seq)
            if seq_len is not None and len(seq) != seq_len:
                # print 'seq_len:', seq_len, ',len(seq):', len(seq)
                raise SyntaxError(' %s lengh mismatched: %s != %s' % (seq, seq_len, len(seq)))
        return seqs

    @staticmethod
    def split_seqs(seq_txt, seq_len=None, delim=' \n\t,;-'):
        striped_txt = seq_txt.strip(' \n\t')
        seqs = None
        try:
            seqs = Utils.split_fasta(fasta=striped_txt, seq_len=seq_len)
        except SyntaxError as e:
            pass

        if seqs is None:
            seqs = re.split("[%s]+" % delim, striped_txt.strip(' \n\t'))
            seqs = [ s.strip(' \n\t') for s in seqs]
        # filter empty seqs
        seqs = filter(lambda seq: len(seq) > 0, seqs)
        if len(seqs) <= 0:
            # print 'empty seqs'
            raise SyntaxError('No sequence with valid format')
        for seq in seqs:
            if not Utils.is_valid_aaseq(seq):
                # print 'not valid amino acid'
                raise SyntaxError('Invalid amino acid sequence:%s' % seq)
            if seq_len is not None and len(seq) != seq_len:
                # print 'seq_len:', seq_len, ',len(seq):', len(seq)
                raise SyntaxError(' %s lengh mismatched: %s != %s' % (seq, seq_len, len(seq)))
        return seqs

    @staticmethod
    def show_bind_image(img, css=None, ax=None, title=None, annot=True, cbar=False):

        p_sites = sorted(np.unique([cs[0] for cs in css]) + 1)
        h_sites = sorted(np.unique([cs[1] for cs in css]) + 1)

        g = sns.heatmap(img, ax=ax, annot=annot, linewidths=.4, cbar=cbar)
        g.set(title=title)
        g.set_xticklabels(h_sites, rotation=90)
        g.set_yticklabels(p_sites[::-1])
        return g

class UtilsTest(unittest.TestCase):
    def setUp(self):
        self.validFastas = [">1\nAGYMNAAK", ">1\nAGYMNAAK\n>2\nMMMTTTAAK\n>3\nLLLYYYRRR\n", ">1\nAGYMNAAK\n\n>2\nMMMTTTAAK\n\n\n>3\nLLLYYYRRR", "MMMTTTAAK\n>1\nMMMTTTAAK"]
        self.invalidFastas = [">XXGYMNAAK\n", "MMMTTTAAK\nLLLYYYRRR\n", ">1\nMMMTTTAAK\n>2\nXLLYYYRRR"]
        self.validSeqs = ["LLLYYYRRR", "AGYMNAAK, LLLYYYRRR, MMMTTTAAK", "MMMTTTAAK\tMMMTTTAAK\tLLLYYYRRR\t", "LLLYYYRRR\nLLLYYYRRR,MMMTTTAAK YYYYKKKKMMM\tMMMYYYYKKK"]
        self.invalidSeqs = ["AGYMNAAK^LLLYYYRRR, MMMTTTAAK", "XXMMTTTAAK\tMMMTTTAAK\tLLLYYYRRR\t", ""]

    def test_split_fasta(self):


        for ss in self.validFastas:
            seqs = Utils.split_fasta(fasta=ss)
            print seqs
            self.assertIsNotNone(seqs)

        for ss in self.invalidFastas:

            with self.assertRaises(SyntaxError) as err:
                Utils.split_fasta(fasta=ss)
            print err.exception

        for ss in self.validFastas:
            with self.assertRaises(SyntaxError) as err:
                Utils.split_fasta(fasta=ss, seq_len=10)
            print err.exception

    def test_split_seqs(self):
        for ss in self.validSeqs:
            seqs = Utils.split_seqs(seq_txt=ss)
            print seqs
            self.assertIsNotNone(seqs)

        for ss in self.invalidSeqs:
            with self.assertRaises(SyntaxError) as err:
                Utils.split_seqs(seq_txt=ss)
            print err.exception

        for ss in self.validSeqs:
            with self.assertRaises(SyntaxError) as err:
                Utils.split_seqs(seq_txt=ss, seq_len=11)
            print err.exception

    def test_split_seqs(self):
        print Utils.split_seqs(seq_txt='RRRRRRRRR	KKKKKKKKK\nMMMMMMMMM', seq_len=9)

    def test_show_bind_imgage(self):
        css = [ (i, j) for i in range(0, 9) for j in range(0, 34) ]
        img = np.random.rand(18, 9, 34).tolist();

        sns.set_context('paper', font_scale=1.1)
        sns.axes_style('white')

        fig, axes = plt.subplots(nrows=1, ncols=1)
        fig.set_figwidth(8)
        fig.set_figheight(3)
        plt.tight_layout()
        fig.subplots_adjust(top=0.87, bottom=0.22)

        ax = Utils.show_bind_image(np.mean(img, axis=0), css=css, annot=False, ax=axes, cbar=False, title="Informative pixels")
        ax.set(xlabel='HLA contact site', ylabel='Peptide position')
        plt.show()

    def test_eval(self):
        with open('datasets/alleles.txt', 'r') as f:
            print type(eval(f.read()))

if __name__ == '__main__':
    unittest.main()
