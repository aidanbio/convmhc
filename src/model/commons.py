import numpy as np
import pandas as pd
import re
import logging
import pickle
import matplotlib.pyplot as plt
import itertools
import os


# Common constancts
class Const(object):
    EPS = 0.000001
    AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
    AA_PAIRS = [(AMINO_ACIDS[i], AMINO_ACIDS[j]) for i in range(len(AMINO_ACIDS)) for j in range(i, len(AMINO_ACIDS))]

    HLA_CLASSI_A = 'HLA-A'
    HLA_CLASSI_B = 'HLA-B'
    HLA_CLASSI_C = 'HLA-C'
    HLA_CLASSI = [HLA_CLASSI_A, HLA_CLASSI_B, HLA_CLASSI_C]
    BINDING_PEPIDE_LENGTHS = range(8, 11)
    BD_G_ALPHA1_POS = (24, 89)
    BD_G_ALPHA2_POS = (90, 182)

    # For loading datasets
    BINDING_MEASUREMENT_TYPE_IC50 = 'ic50'
    BINDING_MEASUREMENT_TYPE_BIN = 'binary'
    BINDING_MEASUREMENT_TYPE_HALF_ALIVE = 't1/2'
    BINDING_MEASUREMENT_TYPES = [BINDING_MEASUREMENT_TYPE_IC50, BINDING_MEASUREMENT_TYPE_BIN,
                                 BINDING_MEASUREMENT_TYPE_HALF_ALIVE]

    CN_BD_TAB_SPECIES = 'species'
    CN_BD_TAB_ALLELE = 'allele'
    CN_BD_TAB_HLA_CLS = 'hla_cls'
    CN_BD_TAB_PEPTIDE_LEN = 'peptide_len'
    CN_BD_TAB_PEPTIDE_SEQ = 'peptide_seq'
    CN_BD_TAB_MEAS_TYPE = 'meas_type'
    CN_BD_TAB_MEAS_VALUE = 'meas_value'
    CN_BD_TAB_BINDER = 'binder'
    CN_BD_TAB = [CN_BD_TAB_SPECIES, CN_BD_TAB_ALLELE, CN_BD_TAB_HLA_CLS, CN_BD_TAB_PEPTIDE_LEN, CN_BD_TAB_PEPTIDE_SEQ,
                 CN_BD_TAB_MEAS_TYPE, CN_BD_TAB_MEAS_VALUE, CN_BD_TAB_BINDER]

    # For multiple sequence alignment files
    FN_IMGT_HLA_CLASSI_A_PROT = 'datasets/A_prot.txt'
    FN_IMGT_HLA_CLASSI_B_PROT = 'datasets/B_prot.txt'
    FN_IMGT_HLA_CLASSI_C_PROT = 'datasets/C_prot.txt'

    # For logging
    LOG_NAME = 'ybind'
    LOG_FILENAME = '/home/hym/trunk/ybind/ybind.log'


class PlotUtils(object):
    @staticmethod
    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')


class FileUtils(object):
    @staticmethod
    def pkl_save(fn, target):
        with open(fn, 'wb') as fh:
            pickle.dump(target, fh)

    @staticmethod
    def pkl_load(fn):
        with open(fn, 'rb') as fh:
            return pickle.load(fh)

    @staticmethod
    def rm_files(path, fn_ptrn):
        #         print 'Pattern:', fn_ptrn
        for fn in os.listdir(path):
            #             print 'Current file:', fn
            if re.match(fn_ptrn, fn) is not None:
                #                 print 'Remove file:', fn
                os.remove('%s/%s' % (os.path.normpath(path), fn))

    @staticmethod
    def list_files(path, fn_ptrn):
        fns = []
        for fn in os.listdir(path):
            if re.match(fn_ptrn, fn) is not None:
                fns.append(fn)
        return fns


# class DataUtils(object):
#     @staticmethod
#     def load_datatab(fn, header=0, index_col=0, corr_cutoff=0.8, dropna=True, scaler=None, dim_reducer=None):
#         datatab = None
#         if FileUtils.is_csv(fn):
#             datatab = pd.read_csv(fn, header=header, index_col=index_col)
#
#         # Dealing with missing data
#         if dropna:
#             datatab = datatab.dropna(axis=1)
#
#         # Scaling
#         if scaler is not None:
#             datatab = pd.DataFrame(scaler.fit_transform(datatab.values), index=datatab.index, columns=datatab.columns)
#
#         # Remove highly correlated features
#         if corr_cutoff is not None:
#             hcorr = StatUtils.find_corr(datatab, corr_cutoff)
#             datatab = datatab.drop(hcorr, 1)
#
#         # Dimension reduction
#         if dim_reducer is not None:
#             X_trans = dim_reducer.fit_transform(datatab.values)
#             datatab = pd.DataFrame(X_trans, index=datatab.index, columns=['F%s' % (i) for i in range(X_trans.shape[1])])
#
#         return datatab


class MHCAlleleName(object):
    GROUP_SEP = '*'
    FIELD_SEP = ':'
    EXP_CHG_TYPES = ['N', 'L', 'S', 'A', 'Q']

    def __init__(self, prefix=None, gene=None, group=None, nonsyn_mut=None, syn_mut=None, noncod_mut=None,
                 exp_chg=None):
        self.prefix = prefix
        self.gene = gene
        self.group = group
        self.nonsyn_mut = nonsyn_mut
        self.syn_mut = syn_mut
        self.noncod_mut = noncod_mut
        self.exp_chg = exp_chg

    def format(self, group_sep=GROUP_SEP, field_sep=FIELD_SEP):
        if group_sep is None:
            group_sep = ''
        if field_sep is None:
            field_sep = ''

        s = '%s-%s' % (self.prefix, self.gene)
        if self.group is not None:
            s += '%s%s' % (group_sep, self.group)
        if self.nonsyn_mut is not None:
            s += '%s%s' % (field_sep, self.nonsyn_mut)
        if self.syn_mut is not None:
            s += '%s%s' % (field_sep, self.syn_mut)
        if self.noncod_mut is not None:
            s += '%s%s' % (field_sep, self.noncod_mut)
        if self.exp_chg is not None:
            s += self.exp_chg
        return s

    @classmethod
    def parse(cls, s, group_sep=GROUP_SEP, field_sep=FIELD_SEP):
        #         Tracer()()
        allele = MHCAlleleName()
        tokens = s.split('-', 1)
        allele.prefix = tokens[0]
        tokens = tokens[1].split(group_sep, 1)
        allele.gene = tokens[0]
        if len(tokens) > 1:
            tmp = tokens[1]
            if field_sep is None:
                allele.group = tmp[0:2]
                if len(tmp) > 2:
                    allele.nonsyn_mut = tmp[2:]
            else:
                fields = tmp.split(field_sep)
                try:
                    allele.group = fields[0]
                    allele.nonsyn_mut = fields[1]
                    allele.syn_mut = fields[2]
                    allele.noncod_mut = fields[3]
                    if allele.noncod_mut is not None:
                        tmp = allele.noncod_mut
                        l = len(tmp)
                        if tmp[l - 1] in cls.EXP_CHG_TYPES:
                            allele.noncod_mut = tmp[:l - 1]
                            allele.exp_chg = tmp[l - 1]
                except IndexError:
                    pass
        return allele


class HLAAlleleUtils(object):
    __super_type_map = {'A0101': 'A01', 'A2601': 'A01', 'A0101': 'A01', 'A0101': 'A01'}

    @staticmethod
    def valid_name(allele):
        #         Tracer()()
        an = re.sub('HLA-', '', allele)
        pattern = '^[A-C][1-9/]*(\*[0-9]{2,}(\:[0-9]{2,}){0,3}[NLSAQ]?)?$'
        m = re.search(pattern, an)

        return m is not None

    @staticmethod
    def hla_class(allele):
        for hcls in Const.HLA_CLASSI:
            loc = allele.find(hcls)
            if loc >= 0:
                return hcls
        return None

    @staticmethod
    def equal_names(allele1, allele2, level=-1):  # level: 0(gene), 1(group), 2(protein), 3(all), and -1(minimal)
        #         Tracer()()

        if (not HLAAlleleUtils.valid_name(allele1)) or (not HLAAlleleUtils.valid_name(allele2)):
            raise ValueError('Invalid format of HLA allele name: %s or %s' % (allele1, allele2))

        an1 = re.sub('HLA-', '', allele1)
        an2 = re.sub('HLA-', '', allele2)

        tks1 = an1.split('*')
        tks2 = an2.split('*')
        if level < 0:
            na = min(an1.count('*'), an2.count('*'))
            nc = min(an1.count(':'), an2.count(':'))
            if na == 0:
                level = 0
            else:
                level = nc + 1

        if level == 0:
            return tks1[0] == tks2[0]
        fds1 = tks1[1].split(':')
        fds2 = tks2[1].split(':')
        if level == 1:
            return (tks1[0] == tks2[0]) and (fds1[0] == fds2[0])
        if level == 2:
            return (tks1[0] == tks2[0]) and (fds1[0] == fds2[0]) and (fds1[1] == fds2[1])

        return an1 == an2

    @staticmethod
    def sub_name(allele,
                 level=0):  # level: 0(gene), 1(group), 2(protein), 3(synonymous mutation in exons), 4(synonymous mutation in introns)
        if not HLAAlleleUtils.valid_name(allele):
            raise ValueError('Invalid format of HLA allele name: %s' % allele)
        if level not in range(5):
            raise ValueError('Level allowed to 0-4: %s' % level)

        tks = allele.split('*')
        if level == 0:
            return tks[0]
        else:
            if len(tks) != 2:
                raise ValueError('Out of bound: %s for level > 0' % allele)
            fds = tks[1].split(':')
            if level == 1:
                return '%s*%s' % (tks[0], fds[0])
            elif level == 2:
                if len(fds) < 2:
                    raise ValueError('Out of bound: %s for level == 2' % allele)
                return '%s*%s:%s' % (tks[0], fds[0], fds[1])
            elif level == 3:
                if len(fds) < 3:
                    raise ValueError('Out of bound: %s for level == 3' % allele)
                return '%s*%s:%s:%s' % (tks[0], fds[0], fds[1], fds[2])
            else:  # level == 4
                if len(fds) < 4:
                    raise ValueError('Out of bound: %s for level == 4' % allele)
                return '%s*%s:%s:%s:%s' % (tks[0], fds[0], fds[1], fds[2], fds[3])


class StatUtils(object):
    @staticmethod
    def minmax(x):
        return (min(x), max(x))

    @staticmethod
    def find_corr(tab, cutoff=0.8):
        corr = tab.corr()
        colnames = tab.columns
        target = []
        for i in range(len(colnames)):
            for j in range(i + 1, len(colnames)):
                cur = np.abs(corr.values[i, j])
                #             print('Current:%s(%s), %s(%s): %s' % (colnames[i], i, colnames[j], j, cur))
                if cur >= cutoff:
                    #                 print('%s\t%s\t%s' % (colnames[i], colnames[j], cur))
                    target.append(colnames[j])
                    #                 print('Appended col:%s' % (colnames[j]))
        return np.unique(target)

    @staticmethod
    def almost_equals(f1, f2):
        return abs(f1 - f2) < Const.EPS


class PrintUtils(object):
    @staticmethod
    def fullprint(*args, **kwargs):
        from pprint import pprint
        import numpy
        import pandas as pd
        opt = numpy.get_printoptions()
        max_rows = pd.options.display.max_rows
        numpy.set_printoptions(threshold='nan')
        pd.options.display.max_rows = None
        pprint(*args, **kwargs)
        numpy.set_printoptions(**opt)
        pd.options.display.max_rows = max_rows


class LogUtils(object):
    @staticmethod
    def has_filehandler(loggername, filename):
        logger = logging.getLogger(loggername)
        for handler in logger.handlers:
            if type(handler) is logging.FileHandler:
                if handler.baseFilename == filename:
                    return True
        return False

    @staticmethod
    def get_default_logger(loggername, filename):
        logger = logging.getLogger(loggername)
        logger.propagate = False
        for handler in logger.handlers:
            if type(handler) is logging.FileHandler:
                if handler.baseFilename == filename:
                    return logger

        fh = logging.FileHandler(filename, mode='a')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s'))
        logger.addHandler(fh)
        logger.setLevel(logging.DEBUG)
        return logger


class StrUtils(object):
    @staticmethod
    def rm_nonwords(s):
        import re
        return re.sub('\\W', '', s)


class SeqUtils(object):
    @staticmethod
    def write_fa(fn, seqs, headers=None):
        with open(fn, 'w') as fh:
            fh.write(SeqUtils.format_fa(seqs, headers))

    @staticmethod
    def format_fa(seqs, headers=None):
        return '\n'.join(
            map(lambda h, seq: '>%s\n%s' % (h, seq), range(1, len(seqs) + 1) if headers is None else headers, seqs))


class TypeConvertUtils(object):
    @staticmethod
    def to_boolean(x):
        #         Tracer()()
        if type(x) is int or type(x) is float:
            return x >= 1
        if type(x) is str:
            upper_x = x.upper()
            return upper_x in ['TRUE', 'T']
        return None


class ArrayUtils(object):
    @staticmethod
    def intersect2d(a1, a2):
        if len(a1.shape) != 2 or len(a2.shape) != 2:
            return None
        return np.array([x for x in set(tuple(x) for x in a1) & set(tuple(x) for x in a2)])

    @staticmethod
    def diff2d(a1, a2):
        if len(a1.shape) != 2 or len(a2.shape) != 2:
            return None
        return np.array([x for x in set(tuple(x) for x in a1) - set(tuple(x) for x in a2)])

    @staticmethod
    def union2d(a1, a2):
        if len(a1.shape) != 2 or len(a2.shape) != 2:
            return None
        return np.array([x for x in set(tuple(x) for x in a1) | set(tuple(x) for x in a2)])


class singleton:
    """
    A non-thread-safe helper class to ease implementing singletons.
    This should be used as a decorator -- not a metaclass -- to the
    class that should be a singleton.

    The decorated class can define one `__init__` function that
    takes only the `self` argument. Other than that, there are
    no restrictions that apply to the decorated class.

    To get the singleton instance, use the `Instance` method. Trying
    to use `__call__` will result in a `TypeError` being raised.

    Limitations: The decorated class cannot be inherited from.

    """

    def __init__(self, decorated):
        self._decorated = decorated

    def get_instance(self):
        """
        Returns the singleton instance. Upon its first call, it creates a
        new instance of the decorated class and calls its `__init__` method.
        On all subsequent calls, the already created instance is returned.

        """
        try:
            return self._instance
        except AttributeError:
            self._instance = self._decorated()
            return self._instance

    def __call__(self):
        raise TypeError('singletons must be accessed through `get_instance()`.')

    def __instancecheck__(self, inst):
        return isinstance(inst, self._decorated)


import unittest

class HLAAlleleUtilsTest(unittest.TestCase):
    def test_valid_name(self):
        self.assertTrue(HLAAlleleUtils.valid_name('HLA-A*01'))
        self.assertTrue(HLAAlleleUtils.valid_name('HLA-A*05:06'))
        self.assertTrue(HLAAlleleUtils.valid_name('HLA-A*77:88888'))
        self.assertTrue(HLAAlleleUtils.valid_name('HLA-A*05:77N'))
        self.assertTrue(HLAAlleleUtils.valid_name('HLA-A*01:01:01'))
        self.assertTrue(HLAAlleleUtils.valid_name('HLA-A*01:01:01L'))
        self.assertTrue(HLAAlleleUtils.valid_name('HLA-A*01:02:02:01'))
        self.assertTrue(HLAAlleleUtils.valid_name('HLA-A*01:02:04:67Q'))
        self.assertTrue(HLAAlleleUtils.valid_name('A*01'))
        self.assertTrue(HLAAlleleUtils.valid_name('A*05:06'))
        self.assertTrue(HLAAlleleUtils.valid_name('A*05:77'))
        self.assertTrue(HLAAlleleUtils.valid_name('A*77:88888'))
        self.assertTrue(HLAAlleleUtils.valid_name('HLA-A13'))
        self.assertTrue(HLAAlleleUtils.valid_name('HLA-A3/11'))
        self.assertTrue(HLAAlleleUtils.valid_name('HLA-A'))

        self.assertFalse(HLAAlleleUtils.valid_name('HLA-A*'))
        self.assertFalse(HLAAlleleUtils.valid_name('HLA-A*XX'))
        self.assertFalse(HLAAlleleUtils.valid_name('HLA-A*01:NY'))
        self.assertFalse(HLAAlleleUtils.valid_name('HLA-A*01:02:03:06:08'))
        self.assertFalse(HLAAlleleUtils.valid_name('HLA-A*01:02:04:67K'))

    def test_equal_names(self):
        # Gene
        self.assertTrue(HLAAlleleUtils.equal_names('HLA-A*01:01', 'HLA-A*02:01', level=0))
        self.assertFalse(HLAAlleleUtils.equal_names('HLA-A*01:01', 'HLA-B*01:01', level=0))

        # Group
        self.assertTrue(HLAAlleleUtils.equal_names('HLA-A*01:01', 'HLA-A*01:02', level=1))
        self.assertFalse(HLAAlleleUtils.equal_names('HLA-A*01:01', 'HLA-B*01:01', level=1))
        self.assertFalse(HLAAlleleUtils.equal_names('HLA-A*01:01', 'HLA-A*02:01', level=1))

        # Protein
        self.assertTrue(HLAAlleleUtils.equal_names('HLA-A*01:01:01', 'HLA-A*01:01:02', level=2))
        self.assertTrue(HLAAlleleUtils.equal_names('HLA-A*01:01', 'HLA-A*01:01:02', level=2))
        self.assertFalse(HLAAlleleUtils.equal_names('HLA-A*01:01:01', 'HLA-B*01:01:01', level=2))
        self.assertFalse(HLAAlleleUtils.equal_names('HLA-A*01:01:01', 'HLA-A*02:01:01', level=2))
        self.assertFalse(HLAAlleleUtils.equal_names('HLA-A*01:01:01', 'HLA-A*01:02:01', level=2))

        # All
        self.assertTrue(HLAAlleleUtils.equal_names('HLA-A*01:01:01', 'HLA-A*01:01:01', level=3))
        self.assertFalse(HLAAlleleUtils.equal_names('HLA-A*01:01:01', 'HLA-A*01:01:02', level=3))
        self.assertFalse(HLAAlleleUtils.equal_names('HLA-A*01:01', 'HLA-A*01:01:02', level=3))
        self.assertFalse(HLAAlleleUtils.equal_names('HLA-A*01:01:01', 'HLA-B*01:01:01', level=3))
        self.assertFalse(HLAAlleleUtils.equal_names('HLA-A*01:01:01', 'HLA-A*02:01:01', level=3))
        self.assertFalse(HLAAlleleUtils.equal_names('HLA-A*01:01:01', 'HLA-A*01:02:01', level=3))

        # Minimal match
        self.assertTrue(HLAAlleleUtils.equal_names('HLA-A', 'HLA-A*01:01:01'))
        self.assertTrue(HLAAlleleUtils.equal_names('HLA-A*11', 'HLA-A*11:01:01'))
        self.assertTrue(HLAAlleleUtils.equal_names('HLA-A*01', 'HLA-A*01:01:01'))
        self.assertTrue(HLAAlleleUtils.equal_names('HLA-A*01:01', 'HLA-A*01:01:01'))
        self.assertTrue(HLAAlleleUtils.equal_names('HLA-A*01:01', 'HLA-A*01:01:02:03N'))
        self.assertFalse(HLAAlleleUtils.equal_names('HLA-A', 'HLA-B*01:01:01'))
        self.assertFalse(HLAAlleleUtils.equal_names('HLA-A*11', 'HLA-A*01:01:01'))
        self.assertFalse(HLAAlleleUtils.equal_names('HLA-A*01:01', 'HLA-A*01:02:02:03N'))
        self.assertFalse(HLAAlleleUtils.equal_names('HLA-A*01:01:01', 'HLA-A*01:01:02:03N'))

        # Invalid format
        with self.assertRaises(ValueError):
            HLAAlleleUtils.equal_names('HLA-A', 'HLA-A*01:01:01')
            HLAAlleleUtils.equal_names('HLA-A*01:02', 'HLA-A*01:02:03:06:08')
            HLAAlleleUtils.equal_names('HLA-A*01:02N', 'HLA-A*01:02:XX')

    def test_sub_name(self):
        # Gene
        self.assertEquals('HLA-A', HLAAlleleUtils.sub_name('HLA-A*01:01:02:03N', level=0))
        self.assertEquals('A', HLAAlleleUtils.sub_name('A*01:01:02:03N', level=0))

        # Group
        self.assertEquals('HLA-A*01', HLAAlleleUtils.sub_name('HLA-A*01:01:02:03N', level=1))
        self.assertEquals('A*21', HLAAlleleUtils.sub_name('A*21:01:02:03N', level=1))

        # Protein
        self.assertEquals('HLA-A*01:01', HLAAlleleUtils.sub_name('HLA-A*01:01:02:03N', level=2))
        self.assertEquals('A*21:22', HLAAlleleUtils.sub_name('A*21:22:02:03N', level=2))

        # Synonymous mutations in exons
        self.assertEquals('HLA-A*01:01:02', HLAAlleleUtils.sub_name('HLA-A*01:01:02:03N', level=3))
        self.assertEquals('A*21:22:02N', HLAAlleleUtils.sub_name('A*21:22:02N', level=3))

        # Synonymous mutations in introns
        self.assertEquals('HLA-A*01:01:02:77A', HLAAlleleUtils.sub_name('HLA-A*01:01:02:77A', level=4))
        self.assertEquals('A*21:22:02:888', HLAAlleleUtils.sub_name('A*21:22:02:888', level=4))

        # Out of level
        with self.assertRaises(ValueError):
            HLAAlleleUtils.sub_name('HLA-A*01:01:02:03N', level=5)
        with self.assertRaises(ValueError):
            HLAAlleleUtils.sub_name('HLA-A*01:01:02:03N', level=-1)

        # Invalid level
        with self.assertRaises(ValueError):
            HLAAlleleUtils.sub_name('HLA-A', level=1)
        with self.assertRaises(ValueError):
            HLAAlleleUtils.sub_name('HLA-A*22', level=2)
        with self.assertRaises(ValueError):
            HLAAlleleUtils.sub_name('HLA-A*22:11', level=3)
        with self.assertRaises(ValueError):
            HLAAlleleUtils.sub_name('HLA-A*22:11:34', level=4)

    def test_hla_class(self):
        self.assertEquals(Const.HLA_CLASSI_A, HLAAlleleUtils.hla_class("HLA-A*02:01"))
        self.assertEquals(Const.HLA_CLASSI_B, HLAAlleleUtils.hla_class("HLA-B*02:01"))



import unittest

@singleton
class ASingleton:
    pass


class SingletonTest(unittest.TestCase):
    def test_singleton(self):
        with self.assertRaises(TypeError):
            ASingleton()

        self.assertEquals(id(ASingleton.get_instance()), id(ASingleton.get_instance()))


class MHCAlleleNameTest(unittest.TestCase):
    def test_parse(self):
        allele = MHCAlleleName.parse('HLA-A*02:101:01:02N')
        self.assertEquals('HLA', allele.prefix)
        self.assertEquals('A', allele.gene)
        self.assertEquals('02', allele.group)
        self.assertEquals('101', allele.nonsyn_mut)
        self.assertEquals('01', allele.syn_mut)
        self.assertEquals('02', allele.noncod_mut)
        self.assertEquals('N', allele.exp_chg)

        allele = MHCAlleleName.parse('HLA-A-02:101', group_sep='-')
        self.assertEquals('HLA', allele.prefix)
        self.assertEquals('A', allele.gene)
        self.assertEquals('02', allele.group)
        self.assertEquals('101', allele.nonsyn_mut)
        self.assertTrue(allele.syn_mut is None)
        self.assertTrue(allele.noncod_mut is None)
        self.assertTrue(allele.exp_chg is None)

        allele = MHCAlleleName.parse('HLA-A-6801', group_sep='-', field_sep=None)
        self.assertEquals('HLA', allele.prefix)
        self.assertEquals('A', allele.gene)
        self.assertEquals('68', allele.group)
        self.assertEquals('01', allele.nonsyn_mut)
        self.assertTrue(allele.syn_mut is None)
        self.assertTrue(allele.noncod_mut is None)
        self.assertTrue(allele.exp_chg is None)

        allele = MHCAlleleName.parse('Gogo-B-0101', group_sep='-', field_sep=None)
        self.assertEquals('Gogo', allele.prefix)
        self.assertEquals('B', allele.gene)
        self.assertEquals('01', allele.group)
        self.assertEquals('01', allele.nonsyn_mut)
        self.assertTrue(allele.syn_mut is None)
        self.assertTrue(allele.noncod_mut is None)
        self.assertTrue(allele.exp_chg is None)

        allele = MHCAlleleName.parse('H-2-Db', group_sep='-', field_sep=None)
        self.assertEquals('H', allele.prefix)
        self.assertEquals('2', allele.gene)
        self.assertEquals('Db', allele.group)
        self.assertTrue(allele.nonsyn_mut is None, 'allele.nonsyn_mut:%s' % allele.nonsyn_mut)
        self.assertTrue(allele.syn_mut is None)
        self.assertTrue(allele.noncod_mut is None)
        self.assertTrue(allele.exp_chg is None)

        allele = MHCAlleleName.parse('HLA-A1', group_sep='-', field_sep=None)
        self.assertEquals('HLA', allele.prefix)
        self.assertEquals('A1', allele.gene)
        self.assertTrue(allele.group is None)
        self.assertTrue(allele.nonsyn_mut is None)
        self.assertTrue(allele.syn_mut is None)
        self.assertTrue(allele.noncod_mut is None)
        self.assertTrue(allele.exp_chg is None)

    def test_format(self):
        allele = MHCAlleleName.parse('HLA-A*02:101:01:02N')
        self.assertEquals('HLA-A*02:101:01:02N', allele.format())

        allele = MHCAlleleName.parse('HLA-A-02:101', group_sep='-')
        self.assertEquals('HLA-A*02:101', allele.format())

        allele = MHCAlleleName.parse('HLA-A-6801', group_sep='-', field_sep=None)
        self.assertEquals('HLA-A*68:01', allele.format())

        allele = MHCAlleleName.parse('Gogo-B-0101', group_sep='-', field_sep=None)
        self.assertEquals('Gogo-B*01:01', allele.format())

        allele = MHCAlleleName.parse('H-2-Db', group_sep='-', field_sep=None)
        self.assertEquals('H-2*Db', allele.format())

        allele = MHCAlleleName.parse('HLA-A1', group_sep='-', field_sep=None)
        self.assertEquals('HLA-A1', allele.format())


# class DataUtilsTest(TestCase):
#     def test_load_datatab_with_corr_cutoff(self):
#         expected_shape_without_corr_cutoff = (20, 11)
#         expected_shape_with_corr_cutoff80 = (20, 7)
#         reduced_ncomp = 3
#         expected_shape_with_dim_reduced = (20, reduced_ncomp)
#
#         tab = DataUtils.load_datatab(fn='datasets/aaprops_Wen_Liu.csv', corr_cutoff=None)
#         self.assertEquals(expected_shape_without_corr_cutoff, tab.shape)
#         tab = DataUtils.load_datatab(fn='datasets/aaprops_Wen_Liu.csv', corr_cutoff=0.8)
#         self.assertEquals(expected_shape_with_corr_cutoff80, tab.shape)
#
#         tab = DataUtils.load_datatab(fn='datasets/aaprops_Wen_Liu.csv', corr_cutoff=0.8,
#                                      dim_reducer=decomposition.PCA(n_components=reduced_ncomp))
#
#         self.assertEquals(expected_shape_with_dim_reduced, tab.shape)
#         print tab


class FileUtilsTest(unittest.TestCase):
    pass
    # def test_rm_files(self):
    #     fns = FileUtils.list_files('tmp/', 'extest*')
    #     self.assertTrue(len(fns) > 0)
    #     FileUtils.rm_files('tmp/', 'extest*')
    #     fns = FileUtils.list_files('tmp/', 'extest*')
    #     self.assertTrue(len(fns) == 0)


class SeqUtilsTest(unittest.TestCase):
    def test_format_fa(self):
        seqs = ['AAA', 'BBB', 'CCC']
        headers = ['HA', 'HB', 'HC']
        expected_with_headers = '>HA\nAAA\n>HB\nBBB\n>HC\nCCC'
        expected_without_headers = '>1\nAAA\n>2\nBBB\n>3\nCCC'
        self.assertEquals(expected_with_headers, SeqUtils.format_fa(seqs=seqs, headers=headers))
        self.assertEquals(expected_without_headers, SeqUtils.format_fa(seqs=seqs))


class TypeConvertUtilsTest(unittest.TestCase):
    def test_to_boolean(self):
        self.assertTrue(TypeConvertUtils.to_boolean(1))
        self.assertTrue(TypeConvertUtils.to_boolean(2))
        self.assertTrue(TypeConvertUtils.to_boolean('True'))
        self.assertTrue(TypeConvertUtils.to_boolean('T'))
        self.assertFalse(TypeConvertUtils.to_boolean(0))
        self.assertFalse(TypeConvertUtils.to_boolean('False'))
        self.assertFalse(TypeConvertUtils.to_boolean('F'))


class ArrayUtilsTest(unittest.TestCase):
    def test_intersect2d(self):
        self.assertTrue(np.all(np.array([[1, 2], [4, 5]]) == ArrayUtils.intersect2d(np.array([[1, 2], [2, 3], [4, 5]]),
                                                                                    np.array([[1, 2], [4, 5]]))))

    def test_diff2d(self):
        self.assertTrue(np.all(
            np.array([[2, 3]]) == ArrayUtils.diff2d(np.array([[1, 2], [2, 3], [4, 5]]), np.array([[1, 2], [4, 5]]))))

    def test_union2d(self):
        print ArrayUtils.union2d(np.array([[1, 2], [2, 3], [4, 5]]), np.array([[1, 2], [4, 5]]))

# self.assertTrue(np.all(np.array([[1, 2], [2, 3], [4, 5]]) == )

if __name__ == '__main__':
    unittest.main()
