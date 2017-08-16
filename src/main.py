from flask import Flask, make_response, render_template, request, jsonify, g, json, url_for
from flask.json import JSONEncoder
import numpy as np
from utils import Utils
import StringIO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import seaborn as sns
from keras.models import load_model
from model.bdomain import PanMHCIBindingDomain
from model.aaprops import WenLiuAAPropScorer
from sklearn.preprocessing import MinMaxScaler
from deeplift.conversion import keras_conversion as kc
from deeplift.blobs import NonlinearMxtsMode
from deeplift.blobs import convolution
import traceback
import os

# Global context
app = Flask(__name__)
model_path = 'datasets/model_AB.h5'

class MyJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, PredictionResult):
            return obj.to_json()
        return super(MyJSONEncoder, self).default(obj)

app.json_encoder = MyJSONEncoder

# def get_model():
#     model = getattr(g, 'model', None)
#     if model is None:
#         model = g.model = PredictionModelWrapper(path=model_path)
#     return model

class PredictionResult(object):
    def __init__(self, allele=None, pep_seq=None, binder_prob=None, binder=None, bind_img=None):
        self.allele = allele
        self.pep_seq = pep_seq
        self.binder_prob = binder_prob
        self.binder = binder
        self.bind_img = bind_img
    def to_json(self):
        return {
            'allele': self.allele,
            'pep_seq': self.pep_seq,
            'binder_prob': str(self.binder_prob),
            'binder': self.binder,
            'bind_img': (self.bind_img.tolist() if self.bind_img is not None else None)
        }

class PredictionModelWrapper(object):
    def __init__(self, path=None):
        self._model = load_model(path)
        print 'Kera model loaded.'

        self.alleles = self.read_alleles()
        print 'alleles:', self.alleles
        self.pep_len = 9
        # Use only 34 NetMHCPan contact sites
        self.bdomain = PanMHCIBindingDomain()
        self.bdomain.set_contact_sites(self.pep_len, self.bdomain._PanMHCIBindingDomain__netmhcpan_contact_sites_9)
        print 'PanMHCIBindingDomain loaded'

        self.aa_scorer = WenLiuAAPropScorer(corr_cutoff=0.85, data_transformer=MinMaxScaler())
        self.aa_scorer.load_score_tab()
        print('aa_scorer.n_scores: %s' % self.aa_scorer.n_scores())
        print('aa_scorer.feature_names: %s' % self.aa_scorer.feature_names())

    def read_alleles(self):
        with open('datasets/alleles.txt', 'r') as f:
            return eval(f.read())

    def predict(self, allele, pep_seqs, pep_len):

        X = self.transform_bind_images([(allele, seq) for seq in pep_seqs])
        print 'X.shape:', X.shape, 'X[0].shape:', X[0].shape

        y_pred = self._model.predict_proba(X, batch_size=16, verbose=0)
        y_pred_cls = np.argmax(y_pred, axis=1)
        y_pred_prob = y_pred[:, 1]
        print 'y_pred:', y_pred_prob, y_pred_cls

        results = []
        for i in range(X.shape[0]):
            results.append(PredictionResult(allele=allele,
                                            pep_seq=pep_seqs[i],
                                            binder_prob=round(y_pred_prob[i], 4),
                                            binder=y_pred_cls[i],
                                            bind_img=X[i]))

        return results


    def transform_bind_images(self, pep_seqs, p_margin=0, h_margin=0):
        ndata = len(pep_seqs)
        print('===>Start to transform. ndata: %s' % (ndata))
        imgs = []
        for i in range(ndata):
            allele = pep_seqs[i][0]
            pep_seq = pep_seqs[i][1]
            img = self.bdomain.binding_image(allele=allele,
                                        pep_seq=pep_seq,
                                        p_margin=p_margin,
                                        h_margin=h_margin,
                                        p_aa_scorer=self.aa_scorer,
                                        h_aa_scorer=self.aa_scorer,
                                        aai_scorer=None)

            print('Progress==>%s/%s, allele:%s, pep_seq:%s' % ((i + 1), ndata, allele, pep_seq))
            imgs.append(img)

        print('===>Done to transform.')
        return np.asarray(imgs)

    def find_informative_pixels(self, target_img, binder):
        print 'target_img.shape:', target_img.shape, 'binder:', binder
        dl_imgs = apply_deeplift(self._model, np.expand_dims(target_img, axis=0), class_index=binder)
        print 'dl_imgs.shape:', dl_imgs.shape
        return np.mean(dl_imgs[0], axis=0)


# data: the list of target data(eg. for MNIST images n images with 1 x 28 x 28)
# input_layer_index: the index of the layer to compute the importance scores. Default is 0, which is for input layer
# target_layer_index: the index of target output layer. For sigmoid or softmax outputs, target_layer_idx should be -2(the default)
#                     (See "a note on final activation layers" in https://arxiv.org/pdf/1605.01713v2.pdf for justification)
#                     For regression tasks with a linear output, target_layer_idx should be -1(which simply refers to the last layer)
# class_index: represents the index of the node in the output layer that we wish to compute scores.
# Eg: if the output is a 10-way softmax, and class_index is 0, we will compute scores for the first class index
def apply_deeplift(keras_model, data, input_layer_index=0, target_layer_index=-2, class_index=None):
    # Convert the Keras model
    # NonlinearMxtsMode defines the method for computing importance scores. Other supported values are:
    # Gradient, DeconvNet, GuidedBackprop and GuidedBackpropDeepLIFT (a hybrid of GuidedBackprop and DeepLIFT where
    # negative multipliers are ignored during backpropagation)
    deeplift_model = kc.convert_sequential_model(keras_model, num_dims=len(keras_model.input_shape),
                                                 nonlinear_mxts_mode=NonlinearMxtsMode.DeepLIFT)

    # get relevant functions
    deeplift_contribs_func = \
        deeplift_model.get_target_contribs_func(find_scores_layer_idx=input_layer_index)

    # input_data_list is a list of arrays for each mode
    # each array in the list are features of cases in the appropriate format
    input_data_list = [data]

    # helper function for running aforementioned functions
    def compute_contribs(func):
        return np.array(
            func(task_idx=class_index, input_data_list=input_data_list, batch_size=10, progress_update=None))

    # output is a list of arrays...
    # list index = index of output neuron (controlled by task_idx)
    # array has dimensions (k, 784), with k= # of samples, 784= # of features
    deeplift_contribs = compute_contribs(deeplift_contribs_func)

    return deeplift_contribs


global model
model = PredictionModelWrapper(path=model_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    global model
    return render_template('main.html', data={'alleles': model.alleles, 'pep_lens': [model.pep_len]})

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global model

    seq_txt = request.form.get('peptide_seqs', '', type=str)
    # print 'Found newline', seq_txt.find("\r\n")
    seq_txt = seq_txt.replace('\r\n', '\n')
    allele = request.form.get('allele', '', type=str)
    pep_len = request.form.get('peptide_len', '', type=int)
    print('allele:%s, seq_text:%s, pep_len:%s' % (allele, seq_txt, pep_len))
    try:
        seqs = Utils.split_seqs(seq_txt=seq_txt, seq_len=pep_len)
        print('sequences:%s' % seqs)

        pred_results = model.predict(allele=allele, pep_seqs=seqs, pep_len=pep_len)
        print 'Pred results:', pred_results
        results = {}
        results['pred_results'] = pred_results
        return jsonify(results=results)
    except Exception as e:
        print(traceback.format_exc())
        return e.message, 500
    # # return json.dumps({'status': 'OK', 'user': user, 'pass': password});

import time

@app.route('/generate_inf_img', methods=['GET', 'POST'])
def generate_informative_img():
    global model
    try:
        allele = request.form.get('target_allele', '')
        pep_seq = request.form.get('target_pepseq', '')
        binder = request.form.get('target_binder', 0, type=int)
        target_img_txt = request.form.get('target_img', '', type=str)
        target_img = json.loads(target_img_txt)
        print 'target_img:', target_img, 'allele', allele, 'pep_seq', pep_seq, 'binder:', binder

        infr_img = model.find_informative_pixels(np.asarray(target_img), binder=binder)
        # plot informative pixels
        p_sites = range(1, 10)
        h_sites = sorted(np.unique([css[1] for css in model.bdomain.contact_sites(9)]) + 1)

        sns.set_context('paper', font_scale=1.1)
        sns.axes_style('white')
        fig, axes = plt.subplots(nrows=1, ncols=1)
        fig.set_figwidth(6)
        fig.set_figheight(2)
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.22)

        g = sns.heatmap(infr_img, ax=axes, annot=False, linewidths=.4, cbar=False)
        # g.set(title='Informative pixels for %s-%s' % (pep_seq, allele))
        g.set_xticklabels(h_sites, rotation=90)
        g.set_yticklabels(p_sites[::-1])
        g.set(xlabel='HLA contact site', ylabel='Peptide position')

        canvas = FigureCanvas(fig)
        output = StringIO.StringIO()
        canvas.print_png(output)

        response = make_response(output.getvalue())
        response.mimetype = 'image/png'
        response.headers['Content-Type'] = 'image/png'
        return response

    except Exception as e:
        print(traceback.format_exc())
        return e.message, 500

@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)

def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path, endpoint, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)

if __name__ == '__main__':
    app.run()

import unittest

class PredictTestCase(unittest.TestCase):

    def setUp(self):
        with app.app_context() as ctx:
            ctx.push()
            g.model = load_model()
            self.client = app.test_client()

    def test_predict(self):
        data = {}
        data['peptide_seqs'] = 'AAAYYYRRR AAAYYYRRR'
        data['allele'] = 'HLA-A*03:01'
        data['peptide_len'] = 9
        response = self.client.post('/predict', data=data)
        print response


# if __name__ == '__main__':
#     unittest.main()