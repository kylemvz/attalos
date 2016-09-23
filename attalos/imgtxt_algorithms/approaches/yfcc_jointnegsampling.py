import numpy as np
import tensorflow as tf
from scipy.special import expit as sigmoid
from scipy.sparse import lil_matrix

from itertools import izip

from attalos.imgtxt_algorithms.approaches.base import AttalosModel
from attalos.util.transformers.onehot import OneHot
from attalos.imgtxt_algorithms.correlation.correlation import construct_W, scale3
from attalos.imgtxt_algorithms.util.negsamp import NegativeSampler

from collections import Counter

import attalos.util.log.log as l
logger = l.getLogger(__name__)

# This is with jointly optimizing words
class YFCCJointNegSamplingModel(AttalosModel):
    def _construct_model_info(self, input_size, output_size, learning_rate, wv_arr,
                              hidden_units=[200,200],
                              use_batch_norm=True):
        model_info = {}
        model_info["input"] = tf.placeholder(shape=(None, input_size), dtype=tf.float32)

        model_info["pos_vecs"] = tf.placeholder(dtype=tf.float32)
        model_info["neg_vecs"] = tf.placeholder(dtype=tf.float32)
        logger.info("Optimization on GPU, word vectors are stored separately.")

        # Construct fully connected layers
        layers = []
        layer = model_info["input"]
        for i, hidden_size in enumerate(hidden_units[:-1]):
            layer = tf.contrib.layers.relu(layer, hidden_size)
            layers.append(layer)
            if use_batch_norm:
                layer = tf.contrib.layers.batch_norm(layer)
                layers.append(layer)

        # Output layer should always be linear
        layer = tf.contrib.layers.linear(layer, wv_arr.shape[1])
        layers.append(layer)

        model_info["layers"] = layers
        model_info["prediction"] = layer

        def meanlogsig(predictions, truth):
            reduction_indices = 2
            return tf.reduce_mean(tf.log(tf.sigmoid(tf.reduce_sum(predictions * truth, reduction_indices=reduction_indices))))

        pos_loss = meanlogsig(model_info["prediction"], model_info["pos_vecs"])
        neg_loss = meanlogsig(-model_info["prediction"], model_info["neg_vecs"])
        model_info["loss"] = -(pos_loss + neg_loss)

        model_info["optimizer"] = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(model_info["loss"])
        #model_info["init_op"] = tf.initialize_all_variables()
        #model_info["saver"] = tf.train.Saver()

        return model_info
    
    def _run_counter(self, datasets):
        word_counter = Counter()
        valid_vocab = set(self.wv_model.vocab)
        for dataset in datasets:
            for filename, tags in dataset.text_feats.iteritems():
                # Lower case, split by "+", and take the last word
                tags = [tag.split("+")[-1].lower() for tag in tags]
                for tag in tags:
                    if tag in valid_vocab:
                        word_counter[tag] += 1
        tag_ordering = [word for word, count in word_counter.most_common()]
        tag_to_index = {tag: i for i, tag in enumerate(tag_ordering)}
        return word_counter, tag_ordering, tag_to_index
       
    def __init__(self, wv_model, datasets, **kwargs):
        self.wv_model = wv_model
        word_counter, tag_ordering, tag_to_index = self._run_counter(datasets)
        self.word_counter = word_counter
        self.tag_ordering = tag_ordering
        self.tag_to_index = tag_to_index      
        self.negsampler = NegativeSampler(np.array([count for word, count in self.word_counter.most_common()]))
        self.w = construct_W(wv_model, self.tag_ordering).T
        self.w = (self.w.T / np.linalg.norm(self.w, axis=1)).T
        
        train_dataset = datasets[0] # train_dataset should always be first in datasets
        self.learning_rate = kwargs.get("learning_rate", 0.0001)
        self.hidden_units = kwargs.get("hidden_units", "200,200")
        self.hidden_units = [int(x) for x in self.hidden_units.split(",")]
        logger.info("Constructing model info.")
        self.model_info = self._construct_model_info(
            input_size=train_dataset.img_feat_size,
            output_size=len(self.tag_ordering),
            hidden_units=self.hidden_units,
            learning_rate=self.learning_rate,
            wv_arr=self.w
        )
        
    def _get_sampled_vecs(self, text_feats_idxs_list, numSamps=(5, 10)):
        # Using more descriptive names
        num_items_in_batch = len(text_feats_idxs_list)
        num_pos_samples = numSamps[0] 
        num_neg_samples = numSamps[1]
        wv_length = self.w.shape[1]
        vocab_size = self.w.shape[0]
        
        pos_idarr = np.zeros((num_items_in_batch, num_pos_samples), dtype=np.int32)
        neg_idarr = np.zeros((num_items_in_batch, num_neg_samples), dtype=np.int32)
        pos_vecs = np.zeros((num_items_in_batch, num_pos_samples, wv_length))
        neg_vecs = np.zeros((num_items_in_batch, num_neg_samples, wv_length))
        
        for row_idx, text_feats_idxs in enumerate(text_feats_idxs_list):
            if text_feats_idxs: # if empty, do nothing
                # Uniformly sample positive tags
                pos_ids = np.random.choice(text_feats_idxs, size=num_pos_samples)
                #logger.info("Positive id choice(s): %s" % pos_ids)
                pos_idarr[row_idx] = pos_ids
                for idx, pos_id in enumerate(pos_ids):
                    wv = self.w[pos_id]
                    #logger.info("WV: %s" % str(wv))
                    pos_vecs[row_idx, idx] = wv
            # Sample negative tags according to the word counts
            # TODO: negsampler.negsamp_ind calls np.copy, creating a copy of the count pdf... may be able to optimize?
            #p = np.ones(vocab_size)
            #p[text_feats_idxs] = 0
            #p = p / p.sum()
            #neg_ids = np.random.choice(vocab_size, num_neg_samples, p=p)
            neg_ids = self.negsampler.negsamp_ind(text_feats_idxs, num_neg_samples)
            #logger.info("Negative id choice(s): %s" % neg_ids)
            neg_idarr[row_idx] = neg_ids
            for idx, neg_id in enumerate(neg_ids):
                wv = self.w[neg_id]
                #logger.info("WV: %s" % str(wv))
                neg_vecs[row_idx, idx] = wv
        self.pos_ids = pos_idarr
        self.neg_ids = neg_idarr
        pos_vecs = pos_vecs.transpose((1, 0, 2))
        neg_vecs = neg_vecs.transpose((1, 0, 2))
        self.pos_vecs = pos_vecs
        self.neg_vecs = neg_vecs
        return pos_vecs, neg_vecs
        
    def prep_fit(self, data):
        image_feats, text_feats = data
        image_feats = np.array(image_feats)  # TODO: is this necessary? is image feats already a numpy array?
        image_feats = scale3(image_feats)
        
        text_feats_idxs_list = [[self.tag_to_index[tag] for tag in tags if tag in self.tag_to_index] for tags in text_feats]
        pos_vecs, neg_vecs = self._get_sampled_vecs(text_feats_idxs_list)
        fetches = [self.model_info["optimizer"], self.model_info["loss"], self.model_info["prediction"]]
        feed_dict = {
            self.model_info["input"] : image_feats,
            self.model_info["pos_vecs"] : pos_vecs,
            self.model_info["neg_vecs"] : neg_vecs
        }
        return fetches, feed_dict 

    def _updatewords(self, vpindex, vnindex, vin):
        for i, (vpi, vni) in enumerate(zip(vpindex, vnindex)):
            self.w[vpi] += self.learning_rate * np.outer(1 - sigmoid(self.w[vpi].dot(vin[i])), vin[i])
            self.w[vni] -= self.learning_rate * np.outer(sigmoid(self.w[vni].dot(vin[i])), vin[i])

    def fit(self, sess, fetches, feed_dict):
        fit_fetches = super(YFCCJointNegSamplingModel, self).fit(sess, fetches, feed_dict)
        if self.pos_ids is None or self.neg_ids is None:
            raise Exception("pos_ids or neg_ids is not set; cannot update word vectors. Did you run prep_fit()?")
        _, _, prediction = fit_fetches
        self._updatewords(self.pos_ids, self.neg_ids, prediction)
        return fit_fetches
    
    # Using numpy slices instead of constructing a new numpy array with get_next_batch
    def iter_batches(self, dataset, batch_size):
        num_batches = int(dataset.num_images / batch_size)
        for count in xrange(num_batches):
            selected_idxs = np.random.randint(low=dataset.num_images, size=batch_size)
            sorted_selected_idxs = sorted(selected_idxs.tolist())
            selected_filenames = dataset.image_ids[sorted_selected_idxs]
            batch_image_feats = dataset.image_feats[sorted_selected_idxs]
            batch_text_feats = [dataset.text_feats[selected_filename] for selected_filename in selected_filenames]
            #batch_text_feats = dataset.text_feats[selected_filenames]
            fetches, feed_dict = self.prep_fit((batch_image_feats, batch_text_feats))
            yield fetches, feed_dict
    
    def get_training_loss(self, fit_fetches):
        return fit_fetches[1]
    
    def prep_predict(self, dataset, cross_eval=False):
        if cross_eval:
            _, self.test_tag_ordering, self.test_tag_to_index = self._run_counter([dataset])
            self.test_w = construct_W(self.wv_model, self.test_tag_ordering).T
        else:
            self.test_tag_ordering = self.tag_ordering
            self.test_tag_to_index = self.tag_to_index
            self.test_w = self.w
            
        if type(dataset.image_feats) != np.ndarray:
            logger.info("Allocating space for image feature matrix.")
            x = np.array(dataset.image_feats)
        else:
            x = dataset.image_feats
        x = scale3(x)
            
        logger.info("Allocating space for text feature matrix.")
        y = lil_matrix((dataset.num_images, len(self.test_tag_ordering)))
        logger.info("Done allocating.")

        count = 0
        logger.info("Starting prep_predict loop.")
        valid_vocab = set(self.wv_model.vocab)
        
        for idx, filename in enumerate(dataset.image_ids):
            if count % 500000 == 0:
                logger.info("%s of %s completed (prep_predict)." % (count, dataset.num_images))
            count += 1
            
            tags = dataset.text_feats[filename]
            for tag in tags:
                if tag in valid_vocab:
                    y_idx = self.test_tag_to_index[tag]
                    y[idx, y_idx] = 1

        fetches = [self.model_info["prediction"], ]
        feed_dict = {
            self.model_info["input"]: x
        }
        truth = y.tocsr()
        return fetches, feed_dict, truth
    
    def post_predict(self, predict_fetches, cross_eval=False):
        predictions = predict_fetches[0]
        if cross_eval and self.test_w is None:
            raise Exception("test_w is not set. Did you call prep_predict?")
        predictions = np.dot(predictions, self.test_w.T)
        return predictions