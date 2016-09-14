import numpy as np
import tensorflow as tf
from scipy.special import expit as sigmoid
import sys
sys.path.append('/home/kni/local-kni/_update_negsamp/kyle_update/')
from attalos.imgtxt_algorithms.regress2sum.attalos_model import AttalosModel
from attalos.dataset.transformers.onehot import OneHot
from attalos.imgtxt_algorithms.util.readw2v import initVo, readvocab

import attalos.util.log.log as l
logger = l.getLogger(__name__)

class NegSamplingModel(AttalosModel):
    """
    Create a tensorflow graph that does regression to a target using a negative sampling loss function
    """
    def __init__(self, w2v_model, train_dataset, test_dataset,
                 learning_rate=0.01,
                 hidden_units=[200,200],
                 optim_words=True,
                 use_batch_norm=True, **kwargs):
        
        self.model_info = dict()

        input_size = train_dataset.img_feat_size
        self.cross_eval = kwargs.get("cross_eval", False)
        self.one_hot = OneHot([train_dataset] if self.cross_eval else [train_dataset])                              
                               #valid_vocab=w2v_model.keys())
        w2v = initVo( w2v_model, self.one_hot.get_key_ordering() )
        self.w2v = w2v
        
         # Placeholders for data
        self.model_info['input'] = tf.placeholder(shape=(None, input_size), dtype=tf.float32)

        # Inputs to cost function
        self.learning_rate = learning_rate
        self.optim_words = optim_words
        if optim_words:
            self.model_info['pos_vecs'] = tf.placeholder(dtype=tf.float32)
            self.model_info['neg_vecs'] = tf.placeholder(dtype=tf.float32)
            logger.info("Optimization on GPU, word vectors stored separately")
        else:
            self.model_info['w2v'] = tf.Variable(w2v)
            w2vgraph=self.model_info['w2v'] 
            self.model_info['pos_ids'] = tf.placeholder(dtype=tf.int32)
            self.model_info['neg_ids'] = tf.placeholder(dtype=tf.int32)  
            self.model_info['pos_vecs'] = tf.transpose(tf.nn.embedding_lookup(w2vgraph,
                                                                              self.model_info['pos_ids']),
                                                                              perm=[1,0,2])
            self.model_info['neg_vecs'] = tf.transpose(tf.nn.embedding_lookup(w2vgraph,
                                                                              self.model_info['neg_ids']),
                                                                              perm=[1,0,2])

        # Construct fully connected layers
        layers = []
        layer = self.model_info['input']
        for i, hidden_size in enumerate(hidden_units[:-1]):
            layer = tf.contrib.layers.relu(layer, hidden_size)
            layers.append(layer)
            if use_batch_norm:
                layer = tf.contrib.layers.batch_norm(layer)
                layers.append(layer)

        # Output layer should always be linear
        layer = tf.contrib.layers.linear(layer, w2v.shape[1])
        layers.append(layer)

        self.model_info['layers'] = layers
        self.model_info['prediction'] = layer

        def meanlogsig(pred, truth):
            reduction_indicies = 2
            return tf.reduce_mean( tf.log( tf.sigmoid( tf.reduce_sum(pred*truth, reduction_indices=reduction_indicies))))
        
        pos_loss = meanlogsig(self.model_info['prediction'], self.model_info['pos_vecs'])
        neg_loss = meanlogsig(-self.model_info['prediction'], self.model_info['neg_vecs'])
        loss = -(pos_loss + neg_loss)
        self.model_info['loss'] = loss 

        logger.info("Learning rate: %s" % learning_rate)
        
        # Initialization operations: check to see which ones actually get initialized (make sure "W2V")
        self.model_info['optimizer'] = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        self.model_info['init_op'] = tf.initialize_all_variables()
        self.model_info['saver'] = tf.train.Saver()

    def initialize_model(self, sess):
        sess.run(self.model_info['init_op'])

    def predict(self, sess, x, y=None):
        return sess.run(self.model_info['prediction'], feed_dict={self.model_info['input']: x})
    
    def predict_words(self, sess, x, y=None):
        pred = sess.run(self.model_info['prediction'], feed_dict={self.model_info['input']: x})
        return pred.dot(self.w2v.T)

    def updatewords(self, vpindex, vnindex, vin):
        for i, (vpi, vni) in enumerate(zip(vpindex, vnindex)):
            self.w2v[vpi]+=self.learning_rate*np.outer(1 - sigmoid(self.w2v[vpi].dot(vin[i])),vin[i])
            self.w2v[vni]-=self.learning_rate*np.outer(sigmoid(self.w2v[vni].dot(vin[i])),vin[i])                                                            
    def fit(self, sess, x, y, **kwargs):

        # If you're not optimizing for the words
        if not self.optim_words:
            _, loss = sess.run([self.model_info['optimizer'], self.model_info['loss']],
                               feed_dict={ self.model_info['input']: x,
                                           self.model_info['pos_ids']: y[0],
                                           self.model_info['neg_ids']: y[1]
                                         })

        # If you are, then you'll need to get the vectors offline
        else:
            pos_ids, neg_ids = y
            
            pvecs = np.zeros((pos_ids.shape[0], pos_ids.shape[1], self.w2v.shape[1]))
            nvecs = np.zeros((neg_ids.shape[0], neg_ids.shape[1], self.w2v.shape[1]))
            for i, ids in enumerate(pos_ids):
                pvecs[i] = self.w2v[ids]
            for i, ids in enumerate(neg_ids):
                nvecs[i] = self.w2v[ids]
            pvecs = pvecs.transpose((1,0,2))
            nvecs = nvecs.transpose((1,0,2))
            _, loss, preds = sess.run([self.model_info['optimizer'], self.model_info['loss'], self.model_info['prediction']],
                                       feed_dict={ self.model_info['input']: x,
                                                   self.model_info['pos_vecs']: pvecs,
                                                   self.model_info['neg_vecs']: nvecs
                                                 })
            self.updatewords(pos_ids, neg_ids, preds)
            #logger.info("Loss: %s" % loss)
        return loss
    
    
    def get_batch(self, pBatch, numSamps=[5,10]):
        nBatch = 1.0 - pBatch
        vpia = []; vnia = [];
        for i,unisamp in enumerate(pBatch):
            vpi = np.random.choice( range(len(unisamp)) , size=numSamps[0],  p=1.0*unisamp/sum(unisamp))
            vpia += [vpi]
        for i,unisamp in enumerate(nBatch):
            vni = np.random.choice( range(len(unisamp)) , size=numSamps[1], p=1.0*unisamp/sum(unisamp))
            vnia += [vni]
        vpia = np.array(vpia)
        vnia = np.array(vnia)
        return vpia, vnia
    
    def to_batches(self, dataset, batch_size = 1024):
        num_batches = int(dataset.num_images / batch_size)
        for batch in xrange(num_batches):
            img_feats_list, text_feats_list = dataset.get_next_batch(batch_size)
            img_feats = np.array(img_feats_list)
            #img_feats = (img_feats.T / np.linalg.norm(img_feats, axis=1)).T
            
            new_text_feats = [self.one_hot.get_multiple(text_feats) for text_feats in text_feats_list]
            new_text_feats = np.array(new_text_feats)
            new_text_feats = (new_text_feats.T / np.linalg.norm(new_text_feats, axis=1)).T
            
            pos_ids, neg_ids = self.get_batch( new_text_feats )
            yield img_feats, (pos_ids, neg_ids)
   
    def to_ndarrs(self, dataset):
        x = []                                                                                                                  
        y = []                                                                                                                  
        for idx in dataset:                                                                                                      
            image_feats, text_feats = dataset.get_index(idx)                                                                    
            text_feats = self.one_hot.get_multiple(text_feats)                                                                  
            x.append(image_feats)                                                                                                
            y.append(text_feats)                                                                                                
        return np.asarray(x), np.asarray(y)  

    def save(self, sess, model_output_path):
        self.model_info['saver'].save(sess, model_output_path)

    def load(self, sess, model_input_path):
        self.model_info['saver'].restore(sess, model_input_path)
