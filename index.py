import tensorflow as tf
import re
import unicodedata
import numpy as np
import os
import io
import time
import json as json
import logging
import sys

#logger part
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(message)s:')
file_handler = logging.FileHandler('EL_LANG.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state = hidden)
        return output, state
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))

# Attention layer
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
    def call(self, query, values):
        hidden_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

# Decoder class
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.dec_units)
    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state, attention_weights
############################################################################

# Data preprocessing step
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w = w.rstrip().strip()
    w = '<start> ' + w + ' <end>'
    return w

# word pair creation
def create_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:num_examples]]
    return zip(*word_pairs)

def max_length(tensor):
    return max(len(t) for t in tensor)

# Tokenizing the words
def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
      filters='')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                         padding='post')
    return tensor, lang_tokenizer

# Loading the dataset
def load_dataset(path, num_examples=None):
    targ_lang, inp_lang = create_dataset(path, num_examples)
    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)
    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


def convert(lang, tensor):
    for t in tensor:
        if t!=0:
            print ("%d ----> %s" % (t, lang.index_word[t]))

# Evaluate function
def evaluate(sentence,inp_lang,targ_lang,max_length_targ,
            max_length_inp,vocab_inp_size,vocab_tar_size,
            encoder,decoder,checkpoint):
    try:
        attention_plot = np.zeros((max_length_targ, max_length_inp))
        sentence = preprocess_sentence(sentence)
        inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                            maxlen=max_length_inp,
                                                            padding='post')
        inputs = tf.convert_to_tensor(inputs)
        result = ''
        hidden = [tf.zeros((1, units))]
        enc_out, enc_hidden = encoder(inputs, hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)
        for t in range(max_length_targ):
            predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                            dec_hidden,
                                                            enc_out)
            attention_weights = tf.reshape(attention_weights, (-1, ))
            predicted_id = tf.argmax(predictions[0]).numpy()
            result += targ_lang.index_word[predicted_id] + ' '
            if targ_lang.index_word[predicted_id] == '<end>':
                return result.replace('<end>',''), sentence
            dec_input = tf.expand_dims([predicted_id], 0)
        logger.info("<p>Successfully translated: %s</p>"% result)
        return result.replace('<end>',''), sentence
    except:
        e = sys.exc_info()[0]
        logger.info( "<p>Error: %s</p>" % e )

# Translate function
def translateLang(sentence,inp_lang,targ_lang,
                max_length_targ,max_length_inp,
                vocab_inp_size,vocab_tar_size,
                encoder,decoder,checkpoint):
    result, sentence = evaluate(sentence,inp_lang,targ_lang,
                        max_length_targ,max_length_inp,
                        vocab_inp_size,vocab_tar_size,
                        encoder,decoder,checkpoint)
    print('Input: %s' % (sentence))
    print('Predicted translation: {}'.format(result))
    return result

#####################################################################
from flask import Flask,request
app = Flask(__name__)

@app.route('/translate', methods=["POST"])
def translate():
    
    sent =request.args.get('sentence')
    if request.method == 'POST':
        sent = request.form['sentence']
        from_lang = request.form['from_lang']
        to_lang = request.form['to_lang']
        
    if from_lang == 'sp' and to_lang == 'en':
        checkpoint_dir = "./checkpoint_folder_sp_en/"
        input_tensor, target_tensor,inp_lang, targ_lang = load_dataset(path_to_file, num_examples)
    elif from_lang  == 'en'and to_lang == 'sp':
        checkpoint_dir = "./checkpoint_folder_en-spa/"
        target_tensor,input_tensor,targ_lang, inp_lang = load_dataset(path_to_file, num_examples)
    else:
        print("\nEnter a valid language \n 1.spanish(sp) \n 2.english(en)\n\n")
       
    max_length_targ, max_length_inp = max_length(target_tensor), max_length(input_tensor)
    vocab_inp_size = len(inp_lang.word_index)+1
    vocab_tar_size = len(targ_lang.word_index)+1
    encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
    attention_layer = BahdanauAttention(10)
    decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)
    optimizer = tf.keras.optimizers.Adam()
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

    translated_text = translateLang(sent,inp_lang,targ_lang,
        max_length_targ,
        max_length_inp,
        vocab_inp_size,
        vocab_tar_size,
        encoder,
        decoder,
        checkpoint)
    event['Input Sentence'] = sent
    event["Output"] = translated_text
    return (json.dumps(event, indent=4, sort_keys=True) )

##########################################################################
event = {}
BATCH_SIZE = 64
embedding_dim = 256
units = 1024
num_examples = 30000

path_to_zip = tf.keras.utils.get_file(
    'spa-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
    extract=True)

path_to_file = os.path.dirname(path_to_zip)+"/spa-eng/spa.txt"

########################################################################
if __name__ == '__main__':
    app.run()
#########################################################################