import numpy as np
import tqdm
import nltk
from src.features.config import NODEF_TOKEN, NULL_TOKEN, NODEF_TOKEN_VEC_VALUE, NULL_TOKEN_VEC_VALUE, SEQ_LENGTH, W2V_SIZE

# TODO Fastext embeding
# TODO Glove embeding


def tokenize_sentences(sentence):
    '''
    Tokenize a sentence using NLTK tokenizer. also decode sentence to UTF-8 if possible/necessary
    :param sentence:
    :return:
    '''
    if hasattr(sentence, "decode"):
        sentence = sentence.decode("utf-8")

    tokens = nltk.tokenize.word_tokenize(sentence)

    return tokens


def get_embeding(token, word_embeding):
    '''
    return the embeding of a word or token.

    :param token: srtring of the word to embed
    :param word_embeding: a gensim Word2Vec object
    :return: Numpy array of the embeding
    '''
    try:
        embeding = word_embeding.wv[token]

    except KeyError:

        # If not part of the Word2Vec
        if token == NULL_TOKEN:
            embeding = np.full((W2V_SIZE,), NULL_TOKEN_VEC_VALUE,  dtype=np.float32)
        elif token == NODEF_TOKEN:
            embeding = np.full((W2V_SIZE,), NODEF_TOKEN_VEC_VALUE, dtype=np.float32)
        else:
            embeding = np.full((W2V_SIZE,), NODEF_TOKEN_VEC_VALUE, dtype=np.float32)

    return embeding


def embed_sentence(sentence, word_embedings):
    '''
    Return the W2V embeding of a sentence. Also pad the sentence
    :param sentence: String
    :param word_embedings: gensim object
    :return: Numpy array nb_token x SEQUENCE_LENGTH
    '''
    tokens = tokenize_sentences(sentence)
    w2v_sentence = list()

    for token in tokens:
        w2v_sentence.append(get_embeding(token, word_embedings))

    if len(w2v_sentence) < SEQ_LENGTH:
        padding_seq = [np.full((W2V_SIZE, ), NULL_TOKEN_VEC_VALUE)] * (SEQ_LENGTH - len(w2v_sentence))

        w2v_sentence = padding_seq + w2v_sentence
    else:
        w2v_sentence = w2v_sentence[:SEQ_LENGTH]

    w2v_sentence = np.asarray(w2v_sentence)

    return w2v_sentence


def create_embeding_space(corpus, w2v, no_def_init="zero"):
    '''
    :param corpus: A pandas list of string
    :param w2v: A gensim object representing a Word2Vec
    :param no_def_init: One of zero, normal_random
    :return: embeding_indexes a dictionnary with key being word and value being an index with relate to embeding_space.
    embeding_space is a list which contain the word representation of the word that as this index as index.
    '''

    if no_def_init == "zero":
        mean = w2v.syn0.flatten().mean()
        std = w2v.syn0.flatten().std()

    embeding_indexes = dict()
    embeding_indexes[NODEF_TOKEN] = 0

    embeding_space = list()
    embeding_space.append(np.full((W2V_SIZE, ), NODEF_TOKEN_VEC_VALUE))

    index = 1
    tokens = set()

    corpus.apply(tokens.update)

    for token in tqdm.tqdm(tokens):
        if token in w2v.vocab:
            embeding_indexes[token] = index
            embeding_space.append(w2v.wv[token])
            index += 1
        elif no_def_init == "normal_random":
            generated_embeding = np.random.normal(loc=mean, scale=std, size=w2v.vector_size)
            embeding_indexes[token] = index
            embeding_space.append(generated_embeding)
            index += 1

    return embeding_indexes, embeding_space


def index_from_token(tokens, indexes, pad=True):
    if not pad:
        raise NotImplementedError

    token_indexes = list()

    for token in tokens:
        try:
            token_indexes.append(indexes[token])
        except:
            token_indexes.append(0)

    if len(token_indexes) < SEQ_LENGTH:
        padding_seq = [0] * (SEQ_LENGTH - len(token_indexes))  # TODO change to function instead

        token_indexes = padding_seq + token_indexes
    else:
        token_indexes = token_indexes[:SEQ_LENGTH]

    token_indexes = np.asarray(token_indexes, dtype=np.int32)

    return token_indexes

