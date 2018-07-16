import pickle, csv
from keras.preprocessing.text import Tokenizer
from zipfile import ZipFile
from os.path import expanduser, exists

from keras.utils import get_file

MAX_NB_WORDS = 80000


# Build tokenized word index for full dataset. Save tokenizer for reuse
def read_full_dataset(file):
    # if not exists(KERAS_DATASETS_DIR + QUESTION_PAIRS_FILE):
    #     get_file(QUESTION_PAIRS_FILE, QUESTION_PAIRS_FILE_URL)

    print("Processing quora question pairs file")
    question1 = []
    question2 = []
    is_duplicate = []
    with open(file, encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            question1.append(row[1])
            question2.append(row[2])
            is_duplicate.append(row[0])
    print('Question pairs from %s: %d' % (file, len(question1)))

    return question1, question2, is_duplicate


def save_tokenizer(question1, question2):
    questions = question1 + question2
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(questions)
    word_index = tokenizer.word_index
    print("Words in index: %d" % len(word_index))

    # save tokenizer
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return word_index


# Download and process GloVe embeddings
def get_glove_embeddings(embeddings):
    keras_dataset_dir = expanduser('~/.keras/datasets/')
    glove_zip_file_url = 'http://nlp.stanford.edu/data/glove.840B.300d.zip'
    glove_zip_file = 'glove.840B.300d.zip'
    glove_file = 'glove.840B.300d.txt'
    if not exists(keras_dataset_dir + glove_zip_file):
        zf = ZipFile(get_file(glove_zip_file, glove_zip_file_url))
        zf.extract(glove_file, path=keras_dataset_dir)

    print("Processing", glove_file)


def prepare_vocab(file, embeddings):
    q1, q2, is_duplicate = read_full_dataset(file)
    word_index = save_tokenizer(q1, q2)

    # download embeddings
    if embeddings == "glove":
        get_glove_embeddings(embeddings)

    print("Finished reading full dataset")

    return word_index


# def get_misclassified_q(misclassified_q, word2idx):
#     idx2word = []
#     for i, pair in enumerate(misclassified_q):
#         q_pair = []
#         for idx,q in enumerate(pair):
#             q_words = []
#             if idx != 2 and idx != 3:
#                 for w in q:
#                     word = [key for key, value in word2idx.items() if value == w][0]
#                     q_words.append(word)
#                 q_pair.append(q_words)
#             else :
#                 q_pair.append(q)
#         idx2word.append(q_pair)
#     return idx2word
#
#
# def write_misclassified_q(misclassified_q, output_file):
#     if output_file:
#         with open(output_file, 'w+') as f:
#             f.write('New Epoch---------------------------\n')
#             for pair in misclassified_q:
#                 f.writelines(str(pair[0])+'\t'+ str(pair[1]) +'\t'+ str(pair[2]) +'\t'+ str(pair[3])+'\n')
#
#
# misclassified_q = []
#     i = 0
#     for l_true, l_pred in zip(dataset.get_data_item('y'), all_plabels):
#         if l_true != l_pred:
#             qq = [dataset.get_data_item('q1')[i],dataset.get_data_item('q2')[i], l_true, l_pred]
#             misclassified_q.append(qq)
#             i +=1