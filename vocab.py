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
    save_tokenizer(q1, q2)

    # download embeddings
    if embeddings == "glove":
        get_glove_embeddings(embeddings)

    print("Finished reading full dataset")
