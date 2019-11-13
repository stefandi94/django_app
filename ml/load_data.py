import os
import pickle

from ml.settings import TRAIN_RAW_DIR, TEST_RAW_DIR, DATA_DIR


def join_raw_data(path):
    text = []
    labels = []
    
    for folder in os.listdir(path):
        file_path = os.path.join(path, folder)
        if os.path.isdir(file_path):
            for file in os.listdir(file_path):
                with open(os.path.join(path, folder, file)) as fp:
                    text.append(fp.readlines()[0])
                labels.append(folder)
    return text, labels


def save_text_data(path, file, filename):
    with open(os.path.join(path, filename), 'w') as fp:
        for line in file:
            fp.write(line + '\n')


def load_file(path):
    with open(path, 'r') as fp:
        file = fp.readlines()
    return [line.strip() for line in file]


def pickle_save(file, filepath, filename):
    with open(os.path.join(filepath, f'{filename}.pckl'), 'wb') as fp:
        pickle.dump(file, fp, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(filepath, filename):
    with open(os.path.join(filepath, f'{filename}.pckl'), 'rb') as handle:
        data = pickle.load(handle)
    return data


def load_data():
    X_train = load_file(os.path.join(DATA_DIR, 'train_text'))
    y_train = load_file(os.path.join(DATA_DIR, 'train_labels'))
    
    X_test = load_file(os.path.join(DATA_DIR, 'test_text'))
    y_test = load_file(os.path.join(DATA_DIR, 'test_labels'))
    return X_train, y_train, X_test, y_test


def clean_unsuper_text(text, labels):
    new_text = []
    new_labels = []
    for index, label in enumerate(labels):
        if label in ['pos', 'neg']:
            new_text.append(text[index])
            new_labels.append(label)
    return new_text, new_labels


if __name__ == '__main__':
    train_text, train_labels = join_raw_data(TRAIN_RAW_DIR)
    test_text, test_labels = join_raw_data(TEST_RAW_DIR)

    train_text, train_labels = clean_unsuper_text(train_text, train_labels)
    test_text, test_labels = clean_unsuper_text(test_text, test_labels)
    
    save_text_data(DATA_DIR, train_text, 'train_text')
    save_text_data(DATA_DIR, train_labels, 'train_labels')
    save_text_data(DATA_DIR, test_text, 'test_text')
    save_text_data(DATA_DIR, test_labels, 'test_labels')
    
    print()