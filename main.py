import json
import os
import pickle
import cv2
import numpy as np
import time
from colorama import Fore
from pathlib import Path
from typing import Tuple
from sklearn import cluster
from sklearn import svm
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score
from tqdm import tqdm
project_Name='ImageClfProject'

def load_dataset(dataset_dir_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for i, class_dir in enumerate(dataset_dir_path.iterdir()):
        for file in class_dir.iterdir():
            img_file = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
            img_file = cv2.resize(img_file, (720, 540))
            x.append(img_file)
            y.append(i)
    return np.asarray(x), np.asarray(y)


def convert_descriptor_to_histogram(descriptors, vocab_model, normalize=True) -> np.ndarray:
    features_words = vocab_model.predict(descriptors)
    histogram = np.zeros(vocab_model.n_clusters, dtype=np.float32)
    unique, counts = np.unique(features_words, return_counts=True)
    histogram[unique] += counts
    if normalize:
        histogram /= histogram.sum()
    return histogram


def apply_feature_transform(
        data: np.ndarray,
        feature_detector_descriptor,
        vocab_model
) -> np.ndarray:
    data_transformed = []
    for image in tqdm(data, desc="Applying transform: ",
                      bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.LIGHTGREEN_EX, Fore.RESET)):
        keypoints, image_descriptors = feature_detector_descriptor.detectAndCompute(image, None)
        bow_features_histogram = convert_descriptor_to_histogram(image_descriptors, vocab_model)
        data_transformed.append(bow_features_histogram)
    return np.asarray(data_transformed)


def data_processing(x: np.ndarray) -> np.ndarray:
    # TODO: add data processing here if needed

    return x


def project(train, NB_WORDS, batch_size_K, random_state=42):
    np.random.seed(42)
    first_name = 'Jakub'
    last_name = 'Bielawski'

    data_path = Path(f'../{project_Name}/train/')  # You can change the path here
    data_path = os.getenv('DATA_PATH', data_path)  # Don't change that line
    x, y = load_dataset(data_path)
    x = data_processing(x)
    X_train, X_valid, y_train, y_valid = train_test_split(x, y, random_state=random_state, test_size=0.33, stratify=y)

    feature_detector_descriptor = cv2.AKAZE_create()
    if train:
        train_features = []
        for image in tqdm(X_train, desc="Description progress: ",
                          bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET)):
            key_points, image_decriptors = feature_detector_descriptor.detectAndCompute(image, None)
            for feature in image_decriptors:
                train_features.append(feature)

        print(f"Training vocab on {len(train_features)} features")
        vocab_model = cluster.MiniBatchKMeans(n_clusters=NB_WORDS, random_state=random_state, batch_size=batch_size_K)

        vocab_model.fit(train_features)
        pickle.dump(vocab_model, open(f"vocab_model_{NB_WORDS}_{batch_size_K}.p", "wb"))

        with Path(f'vocab_model_{NB_WORDS}_{batch_size_K}.p').open('rb') as vocab_file:  # Don't change the path here
            vocab_model = pickle.load(vocab_file)
        print("Saved vocab")

        X_train_transformed = apply_feature_transform(X_train, feature_detector_descriptor, vocab_model)

        # TODO: train a classifier and save it using pickle.dump function
        print("Training clf")

        clf = svm.SVC(C=8, kernel='poly')
        clf.fit(X_train_transformed, y_train)
        pickle.dump(clf, open("clf.p", "wb"))
        print("Saved clf")
        X_to_clasify = X_train_transformed
        y_to_clasify = y_train
    else:
        with Path(f'vocab_model_{NB_WORDS}_{batch_size_K}.p').open('rb') as vocab_file:  # Don't change the path here
            vocab_model = pickle.load(vocab_file)
        X_valid_transformed = apply_feature_transform(X_valid, feature_detector_descriptor, vocab_model)
        X_to_clasify = X_valid_transformed
        y_to_clasify = y_valid
    with Path('clf.p').open('rb') as classifier_file:  # Don't change the path here
        clf = pickle.load(classifier_file)

    score = clf.score(X_to_clasify, y_to_clasify)
    print(f"Loaded clf: {clf}")
    print(f"NB_WORDS: {NB_WORDS}")
    print(f"Batch sie: {batch_size_K}")
    print(f'_______________train___{train}_______________________{first_name} {last_name} score: {score}')
    with Path(f'{last_name}_{first_name}_score.json').open('w') as score_file:  # Don't change the path here
        json.dump({'score': score}, score_file)


def test(NB_WORDS, batch_size_K, random_state=42):
    first_name = 'Jakub'
    last_name = 'Bielawski'

    with Path(f'vocab_model_{NB_WORDS}_{batch_size_K}.p').open('rb') as vocab_file:  # Don't change the path here
        vocab_model = pickle.load(vocab_file)
    with Path('clf.p').open('rb') as classifier_file:  # Don't change the path here
        clf = pickle.load(classifier_file)
    data_path = Path(f'../{project_Name}/test/')  # You can change the path here
    data_path = os.getenv('DATA_PATH', data_path)  # Don't change that line
    X_test, y_test = load_dataset(data_path)

    X_test_transformed = apply_feature_transform(X_test, cv2.AKAZE_create(), vocab_model)
    X_to_test = X_test_transformed
    y_to_test = y_test
    score = clf.score(X_to_test, y_to_test)

    print(f"Loaded clf: {clf}")
    print(f"NB_WORDS: {NB_WORDS}")
    print(f"Batch sie: {batch_size_K}")
    print(f'_______________test__________________________{first_name} {last_name} score: {score}')


if __name__ == '__main__':
    # TODO: git repo

    project(train=True, NB_WORDS=250, batch_size_K=500)
    project(train=False, NB_WORDS=250, batch_size_K=500)
    test(250, 500)
