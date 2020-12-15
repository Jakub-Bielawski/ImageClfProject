import json
import os
import pickle
import cv2
import numpy as np
import pandas as pd
from colorama import Fore
from pathlib import Path
from typing import Tuple
from sklearn import cluster
from sklearn import svm
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from typing import List
import matplotlib.pyplot as plt


project_Name = 'ImageClfProject'


def load_dataset(dataset_dir_path: Path) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    x, y = [], []
    for i, class_dir in enumerate(sorted(dataset_dir_path.iterdir())):
        for file in class_dir.iterdir():
            img_file = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
            # img_file = cv2.resize(img_file, (720, 540))
            x.append(img_file)
            y.append(i)
    return x, y


def convert_descriptor_to_histogram(descriptors, vocab_model, normalize=True) -> np.ndarray:
    descriptors = descriptors.astype(np.float)
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

    return np.asarray(data_transformed,dtype='object')


def data_processing(x: np.ndarray) -> np.ndarray:
    # TODO: add data processing here if needed
    return x

def read_and_describe_train_pictures():
    data_path = Path(f'../{project_Name}/train/')  # You can change the path here
    data_path = os.getenv('DATA_PATH', data_path)  # Don't change that line
    x, y = load_dataset(data_path)
    X_train, X_valid, y_train, y_valid = train_test_split(x, y, random_state=42, test_size=0.33, stratify=y)

    train_features = []
    feature_detector_descriptor = cv2.SIFT_create()

    for image in tqdm(X_train, desc="Description progress: ",
                      bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET)):
        key_points, image_decriptors = feature_detector_descriptor.detectAndCompute(image, None)
        for feature in image_decriptors:
            train_features.append(feature)
    return train_features
def find_parameters(train_features,NB_WORDS=250, batch_size=500):
    np.random.seed(42)
    first_name = 'Jakub'
    last_name = 'Bielawski'
    # feature_detector_descriptor = cv2.AKAZE_create()
    feature_detector_descriptor = cv2.SIFT_create()

    # clf = svm.SVC(C=8, kernel='poly')
    # clf = svm.SVC()
    clf = svm.LinearSVC()
    # clf = Pipeline([('scaler', MinMaxScaler()), ('svc', clf)])
    clf = Pipeline([('scaler', MinMaxScaler()), ('linSVC', clf)])

    data_path = Path(f'../{project_Name}/train/')  # You can change the path here
    data_path = os.getenv('DATA_PATH', data_path)  # Don't change that line
    x, y = load_dataset(data_path)
    X_train, X_valid, y_train, y_valid = train_test_split(x, y, random_state=42, test_size=0.33, stratify=y)
    #
    # train_features = []
    # for image in tqdm(X_train, desc="Description progress: ",
    #                   bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET)):
    #     key_points, image_decriptors = feature_detector_descriptor.detectAndCompute(image, None)
    #     for feature in image_decriptors:
    #         train_features.append(feature)

    print(f"Training vocab on {len(train_features)} features")
    vocab_model = cluster.MiniBatchKMeans(n_clusters=NB_WORDS, random_state=42, batch_size=batch_size)
    # vocab_model = cluster.MiniBatchKMeans(n_clusters=100, random_state=42, batch_size=1000)

    vocab_model.fit(train_features)
    pickle.dump(vocab_model, open(f"vocab_model.p", "wb"))

    with Path(f'vocab_model.p').open('rb') as vocab_file:  # Don't change the path here
        vocab_model = pickle.load(vocab_file)
    print("Saved vocab")

    X_train_transformed = apply_feature_transform(X_train, feature_detector_descriptor, vocab_model)

    # TODO: train a classifier and save it using pickle.dump function
    print("Training clf")
    param_grid_SVC = {'svc__C': [0.1, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                      'svc__gamma': ['scale', 'auto', 1, 0.1, 0.001],
                      'svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid']
                      # 'max_iter': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]}
                      }
    param_grid_LinearSVC = {'linSVC__C': [0.1, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                            'linSVC__penalty': ['l1', 'l2'],
                            'linSVC__loss': ['hinge', 'squared_hinge'],

                            # 'linSVC__max_iter': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]}
                            }
    clf_random = RandomizedSearchCV(clf, param_grid_LinearSVC, refit=True, verbose=0)

    clf_random.fit(X_train_transformed, y_train)
    best_score = clf_random.best_score_
    print(f"{clf} score : {best_score}")
    clf_best = clf_random.best_estimator_
    pickle.dump(clf_best, open("clf.p", "wb"))
    return best_score, clf_best

def project(train,train_features, NB_WORDS=1000, batch_size_K=2048, random_state=42):
    np.random.seed(42)
    first_name = 'Jakub'
    last_name = 'Bielawski'

    feature_detector_descriptor = cv2.SIFT_create()
    clf = svm.LinearSVC(C=7)

    clf = Pipeline([('scaler', MinMaxScaler()), ('svc', clf)])
    data_path = Path(f'../{project_Name}/train/')  # You can change the path here
    data_path = os.getenv('DATA_PATH', data_path)  # Don't change that line
    x, y = load_dataset(data_path)
    x = data_processing(x)
    X_train, X_valid, y_train, y_valid = train_test_split(x, y, random_state=random_state, test_size=0.33, stratify=y)

    if train:
        print(f"Training vocab on {len(train_features)} features")
        vocab_model = cluster.MiniBatchKMeans(n_clusters=NB_WORDS, random_state=random_state, batch_size=batch_size_K)
        vocab_model.fit(train_features)
        pickle.dump(vocab_model, open(f"vocab_model.p", "wb"))

        with Path(f'vocab_model.p').open('rb') as vocab_file:  # Don't change the path here
            vocab_model = pickle.load(vocab_file)
        print("Saved vocab")

        X_train_transformed = apply_feature_transform(X_train, feature_detector_descriptor, vocab_model)

        # TODO: train a classifier and save it using pickle.dump function
        print("Training clf")
        clf.fit(X_train_transformed, y_train)
        pickle.dump(clf, open("clf.p", "wb"))
        print("Saved clf")
        X_to_clasify = X_train_transformed
        y_to_clasify = y_train
    else:
        with Path(f'vocab_model.p').open('rb') as vocab_file:  # Don't change the path here
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
    return score

def base_project_function():
    np.random.seed(42)

    first_name = 'Jakub'
    last_name = 'Bielawski'

    data_path = Path(f'test_testowy/')  # You can change the path here
    data_path = os.getenv('DATA_PATH', data_path)  # Don't change that line
    x, y = load_dataset(data_path)
    x = data_processing(x)
    feature_detector_descriptor = cv2.SIFT_create()

    with Path('vocab_model.p').open('rb') as vocab_file:  # Don't change the path here
        vocab_model = pickle.load(vocab_file)

    x_transformed = apply_feature_transform(x, feature_detector_descriptor, vocab_model)

    with Path('clf.p').open('rb') as classifier_file:  # Don't change the path here
        clf = pickle.load(classifier_file)

    score = clf.score(x_transformed, y)
    print(f'{first_name} {last_name} score: {score}')
    with Path(f'{last_name}_{first_name}_score.json').open('w') as score_file:
        json.dump({'score': score}, score_file)


def test(NB_WORDS, batch_size_K, random_state=42):
    first_name = 'Jakub'
    last_name = 'Bielawski'

    with Path(f'vocab_model.p').open('rb') as vocab_file:  # Don't change the path here
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


def plot_results():
    Scores = pd.read_csv('data.txt',sep=',')
    # Scores = pd.pivot_table(Scores,index='Batch_size',columns='NB_WORDS',margins=True)
    # print(Scores)
    batch_sizes = [64,128,256,512,1024,2048]
    Words=[20,50,100,200,500,700,1000,1500,2000]
    print(Scores)
    Scores = Scores.pivot(index='Batch_size',columns='NB_WORDS',values=['train_scores','test_scores'])
    Scores_train = Scores['train_scores']
    Scores_test = Scores['test_scores']
    # print(Scores_train.sum(axis=0).T)

    # Scores_train['SUM']=Scores_train.sum(axis=1)
    # # Scores_train=Scores_train.append(pd.DataFrame(Scores_train.sum(axis=0)).T,ignore_index=True)
    # Scores_test['SUM']=Scores_test.sum(axis=1)
    # print(Scores_train[Scores_train >=0.84])
    # print(Scores_test[Scores_test >=0.91])
    fig,ax = plt.subplots(2,1)
    ax[0].plot(Scores_train.columns,Scores_train.loc[64,:],label='64')
    ax[0].plot(Scores_train.columns,Scores_train.loc[128,:],label='128')
    ax[0].plot(Scores_train.columns,Scores_train.loc[256,:],label='256')
    ax[0].plot(Scores_train.columns,Scores_train.loc[512,:],label='512')
    ax[0].plot(Scores_train.columns,Scores_train.loc[1024,:],label='1024')
    ax[0].plot(Scores_train.columns,Scores_train.loc[2048,:],label='2048')

    ax[1].plot(Scores_test.columns,Scores_test.loc[64,:],label='64')
    ax[1].plot(Scores_test.columns,Scores_test.loc[128,:],label='128')
    ax[1].plot(Scores_test.columns,Scores_test.loc[256,:],label='256')
    ax[1].plot(Scores_test.columns,Scores_test.loc[512,:],label='512')
    ax[1].plot(Scores_test.columns,Scores_test.loc[1024,:],label='1024')
    ax[1].plot(Scores_test.columns,Scores_test.loc[2048,:],label='2048')
    ax[0].grid()
    ax[0].set_xticks(Words)
    ax[1].set_xticks(Words)

    ax[1].grid()
    plt.legend()

    Scores_train = Scores_train.T
    Scores_test = Scores_test.T

    fig,axs = plt.subplots(2,1)
    axs[0].plot(Scores_train.columns,Scores_train.loc[20,:],label='20')
    axs[0].plot(Scores_train.columns,Scores_train.loc[50,:],label='50')
    axs[0].plot(Scores_train.columns,Scores_train.loc[100,:],label='100')
    axs[0].plot(Scores_train.columns,Scores_train.loc[200,:],label='200')
    axs[0].plot(Scores_train.columns,Scores_train.loc[500,:],label='500')
    axs[0].plot(Scores_train.columns,Scores_train.loc[700,:],label='700')
    axs[0].plot(Scores_train.columns,Scores_train.loc[1000,:],label='1000')
    axs[0].plot(Scores_train.columns,Scores_train.loc[1500,:],label='1500')
    axs[0].plot(Scores_train.columns,Scores_train.loc[2000,:],label='2000')
    axs[0].set_xticks(batch_sizes)


    axs[1].plot(Scores_test.columns,Scores_test.loc[20,:],label='20')
    axs[1].plot(Scores_test.columns,Scores_test.loc[50,:],label='50')
    axs[1].plot(Scores_test.columns,Scores_test.loc[100,:],label='100')
    axs[1].plot(Scores_test.columns,Scores_test.loc[200,:],label='200')
    axs[1].plot(Scores_test.columns,Scores_test.loc[500,:],label='500')
    axs[1].plot(Scores_test.columns,Scores_test.loc[700,:],label='700')
    axs[1].plot(Scores_test.columns,Scores_test.loc[1000,:],label='1000')
    axs[1].plot(Scores_test.columns,Scores_test.loc[1500,:],label='1500')
    axs[1].plot(Scores_test.columns,Scores_test.loc[2000,:],label='2000')
    axs[0].grid()
    axs[1].grid()
    axs[1].set_xticks(batch_sizes)

    plt.legend()
    plt.show()


def find_best_parameters():
    train_features = read_and_describe_train_pictures()

    Scores = pd.DataFrame(columns=['CLF','Batch_size', 'NB_WORDS', 'train_scores', 'test_scores'])
    batch_sizes = [64,128,256,512,1024,2048]
    Words=[20,50,100,200,500,700,1000,1500,2000]
    for batch_size in batch_sizes:
        for word in Words:
            train_score, clf = find_parameters(train_features,word,batch_size)
            test_score = project(False,train_features,word,batch_size,42)
            df = pd.DataFrame({'CLF': [clf],
                               'Batch_size': [batch_size],
                               'NB_WORDS': [word],
                               'train_scores': [train_score],
                               'test_scores': [test_score]})
            Scores = Scores.append(df,ignore_index=True)
            Scores.to_csv('data.txt', sep=',')
            print(Scores)

if __name__ == '__main__':
    # train_features = read_and_describe_train_pictures()
    # find_parameters(train_features,2000,256)
    # project(True,train_features)
    # project(False,train_features)
    base_project_function()

    # base_project_function()




