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
from sklearn.model_selection import StratifiedKFold,cross_val_score
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from typing import List

project_Name = 'ImageClfProject'


def load_dataset(dataset_dir_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for i, class_dir in enumerate(sorted(dataset_dir_path.iterdir())):
        for file in class_dir.iterdir():
            img_file = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)

            x.append(img_file)
            y.append(i)

    return np.asarray(x), np.asarray(y)

def data_processing(x: List[np.ndarray]) -> np.ndarray:
    # if needed
    return x


class DataDescriber:
    def __init__(self, SIFTFeatures=500, vocab_clusters=2_000, vocab_batch_size=5_000, random_state=42):

        self.features = []
        self.vocab_clusters = vocab_clusters
        self.vocab_batch_size = vocab_batch_size
        self.random_state = random_state
        self.vocab_model = cluster.MiniBatchKMeans(n_clusters=self.vocab_clusters,
                                                   random_state=self.random_state,
                                                   batch_size=self.vocab_batch_size)
        self.SIFTFeatures = SIFTFeatures


    #enter i exit
    # kontekst menadżer
    def fit(self, x, y):

        return self

    def get_params(self, deep=True):
        return {"SIFTFeatures": self.SIFTFeatures,
                "vocab_batch_size": self.vocab_batch_size,
                "vocab_clusters": self.vocab_clusters
                }

    def set_params(self,**parameters):

        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def transform(self, x):

        for image in tqdm(x, desc="Description progress: ",
                          bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET)):
            key_points, image_decriptors = cv2.SIFT_create(nfeatures=self.SIFTFeatures).detectAndCompute(image, None)

            for feature in image_decriptors:
                self.features.append(feature)

        return self.apply_feature_transform(x)

    def fit_transform(self, x, y):
        self.fit(x, y)
        # print('fit_transform')
        for image in tqdm(x, desc="Description progress: ",
                          bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET)):
            #
            key_points, image_decriptors = cv2.SIFT_create(nfeatures=self.SIFTFeatures).detectAndCompute(image, None)
            for feature in image_decriptors:
                self.features.append(feature)

        print(f"Fitting vocab model on {len(self.features)} features.")
        self.fit_vocab()
        # self.x_transformed = self.apply_feature_transform(x)
        return self.apply_feature_transform(x)

    def fit_vocab(self):
        self.vocab_model.fit(self.features)
        pickle.dump(self.vocab_model, open(f"vocab_model.p", "wb"))
        print("Saved vocab")

    def apply_feature_transform(self,
                                data: np.ndarray,
                                ) -> np.ndarray:
        self.data_transformed = []
        for image in tqdm(data, desc="Applying transform: ",
                          bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.LIGHTGREEN_EX, Fore.RESET)):
            keypoints, image_descriptors = cv2.SIFT_create(nfeatures=self.SIFTFeatures).detectAndCompute(image, None)
            bow_features_histogram = self.convert_descriptor_to_histogram(image_descriptors)
            self.data_transformed.append(bow_features_histogram)

        # self.features = []

        return np.asarray(self.data_transformed, dtype='object')

    def convert_descriptor_to_histogram(self, descriptors, normalize=True) -> np.ndarray:
        descriptors = descriptors.astype(np.float)
        features_words = self.vocab_model.predict(descriptors)
        histogram = np.zeros(self.vocab_model.n_clusters, dtype=np.float32)
        unique, counts = np.unique(features_words, return_counts=True)
        histogram[unique] += counts
        if normalize:
            histogram /= histogram.sum()
        return histogram


def make_classification(NB_WORDS=2_000, batch_size_K=5_000, random_state=42):
    np.random.seed(42)

    cv = StratifiedKFold(n_splits=7, shuffle=True, random_state=random_state)
    clf = svm.LinearSVC(tol=1e-5, random_state=random_state)

    scaler = MinMaxScaler()
    preproceser = DataDescriber()


    data_path = Path('train/')  # You can change the path here

    data_path = os.getenv('DATA_PATH', data_path)  # Don't change that line
    x, y = load_dataset(data_path)
    print("Data loaded")
    flow = Pipeline([('TransformData', preproceser), ('Scaler', scaler), ('Classyfier', clf)])

    scores = cross_val_score(flow, x, y, cv=cv, scoring='accuracy')
    print("Accuracy: %0.4f (+/- %0.4f)" % (
        scores.mean(), scores.std() * 2))

    pickle.dump(flow, open("clf.p", "wb"))
    print("Saved clf")
    return scores.mean(), scores.std() * 2, max(scores)


def test_data():
    np.random.seed(42)

    first_name = 'Name'
    last_name = 'Surname'

    data_path = Path('./../../test_data/')  # You can change the path here
    data_path = os.getenv('DATA_PATH', data_path)  # Don't change that line
    x, y = load_dataset(data_path)

    with Path('clf.p').open('rb') as classifier_file:  # Don't change the path here
        clf = pickle.load(classifier_file)

    score = clf.score(x, y)
    print(f'{first_name} {last_name} score: {score}')
    with Path(f'{last_name}_{first_name}_score.json').open('w') as score_file:
        json.dump({'score': score}, score_file)

    return score


def find_best_parameters():
    Scores = pd.DataFrame(columns=['NB_WORDS', 'scores_mean', 'scores_std', 'scores_max', 'test_score'])
    Words = [20, 50, 100, 200, 500, 700, 1000, 1500, 2000, 2500, 3000, 5000]

    for word in Words:
        score_mean, score_std, scores_max = make_classification(word)
        score_test = test_data()
        df = pd.DataFrame({'NB_WORDS': [word],
                           'scores_mean': [score_mean],
                           'scores_std': [score_std],
                           'scores_max': [scores_max],
                           'test_score': [score_test]})
        Scores = Scores.append(df, ignore_index=True)
        Scores.to_csv(f'data_new_approach_3.txt', sep=',')
        print(Scores)


if __name__ == '__main__':
    make_classification()
    # test_data()


