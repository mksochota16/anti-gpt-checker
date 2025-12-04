from typing import List, Tuple, Dict, Optional

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.feature_selection import SelectKBest
import pandas as pd

from models.attribute import AttributeInDB


def find_significant_features(data):
    """
    Identifies the most significant features in a list of data.

    Args:
    data (list of tuples): A list where each element is a tuple containing a dictionary of features and a label.
    num_features (int): The number of top features to select.

    Returns:
    list: Names of the top `num_features` significant features.
    """
    # Convert list of tuples into a DataFrame
    df = pd.DataFrame([item[0] for item in data])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    labels = [item[1] for item in data]

    # Fill missing values if necessary
    df.ffill(inplace=True)

    # Compute mutual information
    mi_scores = mutual_info_classif(df, labels)

    # Create a Series with feature names as the index and MI scores as the values
    mi_series = pd.Series(mi_scores, index=df.columns)

    # Sort the features by their mutual information scores in descending order
    sorted_features = mi_series.sort_values(ascending=False).index.tolist()

    return sorted_features


def split_dataset(data_labels: List[Tuple[Dict, int]], train_size: float):
    """
    Shuffles and splits the data into training and testing sets.
    """
    np.random.shuffle(data_labels)
    split_idx = int(len(data_labels) * train_size)
    train_paired, test_paired = data_labels[:split_idx], data_labels[split_idx:]

    # Extract data and labels
    train_data = [x[0] for x in train_paired]
    train_labels = [x[1] for x in train_paired]
    test_data = [x[0] for x in test_paired]
    test_labels = [x[1] for x in test_paired]

    return train_data, train_labels, test_data, test_labels


def prepare_features(data: List[Dict]):
    """
    Converts list of dictionaries into a matrix of features suitable for sklearn models.
    """
    vectorizer = DictVectorizer(sparse=False)
    return vectorizer.fit_transform(data)


def convert_db_attributes_to_input_data(generated: List[AttributeInDB],
                                        real: List[AttributeInDB],
                                        num_of_features: int = 10,
                                        exclude_additionally: Optional[list] = None) -> List[Tuple[Dict, int]]:

    data = [(x.to_flat_dict_normalized(exclude=exclude_additionally), 1) for x in generated]
    data += [(x.to_flat_dict_normalized(exclude=exclude_additionally), 0) for x in real]
    # replace None with 0
    for i in range(len(data)):
        for key in data[i][0].keys():
            if data[i][0][key] is None:
                data[i][0][key] = 0

    significant_features = find_significant_features(data)
    if exclude_additionally:
        # remove features listed in exclude_additionally
        significant_features = [x for x in significant_features if x not in exclude_additionally]

    features_to_exclude = significant_features[num_of_features:]
    # add features to exclude
    if exclude_additionally:
        features_to_exclude.extend(exclude_additionally)

    data = [(x.to_flat_dict_normalized(exclude=features_to_exclude), 1) for x in generated]
    data += [(x.to_flat_dict_normalized(exclude=features_to_exclude), 0) for x in real]
    # replace None with 0
    for i in range(len(data)):
        for key in data[i][0].keys():
            if data[i][0][key] is None:
                data[i][0][key] = 0

    return data

def convert_db_to_attributes_selected_features(generated: List[AttributeInDB],
                             real: List[AttributeInDB],
                             selected_features: list[str]) -> List[Tuple[Dict, int]]:
    data = [(x.to_flat_dict_normalized(), 1) for x in generated]
    data += [(x.to_flat_dict_normalized(), 0) for x in real]

    attributes_dict = []

    for i in range(len(data)):
        filtered_row = {key: value for key, value in data[i][0].items() if key in selected_features}
        attributes_dict.append((filtered_row, data[i][1]))

    return attributes_dict
