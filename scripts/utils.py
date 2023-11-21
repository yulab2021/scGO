import os
import pickle
import numpy as np
import pandas  as pd
import scanpy as sc

import torch
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import Dataset


def read_data(data_file):
    """
    Read data from a file.

    Parameters:
    - data_file: Path to the input data file.

    Returns:
    - Data as a NumPy array.

    Example usage:
    ```python
    my_data = read_data("my_data.txt")
    ```

    """
    data = pd.read_csv(data_file, sep="\t", header=None)
    return data



def rds_to_csv(rds_file):
    """
    Convert an RDS file to a CSV file.

    Parameters:
    - rds_file: Path to the input RDS file.

    Returns:
    - Write data to csv.

    Example usage:
    ```python
    rds_to_csv("my_data.rds")
    ```

    """
    cmd = "Rscript --vanilla rds_to_csv.r " + rds_file +" " +rds_file.replace(".rds", ".csv")

    os.system(cmd)




def h5ad_to_csv(h5ad_file):
    """
    Convert an h5ad file to a CSV file.

    Parameters:
    - h5ad_file: Path to the input h5ad file.

    Returns:
    - Csv format.

    Example usage:
    ```python
    h5ad_to_csv("my_data.h5ad")
    ```

    """

    adata= sc.read_h5ad(h5ad_file)

    dense_matrix = adata.X.todense()

    cell_ids = adata.obs_names

    gene_names = adata.var_names

    
    data = pd.DataFrame(dense_matrix, index=cell_ids, columns=gene_names)
    
    data.to_csv(h5ad_file.replace(".h5ad",".csv"), index=True, header=True)
    


def select_and_pad_features(original_df, feature_list):
    """
    Selects columns from the original DataFrame based on the provided feature_list.
    If a feature is missing in the original DataFrame, it adds the feature and fills it with zeros.
    The resulting DataFrame is then reordered based on the provided feature_list.

    Parameters:
    - original_df: The original DataFrame.
    - feature_list: A list of features to select and order.

    Returns:
    - A new DataFrame with selected and padded features, ordered as specified in feature_list.
    """
    # Select only the features present in the original DataFrame
    selected_features = [feature for feature in feature_list if feature in original_df.columns]

    # Create a new DataFrame with selected features
    new_df = original_df[selected_features].copy()

    # Add missing features and fill with zeros
    missing_features = set(feature_list) - set(selected_features)
    for feature in missing_features:
        new_df[feature] = 0

    # Reorder the columns based on feature_list
    new_df = new_df[feature_list]

    return new_df


def label_encoding(labels):
    """
    Encode categorical labels into numerical representations.

    Parameters:
    - labels: List of categorical labels to be encoded.

    Returns:
    - List of encoded numerical labels.

    Example usage:
    Assuming you have a list of labels,
    classes, encoded_labels = label_encoding(labels)
    """

    # Get unique classes from the input labels
    classes = list(set(labels))

    # Encode each label by its index in the unique classes
    encoded_labels = [classes.index(label) for label in labels]

    return classes, encoded_labels



def label_decoding(classes,labels):
    """
    Decode numerical labels into categorical representations.

    Parameters:
    - labels: List of numerical labels to be decoded.

    Returns:
    - List of decoded categorical labels.

    Example usage:
    Assuming you have a list of labels,
    decoded_labels = label_decoding(classes,labels)
    """


    # Decode each label by its index in the unique classes
    decoded_labels = [classes[label] for label in labels]
    
    return decoded_labels



class MyDataset(Dataset):
    """
    A custom PyTorch dataset for handling features and labels.

    Parameters:
    - x: Input features.
    - y: Corresponding labels.

    Example usage:
    ```python
    dataset = MyDataset(x, y)
    ```

    Usage in a DataLoader:
    ```python
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    for batch in dataloader:
        # Your training/validation loop here
    ```

    """
    def __init__(self, x, y):
        """
        Initializes the dataset with features and labels.

        Parameters:
        - x: Input features.
        - y: Corresponding labels.
        """
        self.x = x
        self.y = y

    def __getitem__(self, index):
        """
        Gets a single data sample.

        Parameters:
        - index: Index of the data sample.

        Returns:
        - features: Input features.
        - label: Corresponding label.
        """
        features = self.x[index]
        label = self.y[index]
        return features, label

    def __len__(self):
        """
        Returns the total number of data samples in the dataset.

        Returns:
        - Length of the dataset.
        """
        return len(self.x)



def generate_mask(row, col, percent=0.5, num_zeros=None):
    """
    Generate a binary mask with a specified percentage of zeros.

    Parameters:
    - row: Number of rows in the mask.
    - col: Number of columns in the mask.
    - percent: Percentage of zeros in the mask (default is 0.5).
    - num_zeros: Number of zeros in the mask (overrides percent if provided).

    Returns:
    - Binary mask as a NumPy array.

    Example usage:
    Assuming you want to generate a mask for a matrix,
    mask = generate_mask(10, 10, percent=0.3)
    """

    if num_zeros is None:
        # Total number being masked is 0.5 by default
        num_zeros = int((row * col) * percent)

    # Create a binary mask with the specified number of zeros
    mask = np.hstack([np.zeros(num_zeros), np.ones(row * col - num_zeros)])
    np.random.shuffle(mask)

    return mask.reshape(row, col)



def accuracy_score(true_labels, predicted_labels):
    """
    Calculate the accuracy score.

    Parameters:
    - true_labels: True labels.
    - predicted_labels: Predicted labels.

    Returns:
    - Accuracy score.
    """
    correct_predictions = 0
    incorrect_predictions = 0

    for true_label, predicted_label in zip(true_labels, predicted_labels):
        if true_label == predicted_label:
            correct_predictions += 1
        else:
            incorrect_predictions += 1

    accuracy = correct_predictions / (correct_predictions + incorrect_predictions)
    return accuracy





def make_weights_for_balanced_classes(dataset: Dataset, nclasses: int):
    """
    Generate weights for balanced classes in a dataset.

    Parameters:
    - dataset: PyTorch dataset containing samples with labels.
    - nclasses: Number of classes in the dataset.

    Returns:
    - List of weights corresponding to each sample in the dataset.

    Example usage:
    ```python
    weights = make_weights_for_balanced_classes(my_dataset, num_classes)
    sampler = WeightedRandomSampler(weights, num_samples=len(my_dataset), replacement=True)
    dataloader = DataLoader(my_dataset, batch_size=batch_size, sampler=sampler)
    ```

    """
    count = [0] * nclasses

    # Count the occurrences of each class in the dataset
    for item in dataset:
        count[item[1]] += 1

    weight_per_class = [0.] * nclasses
    N = float(sum(count))

    # Calculate weights for each class
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])

    weight = [0] * len(dataset)

    # Assign weights to each sample based on its class
    for idx, val in enumerate(dataset):
        weight[idx] = weight_per_class[val[1]]

    return weight





class CustomWeightedRandomSampler(WeightedRandomSampler):
    """
    WeightedRandomSampler extension that allows for more than 2^24 samples to be sampled.

    Parameters:
    - weights: A sequence of weights, not necessarily summing up to one.
    - num_samples: Number of samples to draw.
    - replacement: If True, samples are drawn with replacement. If False, without replacement.

    Example usage:
    ```python
    sampler = CustomWeightedRandomSampler(weights, num_samples, replacement=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    ```

    """
    def __init__(self, *args, **kwargs):
        """
        Initializes the CustomWeightedRandomSampler.

        Parameters:
        - *args: Variable length argument list passed to the WeightedRandomSampler constructor.
        - **kwargs: Keyword arguments passed to the WeightedRandomSampler constructor.
        """
        super().__init__(*args, **kwargs)

    def __iter__(self):
        """
        Generates an iterator for sampling indices based on weights.

        Returns:
        - Iterator for sampled indices.
        """
        rand_tensor = np.random.choice(
            range(0, len(self.weights)),
            size=self.num_samples,
            p=self.weights.numpy() / torch.sum(self.weights).numpy(),
            replace=self.replacement
        )
        rand_tensor = torch.from_numpy(rand_tensor)
        return iter(rand_tensor.tolist())



def prepare_GO_data(annotation_file, min_gene_count):
    """
    Prepare GO dictionaries based on the provided annotation file.

    Parameters:
    - annotation_file: Path to the GO annotation file.
    - min_gene_count: Minimum count of genes associated with a GO term.

    Returns:
    - GO_list: List of GO terms with counts greater than or equal to min_gene_count.
    - gene_dict: Dictionary mapping gene IDs to associated GO terms.

    Example usage:
    GO_list, gene_dict = prepare_GO_data("human/goa_human.gaf", min_gene_count=30)
    print(len(GO_list))

    """
    GO_dict = {}

    with open(annotation_file) as f:
        for line in f:
            if line[0] == "!":
                continue
            
            gene_id = line.split("\t")[2]
            GO_term = line.split("\t")[4]

            if GO_term not in GO_dict:
                GO_dict[GO_term] = []

            GO_dict[GO_term].append(gene_id)

    GO_list = []
    count = 0

    # Select GO terms with counts greater than or equal to min_gene_count
    for item in GO_dict:
        if len(GO_dict[item]) >= min_gene_count:
            count += 1
            GO_list.append(item)

    print(count)

    gene_dict = {}

    with open(annotation_file) as f:
        for line in f:
            if line[0] == "!":
                continue

            gene_id = line.split("\t")[2].upper()
            GO_term = line.split("\t")[4]

            if gene_id not in gene_dict:
                gene_dict[gene_id] = []

            gene_dict[gene_id].append(GO_term)

    return GO_list, gene_dict






def prepare_TF_data(annotation_file, output_TF_gene_dict, output_gene_TF_dict):
    """
    Prepare TF-gene dictionaries based on the provided annotation file.

    Parameters:
    - annotation_file: Path to the TF annotation file.
    - output_TF_gene_dict: Output path for the TF-gene dictionary (TF to genes).
    - output_gene_TF_dict: Output path for the gene-TF dictionary (genes to TF).

    Returns:
    None

    Example usage:
    prepare_TF_data("human/human_TF.annotate", "human/TF_gene_dict.pkl", "human/gene_TF_dict.pkl")

    """
    TF_gene_dict = {}  # {TF: [gene]}
    gene_TF_dict = {}  # {gene: [TF]}

    with open(annotation_file) as f:
        for line in f:
            if "PeakID" in line or "promoter-TSS" in line:
                continue

            TF = line.split(":")[0]
            gene_name = line.split("\t")[15]

            if TF not in TF_gene_dict:
                TF_gene_dict[TF] = []

            TF_gene_dict[TF].append(gene_name)

            if gene_name not in gene_TF_dict:
                gene_TF_dict[gene_name] = []

            gene_TF_dict[gene_name].append(TF)

    print("Number of TFs:", len(TF_gene_dict))

    # Save dictionaries to pickle files
    pickle.dump(TF_gene_dict, open(output_TF_gene_dict, "wb"))
    pickle.dump(gene_TF_dict, open(output_gene_TF_dict, "wb"))






def generate_gene_to_TF_transform_matrix(TF_gene_dict, data_rm_sparse, base_dir):
    """
    Generate a gene-to-TF transformation matrix.

    Parameters:
    - TF_gene_dict: Dictionary mapping TFs to associated genes.
    - data_rm_sparse: DataFrame with genes as columns.
    - base_dir: Base directory to save the transformation matrix.

    Returns:
    - gene_to_TF_transform_matrix: NumPy array representing the gene-to-TF transformation matrix.

    Example usage:
    ```python
    matrix = generate_gene_to_TF_transform_matrix(TF_gene_dict, my_data, "my_directory")
    ```

    """
    gene_number = len(data_rm_sparse.columns.to_list())    
    TF_number = len(TF_gene_dict)

    gene_to_TF_transform_matrix = np.zeros((gene_number, TF_number))

    TF_list = list(TF_gene_dict.keys())

    for i, gene in enumerate(data_rm_sparse.columns):
        try:
            j = TF_list.index(gene)
            gene_to_TF_transform_matrix[i][j] = 1
        except ValueError:
            # Handle the case when the gene is not in the TF list
            pass

    # Save the transformation matrix as a pickle file
    pickle.dump(gene_to_TF_transform_matrix, open(f"{base_dir}/gene_to_TF_transform_matrix.pkl", "wb"))

    return gene_to_TF_transform_matrix




def generate_GO_mask(annotation_file, data_rm_sparse, base_dir, min_gene_count=30):
    """
    Generate a GO mask based on the provided annotation file and data.

    Parameters:
    - annotation_file: Path to the GO annotation file.
    - data_rm_sparse: DataFrame with genes as columns.
    - base_dir: Base directory to save the GO mask.
    - min_gene_count: Minimum count of genes associated with a GO term.

    Returns:
    - GO_mask: NumPy array representing the GO mask.

    Example usage:
    ```python
    mask = generate_GO_mask("human/goa_human.gaf", my_data, "my_directory", min_gene_count=30)
    ```

    """
    GO_dict = {}
    
    with open(annotation_file) as f:
        for line in f:
            if line[0] == "!":
                continue
            
            gene_id = line.split("\t")[2]
            GO_term = line.split("\t")[4]
            
            if GO_term not in GO_dict:
                GO_dict[GO_term] = []
                
            GO_dict[GO_term].append(gene_id)

    GO_list = [item for item in GO_dict if len(GO_dict[item]) >= min_gene_count]
    print(len(GO_list))

    gene_dict = {}
    
    with open(annotation_file) as f:
        for line in f:
            if line[0] == "!":
                continue
            
            gene_id = line.split("\t")[2].upper()
            GO_term = line.split("\t")[4]
            
            if gene_id not in gene_dict:
                gene_dict[gene_id] = []
                
            gene_dict[gene_id].append(GO_term)

    gene_number = len(data_rm_sparse.columns)
    GO_number = len(GO_list)
    GO_mask = np.zeros((gene_number, GO_number))
    error_count = 0

    for i, gene_id in enumerate(data_rm_sparse.columns):
        for j, GO_term in enumerate(GO_list):
            if GO_term in gene_dict.get(gene_id, []):
                GO_mask[i][j] = 1
            else:
                error_count += 1

    print(error_count)

    # Save the GO mask as a pickle file
    pickle.dump(GO_mask, open(f"{base_dir}/GO_mask.pkl", "wb"))

    return GO_mask






def generate_TF_mask(gene_TF_dict, TF_gene_dict, data_rm_sparse, base_dir):
    """
    Generate a TF_mask based on gene-TF associations.

    Parameters:
    - gene_TF_dict: A dictionary mapping gene IDs to associated TFs.
    - TF_gene_dict: A list or dictionary containing TF names or IDs.
    - data_rm_sparse: The input data, possibly a DataFrame with gene columns.
    - base_dir: The base directory to save the TF_mask.

    Returns:
    None (saves the TF_mask as a pickle file).
    
    Example usage:
    Assuming you have the required data structures (gene_TF_dict, TF_gene_dict, data_rm_sparse, base_dir)
    generate_TF_mask(gene_TF_dict, TF_gene_dict, data_rm_sparse, base_dir)
    """
    gene_number = len(data_rm_sparse.columns.to_list())
    TF_number = len(TF_gene_dict)

    TF_mask = np.zeros((gene_number, TF_number))
    error_count = 0

    for i, gene_id in enumerate(data_rm_sparse.columns):
        for j, TF in enumerate(TF_gene_dict):
            if TF in gene_TF_dict.get(gene_id, []):
                TF_mask[i][j] = 1
            else:
                error_count += 1

    print("Total Errors:", error_count)
    print("TF_mask:\n", TF_mask)

    # Save TF_mask as a pickle file
    pickle.dump(TF_mask, open(f"{base_dir}/TF_mask.pkl", "wb"))





def generate_GO_to_TF_mask(TF_gene_dict, gene_dict, GO_list, base_dir):
    """
    Generate a GO-to-TF mask based on TF-gene and gene-GO associations.

    Parameters:
    - TF_gene_dict: Dictionary mapping TFs to associated genes.
    - gene_dict: Dictionary mapping genes to associated GO terms.
    - GO_list: List of GO terms.
    - base_dir: Base directory to save the GO-to-TF mask.

    Returns:
    - GO_TF_mask: NumPy array representing the GO-to-TF mask.

    Example usage:
    ```python
    mask = generate_GO_to_TF_mask(my_TF_gene_dict, my_gene_dict, my_GO_list, "my_directory")
    ```

    """
    TF_number = len(TF_gene_dict)
    GO_number = len(GO_list)
    GO_TF_mask = np.zeros((TF_number, GO_number))
    error_count = 0

    for i, TF in enumerate(TF_gene_dict):
        for j, GO in enumerate(GO_list):
            if GO in gene_dict.get(TF, []):
                GO_TF_mask[i][j] = 1
            else:
                error_count += 1

    print(error_count)

    # Save the GO-to-TF mask as a pickle file
    pickle.dump(GO_TF_mask, open(f"{base_dir}/GO_TF_mask.pkl", "wb"))

    return GO_TF_mask



def calculate_multiclass_f1_score(true_labels, predicted_labels):
    """
    Calculate the macro F1 score for a multiclass classification problem.

    Parameters:
    - true_labels: List of true labels.
    - predicted_labels: List of predicted labels.

    Returns:
    - macro_f1: Macro F1 score.

    Raises:
    - ValueError: If the input lists have different lengths.

    Example usage:
    ```python
    true_labels = [0, 1, 2, 0, 1, 2]
    predicted_labels = [0, 1, 2, 0, 1, 1]
    f1_score = calculate_multiclass_f1_score(true_labels, predicted_labels)
    print(f1_score)
    ```

    """
    if len(true_labels) != len(predicted_labels):
        raise ValueError("Input lists must have the same length.")

    unique_labels = set(true_labels + predicted_labels)
    f1_scores = []

    for label in unique_labels:
        true_positive = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == label and pred == label)
        false_positive = sum(1 for true, pred in zip(true_labels, predicted_labels) if true != label and pred == label)
        false_negative = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == label and pred != label)

        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0

        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / len(f1_scores)
    return macro_f1


def data_normalization(data):
    """
    Normalize the rows of a DataFrame by dividing each row by its sum and multiplying by 1e4.

    Parameters:
    - data: Input DataFrame with rows to be normalized.

    Returns:
    - Normalized DataFrame.

    Example usage:
    Assuming you have your data DataFrame,
    normalized_data = data_normalization(data)
    """

    # Calculate the sum of each row
    row_sums = data.sum(axis=1)

    # Normalize each row by dividing by its sum and multiplying by 1e4
    data_normalized = data.div(row_sums, axis=0) * 1e4

    return data_normalized






