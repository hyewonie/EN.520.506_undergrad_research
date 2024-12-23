# 1214__analysis.py
# Dec 14, 2024

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import cupy as cp
from soft_WL_CUDA import Soft_WL_Subtree_NodeLevel
import time
import gower
from scipy.sparse import csr_matrix
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.sparse as sp
import cupyx.scipy.sparse as cupyx_sp

def main():
    start_time = time.time()
    
    # Preprocessed data
    features = np.load('features_processed.npy')
    death_date = np.load('death_date_processed.npy')
    short_followup = np.load('short_followup_processed.npy')
    print("Shape of features:", features.shape)
    print("Shape of death_date:", death_date.shape)
    print("Shape of short_followup:", short_followup.shape)
    
    
    # Compute Gower similarity

    K = gower.gower_matrix(features)
    
    print("Similarity matrix computed.")
    print("\nGower Similarity Matrix:")
    print(K)
    
    # Remove self-loops
    np.fill_diagonal(K, 0)

    print("Shape of K:", K.shape)
    print("Data type of K:", K.dtype)

    np.savetxt('K_matrix_1215.csv', K, delimiter=',')
    print("K saved to 'K_matrix_1215.csv'.")

    # Create a NetworkX graph from the weighted adjacency matrix
    # G = nx.from_numpy_array(adj_matrix)
    
    # print("Graph created from similarity matrix.")

    """
    # Construct initial adjacency matrix using KNN
    n_nodes = features.shape[0]
    k_neighbors = 5  # Adjust based on your dataset size and requirements

    nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='auto').fit(features)
    distances, indices = nbrs.kneighbors(features)

    # Build the adjacency matrix using sparse representation
    row_indices = np.repeat(np.arange(n_nodes), k_neighbors)
    col_indices = indices.flatten()
    data = np.ones(len(row_indices), dtype=np.float32)  # Ensure data is float32
    adj_matrix = sp.csr_matrix((data, (row_indices, col_indices)), shape=(n_nodes, n_nodes))
    adj_matrix = adj_matrix.maximum(adj_matrix.transpose())  # Make symmetric

    print("Initial adjacency matrix constructed.")

    # Convert to CuPy CSR matrix
    adj_cp = cupyx_sp.csr_matrix(adj_matrix)

    # Get the current active device
    current_device = cp.cuda.Device().id
    print(f"Current active device: {current_device}")

    # Alternatively, get device name
    props = cp.cuda.runtime.getDeviceProperties(current_device)
    device_name = props['name'].decode('utf-8') if isinstance(props['name'], bytes) else props['name']

    print(f"Device name: {device_name}")
    
    # Instantiate and run the modified Soft WL Subtree method
    swl_node = Soft_WL_Subtree_NodeLevel(n_iter=1, n_jobs=-1, k=30, normalize=True)
    K = swl_node.fit_transform_single_graph(adj_cp, features)
    
    print("Similarity matrix computed.")
    """
    data_to_pickle = (K, features, short_followup) # load_data_medical requires 3 inputs.
    # Define the filename for the pickle file
    output_filename = 'Gower_shortest_followup_raw.pkl'

    # Save the data to the pickle file
    with open(output_filename, 'wb') as f:
        pickle.dump(data_to_pickle, f)

    print(f"Data successfully saved to '{output_filename}'.")

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    
    # Optional: Save processed data
    # np.save('features_processed.npy', features)
    # np.save('death_date.npy', death_date)
    # np.save('short_followup.npy', short_followup)


if __name__ == "__main__":
    main()