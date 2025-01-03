import pandas as pd
import numpy as np
import networkx as nx
import scipy.sparse as sp
import pickle
from sklearn.neighbors import kneighbors_graph

# Load the CSV file
# file_path = './diabetes.csv'
# df = pd.read_csv(file_path)

# Define features and labels
# features = df.drop(columns=['Outcome']).values  # drop the outcome label (not a feature)
# labels = df['Outcome'].values  # Outcome is the label
features = np.load('features_processed.npy')
labels = np.load('short_followup_processed.npy')

# Option 1 for adjcency matrix: Create a k-NN Adjacency Matrix

k_array = [5]
# k_array = [2, 16, 32, 50, 64, 100, 256, 512] # Number of nearest neighbors

for k in k_array:
    knn_graph = kneighbors_graph(features, n_neighbors=k, mode='connectivity', include_self=False)
    adj_2 = knn_graph.toarray()


    # Option 2 for adjcency matrix: fully connected graph
    # num_samples = features.shape[0]
    # adj_2 = np.ones((num_samples, num_samples))

    # test removing self-loops in graphs
    # np.fill_diagonal(adj_2, 0)

    print(f"Adjacency Matrix Shape: {adj_2.shape}")  # Should be (768, 768)

    data_to_pickle = (adj_2, features, labels) # load_data_medical requires 3 inputs.

    # Save the data as a .pkl file
    with open(f'shortest_followup_KNN_{k}.pkl', 'wb') as f:
        pickle.dump(data_to_pickle, f)



