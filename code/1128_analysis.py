# analysis.py
# updated 12/14/2024

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from soft_WL import Soft_WL_Subtree_NodeLevel
import time
from scipy.sparse import csr_matrix
import pickle

def main():
    start_time = time.time()
    
    # Preprocessed data
    features = np.load('features_processed.npy')
    death_date = np.load('death_date_processed.npy')
    short_followup = np.load('short_followup_processed.npy')
    print("Shape of features:", features.shape)
    print("Shape of death_date:", death_date.shape)
    print("Shape of short_followup:", short_followup.shape)
    
    # Construct initial adjacency matrix using KNN
    n_nodes = features.shape[0]
    k_neighbors = 5  # Adjust based on your dataset size and requirements
    
    nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='auto').fit(features)
    distances, indices = nbrs.kneighbors(features)
    
    # Build the adjacency matrix using sparse representation
    row_indices = np.repeat(np.arange(n_nodes), k_neighbors)
    col_indices = indices.flatten()
    data = np.ones(len(row_indices))
    adj_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(n_nodes, n_nodes))
    adj_matrix = adj_matrix.maximum(adj_matrix.transpose())  # Make symmetric
    
    print("Initial adjacency matrix constructed.")
    
    # Instantiate and run the modified Soft WL Subtree method
    swl_node = Soft_WL_Subtree_NodeLevel(n_iter=1, n_jobs=-1, k=30, normalize=True)
    K = swl_node.fit_transform_single_graph(adj_matrix, features)
    
    print("Similarity matrix computed.")
    
    # Remove self-loops
    np.fill_diagonal(K, 0)

    print("Shape of K:", K.shape)
    print("Data type of K:", K.dtype)

    np.savetxt('K_matrix_1215.csv', K, delimiter=',')
    print("K saved to 'K_matrix.csv'.")

    # Create a NetworkX graph from the weighted adjacency matrix
    G = nx.from_numpy_array(K)
    
    # print("Graph created from similarity matrix.")

    data_to_pickle = (K, features, short_followup) # load_data_medical requires 3 inputs.
    # Define the filename for the pickle file
    output_filename = 'shortest_followup_raw.pkl'

    # Save the data to the pickle file
    with open(output_filename, 'wb') as f:
        pickle.dump(data_to_pickle, f)

    print(f"Data successfully saved to '{output_filename}'.")


    """
    # Optionally, you can set node colors based on pattern IDs
    node_colors = swl_node.Pattern_ids
    # Draw the graph (limit to a subset for visualization if large)
    if n_nodes > 1000:
        print("Graph too large to visualize. Visualizing a subset of 1000 nodes.")
        subgraph_nodes = list(range(1000))
        G_sub = G.subgraph(subgraph_nodes)
        node_colors_sub = node_colors[:1000]
        
        pos = nx.spring_layout(G_sub, k=0.15, iterations=20)
        nx.draw_networkx_nodes(G_sub, pos, node_color=node_colors_sub, cmap=plt.cm.tab20, node_size=10)
        nx.draw_networkx_edges(G_sub, pos, width=0.5, edge_color='gray', alpha=0.5)
        plt.title("Subgraph Visualization")
        plt.show()
    else:
        pos = nx.spring_layout(G, k=0.15, iterations=20)
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=plt.cm.tab20, node_size=50)
        nx.draw_networkx_edges(G, pos, width=1, edge_color='gray', alpha=0.7)
        plt.title("Graph Visualization")
        plt.show()
    
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    
    # Optional: Save processed data
    # np.save('features_processed.npy', features)
    # np.save('death_date.npy', death_date)
    # np.save('short_followup.npy', short_followup)
    """

if __name__ == "__main__":
    main()