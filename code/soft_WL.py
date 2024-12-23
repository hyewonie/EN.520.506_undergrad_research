# Modified Implementation of Soft WL subtree kernel for Node-Level Analysis
# adapted from: https://github.com/Sulam-Group/BiGraph4TME/blob/master/soft_wl_subtree.py

import numpy as np
import phenograph
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity


class Soft_WL_Subtree_NodeLevel(object):
    """Calculate the Soft WL subtree kernel at the node level for a single graph"""

    def __init__(self, n_iter=0, n_jobs=-1, k=100, normalize=True):
        self.n_iter = n_iter  # Number of iterations of graph convolution
        self.n_jobs = n_jobs  # Number of jobs for parallel computation
        self.k = k  # Number of neighbors for Phenograph clustering
        self.normalize = normalize  # Whether to normalize the kernel matrix
        print(
            "Initialize SoftWL Node-Level: n_iter={}, n_jobs={}, k={}, normalize={}".format(
                self.n_iter, self.n_jobs, self.k, self.normalize
            )
        )
        self.Signatures = None  # Initialize the signatures of the patterns
        self.Histograms = None  # Initialize the histograms
        self.num_patterns = None  # Initialize the number of patterns
        self.subtree_feature = None  # Initialize the subtree features
        self.Pattern_ids = None  # Initialize the pattern IDs
        self.Similarity_matrix = None  # Initialize the similarity matrix

    def graph_convolution(self, adj, x):
        """
        Perform graph convolution on the node features.
        Parameters
        ----------
        adj : numpy array, shape = [n_nodes, n_nodes]
            Adjacency matrix of the graph
        x : numpy array, shape = [n_nodes, n_features]
            Node feature matrix
        Returns
        -------
        x : numpy array, shape = [n_nodes, n_features]
            Updated node feature matrix after graph convolution
        """
        adj = adj.copy()  # Create a copy to avoid modifying the original adjacency matrix
        adj.setdiag(1)    # Include self-loops by setting the diagonal to 1
        x = x.astype(np.float32)  # Convert to float32

        for i in range(self.n_iter):
            x = adj.dot(x)  # Use the dot method for sparse matrix multiplication
        return x

    def cluster_subtrees(self, X):
        """
        Cluster the subtrees (node features) using Phenograph.
        Parameters
        ----------
        X : numpy array, shape = [n_samples, n_features]
            Subtree features (node features after convolution)
        Returns
        -------
        cluster_identities : numpy array, shape = [n_samples]
            Cluster labels for each node
        """
        # Suppress warnings from Phenograph
        import warnings
        warnings.filterwarnings("ignore")
        # Clustering
        cluster_identities, _, _ = phenograph.cluster(X, n_jobs=self.n_jobs, k=self.k)
        return cluster_identities

    def compute_cluster_centroids(self, X, Cluster_identities):
        """
        Compute the centroids of the clusters (patterns).
        Parameters
        ----------
        X : numpy array, shape = [n_samples, n_features]
            Subtree features
        Cluster_identities : numpy array, shape = [n_samples]
            Cluster labels for each node
        Returns
        -------
        Signatures : numpy array, shape = [n_clusters, n_features]
            Centroids of the clusters
        """
        unique_clusters = np.unique(Cluster_identities)
        n_clusters = len(unique_clusters)
        Signatures = np.zeros((n_clusters, X.shape[1]))
        for i, cluster_id in enumerate(unique_clusters):
            Signatures[i] = np.mean(X[Cluster_identities == cluster_id], axis=0)
        return Signatures

    def discover_patterns_single_graph(self, adj, x):
        """
        Discover patterns in a single graph by performing graph convolution and clustering nodes.
        Parameters
        ----------
        adj : numpy array, shape = [n_nodes, n_nodes]
            Adjacency matrix of the graph
        x : numpy array, shape = [n_nodes, n_features]
            Node feature matrix
        Returns
        -------
        subtree_feature : numpy array, shape = [n_nodes, n_features]
            Node features after graph convolution
        Pattern_ids : numpy array, shape = [n_nodes]
            Cluster labels (pattern IDs) for each node
        Signatures : numpy array, shape = [n_clusters, n_features]
            Centroids of the clusters
        """
        print(
            "Discovering patterns in a single graph with {} nodes and node feature dimension {}".format(
                adj.shape[0], x.shape[1]
            )
        )
        # Perform graph convolution
        subtree_feature = self.graph_convolution(adj, x)
        # Cluster the nodes based on their features
        print("\tClustering nodes based on updated features")
        Pattern_ids = self.cluster_subtrees(subtree_feature)
        # Compute cluster centroids (signatures)
        Signatures = self.compute_cluster_centroids(subtree_feature, Pattern_ids)
        # Store results
        self.subtree_feature = subtree_feature
        self.Pattern_ids = Pattern_ids
        self.Signatures = Signatures
        self.num_patterns = len(Signatures)
        return subtree_feature, Pattern_ids, Signatures

    def fit_transform_single_graph(self, adj, x):
        self.discover_patterns_single_graph(adj, x)
        # Build histogram for the graph
        histogram = np.zeros(self.num_patterns)
        for i in range(self.num_patterns):
            histogram[i] = np.sum(self.Pattern_ids == i)
        self.Histograms = [histogram]
        
        # Compute similarity matrix based on cosine similarity of cluster centroids
        K = cosine_similarity(self.Signatures)  # Shape: [n_patterns, n_patterns]
        
        # Create a mapping from node to cluster centroid
        node_centroids = self.Signatures[self.Pattern_ids]  # Shape: [n_nodes, n_features]
        
        # Compute cosine similarity between nodes based on their centroids
        K_nodes = cosine_similarity(node_centroids)  # Shape: [n_nodes, n_nodes]
        
        if self.normalize:
            # Normalize the similarity matrix
            diag = np.diag(K_nodes).copy()
            diag[diag == 0] = 1  # Avoid division by zero
            K_nodes = K_nodes / np.sqrt(np.outer(diag, diag))
        
        self.Similarity_matrix = K_nodes
        return K_nodes

    def transform_single_graph(self, adj, x):
        """
        Transform new node data using the fitted model and compute similarities.
        Parameters
        ----------
        adj : numpy array, shape = [n_nodes, n_nodes]
            Adjacency matrix of the graph
        x : numpy array, shape = [n_nodes, n_features]
            Node feature matrix
        Returns
        -------
        K_new : numpy array, shape = [n_nodes_fitted, n_nodes_new]
            Similarity matrix between fitted nodes and new nodes
        """
        # Perform graph convolution
        subtree_feature_new = self.graph_convolution(adj, x)
        # Map new nodes to the closest existing patterns
        Pattern_ids_new = self.closest_cluster_mapping(subtree_feature_new, self.Signatures)
        # Compute similarity matrix
        n_nodes_fitted = self.subtree_feature.shape[0]
        n_nodes_new = subtree_feature_new.shape[0]
        K_new = np.zeros((n_nodes_fitted, n_nodes_new))
        for i in range(n_nodes_fitted):
            for j in range(n_nodes_new):
                if self.Pattern_ids[i] == Pattern_ids_new[j]:
                    K_new[i, j] = 1
        if self.normalize:
            diag_fitted = np.diag(np.dot(K_new, K_new.T))
            diag_new = np.diag(np.dot(K_new.T, K_new))
            diag_fitted[diag_fitted == 0] = 1
            diag_new[diag_new == 0] = 1
            K_new = K_new / np.sqrt(np.outer(diag_fitted, diag_new))
        return K_new

    def closest_cluster_mapping(self, X, Signatures):
        """
        Map each node to the closest cluster (pattern) based on the signatures.
        Parameters
        ----------
        X : numpy array, shape = [n_nodes, n_features]
            Node features after graph convolution
        Signatures : numpy array, shape = [n_clusters, n_features]
            Centroids of the clusters
        Returns
        -------
        Pattern_ids_hat : numpy array, shape = [n_nodes]
            Cluster labels (pattern IDs) for each node
        """
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(Signatures)
        distances, indices = neigh.kneighbors(X)
        Pattern_ids_hat = indices.flatten()
        return Pattern_ids_hat
