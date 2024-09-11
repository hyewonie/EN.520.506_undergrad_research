import pickle

# Load the pickle file
file_path = './research/graph_data_with_features.pkl'  # Replace with the path to your pickle file
with open(file_path, 'rb') as f:
    adj_matrix, feature_vectors, labels = pickle.load(f)

# Inspect the contents
print("Adjacency Matrix Shape:", adj_matrix.shape)
print("Feature Vectors Shape:", feature_vectors.shape)
print("Labels Shape:", labels.shape)

# Display a sample of the data
print("\nSample of Adjacency Matrix:\n", adj_matrix[:5, :5])  # Show a small part of the matrix
print("\nSample of Feature Vectors:\n", feature_vectors[:5])   # Show a few feature vectors
print("\nSample of Labels:\n", labels[:25])                    # Show a few labels
