import pickle


file_path = './medical_data_2.pkl'


# Function to load the pickle file
def test_pickle(file_path):
    with open(file_path, 'rb') as f:
        adj, features, labels = pickle.load(f)
    print("Adjacency Matrix Shape:", adj.shape)
    print("Features Shape:", features.shape)
    print("Labels Shape:", labels.shape)

# Test the newly created pickle file
test_pickle('medical_data_2.pkl')
