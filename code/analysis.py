# analysis.py
# last edited 12/14/2024
import pandas as pd
import numpy as np
import pickle

# Preprocessed data
features = np.load('features_processed.npy')
death_date = np.load('death_date_processed.npy')
short_followup = np.load('short_followup_processed.npy')

# Load the Adjacency Matrix
df = pd.read_csv('K_matrix_1214.csv', header=None)

# Display the original adjacency matrix
print("Original Adjacency Matrix:")
print(df)

weights = df.values.copy()

# Define Threshold Percentages
threshold_percentages = [80, 85, 90, 95, 99]

# Calculate the threshold values based on percentiles
threshold_values = np.percentile(weights, threshold_percentages)
threshold_dict = dict(zip(threshold_percentages, threshold_values))

print("\nThreshold Values:")
for perc, val in threshold_dict.items():
    print(f"{perc}% Threshold: {val}")

# Create Binary Adjacency Matrices
for perc in threshold_percentages:
    threshold = threshold_dict[perc]
    # Apply threshold: 1 if weight > threshold, else 0
    binary_df = df.applymap(lambda x: 1 if x >= threshold else 0)
    
    # Ensure diagonal remains 0 (no self-connections)
    np.fill_diagonal(binary_df.values, 0)
    K = binary_df.to_numpy()
    
    # Display the binary adjacency matrix
    print(f"\nBinary Adjacency Matrix at {perc}% Threshold:")
    print(K)
    ones = np.count_nonzero(K)
    print("number of 1's: ", ones)

    output_filename = f'Gower_shortest_followup_{perc}.pkl'
    data_to_pickle = (K, features, short_followup)
    # Save the data to the pickle file
    with open(output_filename, 'wb') as f:
        pickle.dump(data_to_pickle, f)

    print(f"Data successfully saved to '{output_filename}'.")
