# preprocess.py

import warnings
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# Preprocess the data for graph generation.
# Drops Oxygen_saturation column (missing too many values).
# Creates 3 .npy files
"""
def preprocess_data(csv_file):
    # Load CSV data
    df = pd.read_csv(csv_file)
    
    # Drop 'XNATSessionID'
    df = df.drop(columns=['XNATSessionID'])
    
    # Extract outcome variables
    death_date = df['Death_Date'].map({'No': 0, 'Yes': 1}).values
    # Step 1: Replace empty strings with NaN
    df['Short_Followup_Less_Than_30_Days'] = df['Short_Followup_Less_Than_30_Days'].replace('', np.nan)
    
    # Step 2: Assign 'FALSE' to NaN values
    df['Short_Followup_Less_Than_30_Days'] = df['Short_Followup_Less_Than_30_Days'].fillna('FALSE')
    
    # Step 3: Standardize the entries (uppercase and strip whitespace)
    short_followup_standardized = df['Short_Followup_Less_Than_30_Days'].astype(str).str.strip().str.upper()
    
    # Step 4: Define the mapping dictionary
    mapping_dict = {
        'TRUE': 1,
        'FALSE': 0,
        # Add more mappings if there are other representations
    }
    
    # Step 5: Map the standardized values to numerical values
    short_followup_mapped = short_followup_standardized.map(mapping_dict)
    
    # Convert to integer type
    short_followup = short_followup_mapped.astype(int).values
        
    # Drop outcome variables from features
    features_df = df.drop(columns=['Death_Date', 'Short_Followup_Less_Than_30_Days'])
    
    # Define categorical and numerical columns
    binary_categorical_cols = ['Altered Mental Status', 'COPD', 
                               'Chronic Heart Failure', 'Chronic Cancer']
    non_binary_categorical_cols = ['Gender', 'Race']
    numerical_cols = ['Age', 'Pulse', 'BP_Systolic', 'Respiratory Rate', 
                      'Temperature', 'Oxygen_saturation', 
                      'Shortest follow-up', 'Follow-up']
    
    # Extract categorical and numerical data
    binary_categorical_df = features_df[binary_categorical_cols]
    non_binary_categorical_df = features_df[non_binary_categorical_cols]
    numerical_df = features_df[numerical_cols]
    
    # First, drop 'Oxygen_saturation'
    numerical_df = numerical_df.drop(columns=['Oxygen_saturation'])
    numerical_cols = ['Age', 'Pulse', 'BP_Systolic', 'Respiratory Rate', 
                      'Temperature', 'Shortest follow-up', 'Follow-up'] #TODO: shortest follow up???
    
    print("Updated Numerical Columns:", numerical_cols)
    
    # Define a function to map 'No' to 0 and any other value to 1
    def map_binary_abnormality(value):
        if isinstance(value, str) and value.strip().lower() == 'no':
            return 0
        else:
            return 1
    
    # Apply the mapping to 'Respiratory Rate' and 'Temperature'
    numerical_df['Respiratory Rate'] = numerical_df['Respiratory Rate'].apply(map_binary_abnormality)
    numerical_df['Temperature'] = numerical_df['Temperature'].apply(map_binary_abnormality)
    
    # Verify the conversion
    print("Converted 'Respiratory Rate' and 'Temperature':\n", numerical_df[['Respiratory Rate', 'Temperature']].head())
    
    # Assign the largest value to missing entries in 'Shortest follow-up' and 'Follow-up'
    followup_cols = ['Shortest follow-up', 'Follow-up']
    for col in followup_cols:
        max_val = numerical_df[col].max()
        numerical_df[col].fillna(max_val, inplace=True)
        print(f"Assigned {max_val} to missing values in '{col}'.")
    
    # Verify the assignment
    print("\nUpdated Numerical Features with Assigned Values:\n", numerical_df[followup_cols].head())
    
    # Impute missing numerical features with mean (if any remaining missing values)
    numerical_imputer = SimpleImputer(strategy='mean')
    numerical_imputed = numerical_imputer.fit_transform(numerical_df)
    
    # Convert to DataFrame
    numerical_imputed_df = pd.DataFrame(numerical_imputed, columns=numerical_cols)
    
    # Convert binary categorical columns ('Yes'/'No') to 1/0
    binary_categorical_df = binary_categorical_df.replace({'Yes': 1, 'No': 0})
    
    # Handle non-binary categorical columns with One-Hot Encoding
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    non_binary_encoded = encoder.fit_transform(non_binary_categorical_df)
    encoded_feature_names = encoder.get_feature_names_out(non_binary_categorical_cols)
    non_binary_encoded_df = pd.DataFrame(non_binary_encoded, columns=encoded_feature_names)
    
    # Normalize numerical features
    scaler = StandardScaler()
    numerical_scaled = scaler.fit_transform(numerical_imputed_df)
    numerical_scaled_df = pd.DataFrame(numerical_scaled, columns=numerical_cols)

    # print("\nFirst 5 rows of 'numerical_scaled':\n", numerical_scaled_df.head())
    # print("\nFirst 5 rows of 'binary_categorical_df.values':\n", binary_categorical_df.values[:5])
    # print("\nFirst 5 rows of 'non_binary_encoded':\n", non_binary_encoded_df.head())
    # gender and race has been expanded into 11 total columns.
    
    # Combine all features: numerical, binary categorical, and one-hot encoded non-binary categorical
    features_processed = np.hstack((numerical_scaled, 
                                     binary_categorical_df.values, 
                                     non_binary_encoded))
    
    return features_processed, death_date, short_followup
"""

# Preprocess the data for graph generation.
# Drops rows without values in Oxygen_saturation column (missing too many values).
# Creates 3 .npy files
def preprocess_data(csv_file):
    # Load CSV data
    df = pd.read_csv(csv_file)
    
    # Drop 'XNATSessionID'
    df = df.drop(columns=['XNATSessionID'])
    print("Before dropping missing SpO2, number of datapoints: ", df.shape[0])
    
    # Extract outcome variables
    death_date = df['Death_Date'].map({'No': 0, 'Yes': 1}).values
    
    # Step 1: Replace empty strings with NaN
    df['Short_Followup_Less_Than_30_Days'] = df['Short_Followup_Less_Than_30_Days'].replace('', np.nan)
    
    # Step 2: Assign 'FALSE' to NaN values
    df['Short_Followup_Less_Than_30_Days'] = df['Short_Followup_Less_Than_30_Days'].fillna('FALSE')
    
    # Step 3: Standardize the entries (uppercase and strip whitespace)
    short_followup_standardized = df['Short_Followup_Less_Than_30_Days'].astype(str).str.strip().str.upper()
    
    # Step 4: Define the mapping dictionary
    mapping_dict = {
        'TRUE': 1,
        'FALSE': 0,
    }
    
    # Step 5: Map the standardized values to numerical values
    short_followup_mapped = short_followup_standardized.map(mapping_dict)
    
    # Convert to integer type
    short_followup = short_followup_mapped.astype(int).values
        
    # Drop outcome variables from features
    features_df = df.drop(columns=['Death_Date', 'Short_Followup_Less_Than_30_Days'])
    
    # Define categorical and numerical columns
    binary_categorical_cols = ['Altered Mental Status', 'COPD', 
                               'Chronic Heart Failure', 'Chronic Cancer']
    non_binary_categorical_cols = ['Gender', 'Race']
    numerical_cols = ['Age', 'Pulse', 'BP_Systolic', 'Respiratory Rate', 
                      'Temperature', 'Oxygen_saturation', 
                      'Shortest follow-up', 'Follow-up']
    
    # Extract categorical and numerical data
    binary_categorical_df = features_df[binary_categorical_cols]
    non_binary_categorical_df = features_df[non_binary_categorical_cols]
    numerical_df = features_df[numerical_cols]
    
    # Define a function to map 'No' to 0 and any other value to 1
    def map_binary_abnormality(value):
        if isinstance(value, str) and value.strip().lower() == 'no':
            return 0
        else:
            return 1
    
    # Apply the mapping to 'Respiratory Rate' and 'Temperature'
    numerical_df['Respiratory Rate'] = numerical_df['Respiratory Rate'].apply(map_binary_abnormality)
    numerical_df['Temperature'] = numerical_df['Temperature'].apply(map_binary_abnormality)
    
    # Verify the conversion
    print("Converted 'Respiratory Rate' and 'Temperature':\n", numerical_df[['Respiratory Rate', 'Temperature']].head())
    
    # Convert all numerical columns to numeric types, coercing errors to NaN
    for col in numerical_cols:
        numerical_df[col] = pd.to_numeric(numerical_df[col], errors='coerce')

    # Assign the largest value to missing entries in 'Shortest follow-up' and 'Follow-up'
    followup_cols = ['Shortest follow-up', 'Follow-up']
    for col in followup_cols:
        max_val = numerical_df[col].max()
        numerical_df[col].fillna(max_val, inplace=True)
        print(f"Assigned {max_val} to missing values in '{col}'.")
    
    # Verify the assignment
    print("\nUpdated Numerical Features with Assigned Values:\n", numerical_df[followup_cols].head())
    
    # Identify columns that still contain NaN after conversion
    remaining_nans = numerical_df.isna().sum()
    print("Remaining NaNs after conversion:\n", remaining_nans)

    # Drop rows where 'Oxygen_saturation' is missing
    df = df.dropna(subset=['Oxygen_saturation'])
    print("After dropping missing SpO2, number of datapoints: ", df.shape[0])
    
    # Handle missing values in numerical columns
    # For simplicity, we'll impute them with the mean of each column
    numerical_imputer = SimpleImputer(strategy='mean')
    numerical_imputed = numerical_imputer.fit_transform(numerical_df)
    
    # Convert to DataFrame
    numerical_imputed_df = pd.DataFrame(numerical_imputed, columns=numerical_cols)
    
    # Assign the largest value to missing entries in 'Shortest follow-up' and 'Follow-up'
    # Note: Since we've already imputed missing values, this step might be redundant.
    # If you still want to assign the max before imputation, ensure it's done before imputation.
    # Here, it's done after imputation, which might not be necessary.
    # If you prefer to assign the max first, move this step before imputation.
    
    # Drop outcome variables from features is already done earlier
    
    # Convert binary categorical columns ('Yes'/'No') to 1/0
    binary_categorical_df = binary_categorical_df.replace({'Yes': 1, 'No': 0})
    
    # Handle non-binary categorical columns with One-Hot Encoding
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    non_binary_encoded = encoder.fit_transform(non_binary_categorical_df)
    encoded_feature_names = encoder.get_feature_names_out(non_binary_categorical_cols)
    non_binary_encoded_df = pd.DataFrame(non_binary_encoded, columns=encoded_feature_names)
    
    # Normalize numerical features
    scaler = StandardScaler()
    numerical_scaled = scaler.fit_transform(numerical_imputed_df)
    numerical_scaled_df = pd.DataFrame(numerical_scaled, columns=numerical_cols)

    # print("\nFirst 5 rows of 'numerical_scaled':\n", numerical_scaled_df.head())
    # print("\nFirst 5 rows of 'binary_categorical_df.values':\n", binary_categorical_df.values[:5])
    # print("\nFirst 5 rows of 'non_binary_encoded':\n", non_binary_encoded_df.head())
    # gender and race has been expanded into 11 total columns.
    
    # Combine all features: numerical, binary categorical, and one-hot encoded non-binary categorical
    features_processed = np.hstack((numerical_scaled, 
                                     binary_categorical_df.values, 
                                     non_binary_encoded))
    
    return features_processed, death_date, short_followup

def main():
    
    # Preprocess the data
    features, death_date, short_followup = preprocess_data('GCN_data_all_demographics_GCN.csv')

    print("Data preprocessing completed.")

    # Optional: Save processed data
    np.save('features_processed.npy', features)
    np.save('death_date_processed.npy', death_date)
    print(short_followup)
    np.save('short_followup_processed.npy', short_followup)

if __name__ == "__main__":
    main()