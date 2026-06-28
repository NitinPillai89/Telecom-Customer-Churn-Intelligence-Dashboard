import os
import pandas as pd
import numpy as np

def preprocess_telecom_data(input_path, output_path):
    """
    Performs robust data cleaning, type coercion, and binary categorical 
    mapping arrays on the raw Telecom Customer Churn dataset.
    """
    print("Initializing Data Preprocessing Pipeline...")
    
    # 1. Check if the input file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Source file not found at: {input_path}")
        
    # 2. Ingest the raw dataset
    print(f"Ingesting raw file from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Successfully loaded dataset with shape: {df.shape}")
    
    # 3. Clean string white-spaces from columns to avoid key mismatches
    df.columns = df.columns.str.strip()
    
    # 4. Handle 'Total Charges' type coercion from text string to float
    # Employs errors='coerce' to force blank spaces/malformed text into NaN
    if 'Total Charges' in df.columns:
        print("Coercing 'Total Charges' to numeric floating-point values...")
        df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
    else:
        # Fallback to older column name string if detected
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # 5. Handle missing values using dropna()
    # Isolates structural shards to prevent data skewing or calculation drops
    print("Executing dropna() to eliminate null values and preserve data integrity...")
    initial_rows = len(df)
    df = df.dropna()
    cleansed_rows = len(df)
    print(f"Dropped {initial_rows - cleansed_rows} records with incomplete metrics.")

    # 6. Convert categorical target variable 'Churn' into a numeric binary outcome flag
    # Maps 'Yes' to 1 (Churned) and 'No' to 0 (Stayed) for statistical calculations
    if 'Churn' in df.columns:
        print("Mapping categorical 'Churn' field to binary outcome arrays...")
        df['Churn Value'] = df['Churn'].map({'Yes': 1, 'No': 0})
    elif 'ChurnStatus' in df.columns:
        df['Churn Value'] = df['ChurnStatus'].map({'Yes': 1, 'No': 0})

    # 7. Validate that output columns conform exactly to the Star Schema
    print("Verifying final data schema transformations...")
    print(df[['CustomerID', 'Tenure in Months', 'Monthly Charge', 'Total Charges', 'Churn Value']].head())
    
    # 8. Export clean dataset to CSV
    df.to_csv(output_path, index=False)
    print(f"Preprocessing complete! Cleaned dataset exported to: {output_path}")
    print(f"Final clean shape: {df.shape}")

if __name__ == "__main__":
    # Define local system execution paths
    # Update these file paths to match your local project workspace settings
    RAW_DATA_PATH = "telecom_customer_churn.csv"
    CLEAN_DATA_PATH = "cleaned_telecom_customer_churn.csv"
    
    try:
        preprocess_telecom_data(RAW_DATA_PATH, CLEAN_DATA_PATH)
    except Exception as e:
        print(f"Pipeline execution failed! Error details: {str(e)}")