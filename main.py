import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Set the title of the app
st.title("Data Upload and Preprocessing App")

# Step 1: File uploader for CSV files
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Step 2: Load the dataset
    df = pd.read_csv(uploaded_file)

    # Step 3: Display the dataframe
    st.write("### Data Preview")
    st.dataframe(df.head())  # Show the first few rows of the dataframe

    # Step 4: Display basic statistics
    st.write("### Data Statistics")
    st.write(df.describe())  # Show statistical summary of the data

    # Step 5: Display shape of the data
    st.write("### Data Shape")
    st.write(f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")

    # Step 6: Handle missing values
    st.write("### Handle Missing Values")
    missing_cols = df.columns[df.isnull().any()]
    st.write("Columns with missing values:", missing_cols.tolist())

    if st.checkbox("Fill missing values with mean for numerical columns"):
        for col in df.select_dtypes(include=['float64', 'int']).columns:
            df[col].fillna(df[col].mean(), inplace=True)

    if st.checkbox("Fill missing values with mode for categorical columns"):
        for col in df.select_dtypes(include=['object']).columns:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # Using SimpleImputer for missing values
    if st.checkbox("Use SimpleImputer for missing values"):
        imputer = SimpleImputer(strategy='mean')  # or 'most_frequent' for categorical
        for col in df.columns:
            if df[col].isnull().any():
                df[col] = imputer.fit_transform(df[[col]])

    # Display updated data after handling missing values
    st.write("### Data After Handling Missing Values")
    st.dataframe(df.head())

    # Step 7: Removing Duplicates
    if st.checkbox("Remove duplicate rows"):
        df.drop_duplicates(inplace=True)
        st.success("Duplicate rows removed.")

    # Display updated data after removing duplicates
    st.write("### Data After Removing Duplicates")
    st.dataframe(df.head())

    # Step 8: Encoding categorical variables
    st.write("### Encode Categorical Variables")
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    if st.checkbox("Encode categorical variables using Label Encoding"):
        le = LabelEncoder()
        for col in categorical_cols:
            df[col] = le.fit_transform(df[col])

    # Display updated data after encoding
    st.write("### Data After Encoding")
    st.dataframe(df.head())

    # Step 9: Feature Scaling
    st.write("### Feature Scaling")
    if st.checkbox("Scale numerical features"):
        scaler = StandardScaler()
        numerical_cols = df.select_dtypes(include=['float64', 'int']).columns.tolist()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Display updated data after scaling
    st.write("### Data After Scaling")
    st.dataframe(df.head())

    # Step 10: Data Validation
    st.write("### Data Quality Checks")

    # Check for duplicates
    if st.checkbox("Check for duplicate rows"):
        duplicate_count = df.duplicated().sum()
        st.write(f"Number of duplicate rows: {duplicate_count}")

    # Check for inconsistent data types
    if st.checkbox("Check data types"):
        st.write("Data Types:")
        st.write(df.dtypes)

    # Check for value ranges in numerical columns
    if st.checkbox("Check value ranges for numerical columns"):
        for col in df.select_dtypes(include=['float64', 'int']).columns:
            st.write(f"### {col} Range")
            st.write(f"Min: {df[col].min()}, Max: {df[col].max()}")

    # Step 11: Final Dataset
    st.write("### Final Preprocessed Dataset")
    st.dataframe(df)

# Run this app with: streamlit run app.py
