import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from itertools import chain 

def compute_similarity_to_starred(df, starred_ids, keyword, column_name="similarity_to_starred", inplace=False):
    """
    Compute similarity of each candidate's job title to the ideal embedding of starred candidates.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing 'id' and 'job_title' columns.
    - starred_ids (list): List of candidate IDs that are starred.
    - keyword (str): Keyword to use as fallback if no starred candidates exist.
    - column_name (str): Name of the column to store similarity scores (default: "similarity_to_starred").
    - inplace (bool): If True, modify the input DataFrame; if False, return a new DataFrame (default: False).
    
    Returns:
    - pd.DataFrame: DataFrame with the new similarity column (if inplace=False).
    """

    # Flatten starred_ids to ensure scalars
    starred_ids = list(chain.from_iterable([x] if not isinstance(x, (list, tuple)) else x for x in starred_ids))

    # Load BERT model once
    BERT_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Step 1: Fetch embeddings of starred candidates
    starred_embeddings = []
    for candidate_id in starred_ids:
        # Find job_title by matching ID (using pandas filtering for efficiency)
        match = df[df["id"] == candidate_id]
        if not match.empty:
            job_title = match["job_title"].iloc[0]
            embedding = BERT_model.encode(job_title)
            starred_embeddings.append(embedding)

    # Step 2: Compute the "ideal" embedding
    if starred_embeddings:
        # Average the embeddings efficiently using numpy
        ideal_embedding = np.mean(starred_embeddings, axis=0)
    else:
        # Fallback to keyword embedding
        ideal_embedding = BERT_model.encode(keyword)

    # Step 3: Compute similarity scores for all candidates
    new_fit_scores = []
    for job_title in df["job_title"]:
        job_embedding = BERT_model.encode(job_title)
        # Compute cosine similarity (2D arrays for sklearn)
        similarity = cosine_similarity([job_embedding], [ideal_embedding])[0][0]
        new_fit_scores.append(similarity)

    # Step 4: Handle output based on inplace parameter
    if inplace:
        df[column_name] = new_fit_scores
        return None  # No return value when modifying in place
    else:
        # Return a copy of the DataFrame with the new column
        df_new = df.copy()
        df_new[column_name] = new_fit_scores
        return df_new

# Example usage:
# Assuming df_combined, starred_ids, and keyword are defined
# Option 1: Modify df_combined in place
# compute_similarity_to_starred(df_combined, starred_ids, keyword, inplace=True)
# print(df_combined[["id", "job_title", "similarity_to_starred"]].head())

# Option 2: Return a new DataFrame
# df_result = compute_similarity_to_starred(df_combined, starred_ids, keyword)
# print(df_result[["id", "job_title", "similarity_to_starred"]].head())



import pandas as pd
import numpy as np

def compute_weighted_geometric_mean(df, col1, col2, w1, w2, output_col="fit_combined", inplace=False, check_weights=True):
    """
    Compute the weighted geometric mean of two columns and add it as a new feature.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing the input columns.
    - col1 (str): Name of the first column (e.g., "fit_BERT").
    - col2 (str): Name of the second column (e.g., "similarity_to_starred").
    - w1 (float): Weight for the first column (e.g., 0.4).
    - w2 (float): Weight for the second column (e.g., 0.6).
    - output_col (str): Name of the output column (default: "fit_combined").
    - inplace (bool): If True, modify the input DataFrame; if False, return a new DataFrame (default: False).
    - check_weights (bool): If True, warn if weights don’t sum to 1 (default: True).
    
    Returns:
    - pd.DataFrame or None: New DataFrame with the combined feature (if inplace=False), None otherwise.
    """
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if col1 not in df.columns or col2 not in df.columns:
        raise KeyError(f"Columns '{col1}' and/or '{col2}' not found in DataFrame")
    if not isinstance(w1, (int, float)) or not isinstance(w2, (int, float)):
        raise TypeError("Weights w1 and w2 must be numeric")
    
    # Optional weight check
    if check_weights and abs(w1 + w2 - 1.0) > 1e-6:
        print(f"Warning: Weights w1 ({w1}) + w2 ({w2}) = {w1 + w2}, do not sum to 1")

    # Compute weighted geometric mean: (col1^w1) * (col2^w2)
    fit_combined_scores = (df[col1] ** w1) * (df[col2] ** w2)

    # Handle output
    if inplace:
        df[output_col] = fit_combined_scores
        return None
    else:
        df_new = df.copy()
        df_new[output_col] = fit_combined_scores
        return df_new

