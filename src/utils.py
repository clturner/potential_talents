from collections import Counter
import pandas as pd
import re

# clean words of extra puctuations and uneede chars in job_title feature
def clean_word(word):
    # Remove unwanted characters: (, ) | , !
    word = re.sub(r"[(),|!-]", "", word)
    
    # Remove "." unless it's part of an abbreviation (capital letter before and after)
    word = re.sub(r"(?<![A-Z])\.(?![A-Z])", "", word)

    return word

# get word frequencies from a specified column in a DataFrame
def get_word_frequencies(df, column_name):
    """
    Computes word frequencies from a specified column in a DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column_name (str): The name of the column containing text data.

    Returns:
    pd.DataFrame: A DataFrame with word frequencies, sorted in descending order.
    """
    
    # Combine all text entries into one string
    all_text = " ".join(df[column_name].astype(str))

    # Tokenize (split into words) and clean them
    words = [clean_word(word) for word in all_text.split() if clean_word(word).strip()]

    # Count word occurrences
    word_counts = Counter(words)

    # Convert to DataFrame and sort by frequency
    df_word_counts = pd.DataFrame(word_counts.items(), columns=["Word", "Count"])
    df_word_counts = df_word_counts.sort_values(by="Count", ascending=False).reset_index(drop=True)

    return df_word_counts


import re
import pandas as pd

def clean_location(location):
    # Handle NaN or empty values
    if pd.isna(location) or not location.strip():
        return "Unknown"

    # Dictionary to standardize country names and full location replacements
    replacements = {
        "Kanada": "Canada",
        "Amerika Birleşik Devletleri": "United States",
        "İzmir, Türkiye": "Izmir, Turkey",
        "USA": "United States",
    }

    # Apply replacements first
    location = replacements.get(location, location)

    # Replace '/' with ',' for consistency
    location = location.replace("/", ", ")

    # Remove 'Area' suffix and handle "Greater [City] Area"
    location = re.sub(r"\s*Area$", "", location)  # e.g., "Houston, Texas Area" → "Houston, Texas"
    location = re.sub(r"^Greater\s+", "", location)  # e.g., "Greater New York City" → "New York City"

    # Dictionary for adding missing state or country
    city_to_state = {
        # Cities without state/country
        "New York": "New York, New York, United States",
        "Houston": "Houston, Texas, United States",
        "Denton": "Denton, Texas, United States",
        "Atlanta": "Atlanta, Georgia, United States",
        "Chicago": "Chicago, Illinois, United States",
        "Austin": "Austin, Texas, United States",
        "San Francisco": "San Francisco, California, United States",
        "San Jose": "San Jose, California, United States",
        "Los Angeles": "Los Angeles, California, United States",
        "Lake Forest": "Lake Forest, California, United States",
        "Virginia Beach": "Virginia Beach, Virginia, United States",
        "Baltimore": "Baltimore, Maryland, United States",
        "Gaithersburg": "Gaithersburg, Maryland, United States",
        "Highland": "Highland, California, United States",
        "Milpitas": "Milpitas, California, United States",
        "Torrance": "Torrance, California, United States",
        "Long Beach": "Long Beach, California, United States",
        "Bridgewater": "Bridgewater, Massachusetts, United States",
        "Lafayette": "Lafayette, Indiana, United States",
        "Cape Girardeau": "Cape Girardeau, Missouri, United States",
        "Katy": "Katy, Texas, United States",
        "Izmir": "Izmir, Turkey",

        # Regions or metro areas
        "New York City": "New York, New York, United States",
        "San Francisco Bay": "San Francisco, California, United States",
        "Philadelphia": "Philadelphia, Pennsylvania, United States",
        "Boston": "Boston, Massachusetts, United States",
        "Atlanta": "Atlanta, Georgia, United States",
        "Chicago": "Chicago, Illinois, United States",
        "Los Angeles": "Los Angeles, California, United States",
        "Grand Rapids, Michigan": "Grand Rapids, Michigan, United States",
        "Dallas/Fort Worth": "Dallas, Texas, United States",
        "Raleigh-Durham, North Carolina": "Raleigh, North Carolina, United States",
        "Jackson, Mississippi": "Jackson, Mississippi, United States",
        "Monroe, Louisiana": "Monroe, Louisiana, United States",
        "Baton Rouge, Louisiana": "Baton Rouge, Louisiana, United States",
        "Myrtle Beach, South Carolina": "Myrtle Beach, South Carolina, United States",
        "Chattanooga, Tennessee": "Chattanooga, Tennessee, United States",
        "Kokomo, Indiana": "Kokomo, Indiana, United States",
        "Las Vegas, Nevada": "Las Vegas, Nevada, United States",
    }

    # Split the location into parts
    parts = [part.strip() for part in location.split(",")]

    # Case 1: Single value (could be city, state, or country)
    if len(parts) == 1:
        loc = parts[0]
        if loc in city_to_state:
            return city_to_state[loc]
        elif loc in replacements:
            return replacements[loc]
        else:
            # Assume it's a country or ambiguous place if not found
            return f"{loc}, Unknown" if loc not in ["United States", "Canada", "Turkey"] else loc

    # Case 2: Two parts (e.g., "Houston, Texas" or "Izmir, Turkey")
    elif len(parts) == 2:
        city, second = parts
        if city in city_to_state:
            return city_to_state[city]
        # If second part is a state or country, format accordingly
        if second in ["Texas", "California", "Georgia", "Illinois", "Virginia", "Maryland", "Massachusetts", "Indiana", "Missouri", "Nevada", "New York"]:
            return f"{city}, {second}, United States"
        elif second in ["Turkey", "Canada", "United States"]:
            return f"{city}, {second}"
        else:
            return f"{city}, {second}, United States"  # Default to US if unclear

    # Case 3: Three or more parts (e.g., "New York, New York, United States")
    else:
        if "United States" in parts or "Canada" in parts or "Turkey" in parts:
            return ", ".join(parts[:3])  # Keep first three parts if country is present
        else:
            return f"{', '.join(parts[:2])}, United States"  # Assume US if no country

    return location  # Fallback



