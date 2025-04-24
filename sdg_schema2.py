import pandas as pd
import json
import os
import numpy as np
from collections import defaultdict
import re
from datetime import datetime
import math # Import math for isnan check
from itertools import combinations, product # Import combinations and product
import inspect # Import inspect for checking faker args later if needed
import google.generativeai as genai # Import the Gemini library
import warnings # To suppress specific warnings if needed

# Configuration
INPUT_CSV = "customer_data.csv" # Example input CSV
OUTPUT_SCHEMA_JSON = "enhanced_schema_v3.json" # Output file name
METADATA_JSON = "metadata.json" # Path to the external metadata file (REQUIRED)
TEMP_RAW_RESPONSE_FILE = "temp_raw_gemini_response_v3.txt" # File to save raw LLM response for debugging

# Thresholds for relationship detection
MIN_UNIQUE_VALUES_FOR_RELATIONSHIP_SOURCE = 5 # Minimum number of unique non-null values for a column to be considered a relationship source
MIN_UNIQUE_VALUE_RATIO_FOR_RELATIONSHIP_SOURCE = 0.05 # Minimum ratio of unique non-null values to total non-null values
RELATIONSHIP_DETECTION_THRESHOLD = 0.999 # Threshold for functional dependency and value relationship consistency

# Thresholds for type detection
DATETIME_CONVERSION_THRESHOLD = 0.80 # Minimum proportion of non-null values that must convert to datetime to be flagged as datetime
CATEGORICAL_UNIQUE_RATIO_THRESHOLD = 0.5 # Maximum ratio of unique values for a column to be considered categorical
CATEGORICAL_MAX_UNIQUE_VALUES = 100 # Maximum number of unique values for a column to be considered categorical (increased)


# --- Gemini API Setup ---
# This will be configured in the main function
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
gemini_model = None # Global variable to hold the configured Gemini model
# --- End Gemini API Setup ---


def load_data(csv_file_path):
    """Loads data from a CSV file."""
    if not os.path.exists(csv_file_path):
        print(f"Error: Input data file not found: {csv_file_path}")
        return None
    try:
        # Attempt to read with UTF-8, fallback to latin-1
        try:
            # Keep original data types if possible, esp. for detecting numeric strings vs actual numbers
            df = pd.read_csv(csv_file_path, encoding='utf-8', low_memory=False)
        except UnicodeDecodeError:
            print("UTF-8 decoding failed, trying latin-1...")
            df = pd.read_csv(csv_file_path, encoding='latin-1', low_memory=False)

        print(f"Successfully loaded data from {csv_file_path}. Shape: {df.shape}")
        # Store original dtypes before any potential conversion
        original_dtypes = df.dtypes.to_dict()
        return df, original_dtypes
    except Exception as e:
        print(f"Error loading data from {csv_file_path}: {e}")
        return None, None

def load_external_metadata(json_file_path):
    """
    Loads external metadata from a JSON file.
    Expects structure like: { "table_name": { "columns": [ {"Column_name": "...", "Key_type": "...", ...} ] } }
    Returns a dictionary keyed by uppercase column names.
    """
    if not os.path.exists(json_file_path):
        print(f"Error: Metadata file not found at {json_file_path}. Metadata is required.")
        return None
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        print(f"Successfully loaded external metadata from {json_file_path}.")

        column_metadata = {}
        # Iterate through tables in the metadata (assuming there might be multiple, use first one found with columns)
        table_key = next((key for key, value in metadata.items() if isinstance(value, dict) and "columns" in value), None)

        if table_key and "columns" in metadata[table_key]:
            for col_info in metadata[table_key].get("columns", []):
                 if isinstance(col_info, dict) and "Column_name" in col_info:
                      # Store metadata keyed by uppercase column name for consistency
                      col_name_upper = col_info["Column_name"].upper()
                      column_metadata[col_name_upper] = col_info
            print(f"Extracted metadata for {len(column_metadata)} columns from table '{table_key}'.")
            return column_metadata
        else:
             print(f"Error: Could not find a valid table structure with 'columns' in {json_file_path}.")
             return None

    except Exception as e:
        print(f"Error loading or parsing external metadata from {json_file_path}: {e}")
        return None

def calculate_basic_stats(df, original_dtypes):
    """
    Calculates basic statistics for each column.
    Uses stricter datetime detection and refined categorical detection.
    Args:
        df (pd.DataFrame): Input DataFrame (assuming uppercase column names).
        original_dtypes (dict): Dictionary mapping original column names to their pandas dtypes.
    Returns:
        dict: Dictionary of statistics keyed by uppercase column name.
    """
    print("\nCalculating basic statistics...")
    stats = {}
    # Assume df columns are already uppercase from main()
    df_columns_upper = df.columns.tolist()
    # Need original case map if original_dtypes keys are not uppercase
    original_case_map_from_dtypes = {name.upper(): name for name in original_dtypes.keys()}


    for col_upper in df_columns_upper:
        # Get original case name if needed for accessing original_dtypes
        # col_original_case = original_case_map_from_dtypes.get(col_upper, col_upper)
        # Use uppercase name directly for df access as columns were converted in main()
        col_original_case = col_upper # Use uppercase name as df columns are now uppercase

        col_stats = {}
        col_series = df[col_original_case]
        original_dtype = original_dtypes.get(original_case_map_from_dtypes.get(col_upper)) # Get original dtype using original case name

        col_stats["original_dtype"] = str(original_dtype) if original_dtype else 'unknown'
        col_stats["null_count"] = col_series.isnull().sum()
        col_stats["total_count"] = len(df)
        col_stats["null_percentage"] = (col_stats["null_count"] / col_stats["total_count"]) * 100 if col_stats["total_count"] > 0 else 0
        col_stats["unique_count"] = col_series.nunique()

        non_null_series = col_series.dropna()
        num_non_null = len(non_null_series)

        col_stats["is_numeric"] = False
        col_stats["is_datetime"] = False
        col_stats["is_categorical"] = False # Initialize categorical flag

        if num_non_null > 0:
            # --- Sample Values ---
            sample_size = min(20, num_non_null) # Increased sample size to 20
            col_stats["sample_values"] = non_null_series.sample(sample_size).tolist()

            # --- Numeric Stats ---
            # Attempt numeric conversion only if original type suggests it or object type
            if pd.api.types.is_numeric_dtype(original_dtype) or pd.api.types.is_object_dtype(original_dtype):
                numeric_series = pd.to_numeric(non_null_series, errors='coerce').dropna()
                if not numeric_series.empty:
                     # Check if a significant portion could be converted if original was object
                     if pd.api.types.is_object_dtype(original_dtype) and len(numeric_series) / num_non_null < 0.5:
                          pass # Not primarily numeric if less than half converted from object
                     else:
                          col_stats["min"] = numeric_series.min()
                          col_stats["max"] = numeric_series.max()
                          col_stats["mean"] = numeric_series.mean()
                          col_stats["median"] = numeric_series.median()
                          col_stats["std_dev"] = numeric_series.std()
                          col_stats["is_numeric"] = True # Flag for relationship detection
            # Ensure numeric stats fields exist even if not numeric
            if not col_stats["is_numeric"]:
                 col_stats["min"] = None
                 col_stats["max"] = None
                 col_stats["mean"] = None
                 col_stats["median"] = None
                 col_stats["std_dev"] = None


            # --- Datetime Stats (Stricter Check) ---
            # Only attempt if not already confirmed numeric and original type is object/string or datetime-like
            if not col_stats["is_numeric"] and \
               (pd.api.types.is_object_dtype(original_dtype) or pd.api.types.is_datetime64_any_dtype(original_dtype)):
                datetime_series = None
                try:
                    # Suppress UserWarning about inferring datetime format
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        temp_datetime_series = pd.to_datetime(non_null_series, errors='coerce')

                    num_converted = temp_datetime_series.notna().sum()

                    # Check if a high proportion converted successfully
                    if num_non_null > 0 and (num_converted / num_non_null) >= DATETIME_CONVERSION_THRESHOLD:
                        datetime_series = temp_datetime_series.dropna()
                        if not datetime_series.empty:
                            col_stats["min_date"] = datetime_series.min().isoformat()
                            col_stats["max_date"] = datetime_series.max().isoformat()
                            col_stats["is_datetime"] = True # Flag for relationship detection
                except Exception: # Catch any other unexpected errors during conversion
                    pass # is_datetime remains False
            # Ensure date stats fields exist even if not datetime
            if not col_stats["is_datetime"]:
                col_stats["min_date"] = None
                col_stats["max_date"] = None


            # --- Categorical Stats (Refined Check) ---
            # Consider categorical if:
            # 1. Not numeric and not datetime.
            # 2. Original type is object, category, or potentially boolean.
            # 3. Unique ratio is below threshold OR absolute unique count is below threshold.
            unique_ratio = col_stats["unique_count"] / num_non_null if num_non_null > 0 else 0
            is_likely_type = pd.api.types.is_object_dtype(original_dtype) or \
                             pd.api.types.is_categorical_dtype(original_dtype) or \
                             pd.api.types.is_bool_dtype(original_dtype)

            if not col_stats["is_numeric"] and not col_stats["is_datetime"] and is_likely_type and \
               (unique_ratio < CATEGORICAL_UNIQUE_RATIO_THRESHOLD or col_stats["unique_count"] < CATEGORICAL_MAX_UNIQUE_VALUES) and \
               col_stats["unique_count"] > 0:

                 col_stats["is_categorical"] = True # Set categorical flag
                 value_counts = non_null_series.value_counts(normalize=True)
                 # Limit number of categories shown in stats
                 top_categories = value_counts.head(20)
                 col_stats["categories"] = [{"value": str(index), "percentage": round(value * 100, 2)} for index, value in top_categories.items()]
            else:
                 col_stats["categories"] = []


        # Store stats using uppercase column name as key
        stats[col_upper] = col_stats
        print(f"  '{col_upper}': Null%={col_stats['null_percentage']:.1f}, Unique={col_stats['unique_count']}, "
              f"Numeric={col_stats['is_numeric']}, Datetime={col_stats['is_datetime']}, Categorical={col_stats['is_categorical']}")
    print("Basic statistics calculation complete.")
    return stats

def detect_key_types(df_columns, external_metadata):
    """
    Detects primary and foreign keys based *only* on the external metadata file.
    Args:
        df_columns (list): List of column names from the DataFrame (uppercase).
        external_metadata (dict): Dictionary of metadata keyed by uppercase column name.
    Returns:
        dict: Dictionary mapping uppercase column names to key types ('Primary Key', 'Foreign Key', 'None').
    """
    print("\nDetecting key types from external metadata...")
    key_types = {}
    if not external_metadata:
        print("  Warning: External metadata not provided or empty. Cannot determine key types.")
        # Default all columns present in the dataframe to 'None'
        for col_upper in df_columns:
             key_types[col_upper] = "None"
        return key_types

    found_pk = False
    for col_upper in df_columns:
        col_meta = external_metadata.get(col_upper)
        if col_meta and "Key_type" in col_meta:
            key_type_str = str(col_meta["Key_type"]).strip().lower()
            if key_type_str == "primary key":
                key_types[col_upper] = "Primary Key"
                print(f"  '{col_upper}': Primary Key (from metadata)")
                found_pk = True
            elif key_type_str == "foreign key":
                key_types[col_upper] = "Foreign Key"
                print(f"  '{col_upper}': Foreign Key (from metadata)")
            elif key_type_str in ["null", "none", ""]:
                 key_types[col_upper] = "None"
            else:
                 # If metadata provides a value other than known types, treat as None for now
                 print(f"  Warning: Unknown Key_type '{col_meta['Key_type']}' for column '{col_upper}' in metadata. Treating as 'None'.")
                 key_types[col_upper] = "None"
        else:
            # If column exists in data but not in metadata, or metadata lacks Key_type
            key_types[col_upper] = "None"
            if col_upper not in external_metadata:
                 print(f"  Warning: Column '{col_upper}' found in data but not in metadata. Key type set to 'None'.")
            elif col_meta and "Key_type" not in col_meta: # Check col_meta exists before accessing Key_type
                 print(f"  Warning: Metadata for column '{col_upper}' lacks 'Key_type'. Key type set to 'None'.")
            elif not col_meta: # Handle case where col exists in data but not in metadata dict
                 print(f"  Warning: Column '{col_upper}' found in data but not in metadata dict. Key type set to 'None'.")


    # Ensure all columns from the dataframe have an entry
    for col_upper in df_columns:
        if col_upper not in key_types:
            key_types[col_upper] = "None"

    if not found_pk:
        print("  Warning: No Primary Key explicitly defined in the provided metadata for the loaded columns.")

    print("Key type detection from metadata complete.")
    return key_types

def infer_column_info_with_llm(col_name_upper, stats, key_type, external_metadata_info=None):
    """
    Infers data type, Faker provider, and args for a single column using the LLM.
    Includes external metadata info in the prompt if available.
    Suggests structured unique providers like unique.random_int for Primary Keys instead of uuid4.
    Requires the global `gemini_model` to be configured.
    Implements robust JSON extraction and error handling.
    Saves raw LLM response to a temporary file for debugging.
    Args:
        col_name_upper (str): The uppercase column name.
        stats (dict): Basic statistics for this column.
        key_type (str): Key type ('Primary Key', 'Foreign Key', 'None').
        external_metadata_info (dict, optional): Specific metadata for this column. Defaults to None.
    Returns:
        dict: Dictionary with 'data_type', 'faker_provider', 'faker_args'.
    """
    print(f"  Inferring info for column '{col_name_upper}' using LLM...")

    global gemini_model # Access the global configured model

    if gemini_model is None:
        print(f"  Skipping LLM inference for '{col_name_upper}': Gemini model not configured.")
        return {"data_type": "unknown", "faker_provider": None, "faker_args": {}, "data_domain": None, "constraints_description": None}

    # --- Convert NumPy/Pandas types in stats to Python types for JSON serialization ---
    cleaned_stats = {}
    for key, value in stats.items():
        if isinstance(value, (np.int64, np.int32)):
            cleaned_stats[key] = int(value)
        elif isinstance(value, (np.float64, np.float32)):
            cleaned_stats[key] = float(value) if not (np.isnan(value) or np.isinf(value)) else None
        elif isinstance(value, pd.Timestamp):
             cleaned_stats[key] = value.isoformat()
        elif isinstance(value, (np.bool_, bool)): # Handle boolean types
             cleaned_stats[key] = bool(value)
        elif isinstance(value, dict):
             cleaned_nested_dict = {}
             for k, v in value.items():
                  if isinstance(v, (np.int64, np.int32)): cleaned_nested_dict[k] = int(v)
                  elif isinstance(v, (np.float64, np.float32)): cleaned_nested_dict[k] = float(v) if not (np.isnan(v) or np.isinf(v)) else None
                  elif isinstance(v, pd.Timestamp): cleaned_nested_dict[k] = v.isoformat()
                  elif isinstance(v, (np.bool_, bool)): cleaned_nested_dict[k] = bool(v)
                  else: cleaned_nested_dict[k] = v # Keep other types as is (e.g., string)
             cleaned_stats[key] = cleaned_nested_dict
        elif isinstance(value, list):
             cleaned_list = []
             for item in value:
                  if isinstance(item, (np.int64, np.int32)): cleaned_list.append(int(item))
                  elif isinstance(item, (np.float64, np.float32)): cleaned_list.append(float(item) if not (np.isnan(item) or np.isinf(item)) else None)
                  elif isinstance(item, pd.Timestamp): cleaned_list.append(item.isoformat())
                  elif isinstance(item, (np.bool_, bool)): cleaned_list.append(bool(item))
                  elif isinstance(item, dict):
                      cleaned_nested_dict_in_list = {}
                      for k_list, v_list in item.items():
                           if isinstance(v_list, (np.int64, np.int32)): cleaned_nested_dict_in_list[k_list] = int(v_list)
                           elif isinstance(v_list, (np.float64, np.float32)): cleaned_nested_dict_in_list[k_list] = float(v_list) if not (np.isnan(v_list) or np.isinf(v_list)) else None
                           elif isinstance(v_list, pd.Timestamp): cleaned_nested_dict_in_list[k_list] = v_list.isoformat()
                           elif isinstance(v_list, (np.bool_, bool)): cleaned_nested_dict_in_list[k_list] = bool(v_list)
                           else: cleaned_nested_dict_in_list[k_list] = v_list
                      cleaned_list.append(cleaned_nested_dict_in_list)
                  else:
                       cleaned_list.append(item) # Keep other types as is (e.g., string)
             cleaned_stats[key] = cleaned_list
        else:
            # Ensure other types like strings are kept
            cleaned_stats[key] = value
    # --- End Conversion ---


    # Construct a detailed prompt for the LLM
    prompt = f"""
Analyze the following information about a column named '{col_name_upper}' from a dataset.
Based on the column name, key type, statistics, sample values, and potentially external metadata, suggest:
1.  The most appropriate data type ('integer', 'float', 'numerical', 'datetime', 'boolean', 'categorical', 'text', 'unknown').
2.  A suitable Faker provider name (e.g., 'name', 'random_int', 'date_time_this_century', 'unique.random_int', 'regexify', 'company'). If suggesting 'regexify', provide a simple regex pattern based on sample values if possible.
    *IMPORTANT*: If the Key Type is 'Primary Key':
        - If the data appears numeric (based on stats like min/max/mean), strongly prefer 'unique.random_int' or similar numeric unique provider. Define appropriate 'min'/'max' in faker_args.
        - If the data appears to be strings with a pattern, suggest 'unique.regexify' with a pattern in faker_args.
        - Otherwise, suggest a suitable unique string provider like 'unique.pystr' or let Faker handle uniqueness via the generation script if a standard provider is chosen. Avoid 'uuid4'.
3.  Appropriate arguments (`faker_args`) for the suggested Faker provider, in a JSON object format (e.g., {{"min": 0, "max": 100}}, {{"elements": ["A", "B"]}}, {{"start_date": "{cleaned_stats.get('min_date', 'YYYY-MM-DD')}", "end_date": "{cleaned_stats.get('max_date', 'YYYY-MM-DD')}"}}, {{"pattern": "regex_pattern"}}, or empty {{}} if no args}}). Try to use min/max/date stats to inform args.
4.  (Optional) A general data domain category (e.g., 'Customer Info', 'Financial Transaction', 'Product Details', 'Date/Time', 'Identifier', 'Address', 'Geography', 'Internal ID'). Respond with null if unsure.
5.  (Optional) A brief, human-readable description of any constraints implied by the data (e.g., "Must be unique", "Date must be after 2020", "References another table"). Respond with null if none obvious.

Column Name: {col_name_upper}
Key Type: {key_type}
External Metadata: {json.dumps(external_metadata_info, indent=2, default=str)}
Statistics: {json.dumps(cleaned_stats, indent=2, default=str)}

Provide the response as a single JSON object with the keys 'data_type', 'faker_provider', 'faker_args', 'data_domain', 'constraints_description'.
Respond ONLY with the JSON object, no conversational text or explanations.
"""
    # print(f"--- LLM Prompt for {col_name_upper} ---\n{prompt}\n------------------------") # Uncomment for debugging prompts

    default_response = {
        "data_type": "unknown",
        "faker_provider": None,
        "faker_args": {},
        "data_domain": None,
        "constraints_description": None
    }

    try:
        # Call the actual Gemini model
        response = gemini_model.generate_content(prompt)
        llm_response_text = response.text

        # --- Debugging: Save Raw Response to File ---
        try:
            with open(TEMP_RAW_RESPONSE_FILE, 'a', encoding='utf-8') as f:
                 f.write(f"\n--- Response for Column: {col_name_upper} ---\n")
                 f.write(llm_response_text)
                 f.write("\n-------------------------------------\n")
        except Exception as e:
            print(f"  Error saving raw response for '{col_name_upper}' to file: {e}")
        # --- End Debugging ---

        # Robustly extract JSON from the LLM response text
        json_match = re.search(r'```json\s*(\{.*?\})\s*```|\{.*?\}', llm_response_text, re.DOTALL | re.IGNORECASE)
        if json_match:
            json_string = json_match.group(1) if json_match.group(1) else json_match.group(0)
            try:
                llm_response = json.loads(json_string)
                if isinstance(llm_response, dict) and \
                   'data_type' in llm_response and \
                   'faker_provider' in llm_response and \
                   'faker_args' in llm_response:
                    llm_response.setdefault('data_domain', None)
                    llm_response.setdefault('constraints_description', None)

                    # --- Post-processing for Primary Key Provider ---
                    # If LLM didn't suggest a 'unique.*' provider for a PK, add 'unique.' prefix if suitable
                    if key_type == "Primary Key" and \
                       isinstance(llm_response['faker_provider'], str) and \
                       not llm_response['faker_provider'].startswith('unique.'):
                           # Check if the base provider is suitable for unique wrapping
                           base_provider = llm_response['faker_provider']
                           # Avoid wrapping providers that manage their own complex state or don't make sense as unique
                           if base_provider not in ['profile', 'json', 'file_path', 'image_url', 'password']:
                                print(f"  Note: Wrapping suggested provider '{base_provider}' with 'unique.' for Primary Key '{col_name_upper}'.")
                                llm_response['faker_provider'] = f"unique.{base_provider}"

                    print(f"  LLM inferred: Type='{llm_response['data_type']}', Provider='{llm_response['faker_provider']}', Args={llm_response['faker_args']}, Domain='{llm_response['data_domain']}'")
                    return llm_response
                else:
                    print(f"  Warning: LLM returned valid JSON but incomplete structure for '{col_name_upper}'. Falling back to default.")
                    return default_response
            except json.JSONDecodeError as json_err:
                print(f"  Warning: Could not decode JSON from LLM response for '{col_name_upper}'. Error: {json_err}. Response text: \n{llm_response_text}\n Falling back to default.")
                return default_response
        else:
            print(f"  Warning: Could not find a JSON object pattern in LLM response for '{col_name_upper}'. Response text: \n{llm_response_text}\n Falling back to default.")
            return default_response

    except Exception as e:
        print(f"  Error calling LLM for inference on '{col_name_upper}': {e}. Falling back to default.")
        return default_response

def detect_column_correlations(df, stats, key_types):
    """
    Detects column relationships (Functional Dependency, One-to-One, Value Relationships, Temporal Relationships).
    Prioritizes One-to-One over Functional Dependency.
    Excludes Pearson Correlation and Formulas.
    Uses strict inequalities for value/temporal relationships.
    Requires stats (for uniqueness, numeric/datetime flags) and key_types.
    Args:
        df (pd.DataFrame): The input dataframe (assuming uppercase column names).
        stats (dict): Dictionary of calculated statistics keyed by uppercase column name.
        key_types (dict): Dictionary of key types keyed by uppercase column name.
    Returns:
        dict: Dictionary of relationships keyed by uppercase column name.
    """
    print("\nDetecting column relationships (FD, O2O, Value, Temporal)...")
    relationships = defaultdict(list) # Use defaultdict to easily append relationships
    # Get column names in original case for df access, and uppercase for stats/key_types access
    # original_case_map = {col.upper(): col for col in df.columns} # Not needed if df cols are uppercase
    all_cols_upper = df.columns.tolist() # df columns assumed to be uppercase

    # Identify numeric and datetime columns from stats (using uppercase keys)
    numeric_cols_upper = [col for col in all_cols_upper if stats.get(col, {}).get("is_numeric")]
    datetime_cols_upper = [col for col in all_cols_upper if stats.get(col, {}).get("is_datetime")]
    print(f"  Identified {len(numeric_cols_upper)} numeric columns and {len(datetime_cols_upper)} datetime columns.")

    def meets_uniqueness_criteria(col_name_upper, stats):
        """Checks if a column meets the minimum uniqueness criteria to be a relationship source."""
        col_stats = stats.get(col_name_upper, {})
        unique_count = col_stats.get("unique_count", 0)
        total_non_null = col_stats.get("total_count", 0) - col_stats.get("null_count", 0)

        if total_non_null == 0:
            return False # Cannot be a source if all values are null

        unique_ratio = unique_count / total_non_null if total_non_null > 0 else 0

        meets_criteria = unique_count >= MIN_UNIQUE_VALUES_FOR_RELATIONSHIP_SOURCE and \
                         unique_ratio >= MIN_UNIQUE_VALUE_RATIO_FOR_RELATIONSHIP_SOURCE
        return meets_criteria

    print(f"  Applying uniqueness thresholds for sources: Min Unique Values = {MIN_UNIQUE_VALUES_FOR_RELATIONSHIP_SOURCE}, Min Unique Ratio = {MIN_UNIQUE_VALUE_RATIO_FOR_RELATIONSHIP_SOURCE}")

    # --- Functional Dependency (A -> B) Detection ---
    print("  Checking for Functional Dependencies...")
    functional_dependencies_detected = defaultdict(list) # Temporarily store FD detections {source_upper: [target_upper, ...]}

    for col1_upper in all_cols_upper:
        if not meets_uniqueness_criteria(col1_upper, stats):
             continue # Skip if col1 doesn't meet uniqueness criteria as a source

        # col1_original = original_case_map[col1_upper] # df cols are uppercase

        for col2_upper in all_cols_upper:
            if col1_upper == col2_upper:
                continue

            # col2_original = original_case_map[col2_upper] # df cols are uppercase

            try:
                df_filtered = df[[col1_upper, col2_upper]].dropna()
                if df_filtered.empty or len(df_filtered) < MIN_UNIQUE_VALUES_FOR_RELATIONSHIP_SOURCE:
                    continue

                hashable_col1 = False
                try:
                    # Check actual dtype in filtered series, not just original_dtype
                    if pd.api.types.is_object_dtype(df_filtered[col1_upper].dtype):
                        # Try converting to string; avoid modifying original df_filtered if possible
                        temp_col1 = df_filtered[col1_upper].astype(str)
                        if pd.api.types.is_hashable(temp_col1.iloc[0]):
                            hashable_col1 = True
                            # Use the converted series for groupby if needed
                            col1_for_groupby = temp_col1
                    elif pd.api.types.is_hashable(df_filtered[col1_upper].iloc[0]):
                         hashable_col1 = True
                         col1_for_groupby = df_filtered[col1_upper] # Use original series

                except Exception:
                    hashable_col1 = False

                if hashable_col1:
                    # Use the potentially converted col1_for_groupby
                    unique_values_per_group = df_filtered.groupby(col1_for_groupby)[col2_upper].nunique()
                    consistent_groups_ratio = (unique_values_per_group <= 1).sum() / len(unique_values_per_group) if len(unique_values_per_group) > 0 else 1.0

                    if consistent_groups_ratio >= RELATIONSHIP_DETECTION_THRESHOLD:
                        functional_dependencies_detected[col1_upper].append(col2_upper)

            except Exception as dep_e:
                # print(f"    Error checking FD {col1_upper} -> {col2_upper}: {dep_e}")
                pass # Ignore errors for this pair

    # --- Prioritize One-to-One (A <-> B) and Finalize FDs ---
    print("  Prioritizing One-to-One relationships...")
    one_to_one_pairs = set() # Track O2O pairs (tuples of sorted uppercase names)

    for col1_upper, targets in functional_dependencies_detected.items():
        for col2_upper in targets:
            if col2_upper in functional_dependencies_detected and \
               col1_upper in functional_dependencies_detected[col2_upper] and \
               meets_uniqueness_criteria(col2_upper, stats):

                pair = tuple(sorted((col1_upper, col2_upper)))
                if pair not in one_to_one_pairs:
                    relationships[pair[0]].append({"column": pair[1], "type": "one_to_one"})
                    relationships[pair[1]].append({"column": pair[0], "type": "one_to_one"})
                    one_to_one_pairs.add(pair)
                    print(f"    Detected One-to-One: {pair[0]} <-> {pair[1]}")

    # Add remaining Functional Dependencies
    for col1_upper, targets in functional_dependencies_detected.items():
        for col2_upper in targets:
            pair = tuple(sorted((col1_upper, col2_upper)))
            if pair not in one_to_one_pairs:
                 if not any(r.get("column") == col2_upper and r.get("type") == "functional_dependency" for r in relationships[col1_upper]):
                      relationships[col1_upper].append({"column": col2_upper, "type": "functional_dependency"})
                      print(f"    Detected Functional Dependency: {col1_upper} -> {col2_upper}")


    # --- Value Relationships (>, <, =) for Numeric Columns ---
    print("  Checking for Numeric Value Relationships (>, <, =)...")
    if len(numeric_cols_upper) >= 2:
         for col1_upper, col2_upper in product(numeric_cols_upper, repeat=2):
             if col1_upper == col2_upper: continue
             if not meets_uniqueness_criteria(col1_upper, stats): continue

             # col1_original = original_case_map[col1_upper] # df cols are uppercase
             # col2_original = original_case_map[col2_upper] # df cols are uppercase

             try:
                 df_filtered = df[[col1_upper, col2_upper]].dropna()
                 if not df_filtered.empty:
                     num_rows_checked = len(df_filtered)
                     if num_rows_checked > 0:
                          series1 = pd.to_numeric(df_filtered[col1_upper], errors='coerce')
                          series2 = pd.to_numeric(df_filtered[col2_upper], errors='coerce')
                          valid_comparison = pd.concat([series1, series2], axis=1).dropna()
                          num_valid_rows = len(valid_comparison)

                          if num_valid_rows > 0:
                               series1_valid = valid_comparison[col1_upper]
                               series2_valid = valid_comparison[col2_upper]

                               # Check A > B
                               if (series1_valid > series2_valid).sum() / num_valid_rows >= RELATIONSHIP_DETECTION_THRESHOLD:
                                   rel_details = {"column": col2_upper, "relationship": "greater_than", "type": "value_relationship"}
                                   if rel_details not in relationships[col1_upper]: relationships[col1_upper].append(rel_details); print(f"    Detected Value Relationship: {col1_upper} > {col2_upper}")

                               # Check A < B
                               if (series1_valid < series2_valid).sum() / num_valid_rows >= RELATIONSHIP_DETECTION_THRESHOLD:
                                   rel_details = {"column": col2_upper, "relationship": "less_than", "type": "value_relationship"}
                                   if rel_details not in relationships[col1_upper]: relationships[col1_upper].append(rel_details); print(f"    Detected Value Relationship: {col1_upper} < {col2_upper}")

                               # Check A == B
                               if np.isclose(series1_valid, series2_valid).sum() / num_valid_rows >= RELATIONSHIP_DETECTION_THRESHOLD:
                                   rel_details = {"column": col2_upper, "relationship": "equal_to", "type": "value_relationship"}
                                   if rel_details not in relationships[col1_upper]: relationships[col1_upper].append(rel_details); print(f"    Detected Value Relationship: {col1_upper} == {col2_upper}")

             except Exception as rel_e:
                 # print(f"    Error checking numeric value relationship {col1_upper} vs {col2_upper}: {rel_e}")
                 pass


    # --- Temporal Relationships (>, <, =) for Datetime Columns ---
    print("  Checking for Temporal Relationships (>, <, =)...")
    if len(datetime_cols_upper) >= 2:
         for col1_upper, col2_upper in product(datetime_cols_upper, repeat=2):
             if col1_upper == col2_upper: continue
             if not meets_uniqueness_criteria(col1_upper, stats): continue

             # col1_original = original_case_map[col1_upper] # df cols are uppercase
             # col2_original = original_case_map[col2_upper] # df cols are uppercase

             try:
                 df_filtered = df[[col1_upper, col2_upper]].dropna()
                 if not df_filtered.empty:
                     num_rows_checked = len(df_filtered)
                     if num_rows_checked > 0:
                          series1 = pd.to_datetime(df_filtered[col1_upper], errors='coerce')
                          series2 = pd.to_datetime(df_filtered[col2_upper], errors='coerce')
                          valid_comparison = pd.concat([series1, series2], axis=1).dropna()
                          num_valid_rows = len(valid_comparison)

                          if num_valid_rows > 0:
                               series1_valid = valid_comparison[col1_upper]
                               series2_valid = valid_comparison[col2_upper]

                               # Check A > B
                               if (series1_valid > series2_valid).sum() / num_valid_rows >= RELATIONSHIP_DETECTION_THRESHOLD:
                                   rel_details = {"column": col2_upper, "relationship": "greater_than", "type": "temporal_relationship"}
                                   if rel_details not in relationships[col1_upper]: relationships[col1_upper].append(rel_details); print(f"    Detected Temporal Relationship: {col1_upper} > {col2_upper}")

                               # Check A < B
                               if (series1_valid < series2_valid).sum() / num_valid_rows >= RELATIONSHIP_DETECTION_THRESHOLD:
                                   rel_details = {"column": col2_upper, "relationship": "less_than", "type": "temporal_relationship"}
                                   if rel_details not in relationships[col1_upper]: relationships[col1_upper].append(rel_details); print(f"    Detected Temporal Relationship: {col1_upper} < {col2_upper}")

                               # Check A == B
                               if (series1_valid == series2_valid).sum() / num_valid_rows >= RELATIONSHIP_DETECTION_THRESHOLD:
                                   rel_details = {"column": col2_upper, "relationship": "equal_to", "type": "temporal_relationship"}
                                   if rel_details not in relationships[col1_upper]: relationships[col1_upper].append(rel_details); print(f"    Detected Temporal Relationship: {col1_upper} == {col2_upper}")

             except Exception as rel_e:
                 # print(f"    Error checking temporal relationship {col1_upper} vs {col2_upper}: {rel_e}")
                 pass


    print("Column relationship detection complete.")
    return dict(relationships)


def generate_enhanced_schema(df_columns, key_types, stats, relationships, column_inferences, external_metadata=None):
    """
    Generates the enhanced schema dictionary using LLM inferences and external metadata.
    Args:
        df_columns (list): List of uppercase column names from the DataFrame.
        key_types (dict): Key types keyed by uppercase column name.
        stats (dict): Basic statistics keyed by uppercase column name.
        relationships (dict): Detected relationships keyed by uppercase column name.
        column_inferences (dict): LLM inferences keyed by uppercase column name.
        external_metadata (dict, optional): Full metadata keyed by uppercase column name. Defaults to None.
    Returns:
        dict: The enhanced schema dictionary.
    """
    print("\nGenerating enhanced schema dictionary...")
    enhanced_schema = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "original_file": INPUT_CSV,
            "row_count": stats[df_columns[0]]["total_count"] if df_columns and df_columns[0] in stats else 0,
            "column_count": len(df_columns)
        },
        "columns": {}, # Store column details here keyed by uppercase name
        "relationships_summary": relationships # Store relationships summary
    }

    for col_upper in df_columns:
        inferred_info = column_inferences.get(col_upper, {
            "data_type": "unknown", "faker_provider": None, "faker_args": {},
            "data_domain": None, "constraints_description": None
        })
        col_stats = stats.get(col_upper, {})
        external_meta_info = external_metadata.get(col_upper, {}) if external_metadata else {}
        description = external_meta_info.get("Description", "")
        key_type = key_types.get(col_upper, "None")

        enhanced_schema["columns"][col_upper] = {
            "description": description,
            "key_type": key_type,
            "data_type": inferred_info.get("data_type"),
            "stats": col_stats,
            "null_count": col_stats.get("null_count", 0),
            "null_percentage": col_stats.get("null_percentage", 0),
            "total_count": col_stats.get("total_count", 0),
            "sample_values": col_stats.get("sample_values", []),
            "faker_provider": inferred_info.get("faker_provider"),
            "faker_args": inferred_info.get("faker_args", {}),
            "data_domain": inferred_info.get("data_domain"),
            "constraints_description": inferred_info.get("constraints_description"),
            "post_processing_rules": relationships.get(col_upper, []),
            "data_quality": {}
        }

    print("Enhanced schema dictionary generated.")
    return enhanced_schema

def save_schema(schema, json_file_path):
    """Saves the enhanced schema to a JSON file."""
    try:
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(schema, f, indent=4, ensure_ascii=False, default=str)
        print(f"Enhanced schema successfully saved to {json_file_path}")
    except Exception as e:
        print(f"Error saving enhanced schema to {json_file_path}: {e}")

def main():
    """Main function to orchestrate the schema generation process."""
    print("--- Starting Enhanced Schema Generation Process ---")

    # --- Configure Gemini Model ---
    global gemini_model, GEMINI_API_KEY
    if not GEMINI_API_KEY:
        print("Warning: GEMINI_API_KEY environment variable not set. Proceeding without Gemini inference.")
        gemini_model = None
    else:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            gemini_model = genai.GenerativeModel("gemini-1.5-flash")
            print("Gemini model configured successfully.")
        except Exception as e:
            print(f"Error configuring Gemini model: {str(e)}. Proceeding without Gemini inference.")
            gemini_model = None
    # --- End Configure Gemini Model ---


    df, original_dtypes = load_data(INPUT_CSV)
    if df is None:
        print("Failed to load input data. Exiting.")
        return

    # Convert DataFrame columns to uppercase for consistent keying
    # Keep original_case_map if needed elsewhere, but main processing uses uppercase cols
    original_case_map = {col.upper(): col for col in df.columns}
    df.columns = df.columns.str.upper()
    print("Converted DataFrame column names to uppercase for processing.")
    df_columns_upper = df.columns.tolist()


    # Load external metadata (REQUIRED)
    external_metadata = load_external_metadata(METADATA_JSON)
    if external_metadata is None:
        print("Failed to load required external metadata. Exiting.")
        return

    # Validate metadata against DataFrame columns
    missing_meta_cols = [col for col in df_columns_upper if col not in external_metadata]
    if missing_meta_cols:
        print(f"Warning: The following columns exist in the CSV but are missing from the metadata file: {missing_meta_cols}")
        # Add default entries for missing columns to allow processing
        for col in missing_meta_cols:
            external_metadata[col] = {"Column_name": original_case_map.get(col, col), "Key_type": "None", "Description": "N/A - Missing from metadata"}
            print(f"  Added default metadata for missing column: {col}")


    # Calculate stats using df (uppercase cols) and original_dtypes (original case keys)
    stats = calculate_basic_stats(df, original_dtypes)
    # Get key types ONLY from metadata
    key_types = detect_key_types(df_columns_upper, external_metadata)

    # Infer column info using LLM
    column_inferences = {}
    print("\nInferring column info using LLM...")
    if os.path.exists(TEMP_RAW_RESPONSE_FILE):
        try:
            os.remove(TEMP_RAW_RESPONSE_FILE)
            print(f"Cleared previous raw Gemini response file: {TEMP_RAW_RESPONSE_FILE}")
        except Exception as e:
            print(f"Warning: Could not clear raw Gemini response file: {e}")

    for col_upper in df_columns_upper:
         col_stats = stats.get(col_upper, {})
         col_key_type = key_types.get(col_upper, "None")
         external_meta_info = external_metadata.get(col_upper, {})
         inferred_info = infer_column_info_with_llm(col_upper, col_stats, col_key_type, external_meta_info)
         column_inferences[col_upper] = inferred_info
    print("LLM inference for all columns complete.")


    # Detect relationships
    relationships = detect_column_correlations(df, stats, key_types)

    # Generate the enhanced schema
    enhanced_schema = generate_enhanced_schema(
        df_columns_upper,
        key_types,
        stats,
        relationships,
        column_inferences,
        external_metadata
    )

    save_schema(enhanced_schema, OUTPUT_SCHEMA_JSON)

    print("\n--- Enhanced schema generation process finished ---")

if __name__ == "__main__":
    main()
