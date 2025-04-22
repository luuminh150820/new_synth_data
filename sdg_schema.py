import pandas as pd
import json
import os
import numpy as np
import google.generativeai as genai
from google.generativeai import GenerativeModel
import math
from faker import Faker
# Using sdv.constraints.tabular for constraints
# We will still list supported constraints in the prompt for Gemini's awareness,
# but rely on text descriptions and manual mapping if needed, or primarily use
# constraints detected by the basic script or manually added.
# from sdv.constraints.tabular import ScalarRange, Unique, OneHotEncoding
from sdv.metadata import SingleTableMetadata
import re
import random
from datetime import datetime, timedelta
import inspect # Import inspect to check function parameters
from itertools import combinations # Import combinations for formula detection

# Configuration
INPUT_CSV = "customer_data.csv"
OUTPUT_SCHEMA_JSON = "enhanced_schema.json"  # Output JSON file for the enhanced schema
METADATA_JSON = "FCT_ENT_TERM_DEPOSIT_metadata.json" # This file is optional now for primary key
BATCH_SIZE = 5
CORRELATION_THRESHOLD = 0.7 # Threshold for reporting correlations
TEMP_RAW_RESPONSE_FILE = "temp_raw_gemini_response.txt" # Temporary file to save raw Gemini response
FORMULA_TOLERANCE = 1e-6 # Tolerance for floating-point comparisons in formula detection

# Set proxy environment variables if needed
# os.environ["HTTP_PROXY"] = "http://dc2-proxyuat.seauat.com.vn:8080"
# os.environ["HTTPS_PROXY"] = "http://dc2-proxyuat.seauat.com.vn:8080"

# Configure Gemini API
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    # In a real application, you might handle this more gracefully
    # For this script, we'll raise an error as the API key is required for enhancement
    raise ValueError("GEMINI_API_KEY environment variable not set.")

try:
    genai.configure(api_key=GEMINI_API_KEY)
    # Use a model suitable for text generation and understanding structure
    model = genai.GenerativeModel("gemini-1.5-flash-latest") # Using a more recent model
    print("Gemini model configured successfully.")
except Exception as e:
    print(f"Error configuring Gemini model: {str(e)}")
    print("Proceeding without Gemini schema enhancement.")
    model = None # Set model to None if configuration fails


fake = Faker(['vi_VN'])
Faker.seed(42)

def read_metadata(json_file_path):
    """Reads metadata from a JSON file."""
    if not os.path.exists(json_file_path):
        print(f"Metadata file {json_file_path} not found. Continuing without metadata.")
        return {}
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        column_metadata = {}
        if "fct_ent_term_deposit" in metadata and "columns" in metadata["fct_ent_term_deposit"]:
            for column in metadata["fct_ent_term_deposit"]["columns"]:
                column_name = column.get("Column_name", "").strip()
                if column_name:
                    column_metadata[column_name] = {
                        "description": column.get("Description", ""),
                        "key_type": column.get("Key_type", "")
                    }
        return column_metadata
    except Exception as e:
        print(f"Error reading metadata: {str(e)}")
        return {}

def detect_relationships(df):
    """
    Detects column relationships (Pearson, basic functional dependency, value relationships, simple formulas)
     and returns them as a dictionary where keys are column names and values are lists
     of detected relationships for that column. This is the primary source
     for structured relationships.
    """
    relationships = {} # Initialize relationships as a dictionary
    RELATIONSHIP_CONSISTENCY_THRESHOLD = 0.999 # Define threshold for value relationship consistency (e.g., 99.9%)
    global CORRELATION_THRESHOLD
    global FORMULA_TOLERANCE

    try:
        # Select only columns that are purely numeric (int or float)
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        # --- Pearson Correlation ---
        if len(numeric_cols) >= 2:
            try:
                df_numeric = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
                df_numeric = df_numeric.dropna(axis=1, how='all') # Drop columns that became all NaN
                numeric_cols_cleaned = df_numeric.columns.tolist() # Update numeric_cols list after cleaning

                if len(numeric_cols_cleaned) >= 2:
                     corr_matrix = df_numeric.corr(method='pearson', numeric_only=True) # numeric_only=True is default

                     for i, col1 in enumerate(numeric_cols_cleaned):
                         for j, col2 in enumerate(numeric_cols_cleaned):
                             if i < j: # Avoid duplicates and self-correlation (A-B is same as B-A)
                                 corr_value = corr_matrix.loc[col1, col2]
                                 if abs(corr_value) >= CORRELATION_THRESHOLD and not pd.isna(corr_value):
                                     # Add correlation for col1 -> col2
                                     if col1 not in relationships:
                                         relationships[col1] = []
                                     relationships[col1].append({
                                         "column": col2,
                                         "correlation": float(round(corr_value, 3)), # Ensure JSON serializable float
                                         "type": "pearson_correlation" # Use specific type name
                                     })
                                     # Add reciprocal correlation for col2 -> col1
                                     if col2 not in relationships:
                                         relationships[col2] = []
                                     relationships[col2].append({
                                         "column": col1,
                                         "correlation": float(round(corr_value, 3)), # Ensure JSON serializable float
                                         "type": "pearson_correlation"
                                     })
                     print(f"Detected Pearson correlations above threshold {CORRELATION_THRESHOLD}.")
                else:
                     pass # print("Not enough valid numeric columns after cleaning for Pearson correlation calculation.")

            except Exception as pearson_e:
                 print(f"Error during Pearson correlation calculation for reporting: {pearson_e}")


        # --- Functional Dependency (Simple Check: one value maps to only one other value) ---
        all_cols = df.columns.tolist() # Check dependency across all columns
        print("\nChecking for functional dependencies...")
        for col1 in all_cols:
            for col2 in all_cols:
                if col1 != col2:
                    try:
                        df_filtered = df[[col1, col2]].dropna()
                        if not df_filtered.empty:
                            if pd.api.types.is_hashable(df_filtered[col1].dtype):
                                if not df_filtered[col1].empty:
                                     if pd.api.types.is_hashable(df_filtered[col1].iloc[0]):
                                        unique_values_per_group = df_filtered.groupby(col1)[col2].nunique()
                                        if (unique_values_per_group <= 1).all():
                                            # Only add dependency from col1 to col2 if col1 comes before col2 alphabetically
                                            # This prevents adding both A -> B and B -> A if both exist
                                            if col1 < col2:
                                                 if col1 not in relationships:
                                                     relationships[col1] = []
                                                 if not any(d.get("column") == col2 and d.get("type") == "functional_dependency" for d in relationships[col1]):
                                                     relationships[col1].append({
                                                         "column": col2,
                                                         "correlation": 1.0, # Represent dependency
                                                         "type": "functional_dependency"
                                                     })
                                                     print(f"Detected potential functional dependency: {col1} -> {col2}")
                    except Exception as dep_e:
                        pass


        # --- Basic Value Relationships (e.g., col1 <= col2, col1 >= col2) ---
        # Check for consistent inequality relationships between numeric columns
        print("\nChecking for basic value relationships (e.g., <=, >=)...")
        if len(numeric_cols) >= 2:
             for col1 in numeric_cols:
                 for col2 in numeric_cols:
                     if col1 != col2:
                         try:
                             df_filtered = df[[col1, col2]].dropna()
                             if not df_filtered.empty:
                                 num_rows_checked = len(df_filtered)

                                 # Check col1 <= col2
                                 if num_rows_checked > 0 and (df_filtered[col1] <= df_filtered[col2]).sum() / num_rows_checked >= RELATIONSHIP_CONSISTENCY_THRESHOLD:
                                     # Only add one direction based on alphabetical order to avoid redundancy
                                     if col1 < col2: # Add A <= B if A comes before B alphabetically
                                         if col1 not in relationships:
                                             relationships[col1] = []
                                         if not any(d.get("column") == col2 and d.get("relationship") == "less_than_or_equal_to" for d in relationships[col1]):
                                              relationships[col1].append({
                                                  "column": col2,
                                                  "relationship": "less_than_or_equal_to",
                                                  "type": "value_relationship"
                                              })
                                              print(f"Detected value relationship: {col1} <= {col2} (holds for >= {RELATIONSHIP_CONSISTENCY_THRESHOLD:.1%} of data)")
                                     # No need to add B >= A explicitly if A <= B is added

                                 # Check col1 >= col2
                                 if num_rows_checked > 0 and (df_filtered[col1] >= df_filtered[col2]).sum() / num_rows_checked >= RELATIONSHIP_CONSISTENCY_THRESHOLD:
                                     # Only add one direction based on alphabetical order to avoid redundancy
                                     if col1 < col2: # Add A >= B if A comes before B alphabetically
                                         if col1 not in relationships:
                                             relationships[col1] = []
                                         if not any(d.get("column") == col2 and d.get("relationship") == "greater_than_or_equal_to" for d in relationships[col1]):
                                              relationships[col1].append({
                                                  "column": col2,
                                                  "relationship": "greater_than_or_equal_to",
                                                  "type": "value_relationship"
                                              })
                                              print(f"Detected value relationship: {col1} >= {col2} (holds for >= {RELATIONSHIP_CONSISTENCY_THRESHOLD:.1%} of data)")
                                     # No need to add B <= A explicitly if A >= B is added


                         except Exception as rel_e:
                             pass


        # --- Simple Formula Relationships (A = B + C) ---
        print("\nChecking for simple formula relationships (A = B + C)...")
        if len(numeric_cols) >= 3:
             # Iterate through all combinations of 3 unique numeric columns
             for col_a, col_b, col_c in combinations(numeric_cols, 3):
                 try:
                     # Drop rows where any of the three columns are NaN
                     df_filtered = df[[col_a, col_b, col_c]].dropna()
                     if not df_filtered.empty:
                         num_rows_checked = len(df_filtered)

                         # Check if col_a = col_b + col_c
                         if num_rows_checked > 0 and np.allclose(df_filtered[col_a], df_filtered[col_b] + df_filtered[col_c], atol=FORMULA_TOLERANCE):
                             formula = f"{col_b} + {col_c}"
                             if col_a not in relationships:
                                 relationships[col_a] = []
                             # Avoid adding duplicate formula entries
                             if not any(d.get("formula") == formula and d.get("type") == "formula" for d in relationships[col_a]):
                                  relationships[col_a].append({
                                      "target_column": col_a,
                                      "source_columns": [col_b, col_c],
                                      "formula": formula,
                                      "type": "formula"
                                  })
                                  print(f"Detected formula relationship: {col_a} = {formula}")

                         # Check if col_b = col_a + col_c
                         if num_rows_checked > 0 and np.allclose(df_filtered[col_b], df_filtered[col_a] + df_filtered[col_c], atol=FORMULA_TOLERANCE):
                             formula = f"{col_a} + {col_c}"
                             if col_b not in relationships:
                                 relationships[col_b] = []
                             if not any(d.get("formula") == formula and d.get("type") == "formula" for d in relationships[col_b]):
                                  relationships[col_b].append({
                                      "target_column": col_b,
                                      "source_columns": [col_a, col_c],
                                      "formula": formula,
                                      "type": "formula"
                                  })
                                  print(f"Detected formula relationship: {col_b} = {formula}")

                         # Check if col_c = col_a + col_b
                         if num_rows_checked > 0 and np.allclose(df_filtered[col_c], df_filtered[col_a] + df_filtered[col_b], atol=FORMULA_TOLERANCE):
                             formula = f"{col_a} + {col_b}"
                             if col_c not in relationships:
                                 relationships[col_c] = []
                             if not any(d.get("formula") == formula and d.get("type") == "formula" for d in relationships[col_c]):
                                  relationships[col_c].append({
                                      "target_column": col_c,
                                      "source_columns": [col_a, col_b],
                                      "formula": formula,
                                      "type": "formula"
                                  })
                                  print(f"Detected formula relationship: {col_c} = {formula}")

                 except Exception as formula_e:
                     pass # Suppress errors during formula check


    except Exception as e:
        print(f"Error detecting relationships for reporting: {str(e)}")
        import traceback
        traceback.print_exc()

    # Clean up empty relationship lists (columns with no detected relationships)
    relationships = {col: rels for col, rels in relationships.items() if rels}

    return relationships


def is_categorical(series, threshold=0.5):
    """Determines if a series is categorical."""
    if pd.api.types.is_categorical_dtype(series):
        return True
    if series.dtype == object:
        if series.nunique() / series.count() <= threshold:
            return True
    if pd.api.types.is_numeric_dtype(series):
        unique_values = series.dropna().nunique()
        if unique_values <= 20 and unique_values / series.count() <= threshold:
            return True
    return False

def read_csv_and_generate_schema(csv_file_path, metadata=None):
    """Reads a CSV file and generates a schema."""
    try:
        df = pd.read_csv(csv_file_path)
        if df.empty:
            print("Warning: CSV file is empty")
            return {}
        schema = {}
        # Relationships are now detected separately and returned by main
        # correlations = detect_relationships(df) # Removed this call here

        for column in df.columns:
            col_data = df[column]
            non_null_count = col_data.count()
            null_count = col_data.isna().sum()
            total_count = len(col_data)
            null_percentage = round((null_count / total_count) * 100, 2) if total_count > 0 else 0
            sample_values = []
            if non_null_count > 0:
                sample_size = min(5, non_null_count)
                sample_values = col_data.dropna().sample(sample_size).tolist() if sample_size > 0 else []

            # --- Determine Data Type ---
            data_type = "string" # Default to string
            stats = {"unique_values_count": int(col_data.nunique())} # Default stats

            if is_categorical(col_data):
                data_type = "categorical"
                if non_null_count > 0:
                    value_counts = col_data.value_counts(dropna=True)
                    value_percentages = (value_counts / non_null_count * 100).round(2)
                    categories = []
                    for val, count in value_counts.items():
                        percentage = value_percentages[val]
                        categories.append({
                            "value": val,
                            "count": int(count),
                            "percentage": float(percentage)
                        })
                    if len(categories) > 10:
                        stats["top_categories"] = categories[:10]
                    else:
                        stats["categories"] = categories

            elif pd.api.types.is_numeric_dtype(col_data):
                non_null_data = col_data.dropna()
                data_type = "integer" if non_null_data.apply(lambda x: float(x).is_integer()).all() else "float"
                if non_null_count > 0:
                    stats = {
                        "min": float(non_null_data.min()),
                        "max": float(non_null_data.max()),
                        "mean": float(non_null_data.mean()),
                        "median": float(non_null_data.median()),
                        "std_dev": float(non_null_data.std()),
                        "unique_values_count": int(col_data.nunique())
                    }
            elif pd.api.types.is_datetime64_dtype(col_data):
                data_type = "datetime"
                if non_null_count > 0:
                    date_min = col_data.min()
                    date_max = col_data.max()
                    stats = {
                        "min": str(date_min),
                        "max": str(date_max),
                        "unique_values_count": int(col_data.nunique()),
                        "date_range_days": (date_max - date_min).days if not pd.isna(date_min) and not pd.isna(date_max) else 0
                    }
            else: # Handle as string if not caught by other types
                 if non_null_count > 0:
                    str_lengths = col_data.dropna().astype(str).str.len()
                    stats.update({
                        "max_length": int(str_lengths.max()),
                        "min_length": int(str_lengths.min()),
                        "mean_length": float(str_lengths.mean())
                    })


            # --- Determine Default Faker Provider based on Data Type ---
            default_faker_provider = None
            default_faker_args = {}

            if data_type == "string":
                default_faker_provider = "word" # Default for generic strings
            elif data_type in ["numerical", "integer", "float"]:
                 default_faker_provider = "random_number" # Default for numbers
                 # Add basic args based on detected range if available
                 if 'min' in stats and 'max' in stats:
                      # Use random_int or pyfloat based on integer/float type
                      if data_type == "integer":
                           default_faker_provider = "random_int"
                           default_faker_args = {"min": int(stats['min']), "max": int(stats['max'])}
                      else: # float or general numerical
                           default_faker_provider = "pyfloat"
                           # Add nb_digits based on analysis if needed, or use a default
                           # For simplicity, let's not add nb_digits by default unless needed
                           default_faker_args = {"min_value": float(stats['min']), "max_value": float(stats['max'])}
                           # Set positive arg correctly based on min_value
                           if stats['min'] > 0:
                                default_faker_args["positive"] = True
                           elif stats['max'] < 0:
                                default_faker_args["positive"] = False # Negative numbers
                           # If min <= 0 <= max, omit positive arg


            elif data_type == "datetime":
                 default_faker_provider = "date_object" # Default for datetime
                 # Add basic args based on detected range if available
                 if 'min' in stats and 'max' in stats:
                      # Faker date_object uses start/end_date (as date objects or strings)
                      # Need to handle potential conversion from string stats min/max
                      try:
                           from datetime import datetime
                           # Attempt to parse common date formats
                           start_date_str = str(stats['min']).split(' ')[0] # Take date part
                           end_date_str = str(stats['max']).split(' ')[0] # Take date part
                           date_formats_to_try = ['%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y', '%Y/%m/%d'] # Add other common formats
                           start_date_obj = None
                           end_date_obj = None

                           for fmt in date_formats_to_try:
                               try:
                                   start_date_obj = datetime.strptime(start_date_str, fmt).date()
                                   break # Stop if parsing successful
                               except ValueError:
                                   pass # Try next format

                           for fmt in date_formats_to_try:
                               try:
                                   end_date_obj = datetime.strptime(end_date_str, fmt).date()
                                   break # Stop if parsing successful
                               except ValueError:
                                   pass # Try next format

                           if start_date_obj and end_date_obj:
                               default_faker_args = {"start_date": start_date_obj, "end_date": end_date_obj}
                           else:
                               print(f"Warning: Could not parse date range '{start_date_str}' to '{end_date_str}' for '{column}'. Using default date_object.")
                               default_faker_args = {} # Use default Faker behavior if range parsing fails

                      except Exception as date_e:
                           print(f"Warning: Error setting default Faker date range for '{column}': {date_e}. Using default date_object.")
                           default_faker_args = {} # Use default Faker behavior if range parsing fails

            elif data_type == "categorical":
                 # For categorical, rely on the distribution handling in the synthesis script
                 # Setting faker_provider to None is appropriate here.
                 default_faker_provider = None
                 default_faker_args = {}
                 # If categories list is small, could consider "random_element" with elements from categories
                 # if 'categories' in stats and 'categories' in stats and len(stats['categories']) <= 10: # Example threshold
                 #      default_faker_provider = "random_element"
                 #      default_faker_args = {"elements": [cat['value'] for cat in stats['categories']]}


            # --- Populate Schema Dictionary ---
            schema[column] = {
                "description": metadata.get(column, {}).get("description", "") if metadata else "",
                "key_type": metadata.get(column, {}).get("key_type", "") if metadata else "",
                "data_type": data_type, # Add detected data type
                "stats": stats, # Add detected stats
                "null_count": int(null_count),
                "null_percentage": null_percentage,
                "total_count": int(total_count),
                "sample_values": sample_values,
                "faker_provider": default_faker_provider, # Add default Faker provider
                "faker_args": default_faker_args, # Add default Faker args
                "sdv_constraints": [], # Keep this field, but we won't use it for add_constraints
                "post_processing": None,
            }

            # --- Hardcode Primary Key (Override if column is CONTRACT) ---
            if column == "CONTRACT":
                 schema[column]["key_type"] = "Primary Key"
                 print(f"Hardcoded '{column}' as Primary Key in initial schema.")
                 # Also set sdtype for PK if needed later, though create_sdv_metadata handles it
                 # schema[column]["sdtype"] = "id"
            # --- End Hardcode Primary Key ---

        return schema
    except Exception as e:
        print(f"Error generating schema: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def enhance_schema_batch_alternative(schema_batch):
    """
    Enhances a batch of schema using Gemini, parsing a custom text format.
    More robust parsing logic.
    """
    try:
        # --- Prompt for Custom Text Format ---
        # Removed "Relationships" field from the prompt
        prompt = f"""
        Enhance the schema details for the following columns from a Vietnamese banking dataset.
        Provide the enhanced information for each column using the exact format specified below.
        Ensure values for Faker Args are valid JSON objects.

        For each column in the batch:
        ## Column: [Column Name]
        Description: [Detailed description of what this column represents]
        Domain: [Domain and realistic ranges for values]
        Constraints Description: [Human-readable patterns or constraints that should be maintained]
        Data Quality: [Recommendations for data validation]
        Faker Provider: [The most appropriate specific Faker provider method to generate realistic values for this column's data type and domain, or null if SDV's learned distribution is preferred, or 'random_element' for limited categorical values]
        Faker Args: {{JSON object with parameters for the chosen Faker provider, e.g., {{"min": 0, "max": 100}}, {{"elements": ["A", "B"]}}, {{"start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"}}, or empty {{}} if no args}}
        Post Processing: [Any special post-processing function needed (e.g., 'format_as_currency', 'ensure_valid_id', 'format_percentage', 'format_date'), or null]
        ---

        Here is the schema batch to enhance:
        {json.dumps(schema_batch, indent=2)}

        Provide ONLY the enhanced information in the specified format, starting with the first '## Column:'.
        """
        # --- End Prompt ---

        response = model.generate_content(prompt)
        response_text = response.text

        # --- Debugging: Save Raw Response to File ---
        try:
            with open(TEMP_RAW_RESPONSE_FILE, 'w', encoding='utf-8') as f:
                 f.write(response_text)
            print(f"\nRaw Gemini response for this batch saved to '{TEMP_RAW_RESPONSE_FILE}' for inspection.")
        except Exception as e:
            print(f"Error saving raw response to file: {e}")
        # --- End Debugging ---

        # --- Parse Custom Text Format (More Robust) ---
        enhanced_batch = {}
        # Split by '## Column:' to get blocks for each column. The first element is usually preamble.
        column_blocks = re.split(r'## Column:', response_text)

        print(f"Attempting to parse {len(column_blocks) - 1} potential column blocks from Gemini response (excluding preamble)...")

        # Regex to find fields within a column block
        # Captures Field Name and the text following it until the next field name or end of block
        # Removed "Relationships" from the regex pattern
        field_pattern = re.compile(r"^(Description|Domain|Constraints Description|Data Quality|Faker Provider|Faker Args|Post Processing):\s*(.*)", re.MULTILINE)

        for i, block in enumerate(column_blocks):
            if i == 0: # Skip the preamble before the first '## Column:'
                continue

            block = block.strip()
            if not block: # Skip empty blocks
                continue

            # The first line of the block after '## Column:' should be the column name
            lines = block.split('\n', 1) # Split into first line (name) and rest of block
            if not lines or not lines[0].strip():
                 print(f"  Warning: Could not extract column name from block {i}. Skipping block.")
                 continue

            column_name = lines[0].strip()
            rest_of_block = lines[1] if len(lines) > 1 else ""

            column_details = {}
            # Use finditer to get all field matches in order
            field_matches = list(field_pattern.finditer(rest_of_block))

            if not field_matches:
                 print(f"  Warning: No fields found for column '{column_name}'. Skipping enhancement for this column.")
                 # Add column name to enhanced_batch with default info if needed, or just skip
                 # Let's just skip enhancing this column and rely on basic schema fallback
                 continue

            # Iterate through matches to extract field values
            for j, match in enumerate(field_matches):
                field_name_prompt = match.group(1)
                field_value_start = match.group(2).strip() # Text on the same line as field name

                # Determine the end of the current field's value
                value_end_index = len(rest_of_block) # Default to end of block

                # If there's a next field match, the value ends just before it
                if j + 1 < len(field_matches):
                    next_match_start_index = field_matches[j+1].start()
                    value_end_index = next_match_start_index

                # Extract the full value text for the current field
                # Need to be careful with slicing if match.end(1) + 1 is out of bounds
                start_slice = match.end(1) + 1
                if start_slice >= len(rest_of_block):
                     full_value_text = "" # No text after the field name and colon
                else:
                     full_value_text = rest_of_block[start_slice:value_end_index].strip()


                # Map prompt field names to schema field names
                schema_field_map = {
                    "Description": "description",
                    "Domain": "domain",
                    "Constraints Description": "constraints_description",
                    # Removed "Relationships" from mapping
                    "Data Quality": "data_quality",
                    "Faker Provider": "faker_provider",
                    "Faker Args": "faker_args",
                    "Post Processing": "post_processing"
                }
                schema_field_name = schema_field_map.get(field_name_prompt)

                if schema_field_name:
                    column_details[schema_field_name] = full_value_text # Store the extracted value text


            # --- Process Faker Provider (Clean extra quotes/prefix) ---
            if "faker_provider" in column_details and column_details["faker_provider"] is not None:
                 provider_text = str(column_details["faker_provider"]).strip()
                 # Remove leading/trailing quotes (single or double)
                 provider_text = re.sub(r'^[\'"]|[\'"]$', '', provider_text)
                 # Remove potential 'Faker.' prefix
                 provider_text = provider_text.replace('Faker.', '')
                 column_details["faker_provider"] = provider_text if provider_text else None # Set to None if empty after cleaning


            # --- Process Faker Args specifically as JSON (More Robust) ---
            if "faker_args" in column_details:
                faker_args_text_raw = column_details["faker_args"].strip()
                column_details["faker_args"] = {} # Default to empty dict

                if faker_args_text_raw:
                    # Try to find the first '{' and the last '}' to isolate the JSON object
                    json_start = faker_args_text_raw.find('{')
                    json_end = faker_args_text_raw.rfind('}')

                    if json_start != -1 and json_end != -1 and json_end > json_start:
                         faker_args_json_string = faker_args_text_raw[json_start : json_end + 1]
                         try:
                            # Attempt to parse the isolated JSON string
                            column_details["faker_args"] = json.loads(faker_args_json_string)
                            # print(f"  Successfully parsed Faker Args for '{column_name}'.")
                         except json.JSONDecodeError as e:
                            print(f"  Warning: Could not parse isolated Faker Args JSON for column '{column_name}': {e}. Raw text: '{faker_args_text_raw}'. Isolated: '{faker_args_json_string}'. Setting faker_args to empty dictionary.")
                            # Keep default empty dict
                         except Exception as e:
                             print(f"  Warning: Unexpected error parsing isolated Faker Args for '{column_name}': {e}. Raw text: '{faker_args_text_raw}'. Setting faker_args to empty dictionary.")
                             # Keep default empty dict
                    else:
                         print(f"  Warning: Could not find valid JSON object {{...}} pattern in Faker Args text for column '{column_name}'. Raw text: '{faker_args_text_raw}'. Setting faker_args to empty dictionary.")
                         # Keep default empty dict
                # else: # If raw text is empty, faker_args remains {} (the default)


            # Add the extracted details to the batch result if column name was found
            if column_name:
                # Before adding, ensure column_name exists in the original schema_batch keys
                # This prevents adding data for columns not in the current batch if Gemini hallucinates
                if column_name in schema_batch:
                     # Start with the original schema info for this column
                     enhanced_column_info = schema_batch[column_name].copy()
                     # Update with details parsed from Gemini's response
                     enhanced_column_info.update(column_details)
                     enhanced_batch[column_name] = enhanced_column_info
                     print(f"  Parsed details for column: {column_name}")
                else:
                     print(f"  Warning: Parsed column '{column_name}' from Gemini response is not in the original schema batch. Skipping.")


        if not enhanced_batch:
             print("Warning: No column details successfully parsed from Gemini response.")

        return enhanced_batch # Return the dictionary of enhanced column details

    except Exception as e:
        print(f"Error during custom format parsing in alternative enhancement approach: {str(e)}")
        import traceback
        traceback.print_exc() # Print traceback for the error
        return schema_batch # Return original batch on error


def enhance_schema_batch(schema_batch):
    """Enhances a batch of schema information using the Gemini API."""
    # This function now primarily acts as a fallback to the alternative approach
    # as the structured output was causing issues.
    print("Attempting schema enhancement using the alternative text parsing approach...")
    return enhance_schema_batch_alternative(schema_batch)


def enhance_schema_with_gemini(schema):
    """Enhances the schema using the Gemini API in batches."""
    try:
        column_names = list(schema.keys())
        num_columns = len(column_names)
        num_batches = math.ceil(num_columns / BATCH_SIZE)
        print(f"Processing {num_columns} columns in {num_batches} batches of {BATCH_SIZE}...")
        enhanced_schema = {}
        # Initialize enhanced_schema with the basic schema content first
        # This ensures all columns are present even if enhancement fails for some
        enhanced_schema.update(schema)


        for i in range(num_batches):
            start_idx = i * BATCH_SIZE
            end_idx = min((i + 1) * BATCH_SIZE, num_columns)
            batch_columns = column_names[start_idx:end_idx]
            print(f"Processing batch {i+1}/{num_batches} with columns: {batch_columns}")
            schema_batch = {col: schema[col] for col in batch_columns} # Get basic schema for the batch

            # Call enhance_schema_batch (which now calls the alternative approach)
            enhanced_batch_result = enhance_schema_batch(schema_batch)

            # Update the enhanced_schema with the results from the batch
            # This will overwrite the basic schema info if enhancement was successful
            if enhanced_batch_result:
                 enhanced_schema.update(enhanced_batch_result)
                 print(f"Successfully processed batch {i+1}. Updated schema with enhanced details.")
            else:
                 print(f"Warning: Batch {i+1} enhancement failed or returned empty. Keeping basic schema for these columns.")


        return enhanced_schema # Return the fully enhanced schema (or basic if enhancement failed)

    except Exception as e:
        print(f"Error in batch enhancement process: {str(e)}")
        import traceback
        traceback.print_exc()
        print("Returning original basic schema due to enhancement process error.")
        return schema # Return original basic schema on overall process error


def main():
    """Main function to generate the enhanced schema."""
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Input file '{INPUT_CSV}' does not exist.")
        return

    print("Starting schema generation process")
    # Metadata file is optional now for primary key due to hardcoding
    metadata = read_metadata(METADATA_JSON)

    # Read the CSV to get the DataFrame for analysis
    try:
        df = pd.read_csv(INPUT_CSV)
        if df.empty:
            print("Error: Input CSV file is empty.")
            return
    except Exception as e:
        print(f"Error reading input CSV file: {str(e)}")
        return


    print(f"Generating basic schema from {INPUT_CSV}...")
    # Generate the basic schema (column details)
    schema = read_csv_and_generate_schema(INPUT_CSV, metadata)
    if not schema:
        print("Failed to generate schema. Exiting.")
        return

    print("Detecting relationships based on data analysis...")
    # Detect relationships separately
    relationships = detect_relationships(df)
    if relationships is None: # Check for errors during relationship detection
         print("Warning: Relationship detection failed. Proceeding without structured relationships.")
         relationships = {} # Ensure relationships is an empty dict if detection fails


    # Basic schema structure includes columns and relationships detected by the script
    # Relationships are now at the top level
    basic_schema_structure = {
        "columns": schema,
        "relationships": relationships # Add detected relationships at the top level
    }

    basic_schema_file = "basic_schema.json"
    with open(basic_schema_file, 'w', encoding='utf-8') as f:
        json.dump(basic_schema_structure, f, indent=2, ensure_ascii=False)
    print(f"Basic schema structure saved to {basic_schema_file}")

    # Only proceed with Gemini enhancement if the model was configured successfully
    if model:
        print("Enhancing schema structure with Gemini...")
        # Pass the basic schema (columns) to the enhancement process
        # The enhancement process will only update column details
        enhanced_columns = enhance_schema_with_gemini(schema)

        # The final enhanced schema combines enhanced columns and script-detected relationships
        enhanced_schema_structure = {
            "columns": enhanced_columns,
            "relationships": relationships # Keep the script-detected relationships
        }
    else:
        print("Skipping Gemini schema enhancement due to configuration error.")
        # If Gemini is not available, the final schema is just the basic schema structure
        enhanced_schema_structure = basic_schema_structure


    # Save the final enhanced schema (which might be the basic schema if enhancement failed)
    try:
        with open(OUTPUT_SCHEMA_JSON, 'w', encoding='utf-8') as f:
            json.dump(enhanced_schema_structure, f, indent=2, ensure_ascii=False)
        print(f"Final schema (enhanced or basic fallback) saved to {OUTPUT_SCHEMA_JSON}")
    except Exception as e:
        print(f"Error saving final schema to {OUTPUT_SCHEMA_JSON}: {str(e)}")
        import traceback
        traceback.print_exc()


    print("Schema generation completed.")

if __name__ == "__main__":
    main()
