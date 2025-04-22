import pandas as pd
import json
import os
import numpy as np
import google.generativeai as genai
from google.generativeai import GenerativeModel
import math
from faker import Faker
# Using sdv.constraints.tabular for constraints
# Importing only constraints confirmed to be in the user's SDV version based on their list
from sdv.constraints.tabular import ScalarRange, Unique, OneHotEncoding
from sdv.metadata import SingleTableMetadata
import re
import random
from datetime import datetime, timedelta
import inspect # Import inspect to check function parameters

# Configuration
INPUT_CSV = "customer_data.csv"
OUTPUT_SCHEMA_JSON = "enhanced_schema.json"  # Output JSON file for the enhanced schema
METADATA_JSON = "FCT_ENT_TERM_DEPOSIT_metadata.json" # This file is optional now for primary key
BATCH_SIZE = 5
CORRELATION_THRESHOLD = 0.7 # Threshold for reporting correlations
TEMP_RAW_RESPONSE_FILE = "temp_raw_gemini_response.txt" # Temporary file to save raw Gemini response

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
    Detects column relationships (Pearson, basic functional dependency, value relationships)
     and returns them as a list of relationship objects.
    """
    relationships = []
    RELATIONSHIP_CONSISTENCY_THRESHOLD = 0.999 # Define threshold for value relationship consistency (e.g., 99.9%)
    # Use the same CORRELATION_THRESHOLD from the main script config
    global CORRELATION_THRESHOLD

    # print("\nDetecting column relationships...") # Removed print statement

    try:
        # Select only columns that are purely numeric (int or float)
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        # --- Pearson Correlation ---
        if len(numeric_cols) >= 2:
            try:
                df_numeric = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
                df_numeric = df_numeric.dropna(axis=1, how='all')
                numeric_cols_cleaned = df_numeric.columns.tolist()

                if len(numeric_cols_cleaned) >= 2:
                     corr_matrix = df_numeric.corr(method='pearson', numeric_only=True)

                     for i, col1 in enumerate(numeric_cols_cleaned):
                         for j, col2 in enumerate(numeric_cols_cleaned):
                             if i < j: # Avoid duplicates and self-correlation
                                 corr_value = corr_matrix.loc[col1, col2]
                                 if abs(corr_value) >= CORRELATION_THRESHOLD and not pd.isna(corr_value):
                                     relationships.append({
                                         "type": "pearson_correlation",
                                         "column1": col1,
                                         "column2": col2,
                                         "correlation": float(round(corr_value, 3)),
                                         "note": "Detected by basic script"
                                     })
                     # print(f"Detected Pearson correlations above threshold {CORRELATION_THRESHOLD}.") # Removed print statement
                else:
                     pass

            except Exception as pearson_e:
                 print(f"Error during Pearson correlation detection: {pearson_e}")


        # --- Functional Dependency (Simple Check: one value maps to only one other value) ---
        all_cols = df.columns.tolist()
        # print("\nChecking for simple functional dependencies...") # Removed print statement
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
                                            # Check if this dependency is already added to avoid duplicates
                                            is_duplicate = False
                                            for rel in relationships:
                                                 if rel.get("type") == "functional_dependency" and rel.get("source_column") == col1 and rel.get("target_column") == col2:
                                                      is_duplicate = True
                                                      break
                                            if not is_duplicate:
                                                 relationships.append({
                                                     "type": "functional_dependency",
                                                     "source_column": col1,
                                                     "target_column": col2,
                                                     "note": "Detected by basic script"
                                                 })
                                                 # print(f"Detected potential simple functional dependency: {col1} -> {col2}") # Removed print statement
                    except Exception as dep_e:
                        pass


        # --- Basic Value Relationships (e.g., col1 <= col2, col1 >= col2) ---
        # print("\nChecking for basic value relationships (e.g., <=, >=)...") # Removed print statement
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
                                     is_duplicate = False
                                     for rel in relationships:
                                          if rel.get("type") == "value_relationship" and rel.get("column1") == col1 and rel.get("column2") == col2 and rel.get("relationship") == "less_than_or_equal_to":
                                               is_duplicate = True
                                               break
                                     if not is_duplicate:
                                          relationships.append({
                                              "type": "value_relationship",
                                              "column1": col1,
                                              "column2": col2,
                                              "relationship": "less_than_or_equal_to",
                                              "note": "Detected by basic script"
                                          })
                                          # print(f"Detected value relationship: {col1} <= {col2} (holds for >= {RELATIONSHIP_CONSISTENCY_THRESHOLD:.1%} of data)") # Removed print statement

                                 # Check col1 >= col2
                                 if num_rows_checked > 0 and (df_filtered[col1] >= df_filtered[col2]).sum() / num_rows_checked >= RELATIONSHIP_CONSISTENCY_THRESHOLD:
                                     is_duplicate = False
                                     for rel in relationships:
                                          if rel.get("type") == "value_relationship" and rel.get("column1") == col1 and rel.get("column2") == col2 and rel.get("relationship") == "greater_than_or_equal_to":
                                               is_duplicate = True
                                               break
                                     if not is_duplicate:
                                          relationships.append({
                                              "type": "value_relationship",
                                              "column1": col1,
                                              "column2": col2,
                                              "relationship": "greater_than_or_equal_to",
                                              "note": "Detected by basic script"
                                          })
                                          # print(f"Detected value relationship: {col1} >= {col2} (holds for >= {RELATIONSHIP_CONSISTENCY_THRESHOLD:.1%} of data)") # Removed print statement

                         except Exception as rel_e:
                             pass


    except Exception as e:
        print(f"Error detecting relationships: {str(e)}")
        import traceback
        traceback.print_exc()

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

def read_csv_and_generate_basic_schema(csv_file_path, metadata=None):
    """Reads a CSV file and generates a basic schema and relationships."""
    try:
        df = pd.read_csv(csv_file_path)
        if df.empty:
            print("Warning: CSV file is empty")
            return {}, [] # Return empty schema and empty relationships list

        schema = {}
        # Detect relationships separately
        relationships = detect_relationships(df)

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
                "constraints": [], # Constraints will still be per-column
                "post_processing": None,
            }

            # --- Hardcode Primary Key (Override if column is CONTRACT) ---
            if column == "CONTRACT":
                 schema[column]["key_type"] = "Primary Key"
                 print(f"Hardcoded '{column}' as Primary Key in initial schema.")
                 # Also set sdtype for PK if needed later, though create_sdv_metadata handles it
                 # schema[column]["sdtype"] = "id"
            # --- End Hardcode Primary Key ---


        return schema, relationships # Return both schema (columns) and relationships

    except Exception as e:
        print(f"Error generating basic schema or relationships: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None # Return None on error


def enhance_schema_batch_alternative(schema_batch, basic_relationships):
    """
    Enhances a batch of schema using Gemini, parsing a custom text format.
    Includes handling for relationships.
    """
    try:
        # Define the list of supported constraints based on the user's findings
        supported_constraints = ["scalar_range", "unique", "one_hot_encoding", "fixed_combinations", "positive", "negative", "fixed_increments"]

        # --- Prompt for Custom Text Format ---
        # Made supported constraints list more prominent and explicit
        prompt = f"""
        Enhance the schema details and identify potential relationships for the following columns from a Vietnamese banking dataset.
        Provide the enhanced information for each column and list identified relationships using the exact format specified below.
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
        Constraints: [List of SDV constraint objects (JSON format). **IMPORTANT: ONLY use constraint types from this exact list: {', '.join([c.replace('_', ' ').title() for c in supported_constraints])}.** Ensure each constraint is a JSON object with a "type" key and necessary parameters.]
        ---

        Relationships (identified in this batch or related to these columns):
        [List of relationship objects (JSON format). IMPORTANT: DO NOT identify or suggest "functional_dependency" type relationships. Focus on other types like 'value_relationship', 'formula', etc. For 'value_relationship', include 'column1', 'column2', 'relationship' (e.g., '<=', '>='). For 'formula', include 'target_column', 'formula' (e.g., 'col_a + col_b'), 'source_columns' (list of columns used in formula). Ensure column names match exactly.]
        ---

        Here is the schema batch to enhance (includes basic relationships detected):
        {{
             "columns": {json.dumps(schema_batch, indent=2)},
             "relationships": {json.dumps(basic_relationships, indent=2)}
        }}


        Provide ONLY the enhanced information in the specified format, starting with the first '## Column:'.
        """
        # --- End Prompt ---

        # Removed safety_settings from the generate_content call
        response = model.generate_content(prompt, request_options={"timeout": 600})

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
        enhanced_batch_columns = {}
        enhanced_batch_relationships = []

        # Split the response into column blocks and the relationships block
        parts = re.split(r'Relationships \(identified in this batch or related to these columns\):', response_text, 1)
        column_blocks_text = parts[0] if parts else ""
        relationships_block_text = parts[1] if len(parts) > 1 else ""

        # Parse column blocks
        column_blocks = re.split(r'## Column:', column_blocks_text)

        print(f"Attempting to parse {len(column_blocks) - 1} potential column blocks from Gemini response...")

        # Regex to find fields within a column block (updated to include Constraints)
        field_pattern = re.compile(r"^(Description|Domain|Constraints Description|Data Quality|Faker Provider|Faker Args|Post Processing|Constraints):\s*(.*)", re.MULTILINE)


        for i, block in enumerate(column_blocks):
            if i == 0: # Skip the preamble before the first '## Column:'
                continue

            block = block.strip()
            if not block: continue

            lines = block.split('\n', 1)
            if not lines or not lines[0].strip():
                 print(f"  Warning: Could not extract column name from block {i}. Skipping block.")
                 continue

            column_name = lines[0].strip()
            rest_of_block = lines[1] if len(lines) > 1 else ""

            column_details = {}
            field_matches = list(field_pattern.finditer(rest_of_block))

            if not field_matches:
                 print(f"  Warning: No fields found for column '{column_name}'. Skipping enhancement for this column.")
                 continue

            for j, match in enumerate(field_matches):
                field_name_prompt = match.group(1)
                start_of_value_in_block = match.end(1) + 1 # Position after field name and colon

                # Determine the end of the current field's value
                value_end_index_in_block = len(rest_of_block)
                if j + 1 < len(field_matches):
                    next_match_start_index_in_block = field_matches[j+1].start()
                    value_end_index_in_block = next_match_start_index_in_block

                # Extract the full value text for the current field
                full_value_text = rest_of_block[start_of_value_in_block:value_end_index_in_block].strip()


                schema_field_map = {
                    "Description": "description",
                    "Domain": "domain",
                    "Constraints Description": "constraints_description",
                    "Data Quality": "data_quality",
                    "Faker Provider": "faker_provider",
                    "Faker Args": "faker_args",
                    "Post Processing": "post_processing",
                    "Constraints": "constraints" # Added Constraints mapping
                }
                schema_field_name = schema_field_map.get(field_name_prompt)

                if schema_field_name:
                    column_details[schema_field_name] = full_value_text


            # --- Process Faker Provider (Clean extra quotes/prefix) ---
            if "faker_provider" in column_details and column_details["faker_provider"] is not None:
                 provider_text = str(column_details["faker_provider"]).strip()
                 provider_text = re.sub(r'^[\'"]|[\'"]$', '', provider_text)
                 provider_text = provider_text.replace('Faker.', '')
                 column_details["faker_provider"] = provider_text if provider_text else None

            # --- Process Faker Args specifically as JSON (More Robust) ---
            if "faker_args" in column_details:
                faker_args_text_raw = column_details["faker_args"].strip()
                column_details["faker_args"] = {} # Default to empty dict

                if faker_args_text_raw:
                    # Clean the raw text: replace Python 'None' with JSON 'null'
                    cleaned_faker_args_text = faker_args_text_raw.replace('None', 'null')
                    # Also remove the '... ]}' pattern before attempting to find braces
                    cleaned_faker_args_text = cleaned_faker_args_text.replace('... ]}', ']}')

                    # Find the first '{' and the last '}' to isolate the JSON object
                    json_start = cleaned_faker_args_text.find('{')
                    json_end = cleaned_faker_args_text.rfind('}')

                    if json_start != -1 and json_end != -1 and json_end > json_start:
                         # Extract the substring from the first '{' to the last '}'
                         faker_args_json_string = cleaned_faker_args_text[json_start : json_end + 1]
                         try:
                            column_details["faker_args"] = json.loads(faker_args_json_string)
                            # print(f"  Successfully parsed Faker Args for '{column_name}'.")
                         except json.JSONDecodeError as e:
                            print(f"  Warning: Could not parse isolated Faker Args JSON for column '{column_name}': {e}. Raw text: '{faker_args_text_raw}'. Isolated: '{faker_args_json_string}'. Setting faker_args to empty dictionary.")
                         except Exception as e:
                             print(f"  Warning: Unexpected error parsing isolated Faker Args for '{column_name}': {e}. Raw text: '{faker_args_text_raw}'. Setting faker_args to empty dictionary.")
                    else:
                         print(f"  Warning: Could not find valid JSON object {{...}} pattern in Faker Args text for column '{column_name}'. Raw text: '{faker_args_text_raw}'. Setting faker_args to empty dictionary.")
                 # else: # If raw text is empty, faker_args remains {} (the default)


            # --- Process Constraints specifically as JSON List (More Robust) ---
            if "constraints" in column_details:
                 constraints_text_raw = column_details["constraints"].strip()
                 column_details["constraints"] = [] # Default to empty list

                 if constraints_text_raw:
                      # Try to find the first '[' and the last ']' to isolate the JSON list
                      json_list_start = constraints_text_raw.find('[')
                      json_list_end = constraints_text_raw.rfind(']')

                      if json_list_start != -1 and json_list_end != -1 and json_list_end > json_list_start:
                           constraints_json_string = constraints_text_raw[json_list_start : json_list_end + 1]
                           try:
                                # Attempt to parse the isolated JSON string as a list
                                parsed_constraints = json.loads(constraints_json_string)
                                if isinstance(parsed_constraints, list):
                                     # Define the list of supported constraints based on the user's findings
                                     supported_constraints = ["scalar_range", "unique", "one_hot_encoding", "fixed_combinations", "positive", "negative", "fixed_increments"]
                                     filtered_constraints = []
                                     existing_constraint_types = [] # Track types to avoid duplicates

                                     for constraint in parsed_constraints:
                                          if isinstance(constraint, dict):
                                               constraint_type = constraint.get("type", "").lower()
                                               # Normalize constraint type string (remove spaces, convert to lowercase)
                                               normalized_constraint_type = constraint_type.replace(" ", "_")

                                               # Debug print for constraints
                                               # print(f"    Debug: Processing constraint dict for '{column_name}': type='{constraint_type}', normalized='{normalized_constraint_type}'")
                                               # print(f"    Debug: Supported constraints: {supported_constraints}")


                                               if normalized_constraint_type in supported_constraints:
                                                    # Simple check to avoid adding the same constraint type multiple times
                                                    if normalized_constraint_type not in existing_constraint_types:
                                                         # Add the original constraint dict, not the normalized string
                                                         filtered_constraints.append(constraint)
                                                         existing_constraint_types.append(normalized_constraint_type)
                                                    # else: # Optional: log skipped duplicate
                                                         # print(f"    Skipping duplicate supported constraint type '{constraint_type}' for column '{column_name}'.")
                                               else:
                                                    print(f"    Warning: Gemini suggested unsupported constraint type '{constraint_type}' for column '{column_name}'. Skipping.")
                                          # --- Handle String-based Constraints suggested by Gemini ---
                                          elif isinstance(constraint, str):
                                               constraint_str = constraint.strip()
                                               # Attempt to parse the string as JSON first, after cleaning
                                               cleaned_constraint_str = constraint_str.replace('None', 'null').replace('_', '').replace("'", '"') # Clean common issues
                                               try:
                                                    parsed_string_as_json = json.loads(cleaned_constraint_str)
                                                    if isinstance(parsed_string_as_json, dict):
                                                         # If it successfully parsed into a dict, process it like a dict constraint
                                                         constraint_type = parsed_string_as_json.get("type", "").lower()
                                                         normalized_constraint_type = constraint_type.replace(" ", "_")

                                                         # Debug print for constraints
                                                         # print(f"    Debug: Processing constraint string (as JSON) for '{column_name}': raw='{constraint_str}', cleaned='{cleaned_constraint_str}', normalized='{normalized_constraint_type}'")
                                                         # print(f"    Debug: Supported constraints: {supported_constraints}")

                                                         if normalized_constraint_type in supported_constraints:
                                                              if normalized_constraint_type not in existing_constraint_types:
                                                                   # Add the parsed dictionary
                                                                   filtered_constraints.append(parsed_string_as_json)
                                                                   existing_constraint_types.append(normalized_constraint_type)
                                                              # else: # Optional: log skipped duplicate
                                                                   # print(f"    Skipping duplicate supported constraint string (as JSON) '{constraint_str}' for column '{column_name}'.")
                                                         else:
                                                              print(f"    Warning: Gemini suggested unsupported constraint type (from string as JSON) '{constraint_type}' for column '{column_name}'. Skipping.")
                                                    else:
                                                         # If it parsed but wasn't a dict, treat as unsupported string
                                                         print(f"    Warning: Gemini suggested unsupported constraint string format '{constraint_str}' for column '{column_name}' (parsed as {type(parsed_string_as_json)}). Skipping.")

                                               except json.JSONDecodeError:
                                                     # If it didn't parse as JSON, check if it's a simple supported string name
                                                     normalized_constraint_str = constraint_str.lower().replace(" ", "_")

                                                     # Debug print for constraints
                                                     # print(f"    Debug: Processing constraint string (simple) for '{column_name}': raw='{constraint_str}', normalized='{normalized_constraint_str}'")
                                                     # print(f"    Debug: Supported constraints: {supported_constraints}")

                                                     if normalized_constraint_str in supported_constraints:
                                                          # Convert simple string constraint to dictionary format
                                                          constraint_dict = {"type": normalized_constraint_str}
                                                          if normalized_constraint_str == "unique":
                                                               constraint_dict["column_names"] = [column_name]
                                                          # Add other default parameters for other types if necessary
                                                          # For ScalarRange, Positive, Negative, FixedIncrements, FixedCombinations
                                                          # Gemini should ideally provide parameters in the list, but if it only gives the type string,
                                                          # we might not have enough info here. We'll rely on the dict format for those for now.
                                                          # This string handling is primarily for 'unique' and 'one_hot_encoding' if they appear as strings.
                                                          # Also handle 'positive' and 'negative' as they were reported as strings
                                                          if normalized_constraint_str in ["unique", "one_hot_encoding", "positive", "negative", "fixed_increments", "fixed_combinations"]: # Handle these as strings
                                                               if normalized_constraint_str not in existing_constraint_types:
                                                                    filtered_constraints.append(constraint_dict)
                                                                    existing_constraint_types.append(normalized_constraint_str)
                                                               # else: # Optional: log skipped duplicate
                                                                    # print(f"    Skipping duplicate supported constraint string '{constraint_str}' for column '{column_name}'.")
                                                          else:
                                                               print(f"    Warning: Gemini suggested unsupported constraint string '{normalized_constraint_str}' for column '{column_name}'. Skipping.") # Use normalized string in warning

                                                     else:
                                                          print(f"    Warning: Gemini suggested unsupported constraint string '{normalized_constraint_str}' for column '{column_name}'. Skipping.") # Use normalized string in warning

                                               except Exception as e:
                                                    print(f"    Warning: Unexpected error processing constraint string '{constraint_str}' for column '{column_name}': {e}. Skipping.")


                                          else:
                                               print(f"    Warning: Found non-dictionary or non-string item in constraints list for column '{column_name}'. Skipping: {constraint}")

                                     column_details["constraints"] = filtered_constraints
                                     # print(f"  Successfully parsed and filtered Constraints for '{column_name}'.")
                                else:
                                     print(f"  Warning: Parsed Constraints JSON for column '{column_name}' is not a list ({type(parsed_constraints)}). Raw text: '{constraints_text_raw}'. Setting constraints to empty list.")
                                     # Keep default empty list
                           except json.JSONDecodeError as e:
                              print(f"  Warning: Could not parse isolated Constraints JSON list for column '{column_name}': {e}. Raw text: '{constraints_text_raw}'. Isolated: '{constraints_json_string}'. Setting constraints to empty list.")
                              # Keep default empty list
                           except Exception as e:
                               print(f"  Warning: Unexpected error parsing isolated Constraints list for '{column_name}': {e}. Raw text: '{constraints_text_raw}'. Setting constraints to empty list.")
                               # Keep default empty list
                      else:
                           print(f"  Warning: Could not find valid JSON list pattern [...] in Constraints text for column '{column_name}'. Raw text: '{constraints_text_raw}'. Setting constraints to empty list.")
                           # Keep default empty list
                 # else: # If raw text is empty, constraints remains [] (the default)


            # Add the extracted details to the batch result if column name was found
            if column_name:
                if column_name in schema_batch:
                     enhanced_column_info = schema_batch[column_name].copy()
                     enhanced_column_info.update(column_details)
                     enhanced_batch_columns[column_name] = enhanced_column_info
                     # print(f"  Parsed details for column: {column_name}") # Removed print statement
                else:
                     print(f"  Warning: Parsed column '{column_name}' from Gemini response is not in the original schema batch. Skipping.")


        # Parse relationships block
        if relationships_block_text:
             # Try to find the first '[' and the last ']' to isolate the JSON list
             json_list_start = relationships_block_text.find('[')
             json_list_end = relationships_block_text.rfind(']')

             if json_list_start != -1 and json_list_end != -1 and json_list_end > json_list_start:
                  relationships_json_string = relationships_block_text[json_list_start : json_list_end + 1]
                  try:
                       parsed_relationships = json.loads(relationships_json_string)
                       if isinstance(parsed_relationships, list):
                            # Filter relationships: ONLY add if NOT functional_dependency
                            filtered_relationships = []
                            for rel in parsed_relationships:
                                 if isinstance(rel, dict):
                                      rel_type = rel.get("type", "").lower()
                                      if rel_type != "functional_dependency":
                                           filtered_relationships.append(rel)
                                           # print(f"  Added non-functional dependency relationship: {rel.get('type')}") # Optional logging
                                      # else: # Optional: log skipped functional dependency
                                           # print(f"  Skipping functional dependency relationship suggested by Gemini.")
                                 else:
                                      print(f"  Warning: Found non-dictionary item in relationships list. Skipping: {rel}")

                            enhanced_batch_relationships = filtered_relationships
                            # print(f"  Successfully parsed and filtered relationships from Gemini response.") # Removed print statement
                       else:
                            print(f"  Warning: Parsed Relationships JSON is not a list ({type(parsed_relationships)}). Raw text: '{relationships_block_text}'. Setting relationships to empty list.")
                            # Keep default empty list
                  except json.JSONDecodeError as e:
                     print(f"  Warning: Could not parse isolated Relationships JSON list: {e}. Raw text: '{relationships_block_text}'. Setting relationships to empty list.")
                     # Keep default empty list
                  except Exception as e:
                      print(f"  Warning: Unexpected error parsing isolated Relationships list: {e}. Raw text: '{relationships_block_text}'. Setting relationships to empty list.")
                      # Keep default empty list
             else:
                  print(f"  Warning: Could not find valid JSON list pattern [...] in Relationships text. Raw text: '{relationships_block_text}'. Setting relationships to empty list.")
                  # Keep default empty list


        return enhanced_batch_columns, enhanced_batch_relationships # Return both enhanced columns and relationships

    except Exception as e:
        print(f"Error during custom format parsing in alternative enhancement approach: {str(e)}")
        import traceback
        traceback.print_exc() # Print traceback for the error
        return schema_batch, basic_relationships # Return original batch info on error


def enhance_schema_batch(schema_batch, basic_relationships):
    """Enhances a batch of schema information using the Gemini API."""
    print("Attempting schema enhancement using the alternative text parsing approach...")
    return enhance_schema_batch_alternative(schema_batch, basic_relationships)


def enhance_schema_with_gemini(schema, relationships):
    """Enhances the schema and relationships using the Gemini API in batches."""
    try:
        column_names = list(schema.keys())
        num_columns = len(column_names)
        num_batches = math.ceil(num_columns / BATCH_SIZE)
        print(f"Processing {num_columns} columns in {num_batches} batches of {BATCH_SIZE}...")

        # Initialize enhanced schema structure
        enhanced_schema = {
            "columns": {},
            "relationships": []
        }
        # Start with the basic schema and relationships
        enhanced_schema["columns"].update(schema)
        # Add basic relationships, avoiding duplicates if Gemini also suggests them later
        basic_relationship_keys = set()
        for rel in relationships:
             # Create a simple key for basic relationships to check for duplicates later
             # Functional dependency key: (type, source, target)
             # Value relationship key: (type, col1, col2, relationship) - sort cols for consistency
             if rel.get("type") == "functional_dependency":
                  key = ("functional_dependency", rel.get("source_column"), rel.get("target_column"))
             elif rel.get("type") == "value_relationship":
                  key_cols = tuple(sorted((rel.get("column1"), rel.get("column2"))))
                  key = ("value_relationship", key_cols, rel.get("relationship"))
             elif rel.get("type") == "formula": # Handle potential new formula type
                  key = ("formula", rel.get("target_column"), rel.get("formula")) # Simple key for formula
             else:
                  # For other types, create a key from sorted items
                  key = tuple(sorted(rel.items()))

             if key not in basic_relationship_keys:
                  enhanced_schema["relationships"].append(rel)
                  basic_relationship_keys.add(key)


        for i in range(num_batches):
            start_idx = i * BATCH_SIZE
            end_idx = min((i + 1) * BATCH_SIZE, num_columns)
            batch_columns = column_names[start_idx:end_idx]
            print(f"Processing batch {i+1}/{num_batches} with columns: {batch_columns}")

            # Get basic schema and relevant basic relationships for the batch
            schema_batch = {col: schema[col] for col in batch_columns}
            # Filter basic relationships to include only those where source or target is in the current batch
            basic_relationships_batch = [
                 rel for rel in relationships
                 if (rel.get("source_column") in batch_columns or rel.get("target_column") in batch_columns or
                     rel.get("column1") in batch_columns or rel.get("column2") in batch_columns or
                     # Check for formula relationships if target or source columns are in batch
                     (rel.get("type") == "formula" and (rel.get("target_column") in batch_columns or any(src in batch_columns for src in rel.get("source_columns", []))))
                    )
            ]


            # Call enhance_schema_batch (which now calls the alternative approach)
            enhanced_batch_columns_result, enhanced_batch_relationships_result = enhance_schema_batch(schema_batch, basic_relationships_batch)

            # Update the enhanced_schema with the results from the batch
            # This will overwrite the basic schema info if enhancement was successful
            if enhanced_batch_columns_result:
                 enhanced_schema["columns"].update(enhanced_batch_columns_result)
                 print(f"Successfully processed column details for batch {i+1}.")
            else:
                 print(f"Warning: Batch {i+1} column enhancement failed or returned empty. Keeping basic schema for these columns.")

            # Add enhanced relationships from Gemini, avoiding duplicates with existing ones
            if enhanced_batch_relationships_result:
                 for rel in enhanced_batch_relationships_result:
                      # Create a key for the new relationship
                      if rel.get("type") == "functional_dependency": # Should be filtered out by parsing, but double check
                           continue # Skip functional dependencies from Gemini

                      elif rel.get("type") == "value_relationship":
                           key_cols = tuple(sorted((rel.get("column1"), rel.get("column2"))))
                           key = ("value_relationship", key_cols, rel.get("relationship"))
                      elif rel.get("type") == "formula": # Handle potential new formula type
                           # Formula key should include target and formula string
                           key = ("formula", rel.get("target_column"), rel.get("formula"))
                      else:
                           # Generic key for other types
                           key = tuple(sorted(rel.items()))


                      # Check if this relationship (or a similar one) is already in the enhanced schema
                      is_duplicate_or_similar = False
                      for existing_rel in enhanced_schema["relationships"]:
                           # Simple check: If types and key columns/properties match
                           if existing_rel.get("type") == rel.get("type"):
                                if rel.get("type") == "functional_dependency": # Should be filtered
                                     continue
                                elif rel.get("type") == "value_relationship":
                                     existing_key_cols = tuple(sorted((existing_rel.get("column1"), existing_rel.get("column2"))))
                                     if existing_key_cols == key_cols and existing_rel.get("relationship") == rel.get("relationship"):
                                          is_duplicate_or_similar = True
                                          break
                                elif rel.get("type") == "formula":
                                     # Compare target column and formula string for formula type
                                     if existing_rel.get("target_column") == rel.get("target_column") and existing_rel.get("formula") == rel.get("formula"):
                                          is_duplicate_or_similar = True
                                          break
                                else: # Generic check for other types
                                     if tuple(sorted(existing_rel.items())) == key:
                                          is_duplicate_or_similar = True
                                          break

                      if not is_duplicate_or_similar:
                           enhanced_schema["relationships"].append(rel)
                           # print(f"  Added non-functional dependency relationship: {rel.get('type')}") # Optional logging
                      # else: # Optional logging for skipping duplicates
                           # print(f"  Skipping duplicate or similar relationship from Gemini: {rel.get('type')}")


            # else: # Warning about failed relationship parsing is already in enhance_schema_batch_alternative
                 # print(f"Warning: Batch {i+1} relationship enhancement failed or returned empty.")


        return enhanced_schema # Return the fully enhanced schema (or basic if enhancement failed)

    except Exception as e:
        print(f"Error in batch enhancement process: {str(e)}")
        import traceback
        traceback.print_exc()
        print("Returning original basic schema due to enhancement process error.")
        # If overall process fails, return a structure containing the basic schema and relationships
        return {
            "columns": schema,
            "relationships": relationships
        }


def main():
    """Main function to generate the enhanced schema."""
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Input file '{INPUT_CSV}' does not exist.")
        return

    print("Starting schema generation process")
    metadata = read_metadata(METADATA_JSON)

    print(f"Generating basic schema and relationships from {INPUT_CSV}...")
    # read_csv_and_generate_basic_schema now returns schema (columns) and relationships
    schema, relationships = read_csv_and_generate_basic_schema(INPUT_CSV, metadata)
    if schema is None or relationships is None: # Check if either returned None due to error
        print("Failed to generate basic schema or relationships. Exiting.")
        return

    basic_schema_structure = {
        "columns": schema,
        "relationships": relationships
    }

    basic_schema_file = "basic_schema.json"
    with open(basic_schema_file, 'w', encoding='utf-8') as f:
        json.dump(basic_schema_structure, f, indent=2, ensure_ascii=False)
    print(f"Basic schema structure saved to {basic_schema_file}")

    print("Enhancing schema structure with Gemini...")
    # Pass the basic schema (columns) and relationships to the enhancement process
    enhanced_schema_structure = enhance_schema_with_gemini(schema, relationships)

    # Save the final enhanced schema structure
    try:
        with open(OUTPUT_SCHEMA_JSON, 'w', encoding='utf-8') as f:
            json.dump(enhanced_schema_structure, f, indent=2, ensure_ascii=False)
        print(f"Final schema structure (enhanced or basic fallback) saved to {OUTPUT_SCHEMA_JSON}")
    except Exception as e:
        print(f"Error saving final schema structure to {OUTPUT_SCHEMA_JSON}: {str(e)}")
        import traceback
        traceback.print_exc()


    print("Schema generation completed.")

if __name__ == "__main__":
    main()
