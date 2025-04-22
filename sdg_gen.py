import pandas as pd
import json
import os
import numpy as np
from sdv.single_table import GaussianCopulaSynthesizer
# No longer need specific constraint imports as we bypass add_constraints
# from sdv.constraints.tabular import ScalarRange, Unique
from sdv.metadata import SingleTableMetadata
import random
from faker import Faker
import inspect
import warnings
import re # Import re for functional dependency mapping
from datetime import datetime, date, timedelta # Import datetime, date, and timedelta for date parsing and Faker args
import string # Import string for character sets

# Configuration
INPUT_CSV = "customer_data.csv"  # Input CSV file
OUTPUT_CSV = "synthetic_data.csv"  # Output CSV file for synthetic data
# Change this to read the ENHANCED schema
INPUT_SCHEMA_JSON = "enhanced_schema.json"  # Input JSON file for the ENHANCED schema
NUM_ROWS = 1000  # Number of synthetic rows to generate
CORRELATION_THRESHOLD = 0.7  # Define the correlation threshold (used in schema generation)

# Initialize Faker
fake = Faker(['vi_VN'])
Faker.seed(42)

# --- Helper Functions ---

def read_schema(json_file_path):
    """Reads the schema (basic or enhanced) from a JSON file."""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)
        print(f"Successfully read schema from {json_file_path}")
        return schema
    except FileNotFoundError:
        print(f"Error: Schema file not found at {json_file_path}")
        print("Please run the schema generation script first to create enhanced_schema.json")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {json_file_path}: {str(e)}")
        print("Please ensure the schema file is valid JSON.")
        return None
    except Exception as e:
        print(f"Error reading schema: {str(e)}")
        return None

# Keep this function as it's used for correlation reporting
def detect_column_correlations(df):
    """
    Detects column correlations, including functional dependencies and basic value relationships.
    Note: This function is copied from the schema generation script for correlation reporting.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        dict: A dictionary where keys are column names and values are lists of detected relationships.
    """
    correlations = {}
    RELATIONSHIP_CONSISTENCY_THRESHOLD = 0.999 # Define threshold for value relationship consistency (e.g., 99.9%)
    # Use the same CORRELATION_THRESHOLD from the main script config
    global CORRELATION_THRESHOLD

    try:
        # Select only columns that are purely numeric (int or float)
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        # --- Pearson Correlation ---
        if len(numeric_cols) >= 2:
            try:
                # Ensure all selected columns are indeed numeric before calculating correlation
                df_numeric = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
                df_numeric = df_numeric.dropna(axis=1, how='all') # Drop columns that became all NaN
                numeric_cols_cleaned = df_numeric.columns.tolist() # Update numeric_cols list after cleaning

                if len(numeric_cols_cleaned) >= 2:
                     corr_matrix = df_numeric.corr(method='pearson', numeric_only=True) # numeric_only=True is default

                     for i, col1 in enumerate(numeric_cols_cleaned):
                         for j, col2 in enumerate(numeric_cols_cleaned):
                             if i < j:
                                 corr_value = corr_matrix.loc[col1, col2]
                                 if abs(corr_value) >= CORRELATION_THRESHOLD and not pd.isna(corr_value):
                                     if col1 not in correlations:
                                         correlations[col1] = []
                                     correlations[col1].append({
                                         "column": col2,
                                         "correlation": float(round(corr_value, 3)), # Ensure JSON serializable float
                                         "type": "pearson_correlation" # Use specific type name
                                     })
                                     # Add reciprocal correlation
                                     if col2 not in correlations:
                                         correlations[col2] = []
                                     correlations[col2].append({
                                         "column": col1,
                                         "correlation": float(round(corr_value, 3)), # Ensure JSON serializable float
                                         "type": "pearson_correlation"
                                     })
                     # print(f"Detected Pearson correlations above threshold {CORRELATION_THRESHOLD}.") # Suppress in synthesis script
                else:
                     pass # print("Not enough valid numeric columns after cleaning for Pearson correlation calculation.")

            except Exception as pearson_e:
                 print(f"Error during Pearson correlation calculation for reporting: {pearson_e}")


        # --- Functional Dependency (Simple Check: one value maps to only one other value) ---
        all_cols = df.columns.tolist() # Check dependency across all columns
        # print("\nChecking for functional dependencies...") # Suppress in synthesis script
        for col1 in all_cols:
            for col2 in all_cols:
                if col1 != col2:
                    try:
                        # Drop rows where either column is NaN for this check
                        df_filtered = df[[col1, col2]].dropna()
                        if not df_filtered.empty:
                            if pd.api.types.is_hashable(df_filtered[col1].dtype):
                                if not df_filtered[col1].empty:
                                     if pd.api.types.is_hashable(df_filtered[col1].iloc[0]):
                                        unique_values_per_group = df_filtered.groupby(col1)[col2].nunique()
                                        if (unique_values_per_group <= 1).all():
                                            if col1 not in correlations:
                                                correlations[col1] = []
                                            if not any(d.get("column") == col2 and d.get("type") == "functional_dependency" for d in correlations[col1]):
                                                correlations[col1].append({
                                                    "column": col2,
                                                    "correlation": 1.0, # Represent dependency
                                                    "type": "functional_dependency"
                                                })
                                                # print(f"Detected potential functional dependency: {col1} -> {col2}") # Suppress in synthesis script
                    except Exception as dep_e:
                        pass # Suppress errors


        # --- Basic Value Relationships (e.g., col1 <= col2, col1 >= col2) ---
        # Check for consistent inequality relationships between numeric columns
        # print("\nChecking for basic value relationships (e.g., <=, >=)...") # Suppress in synthesis script
        if len(numeric_cols) >= 2:
             for col1 in numeric_cols:
                 for col2 in numeric_cols:
                     if col1 != col2:
                         try:
                             # Drop rows where either column is NaN for this check
                             df_filtered = df[[col1, col2]].dropna()
                             if not df_filtered.empty:
                                 num_rows_checked = len(df_filtered)

                                 # Check col1 <= col2
                                 if num_rows_checked > 0 and (df_filtered[col1] <= df_filtered[col2]).sum() / num_rows_checked >= RELATIONSHIP_CONSISTENCY_THRESHOLD:
                                     if col1 not in correlations:
                                         correlations[col1] = []
                                     if not any(d.get("column") == col2 and d.get("relationship") == "less_than_or_equal_to" for d in correlations[col1]):
                                         correlations[col1].append({
                                             "column": col2,
                                             "relationship": "less_than_or_equal_to",
                                             "type": "value_relationship" # Use specific type
                                         })
                                         # print(f"Detected value relationship: {col1} <= {col2} (holds for >= {RELATIONSHIP_CONSISTENCY_THRESHOLD:.1%} of data)") # Suppress

                                 # Check col1 >= col2
                                 if num_rows_checked > 0 and (df_filtered[col1] >= df_filtered[col2]).sum() / num_rows_checked >= RELATIONSHIP_CONSISTENCY_THRESHOLD:
                                     if col1 not in correlations:
                                         correlations[col1] = []
                                     if not any(d.get("column") == col2 and d.get("relationship") == "greater_than_or_equal_to" for d in correlations[col1]):
                                         correlations[col1].append({
                                             "column": col2,
                                             "relationship": "greater_than_or_equal_to",
                                             "type": "value_relationship" # Use specific type
                                         })
                                         # print(f"Detected value relationship: {col1} >= {col2} (holds for >= {RELATIONSHIP_CONSISTENCY_THRESHOLD:.1%} of data)") # Suppress

                         except Exception as rel_e:
                             pass # print(f"Could not check value relationship for {col1} vs {col2} for reporting: {rel_e}")


    except Exception as e:
        print(f"Error detecting correlations for reporting: {str(e)}")
        import traceback
        traceback.print_exc()

    # Clean up empty correlation lists
    correlations = {col: rels for col, rels in correlations.items() if rels}

    return correlations


def create_sdv_metadata(df, schema):
    """Creates SDV metadata from the DataFrame and enhanced schema."""
    try:
        metadata = SingleTableMetadata()
        # Detect initial metadata, including data types
        metadata.detect_from_dataframe(data=df)
        print("Initial metadata detected.")
        # print(metadata.to_dict()) # Optional: print initial detection

        primary_key = None

        # Update metadata based on the enhanced schema
        for col_name, col_schema in schema.items():
            if col_name not in df.columns:
                print(f"Warning: Column '{col_name}' from schema not found in DataFrame. Skipping metadata update.")
                continue

            sdtype_override = None

            # --- Determine Primary Key from schema ---
            if col_schema.get("key_type", "").lower() == "primary key":
                if primary_key is None:
                    primary_key = col_name
                    sdtype_override = 'id' # Set sdtype for primary key
                    print(f"Setting primary key: '{primary_key}' with sdtype 'id'")
                else:
                    # Handle composite keys if necessary, though SDV single table often assumes one PK
                    print(f"Warning: Multiple primary keys defined in schema ('{primary_key}', '{col_name}'). Using the first one found: '{primary_key}'.")

            # --- Determine Data Type (sdtype) from schema ---
            data_type = col_schema.get("data_type", "").lower()
            if data_type in ["numerical", "integer", "float"]:
                 # Ensure SDV's sdtype matches schema's intent if needed
                 current_sdtype = metadata.columns.get(col_name, {}).get('sdtype')
                 if current_sdtype not in ['numerical', 'float', 'integer']:
                      # print(f"Overriding sdtype for '{col_name}' from '{current_sdtype}' to 'numerical' based on schema.")
                      sdtype_override = 'numerical' # Override if detection was different

            elif data_type == "categorical":
                current_sdtype = metadata.columns.get(col_name, {}).get('sdtype')
                if current_sdtype != 'categorical':
                    # print(f"Overriding sdtype for '{col_name}' from '{current_sdtype}' to 'categorical' based on schema.")
                    sdtype_override = 'categorical'

            elif data_type == "datetime":
                # print(f"Setting sdtype for '{col_name}' to 'datetime' based on schema.")
                sdtype_override = 'datetime' # Always override if schema says datetime

            elif data_type == "boolean":
                 current_sdtype = metadata.columns.get(col_name, {}).get('sdtype')
                 if current_sdtype != 'boolean':
                    # print(f"Overriding sdtype for '{col_name}' from '{current_sdtype}' to 'boolean' based on schema.")
                    sdtype_override = 'boolean'

            elif data_type == "id":
                 # Can be used for non-primary key identifiers
                 current_sdtype = metadata.columns.get(col_name, {}).get('sdtype')
                 if current_sdtype != 'id':
                    # print(f"Overriding sdtype for '{col_name}' from '{current_sdtype}' to 'id' based on schema.")
                    sdtype_override = 'id'


            # --- Apply sdtype Update to Metadata ---
            if sdtype_override:
                try:
                    metadata.update_column(column_name=col_name, sdtype=sdtype_override)
                    # print(f"Updated sdtype for column '{col_name}' to '{sdtype_override}'.")
                except Exception as update_e:
                    print(f"Error updating sdtype for column '{col_name}' to '{sdtype_override}': {update_e}")


        # --- Set Primary Key after iterating through all columns ---
        if primary_key:
             try:
                 metadata.set_primary_key(column_name=primary_key)
                 print(f"Successfully set '{primary_key}' as the primary key in metadata.")
             except Exception as pk_e:
                 print(f"Error setting primary key '{primary_key}': {pk_e}")
        else:
            print("Warning: No primary key explicitly defined in the schema or detected.")


        print("Final metadata after applying schema.")
        # print(metadata.to_dict()) # Optional: print final metadata for verification
        try:
            metadata.validate() # Validate the final metadata
            print("Metadata validation successful.")
        except Exception as validate_e:
             print(f"Metadata validation failed: {validate_e}")
             # Decide if you want to proceed with potentially invalid metadata
             # return None # Or raise error

        return metadata

    except Exception as e:
        print(f"Error creating SDV metadata: {str(e)}")
        import traceback
        traceback.print_exc()
        # Fallback to basic detection if schema application fails
        print("Falling back to basic metadata detection.")
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=df)
        try:
            metadata.validate()
        except Exception as validate_e:
             print(f"Fallback metadata validation failed: {validate_e}")
        return metadata

# --- Custom Regex Generation Function ---
def generate_from_regex_pattern(pattern):
    """
    Generates a string matching a simplified regex pattern.
    Handles # (digit), ?, *, +, {}, [], and literal characters.
    Does NOT support full regex syntax.
    """
    if not isinstance(pattern, str):
        return None # Cannot process non-string patterns

    result = []
    i = 0
    while i < len(pattern):
        char = pattern[i]

        if char == '#':
            result.append(random.choice(string.digits))
            i += 1
        elif char == '[':
            # Find the end of the character set
            end_bracket = pattern.find(']', i + 1)
            if end_bracket == -1:
                print(f"Warning: Unclosed '[' in regex pattern: {pattern}. Skipping.")
                return "".join(result) # Return current result
            char_set_content = pattern[i + 1:end_bracket]
            possible_chars = []

            # Parse character set content (simplified: handles a-z, A-Z, 0-9, and literals)
            j = 0
            while j < len(char_set_content):
                if j + 2 < len(char_set_content) and char_set_content[j+1] == '-':
                    start_char = char_set_content[j]
                    end_char = char_set_content[j+2]
                    if ord(start_char) <= ord(end_char):
                        possible_chars.extend([chr(k) for k in range(ord(start_char), ord(end_char) + 1)])
                    j += 3
                else:
                    possible_chars.append(char_set_content[j])
                    j += 1

            if possible_chars:
                result.append(random.choice(possible_chars))
            else:
                print(f"Warning: Empty or unparseable character set in regex pattern: {pattern}. Skipping.")
                # Decide on fallback: append a placeholder or skip? Let's skip for now.
                pass # Do nothing, effectively skipping this part of the pattern

            i = end_bracket + 1
        elif char == '{':
            # Find the end of the repetition count
            end_brace = pattern.find('}', i + 1)
            if end_brace == -1:
                print(f"Warning: Unclosed '{{' in regex pattern: {pattern}. Skipping.")
                return "".join(result) # Return current result

            count_str = pattern[i + 1:end_brace]
            repetition_count = 1 # Default to 1 if parsing fails
            try:
                if ',' in count_str:
                    min_max = count_str.split(',')
                    min_rep = int(min_max[0].strip()) if min_max[0].strip() else 0
                    max_rep = int(min_max[1].strip()) if min_max[1].strip() else min_rep # If max is empty, use min
                    repetition_count = random.randint(min_rep, max_rep)
                else:
                    repetition_count = int(count_str.strip())
            except ValueError:
                print(f"Warning: Invalid repetition count '{{{count_str}}}' in regex pattern: {pattern}. Using count 1.")
                repetition_count = 1 # Fallback to 1 on error

            # Apply repetition to the *last* added character/group
            if result:
                last_char_or_group = result.pop() # Remove the last element
                result.extend([last_char_or_group] * repetition_count) # Add it back the specified number of times
            else:
                 print(f"Warning: Repetition '{{{count_str}}}' found at the start or after an empty group in regex pattern: {pattern}. Ignoring repetition.")

            i = end_brace + 1

        elif char == '\\':
             # Handle escaped characters (e.g., '\-') - simplified
             if i + 1 < len(pattern):
                  result.append(pattern[i+1])
                  i += 2
             else:
                  print(f"Warning: Trailing '\\' in regex pattern: {pattern}. Ignoring.")
                  i += 1 # Just skip the '\'
        # Simplified handling for quantifiers ?, *, +
        elif char in '?*+':
             # Apply to the *last* added character/group if it exists
             if result:
                  last_char_or_group = result.pop() # Remove the last element
                  if char == '?': # 0 or 1
                       if random.random() < 0.5: # 50% chance to include
                            result.append(last_char_or_group)
                  elif char == '*': # 0 or more
                       num_repetitions = random.randint(0, 5) # Arbitrary max repetition
                       result.extend([last_char_or_group] * num_repetitions)
                  elif char == '+': # 1 or more
                       num_repetitions = random.randint(1, 5) # Arbitrary max repetition
                       result.extend([last_char_or_group] * num_repetitions)
             else:
                  print(f"Warning: Quantifier '{char}' found at the start or after an empty group in regex pattern: {pattern}. Ignoring.")
             i += 1
        else:
            # Literal character
            result.append(char)
            i += 1

    return "".join(result)


# --- New Function for Post-processing Constraints (Method 1 for Functional Dependencies) ---
def apply_post_processing_rules(df, schema, original_df):
    """
    Applies post-processing rules to the synthetic DataFrame based on the schema.
    Includes Faker application (excluding problematic providers), custom regex generation,
    preserved categorical distributions, functional dependencies (Method 1),
    value relationships, range constraints, and uniqueness enforcement.

    Args:
        df (pd.DataFrame): The synthetic DataFrame to modify.
        schema (dict): The enhanced schema dictionary.
        original_df (pd.DataFrame): The original DataFrame, used for functional dependency mapping and sampling.

    Returns:
        pd.DataFrame: The DataFrame after applying post-processing rules.
    """
    print("\nApplying post-processing constraints and Faker...")

    # List to keep track of columns that should use custom regex generation
    custom_regex_columns = []

    # --- Apply Faker providers and preserved categorical distributions ---
    # This is now done BEFORE functional dependencies.
    print("  Applying preserved categorical distributions and Faker providers...")
    def generate_weighted_random_element(categories):
        # Ensure categories is a list of dicts with 'value' and 'percentage'
        if not isinstance(categories, list) or not all(isinstance(item, dict) and 'value' in item and 'percentage' in item for item in categories):
             # print(f"     - Warning: Invalid categories data format for weighted random selection.") # Reduced logging
             if isinstance(categories, list) and categories:
                  # print("     - Attempting uniform random choice from available list elements.") # Reduced logging
                  try:
                       return random.choice(categories)
                  except Exception:
                       return None
             return None

        values = [item["value"] for item in categories]
        weights = [item["percentage"] for item in categories]

        if not all(isinstance(w, (int, float)) for w in weights) or sum(weights) <= 0:
             # print(f"     - Warning: Invalid or non-positive sum of weights for weighted random selection. Using uniform random choice.") # Reduced logging
             if values:
                 return random.choice(values)
             else:
                 return None

        try:
            return random.choices(values, weights=weights, k=1)[0]
        except Exception as e:
             # print(f"     - Error during weighted random selection: {e}. Falling back to uniform.") # Reduced logging
             if values:
                  return random.choice(values)
             else:
                  return None


    for col_name, col_schema in schema.items():
         if col_name not in df.columns: continue

         is_cat_in_schema = col_schema.get("data_type") == "categorical"
         categories_data = None
         if "stats" in col_schema:
             categories_data = col_schema["stats"].get("categories") or col_schema["stats"].get("top_categories")

         faker_provider = col_schema.get("faker_provider")

         # --- Apply Faker ONLY if provider is specified and not None ---
         if faker_provider is not None:
             if isinstance(faker_provider, str):
                 cleaned_provider = str(faker_provider).strip()
                 cleaned_provider = re.sub(r'^[\'"]|[\'"]$', '', cleaned_provider)
                 cleaned_provider = cleaned_provider.replace('Faker.', '').replace('Faker::Date.', '').replace('Faker::Regex.', '')

                 provider_name_map = {
                     'regex': 'regexify', 'date': 'date_object', 'date_between': 'date_object', 'bban': 'bban'
                 }
                 cleaned_provider = provider_name_map.get(cleaned_provider, cleaned_provider)

                 # --- Handle 'regexify' specifically with custom logic ---
                 if cleaned_provider == 'regexify':
                      print(f"     - Using custom regex generation for column '{col_name}'")
                      custom_regex_columns.append(col_name)
                      continue # Skip Faker application for this column


                 # --- Apply other Faker providers if found ---
                 if hasattr(fake, cleaned_provider):
                      # print(f"     - Applying Faker provider '{cleaned_provider}' to column '{col_name}'") # Reduced logging
                      faker_args = col_schema.get("faker_args", {}).copy()

                      # --- Argument Filtering and Conversion based on Provider ---
                      if cleaned_provider == 'pyfloat':
                           faker_args.pop('step', None)
                           faker_args.pop('precision', None)
                           if 'min_value' in faker_args and faker_args['min_value'] is not None:
                                if faker_args['min_value'] > 0:
                                     faker_args['positive'] = True
                                elif faker_args['min_value'] <= 0 and 'positive' in faker_args:
                                     faker_args.pop('positive', None)
                           elif 'positive' in faker_args:
                                pass

                      elif cleaned_provider == 'random_element':
                           faker_args.pop('weights', None)


                      elif cleaned_provider == 'date_object':
                           faker_args.pop('pattern', None)
                           for date_arg_key in ['start_date', 'end_date']:
                                if date_arg_key in faker_args:
                                     date_val = faker_args[date_arg_key]
                                     if isinstance(date_val, str):
                                          date_str = date_val.strip()
                                          if date_str.lower() == 'today':
                                               faker_args[date_arg_key] = date.today()
                                               continue

                                          relative_match = re.match(r'^([+-]?\d+)([ymd])$', date_str)
                                          if relative_match:
                                               value = int(relative_match.group(1))
                                               unit = relative_match.group(2)
                                               try:
                                                    today = date.today()
                                                    if unit == 'y':
                                                         faker_args[date_arg_key] = today + timedelta(days=value * 365)
                                                    elif unit == 'm':
                                                         faker_args[date_arg_key] = today + timedelta(days=value * 30)
                                                    elif unit == 'd':
                                                         faker_args[date_arg_key] = today + timedelta(days=value)
                                               except Exception as parse_e:
                                                    print(f"         Warning: Could not parse relative date string '{date_str}' for '{col_name}': {parse_e}. Keeping original value.")
                                          else:
                                               try:
                                                    date_formats_to_try = ['%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y', '%Y/%m/%d', '%Y%m%d']
                                                    parsed_date = None
                                                    for fmt in date_formats_to_try:
                                                         try:
                                                              parsed_date = datetime.strptime(date_str.split(' ')[0], fmt).date()
                                                              break
                                                         except ValueError:
                                                              pass

                                                    if parsed_date:
                                                         faker_args[date_arg_key] = parsed_date
                                                    else:
                                                          print(f"         Warning: Could not parse date string '{date_str}' for '{col_name}' with common formats. Keeping original value.")
                                               except Exception as parse_e:
                                                    print(f"         Warning: Error parsing date string '{date_str}' for '{col_name}': {parse_e}. Keeping original value.")
                                     elif not isinstance(date_val, date):
                                          print(f"         Warning: {date_arg_key} arg for '{col_name}' is not a string or date object ({type(date_val)}). Keeping original value.")


                      # General Faker provider handling (including random_element, but NOT regexify)
                      try:
                          faker_method = getattr(fake, cleaned_provider)
                          valid_args = inspect.signature(faker_method).parameters
                          filtered_args = {k: v for k, v in faker_args.items() if k in valid_args}

                          if cleaned_provider == 'random_element' and 'elements' in filtered_args:
                               if not isinstance(filtered_args['elements'], list):
                                    print(f"         Warning: 'elements' argument for random_element on '{col_name}' is not a list ({type(filtered_args['elements'])}). Skipping Faker.")
                                    continue

                          df.loc[:, col_name] = df[col_name].apply(
                              lambda _: faker_method(**filtered_args)
                          )
                      except AttributeError:
                          print(f"       Warning: Faker provider '{cleaned_provider}' not found on fake object. Keeping generated values for '{col_name}'.")
                      except Exception as e:
                          print(f"       Error applying Faker provider '{cleaned_provider}' to '{col_name}': {str(e)}")
                 else:
                     # print(f"     - Skipping Faker application for column '{col_name}': cleaned faker_provider name is empty.") # Reduced logging
                     pass

         # --- Preserve Categorical Distribution if Faker was NOT applied or is None ---
         current_faker_provider_status = schema.get(col_name, {}).get("faker_provider")
         cleaned_provider_status = None
         if isinstance(current_faker_provider_status, str):
              cleaned_provider_status = str(current_faker_provider_status).strip()
              cleaned_provider_status = re.sub(r'^[\'"]|[\'"]$', '', cleaned_provider_status)
              cleaned_provider_status = cleaned_provider_status.replace('Faker.', '').replace('Faker::Date.', '').replace('Faker::Regex.', '')
              provider_name_map = {
                  'regex': 'regexify', 'date': 'date_object', 'date_between': 'date_object', 'bban': 'bban'
              }
              cleaned_provider_status = provider_name_map.get(cleaned_provider_status, cleaned_provider_status)


         if is_cat_in_schema and categories_data and col_name not in custom_regex_columns:
              faker_applied_successfully = False
              if cleaned_provider_status and hasattr(fake, cleaned_provider_status) and cleaned_provider_status != 'regexify':
                   faker_applied_successfully = True

              if not faker_applied_successfully:
                  # print(f"     - Preserving value distribution for categorical column '{col_name}' (Faker not applicable/found).") # Reduced logging
                  df.loc[:, col_name] = df[col_name].apply(
                      lambda _: generate_weighted_random_element(categories_data)
                  )


    # --- Apply Custom Regex Generation ---
    # This is now done AFTER Faker.
    print("  Applying custom regex generation...")
    for col_name in custom_regex_columns:
         if col_name in df.columns:
              col_schema = schema.get(col_name, {})
              faker_args = col_schema.get("faker_args", {})
              regex_pattern = faker_args.get("text") or faker_args.get("pattern")

              if isinstance(regex_pattern, str):
                   print(f"    - Generating values for '{col_name}' using custom regex pattern: '{regex_pattern}'")
                   try:
                        df.loc[:, col_name] = df[col_name].apply(
                             lambda _: generate_from_regex_pattern(regex_pattern)
                        )
                   except Exception as e:
                        print(f"    - Error applying custom regex generation to '{col_name}': {e}")
              else:
                   print(f"    - Skipping custom regex generation for '{col_name}': No valid regex pattern found in faker_args.")


    # --- Log state after Faker/Custom Regex and Categorical Distribution ---
    print("\n--- State after Faker/Custom Regex and Categorical Distribution ---")
    print(df.head())
    print(df.nunique()) # Check unique values per column


    # --- Apply Post-processing Constraints (Functional Dependencies (Method 1), Value Relationships, Ranges, Uniqueness) ---

    # Apply functional dependencies (Method 1) - This is now done AFTER Faker/Categorical/Custom Regex
    print("  Applying functional dependencies (Method 1)...")
    if original_df is not None:
        for col_name, col_schema in schema.items():
            if col_name in df.columns and "correlations" in col_schema:
                for correlation in col_schema["correlations"]:
                    if correlation.get("type") == "functional_dependency":
                        source_col = col_name
                        target_col = correlation.get("column")

                        # --- Reduced Logging Before Functional Dependency ---
                        # print(f"\n--- Debugging Functional Dependency: {source_col} -> {target_col} ---")
                        # if original_df is not None:
                        #      print("\nOriginal DataFrame unique values in source column (for mapping):")
                        #      print(original_df[source_col].dropna().unique())
                        #      print("\nOriginal DataFrame unique values in target column (for sampling):")
                        #      print(original_df[target_col].dropna().unique())
                        # print("\nSynthetic DataFrame unique values in source column (before applying dependency):")
                        # print(df[source_col].dropna().unique())
                        # --- End Reduced Logging ---


                        if target_col in df.columns and source_col != target_col:
                            try:
                                print(f"    - Enforcing functional dependency: {source_col} -> {target_col} (Method 1)")

                                # 1. Create mapping from original data (handle NaN and duplicates)
                                original_mapping = original_df.dropna(subset=[source_col, target_col]).drop_duplicates(subset=[source_col], keep='first').set_index(source_col)[target_col].to_dict()

                                # --- Reduced Logging of Mappings ---
                                # print("\nOriginal Mapping (from original_df):")
                                # print(original_mapping)
                                # --- End Reduced Logging ---


                                # 2. Get unique synthetic source values
                                unique_synthetic_source_values = df[source_col].dropna().unique()

                                # 3. Create a mapping for synthetic data
                                synthetic_mapping = {}
                                original_target_values_for_sampling = original_df[target_col].dropna().tolist()

                                if not original_target_values_for_sampling:
                                     print(f"      - Warning: Original target column '{target_col}' has no non-null values for sampling. Functional dependency for {source_col} -> {target_col} cannot be fully enforced for new values.")


                                for synthetic_source_value in unique_synthetic_source_values:
                                    if synthetic_source_value in original_mapping:
                                        synthetic_mapping[synthetic_source_value] = original_mapping[synthetic_source_value]
                                    else:
                                        if original_target_values_for_sampling:
                                             sampled_target_value = random.choice(original_target_values_for_sampling)
                                             synthetic_mapping[synthetic_source_value] = sampled_target_value


                                # --- Reduced Logging of Mappings ---
                                # print("\nSynthetic Mapping (used for .map()):")
                                # print(synthetic_mapping)
                                # --- End Reduced Logging ---


                                # 4. Apply the synthetic mapping to the synthetic DataFrame
                                mapped_series = df[source_col].map(synthetic_mapping)

                                # --- Reduced Logging of Mapped Series ---
                                # print("\nResult of .map() operation (mapped_series):")
                                # print(mapped_series.head())
                                print(f"      - NaN count after .map() for {source_col} -> {target_col}: {mapped_series.isna().sum()}")
                                # --- End Reduced Logging ---


                                # Identify rows where mapping was successful (not NaN after map)
                                mapped_mask = mapped_series.notna()

                                original_values_preserved_count = (~mapped_mask).sum()

                                if mapped_mask.any():
                                     df.loc[mapped_mask, target_col] = mapped_series[mapped_mask]
                                     print(f"      - Updated {mapped_mask.sum()} values in '{target_col}' based on '{source_col}' functional dependency (Method 1).")
                                else:
                                     print(f"      - No values updated in '{target_col}' for functional dependency {source_col} -> {target_col} (Method 1): No successful mappings.")


                                if original_values_preserved_count > 0:
                                     print(f"      - Preserved {original_values_preserved_count} original SDV-generated values in '{target_col}' where functional dependency mapping was not applicable.")


                            except Exception as e:
                                print(f"    - Error applying functional dependency '{source_col} -> {target_col}' (Method 1): {e}")

    else:
        print("  Skipping functional dependency enforcement: Original data not loaded for mapping.")

    # --- Log state after Functional Dependencies ---
    print("\n--- State after Functional Dependencies ---")
    print(df.head())
    print(df.nunique())


    # Apply detected value relationships (e.g., colA <= colB) - Do this AFTER functional dependencies
    print("  Applying detected value relationships...")
    for col_name, col_schema in schema.items():
        if col_name in df.columns and "correlations" in col_schema:
            for correlation in col_schema["correlations"]:
                if correlation.get("type") == "value_relationship":
                    target_col = correlation.get("column")
                    relationship = correlation.get("relationship")

                    if target_col in df.columns and col_name != target_col:
                        if pd.api.types.is_numeric_dtype(df[col_name]) and pd.api.types.is_numeric_dtype(df[target_col]):
                            try:
                                violations_mask = None
                                if relationship == "less_than_or_equal_to":
                                    violations_mask = (df[col_name] > df[target_col]) & df[col_name].notna() & df[target_col].notna()
                                    if violations_mask is not None and violations_mask.any():
                                        num_violations = violations_mask.sum()
                                        df.loc[violations_mask, col_name] = pd.to_numeric(df.loc[violations_mask, target_col], errors='coerce')
                                        print(f"    - Enforced '{col_name} <= {target_col}' on {num_violations} rows by setting '{col_name}' = '{target_col}'.")

                                elif relationship == "greater_than_or_equal_to":
                                    violations_mask = (df[col_name] < df[target_col]) & df[col_name].notna() & df[target_col].notna()
                                    if violations_mask is not None and violations_mask.any():
                                        num_violations = violations_mask.sum()
                                        df.loc[violations_mask, col_name] = pd.to_numeric(df.loc[violations_mask, target_col], errors='coerce')
                                        print(f"    - Enforced '{col_name} >= {target_col}' on {num_violations} rows by setting '{col_name}' = '{target_col}'.")

                            except Exception as e:
                                print(f"    - Error applying value relationship '{col_name} {relationship} {target_col}': {e}")


    # --- Log state after Value Relationships ---
    print("\n--- State after Value Relationships ---")
    print(df.head())
    print(df.nunique())


    # Apply range constraints (based on min/max from basic stats) - Do this AFTER relationships
    print("  Applying range constraints...")
    for col_name, col_schema in schema.items():
        if col_name in df.columns:
            data_type = col_schema.get("data_type")
            stats = col_schema.get("stats", {})
            min_val = stats.get("min")
            max_val = stats.get("max")

            if data_type in ['numerical', 'integer', 'float'] and min_val is not None and max_val is not None:
                try:
                    df[col_name] = pd.to_numeric(df[col_name], errors='coerce')

                    original_violations = df[(df[col_name] < min_val) | (df[col_name] > max_val)].shape[0]
                    if original_violations > 0:
                         df[col_name] = df[col_name].clip(lower=min_val, upper=max_val)
                         print(f"    - Clipped {original_violations} values in '{col_name}' to range [{min_val}, {max_val}].")

                except Exception as e:
                    print(f"    - Error applying range constraint to '{col_name}' (min={min_val}, max={max_val}): {e}")

    # --- Log state after Range Constraints ---
    print("\n--- State after Range Constraints ---")
    print(df.head())
    print(df.nunique())


    # Apply uniqueness constraint on Primary Key (if identified) - Do this LAST
    print("  Enforcing uniqueness on Primary Key...")
    primary_key = None
    for col_name, col_schema in schema.items():
        if col_schema.get("key_type", "").lower() == "primary key":
            primary_key = col_name
            break

    if primary_key and primary_key in df.columns:
        try:
            duplicates_mask = df.duplicated(subset=[primary_key], keep='first')
            duplicate_count = duplicates_mask.sum()

            if duplicate_count > 0:
                print(f"    - Found {duplicate_count} duplicate values in primary key column '{primary_key}'. Attempting to regenerate.")
                duplicate_indices = df[duplicates_mask].index

                pk_schema = schema.get(primary_key, {})
                faker_provider = pk_schema.get("faker_provider")
                faker_args = pk_schema.get("faker_args", {}).copy()

                current_max_id = None
                if pd.api.types.is_numeric_dtype(df[primary_key]):
                     try:
                         current_max_id = df[primary_key].max()
                         if pd.isna(current_max_id): current_max_id = 0
                     except Exception:
                          current_max_id = 0


                for index in duplicate_indices:
                    new_value = None
                    if faker_provider:
                        try:
                            if isinstance(faker_provider, str):
                                cleaned_provider = str(faker_provider).strip()
                                cleaned_provider = re.sub(r'^[\'"]|[\'"]$', '', cleaned_provider)
                                cleaned_provider = cleaned_provider.replace('Faker.', '').replace('Faker::Date.', '').replace('Faker::Regex.', '')
                                provider_name_map = {
                                     'regex': 'regexify', 'date': 'date_object', 'date_between': 'date_object', 'bban': 'bban'
                                }
                                cleaned_provider = provider_name_map.get(cleaned_provider, cleaned_provider)

                                if cleaned_provider == 'regexify':
                                     regex_pattern = faker_args.get("text") or faker_args.get("pattern")
                                     if isinstance(regex_pattern, str):
                                          retry_count = 0
                                          while new_value is None or (new_value in df[primary_key].values and retry_count < 20):
                                               new_value = generate_from_regex_pattern(regex_pattern)
                                               retry_count += 1
                                          if new_value in df[primary_key].values:
                                               print(f"      - Warning: Could not generate a unique value for duplicate at index {index} using custom regex after retries. Falling back to sequential/placeholder.")
                                               new_value = None
                                     else:
                                          print(f"      - Warning: No valid regex pattern found in faker_args for PK '{primary_key}' with regexify. Falling back to sequential/placeholder.")
                                          new_value = None

                                elif cleaned_provider and hasattr(fake, cleaned_provider):
                                     faker_method = getattr(fake, cleaned_provider)
                                     retry_count = 0
                                     while new_value is None or (new_value in df[primary_key].values and retry_count < 20):
                                         current_faker_args = faker_args.copy()
                                         if cleaned_provider == 'random_int':
                                              current_max_val = df[primary_key].max()
                                              if pd.isna(current_max_val): current_max_val = 0
                                              current_faker_args['min'] = int(current_max_val) + 1
                                              if 'max' not in current_faker_args or current_faker_args['max'] <= current_faker_args['min']:
                                                   current_faker_args['max'] = current_faker_args['min'] + 100000

                                         valid_args = inspect.signature(faker_method).parameters
                                         filtered_args = {k: v for k, v in current_faker_args.items() if k in valid_args}

                                         new_value = faker_method(**filtered_args)
                                         retry_count += 1

                                     if new_value in df[primary_key].values:
                                          print(f"      - Warning: Could not generate a unique value for duplicate at index {index} using Faker after retries. Falling back to sequential/placeholder.")
                                          new_value = None
                                else:
                                     print(f"      - Warning: Cleaned Faker provider name for PK '{primary_key}' is empty or not found. Falling back to sequential/placeholder.")
                                     new_value = None

                            else:
                                 print(f"      - Warning: Faker provider for PK '{primary_key}' is not a string ({type(faker_provider)}). Falling back to sequential/placeholder.")
                                 new_value = None


                        except Exception as e:
                            print(f"      - Error generating new ID for duplicate at index {index} with Faker: {e}. Falling back to sequential/placeholder.")
                            new_value = None

                    if new_value is None:
                        if pd.api.types.is_numeric_dtype(df[primary_key]) and current_max_id is not None:
                             current_max_id += 1
                             new_value = current_max_id
                             while new_value in df[primary_key].values:
                                  current_max_id += 1
                                  new_value = current_max_id
                        else:
                             new_value = f"GENERATED_DUP_{index}_{random.randint(1000, 9999)}"
                             print(f"      - Warning: Could not regenerate unique non-numeric ID for duplicate at index {index}. Using placeholder '{new_value}'.")


                    df.loc[index, primary_key] = new_value

                print(f"    - Attempted to regenerate {duplicate_count} duplicate primary key values.")

        except Exception as e:
            print(f"    - Error enforcing uniqueness on primary key '{primary_key}': {e}")
    elif not primary_key:
         print("    - Skipping uniqueness enforcement: No primary key defined in schema.")
    else:
         print(f"    - Skipping uniqueness enforcement: Primary key column '{primary_key}' not found in synthetic data.")

    # --- Log state after Uniqueness Enforcement ---
    print("\n--- State after Uniqueness Enforcement ---")
    print(df.head())
    print(df.nunique())


    print("Post-processing constraints applied.")
    return df


def generate_synthetic_data_with_sdv(df, schema, num_rows, output_file):
    """Generates synthetic data using SDV, applies post-processing constraints, Faker, etc."""
    try:
        print("\n--- Starting SDV Synthetic Data Generation ---")
        print("1. Creating SDV metadata from DataFrame and enhanced schema...")
        metadata = create_sdv_metadata(df, schema)
        if not metadata:
            print("Error: Failed to create metadata. Aborting.")
            return False

        print("\n2. Setting up SDV synthesizer (GaussianCopulaSynthesizer)...")
        synthesizer = GaussianCopulaSynthesizer(metadata=metadata)


        print("\n3. Fitting synthesizer to the original data...")
        try:
             with warnings.catch_warnings():
                synthesizer.fit(df)
             print("   Synthesizer fitting completed.")
        except Exception as fit_e:
            print(f"   Error fitting synthesizer: {fit_e}")
            import traceback
            traceback.print_exc()
            print("   Aborting due to fitting error.")
            return False


        print(f"\n4. Generating {num_rows} rows of synthetic data...")
        try:
            with warnings.catch_warnings():
                synthetic_data = synthesizer.sample(num_rows=num_rows)
            print(f"   Successfully generated {len(synthetic_data)} raw synthetic rows.")
        except Exception as sample_e:
            print(f"   Error during sampling: {sample_e}")
            import traceback
            traceback.print_exc()
            print("   Aborting due to sampling error.")
            return False

        # --- Post-processing Steps ---
        original_df_for_postprocessing = None
        try:
             original_df_for_postprocessing = pd.read_csv(INPUT_CSV, encoding='utf-8')
             try:
                 original_df_for_postprocessing = pd.read_csv(INPUT_CSV, encoding='utf-8')
             except UnicodeDecodeError:
                 print("UTF-8 decoding failed for original data reload for post-processing, trying latin-1...")
                 original_df_for_postprocessing = pd.read_csv(INPUT_CSV, encoding='latin-1')
             print("\n   Loaded original data for post-processing (functional dependency mapping, mode calculation).")
        except Exception as e:
             print(f"\n   Error loading original data for post-processing: {e}. Functional dependencies and mode filling cannot be applied.")
             original_df_for_postprocessing = None


        print("\n5. Applying post-processing (Faker, categorical distributions, custom rules, constraints)...")
        synthetic_data = apply_post_processing_rules(synthetic_data, schema, original_df_for_postprocessing)

        if original_df_for_postprocessing is not None:
             del original_df_for_postprocessing


        print("\n6. Comparing correlations between original and synthetic data and generating report...")
        original_df_for_corr = None
        try:
             original_df_for_corr = pd.read_csv(INPUT_CSV, encoding='utf-8')
             try:
                 original_df_for_corr = pd.read_csv(INPUT_CSV, encoding='utf-8')
             except UnicodeDecodeError:
                 print("UTF-8 decoding failed for original data reload for correlation report, trying latin-1...")
                 original_df_for_corr = pd.read_csv(INPUT_CSV, encoding='latin-1')
             print("   Loaded original data for correlation reporting.")

             original_correlations_detected = detect_column_correlations(original_df_for_corr)

             original_numeric_cols = original_df_for_corr.select_dtypes(include=np.number).columns.tolist()
             potential_numeric_synthetic = synthetic_data[[col for col in original_numeric_cols if col in synthetic_data.columns]]

             numeric_synthetic = potential_numeric_synthetic.apply(pd.to_numeric, errors='coerce')

             numeric_synthetic = numeric_synthetic.dropna(axis=1, how='all')

             synthetic_correlations_detected = detect_column_correlations(numeric_synthetic)


             correlation_report_data = []
             all_relationships = {}

             for col1, rels in original_correlations_detected.items():
                 for rel in rels:
                     col2 = rel.get("column")
                     rel_type = rel.get("type")
                     rel_detail = rel.get("relationship")

                     if col2:
                         key_cols = tuple(sorted((col1, col2))) if rel_type == "pearson_correlation" else (col1, col2)
                         key = (key_cols, rel_type, rel_detail)

                         if key not in all_relationships:
                             all_relationships[key] = {
                                 "Column 1": col1,
                                 "Column 2": col2,
                                 "Type": rel_type,
                                 "Relationship": rel_detail if rel_detail else "",
                                 "Original Value": None,
                                 "Synthetic Value": None
                             }

                         if rel_type == "pearson_correlation":
                             all_relationships[key]["Original Value"] = rel.get("correlation")
                         elif rel_type == "functional_dependency":
                             all_relationships[key]["Original Value"] = "Exists"
                         elif rel_type == "value_relationship":
                              all_relationships[key]["Original Value"] = rel_detail


             for col1, rels in synthetic_correlations_detected.items():
                  for rel in rels:
                     col2 = rel.get("column")
                     rel_type = rel.get("type")
                     rel_detail = rel.get("relationship")

                     if col2:
                         key_cols = tuple(sorted((col1, col2))) if rel_type == "pearson_correlation" else (col1, col2)
                         key = (key_cols, rel_type, rel_detail)

                         if key not in all_relationships:
                              all_relationships[key] = {
                                 "Column 1": col1,
                                 "Column 2": col2,
                                 "Type": rel_type,
                                 "Relationship": rel_detail if rel_detail else "",
                                 "Original Value": "Not Detected",
                                 "Synthetic Value": None
                             }

                         if rel_type == "pearson_correlation":
                             all_relationships[key]["Synthetic Value"] = rel.get("correlation")
                         elif rel_type == "functional_dependency":
                             all_relationships[key]["Synthetic Value"] = "Exists"
                         elif rel_type == "value_relationship":
                              all_relationships[key]["Synthetic Value"] = rel_detail


             correlation_report_data = list(all_relationships.values())

             if correlation_report_data:
                 correlation_df = pd.DataFrame(correlation_report_data)
                 column_order = ["Column 1", "Column 2", "Type", "Relationship", "Original Value", "Synthetic Value"]
                 correlation_df = correlation_df[column_order]

                 correlation_report_csv_file = f"{os.path.splitext(output_file)[0]}_correlation_report.csv"
                 try:
                     correlation_df.to_csv(correlation_report_csv_file, index=False, encoding='utf-8')
                     print(f"   Correlation comparison report saved to {correlation_report_csv_file}")
                 except Exception as csv_e:
                     print(f"   Error saving correlation report CSV: {csv_e}")
             else:
                 print("   No correlations or relationships detected for reporting.")


        except Exception as corr_e:
            print(f"   Error during correlation comparison step: {corr_e}")
        finally:
             if 'original_df_for_corr' in locals() and original_df_for_corr is not None:
                  del original_df_for_corr


        print(f"\n7. Saving final synthetic data to {output_file}...")
        synthetic_data.to_csv(output_file, index=False, encoding='utf-8')
        print(f"   Successfully generated and saved {len(synthetic_data)} rows of synthetic data!")
        print("\n--- Synthetic Data Generation Finished ---")
        return True

    except Exception as e:
        print(f"\n--- Error during Synthetic Data Generation ---")
        print(f"An unexpected error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to orchestrate the synthetic data generation process."""
    print("--- Starting Script ---")
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Input data file '{INPUT_CSV}' not found.")
        return

    print(f"\nReading enhanced schema from: {INPUT_SCHEMA_JSON}")
    schema = read_schema(INPUT_SCHEMA_JSON)
    if not schema:
        print("Failed to read the enhanced schema. Please run the schema generation script first to create enhanced_schema.json. Exiting.")
        return


    print(f"\nLoading original data from: {INPUT_CSV}")
    try:
        try:
            original_df = pd.read_csv(INPUT_CSV, encoding='utf-8')
        except UnicodeDecodeError:
            print("UTF-8 decoding failed, trying latin-1...")
            original_df = pd.read_csv(INPUT_CSV, encoding='latin-1')

        print(f"Loaded original data: {original_df.shape[0]} rows, {original_df.shape[1]} columns")

    except FileNotFoundError:
         print(f"Error: Input data file '{INPUT_CSV}' not found during loading.")
         return
    except Exception as e:
        print(f"Error loading or preparing original data from '{INPUT_CSV}': {str(e)}")
        return

    print(f"\nInitiating synthetic data generation for {NUM_ROWS} rows...")
    success = generate_synthetic_data_with_sdv(original_df, schema, NUM_ROWS, OUTPUT_CSV)

    print("\n--- Script Finished ---")
    if success:
        print(f"Synthetic data generation process completed successfully.")
        print(f"Output saved to: {OUTPUT_CSV}")
        report_file_csv = f"{os.path.splitext(OUTPUT_CSV)[0]}_correlation_report.csv"
        if os.path.exists(report_file_csv):
             print(f"Correlation report saved to: {report_file_csv}")

    else:
        print("Synthetic data generation process encountered errors.")

if __name__ == "__main__":
    main()
