import pandas as pd
import json
import os
import numpy as np
from sdv.single_table import GaussianCopulaSynthesizer
# No longer need specific constraint imports as we bypass add_constraints for most
# from sdv.constraints.tabular import ScalarRange, Unique, Inequality, Formula
from sdv.metadata import SingleTableMetadata
import random
from faker import Faker
import inspect
import warnings
import re # Import re for functional dependency mapping and regex generation
from datetime import datetime, date, timedelta # Import datetime, date, and timedelta for date parsing and Faker args
import string # Import string for character sets
from collections import defaultdict # Import defaultdict for relationship detection
from itertools import combinations # Import combinations for formula detection in reporting

# Configuration
INPUT_CSV = "customer_data.csv"  # Input CSV file
OUTPUT_CSV = "synthetic_data.csv"  # Output CSV file for synthetic data
# Change this to read the ENHANCED schema
INPUT_SCHEMA_JSON = "enhanced_schema.json"  # Input JSON file for the ENHANCED schema
NUM_ROWS = 1000  # Number of synthetic rows to generate
CORRELATION_THRESHOLD = 0.7  # Threshold for reporting Pearson correlation
RELATIONSHIP_CONSISTENCY_THRESHOLD = 0.999 # Threshold for functional dependency and value relationship consistency in reporting
FORMULA_TOLERANCE = 1e-6 # Tolerance for floating point comparisons in formula detection for reporting
REPORT_CSV = "relationship_comparison_report.csv" # Output file for the relationship report

# Initialize Faker
fake = Faker(['vi_VN'])
Faker.seed(42) # Seed Faker for reproducibility

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

# --- Custom Regex Generation Function ---
def generate_from_regex_pattern(pattern):
    """
    Generates a string matching a simplified regex pattern.
    Handles # (digit), ?, *, +, {}, [], and literal characters.
    Does NOT support full regex syntax.
    Added support for character ranges like [a-z], [A-Z], [0-9].
    Added support for non-capturing groups (?:...).
    """
    if not isinstance(pattern, str):
        return None # Cannot process non-string patterns

    result = []
    i = 0
    while i < len(pattern):
        char = pattern[i]

        if char == '\\':
             # Handle escaped characters (e.g., '\.', '\d')
             if i + 1 < len(pattern):
                  next_char = pattern[i+1]
                  if next_char == 'd': # \d for digit
                       result.append(random.choice(string.digits))
                  elif next_char == '.': # \. for literal dot
                       result.append('.')
                  # Add more escaped characters if needed
                  else:
                       result.append(next_char) # Treat as literal if not a special escape
                  i += 2
             else:
                  print(f"Warning: Trailing '\\' in regex pattern: {pattern}. Ignoring.")
                  i += 1 # Just skip the '\'
        elif char == '#': # Custom shorthand for digit
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

            # Parse character set content (handles ranges and literals)
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
        elif char == '(':
             # Simple handling for non-capturing group (?:...)
             if i + 3 < len(pattern) and pattern[i+1:i+3] == '?:':
                  end_paren = pattern.find(')', i + 3)
                  if end_paren != -1:
                       group_content = pattern[i + 3:end_paren]
                       # Recursively process the group content
                       generated_group_content = generate_from_regex_pattern(group_content)
                       if generated_group_content is not None:
                            result.append(generated_group_content)
                       i = end_paren + 1
                  else:
                       print(f"Warning: Unclosed non-capturing group '(?:' in regex pattern: {pattern}. Skipping.")
                       return "".join(result) # Return current result
             else:
                  # Treat as literal if not a non-capturing group
                  result.append(char)
                  i += 1
        elif char == ')':
             # Ignore closing parenthesis if not part of a handled group
             i += 1
        else:
            # Literal character
            result.append(char)
            i += 1

    return "".join(result)


# --- New Function for Post-processing Constraints (Method 1) ---
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


    for col_name, col_schema in schema.get("columns", {}).items(): # Iterate through columns defined in schema
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
                      # This is a simplified version; a more robust approach would inspect Faker method signatures
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
                           # Faker's random_element takes 'elements' as a list or dict {value: weight}
                           # If schema has categories with weights, use that
                           if categories_data:
                                # Convert categories list to dict {value: percentage}
                                elements_dict = {item['value']: item['percentage'] for item in categories_data if 'value' in item and 'percentage' in item}
                                if elements_dict:
                                     faker_args['elements'] = elements_dict
                                     faker_args.pop('weights', None) # Ensure weights arg is not also present
                                else:
                                     print(f"         Warning: Categories data for '{col_name}' is invalid for random_element. Skipping Faker.")
                                     continue # Skip Faker if categories data is bad for this purpose
                           elif 'elements' not in faker_args:
                                print(f"         Warning: random_element for '{col_name}' has no 'elements' arg and no categories data. Skipping Faker.")
                                continue # Skip if no elements provided


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
                                                         faker_args[date_arg_key] = today + timedelta(days=value * 365) # Approximation for years
                                                    elif unit == 'm':
                                                         faker_args[date_arg_key] = today + timedelta(days=value * 30) # Approximation for months
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
                          # Inspecting signature can be complex with *args, **kwargs, etc.
                          # A simpler approach is to filter args based on common parameter names
                          # Or just pass all args and let Faker handle errors (less safe)
                          # Let's filter based on common parameter names for now
                          common_faker_params = ['min', 'max', 'elements', 'pattern', 'start_date', 'end_date', 'digits', 'right_digits', 'locale', 'country_code', 'area_code', 'extension', 'length', 'data_range', 'types', 'levels', 'zone', 'model', 'make', 'year', 'prefix', 'suffix', 'amount', 'currency', 'vat_id_type', 'country', 'city_prefix', 'city_suffix', 'street_name', 'building_number', 'postcode', 'state', 'state_abbr', 'latitude', 'longitude', 'ean_type', 'isbn_type', 'protocol', 'domain_name', 'tld', 'user_name', 'email', 'name', 'first_name', 'last_name', 'ssn', 'iban', 'bban', 'swift8', 'swift11', 'credit_card_number', 'credit_card_expire', 'credit_card_security_code', 'file_extension', 'mime_type', 'binary', 'size', 'text', 'max_nb_chars', 'min_nb_chars', 'ext_word_list', 'nb_sentences', 'variable_nb_sentences', 'nb_words', 'variable_nb_words']
                          filtered_args = {k: v for k, v in faker_args.items() if k in common_faker_params}

                          # Special handling for random_int min/max if they are floats in schema
                          if cleaned_provider == 'random_int':
                               if 'min' in filtered_args and isinstance(filtered_args['min'], float):
                                    filtered_args['min'] = int(filtered_args['min'])
                               if 'max' in filtered_args and isinstance(filtered_args['max'], float):
                                    filtered_args['max'] = int(filtered_args['max'])


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
         # Check if Faker was successfully applied (by checking if it was 'regexify' or a valid method)
         faker_applied_successfully = False
         if faker_provider is not None and isinstance(faker_provider, str):
              cleaned_provider_status = str(faker_provider).strip()
              cleaned_provider_status = re.sub(r'^[\'"]|[\'"]$', '', cleaned_provider_status)
              cleaned_provider_status = cleaned_provider_status.replace('Faker.', '').replace('Faker::Date.', '').replace('Faker::Regex.', '')
              provider_name_map = {
                  'regex': 'regexify', 'date': 'date_object', 'date_between': 'date_object', 'bban': 'bban'
              }
              cleaned_provider_status = provider_name_map.get(cleaned_provider_status, cleaned_provider_status)

              if cleaned_provider_status == 'regexify' and col_name in custom_regex_columns:
                   faker_applied_successfully = True # Custom regex counts as Faker applied
              elif cleaned_provider_status and hasattr(fake, cleaned_provider_status):
                   faker_applied_successfully = True


         if is_cat_in_schema and categories_data and not faker_applied_successfully:
              # If Faker wasn't applied (e.g., provider not found, or was regexify)
              # and it's a categorical column with category data, use weighted random sampling
              # print(f"     - Preserving value distribution for categorical column '{col_name}' (Faker not applicable/found).") # Reduced logging
              df.loc[:, col_name] = df[col_name].apply(
                  lambda _: generate_weighted_random_element(categories_data)
              )


    # --- Apply Custom Regex Generation ---
    # This is now done AFTER Faker.
    print("  Applying custom regex generation...")
    for col_name in custom_regex_columns:
         if col_name in df.columns:
              col_schema = schema.get("columns", {}).get(col_name, {}) # Get from 'columns' key
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
    print("\n--- State after Initial Generation (Faker/Categorical/Regex) ---")
    print(df.head())
    print(df.nunique())


    # --- Apply Post-processing Constraints (Functional Dependencies (Method 1), Value Relationships, Formulas, Ranges, Uniqueness) ---
    # Order of application: FD -> Value Relationships -> Formulas -> Ranges -> Uniqueness

    # Apply functional dependencies (Method 1) - This is now done AFTER Faker/Categorical/Custom Regex
    print("\n  Applying functional dependencies (Method 1)...")
    # Extract functional dependencies from the schema's post_processing lists
    functional_dependencies_to_apply = {}
    for col_name, col_schema in schema.get("columns", {}).items():
        if col_name in df.columns and "post_processing" in col_schema:
            for rule in col_schema["post_processing"]:
                if rule.get("type") == "functional_dependency":
                    source_col = col_name
                    target_col = rule.get("column")
                    if source_col not in functional_dependencies_to_apply:
                         functional_dependencies_to_apply[source_col] = []
                    functional_dependencies_to_apply[source_col].append(rule)


    if original_df is not None:
        for source_col, rel_list in functional_dependencies_to_apply.items():
            for rel in rel_list:
                target_col = rel.get("column")

                if source_col in df.columns and target_col in df.columns and \
                   source_col in original_df.columns and target_col in original_df.columns:

                    print(f"    - Enforcing functional dependency: {source_col} -> {target_col} (Method 1)")

                    try:
                        # 1. Create mapping from original data (handle NaN and duplicates)
                        original_mapping = original_df.dropna(subset=[source_col, target_col]).drop_duplicates(subset=[source_col], keep='first').set_index(source_col)[target_col].to_dict()

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


                        # 4. Apply the synthetic mapping to the synthetic DataFrame
                        # Use .map() - values not in synthetic_mapping will become NaN
                        mapped_series = df[source_col].map(synthetic_mapping)

                        # Identify rows where mapping was successful (not NaN after map)
                        mapped_mask = mapped_series.notna()

                        # Only update the target column where the mapping was successful
                        if mapped_mask.any():
                             df.loc[mapped_mask, target_col] = mapped_series[mapped_mask]
                             print(f"      - Updated {mapped_mask.sum()} values in '{target_col}' based on '{source_col}' functional dependency (Method 1).")
                        else:
                             print(f"      - No values updated in '{target_col}' for functional dependency {source_col} -> {target_col} (Method 1): No successful mappings.")

                        # Note: Handling of rows where synthetic_source_value was NOT in original_mapping
                        # and original_target_values_for_sampling was empty is implicitly handled:
                        # mapped_series will be NaN for these, and the .loc[mapped_mask] assignment
                        # will simply not update these rows, preserving whatever value was there.


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
    # Extract value relationships from the schema's post_processing lists
    value_relationships_to_apply = {}
    for col_name, col_schema in schema.get("columns", {}).items():
        if col_name in df.columns and "post_processing" in col_schema:
            for rule in col_schema["post_processing"]:
                if rule.get("type") == "value_relationship":
                    source_col = col_name
                    target_col = rule.get("column")
                    if source_col not in value_relationships_to_apply:
                         value_relationships_to_apply[source_col] = []
                    value_relationships_to_apply[source_col].append(rule)


    for col_name, rel_list in value_relationships_to_apply.items():
        for rel in rel_list:
            target_col = rel.get("column")
            relationship = rel.get("relationship")

            if target_col in df.columns and col_name != target_col:
                # Ensure both columns are numeric before attempting comparison
                if pd.api.types.is_numeric_dtype(df[col_name]) and pd.api.types.is_numeric_dtype(df[target_col]):
                    try:
                        violations_mask = None
                        if relationship in ["<=", "less_than_or_equal_to"]:
                            violations_mask = (df[col_name] > df[target_col]) & df[col_name].notna() & df[target_col].notna()
                            if violations_mask is not None and violations_mask.any():
                                num_violations = violations_mask.sum()
                                # Set the violating column's value to the target column's value
                                df.loc[violations_mask, col_name] = pd.to_numeric(df.loc[violations_mask, target_col], errors='coerce')
                                print(f"    - Enforced '{col_name} <= {target_col}' on {num_violations} rows by setting '{col_name}' = '{target_col}'.")

                        elif relationship in [">=", "greater_than_or_equal_to"]:
                            violations_mask = (df[col_name] < df[target_col]) & df[col_name].notna() & df[target_col].notna()
                            if violations_mask is not None and violations_mask.any():
                                num_violations = violations_mask.sum()
                                # Set the violating column's value to the target column's value
                                df.loc[violations_mask, col_name] = pd.to_numeric(df.loc[violations_mask, target_col], errors='coerce')
                                print(f"    - Enforced '{col_name} >= {target_col}' on {num_violations} rows by setting '{col_name}' = '{target_col}'.")

                    except Exception as e:
                        print(f"    - Error applying value relationship '{col_name} {relationship} {target_col}': {e}")
                else:
                    print(f"    - Skipping value relationship '{col_name} {relationship} {target_col}': One or both columns are not numeric.")


    # --- Log state after Value Relationships ---
    print("\n--- State after Value Relationships ---")
    print(df.head())
    print(df.nunique())


    # Apply formula relationships (e.g., colA = colB + colC) - Do this AFTER value relationships
    print("  Applying formula relationships...")
    # Extract formula relationships from the schema's post_processing lists
    formula_relationships_to_apply = {}
    for col_name, col_schema in schema.get("columns", {}).items():
        if col_name in df.columns and "post_processing" in col_schema:
            for rule in col_schema["post_processing"]:
                if rule.get("type") == "formula":
                    target_col = rule.get("target_column")
                    source_cols = rule.get("source_columns")
                    formula_str = rule.get("formula")
                    if target_col not in formula_relationships_to_apply:
                         formula_relationships_to_apply[target_col] = []
                    formula_relationships_to_apply[target_col].append(rule)


    for target_col, rel_list in formula_relationships_to_apply.items():
        for rel in rel_list:
            source_cols = rel.get("source_columns")
            formula_str = rel.get("formula")

            # Ensure target column and all source columns exist and are numeric
            if target_col in df.columns and all(col in df.columns and pd.api.types.is_numeric_dtype(df[col]) for col in source_cols):
                
                    print(f"    - Enforcing formula: {target_col} = {formula_str}")

                    # Evaluate the formula using the synthetic DataFrame
                    # Use .eval() for formula evaluation
                    # Handle potential errors during evaluation (e.g., division by zero if formula includes division)
                    try:
                         # Ensure formula string is safe for eval (only allows column names and basic ops)
                         # A more robust approach would parse the formula string.
                         # For now, assume formula_str is safe (e.g., "colA + colB", "colA - colB", "colA * colB", "colA / colB")
                         # Check if all parts of the formula string are valid column names or operators
                         # This is a simplified check; a full parser would be more robust.
                         valid_formula = True
                         formula_parts = re.findall(r'[a-zA-Z0-9_]+|[+\-*/]', formula_str) # Find column names and operators
                         for part in formula_parts:
                              if re.match(r'[a-zA-Z0-9_]+', part) and part not in df.columns:
                                   print(f"      - Warning: Formula '{formula_str}' contains unknown column '{part}'. Skipping formula enforcement.")
                                   valid_formula = False
                                   break
                              if re.match(r'[+\-*/]', part) is None and re.match(r'[a-zA-Z0-9_]+', part) is None:
                                   print(f"      - Warning: Formula '{formula_str}' contains invalid characters or operators. Skipping formula enforcement.")
                                   valid_formula = False
                                   break


                         if valid_formula:
                              # Use .eval() to calculate the expected value based on the formula
                              # Handle potential division by zero or other evaluation errors
                              with np.errstate(divide='ignore', invalid='ignore'): # Ignore division warnings/errors
                                   expected_values = df.eval(formula_str)

                              # Identify rows where the current value in target_col does NOT match the expected value
                              # Use np.isclose for floating point comparison
                              # Handle NaN values during comparison - isclose treats NaN as not close
                              violations_mask = ~np.isclose(df[target_col], expected_values, atol=FORMULA_TOLERANCE, equal_nan=True) & df[target_col].notna() & expected_values.notna()

                              if violations_mask.any():
                                   num_violations = violations_mask.sum()
                                   # Update the target column with the calculated expected values where there were violations
                                   df.loc[violations_mask, target_col] = expected_values[violations_mask]
                                   print(f"      - Enforced formula '{target_col} = {formula_str}' on {num_violations} rows.")
                              else:
                                   print(f"      - Formula '{target_col} = {formula_str}' already holds for all relevant rows.")

                         else:
                              print(f"    - Skipping formula enforcement for '{target_col} = {formula_str}' due to invalid formula string.")

                    except Exception as eval_e:
                        print(f"      - Error evaluating or applying formula '{formula_str}' for '{target_col}': {eval_e}. Skipping.")

            else:
                missing_cols = [col for col in source_cols + [target_col] if col not in df.columns]
                non_numeric_cols = [col for col in source_cols + [target_col] if col in df.columns and not pd.api.types.is_numeric_dtype(df[col])]
                if missing_cols:
                     print(f"    - Skipping formula enforcement for '{target_col} = {formula_str}': Missing columns in synthetic data: {missing_cols}.")
                if non_numeric_cols:
                     print(f"    - Skipping formula enforcement for '{target_col} = {formula_str}': Non-numeric columns involved: {non_numeric_cols}.")


    # --- Log state after Formula Relationships ---
    print("\n--- State after Formula Relationships ---")
    print(df.head())
    print(df.nunique())


    # Apply range constraints (based on min/max from basic stats) - Do this AFTER relationships
    print("  Applying range constraints...")
    for col_name, col_schema in schema.get("columns", {}).items(): # Iterate through columns defined in schema
        if col_name in df.columns:
            data_type = col_schema.get("data_type")
            stats = col_schema.get("stats", {})
            min_val = stats.get("min")
            max_val = stats.get("max")

            # Apply range constraints only to numeric or datetime columns with valid bounds
            if data_type in ['numerical', 'integer', 'float'] and min_val is not None and max_val is not None:
                try:
                    # Ensure column is numeric before clipping
                    df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
                    # Clip values to the detected range
                    original_violations = df[(df[col_name] < min_val) | (df[col_name] > max_val)].shape[0]
                    if original_violations > 0:
                         df[col_name] = df[col_name].clip(lower=min_val, upper=max_val)
                         print(f"    - Clipped {original_violations} values in '{col_name}' to range [{min_val}, {max_val}].")

                except Exception as e:
                    print(f"    - Error applying range constraint to '{col_name}' (min={min_val}, max={max_val}): {e}")

            elif data_type == 'datetime' and stats.get('min_date') is not None and stats.get('max_date') is not None:
                 min_date_str = stats.get('min_date')
                 max_date_str = stats.get('max_date')
                 try:
                      # Attempt to parse dates and clip
                      min_date_obj = datetime.fromisoformat(min_date_str)
                      max_date_obj = datetime.fromisoformat(max_date_str)

                      # Convert column to datetime, coercing errors to NaT
                      df[col_name] = pd.to_datetime(df[col_name], errors='coerce')

                      # Identify violations (dates before min or after max, excluding NaT)
                      violations_mask = ((df[col_name] < min_date_obj) | (df[col_name] > max_date_obj)) & df[col_name].notna()

                      if violations_mask.any():
                           num_violations = violations_mask.sum()
                           # For date clipping, we might need a different strategy than numeric clip.
                           # Simple approach: replace violating dates with the boundary date.
                           df.loc[violations_mask & (df[col_name] < min_date_obj), col_name] = min_date_obj
                           df.loc[violations_mask & (df[col_name] > max_date_obj), col_name] = max_date_obj
                           print(f"    - Clipped {num_violations} datetime values in '{col_name}' to range [{min_date_str}, {max_date_str}].")

                 except Exception as e:
                      print(f"    - Error applying datetime range constraint to '{col_name}' (min={min_date_str}, max={max_date_str}): {e}")
            # Note: We don't apply range constraints if only one bound is provided, as ScalarRange
            # detection in schema generation currently focuses on pairs.

    # --- Log state after Range Constraints ---
    print("\n--- State after Range Constraints ---")
    print(df.head())
    print(df.nunique())


    # Apply uniqueness constraint on Primary Key (if identified) - Do this LAST
    print("  Enforcing uniqueness on Primary Key...")
    primary_key = None
    for col_name, col_schema in schema.get("columns", {}).items(): # Iterate through columns defined in schema
        if col_schema.get("key_type", "").lower() == "primary key":
            primary_key = col_name
            break

    if primary_key and primary_key in df.columns:
        try:
            # Find duplicate rows based on the primary key column
            duplicates_mask = df.duplicated(subset=[primary_key], keep='first')
            duplicate_count = duplicates_mask.sum()

            if duplicate_count > 0:
                print(f"    - Found {duplicate_count} duplicate values in primary key column '{primary_key}'. Attempting to regenerate.")
                duplicate_indices = df[duplicates_mask].index

                pk_schema = schema.get("columns", {}).get(primary_key, {}) # Get PK schema from 'columns'
                faker_provider = pk_schema.get("faker_provider")
                faker_args = pk_schema.get("faker_args", {}).copy()

                # Get current existing PK values to ensure new ones are unique
                existing_pk_values = set(df[primary_key].dropna().unique())

                # Attempt to regenerate unique values for duplicates
                for index in duplicate_indices:
                    new_value = None
                    retry_count = 0
                    max_retries = 50 # Increased retries for uniqueness

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
                                          while new_value is None or (new_value in existing_pk_values and retry_count < max_retries):
                                               new_value = generate_from_regex_pattern(regex_pattern)
                                               retry_count += 1
                                     else:
                                          print(f"      - Warning: No valid regex pattern found in faker_args for PK '{primary_key}' with regexify. Falling back to sequential/placeholder.")
                                          new_value = None

                                elif cleaned_provider and hasattr(fake, cleaned_provider):
                                     faker_method = getattr(fake, cleaned_provider)
                                     while new_value is None or (new_value in existing_pk_values and retry_count < max_retries):
                                         current_faker_args = faker_args.copy()
                                         # Adjust args for random_int to generate higher values if needed
                                         if cleaned_provider == 'random_int':
                                              # Try generating from a range above the current max
                                              current_max_id = None
                                              if existing_pk_values:
                                                   try:
                                                        current_max_id = max(v for v in existing_pk_values if isinstance(v, (int, float)))
                                                   except ValueError: # Handle case where set is not all numeric
                                                        pass

                                              if current_max_id is not None:
                                                   current_faker_args['min'] = int(current_max_id) + 1
                                                   if 'max' not in current_faker_args or current_faker_args['max'] <= current_faker_args['min']:
                                                        current_faker_args['max'] = current_faker_args['min'] + 100000 # Extend range if needed

                                         # Filter args based on common parameter names (simplified)
                                         common_faker_params = ['min', 'max', 'elements', 'pattern', 'start_date', 'end_date', 'digits', 'right_digits', 'locale', 'country_code', 'area_code', 'extension', 'length', 'data_range', 'types', 'levels', 'zone', 'model', 'make', 'year', 'prefix', 'suffix', 'amount', 'currency', 'vat_id_type', 'country', 'city_prefix', 'city_suffix', 'street_name', 'building_number', 'postcode', 'state', 'state_abbr', 'latitude', 'longitude', 'ean_type', 'isbn_type', 'protocol', 'domain_name', 'tld', 'user_name', 'email', 'name', 'first_name', 'last_name', 'ssn', 'iban', 'bban', 'swift8', 'swift11', 'credit_card_number', 'credit_card_expire', 'credit_card_security_code', 'file_extension', 'mime_type', 'binary', 'size', 'text', 'max_nb_chars', 'min_nb_chars', 'ext_word_list', 'nb_sentences', 'variable_nb_sentences', 'nb_words', 'variable_nb_words']
                                         filtered_args = {k: v for k, v in current_faker_args.items() if k in common_faker_params}


                                         new_value = faker_method(**filtered_args)
                                         retry_count += 1

                                else:
                                     print(f"      - Warning: Cleaned Faker provider name for PK '{primary_key}' is empty or not found. Falling back to sequential/placeholder.")
                                     new_value = None

                            else:
                                 print(f"      - Warning: Faker provider for PK '{primary_key}' is not a string ({type(faker_provider)}). Falling back to sequential/placeholder.")
                                 new_value = None


                        except Exception as e:
                            print(f"      - Error generating new ID for duplicate at index {index} with Faker: {e}. Falling back to sequential/placeholder.")
                            new_value = None

                    # Fallback to a simple sequential or placeholder if Faker/regexify failed
                    if new_value is None or new_value in existing_pk_values:
                        if pd.api.types.is_numeric_dtype(df[primary_key]):
                             # Find the current max value and add 1
                             current_max_id = None
                             try:
                                  current_max_id = df[primary_key].max()
                                  if pd.isna(current_max_id): current_max_id = 0
                             except Exception:
                                  current_max_id = 0

                             new_value = int(current_max_id) + 1
                             # Ensure the new sequential value is also unique
                             while new_value in existing_pk_values:
                                  new_value += 1
                        else:
                             # Create a unique placeholder for non-numeric keys
                             new_value = f"GENERATED_DUP_{index}_{random.randint(10000, 99999)}"
                             # Ensure placeholder is also unique
                             while new_value in existing_pk_values:
                                  new_value = f"GENERATED_DUP_{index}_{random.randint(10000, 99999)}"

                        print(f"      - Using fallback sequential/placeholder value '{new_value}' for duplicate at index {index}.")


                    df.loc[index, primary_key] = new_value
                    existing_pk_values.add(new_value) # Add the new value to the set of existing values


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


    print("\nPost-processing constraints applied.")
    return df


# Keep this function for reporting purposes, adapted to use global thresholds
def detect_column_correlations_for_reporting(df):
    """
    Detects column correlations and relationships for reporting purposes.
    Includes functional dependencies, basic value relationships, and Pearson correlation.
    Uses global thresholds.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        dict: A dictionary where keys are column names and values are lists of detected relationships.
    """
    correlations = {}
    global CORRELATION_THRESHOLD, RELATIONSHIP_CONSISTENCY_THRESHOLD, FORMULA_TOLERANCE

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
                                                    # "correlation": 1.0, # Represent dependency - removed correlation value for clarity
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

        # --- Simple Formula Relationships (A = B + C) for Reporting ---
        # This check is also included for reporting purposes to see if formulas hold in synthetic data
        if len(numeric_cols) >= 3:
             for col_a, col_b, col_c in combinations(numeric_cols, 3):
                  # Ensure columns exist and are numeric
                  if all(col in df.columns and pd.api.types.is_numeric_dtype(df[col]) for col in [col_a, col_b, col_c]):
                       try:
                            df_filtered = df[[col_a, col_b, col_c]].dropna()
                            if not df_filtered.empty:
                                series_a = pd.to_numeric(df_filtered[col_a], errors='coerce')
                                series_b = pd.to_numeric(df_filtered[col_b], errors='coerce')
                                series_c = pd.to_numeric(df_filtered[col_c], errors='coerce')

                                # Check A = B + C
                                if np.allclose(series_a, series_b + series_c, atol=FORMULA_TOLERANCE, equal_nan=True):
                                    formula = f"{col_b} + {col_c}"
                                    rel_details = {"target_column": col_a, "source_columns": sorted([col_b, col_c]), "formula": formula, "type": "formula"}
                                    # Add to correlations under the target column
                                    if col_a not in correlations: correlations[col_a] = []
                                    if rel_details not in correlations[col_a]: correlations[col_a].append(rel_details)
                                    # print(f"Detected Formula: {col_a} = {formula}") # Suppress

                                # Check B = A + C
                                if np.allclose(series_b, series_a + series_c, atol=FORMULA_TOLERANCE, equal_nan=True):
                                    formula = f"{col_a} + {col_c}"
                                    rel_details = {"target_column": col_b, "source_columns": sorted([col_a, col_c]), "formula": formula, "type": "formula"}
                                    if col_b not in correlations: correlations[col_b] = []
                                    if rel_details not in correlations[col_b]: correlations[col_b].append(rel_details)
                                    # print(f"Detected Formula: {col_b} = {formula}") # Suppress

                                # Check C = A + B
                                if np.allclose(series_c, series_a + series_b, atol=FORMULA_TOLERANCE, equal_nan=True):
                                    formula = f"{col_a} + {col_b}"
                                    rel_details = {"target_column": col_c, "source_columns": sorted([col_a, col_b]), "formula": formula, "type": "formula"}
                                    if col_c not in correlations: correlations[col_c] = []
                                    if rel_details not in correlations[col_c]: correlations[col_c].append(rel_details)
                                    # print(f"Detected Formula: {col_c} = {formula}") # Suppress

                       except Exception as formula_e:
                            pass # print(f"Could not check formula {col_a}, {col_b}, {col_c} for reporting: {formula_e}")


    except Exception as e:
        print(f"Error detecting correlations for reporting: {str(e)}")
        import traceback
        traceback.print_exc()

    # Clean up empty correlation lists
    correlations = {col: rels for col, rels in correlations.items() if rels}

    return correlations


def generate_synthetic_data_with_sdv(df, schema, num_rows, output_file):
    """Generates synthetic data using SDV, applies post-processing constraints, Faker, etc."""
    try:
        print("\n--- Starting SDV Synthetic Data Generation ---")
        print("1. Creating SDV metadata from DataFrame and enhanced schema...")
        # create_sdv_metadata is not used in this revised flow as we use Faker for initial generation
        # and post-processing for constraints.
        # We still need the original data and schema for post-processing logic.
        print("   Skipping SDV metadata creation as Faker and post-processing are used for generation.")


        print("\n2. Initializing synthetic DataFrame with correct columns...")
        # Create an empty DataFrame with the same columns as the original data
        synthetic_data = pd.DataFrame(columns=df.columns)
        # Pre-populate with NaNs or a placeholder if needed, or fill during Faker step

        # Resize the DataFrame to the desired number of rows
        synthetic_data = synthetic_data.reindex(range(num_rows))
        print(f"   Initialized synthetic DataFrame with {len(synthetic_data)} rows and {len(synthetic_data.columns)} columns.")


        # --- Post-processing Steps (Includes Initial Generation with Faker/Regex) ---
        # Load original data again for post-processing (functional dependency mapping etc.)
        original_df_for_postprocessing = None
        try:
             original_df_for_postprocessing = pd.read_csv(INPUT_CSV, encoding='utf-8')
             try:
                 original_df_for_postprocessing = pd.read_csv(INPUT_CSV, encoding='utf-8')
             except UnicodeDecodeError:
                 print("UTF-8 decoding failed for original data reload for post-processing, trying latin-1...")
                 original_df_for_postprocessing = pd.read_csv(INPUT_CSV, encoding='latin-1')
             print("\n   Loaded original data for post-processing (functional dependency mapping, sampling).")
        except Exception as e:
             print(f"\n   Error loading original data for post-processing: {e}. Functional dependencies and sampling from original distribution cannot be fully applied.")
             original_df_for_postprocessing = None


        print("\n3. Applying post-processing (Initial Faker/Regex generation, categorical distributions, custom rules, constraints)...")
        # The apply_post_processing_rules function now handles the initial filling of the DataFrame
        # using Faker/Regex/Categorical distributions, and then applies the constraints.
        synthetic_data = apply_post_processing_rules(synthetic_data, schema, original_df_for_postprocessing)

        if original_df_for_postprocessing is not None:
             del original_df_for_postprocessing


        print("\n4. Comparing relationships between original and synthetic data and generating report...")
        original_df_for_corr = None
        try:
             original_df_for_corr = pd.read_csv(INPUT_CSV, encoding='utf-8')
             try:
                 original_df_for_corr = pd.read_csv(INPUT_CSV, encoding='utf-8')
             except UnicodeDecodeError:
                 print("UTF-8 decoding failed for original data reload for correlation report, trying latin-1...")
                 original_df_for_corr = pd.read_csv(INPUT_CSV, encoding='latin-1')
             print("   Loaded original data for relationship reporting.")

             # Detect relationships in original and synthetic data
             original_relationships_detected = detect_column_correlations_for_reporting(original_df_for_corr)
             synthetic_relationships_detected = detect_column_correlations_for_reporting(synthetic_data)

             # --- Generate Comparison Report Data ---
             correlation_report_data = []
             # Use a set to track unique relationships added to the report data
             reported_relationships_keys = set()

             # Helper to create a unique key for a relationship for tracking
             def get_relationship_key(rel_info):
                  rel_type = rel_info.get("type")
                  # Ensure consistent key creation based on relationship type
                  if rel_type == "pearson_correlation":
                       # Pearson correlation is symmetric, sort columns for the key
                       cols = tuple(sorted((rel_info.get("column"), rel_info.get("source_column"))))
                       return (cols, rel_type)
                  elif rel_type in ["functional_dependency", "one_to_one", "value_relationship"]:
                       # For directed/value relationships, order matters for the key
                       # Use source_column and target_column if available, fallback to column
                       source_col = rel_info.get("source_column") or rel_info.get("column")
                       target_col = rel_info.get("target_column") or rel_info.get("column")
                       return (source_col, target_col, rel_type, rel_info.get("relationship")) # Include relationship detail for value relationships
                  elif rel_type == "formula":
                       # Formula key includes target and sorted sources
                       return (rel_info.get("target_column"), tuple(sorted(rel_info.get("source_columns", []))), rel_type, rel_info.get("formula"))
                  return None # Return None for unsupported types


             # Combine all relationships from original and synthetic data
             all_relationships_combined = defaultdict(lambda: {"Original": None, "Synthetic": None})

             # Add original relationships
             for col, rels in original_relationships_detected.items():
                 for rel in rels:
                     # Pass the source column explicitly to get_relationship_key
                     key = get_relationship_key({**rel, "source_column": col})
                     if key:
                          # Store the original relationship details under the 'Original' key
                          all_relationships_combined[key]["Original"] = rel


             # Add synthetic relationships
             for col, rels in synthetic_relationships_detected.items():
                  for rel in rels:
                      # Pass the source column explicitly to get_relationship_key
                      key = get_relationship_key({**rel, "source_column": col})
                      if key:
                           # Store the synthetic relationship details under the 'Synthetic' key
                           all_relationships_combined[key]["Synthetic"] = rel


             # Populate the report data list, prioritizing relationship types
             for key, data in all_relationships_combined.items():
                  original_rel = data["Original"]
                  synthetic_rel = data["Synthetic"]

                  if original_rel is None and synthetic_rel is None:
                       continue # Skip if relationship was not detected in either

                  # Extract details from the key for the report row
                  rel_type = key[2] if len(key) > 2 else None
                  col1 = key[0]
                  col2 = key[1] if len(key) > 1 else None
                  relationship_detail = key[3] if len(key) > 3 else ""


                  # Determine the most specific type detected for this pair/formula
                  most_specific_type = None
                  if original_rel and original_rel.get("type") == "one_to_one" or synthetic_rel and synthetic_rel.get("type") == "one_to_one":
                       most_specific_type = "one_to_one"
                  elif original_rel and original_rel.get("type") == "functional_dependency" or synthetic_rel and synthetic_rel.get("type") == "functional_dependency":
                       most_specific_type = "functional_dependency"
                  elif original_rel and original_rel.get("type") == "formula" or synthetic_rel and synthetic_rel.get("type") == "formula":
                       most_specific_type = "formula"
                  elif original_rel and original_rel.get("type") == "value_relationship" or synthetic_rel and synthetic_rel.get("type") == "value_relationship":
                       most_specific_type = "value_relationship"
                  elif original_rel and original_rel.get("type") == "pearson_correlation" or synthetic_rel and synthetic_rel.get("type") == "pearson_correlation":
                       most_specific_type = "pearson_correlation"


                  # Decide whether to report this relationship based on priority
                  # Prioritization: O2O > FD > Formula > Value Relationship > Pearson Correlation
                  # If a more specific relationship exists for the same pair, don't report less specific ones.
                  # Pearson correlation is reported alongside the most specific structural relationship for the pair.

                  report_this = False
                  if most_specific_type in ["one_to_one", "functional_dependency", "formula"]:
                       # Always report O2O, FD, and Formula if detected in either
                       report_this = True
                  elif most_specific_type == "value_relationship":
                       # Only report Value Relationship if no O2O, FD, or Formula was detected for the same pair
                       pair_has_more_specific = False
                       # Need to check all relationships involving these columns to see if a more specific type exists
                       cols_involved = {col1}
                       if col2: cols_involved.add(col2)
                       # For formulas, need to check if the target or any source columns match
                       if rel_type == "formula":
                            cols_involved.add(key[0]) # Target column
                            cols_involved.update(key[1]) # Source columns tuple

                       for other_key, other_data in all_relationships_combined.items():
                            other_rel_type = other_key[2] if len(other_key) > 2 else None
                            other_cols_involved = {other_key[0]}
                            if len(other_key) > 1: other_cols_involved.add(other_key[1])
                            if other_rel_type == "formula":
                                other_cols_involved.add(other_key[0])
                                other_cols_involved.update(other_key[1])

                            # Check if the sets of columns involved overlap significantly (or are the same for pairs)
                            # For pairs, check if the pair is the same (sorted tuple)
                            if rel_type != "formula" and other_rel_type != "formula":
                                 pair_key_current = tuple(sorted((col1, col2))) if col2 else (col1,)
                                 pair_key_other = tuple(sorted((other_key[0], other_key[1]))) if len(other_key) > 1 else (other_key[0],)
                                 if pair_key_current == pair_key_other and other_rel_type in ["one_to_one", "functional_dependency", "formula"]:
                                      pair_has_more_specific = True
                                      break
                            # For formulas, check if the target column is the same or if source columns overlap significantly
                            elif rel_type == "formula" or other_rel_type == "formula":
                                 # Simplified check: if target columns are the same or if the set of involved columns is the same
                                 if (rel_type == "formula" and other_rel_type == "formula" and key[0] == other_key[0]) or \
                                    (cols_involved == other_cols_involved and other_rel_type in ["one_to_one", "functional_dependency", "formula"]):
                                      pair_has_more_specific = True
                                      break


                       if not pair_has_more_specific:
                            report_this = True
                  elif most_specific_type == "pearson_correlation":
                       # Always report Pearson correlation if detected in either
                       report_this = True


                  if report_this:
                       report_entry = {
                           "Column 1": col1,
                           "Column 2": col2 if col2 else "",
                           "Type": rel_type,
                           "Relationship Detail": relationship_detail,
                           "Original Value": None,
                           "Synthetic Value": None
                       }

                       if original_rel:
                            if rel_type == "pearson_correlation":
                                 report_entry["Original Value"] = original_rel.get("correlation")
                            elif rel_type in ["functional_dependency", "one_to_one"]:
                                 report_entry["Original Value"] = "Detected"
                            elif rel_type == "value_relationship":
                                 report_entry["Original Value"] = original_rel.get("relationship")
                            elif rel_type == "formula":
                                 report_entry["Original Value"] = original_rel.get("formula")


                       if synthetic_rel:
                            if rel_type == "pearson_correlation":
                                 report_entry["Synthetic Value"] = synthetic_rel.get("correlation")
                            elif rel_type in ["functional_dependency", "one_to_one"]:
                                 report_entry["Synthetic Value"] = "Detected"
                            elif rel_type == "value_relationship":
                                 report_entry["Synthetic Value"] = synthetic_rel.get("relationship")
                            elif rel_type == "formula":
                                 report_entry["Synthetic Value"] = synthetic_rel.get("formula")


                       # Add to report data only if not already added (handles symmetric relationships added from both sides)
                       # Create a hashable key from the entry, excluding values for uniqueness check
                       report_entry_key_for_uniqueness = (report_entry["Column 1"], report_entry["Column 2"], report_entry["Type"], report_entry["Relationship Detail"])
                       if report_entry_key_for_uniqueness not in reported_relationships_keys:
                            correlation_report_data.append(report_entry)
                            reported_relationships_keys.add(report_entry_key_for_uniqueness)


             if correlation_report_data:
                 correlation_df = pd.DataFrame(correlation_report_data)
                 # Define a clear order for report columns
                 column_order = ["Column 1", "Column 2", "Type", "Relationship Detail", "Original Value", "Synthetic Value"]
                 correlation_df = correlation_df[column_order]

                 try:
                     correlation_df.to_csv(REPORT_CSV, index=False, encoding='utf-8')
                     print(f"   Relationship comparison report saved to {REPORT_CSV}")
                 except Exception as csv_e:
                     print(f"   Error saving relationship report CSV: {csv_e}")
             else:
                 print("   No correlations or relationships detected for reporting.")


        except Exception as corr_e:
            print(f"   Error during relationship comparison step: {corr_e}")
            import traceback
            traceback.print_exc()
        finally:
             if 'original_df_for_corr' in locals() and original_df_for_corr is not None:
                  del original_df_for_corr


        print(f"\n5. Saving final synthetic data to {output_file}...")
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
    print("--- Starting working synthetic data generation script ---")
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Input data file '{INPUT_CSV}' not found.")
        return

    print(f"\nReading enhanced schema from: {INPUT_SCHEMA_JSON}")
    schema = read_schema(INPUT_SCHEMA_JSON)
    if not schema:
        print("Failed to read the enhanced schema. Please run the schema generation script first to create enhanced_schema.json. Exiting.")
        return

    # Load original data for post-processing and reporting
    original_df = None
    try:
        try:
            original_df = pd.read_csv(INPUT_CSV, encoding='utf-8')
        except UnicodeDecodeError:
            print("UTF-8 decoding failed, trying latin-1...")
            original_df = pd.read_csv(INPUT_CSV, encoding='latin-1')

        print(f"Loaded original data: {original_df.shape[0]} rows, {original_df.shape[1]} columns")

    except FileNotFoundError:
         print(f"Error: Input data file '{INPUT_CSV}' not found during loading for post-processing/reporting.")
         # We can still proceed with generation but FD mapping/reporting will be limited
         pass # Allow script to continue
    except Exception as e:
        print(f"Error loading or preparing original data from '{INPUT_CSV}' for post-processing/reporting: {str(e)}")
        # Allow script to continue but with limitations
        pass # Allow script to continue


    print(f"\nInitiating synthetic data generation for {NUM_ROWS} rows...")
    # Pass the original_df to generate_synthetic_data_with_sdv for post-processing
    success = generate_synthetic_data_with_sdv(original_df.copy() if original_df is not None else pd.DataFrame(columns=schema.get("columns", {}).keys()), schema, NUM_ROWS, OUTPUT_CSV)

    if original_df is not None:
         del original_df # Free up memory


    print("\n--- working synthetic data generation script Finished ---")
    if success:
        print(f"Synthetic data generation process completed successfully.")
        print(f"Output saved to: {OUTPUT_CSV}")
        if os.path.exists(REPORT_CSV):
             print(f"Relationship comparison report saved to: {REPORT_CSV}")

    else:
        print("Synthetic data generation process encountered errors.")

if __name__ == "__main__":
    main()
