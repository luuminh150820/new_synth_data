import pandas as pd
import json
import os
import numpy as np
import random
from faker import Faker
import inspect
import warnings
import re
from datetime import datetime, date, timedelta
import string
from collections import defaultdict
from itertools import combinations

# Configuration
INPUT_CSV = "customer_data.csv"
OUTPUT_CSV = "synthetic_data.csv"
INPUT_SCHEMA_JSON = "enhanced_schema.json"
NUM_ROWS = 1000
# Removed correlation and formula tolerance as they are only used in reporting now
# RELATIONSHIP_CONSISTENCY_THRESHOLD is used in reporting
REPORT_CSV = "relationship_comparison_report.csv"
MAX_SAMPLING_RETRIES = 10 # Retries for relationship-aware sampling fallback and Faker fallback

# Initialize Faker
fake = Faker(['vi_VN'])
Faker.seed(42)

# --- Helper Functions ---

def read_schema(json_file_path):
    """Reads the schema from a JSON file."""
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

def generate_from_regex_pattern(pattern):
    """Generates a string matching a simplified regex pattern."""
    if not isinstance(pattern, str):
        return None

    result = []
    i = 0
    while i < len(pattern):
        char = pattern[i]

        if char == '\\':
             if i + 1 < len(pattern):
                  next_char = pattern[i+1]
                  if next_char == 'd': result.append(random.choice(string.digits))
                  elif next_char == '.': result.append('.')
                  else: result.append(next_char)
                  i += 2
             else:
                  print(f"Warning: Trailing '\\' in regex pattern: {pattern}. Ignoring.")
                  i += 1
        elif char == '#': result.append(random.choice(string.digits)); i += 1
        elif char == '[':
            end_bracket = pattern.find(']', i + 1)
            if end_bracket == -1:
                print(f"Warning: Unclosed '[' in regex pattern: {pattern}. Skipping.")
                return "".join(result)
            char_set_content = pattern[i + 1:end_bracket]
            possible_chars = []
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
            if possible_chars: result.append(random.choice(possible_chars))
            else: print(f"Warning: Empty or unparseable character set in regex pattern: {pattern}. Skipping.")
            i = end_bracket + 1
        elif char == '{':
            end_brace = pattern.find('}', i + 1)
            if end_brace == -1:
                print(f"Warning: Unclosed '{{' in regex pattern: {pattern}. Skipping.")
                return "".join(result)
            count_str = pattern[i + 1:end_brace]
            repetition_count = 1
            try:
                if ',' in count_str:
                    min_max = count_str.split(',')
                    min_rep = int(min_max[0].strip()) if min_max[0].strip() else 0
                    max_rep = int(min_max[1].strip()) if min_max[1].strip() else min_rep
                    repetition_count = random.randint(min_rep, max_rep)
                else: repetition_count = int(count_str.strip())
            except ValueError: print(f"Warning: Invalid repetition count '{{{count_str}}}' in regex pattern: {pattern}. Using count 1."); repetition_count = 1
            if result: last_char_or_group = result.pop(); result.extend([last_char_or_group] * repetition_count)
            else: print(f"Warning: Repetition '{{{count_str}}}' found at the start or after an empty group in regex pattern: {pattern}. Ignoring repetition.")
            i = end_brace + 1
        elif char in '?*+':
             if result:
                  last_char_or_group = result.pop()
                  if char == '?':
                       if random.random() < 0.5: result.append(last_char_or_group)
                  elif char == '*':
                       num_repetitions = random.randint(0, 5); result.extend([last_char_or_group] * num_repetitions)
                  elif char == '+':
                       num_repetitions = random.randint(1, 5); result.extend([last_char_or_group] * num_repetitions)
             else: print(f"Warning: Quantifier '{char}' found at the start or after an empty group in regex pattern: {pattern}. Ignoring.")
             i += 1
        elif char == '(':
             if i + 3 < len(pattern) and pattern[i+1:i+3] == '?:':
                  end_paren = pattern.find(')', i + 3)
                  if end_paren != -1:
                       group_content = pattern[i + 3:end_paren]
                       generated_group_content = generate_from_regex_pattern(group_content)
                       if generated_group_content is not None: result.append(generated_group_content)
                       i = end_paren + 1
                  else: print(f"Warning: Unclosed non-capturing group '(?:' in regex pattern: {pattern}. Skipping."); return "".join(result)
             else: result.append(char); i += 1
        elif char == ')': i += 1
        else: result.append(char); i += 1
    return "".join(result)


# --- Post-processing Function ---
def apply_post_processing_rules(df, schema, original_df):
    """
    Applies post-processing rules to the synthetic DataFrame based on the schema.
    Includes Faker application, custom regex generation, preserved categorical distributions,
    functional dependencies (Method 1 with relationship-aware sampling),
    range constraints, and uniqueness enforcement.
    Does NOT enforce Value or Formula relationships; these are for reporting.

    Args:
        df (pd.DataFrame): The synthetic DataFrame to modify.
        schema (dict): The enhanced schema dictionary.
        original_df (pd.DataFrame): The original DataFrame, used for functional dependency mapping and sampling.

    Returns:
        pd.DataFrame: The DataFrame after applying post-processing rules.
    """
    print("\nApplying post-processing constraints and Faker...")

    # --- Apply Faker providers and preserved categorical distributions ---
    print("  Applying preserved categorical distributions and Faker providers...")

    def generate_weighted_random_element(categories):
        if not isinstance(categories, list) or not all(isinstance(item, dict) and 'value' in item and 'percentage' in item for item in categories):
             return None
        values = [item["value"] for item in categories]
        weights = [item["percentage"] for item in categories]
        if not values or not all(isinstance(w, (int, float)) for w in weights) or sum(weights) <= 0: return None
        try: return random.choices(values, weights=weights, k=1)[0]
        except Exception as e: print(f"     - Error during weighted random selection: {e}. Returning None."); return None

    for col_name, col_schema in schema.get("columns", {}).items():
         if col_name not in df.columns: continue
         is_cat_in_schema = col_schema.get("data_type") == "categorical"
         categories_data = col_schema.get("stats", {}).get("categories") or col_schema.get("stats", {}).get("top_categories")
         faker_provider = col_schema.get("faker_provider")

         if faker_provider is not None and isinstance(faker_provider, str):
             cleaned_provider = faker_provider.strip().replace('Faker.', '').replace('Faker::Date.', '').replace('Faker::Regex.', '')
             provider_name_map = {'regex': 'regexify', 'date': 'date_object', 'date_between': 'date_object', 'bban': 'bban', 'random_float': 'pyfloat'}
             cleaned_provider = provider_name_map.get(cleaned_provider, cleaned_provider)

             if cleaned_provider == 'regexify':
                  print(f"     - Applying custom regex generation for column '{col_name}'")
                  regex_pattern = col_schema.get("faker_args", {}).get("text") or col_schema.get("faker_args", {}).get("pattern")
                  if isinstance(regex_pattern, str):
                       try: df.loc[:, col_name] = df[col_name].apply(lambda x: generate_from_regex_pattern(regex_pattern))
                       except Exception as e: print(f"     - Error applying custom regex generation to '{col_name}': {e}")
                  else: print(f"     - Skipping custom regex generation for '{col_name}': No valid regex pattern found in faker_args.")
                  continue # Skip to next column after applying regex

             # --- Handle Nested and Simple Providers ---
             faker_method = None
             provider_parts = cleaned_provider.split('.', 1) # Split by the first dot

             if len(provider_parts) > 1:
                  # Nested provider (e.g., unique.random_int)
                  attribute_name = provider_parts[0]
                  method_name = provider_parts[1]
                  if hasattr(fake, attribute_name):
                       attribute_obj = getattr(fake, attribute_name)
                       if hasattr(attribute_obj, method_name):
                            faker_method = getattr(attribute_obj, method_name)
                            # Check for unhashable elements if it's random_sample
                            if method_name == 'random_sample':
                                 elements_arg = col_schema.get("faker_args", {}).get('elements')
                                 if isinstance(elements_arg, list):
                                      try: hash(tuple(elements_arg)) # Try hashing a tuple version
                                      except TypeError:
                                           print(f"     - Skipping Faker application for column '{col_name}': Provider '{cleaned_provider}' requires hashable elements, but list contains unhashable types. Cleaned provider: '{cleaned_provider}'")
                                           faker_method = None # Invalidate method if elements are unhashable
                       else:
                            print(f"     - Skipping Faker application for column '{col_name}': Method '{method_name}' not found on '{attribute_name}'. Cleaned provider: '{cleaned_provider}'")
                  else:
                       print(f"     - Skipping Faker application for column '{col_name}': Attribute '{attribute_name}' not found on fake object. Cleaned provider: '{cleaned_provider}'")
             else:
                  # Simple provider (e.g., name, random_int)
                  if hasattr(fake, cleaned_provider):
                       faker_method = getattr(fake, cleaned_provider)
                  else:
                       print(f"     - Skipping Faker application for column '{col_name}': Provider '{cleaned_provider}' not found on fake object.")


             if faker_method:
                  print(f"     - Applying Faker provider '{cleaned_provider}' to column '{col_name}'")
                  faker_args = col_schema.get("faker_args", {}).copy()

                  if cleaned_provider == 'pyfloat':
                       # --- Fix for empty range in pyfloat ---
                       min_val_arg = faker_args.get('min', faker_args.get('min_value'))
                       max_val_arg = faker_args.get('max', faker_args.get('max_value'))

                       if min_val_arg is not None and max_val_arg is not None:
                            try:
                                 min_float = float(min_val_arg)
                                 max_float = float(max_val_arg)
                                 if min_float >= max_float:
                                      print(f"         Warning: Invalid or equal min/max range [{min_val_arg}, {max_val_arg}] for pyfloat in '{col_name}'. Adjusting max slightly.")
                                      # Adjust max to create a valid range, ensure it's still significantly larger than min
                                      faker_args['max_value'] = max_float + abs(max_float - min_float) * 0.01 + 1.0 if max_float != min_float else min_float + 1.0
                                      faker_args.pop('max', None) # Remove potential 'max' key if 'max_value' is used
                                 else:
                                      faker_args['min_value'] = min_float
                                      faker_args['max_value'] = max_float
                                      faker_args.pop('min', None); faker_args.pop('max', None) # Use min/max_value consistently
                            except (ValueError, TypeError):
                                 print(f"         Warning: Could not convert min/max args to float for pyfloat in '{col_name}'. Skipping Faker.")
                                 faker_method = None # Invalidate faker_method to skip application
                       else:
                            print(f"         Warning: Missing min/max args for pyfloat in '{col_name}'. Skipping Faker.")
                            faker_method = None # Invalidate faker_method to skip application

                       faker_args.pop('step', None); faker_args.pop('precision', None)
                       if faker_args.get('min_value') is not None and faker_args['min_value'] > 0: faker_args['positive'] = True
                       elif 'positive' in faker_args: faker_args.pop('positive', None)

                  elif cleaned_provider == 'random_int':
                       if 'min' in faker_args and isinstance(faker_args['min'], float): faker_args['min'] = int(faker_args['min'])
                       if 'max' in faker_args and isinstance(faker_args['max'], float): faker_args['max'] = int(faker_args['max'])
                       # --- Fix for empty range in random_int ---
                       min_val_arg = faker_args.get('min')
                       max_val_arg = faker_args.get('max')
                       if min_val_arg is not None and max_val_arg is not None:
                           try:
                               min_int = int(min_val_arg)
                               max_int = int(max_val_arg)
                               if min_int >= max_int:
                                   print(f"         Warning: Invalid or equal min/max range [{min_val_arg}, {max_val_arg}] for random_int in '{col_name}'. Adjusting max to be at least min + 1.")
                                   faker_args['max'] = min_int + 1 # Adjust max to create a valid range
                           except (ValueError, TypeError):
                                print(f"         Warning: Could not convert min/max args to integer for random_int in '{col_name}'. Skipping Faker.")
                                faker_method = None # Invalidate faker_method to skip application
                       else:
                            print(f"         Warning: Missing min/max args for random_int in '{col_name}'. Skipping Faker.")
                            faker_method = None # Invalidate faker_method to skip application
                       # --- End Fix ---

                  elif cleaned_provider == 'random_element':
                       # --- Fix for random_element OrderedDict issue ---
                       if categories_data:
                           # Pass elements as a list of (value, weight) tuples
                           elements_list_of_tuples = [(item['value'], item['percentage']) for item in categories_data if 'value' in item and 'percentage' in item]
                           if elements_list_of_tuples and sum(w for v, w in elements_list_of_tuples) > 0:
                                faker_args['elements'] = elements_list_of_tuples
                                faker_args.pop('weights', None) # Ensure weights arg is not also present
                                print(f"         - Using list of tuples for random_element elements for '{col_name}'.")
                           else:
                                print(f"         Warning: Categories data for '{col_name}' is invalid or empty for random_element. Skipping Faker.")
                                faker_method = None # Invalidate faker_method if categories data is invalid
                       elif 'elements' not in faker_args:
                            print(f"         Warning: random_element for '{col_name}' has no 'elements' arg and no categories data. Skipping Faker.")
                            faker_method = None # Invalidate faker_method if no elements are provided
                       # --- End Fix ---
                  elif cleaned_provider == 'date_object':
                       faker_args.pop('pattern', None)
                       for date_arg_key in ['start_date', 'end_date']:
                            if date_arg_key in faker_args:
                                 date_val = faker_args[date_arg_key]
                                 if isinstance(date_val, str):
                                      date_str = date_val.strip()
                                      if date_str.lower() == 'today': faker_args[date_arg_key] = date.today(); continue
                                      relative_match = re.match(r'^([+-]?\d+)([ymd])$', date_str)
                                      if relative_match:
                                           value = int(relative_match.group(1)); unit = relative_match.group(2)
                                           try:
                                                today = date.today()
                                                if unit == 'y': faker_args[date_arg_key] = today + timedelta(days=value * 365)
                                                elif unit == 'm': faker_args[date_arg_key] = today + timedelta(days=value * 30)
                                                elif unit == 'd': faker_args[date_arg_key] = today + timedelta(days=value)
                                           except Exception as parse_e: print(f"         Warning: Could not parse relative date string '{date_str}' for '{col_name}': {parse_e}. Keeping original value.")
                                      else:
                                           try:
                                                date_formats_to_try = ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y', '%Y/%m/%d', '%Y%m%d']
                                                parsed_date = None
                                                for fmt in date_formats_to_try:
                                                     try: parsed_date = datetime.strptime(date_str.split(' ')[0], fmt); break
                                                     except ValueError: pass
                                                if parsed_date: faker_args[date_arg_key] = parsed_date
                                                else: print(f"         Warning: Could not parse date string '{date_str}' for '{col_name}' with common formats. Keeping original value.")
                                           except Exception as parse_e: print(f"         Warning: Error parsing date string '{date_str}' for '{col_name}': {parse_e}. Keeping original value.")
                            elif not isinstance(date_val, date): print(f"         Warning: {date_arg_key} arg for '{col_name}' is not a string or date object ({type(date_val)}). Keeping original value.")

                  # --- Apply the Faker method if it's still valid ---
                  if faker_method:
                       try:
                           valid_args = inspect.signature(faker_method).parameters
                           filtered_args = {k: v for k, v in faker_args.items() if k in valid_args}
                           # --- Specific Error Handling for pyfloat empty range ---
                           if cleaned_provider == 'pyfloat':
                               try:
                                   df.loc[:, col_name] = df[col_name].apply(lambda x: faker_method(**filtered_args))
                               except ValueError as ve:
                                   if "empty range in randrange" in str(ve):
                                       print(f"       Error applying Faker provider '{cleaned_provider}' to '{col_name}' due to internal empty range: {ve}. Keeping generated values.")
                                   else:
                                        # Re-raise if it's a different ValueError
                                        raise ve
                               except Exception as e: # Catch any other exception during pyfloat application
                                   print(f"       Error applying Faker provider '{cleaned_provider}' to '{col_name}': {str(e)}. Keeping generated values.")
                           else:
                                # Apply for all other valid providers
                                df.loc[:, col_name] = df[col_name].apply(lambda x: faker_method(**filtered_args))

                       except Exception as e: # Catch any exception during Faker application for non-pyfloat
                           print(f"       Error applying Faker provider '{cleaned_provider}' to '{col_name}': {str(e)}. Keeping generated values.")


         else: # Handle case where faker_provider is None or not a string
             print(f"     - Skipping Faker application for column '{col_name}': Provider is None or not a string.")


         # --- Apply Categorical Distribution if Faker was NOT applied ---
         # Check if Faker was successfully applied by seeing if the provider was valid and not regexify
         # Re-check faker_method as it might have been invalidated for pyfloat/random_int/random_element or unhashable random_sample
         faker_applied_successfully = (faker_provider is not None and isinstance(faker_provider, str) and
                                       cleaned_provider != 'regexify' and faker_method is not None)

         if is_cat_in_schema and categories_data and not faker_applied_successfully:
              if categories_data and all(isinstance(item, dict) and 'value' in item and 'percentage' in item for item in categories_data):
                   values = [item["value"] for item in categories_data]
                   weights = [item["percentage"] for item in categories_data]
                   if values and all(isinstance(w, (int, float)) for w in weights) and sum(weights) > 0:
                        print(f"     - Adjusting distribution for categorical column '{col_name}' by sampling from original categories.")
                        try: df.loc[:, col_name] = random.choices(values, weights=weights, k=len(df))
                        except Exception as e: print(f"     - Error sampling for categorical distribution adjustment for '{col_name}': {e}. Keeping current values.")
                   else: print(f"     - Skipping categorical distribution adjustment for '{col_name}': Invalid categories data for sampling.")
              else: print(f"     - Skipping categorical distribution adjustment for '{col_name}': No valid categories data found in schema stats.")


    print("\n--- State after Initial Generation (Faker/Categorical/Regex) ---")
    print(df.head())
    print(df.nunique())

    # --- Apply Functional Dependencies (Method 1) ---
    print("\n  Applying functional dependencies (Method 1) with relationship-aware sampling...")
    functional_dependencies_to_apply = {}
    for col_name, col_schema in schema.get("columns", {}).items():
        if col_name in df.columns and "post_processing_rules" in col_schema: # Use post_processing_rules
            for rule in col_schema["post_processing_rules"]:
                if rule.get("type") in ["functional_dependency", "one_to_one"]:
                    source_col = col_name
                    target_col = rule.get("column")
                    if source_col not in functional_dependencies_to_apply: functional_dependencies_to_apply[source_col] = []
                    functional_dependencies_to_apply[source_col].append(rule)

    if original_df is not None:
        original_target_values_pool = {col: original_df[col].dropna().tolist() for col in original_df.columns}
        used_target_values_for_new_o2o = defaultdict(set) # {target_col: {used_values}}

        for source_col, rel_list in functional_dependencies_to_apply.items():
            for rel in rel_list:
                target_col = rel.get("column"); rel_type = rel.get("type")

                if source_col in df.columns and target_col in df.columns and source_col in original_df.columns and target_col in original_df.columns:
                    if df[source_col].dropna().empty: print(f"    - Skipping functional dependency '{source_col} -> {target_col}': Synthetic source column is entirely null."); continue
                    print(f"    - Enforcing functional dependency: {source_col} -> {target_col} ({rel_type}, relationship-aware)")

                    try:
                        original_mapping_df = original_df.dropna(subset=[source_col, target_col]).drop_duplicates(subset=[source_col], keep='first')
                        if original_mapping_df.empty: print(f"      - Warning: Original data has no non-null pairs for FD {source_col} -> {target_col}. Skipping.")
                        else:
                            original_mapping = original_mapping_df.set_index(source_col)[target_col].to_dict()

                            for index, row in df.iterrows():
                                 synthetic_source_value = row[source_col]
                                 if pd.isna(synthetic_source_value): continue

                                 if synthetic_source_value in original_mapping:
                                      df.loc[index, target_col] = original_mapping[synthetic_source_value]
                                 else:
                                      original_target_pool = original_target_values_pool.get(target_col, [])
                                      col_schema = schema.get("columns", {}).get(target_col, {})
                                      target_col_data_type = col_schema.get("data_type")
                                      target_faker_provider = col_schema.get("faker_provider")
                                      target_faker_args = col_schema.get("faker_args", {})

                                      if not original_target_pool and (target_faker_provider is None or not isinstance(target_faker_provider, str)):
                                           print(f"      - Warning: Original target column '{target_col}' has no non-null values for sampling, and no valid Faker provider for fallback. Cannot enforce FD for new '{synthetic_source_value}'. Setting target to NaN.")
                                           df.loc[index, target_col] = np.nan
                                           continue

                                      valid_candidates = []
                                      # First attempt: Sample from original pool with uniqueness check for O2O
                                      if original_target_pool:
                                           for _ in range(MAX_SAMPLING_RETRIES):
                                                sampled_target_value = random.choice(original_target_pool)
                                                is_valid_candidate = True

                                                # Check Uniqueness for O2O
                                                if rel_type == "one_to_one":
                                                     # Ensure the sampled value is hashable before checking set
                                                     try: hash(sampled_target_value)
                                                     except TypeError:
                                                          # If sampled value from original is unhashable, it can't be in the set, so it's unique among *hashable* used values.
                                                          # However, if we need strict O2O, unhashable values from the pool are problematic.
                                                          # For now, we'll treat unhashable sampled values as not valid candidates for O2O uniqueness check.
                                                          is_valid_candidate = False
                                                          # print(f"      - Warning: Sampled value from original pool for O2O column '{target_col}' is unhashable ({type(sampled_target_value)}). Cannot check uniqueness.") # Keep this for debugging if needed

                                                     if is_valid_candidate and sampled_target_value in used_target_values_for_new_o2o[target_col]:
                                                          is_valid_candidate = False

                                                if is_valid_candidate:
                                                     valid_candidates.append(sampled_target_value)
                                                     if rel_type == "one_to_one":
                                                          # Add to the used set if successfully sampled and valid for O2O
                                                          # Ensure it's hashable before adding
                                                          try: used_target_values_for_new_o2o[target_col].add(sampled_target_value)
                                                          except TypeError: pass # Should be caught above, but safeguard
                                                     break # Found a valid candidate from original pool

                                      if valid_candidates:
                                           df.loc[index, target_col] = valid_candidates[0]
                                      else:
                                           # --- Fallback Strategy ---
                                           fallback_value = np.nan # Default fallback
                                           fallback_attempted = False # Flag to track if any fallback was tried

                                           # Fallback 1: Add offset if numeric or datetime (requires original pool)
                                           if target_col_data_type in ['numerical', 'integer', 'float', 'datetime'] and original_target_pool:
                                                fallback_attempted = True
                                                try:
                                                     sampled_base_value = random.choice(original_target_pool)
                                                     offset = random.randint(1, 10)
                                                     if target_col_data_type in ['numerical', 'integer', 'float']:
                                                          fallback_value = float(sampled_base_value) + offset
                                                          print(f"      - Fallback (Offset): Added numeric offset {offset} to sampled value for '{target_col}'.")
                                                     elif target_col_data_type == 'datetime':
                                                          # Attempt to convert to datetime before adding timedelta
                                                          if isinstance(sampled_base_value, str):
                                                               date_formats_to_try = ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y', '%Y/%m/%d', '%Y%m%d']
                                                               parsed_date = None
                                                               for fmt in date_formats_to_try:
                                                                    try: parsed_date = datetime.strptime(sampled_base_value, fmt); break
                                                                    except ValueError: pass
                                                               if parsed_date: sampled_base_value = parsed_date
                                                               else: raise ValueError("Could not parse date string")
                                                          elif isinstance(sampled_base_value, date):
                                                               sampled_base_value = datetime.combine(sampled_base_value, datetime.min.time())
                                                          elif not isinstance(sampled_base_value, datetime):
                                                               raise TypeError("Not a datetime or date object")

                                                          fallback_value = sampled_base_value + timedelta(days=offset)
                                                          print(f"      - Fallback (Offset): Added datetime offset {offset} days to sampled value for '{target_col}'.")
                                                     # Note: This offset fallback does NOT re-check O2O uniqueness here.
                                                     # The O2O check is done during the initial sampling attempt.

                                                except Exception as e:
                                                     print(f"      - Fallback (Offset) failed for '{target_col}': {e}. Trying Faker fallback.")
                                                     fallback_value = np.nan # Ensure NaN on failure, proceed to next fallback


                                           # Fallback 2: Generate using Faker with uniqueness check for O2O (if offset failed or wasn't applicable)
                                           if pd.isna(fallback_value) and target_faker_provider and isinstance(target_faker_provider, str):
                                                fallback_attempted = True
                                                cleaned_provider = target_faker_provider.strip().replace('Faker.', '').replace('Faker::Date.', '').replace('Faker::Regex.', '')
                                                provider_name_map = {'regex': 'regexify', 'date': 'date_object', 'date_between': 'date_object', 'bban': 'bban', 'random_float': 'pyfloat'}
                                                cleaned_provider = provider_name_map.get(cleaned_provider, cleaned_provider)

                                                faker_method = None
                                                provider_parts = cleaned_provider.split('.', 1)

                                                if len(provider_parts) > 1:
                                                     attribute_name = provider_parts[0]
                                                     method_name = provider_parts[1]
                                                     if hasattr(fake, attribute_name):
                                                          attribute_obj = getattr(fake, attribute_name)
                                                          if hasattr(attribute_obj, method_name):
                                                               faker_method = getattr(attribute_obj, method_name)
                                                else:
                                                     if hasattr(fake, cleaned_provider):
                                                          faker_method = getattr(fake, cleaned_provider)

                                                if faker_method:
                                                     print(f"      - Fallback (Faker): Attempting to generate value for '{target_col}' using '{cleaned_provider}'.")
                                                     fallback_faker_args = target_faker_args.copy()
                                                     valid_args = inspect.signature(faker_method).parameters
                                                     filtered_args = {k: v for k, v in fallback_faker_args.items() if k in valid_args}

                                                     generated_fallback_value = None
                                                     for _ in range(MAX_SAMPLING_RETRIES): # Use same retry limit
                                                          try:
                                                               generated_fallback_value = faker_method(**filtered_args)
                                                               is_valid_fallback_candidate = True

                                                               # Check Uniqueness for O2O for the generated value
                                                               if rel_type == "one_to_one":
                                                                    # Ensure the generated value is hashable before checking set
                                                                    try: hash(generated_fallback_value)
                                                                    except TypeError:
                                                                         # If generated value is unhashable, it can't be in the set, so it's unique among *hashable* used values.
                                                                         # However, if we need strict O2O, unhashable values are problematic.
                                                                         # For now, we'll treat unhashable generated values as not valid candidates for O2O uniqueness check.
                                                                         is_valid_fallback_candidate = False
                                                                         # print(f"      - Fallback (Faker) Warning: Generated value for O2O column '{target_col}' is unhashable ({type(generated_fallback_value)}). Cannot check uniqueness. Skipping this candidate.") # Keep for debugging

                                                                    if is_valid_fallback_candidate and generated_fallback_value in used_target_values_for_new_o2o[target_col]:
                                                                         is_valid_fallback_candidate = False

                                                               if is_valid_fallback_candidate:
                                                                    fallback_value = generated_fallback_value
                                                                    if rel_type == "one_to_one":
                                                                         # Add to the used set if successfully generated and valid for O2O
                                                                         # Ensure it's hashable before adding
                                                                         try: used_target_values_for_new_o2o[target_col].add(fallback_value)
                                                                         except TypeError: pass # Should be caught above, but safeguard
                                                                    print(f"      - Fallback (Faker): Successfully generated valid value for '{target_col}'.")
                                                                    break # Found a valid candidate from Faker

                                                          except Exception as faker_e:
                                                               print(f"      - Fallback (Faker) Error generating value for '{target_col}' with '{cleaned_provider}': {faker_e}. Retrying.")
                                                          generated_fallback_value = None # Ensure None on error

                                                     if pd.isna(fallback_value):
                                                          print(f"      - Fallback (Faker) failed after {MAX_SAMPLING_RETRIES} retries for '{target_col}'. Setting to NaN.")
                                                else:
                                                     print(f"      - Fallback (Faker) skipped for '{target_col}': No valid Faker method found for '{cleaned_provider}'. Setting to NaN.")

                                           # Fallback 3: Default to NaN if no other fallback applied
                                           if pd.isna(fallback_value): # Only print if it's still NaN
                                                # Check if the target column is sparse categorical (high null %, low unique count)
                                                # Use thresholds similar to schema detection, but based on original data stats
                                                original_stats = schema.get("columns", {}).get(target_col, {}).get("stats", {})
                                                original_null_percentage = original_stats.get("null_percentage", 0)
                                                original_unique_count = original_stats.get("unique_count", 0)
                                                original_total_count = original_stats.get("total_count", 0)
                                                original_non_null_count = original_total_count - original_stats.get("null_count", 0)

                                                is_sparse_categorical = False
                                                if original_non_null_count > 0:
                                                     original_unique_ratio = original_unique_count / original_non_null_count
                                                     # Using heuristic thresholds for "sparse categorical"
                                                     if original_null_percentage > 90 and original_unique_count < 10: # Example thresholds
                                                          is_sparse_categorical = True

                                                if not is_sparse_categorical:
                                                     # Only print the warning if it's NOT a sparse categorical column that failed fallbacks
                                                     print(f"      - Fallback: Target column '{target_col}' is not numeric/datetime, Faker fallback failed or skipped. Setting to NaN after retries failed.")
                                                # If it is sparse categorical, we are okay with NaN and suppress the warning as requested.


                                           df.loc[index, target_col] = fallback_value
                                           if pd.isna(fallback_value) and not is_sparse_categorical: # Only print warning if not sparse categorical
                                                print(f"      - Warning: Could not find a valid target value for new source '{synthetic_source_value}' after all fallbacks. Setting target '{target_col}' to NaN.")


                    except Exception as e: print(f"    - Error applying functional dependency '{source_col} -> {target_col}' (Method 1) during row iteration: {e}")
                else:
                    missing_cols = [col for col in [source_col, target_col] if col not in df.columns]
                    if missing_cols: print(f"    - Skipping functional dependency '{source_col} -> {target_col}': Missing columns in synthetic data: {missing_cols}.")
                    elif source_col not in original_df.columns or target_col not in original_df.columns: print(f"    - Skipping functional dependency '{source_col} -> {target_col}': Source or target column not found in original data.")
    else: print("  Skipping functional dependency enforcement: Original data not loaded for mapping.")

    print("\n--- State after Functional Dependencies ---")
    print(df.head())
    print(df.nunique())

    # --- Apply Range Constraints ---
    print("  Applying range constraints...")
    for col_name, col_schema in schema.get("columns", {}).items():
        if col_name in df.columns:
            data_type = col_schema.get("data_type")
            stats = col_schema.get("stats", {})
            min_val = stats.get("min")
            max_val = stats.get("max")

            if data_type in ['numerical', 'integer', 'float'] and min_val is not None and max_val is not None:
                try:
                    # Ensure column is numeric, coercing errors to NaN
                    df[col_name] = pd.to_numeric(df[col_name], errors='coerce')

                    # Convert min/max from schema to appropriate numeric type
                    try:
                         if data_type == 'integer':
                              min_numeric = int(min_val)
                              max_numeric = int(max_val)
                         else: # numerical or float
                              min_numeric = float(min_val)
                              max_numeric = float(max_val)

                         # Apply range check only to non-null numeric values
                         numeric_series = df[col_name].dropna()
                         if not numeric_series.empty:
                              violations_mask = ((numeric_series < min_numeric) | (numeric_series > max_numeric))
                              if violations_mask.any():
                                   # Apply clipping to the original DataFrame using the index of violations
                                   df.loc[numeric_series[violations_mask].index, col_name] = numeric_series[violations_mask].clip(lower=min_numeric, upper=max_numeric)
                                   print(f"    - Clipped {violations_mask.sum()} values in '{col_name}' to range [{min_numeric}, {max_numeric}].")
                         # else: print(f"    - No non-null numeric values in '{col_name}' to apply range constraint.") # Optional verbose logging

                    except (ValueError, TypeError) as e:
                         print(f"    - Error converting min/max values for range constraint on '{col_name}' (min={min_val}, max={max_val}): {e}. Skipping constraint.")
                    except Exception as e:
                         print(f"    - Unexpected error applying numeric range constraint to '{col_name}' (min={min_val}, max={max_val}): {e}")

                except Exception as e: print(f"    - Error during initial numeric coercion for range constraint on '{col_name}': {e}. Skipping constraint.")

            elif data_type == 'datetime' and stats.get('min_date') is not None and stats.get('max_date') is not None:
                 min_date_str = stats.get('min_date'); max_date_str = stats.get('max_date')
                 try:
                      min_date_obj = datetime.fromisoformat(min_date_str); max_date_obj = datetime.fromisoformat(max_date_str)
                      df[col_name] = pd.to_datetime(df[col_name], errors='coerce')
                      violations_mask = ((df[col_name] < min_date_obj) | (df[col_name] > max_date_obj)) & df[col_name].notna()
                      if violations_mask.any():
                           df.loc[violations_mask & (df[col_name] < min_date_obj), col_name] = min_date_obj
                           df.loc[violations_mask & (df[col_name] > max_date_obj), col_name] = max_date_obj
                           print(f"    - Clipped {violations_mask.sum()} datetime values in '{col_name}' to range [{min_date_str}, {max_date_str}].")
                 except Exception as e: print(f"    - Error applying datetime range constraint to '{col_name}' (min={min_date_str}, max={max_date_str}): {e}")

    print("\n--- State after Range Constraints ---")
    print(df.head())
    print(df.nunique())

    # --- Apply Uniqueness Constraint on Primary Key ---
    print("  Enforcing uniqueness on Primary Key...")
    primary_key = None
    for col_name, col_schema in schema.get("columns", {}).items():
        if col_schema.get("key_type", "").lower() == "primary key":
            primary_key = col_name
            break

    if primary_key and primary_key in df.columns:
        try:
            duplicates_mask = df[primary_key].duplicated(keep='first') & df[primary_key].notna()
            duplicate_count = duplicates_mask.sum()
            if duplicate_count > 0:
                print(f"    - Found {duplicate_count} duplicate values in primary key column '{primary_key}'. Attempting to regenerate.")
                duplicate_indices = df[duplicates_mask].index
                pk_schema = schema.get("columns", {}).get(primary_key, {})
                faker_provider = pk_schema.get("faker_provider")
                faker_args = pk_schema.get("faker_args", {}).copy()
                existing_pk_values = set(df[primary_key].dropna().unique())

                for index in duplicate_indices:
                    new_value = None; retry_count = 0; max_retries = 50
                    if faker_provider and isinstance(faker_provider, str):
                        cleaned_provider = faker_provider.strip().replace('Faker.', '').replace('Faker::Date.', '').replace('Faker::Regex.', '')
                        provider_name_map = {'regex': 'regexify', 'date': 'date_object', 'date_between': 'date_object', 'bban': 'bban', 'random_float': 'pyfloat'}
                        cleaned_provider = provider_name_map.get(cleaned_provider, cleaned_provider)

                        # --- Handle Nested and Simple Providers for PK ---
                        faker_method = None
                        provider_parts = cleaned_provider.split('.', 1) # Split by the first dot

                        if len(provider_parts) > 1:
                             # Nested provider (e.g., unique.random_int)
                             attribute_name = provider_parts[0]
                             method_name = provider_parts[1]
                             if hasattr(fake, attribute_name):
                                  attribute_obj = getattr(fake, attribute_name)
                                  if hasattr(attribute_obj, method_name):
                                       faker_method = getattr(attribute_obj, method_name)
                                  else:
                                       print(f"      - Skipping Faker application for PK '{primary_key}': Method '{method_name}' not found on '{attribute_name}'. Cleaned provider: '{cleaned_provider}'")
                             else:
                                  print(f"      - Skipping Faker application for PK '{primary_key}': Attribute '{attribute_name}' not found on fake object. Cleaned provider: '{cleaned_provider}'")
                        else:
                             # Simple provider (e.g., name, random_int)
                             if hasattr(fake, cleaned_provider):
                                  faker_method = getattr(fake, cleaned_provider)
                             else:
                                  print(f"      - Skipping Faker application for PK '{primary_key}': Provider '{cleaned_provider}' not found on fake object.")

                        if cleaned_provider == 'regexify':
                             regex_pattern = faker_args.get("text") or faker_args.get("pattern")
                             if isinstance(regex_pattern, str):
                                  while (new_value is None or new_value in existing_pk_values) and retry_count < max_retries:
                                       new_value = generate_from_regex_pattern(regex_pattern)
                                       retry_count += 1
                             else: print(f"      - Warning: No valid regex pattern found in faker_args for PK '{primary_key}' with regexify. Falling back.")
                        elif faker_method: # Only proceed if faker_method was successfully found
                             valid_args = inspect.signature(faker_method).parameters
                             filtered_args = {k: v for k, v in faker_args.items() if k in valid_args}

                             while (new_value is None or new_value in existing_pk_values) and retry_count < max_retries:
                                 current_faker_args = filtered_args.copy()
                                 if cleaned_provider == 'random_int' or cleaned_provider == 'unique.random_int': # Handle both simple and unique random_int
                                      current_max_id = None
                                      try: current_max_id = max(v for v in existing_pk_values if isinstance(v, (int, float))); current_max_id = int(current_max_id) if not pd.isna(current_max_id) else 0
                                      except ValueError: current_max_id = 0 # Handle empty sequence
                                      current_faker_args['min'] = current_max_id + 1
                                      if 'max' not in current_faker_args or current_faker_args['max'] <= current_faker_args['min']:
                                           current_faker_args['max'] = current_faker_args['min'] + 100000 # Ensure a range

                                 try: new_value = faker_method(**current_faker_args)
                                 except Exception as e: print(f"      - Error calling Faker method '{cleaned_provider}': {e}. Retrying."); new_value = None # Ensure new_value is None on error
                                 retry_count += 1
                        else:
                             # If faker_method was not found (handled above), the fallback will be used below
                             pass # Do nothing here, let the fallback handle it

                    # --- Fallback for PK regeneration if Faker/Regex failed ---
                    if new_value is None or new_value in existing_pk_values:
                        if pd.api.types.is_numeric_dtype(df[primary_key]):
                             current_max_id = df[primary_key].max()
                             current_max_id = int(current_max_id) if not pd.isna(current_max_id) else 0
                             new_value_seq = current_max_id + 1
                             while new_value_seq in existing_pk_values: new_value_seq += 1
                             new_value = new_value_seq
                        else:
                             new_value_placeholder = f"GENERATED_DUP_{index}_{random.randint(10000, 99999)}"
                             while new_value_placeholder in existing_pk_values: new_value_placeholder = f"GENERATED_DUP_{index}_{random.randint(10000, 99999)}"
                             new_value = new_value_placeholder
                        print(f"      - Using fallback sequential/placeholder value '{new_value}' for duplicate at index {index}.")

                    df.loc[index, primary_key] = new_value
                    existing_pk_values.add(new_value)
                print(f"    - Attempted to regenerate {duplicate_count} duplicate primary key values.")
        except Exception as e: print(f"    - Error enforcing uniqueness on primary key '{primary_key}': {e}")
    elif not primary_key: print("    - Skipping uniqueness enforcement: No primary key defined in schema.")
    else: print(f"    - Skipping uniqueness enforcement: Primary key column '{primary_key}' not found in synthetic data.")

    print("\n--- State after Uniqueness Enforcement ---")
    print(df.head())
    print(df.nunique())

    # --- Introduce Nulls based on null_percentage ---
    print("  Introducing nulls based on schema null percentages...")
    for col_name, col_schema in schema.get("columns", {}).items():
        if col_name in df.columns:
            null_percentage = col_schema.get("null_percentage", 0)
            key_type = col_schema.get("key_type", "None")

            # Do not introduce nulls for non-nullable Primary Keys
            if key_type.lower() == "primary key" and null_percentage == 0:
                 print(f"    - Skipping null introduction for Primary Key '{col_name}' (non-nullable).")
                 continue

            if null_percentage > 0:
                try:
                    num_rows_to_null = int(len(df) * (null_percentage / 100))
                    # Ensure we don't nullify more non-null values than exist
                    non_null_indices = df.index[df[col_name].notna()].tolist()
                    if len(non_null_indices) > num_rows_to_null:
                         indices_to_null = random.sample(non_null_indices, num_rows_to_null)
                         df.loc[indices_to_null, col_name] = np.nan
                         print(f"    - Introduced {num_rows_to_null} nulls into '{col_name}' ({null_percentage:.1f}%).")
                    elif len(non_null_indices) > 0: # Nullify all non-null if target is more than available
                         df.loc[non_null_indices, col_name] = np.nan
                         print(f"    - Introduced {len(non_null_indices)} nulls into '{col_name}' (all remaining non-nulls).")
                    else:
                         print(f"    - Skipping null introduction for '{col_name}': No non-null values to nullify.")

                except Exception as e:
                    print(f"    - Error introducing nulls for '{col_name}' ({null_percentage:.1f}%): {e}")

    print("\n--- State after Null Introduction ---")
    print(df.head())
    print(df.isnull().sum())


    print("\nPost-processing constraints applied.")
    return df


# --- Relationship Consistency Calculation for Reporting ---
def calculate_relationship_consistency(df, schema):
    """
    Calculates the consistency percentage for schema-defined relationships in the synthetic data.
    Focuses on Functional Dependencies, One-to-One, and Value Relationships (<, >, =).
    Args:
        df (pd.DataFrame): The synthetic DataFrame.
        schema (dict): The enhanced schema dictionary.
    Returns:
        dict: A dictionary where keys are relationship descriptions and values are consistency percentages.
    """
    print("\nCalculating relationship consistency in synthetic data for reporting...")
    consistency_report = {}
    # Use a tolerance for float comparisons in value relationships
    VALUE_RELATIONSHIP_TOLERANCE = 1e-6 # Same tolerance as used in schema detection

    # Collect all relationships from the schema that we want to report on
    relationships_to_report = []
    for col_name, col_schema in schema.get("columns", {}).items():
        if "post_processing_rules" in col_schema:
            for rule in col_schema["post_processing_rules"]:
                # Only consider FD, O2O, and Value Relationships (<, >, =)
                if rule.get("type") in ["functional_dependency", "one_to_one"]:
                     # Add source column to the rule for easier processing
                     relationships_to_report.append({**rule, "source_column": col_name})
                elif rule.get("type") == "value_relationship" and rule.get("relationship") in ["less_than", "greater_than", "equal_to"]:
                     relationships_to_report.append({**rule, "source_column": col_name})

    if not relationships_to_report:
        print("  No relevant relationships found in schema for consistency reporting.")
        return consistency_report

    for rel_info in relationships_to_report:
        rel_type = rel_info.get("type")
        source_col = rel_info.get("source_column")
        target_col = rel_info.get("column") # For FD/O2O/Value
        relationship_detail = rel_info.get("relationship") # For Value

        if source_col not in df.columns or target_col not in df.columns:
            print(f"  Skipping consistency check for relationship involving missing columns: {source_col}, {target_col}")
            continue

        # Filter out rows where either column is null for the consistency check
        df_filtered = df[[source_col, target_col]].dropna()
        num_rows_checked = len(df_filtered)

        if num_rows_checked == 0:
            print(f"  Skipping consistency check for relationship {source_col} -> {target_col} (Type: {rel_type}, Detail: {relationship_detail}): No non-null rows in synthetic data.")
            consistency_report[f"{source_col} -> {target_col} (Type: {rel_type}, Detail: {relationship_detail})"] = "N/A (No non-null rows)"
            continue

        num_consistent_rows = 0

        try:
            if rel_type in ["functional_dependency", "one_to_one"]:
                # For FD/O2O, check if target value is unique for each source value
                # This is a simplified check for reporting consistency in synthetic data
                # A more rigorous check would involve mapping back to original, but this gives an idea.
                # Check if the number of unique pairs equals the number of unique source values in the filtered data
                num_unique_pairs = df_filtered.drop_duplicates(subset=[source_col, target_col]).shape[0]
                num_unique_sources = df_filtered[source_col].nunique()

                # If number of unique pairs equals number of unique sources, it's functionally dependent
                if num_unique_pairs == num_unique_sources:
                     num_consistent_rows = num_rows_checked # Consider all checked rows as consistent if FD holds

                relationship_desc = f"{source_col} -> {target_col} (Type: {rel_type})"

            elif rel_type == "value_relationship" and relationship_detail in ["less_than", "greater_than", "equal_to"]:
                # Ensure both columns are numeric for value relationship checks
                if pd.api.types.is_numeric_dtype(df_filtered[source_col]) and pd.api.types.is_numeric_dtype(df_filtered[target_col]):
                    series1 = pd.to_numeric(df_filtered[source_col], errors='coerce')
                    series2 = pd.to_numeric(df_filtered[target_col], errors='coerce')

                    # Filter out rows where coercion failed (should be handled by dropna but double check)
                    valid_comparison_mask = series1.notna() & series2.notna()
                    series1_valid = series1[valid_comparison_mask]
                    series2_valid = series2[valid_comparison_mask]
                    num_valid_comparison_rows = len(series1_valid)

                    if num_valid_comparison_rows > 0:
                         if relationship_detail == "less_than":
                              num_consistent_rows = (series1_valid < series2_valid).sum()
                         elif relationship_detail == "greater_than":
                              num_consistent_rows = (series1_valid > series2_valid).sum()
                         elif relationship_detail == "equal_to":
                              # Use numpy.isclose for floating point equality check
                              num_consistent_rows = np.isclose(series1_valid, series2_valid, atol=VALUE_RELATIONSHIP_TOLERANCE).sum()

                         # Adjust num_rows_checked to reflect only rows where comparison was valid
                         num_rows_checked = num_valid_comparison_rows
                    else:
                         print(f"  Skipping consistency check for numeric value relationship {source_col} {relationship_detail} {target_col}: No valid numeric pairs in synthetic data.")
                         consistency_report[f"{source_col} {relationship_detail} {target_col} (Type: {rel_type})"] = "N/A (No valid numeric pairs)"
                         continue # Skip to next relationship

                else:
                     print(f"  Skipping consistency check for value relationship {source_col} {relationship_detail} {target_col}: One or both columns are not numeric in synthetic data.")
                     consistency_report[f"{source_col} {relationship_detail} {target_col} (Type: {rel_type})"] = "N/A (Non-numeric columns)"
                     continue # Skip to next relationship

                relationship_desc = f"{source_col} {relationship_detail} {target_col} (Type: {rel_type})"

            else:
                 # Should not happen with current filtering, but as a fallback
                 relationship_desc = f"Unknown Relationship: {rel_info}"
                 consistency_report[relationship_desc] = "N/A (Unsupported Type)"
                 continue

            consistency_percentage = (num_consistent_rows / num_rows_checked) * 100 if num_rows_checked > 0 else 0
            consistency_report[relationship_desc] = f"{consistency_percentage:.2f}%"

        except Exception as e:
            print(f"  Error calculating consistency for relationship {source_col} -> {target_col} (Type: {rel_type}, Detail: {relationship_detail}): {e}")
            consistency_report[f"{source_col} -> {target_col} (Type: {rel_type}, Detail: {relationship_detail})"] = f"Error: {e}"


    print("Relationship consistency calculation complete.")
    return consistency_report


def generate_synthetic_data_with_sdv(original_df, schema, num_rows, output_file):
    """Generates synthetic data, applies post-processing constraints, Faker, etc."""
    try:
        print("\n--- Starting Synthetic Data Generation ---")

        print("\n1. Initializing synthetic DataFrame with correct columns...")
        # Use column names from schema to ensure correct column order and presence
        synthetic_data = pd.DataFrame(columns=schema.get("columns", {}).keys())
        synthetic_data = synthetic_data.reindex(range(num_rows))
        print(f"   Initialized empty synthetic DataFrame with {len(synthetic_data)} rows and {len(synthetic_data.columns)} columns.")

        # original_df is passed in from main for post-processing
        if original_df is None:
             print("\n   Warning: Original data not loaded for post-processing. Functional dependencies and sampling from original distribution cannot be fully applied.")


        print("\n2. Applying post-processing (Initial Faker/Regex generation, categorical distributions, custom rules, constraints)...")
        synthetic_data = apply_post_processing_rules(synthetic_data, schema, original_df)

        print("\n3. Calculating relationship consistency in synthetic data and generating report...")
        # Calculate consistency based on the generated synthetic data and the schema rules
        consistency_report_data = calculate_relationship_consistency(synthetic_data, schema)

        if consistency_report_data:
            report_rows = []
            for rel_desc, consistency in consistency_report_data.items():
                report_rows.append({"Relationship": rel_desc, "Consistency in Synthetic Data": consistency})

            report_df = pd.DataFrame(report_rows)
            try:
                report_df.to_csv(REPORT_CSV, index=False, encoding='utf-8')
                print(f"   Relationship consistency report saved to {REPORT_CSV}")
            except Exception as csv_e:
                print(f"   Error saving relationship consistency report CSV: {csv_e}")
        else:
            print("   No relationships to report consistency for.")


        print(f"\n4. Saving final synthetic data to {output_file}...")
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

    print(f"\nLoading original data from: {INPUT_CSV} for post-processing and reporting...")
    original_df = None
    try:
        # Load original data for use in post-processing (FD mapping, sampling)
        # and for comparison in the reporting step.
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
    # If original_df loading failed, pass None.
    success = generate_synthetic_data_with_sdv(original_df, schema, NUM_ROWS, OUTPUT_CSV)

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
