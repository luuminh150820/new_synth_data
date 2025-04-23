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
CORRELATION_THRESHOLD = 0.7
RELATIONSHIP_CONSISTENCY_THRESHOLD = 0.999
FORMULA_TOLERANCE = 1e-6
REPORT_CSV = "relationship_comparison_report.csv"
MAX_SAMPLING_RETRIES = 10 # Increased retries for relationship-aware sampling

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
        return None

    result = []
    i = 0
    while i < len(pattern):
        char = pattern[i]

        if char == '\\':
             if i + 1 < len(pattern):
                  next_char = pattern[i+1]
                  if next_char == 'd':
                       result.append(random.choice(string.digits))
                  elif next_char == '.':
                       result.append('.')
                  else:
                       result.append(next_char)
                  i += 2
             else:
                  print(f"Warning: Trailing '\\' in regex pattern: {pattern}. Ignoring.")
                  i += 1
        elif char == '#':
            result.append(random.choice(string.digits))
            i += 1
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

            if possible_chars:
                result.append(random.choice(possible_chars))
            else:
                print(f"Warning: Empty or unparseable character set in regex pattern: {pattern}. Skipping.")
                pass

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
                else:
                    repetition_count = int(count_str.strip())
            except ValueError:
                print(f"Warning: Invalid repetition count '{{{count_str}}}' in regex pattern: {pattern}. Using count 1.")
                repetition_count = 1

            if result:
                last_char_or_group = result.pop()
                result.extend([last_char_or_group] * repetition_count)
            else:
                 print(f"Warning: Repetition '{{{count_str}}}' found at the start or after an empty group in regex pattern: {pattern}. Ignoring repetition.")

            i = end_brace + 1

        elif char in '?*+':
             if result:
                  last_char_or_group = result.pop()
                  if char == '?':
                       if random.random() < 0.5:
                            result.append(last_char_or_group)
                  elif char == '*':
                       num_repetitions = random.randint(0, 5)
                       result.extend([last_char_or_group] * num_repetitions)
                  elif char == '+':
                       num_repetitions = random.randint(1, 5)
                       result.extend([last_char_or_group] * num_repetitions)
             else:
                  print(f"Warning: Quantifier '{char}' found at the start or after an empty group in regex pattern: {pattern}. Ignoring.")
             i += 1
        elif char == '(':
             if i + 3 < len(pattern) and pattern[i+1:i+3] == '?:':
                  end_paren = pattern.find(')', i + 3)
                  if end_paren != -1:
                       group_content = pattern[i + 3:end_paren]
                       generated_group_content = generate_from_regex_pattern(group_content)
                       if generated_group_content is not None:
                            result.append(generated_group_content)
                       i = end_paren + 1
                  else:
                       print(f"Warning: Unclosed non-capturing group '(?:' in regex pattern: {pattern}. Skipping.")
                       return "".join(result)
             else:
                  result.append(char)
                  i += 1
        elif char == ')':
             i += 1
        else:
            result.append(char)
            i += 1

    return "".join(result)


# --- New Function for Post-processing Constraints (Method 1) ---
def apply_post_processing_rules(df, schema, original_df):
    """
    Applies post-processing rules to the synthetic DataFrame based on the schema.
    Includes Faker application, custom regex generation, preserved categorical distributions,
    functional dependencies (Method 1 with relationship-aware sampling),
    value relationships, range constraints, and uniqueness enforcement.

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

        if not values or not all(isinstance(w, (int, float)) for w in weights) or sum(weights) <= 0:
             return None

        try:
            return random.choices(values, weights=weights, k=1)[0]
        except Exception as e:
             print(f"     - Error during weighted random selection: {e}. Returning None.")
             return None


    for col_name, col_schema in schema.get("columns", {}).items():
         if col_name not in df.columns: continue

         is_cat_in_schema = col_schema.get("data_type") == "categorical"
         categories_data = None
         if "stats" in col_schema:
             categories_data = col_schema["stats"].get("categories") or col_schema["stats"].get("top_categories")

         faker_provider = col_schema.get("faker_provider")

         if faker_provider is not None:
             if isinstance(faker_provider, str):
                 cleaned_provider = str(faker_provider).strip()
                 cleaned_provider = re.sub(r'^[\'"]|[\'"]$', '', cleaned_provider)
                 cleaned_provider = cleaned_provider.replace('Faker.', '').replace('Faker::Date.', '').replace('Faker::Regex.', '')

                 provider_name_map = {
                     'regex': 'regexify', 'date': 'date_object', 'date_between': 'date_object', 'bban': 'bban',
                     'random_float': 'pyfloat'
                 }
                 cleaned_provider = provider_name_map.get(cleaned_provider, cleaned_provider)

                 if cleaned_provider == 'regexify':
                      print(f"     - Applying custom regex generation for column '{col_name}'")
                      faker_args = col_schema.get("faker_args", {})
                      regex_pattern = faker_args.get("text") or faker_args.get("pattern")

                      if isinstance(regex_pattern, str):
                           try:
                                df.loc[:, col_name] = df[col_name].apply(
                                     lambda x: generate_from_regex_pattern(regex_pattern)
                                )
                           except Exception as e:
                                print(f"     - Error applying custom regex generation to '{col_name}': {e}")
                      else:
                           print(f"     - Skipping custom regex generation for '{col_name}': No valid regex pattern found in faker_args.")

                      continue

                 if hasattr(fake, cleaned_provider):
                      print(f"     - Applying Faker provider '{cleaned_provider}' to column '{col_name}'")
                      faker_args = col_schema.get("faker_args", {}).copy()

                      if cleaned_provider == 'pyfloat':
                           faker_args['min_value'] = faker_args.pop('min', None)
                           faker_args['max_value'] = faker_args.pop('max', None)
                           faker_args.pop('step', None)
                           faker_args.pop('precision', None)
                           if faker_args.get('min_value') is not None and faker_args['min_value'] > 0:
                                faker_args['positive'] = True
                           elif 'positive' in faker_args:
                                faker_args.pop('positive', None)

                      elif cleaned_provider == 'random_int':
                           if 'min' in faker_args and isinstance(faker_args['min'], float):
                                faker_args['min'] = int(faker_args['min'])
                           if 'max' in faker_args and isinstance(faker_args['max'], float):
                                faker_args['max'] = int(faker_args['max'])

                      elif cleaned_provider == 'random_element':
                           if categories_data:
                                elements_dict = {item['value']: item['percentage'] for item in categories_data if 'value' in item and 'percentage' in item}
                                if elements_dict:
                                     faker_args['elements'] = elements_dict
                                     faker_args.pop('weights', None)
                                else:
                                     print(f"         Warning: Categories data for '{col_name}' is invalid for random_element. Skipping Faker.")
                                     continue
                           elif 'elements' not in faker_args:
                                print(f"         Warning: random_element for '{col_name}' has no 'elements' arg and no categories data. Skipping Faker.")
                                continue

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


                      try:
                          faker_method = getattr(fake, cleaned_provider)
                          valid_args = inspect.signature(faker_method).parameters
                          filtered_args = {k: v for k, v in faker_args.items() if k in valid_args}

                          df.loc[:, col_name] = df[col_name].apply(
                              lambda x: faker_method(**filtered_args)
                          )
                      except AttributeError:
                          print(f"       Warning: Faker provider '{cleaned_provider}' not found on fake object. Keeping generated values for '{col_name}'.")
                      except Exception as e:
                          print(f"       Error applying Faker provider '{cleaned_provider}' to '{col_name}': {str(e)}. Keeping generated values.")
                 else:
                     print(f"     - Skipping Faker application for column '{col_name}': Invalid provider name format.")

         faker_applied_successfully = False
         if faker_provider is not None and isinstance(faker_provider, str):
              cleaned_provider_status = str(faker_provider).strip()
              cleaned_provider_status = re.sub(r'^[\'"]|[\'"]$', '', cleaned_provider_status)
              cleaned_provider_status = cleaned_provider_status.replace('Faker.', '').replace('Faker::Date.', '').replace('Faker::Regex.', '')
              provider_name_map = {
                  'regex': 'regexify', 'date': 'date_object', 'date_between': 'date_object', 'bban': 'bban',
                  'random_float': 'pyfloat'
              }
              cleaned_provider_status = provider_name_map.get(cleaned_provider_status, cleaned_provider_status)

              if cleaned_provider_status == 'regexify':
                   faker_applied_successfully = True
              elif cleaned_provider_status and hasattr(fake, cleaned_provider_status):
                   faker_applied_successfully = True


         if is_cat_in_schema and categories_data and not faker_applied_successfully:
              if categories_data and all(isinstance(item, dict) and 'value' in item and 'percentage' in item for item in categories_data):
                   values = [item["value"] for item in categories_data]
                   weights = [item["percentage"] for item in categories_data]

                   if values and all(isinstance(w, (int, float)) for w in weights) and sum(weights) > 0:
                        print(f"     - Adjusting distribution for categorical column '{col_name}' by sampling from original categories.")
                        try:
                            sampled_values = random.choices(values, weights=weights, k=len(df))
                            df.loc[:, col_name] = sampled_values
                        except Exception as e:
                             print(f"     - Error sampling for categorical distribution adjustment for '{col_name}': {e}. Keeping current values.")
                   else:
                        print(f"     - Skipping categorical distribution adjustment for '{col_name}': Invalid categories data for sampling.")
              else:
                   print(f"     - Skipping categorical distribution adjustment for '{col_name}': No valid categories data found in schema stats.")


    print("\n--- State after Initial Generation (Faker/Categorical/Regex) ---")
    print(df.head())
    print(df.nunique())


    # --- Apply Post-processing Constraints (Functional Dependencies (Method 1), Value Relationships, Formulas, Ranges, Uniqueness) ---
    # Order of application: FD -> Value Relationships -> Formulas -> Ranges -> Uniqueness

    # Apply functional dependencies (Method 1)
    print("\n  Applying functional dependencies (Method 1) with relationship-aware sampling...")
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

    # Identify relevant value comparison and formula relationships for conditional sampling
    relevant_value_relationships = defaultdict(list) # {(col1, col2): [rel_info]}
    relevant_formula_relationships = defaultdict(list) # {target_col: [rel_info]}

    for col_name, col_schema in schema.get("columns", {}).items():
         if col_name in df.columns and "post_processing" in col_schema:
              for rule in col_schema["post_processing"]:
                   if rule.get("type") == "value_relationship":
                        source_col = col_name
                        target_col = rule.get("column")
                        # Only consider numerical value relationships for now
                        if source_col in df.columns and target_col in df.columns and \
                           pd.api.types.is_numeric_dtype(df[source_col]) and pd.api.types.is_numeric_dtype(df[target_col]):
                             relevant_value_relationships[(source_col, target_col)].append(rule)
                   elif rule.get("type") == "formula":
                        target_col = rule.get("target_column")
                        source_cols = rule.get("source_columns")
                        # Only consider numerical formulas where all involved columns are numeric
                        if target_col in df.columns and all(col in df.columns and pd.api.types.is_numeric_dtype(df[col]) for col in source_cols):
                             relevant_formula_relationships[target_col].append(rule)


    if original_df is not None:
        # Pre-calculate original target values for sampling
        original_target_values_pool = {}
        for col in original_df.columns:
             original_target_values_pool[col] = original_df[col].dropna().tolist()


        # Track used target values for O2O relationships involving new source values
        used_target_values_for_new_o2o = defaultdict(set) # {target_col: {used_values}}


        for source_col, rel_list in functional_dependencies_to_apply.items():
            for rel in rel_list:
                target_col = rel.get("column")
                rel_type = rel.get("type") # Should be 'functional_dependency' or 'one_to_one'

                if source_col in df.columns and target_col in df.columns and \
                   source_col in original_df.columns and target_col in original_df.columns:

                    if df[source_col].dropna().empty:
                         print(f"    - Skipping functional dependency '{source_col} -> {target_col}': Synthetic source column is entirely null.")
                         continue

                    print(f"    - Enforcing functional dependency: {source_col} -> {target_col} (Method 1, relationship-aware)")

                    try:
                        original_mapping_df = original_df.dropna(subset=[source_col, target_col]).drop_duplicates(subset=[source_col], keep='first')
                        if original_mapping_df.empty:
                            print(f"      - Warning: Original data has no non-null pairs for functional dependency {source_col} -> {target_col}. Skipping.")
                            continue

                        original_mapping = original_mapping_df.set_index(source_col)[target_col].to_dict()

                        # Iterate through synthetic data rows
                        for index, row in df.iterrows():
                             synthetic_source_value = row[source_col]

                             if pd.isna(synthetic_source_value):
                                  continue # Skip if the source value is null in synthetic data

                             # If the synthetic source value exists in the original data, use the direct mapping
                             if synthetic_source_value in original_mapping:
                                  df.loc[index, target_col] = original_mapping[synthetic_source_value]
                             else:
                                  # This is a new synthetic source value, need relationship-aware sampling
                                  # Get the pool of original target values for sampling
                                  original_target_pool = original_target_values_pool.get(target_col, [])

                                  if not original_target_pool:
                                       print(f"      - Warning: Original target column '{target_col}' has no non-null values for sampling for new source values. Cannot enforce FD for new '{synthetic_source_value}'. Setting target to NaN.")
                                       df.loc[index, target_col] = np.nan
                                       continue

                                  valid_candidates = []
                                  # Track used values for O2O for this specific new source value attempt
                                  used_in_this_attempt = set()

                                  # Attempt to find a valid candidate from the original pool
                                  for _ in range(MAX_SAMPLING_RETRIES):
                                       sampled_target_value = random.choice(original_target_pool)

                                       # Check if this sampled value satisfies relevant relationships with the synthetic source value
                                       is_valid_candidate = True

                                       # Check Value Relationships (A < B, A > B)
                                       for (s_col, t_col), rels in relevant_value_relationships.items():
                                            if (s_col == source_col and t_col == target_col) or (s_col == target_col and t_col == source_col):
                                                 # Ensure both values are numeric for comparison
                                                 if pd.api.types.is_numeric_dtype(pd.Series([synthetic_source_value])) and pd.api.types.is_numeric_dtype(pd.Series([sampled_target_value])):
                                                      num_synthetic_source = float(synthetic_source_value)
                                                      num_sampled_target = float(sampled_target_value)

                                                      for val_rel in rels:
                                                           relationship = val_rel.get("relationship")
                                                           if relationship in ["<=", "less_than_or_equal_to"]:
                                                                if not (num_synthetic_source <= num_sampled_target):
                                                                     is_valid_candidate = False
                                                                     break
                                                           elif relationship in [">=", "greater_than_or_equal_to"]:
                                                                if not (num_synthetic_source >= num_sampled_target):
                                                                     is_valid_candidate = False
                                                                     break
                                                 else:
                                                      # If not numeric, value relationship cannot be checked for this pair
                                                      pass # Assume valid if relationship is numeric but values aren't

                                       if not is_valid_candidate:
                                            continue # Try another sample if value relationship is violated

                                       # Check Formula Relationships (Target = Source1 + Source2)
                                       for target, rels in relevant_formula_relationships.items():
                                            if target == target_col: # Check formulas where the target column is our current target
                                                 for formula_rel in rels:
                                                      source_cols_formula = formula_rel.get("source_columns", [])
                                                      formula_str = formula_rel.get("formula")

                                                      # If the formula involves the current source_col and target_col,
                                                      # we need to see if the relationship holds.
                                                      # This is complex as it might require values from other source columns.
                                                      # For simplicity in this iteration, we will primarily focus on
                                                      # direct value comparisons (A < B, A > B) for relationship-aware sampling.
                                                      # Formula relationships are better enforced *after* all values are set.
                                                      pass # Skipping formula check in sampling for now for simplicity


                                       if not is_valid_candidate:
                                            continue # Try another sample if formula relationship is violated (if implemented)


                                       # Check Uniqueness for O2O (if applicable)
                                       if rel_type == "one_to_one":
                                            if sampled_target_value in used_target_values_for_new_o2o[target_col]:
                                                 is_valid_candidate = False # Already used for another new source value


                                       if is_valid_candidate:
                                            valid_candidates.append(sampled_target_value)
                                            # If O2O, mark this target value as used for this new source value
                                            if rel_type == "one_to_one":
                                                 used_target_values_for_new_o2o[target_col].add(sampled_target_value)
                                            break # Found a valid candidate, exit retry loop
                                       else:
                                            used_in_this_attempt.add(sampled_target_value) # Track failed attempts

                                  # Assign the chosen value or fallback
                                  if valid_candidates:
                                       df.loc[index, target_col] = valid_candidates[0] # Assign the first valid candidate found
                                       # print(f"      - Sampled valid target value '{valid_candidates[0]}' for new source '{synthetic_source_value}'.") # Reduced logging
                                  else:
                                       # Fallback: If no valid candidate found after retries, assign NaN
                                       print(f"      - Warning: Could not find a valid target value for new source '{synthetic_source_value}' after {MAX_SAMPLING_RETRIES} retries that satisfies relevant constraints. Setting target '{target_col}' to NaN.")
                                       df.loc[index, target_col] = np.nan


                    except Exception as e:
                        print(f"    - Error applying functional dependency '{source_col} -> {target_col}' (Method 1) during row iteration: {e}")

                else:
                    missing_cols = [col for col in [source_col, target_col] if col not in df.columns]
                    if missing_cols:
                         print(f"    - Skipping functional dependency '{source_col} -> {target_col}': Missing columns in synthetic data: {missing_cols}.")
                    elif source_col not in original_df.columns or target_col not in original_df.columns:
                         print(f"    - Skipping functional dependency '{source_col} -> {target_col}': Source or target column not found in original data.")


    else:
        print("  Skipping functional dependency enforcement: Original data not loaded for mapping.")

    print("\n--- State after Functional Dependencies ---")
    print(df.head())
    print(df.nunique())


    # Apply detected value relationships (e.g., colA <= colB) - Do this AFTER functional dependencies
    print("  Applying detected value relationships...")
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
                    # Check if the synthetic source or target column has any non-null values to base the relationship on
                    if df[col_name].dropna().empty or df[target_col].dropna().empty:
                         print(f"    - Skipping value relationship '{col_name} {relationship} {target_col}': One or both synthetic columns are entirely null.")
                         continue # Skip if either source or target is null in synthetic data

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


    print("\n--- State after Value Relationships ---")
    print(df.head())
    print(df.nunique())


    # Apply formula relationships (e.g., colA = colB + colC) - Do this AFTER value relationships
    print("  Applying formula relationships...")
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
                # Check if ALL synthetic source columns for the formula have any non-null values
                if any(df[col].dropna().empty for col in source_cols):
                     null_sources = [col for col in source_cols if df[col].dropna().empty]
                     print(f"    - Skipping formula enforcement for '{target_col} = {formula_str}': One or more synthetic source columns are entirely null: {null_sources}.")
                     continue

                try:
                    print(f"    - Enforcing formula: {target_col} = {formula_str}")

                    try:
                         valid_formula = True
                         formula_parts = re.findall(r'[a-zA-Z0-9_]+|[+\-*/]', formula_str)
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
                              with np.errstate(divide='ignore', invalid='ignore'):
                                   expected_values = df.eval(formula_str)

                              violations_mask = ~np.isclose(df[target_col], expected_values, atol=FORMULA_TOLERANCE, equal_nan=True) & df[target_col].notna() & expected_values.notna()

                              if violations_mask.any():
                                   num_violations = violations_mask.sum()
                                   df.loc[violations_mask, target_col] = expected_values[violations_mask]
                                   print(f"      - Enforced formula '{target_col} = {formula_str}' on {num_violations} rows.")
                              else:
                                   print(f"      - Formula '{target_col} = {formula_str}' already holds for all relevant rows.")

                         else:
                              print(f"    - Skipping formula enforcement for '{target_col} = {formula_str}' due to invalid formula string.")

                    except Exception as eval_e:
                        print(f"      - Error evaluating or applying formula '{formula_str}' for '{target_col}': {eval_e}. Skipping.")
                except Exception as e:
                     print(e)

            else:
                missing_cols = [col for col in source_cols + [target_col] if col not in df.columns]
                non_numeric_cols = [col for col in source_cols + [target_col] if col in df.columns and not pd.api.types.is_numeric_dtype(df[col])]
                if missing_cols:
                     print(f"    - Skipping formula enforcement for '{target_col} = {formula_str}': Missing columns in synthetic data: {missing_cols}.")
                if non_numeric_cols:
                     print(f"    - Skipping formula enforcement for '{target_col} = {formula_str}': Non-numeric columns involved: {non_numeric_cols}.")


    print("\n--- State after Formula Relationships ---")
    print(df.head())
    print(df.nunique())


    # Apply range constraints (based on min/max from basic stats)
    print("  Applying range constraints...")
    for col_name, col_schema in schema.get("columns", {}).items():
        if col_name in df.columns:
            data_type = col_schema.get("data_type")
            stats = col_schema.get("stats", {})
            min_val = stats.get("min")
            max_val = stats.get("max")

            if data_type in ['numerical', 'integer', 'float'] and min_val is not None and max_val is not None:
                try:
                    df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
                    violations_mask = ((df[col_name] < min_val) | (df[col_name] > max_val)) & df[col_name].notna()
                    original_violations = violations_mask.sum()

                    if original_violations > 0:
                         df.loc[violations_mask, col_name] = df.loc[violations_mask, col_name].clip(lower=min_val, upper=max_val)
                         print(f"    - Clipped {original_violations} values in '{col_name}' to range [{min_val}, {max_val}].")

                except Exception as e:
                    print(f"    - Error applying range constraint to '{col_name}' (min={min_val}, max={max_val}): {e}")

            elif data_type == 'datetime' and stats.get('min_date') is not None and stats.get('max_date') is not None:
                 min_date_str = stats.get('min_date')
                 max_date_str = stats.get('max_date')
                 try:
                      min_date_obj = datetime.fromisoformat(min_date_str)
                      max_date_obj = datetime.fromisoformat(max_date_str)

                      df[col_name] = pd.to_datetime(df[col_name], errors='coerce')

                      violations_mask = ((df[col_name] < min_date_obj) | (df[col_name] > max_date_obj)) & df[col_name].notna()

                      if violations_mask.any():
                           num_violations = violations_mask.sum()
                           df.loc[violations_mask & (df[col_name] < min_date_obj), col_name] = min_date_obj
                           df.loc[violations_mask & (df[col_name] > max_date_obj), col_name] = max_date_obj
                           print(f"    - Clipped {num_violations} datetime values in '{col_name}' to range [{min_date_str}, {max_date_str}].")

                 except Exception as e:
                      print(f"    - Error applying datetime range constraint to '{col_name}' (min={min_date_str}, max={max_date_str}): {e}")

    print("\n--- State after Range Constraints ---")
    print(df.head())
    print(df.nunique())


    # Apply uniqueness constraint on Primary Key (if identified)
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
                    new_value = None
                    retry_count = 0
                    max_retries = 50

                    if faker_provider:
                        try:
                            if isinstance(faker_provider, str):
                                cleaned_provider = str(faker_provider).strip()
                                cleaned_provider = re.sub(r'^[\'"]|[\'"]$', '', cleaned_provider)
                                cleaned_provider = cleaned_provider.replace('Faker.', '').replace('Faker::Date.', '').replace('Faker::Regex.', '')
                                provider_name_map = {
                                     'regex': 'regexify', 'date': 'date_object', 'date_between': 'date_object', 'bban': 'bban',
                                     'random_float': 'pyfloat'
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
                                     valid_args = inspect.signature(faker_method).parameters
                                     filtered_args = {k: v for k, v in faker_args.items() if k in valid_args}

                                     while new_value is None or (new_value in existing_pk_values and retry_count < max_retries):
                                         current_faker_args = filtered_args.copy()
                                         if cleaned_provider == 'random_int':
                                              current_max_id = None
                                              if existing_pk_values:
                                                   try:
                                                        current_max_id = max(v for v in existing_pk_values if isinstance(v, (int, float)))
                                                   except ValueError:
                                                        pass

                                              if current_max_id is not None:
                                                   current_faker_args['min'] = int(current_max_id) + 1
                                                   if 'max' not in current_faker_args or current_faker_args['max'] <= current_faker_args['min']:
                                                        current_faker_args['max'] = current_faker_args['min'] + 100000

                                         new_value = faker_method(**current_faker_args)
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

                    if new_value is None or new_value in existing_pk_values:
                        if pd.api.types.is_numeric_dtype(df[primary_key]):
                             current_max_id = None
                             try:
                                  current_max_id = df[primary_key].max()
                                  if pd.isna(current_max_id): current_max_id = 0
                             except Exception:
                                  current_max_id = 0

                             new_value_seq = int(current_max_id) + 1
                             while new_value_seq in existing_pk_values:
                                  new_value_seq += 1
                             new_value = new_value_seq

                        else:
                             new_value_placeholder = f"GENERATED_DUP_{index}_{random.randint(10000, 99999)}"
                             while new_value_placeholder in existing_pk_values:
                                  new_value_placeholder = f"GENERATED_DUP_{index}_{random.randint(10000, 99999)}"
                             new_value = new_value_placeholder

                        print(f"      - Using fallback sequential/placeholder value '{new_value}' for duplicate at index {index}.")


                    df.loc[index, primary_key] = new_value
                    existing_pk_values.add(new_value)


                print(f"    - Attempted to regenerate {duplicate_count} duplicate primary key values.")

        except Exception as e:
            print(f"    - Error enforcing uniqueness on primary key '{primary_key}': {e}")
    elif not primary_key:
         print("    - Skipping uniqueness enforcement: No primary key defined in schema.")
    else:
         print(f"    - Skipping uniqueness enforcement: Primary key column '{primary_key}' not found in synthetic data.")

    print("\n--- State after Uniqueness Enforcement ---")
    print(df.head())
    print(df.nunique())


    print("\nPost-processing constraints applied.")
    return df


def detect_column_correlations_for_reporting(df):
    """
    Detects column correlations and relationships for reporting purposes.
    Includes functional dependencies, basic value relationships, and Pearson correlation.
    Uses global thresholds.
    Limits value relationships to numerical columns.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        dict: A dictionary where keys are column names and values are lists of detected relationships.
    """
    correlations = {}
    global CORRELATION_THRESHOLD, RELATIONSHIP_CONSISTENCY_THRESHOLD, FORMULA_TOLERANCE

    try:
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
                             if i < j:
                                 corr_value = corr_matrix.loc[col1, col2]
                                 if abs(corr_value) >= CORRELATION_THRESHOLD and not pd.isna(corr_value):
                                     if col1 not in correlations:
                                         correlations[col1] = []
                                     correlations[col1].append({
                                         "column": col2,
                                         "correlation": float(round(corr_value, 3)),
                                         "type": "pearson_correlation"
                                     })
                                     if col2 not in correlations:
                                         correlations[col2] = []
                                     correlations[col2].append({
                                         "column": col1,
                                         "correlation": float(round(corr_value, 3)),
                                         "type": "pearson_correlation"
                                     })
            except Exception as pearson_e:
                 print(f"Error during Pearson correlation calculation for reporting: {pearson_e}")


        # --- Functional Dependency (Simple Check) ---
        all_cols = df.columns.tolist()
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
                                            if col1 not in correlations:
                                                correlations[col1] = []
                                            if not any(d.get("column") == col2 and d.get("type") == "functional_dependency" for d in correlations[col1]):
                                                correlations[col1].append({
                                                    "column": col2,
                                                    "type": "functional_dependency"
                                                })
                    except Exception as dep_e:
                        pass


        # --- Basic Value Relationships (<=, >=) - LIMITED TO NUMERIC ---
        if len(numeric_cols) >= 2:
             for col1 in numeric_cols:
                 for col2 in numeric_cols:
                     if col1 != col2:
                         try:
                             df_filtered = df[[col1, col2]].dropna()
                             if not df_filtered.empty:
                                 num_rows_checked = len(df_filtered)

                                 if num_rows_checked > 0:
                                      series1 = pd.to_numeric(df_filtered[col1], errors='coerce')
                                      series2 = pd.to_numeric(df_filtered[col2], errors='coerce')

                                      if (series1 <= series2).sum() / num_rows_checked >= RELATIONSHIP_CONSISTENCY_THRESHOLD:
                                          if col1 not in correlations:
                                              correlations[col1] = []
                                          if not any(d.get("column") == col2 and d.get("relationship") == "less_than_or_equal_to" for d in correlations[col1]):
                                              correlations[col1].append({
                                                  "column": col2,
                                                  "relationship": "less_than_or_equal_to",
                                                  "type": "value_relationship"
                                              })

                                      if (series1 >= series2).sum() / num_rows_checked >= RELATIONSHIP_CONSISTENCY_THRESHOLD:
                                          if col1 not in correlations:
                                              correlations[col1] = []
                                          if not any(d.get("column") == col2 and d.get("relationship") == "greater_than_or_equal_to" for d in correlations[col1]):
                                              correlations[col1].append({
                                                  "column": col2,
                                                  "relationship": "greater_than_or_equal_to",
                                                  "type": "value_relationship"
                                              })

                         except Exception as rel_e:
                             pass

        # --- Simple Formula Relationships (A = B + C) for Reporting ---
        if len(numeric_cols) >= 3:
             for col_a, col_b, col_c in combinations(numeric_cols, 3):
                  if all(col in df.columns and pd.api.types.is_numeric_dtype(df[col]) for col in [col_a, col_b, col_c]):
                       try:
                            df_filtered = df[[col_a, col_b, col_c]].dropna()
                            if not df_filtered.empty:
                                series_a = pd.to_numeric(df_filtered[col_a], errors='coerce')
                                series_b = pd.to_numeric(df_filtered[col_b], errors='coerce')
                                series_c = pd.to_numeric(df_filtered[col_c], errors='coerce')

                                if np.allclose(series_a, series_b + series_c, atol=FORMULA_TOLERANCE, equal_nan=True):
                                    formula = f"{col_b} + {col_c}"
                                    rel_details = {"target_column": col_a, "source_columns": sorted([col_b, col_c]), "formula": formula, "type": "formula"}
                                    if col_a not in correlations: correlations[col_a] = []
                                    if rel_details not in correlations[col_a]: correlations[col_a].append(rel_details)

                                if np.allclose(series_b, series_a + series_c, atol=FORMULA_TOLERANCE, equal_nan=True):
                                    formula = f"{col_a} + {col_c}"
                                    rel_details = {"target_column": col_b, "source_columns": sorted([col_a, col_c]), "formula": formula, "type": "formula"}
                                    if col_b not in correlations: correlations[col_b] = []
                                    if rel_details not in correlations[col_b]: correlations[col_b].append(rel_details)

                                if np.allclose(series_c, series_a + series_b, atol=FORMULA_TOLERANCE, equal_nan=True):
                                    formula = f"{col_a} + {col_b}"
                                    rel_details = {"target_column": col_c, "source_columns": sorted([col_a, col_b]), "formula": formula, "type": "formula"}
                                    if col_c not in correlations: correlations[col_c] = []
                                    if rel_details not in correlations[col_c]: correlations[col_c].append(rel_details)

                       except Exception as formula_e:
                            pass


    except Exception as e:
        print(f"Error detecting correlations for reporting: {str(e)}")
        import traceback
        traceback.print_exc()

    correlations = {col: rels for col, rels in correlations.items() if rels}

    return correlations


def generate_synthetic_data_with_sdv(df, schema, num_rows, output_file):
    """Generates synthetic data, applies post-processing constraints, Faker, etc."""
    try:
        print("\n--- Starting Synthetic Data Generation ---")

        print("\n1. Initializing synthetic DataFrame with correct columns...")
        synthetic_data = pd.DataFrame(columns=schema.get("columns", {}).keys())
        synthetic_data = synthetic_data.reindex(range(num_rows))
        print(f"   Initialized empty synthetic DataFrame with {len(synthetic_data)} rows and {len(synthetic_data.columns)} columns.")

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

        print("\n2. Applying post-processing (Initial Faker/Regex generation, categorical distributions, custom rules, constraints)...")
        synthetic_data = apply_post_processing_rules(synthetic_data, schema, original_df_for_postprocessing)

        if original_df_for_postprocessing is not None:
             del original_df_for_postprocessing

        print("\n3. Comparing relationships between original and synthetic data and generating report...")
        original_df_for_corr = None
        try:
             original_df_for_corr = pd.read_csv(INPUT_CSV, encoding='utf-8')
             try:
                 original_df_for_corr = pd.read_csv(INPUT_CSV, encoding='utf-8')
             except UnicodeDecodeError:
                 print("UTF-8 decoding failed for original data reload for correlation report, trying latin-1...")
                 original_df_for_corr = pd.read_csv(INPUT_CSV, encoding='latin-1')
             print("   Loaded original data for relationship reporting.")

             original_relationships_detected = detect_column_correlations_for_reporting(original_df_for_corr)
             synthetic_relationships_detected = detect_column_correlations_for_reporting(synthetic_data)

             correlation_report_data = []
             reported_relationships_keys = set()

             def get_relationship_key(rel_info):
                  rel_type = rel_info.get("type")
                  if rel_type == "pearson_correlation":
                       cols = tuple(sorted((rel_info.get("column"), rel_info.get("source_column"))))
                       return (cols, rel_type)
                  elif rel_type in ["functional_dependency", "one_to_one", "value_relationship"]:
                       source_col = rel_info.get("source_column") or rel_info.get("column")
                       target_col = rel_info.get("target_column") or rel_info.get("column")
                       return (source_col, target_col, rel_type, rel_info.get("relationship"))
                  elif rel_type == "formula":
                       return (rel_info.get("target_column"), tuple(sorted(rel_info.get("source_columns", []))), rel_type, rel_info.get("formula"))
                  return None

             all_relationships_combined = defaultdict(lambda: {"Original": None, "Synthetic": None})

             for col, rels in original_relationships_detected.items():
                 for rel in rels:
                     key = get_relationship_key({**rel, "source_column": col})
                     if key:
                          all_relationships_combined[key]["Original"] = rel

             for col, rels in synthetic_relationships_detected.items():
                  for rel in rels:
                      key = get_relationship_key({**rel, "source_column": col})
                      if key:
                           all_relationships_combined[key]["Synthetic"] = rel

             for key, data in all_relationships_combined.items():
                  original_rel = data["Original"]
                  synthetic_rel = data["Synthetic"]

                  if original_rel is None and synthetic_rel is None:
                       continue

                  rel_type = key[2] if len(key) > 2 else None
                  col1 = key[0]
                  col2 = key[1] if len(key) > 1 else None
                  relationship_detail = key[3] if len(key) > 3 else ""

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

                  report_this = False
                  if most_specific_type in ["one_to_one", "functional_dependency", "formula"]:
                       report_this = True
                  elif most_specific_type == "value_relationship":
                       pair_has_more_specific = False
                       cols_involved = {col1}
                       if col2: cols_involved.add(col2)
                       if rel_type == "formula":
                            cols_involved.add(key[0])
                            cols_involved.update(key[1])

                       for other_key, other_data in all_relationships_combined.items():
                            other_rel_type = other_key[2] if len(other_key) > 2 else None
                            other_cols_involved = {other_key[0]}
                            if len(other_key) > 1: other_cols_involved.add(other_key[1])
                            if other_rel_type == "formula":
                                other_cols_involved.add(other_key[0])
                                other_cols_involved.update(other_key[1])

                            if rel_type != "formula" and other_rel_type != "formula":
                                 pair_key_current = tuple(sorted((col1, col2))) if col2 else (col1,)
                                 pair_key_other = tuple(sorted((other_key[0], other_key[1]))) if len(other_key) > 1 else (other_key[0],)
                                 if pair_key_current == pair_key_other and other_rel_type in ["one_to_one", "functional_dependency", "formula"]:
                                      pair_has_more_specific = True
                                      break
                            elif rel_type == "formula" or other_rel_type == "formula":
                                 if (rel_type == "formula" and other_rel_type == "formula" and key[0] == other_key[0]) or \
                                    (cols_involved == other_cols_involved and other_rel_type in ["one_to_one", "functional_dependency", "formula"]):
                                      pair_has_more_specific = True
                                      break

                       if not pair_has_more_specific:
                            report_this = True
                  elif most_specific_type == "pearson_correlation":
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

                       report_entry_key_for_uniqueness = (report_entry["Column 1"], report_entry["Column 2"], report_entry["Type"], report_entry["Relationship Detail"])
                       if report_entry_key_for_uniqueness not in reported_relationships_keys:
                            correlation_report_data.append(report_entry)
                            reported_relationships_keys.add(report_entry_key_for_uniqueness)

             if correlation_report_data:
                 correlation_df = pd.DataFrame(correlation_report_data)
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

    print(f"\nLoading original data from: {INPUT_CSV}")
    original_df = None
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

    if original_df is not None:
         del original_df

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
