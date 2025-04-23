import pandas as pd
import json
import os
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta, date
import re
from itertools import combinations
import inspect
import string
import warnings

# Configuration
INPUT_CSV = "customer_data.csv"
SCHEMA_JSON = "enhanced_schema.json"
OUTPUT_SYNTHETIC_CSV = "synthetic_customer_data_faker.csv"
OUTPUT_RELATIONSHIP_COMPARISON_CSV = "relationship_comparison.csv"
NUM_SYNTHETIC_ROWS = 10000
FORMULA_TOLERANCE = 1e-6
INEQUALITY_CONSISTENCY_THRESHOLD = 0.999
RELATIONSHIP_DETECTION_THRESHOLD = 0.999
CORRELATION_REPORTING_THRESHOLD = 0.7
REGEX_MAX_ATTEMPTS = 10 # Limit attempts for custom regex generation

# Initialize Faker
fake = Faker(['vi_VN'])
Faker.seed(42)

def load_data(csv_file_path):
    """Loads data from a CSV file."""
    if not os.path.exists(csv_file_path):
        print(f"Error: Input data file not found: {csv_file_path}")
        return None
    try:
        df = pd.read_csv(csv_file_path)
        print(f"Successfully loaded data from {csv_file_path}. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data from {csv_file_path}: {e}")
        return None

def load_schema(json_file_path):
    """Loads the enhanced schema from a JSON file."""
    if not os.path.exists(json_file_path):
        print(f"Error: Schema file not found: {json_file_path}")
        return None
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)
        print(f"Successfully loaded schema from {json_file_path}.")
        return schema
    except Exception as e:
        print(f"Error loading schema from {json_file_path}: {e}")
        return None

# --- Enhanced Custom Regex Generation Functions ---
def parse_regex_pattern(pattern):
    """
    Parses a simplified regex pattern into a list of components (literals, sets, groups, etc.).
    Supports basic: ., \d, \w, \s, [], {}, ?, *, +, \, |, (), (?:), ^, $.
    Returns a list of tokens or None if parsing fails.
    """
    tokens = []
    i = 0
    while i < len(pattern):
        char = pattern[i]

        if char == '\\':
            if i + 1 < len(pattern):
                tokens.append(('ESCAPED', pattern[i+1]))
                i += 2
            else:
                print(f"Warning: Trailing '\\' in pattern '{pattern}'. Ignoring.")
                i += 1
        elif char == '.':
            tokens.append(('ANY',))
            i += 1
        elif char == '[':
            end_bracket = pattern.find(']', i + 1)
            if end_bracket == -1:
                print(f"Warning: Unclosed '[' in pattern '{pattern}'. Parsing failed.")
                return None
            char_set_content = pattern[i + 1:end_bracket]
            tokens.append(('SET', char_set_content))
            i = end_bracket + 1
        elif char == '{':
            end_brace = pattern.find('}', i + 1)
            if end_brace == -1:
                print(f"Warning: Unclosed '{{' in pattern '{pattern}'. Parsing failed.")
                return None
            count_str = pattern[i + 1:end_brace]
            tokens.append(('REPETITION', count_str))
            i += len(count_str) + 2 # Consume {count}

        elif char in '?*+':
            if not tokens:
                 print(f"Warning: Quantifier '{char}' at start of pattern '{pattern}'. Parsing failed.")
                 return None
            last_token = tokens[-1]
            last_token_type = last_token[0]

            if last_token_type in ('LITERAL', 'ESCAPED', 'ANY', 'SET', 'GROUP'):
                 if len(last_token) > 2 and (last_token[2] in '?*+' or last_token[0] == 'REPETITION'):
                     print(f"Warning: Quantifier '{char}' applied after already quantified/repeated token '{last_token}'. Parsing failed.")
                     return None

                 tokens.pop()
                 tokens.append((last_token_type, last_token[1] if len(last_token) > 1 else None, char))
            else:
                 print(f"Warning: Quantifier '{char}' after unsupported token type '{last_token_type}' in pattern '{pattern}'. Parsing failed.")
                 return None

            i += 1
        elif char == '(':
            if i + 2 < len(pattern) and pattern[i+1] == '?' and pattern[i+2] == ':':
                 group_start_index = i + 3
                 group_type = 'NON_CAPTURING_GROUP'
            else:
                 group_start_index = i + 1
                 group_type = 'CAPTURING_GROUP'

            open_count = 1
            end_paren = -1
            for j in range(group_start_index, len(pattern)):
                 if pattern[j] == '(':
                      open_count += 1
                 elif pattern[j] == ')':
                      open_count -= 1
                      if open_count == 0:
                           end_paren = j
                           break
            if end_paren == -1:
                 print(f"Warning: Unclosed '(' in pattern '{pattern}'. Parsing failed.")
                 return None

            group_content = pattern[group_start_index:end_paren]
            tokens.append(('GROUP', group_content))
            i = end_paren + 1
        elif char == '|':
             tokens.append(('ALTERNATION',))
             i += 1
        elif char == '^':
             tokens.append(('ANCHOR_START',))
             i += 1
        elif char == '$':
             tokens.append(('ANCHOR_END',))
             i += 1
        else:
            tokens.append(('LITERAL', char))
            i += 1
    return tokens

def generate_from_parsed_tokens(tokens):
    """
    Generates a string from a list of parsed regex tokens.
    Handles simple alternation by picking one path.
    Correctly handles repetitions by generating the preceding token's content
    multiple times independently.
    """
    if not tokens:
        return ""

    result = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        token_type = token[0]
        quantifier = token[2] if len(token) > 2 else None

        # Check if the NEXT token is a REPETITION
        num_repetitions = 1 # Default to 1 repetition
        if i + 1 < len(tokens) and tokens[i+1][0] == 'REPETITION':
            rep_token = tokens[i+1]
            count_str = rep_token[1]
            try:
                if ',' in count_str:
                    min_max = count_str.split(',')
                    min_rep = int(min_max[0].strip()) if min_max[0].strip() else 0
                    max_rep = int(min_max[1].strip()) if min_max[1].strip() else min_rep
                    num_repetitions = random.randint(min_rep, max_rep)
                else:
                    num_repetitions = int(count_str.strip())
            except ValueError:
                num_repetitions = 1 # Fallback
            i += 1 # Consume the REPETITION token in the next iteration

        # Apply quantifier if present (overrides explicit repetition token if both exist, though regex rules usually prevent this)
        if quantifier == '?': num_repetitions = random.choice([0, 1])
        elif quantifier == '*': num_repetitions = random.randint(0, 5) # Limited repetition
        elif quantifier == '+': num_repetitions = random.randint(1, 5) # Limited repetition


        generated_part_base = "" # Part generated once before repetition

        if token_type == 'LITERAL':
            generated_part_base = token[1]
        elif token_type == 'ESCAPED':
            escaped_char = token[1]
            if escaped_char == 'd': generated_part_base = random.choice(string.digits)
            elif escaped_char == 'w': generated_part_base = random.choice(string.ascii_letters + string.digits + '_')
            elif escaped_char == 's': generated_part_base = random.choice(string.whitespace)
            else: generated_part_base = escaped_char
        elif token_type == 'ANY':
            generated_part_base = random.choice(string.printable.replace('\n', ''))
        elif token_type == 'SET':
            char_set_content = token[1]
            possible_chars = []
            j = 0
            while j < len(char_set_content):
                if j + 2 < len(char_set_content) and char_set_content[j+1] == '-':
                    start_char = char_set_content[j]
                    end_char = char_set_content[j+2]
                    if ord(start_char) <= ord(end_char):
                        possible_chars.extend([chr(k) for k in range(ord(start_char), ord(end_char) + 1)])
                    j += 3
                elif char_set_content[j] == '\\' and j + 1 < len(char_set_content):
                     possible_chars.append(char_set_content[j+1])
                     j += 2
                else:
                    possible_chars.append(char_set_content[j])
                    j += 1
            if possible_chars: generated_part_base = random.choice(possible_chars)
            else: generated_part_base = ""

        elif token_type == 'GROUP':
             group_content = token[1]
             alternation_options = re.split(r'(?<!\\)\|', group_content)

             if len(alternation_options) > 1:
                  chosen_option = random.choice(alternation_options).strip()
                  parsed_option_tokens = parse_regex_pattern(chosen_option)
                  if parsed_option_tokens is not None:
                       generated_part_base = generate_from_parsed_tokens(parsed_option_tokens)
                  else:
                       generated_part_base = ""
             else:
                  parsed_group_tokens = parse_regex_pattern(group_content)
                  if parsed_group_tokens is not None:
                       generated_part_base = generate_from_parsed_tokens(parsed_group_tokens)
                  else:
                       generated_part_base = ""

        elif token_type in ('ALTERNATION', 'ANCHOR_START', 'ANCHOR_END', 'NON_CAPTURING_GROUP', 'REPETITION'):
             # These tokens are structural or already consumed as repetition count
             generated_part_base = ""
             num_repetitions = 1 # Do not repeat these markers


        # Append the generated part, repeated num_repetitions times
        # For tokens that represent a single character/sequence type (\d, \w, ., [], literal),
        # we need to generate *independently* inside the repetition loop.
        if token_type in ('LITERAL', 'ESCAPED', 'ANY', 'SET'):
             for _ in range(num_repetitions):
                  # Regenerate the base part for each repetition
                  if token_type == 'LITERAL': part = token[1]
                  elif token_type == 'ESCAPED':
                       escaped_char = token[1]
                       if escaped_char == 'd': part = random.choice(string.digits)
                       elif escaped_char == 'w': part = random.choice(string.ascii_letters + string.digits + '_')
                       elif escaped_char == 's': part = random.choice(string.whitespace)
                       else: part = escaped_char
                  elif token_type == 'ANY': part = random.choice(string.printable.replace('\n', ''))
                  elif token_type == 'SET':
                       char_set_content = token[1]
                       possible_chars = []
                       j = 0
                       while j < len(char_set_content):
                           if j + 2 < len(char_set_content) and char_set_content[j+1] == '-':
                               start_char = char_set_content[j]
                               end_char = char_set_content[j+2]
                               if ord(start_char) <= ord(end_char):
                                   possible_chars.extend([chr(k) for k in range(ord(start_char), ord(end_char) + 1)])
                               j += 3
                           elif char_set_content[j] == '\\' and j + 1 < len(char_set_content):
                                possible_chars.append(char_set_content[j+1])
                                j += 2
                           else:
                               possible_chars.append(char_set_content[j])
                               j += 1
                       if possible_chars: part = random.choice(possible_chars)
                       else: part = ""
                  else: part = "" # Should not happen

                  result.append(part)
        else:
             # For GROUPs or structural tokens, just repeat the generated_part_base string
             result.append(generated_part_base * num_repetitions)


        i += 1

    return "".join(result)


def generate_from_regex_pattern(pattern, max_attempts=REGEX_MAX_ATTEMPTS):
    """
    Generates a single string matching a simplified regex pattern.
    Uses parsing and re.fullmatch for robust validation.
    """
    if not isinstance(pattern, str) or not pattern:
        return None

    parsed_tokens = parse_regex_pattern(pattern)
    if parsed_tokens is None:
         print(f"Warning: Failed to parse regex pattern '{pattern}'. Cannot generate.")
         return None

    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        generated_string = generate_from_parsed_tokens(parsed_tokens)

        try:
            if re.fullmatch(pattern, generated_string):
                return generated_string
        except Exception as e:
            print(f"Warning: Error during regex fullmatch validation for pattern '{pattern}' and generated '{generated_string}': {e}. Returning generated string without validation.")
            return generated_string

    print(f"Warning: Could not generate a string matching pattern '{pattern}' after {max_attempts} attempts.")
    return None


def generate_initial_data_with_faker(schema, num_rows, original_df=None):
    """
    Generates initial synthetic data using Faker based on schema and original data stats.
    Uses custom regex generation for 'regexify' provider.
    Reads column info from the 'columns' key in the schema.
    """
    print(f"\nGenerating {num_rows} initial rows using Faker and custom methods...")
    synthetic_data = {}
    # Access column info from the 'columns' key
    column_infos = schema.get("columns", {})
    columns = list(column_infos.keys())

    for col_name in columns:
        col_info = column_infos.get(col_name, {})
        faker_provider_name = col_info.get("faker_provider")
        faker_args = col_info.get("faker_args", {})
        data_type = col_info.get("data_type")
        stats = col_info.get("stats", {})
        null_percentage = col_info.get("null_percentage", 0)

        generated_values = []
        use_faker = False
        use_custom_regex = False

        regex_pattern = faker_args.get("pattern") or faker_args.get("text")
        if faker_provider_name == "regexify" and isinstance(regex_pattern, str):
             print(f"  Generating data for column '{col_name}' using custom regex generation with pattern '{regex_pattern}'...")
             for _ in range(num_rows):
                  generated_values.append(generate_from_regex_pattern(regex_pattern))
             use_custom_regex = True
             use_faker = True

        elif faker_provider_name:
            faker_method = getattr(fake, faker_provider_name, None)
            if faker_method:
                print(f"  Generating data for column '{col_name}' using Faker provider '{faker_provider_name}'...")
                try:
                     sig = inspect.signature(faker_method)
                     valid_args = {arg: faker_args[arg] for arg in faker_args if arg in sig.parameters}
                     if 'locale' in sig.parameters and 'locale' not in valid_args:
                          valid_args['locale'] = fake.locale

                     for _ in range(num_rows):
                         try:
                             value = faker_method(**valid_args)
                             generated_values.append(value)
                         except Exception as e:
                             if original_df is not None and col_name in original_df.columns and not original_df[col_name].empty:
                                 original_values = original_df[col_name].dropna().tolist()
                                 if original_values:
                                      sample_value = random.choice(original_values)
                                      generated_values.append(sample_value)
                                 else:
                                      if data_type in ["integer", "float", "numerical"]: generated_values.append(0)
                                      elif data_type == "datetime": generated_values.append(datetime.now())
                                      else: generated_values.append("FAKER_GEN_ERROR")
                             else:
                                 if data_type in ["integer", "float", "numerical"]: generated_values.append(0)
                                 elif data_type == "datetime": generated_values.append(datetime.now())
                                 else: generated_values.append("PLACEHOLDER")

                     use_faker = True

                except Exception as e:
                     print(f"  Warning: Could not call Faker provider '{faker_provider_name}' for '{col_name}' with args {faker_args}: {e}. Falling back to sampling/placeholders.")
                     use_faker = False

        if not use_faker and original_df is not None and col_name in original_df.columns and not original_df[col_name].empty:
            print(f"  No Faker provider specified or failed for '{col_name}'. Sampling from original data distribution.")
            original_values = original_df[col_name].dropna().tolist()
            if original_values:
                 if data_type == "categorical" and "categories" in stats:
                      categories_info = stats["categories"]
                      values = [cat['value'] for cat in categories_info]
                      weights = [cat['percentage'] for cat in categories_info]
                      total_weight = sum(weights)
                      if total_weight > 0:
                           normalized_weights = [w / total_weight for w in weights]
                           generated_values = random.choices(values, weights=normalized_weights, k=num_rows)
                      else:
                           generated_values = random.choices(original_values, k=num_rows)
                 else:
                      generated_values = random.choices(original_values, k=num_rows)
            else:
                 for _ in range(num_rows):
                     if data_type in ["integer", "float", "numerical"]: generated_values.append(0)
                     elif data_type == "datetime": generated_values.append(datetime.now())
                     else: generated_values.append("SAMPLING_ERROR")

        elif not use_faker and not use_custom_regex:
            for _ in range(num_rows):
                if data_type in ["integer", "float", "numerical"]: generated_values.append(0)
                elif data_type == "datetime": generated_values.append(datetime.now())
                else: generated_values.append("PLACEHOLDER")

        num_nulls_to_add = int(num_rows * (null_percentage / 100))
        num_nulls_to_add = min(num_nulls_to_add, num_rows)

        current_non_nan_indices = [i for i, x in enumerate(generated_values) if pd.notna(x)]
        num_possible_nulls = len(current_non_nan_indices)

        if num_nulls_to_add > 0 and num_possible_nulls > 0:
             indices_to_null = random.sample(current_non_nan_indices, min(num_nulls_to_add, num_possible_nulls))
             for idx in indices_to_null:
                 generated_values[idx] = np.nan
        elif num_nulls_to_add > 0 and num_possible_nulls == 0:
             pass

        synthetic_data[col_name] = generated_values

    synthetic_df = pd.DataFrame(synthetic_data, columns=columns)
    print("Initial data generation complete.")
    return synthetic_df

def enforce_primary_key_uniqueness(synthetic_df, schema, original_df):
    """
    Enforces uniqueness for the primary key column.
    Regenerates duplicate values using the column's Faker provider (including custom regex) or a fallback.
    Reads column info from the 'columns' key in the schema.
    """
    print("\nEnforcing Primary Key uniqueness...")
    pk_column = None
    pk_col_info = None

    # Access column info from the 'columns' key
    column_infos = schema.get("columns", {})

    # Iterate over column info to find the primary key
    for col_name, col_info in column_infos.items():
        if col_info.get("key_type") == "Primary Key":
            pk_column = col_name
            pk_col_info = col_info
            break

    if pk_column is None or pk_column not in synthetic_df.columns:
        print("  No Primary Key defined or found in synthetic data. Skipping uniqueness enforcement.")
        return synthetic_df

    print(f"  Enforcing uniqueness for Primary Key column: '{pk_column}'")

    duplicate_mask = synthetic_df[pk_column].duplicated(keep='first')
    duplicate_indices = synthetic_df.index[duplicate_mask].tolist()

    if not duplicate_indices:
        print(f"  No duplicate values found in Primary Key column '{pk_column}'.")
        return synthetic_df

    print(f"  Found {len(duplicate_indices)} duplicate values in '{pk_column}'. Regenerating...")

    faker_provider_name = pk_col_info.get("faker_provider")
    faker_args = pk_col_info.get("faker_args", {})
    data_type = pk_col_info.get("data_type")

    existing_unique_values = set(original_df[pk_column].dropna().tolist() if original_df is not None and pk_column in original_df.columns else [])
    existing_unique_values.update(synthetic_df[pk_column].dropna().unique().tolist())

    regenerated_count = 0
    max_attempts = 1000 # Keep higher attempts for PK uniqueness as it's critical

    for idx in duplicate_indices:
        attempt = 0
        new_value = None

        while attempt < max_attempts:
            attempt += 1
            try:
                regex_pattern = faker_args.get("pattern") or faker_args.get("text")
                if faker_provider_name == "regexify" and isinstance(regex_pattern, str):
                     new_value = generate_from_regex_pattern(regex_pattern) # Uses REGEX_MAX_ATTEMPTS
                     if new_value == "REGEX_GEN_ERROR": new_value = None

                elif faker_provider_name:
                    faker_method = getattr(fake, faker_provider_name, None)
                    if faker_method:
                         sig = inspect.signature(faker_method)
                         valid_args = {arg: faker_args[arg] for arg in faker_args if arg in sig.parameters}
                         if 'locale' in sig.parameters and 'locale' not in valid_args:
                              valid_args['locale'] = fake.locale
                         new_value = faker_method(**valid_args)

                if new_value is None or pd.isna(new_value):
                     if original_df is not None and pk_column in original_df.columns and not original_df[pk_column].empty:
                          original_unique_pk_values = original_df[pk_column].dropna().unique().tolist()
                          if original_unique_pk_values:
                               new_value = random.choice(original_unique_pk_values)
                          else:
                               if data_type in ["integer", "float", "numerical"]:
                                    current_max_idx = synthetic_df.index.max() if not synthetic_df.empty else -1
                                    new_value = int(current_max_idx + attempt)
                               elif data_type == "datetime":
                                    new_value = datetime.now() + timedelta(seconds=attempt)
                               else:
                                    current_max_idx = synthetic_df.index.max() if not synthetic_df.empty else -1
                                    new_value = f"GENERATED_{current_max_idx + attempt}"

                     else:
                          if data_type in ["integer", "float", "numerical"]:
                               current_max_idx = synthetic_df.index.max() if not synthetic_df.empty else -1
                               new_value = int(current_max_idx + attempt)
                          elif data_type == "datetime":
                               new_value = datetime.now() + timedelta(seconds=attempt)
                          else:
                               current_max_idx = synthetic_df.index.max() if not synthetic_df.empty else -1
                               new_value = f"GENERATED_{current_max_idx + attempt}"

                if pd.notna(new_value) and new_value not in existing_unique_values:
                    synthetic_df.loc[idx, pk_column] = new_value
                    existing_unique_values.add(new_value)
                    regenerated_count += 1
                    break

            except Exception as e:
                print(f"    Error regenerating value for index {idx} in '{pk_column}': {e}. Attempt {attempt}. Using placeholder.")
                synthetic_df.loc[idx, pk_column] = f"ERROR_PK_GEN_{idx}"
                break

        if attempt == max_attempts:
            print(f"    Warning: Could not generate a unique value for index {idx} in '{pk_column}' after {max_attempts} attempts. Value remains a duplicate or placeholder.")

    print(f"  Finished Primary Key uniqueness enforcement. Regenerated {regenerated_count} values.")
    return synthetic_df

def apply_one_to_one_relationships(synthetic_df, original_df, relationships, schema):
    """
    Applies one-to-one relationships as a post-processing step.
    Ensures strict uniqueness for both source and target columns in the pair.
    Uses column's Faker provider (including custom regex) for regeneration if needed.
    Reads column info from the 'columns' key in the schema.
    Reads relationships from the top-level 'relationships' key.
    """
    print("\nApplying one-to-one relationships...")

    # Access column info from the 'columns' key
    column_infos = schema.get("columns", {})
    # Access relationships from the top-level 'relationships' key
    relationships_data = relationships.get("relationships", {})

    one_to_one_relationships_to_apply = {}
    for col, rel_list in relationships_data.items():
         oto_rels = [rel for rel in rel_list if rel.get("type") == "one_to_one"]
         if oto_rels:
              one_to_one_relationships_to_apply[col] = oto_rels


    if not one_to_one_relationships_to_apply:
         print("  No one-to-one relationships found in schema to apply.")
         return synthetic_df


    original_unique_values = {}
    for col in original_df.columns:
         original_unique_values[col] = set(original_df[col].dropna().unique().tolist())

    for col1, rel_list in one_to_one_relationships_to_apply.items():
        for rel in rel_list:
            col2 = rel.get("column")

            if col1 in synthetic_df.columns and col2 in synthetic_df.columns and \
               col1 in original_df.columns and col2 in original_df.columns:

                print(f"  Enforcing one-to-one relationship: {col1} <-> {col2}")

                try:
                    original_mapping_forward = original_df.dropna(subset=[col1, col2]).drop_duplicates(subset=[col1], keep='first').set_index(col1)[col2].to_dict()
                    original_mapping_backward = original_df.dropna(subset=[col1, col2]).drop_duplicates(subset=[col2], keep='first').set_index(col2)[col1].to_dict()

                    used_synthetic_col2_values = set(synthetic_df[col2].dropna().unique().tolist())
                    used_synthetic_col2_values.update(original_mapping_forward.values())

                    regenerated_count = 0
                    max_attempts_per_row = 100 # Keep higher attempts for O2O as it's critical

                    for idx in synthetic_df.index:
                        synth_col1_value = synthetic_df.loc[idx, col1]
                        synth_col2_value = synthetic_df.loc[idx, col2]

                        if pd.isna(synth_col1_value) or pd.isna(synth_col2_value):
                            continue

                        is_valid_pair = False

                        if synth_col1_value in original_mapping_forward:
                            expected_col2_value = original_mapping_forward[synth_col1_value]
                            if synth_col2_value == expected_col2_value:
                                is_valid_pair = True
                            else:
                                new_col2_value = expected_col2_value
                                is_new_col2_already_used = new_col2_value in used_synthetic_col2_values and not (new_col2_value == synth_col2_value and list(synthetic_df[col2][synthetic_df[col2] == new_col2_value].index).count(idx) == 1)

                                if not is_new_col2_already_used:
                                     synthetic_df.loc[idx, col2] = new_col2_value
                                     used_synthetic_col2_values.add(new_col2_value)
                                     regenerated_count += 1
                                     is_valid_pair = True
                                else:
                                     is_valid_pair = False

                        elif synth_col1_value not in original_unique_values.get(col1, set()):
                             is_col2_unique_in_synth = list(synthetic_df[col2][synthetic_df[col2] == synth_col2_value].index).count(idx) == 1
                             is_col2_originally_mapped_to_different_col1 = synth_col2_value in original_mapping_backward and original_mapping_backward[synth_col2_value] != synth_col1_value

                             if is_col2_unique_in_synth and not is_col2_originally_mapped_to_different_col1:
                                  is_valid_pair = True
                                  used_synthetic_col2_values.add(synth_col2_value)
                             else:
                                  is_valid_pair = False

                        if not is_valid_pair:
                            attempt = 0
                            regenerated_pair_found = False
                            while attempt < max_attempts_per_row:
                                attempt += 1
                                new_col1_value = None
                                new_col2_value = None

                                col1_schema_info = column_infos.get(col1, {}) # Access from column_infos
                                if col1_schema_info:
                                     col1_faker_provider_name = col1_schema_info.get("faker_provider")
                                     col1_faker_args = col1_schema_info.get("faker_args", {})
                                     col1_data_type = col1_schema_info.get("data_type")

                                     regex_pattern_col1 = col1_faker_args.get("pattern") or col1_faker_args.get("text")
                                     if col1_faker_provider_name == "regexify" and isinstance(regex_pattern_col1, str):
                                          new_col1_value = generate_from_regex_pattern(regex_pattern_col1) # Uses REGEX_MAX_ATTEMPTS
                                          if new_col1_value == "REGEX_GEN_ERROR": new_col1_value = None

                                     elif col1_faker_provider_name:
                                          faker_method = getattr(fake, col1_faker_provider_name, None)
                                          if faker_method:
                                               try:
                                                    sig = inspect.signature(faker_method)
                                                    valid_args = {arg: col1_faker_args[arg] for arg in col1_faker_args if arg in sig.parameters}
                                                    if 'locale' in sig.parameters and 'locale' not in valid_args:
                                                         valid_args['locale'] = fake.locale
                                                    new_col1_value = faker_method(**valid_args)
                                               except Exception:
                                                    pass

                                if new_col1_value is None or pd.isna(new_col1_value):
                                     if original_df is not None and col1 in original_df.columns and not original_df[col1].empty:
                                          original_unique_col1_values = original_df[col1].dropna().unique().tolist()
                                          if original_unique_col1_values:
                                               new_col1_value = random.choice(original_unique_col1_values)
                                          else:
                                               new_col1_value = f"GEN_C1_{idx}_{attempt}"
                                     else:
                                          new_col1_value = f"GEN_C1_{idx}_{attempt}"

                                if pd.notna(new_col1_value):
                                     if new_col1_value in original_mapping_forward:
                                          new_col2_value = original_mapping_forward[new_col1_value]
                                     else:
                                          col2_schema_info = column_infos.get(col2, {}) # Access from column_infos
                                          if col2_schema_info:
                                               col2_faker_provider_name = col2_schema_info.get("faker_provider")
                                               col2_faker_args = col2_schema_info.get("faker_args", {})
                                               col2_data_type = col2_schema_info.get("data_type")

                                               regex_pattern_col2 = col2_faker_args.get("pattern") or col2_faker_args.get("text")
                                               if col2_faker_provider_name == "regexify" and isinstance(regex_pattern_col2, str):
                                                    new_col2_value = generate_from_regex_pattern(regex_pattern_col2) # Uses REGEX_MAX_ATTEMPTS
                                                    if new_col2_value == "REGEX_GEN_ERROR": new_col2_value = None

                                               elif col2_faker_provider_name:
                                                    faker_method = getattr(fake, col2_faker_provider_name, None)
                                                    if faker_method:
                                                         try:
                                                              sig = inspect.signature(faker_method)
                                                              valid_args = {arg: col2_faker_args[arg] for arg in col2_faker_args if arg in sig.parameters}
                                                              if 'locale' in sig.parameters and 'locale' not in valid_args:
                                                                   valid_args['locale'] = fake.locale
                                                              new_col2_value = faker_method(**valid_args)
                                                         except Exception:
                                                              pass

                                          if new_col2_value is None or pd.isna(new_col2_value):
                                               if col2_data_type in ["integer", "float", "numerical"]:
                                                    current_max_val = synthetic_df[col2].dropna().max() if not synthetic_df[col2].dropna().empty else 0
                                                    new_col2_value = int(current_max_val + idx + attempt)
                                               elif col2_data_type == "datetime":
                                                    new_col2_value = datetime.now() + timedelta(seconds=idx + attempt)
                                               else:
                                                    new_col2_value = f"GEN_C2_{idx}_{attempt}"

                                if pd.notna(new_col1_value) and pd.notna(new_col2_value) and \
                                   new_col2_value not in used_synthetic_col2_values:
                                     is_new_col2_originally_mapped_to_different_col1 = new_col2_value in original_mapping_backward and original_mapping_backward[new_col2_value] != new_col1_value

                                     if not is_new_col2_originally_mapped_to_different_col1:
                                          synthetic_df.loc[idx, col1] = new_col1_value
                                          synthetic_df.loc[idx, col2] = new_col2_value
                                          used_synthetic_col2_values.add(new_col2_value)
                                          regenerated_count += 1
                                          regenerated_pair_found = True
                                          break

                            if not regenerated_pair_found:
                                synthetic_df.loc[idx, col1] = f"O2O_ERR_{col1}_{idx}"
                                synthetic_df.loc[idx, col2] = f"O2O_ERR_{col2}_{idx}"

                except Exception as e:
                    print(f"    Error applying one-to-one for index {idx} in '{col1}' <-> '{col2}': {e}")
                    import traceback
                    traceback.print_exc()
                    synthetic_df.loc[idx, col1] = f"O2O_ERR_{col1}_{idx}"
                    synthetic_df.loc[idx, col2] = f"O2O_ERR_{col2}_{idx}"

            else:
                print(f"  Skipping one-to-one relationship {col1} <-> {col2}: Columns not found in synthetic or original data.")

    print("Finished applying one-to-one relationships.")
    return synthetic_df

def apply_functional_dependencies(synthetic_df, original_df, relationships, schema):
    """
    Applies functional dependencies (A -> B) as a post-processing step.
    Maps original A values to B. For new synthetic A values, samples from original B values.
    Assumes one-to-one dependencies are handled separately with stricter logic.
    Reads column info from the 'columns' key in the schema.
    Reads relationships from the top-level 'relationships' key.
    """
    print("\nApplying functional dependencies (non-one-to-one)...")

    # Access column info from the 'columns' key
    column_infos = schema.get("columns", {})
    # Access relationships from the top-level 'relationships' key
    relationships_data = relationships.get("relationships", {})

    functional_dependencies_to_apply = {}
    for col, rel_list in relationships_data.items():
         fd_rels = [rel for rel in rel_list if rel.get("type") == "functional_dependency"]
         if fd_rels:
              functional_dependencies_to_apply[col] = fd_rels


    if not functional_dependencies_to_apply:
         print("  No functional dependencies found in schema to apply.")
         return synthetic_df

    for source_col, rel_list in functional_dependencies_to_apply.items():
        for rel in rel_list:
            target_col = rel.get("column")

            # Ensure it's not a one_to_one relationship (handled separately)
            is_one_to_one = False
            if target_col in relationships_data:
                 if any(other_rel.get("type") == "one_to_one" and other_rel.get("column") == source_col for other_rel in relationships_data[target_col]):
                      is_one_to_one = True

            if not is_one_to_one and source_col in synthetic_df.columns and target_col in synthetic_df.columns and \
               source_col in original_df.columns and target_col in original_df.columns:

                print(f"  Enforcing functional dependency: {source_col} -> {target_col}")

                try:
                    original_mapping = original_df.dropna(subset=[source_col, target_col]).drop_duplicates(subset=[source_col], keep='first').set_index(source_col)[target_col].to_dict()
                    mapped_series = synthetic_df[source_col].map(original_mapping)
                    mapped_mask = mapped_series.notna()

                    if mapped_mask.any():
                         synthetic_df.loc[mapped_mask, target_col] = mapped_series[mapped_mask]
                         print(f"    Updated {mapped_mask.sum()} values in '{target_col}' based on '{source_col}' mapping.")
                    else:
                         print(f"    Warning: No values updated in '{target_col}': No successful mappings found for '{source_col}' values in synthetic data.")

                    unmapped_mask = mapped_series.isna() & synthetic_df[source_col].notna()

                    if unmapped_mask.any():
                         print(f"    Handling {unmapped_mask.sum()} synthetic '{source_col}' values not found in original data mapping.")
                         original_target_values = original_df[target_col].dropna().unique().tolist()

                         if original_target_values:
                             num_unmapped = unmapped_mask.sum()
                             sampled_values = random.choices(original_target_values, k=num_unmapped)
                             synthetic_df.loc[unmapped_mask, target_col] = sampled_values
                             print(f"    Sampled and assigned {num_unmapped} values to '{target_col}' for new synthetic '{source_col}' values.")
                         else:
                             print(f"    Warning: No non-null values in original target column '{target_col}' to sample from. Unmapped synthetic '{source_col}' values will remain as generated by Faker (potentially incorrect).")

                except Exception as e:
                    print(f"    Error applying functional dependency {source_col} -> {target_col}: {e}")
                    import traceback
                    traceback.print_exc()
            elif is_one_to_one:
                 # print(f"  Skipping functional dependency {source_col} -> {target_col}: It's a one-to-one relationship (handled separately).")
                 pass # Skip if it's a one-to-one, already handled
            else:
                print(f"  Skipping functional dependency {source_col} -> {target_col}: Source or target column not found in synthetic or original data.")

    print("Finished applying functional dependencies.")
    return synthetic_df

def apply_inequalities(synthetic_df, relationships, schema):
    """
    Applies inequality relationships (<=, >=) as a post-processing step.
    Reads column info from the 'columns' key in the schema.
    Reads relationships from the top-level 'relationships' key.
    """
    print("\nApplying inequality constraints...")

    # Access column info from the 'columns' key
    column_infos = schema.get("columns", {})
    # Access relationships from the top-level 'relationships' key
    relationships_data = relationships.get("relationships", {})

    inequalities_to_apply = {}
    for col, rel_list in relationships_data.items():
         ineq_rels = [rel for rel in rel_list if rel.get("type") == "value_relationship" and rel.get("relationship") in ['<=', '>=', 'less_than_or_equal_to', 'greater_than_or_equal_to']]
         if ineq_rels:
              inequalities_to_apply[col] = ineq_rels

    if not inequalities_to_apply:
        print("  No inequality relationships found in schema to apply.")
        return synthetic_df


    for col1, rel_list in inequalities_to_apply.items():
        for rel in rel_list:
            col2 = rel.get("column")
            relation = rel.get("relationship")
            normalized_relation = '<=' if relation in ['<=', 'less_than_or_equal_to'] else '>='

            if col1 in synthetic_df.columns and col2 in synthetic_df.columns:
                 print(f"  Enforcing inequality: {col1} {normalized_relation} {col2}")

                 try:
                     series1 = pd.to_numeric(synthetic_df[col1], errors='coerce')
                     series2 = pd.to_numeric(synthetic_df[col2], errors='coerce')

                     if normalized_relation == '<=':
                         violation_mask = (series1 > series2) & series1.notna() & series2.notna()
                     else:
                         violation_mask = (series1 < series2) & series1.notna() & series2.notna()

                     if violation_mask.any():
                         synthetic_df.loc[violation_mask, col1] = series2[violation_mask]
                         print(f"    Corrected {violation_mask.sum()} values in '{col1}'.")

                 except Exception as e:
                     print(f"    Error applying inequality {col1} {normalized_relation} {col2}: {e}")
                     import traceback
                     traceback.print_exc()
            else:
                print(f"  Skipping inequality {col1} {normalized_relation} {col2}: Columns not found in synthetic data.")

    print("Finished applying inequality constraints.")
    return synthetic_df

def apply_formulas(synthetic_df, relationships, schema):
    """
    Applies formula relationships (A = B + C) as a post-processing step.
    Reads column info from the 'columns' key in the schema.
    Reads relationships from the top-level 'relationships' key.
    """
    print("\nApplying formula constraints (A = B + C)...")

    # Access column info from the 'columns' key
    column_infos = schema.get("columns", {})
    # Access relationships from the top-level 'relationships' key
    relationships_data = relationships.get("relationships", {})

    formulas_to_apply = {}
    for col, rel_list in relationships_data.items():
         formula_rels = [rel for rel in rel_list if rel.get("type") == "formula"]
         if formula_rels:
              formulas_to_apply[col] = formula_rels


    if not formulas_to_apply:
         print("  No formula relationships found in schema to apply.")
         return synthetic_df


    for target_col, rel_list in formulas_to_apply.items():
        for rel in rel_list:
            source_cols = rel.get("source_columns")
            formula_str = rel.get("formula")

            if target_col in synthetic_df.columns and all(src_col in synthetic_df.columns for src_col in source_cols):
                 print(f"  Enforcing formula: {target_col} = {formula_str}")

                 try:
                     target_series = pd.to_numeric(synthetic_df[target_col], errors='coerce')
                     source_series_list = [pd.to_numeric(synthetic_df[src_col], errors='coerce') for src_col in source_cols]

                     if len(source_series_list) == 2: # Assuming simple A = B + C for now
                          expected_value = source_series_list[0] + source_series_list[1]
                          violation_mask = ~np.isclose(target_series, expected_value, atol=FORMULA_TOLERANCE, equal_nan=True) & \
                                           target_series.notna() & source_series_list[0].notna() & source_series_list[1].notna()

                          if violation_mask.any():
                              synthetic_df.loc[violation_mask, target_col] = expected_value[violation_mask]
                              print(f"    Corrected {violation_mask.sum()} values in '{target_col}'.")

                 except Exception as e:
                     print(f"    Error applying formula {target_col} = {formula_str}: {e}")
                     import traceback
                     traceback.print_exc()
            else:
                print(f"  Skipping formula {target_col} = {formula_str}: Target or source columns not found in synthetic data.")

    print("Finished applying formula constraints.")
    return synthetic_df


def apply_range_constraints(synthetic_df, schema):
    """
    Applies range constraints (min/max) as a post-processing step for numeric and date columns.
    Clips values outside the detected range.
    Reads column info from the 'columns' key in the schema.
    """
    print("\nApplying range constraints (min/max)...")

    # Access column info from the 'columns' key
    column_infos = schema.get("columns", {})

    for col_name, col_info in column_infos.items():
        data_type = col_info.get("data_type")
        stats = col_info.get("stats", {})

        if col_name in synthetic_df.columns:
             if data_type in ["integer", "float", "numerical"] and 'min' in stats and 'max' in stats:
                  min_val = stats['min']
                  max_val = stats['max']

                  try:
                      series = pd.to_numeric(synthetic_df[col_name], errors='coerce')
                      violation_mask = ((series < min_val) | (series > max_val)) & series.notna()

                      if violation_mask.any():
                          synthetic_df.loc[violation_mask, col_name] = np.clip(series[violation_mask], min_val, max_val)
                          print(f"    Clipped {violation_mask.sum()} values in '{col_name}'.")

                  except Exception as e:
                      print(f"    Error applying range constraint for '{col_name}' (min={min_val}, max={max_val}): {e}")
                      import traceback
                      traceback.print_exc()

             elif data_type == "datetime" and 'min' in stats and 'max' in stats:
                  min_date_str = stats['min']
                  max_date_str = stats['max']

                  try:
                      date_formats_to_try = ['%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y', '%Y/%m/%d', '%Y%m%d']
                      min_date_obj = None
                      max_date_obj = None

                      for fmt in date_formats_to_try:
                           try:
                               min_date_obj = datetime.strptime(min_date_str, fmt)
                               break
                           except ValueError:
                               pass

                      for fmt in date_formats_to_try:
                           try:
                               max_date_obj = datetime.strptime(max_date_str, fmt)
                               break
                           except ValueError:
                               pass

                      if min_date_obj and max_date_obj:
                           series = pd.to_datetime(synthetic_df[col_name], errors='coerce')
                           violation_mask = ((series < min_date_obj) | (series > max_date_obj)) & series.notna()

                           if violation_mask.any():
                               adjusted_values = series[violation_mask].apply(lambda x: min_date_obj if x < min_date_obj else max_date_obj)
                               synthetic_df.loc[violation_mask, col_name] = adjusted_values
                               print(f"    Adjusted {violation_mask.sum()} values in '{col_name}'.")

                      else:
                           pass

                  except Exception as e:
                      print(f"    Error applying date range constraint for '{col_name}': {e}")
                      import traceback
                      traceback.print_exc()

    print("Finished applying range constraints.")
    return synthetic_df


def detect_relationships_in_synthetic_data(df):
    """
    Detects column relationships (Functional Dependency, One-to-One, Value Relationships,
    Simple Formulas, Pearson Correlation) in the synthetic data for reporting.
    """
    functional_dependencies_forward = {}
    functional_dependencies_backward = {}
    one_to_one_relationships = {}
    value_relationships = {}
    pearson_correlations = {}
    formula_relationships = {}

    global CORRELATION_REPORTING_THRESHOLD
    global FORMULA_TOLERANCE
    global RELATIONSHIP_DETECTION_THRESHOLD

    try:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        all_cols = df.columns.tolist()

        print("\nDetecting functional dependencies (forward and backward) in synthetic data...")
        for col1 in all_cols:
            for col2 in all_cols:
                if col1 != col2:
                    try:
                        df_filtered = df[[col1, col2]].dropna()
                        if not df_filtered.empty:
                            if pd.api.types.is_hashable(df_filtered[col1].dtype):
                                if not df_filtered[col1].empty:
                                     if pd.api.types.is_hashable(df_filtered[col1].iloc[0]):
                                        unique_values_per_group_forward = df_filtered.groupby(col1)[col2].nunique()
                                        consistent_groups_forward = (unique_values_per_group_forward <= 1).sum()
                                        if consistent_groups_forward / len(unique_values_per_group_forward) >= RELATIONSHIP_DETECTION_THRESHOLD:
                                            if col1 not in functional_dependencies_forward: functional_dependencies_forward[col1] = []
                                            functional_dependencies_forward[col1].append({"column": col2})

                            if pd.api.types.is_hashable(df_filtered[col2].dtype):
                                if not df_filtered[col2].empty:
                                     if pd.api.types.is_hashable(df_filtered[col2].iloc[0]):
                                        unique_values_per_group_backward = df_filtered.groupby(col2)[col1].nunique()
                                        consistent_groups_backward = (unique_values_per_group_backward <= 1).sum()
                                        if consistent_groups_backward / len(unique_values_per_group_backward) >= RELATIONSHIP_DETECTION_THRESHOLD:
                                            if col2 not in functional_dependencies_backward: functional_dependencies_backward[col2] = []
                                            functional_dependencies_backward[col2].append({"column": col1})

                    except Exception as dep_e:
                        pass

        print("\nDetecting one-to-one relationships in synthetic data...")
        for col1 in all_cols:
            for col2 in all_cols:
                if col1 < col2:
                    col1_to_col2_found = col1 in functional_dependencies_forward and any(d["column"] == col2 for d in functional_dependencies_forward[col1])
                    col2_to_col1_found = col2 in functional_dependencies_backward and any(d["column"] == col1 for d in functional_dependencies_backward[col2])

                    if col1_to_col2_found and col2_to_col1_found:
                        if col1 not in one_to_one_relationships: one_to_one_relationships[col1] = []
                        one_to_one_relationships[col1].append({"column": col2, "type": "one_to_one"})
                        if col2 not in one_to_one_relationships: one_to_one_relationships[col2] = []
                        one_to_one_relationships[col2].append({"column": col1, "type": "one_to_one"})
                        print(f"Detected one-to-one relationship in synthetic: {col1} <-> {col2}")

        print("\nChecking for basic value relationships (e.g., <=, >=) in synthetic data...")
        if len(numeric_cols) >= 2:
             for col1 in numeric_cols:
                 for col2 in numeric_cols:
                     if col1 != col2:
                         try:
                             df_filtered = df[[col1, col2]].dropna()
                             if not df_filtered.empty:
                                 num_rows_checked = len(df_filtered)

                                 if num_rows_checked > 0 and (pd.to_numeric(df_filtered[col1], errors='coerce') <= pd.to_numeric(df_filtered[col2], errors='coerce')).sum() / num_rows_checked >= RELATIONSHIP_DETECTION_THRESHOLD:
                                     if col1 < col2:
                                         if col1 not in value_relationships: value_relationships[col1] = []
                                         if not any(d.get("column") == col2 and d.get("relationship") == "less_than_or_equal_to" for d in value_relationships[col1]):
                                              value_relationships[col1].append({"column": col2, "relationship": "less_than_or_equal_to", "type": "value_relationship"})
                                              print(f"Detected value relationship in synthetic: {col1} <= {col2}")

                                 if num_rows_checked > 0 and (pd.to_numeric(df_filtered[col1], errors='coerce') >= pd.to_numeric(df_filtered[col2], errors='coerce')).sum() / num_rows_checked >= RELATIONSHIP_DETECTION_THRESHOLD:
                                     if col1 < col2:
                                         if col1 not in value_relationships: value_relationships[col1] = []
                                         if not any(d.get("column") == col2 and d.get("relationship") == "greater_than_or_equal_to" for d in value_relationships[col1]):
                                              value_relationships[col1].append({"column": col2, "relationship": "greater_than_or_equal_to", "type": "value_relationship"})
                                              print(f"Detected value relationship in synthetic: {col1} >= {col2}")

                         except Exception as rel_e:
                             pass

        print("\nChecking for simple formula relationships (A = B + C) in synthetic data...")
        if len(numeric_cols) >= 3:
             for col_a, col_b, col_c in combinations(numeric_cols, 3):
                 try:
                     df_filtered = df[[col_a, col_b, col_c]].dropna()
                     if not df_filtered.empty:
                         series_a = pd.to_numeric(df_filtered[col_a], errors='coerce')
                         series_b = pd.to_numeric(df_filtered[col_b], errors='coerce')
                         series_c = pd.to_numeric(df_filtered[col_c], errors='coerce')

                         if np.allclose(series_a, series_b + series_c, atol=FORMULA_TOLERANCE, equal_nan=True):
                             formula = f"{col_b} + {col_c}"
                             if col_a not in formula_relationships: formula_relationships[col_a] = []
                             if not any(d.get("formula") == formula for d in formula_relationships[col_a]):
                                  formula_relationships[col_a].append({"target_column": col_a, "source_columns": [col_b, col_c], "formula": formula, "type": "formula"})
                                  print(f"Detected formula relationship in synthetic: {col_a} = {formula}")

                         if np.allclose(series_b, series_a + series_c, atol=FORMULA_TOLERANCE, equal_nan=True):
                             formula = f"{col_a} + {col_c}"
                             if col_b not in formula_relationships: formula_relationships[col_b] = []
                             if not any(d.get("formula") == formula for d in formula_relationships[col_b]):
                                  formula_relationships[col_b].append({"target_column": col_b, "source_columns": [col_a, col_c], "formula": formula, "type": "formula"})
                                  print(f"Detected formula relationship in synthetic: {col_b} = {formula}")

                         if np.allclose(series_c, series_a + series_b, atol=FORMULA_TOLERANCE, equal_nan=True):
                             formula = f"{col_a} + {col_b}"
                             if col_c not in formula_relationships: formula_relationships[col_c] = []
                             if not any(d.get("formula") == formula for d in formula_relationships[col_c]):
                                  formula_relationships[col_c].append({"target_column": col_c, "source_columns": [col_a, col_b], "formula": formula, "type": "formula"})
                                  print(f"Detected formula relationship in synthetic: {col_c} = {formula}")

                 except Exception as formula_e:
                     pass

        print("\nDetecting Pearson correlations in synthetic data...")
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
                                 if abs(corr_value) >= CORRELATION_REPORTING_THRESHOLD and not pd.isna(corr_value):
                                     if col1 not in pearson_correlations: pearson_correlations[col1] = []
                                     pearson_correlations[col1].append({"column": col2, "correlation": float(round(corr_value, 3)), "type": "pearson_correlation"})
                                     if col2 not in pearson_correlations: pearson_correlations[col2] = []
                                     pearson_correlations[col2].append({"column": col1, "correlation": float(round(corr_value, 3)), "type": "pearson_correlation"})
                     print(f"Detected Pearson correlations above threshold {CORRELATION_REPORTING_THRESHOLD} in synthetic data.")
                else:
                     pass

            except Exception as pearson_e:
                 print(f"Error during Pearson correlation calculation for reporting in synthetic data: {pearson_e}")

    except Exception as e:
        print(f"Error detecting relationships for reporting in synthetic data: {str(e)}")
        import traceback
        traceback.print_exc()

    detected_relationships = {}
    def add_to_detected(col, rel):
        if col not in detected_relationships: detected_relationships[col] = []
        if rel not in detected_relationships[col]: detected_relationships[col].append(rel)

    for col, rel_list in functional_dependencies_forward.items():
         for rel in rel_list: add_to_detected(col, {**rel, "type": "functional_dependency"})
    for col, rel_list in one_to_one_relationships.items():
         for rel in rel_list: add_to_detected(col, {**rel, "type": "one_to_one"})
    for col, rel_list in value_relationships.items():
         for rel in rel_list: add_to_detected(col, rel)
    for col, rel_list in formula_relationships.items():
         for rel in rel_list: add_to_detected(col, rel)
    for col, rel_list in pearson_correlations.items():
         for rel in rel_list: add_to_detected(col, rel)

    detected_relationships = {col: rels for col, rels in detected_relationships.items() if rels}
    return detected_relationships

def compare_relationships(original_relationships, synthetic_relationships, exclude_types=None):
    """
    Compares relationships detected in original and synthetic data and returns a DataFrame.
    Original relationships are expected to be from the top-level 'relationships' key.
    Synthetic relationships are expected to be from the detect_relationships_in_synthetic_data function.
    Optionally excludes certain relationship types from the comparison.
    """
    print("\nComparing relationships...")
    comparison_data = []
    exclude_types = exclude_types if exclude_types is not None else []

    all_relationships = {}

    # Add original relationships (from schema's top-level 'relationships')
    for col, rel_list in original_relationships.get("relationships", {}).items(): # Access relationships from the key
         for rel in rel_list:
              if rel.get("type") in exclude_types: continue

              # Create a unique key for each relationship
              if rel.get("type") in ["one_to_one", "pearson_correlation", "value_relationship", "functional_dependency"] and "column" in rel:
                   # For symmetric relationships (O2O, Pearson, Value), use sorted columns in the key
                   if rel.get("type") in ["one_to_one", "pearson_correlation", "value_relationship"]:
                        cols_in_pair = tuple(sorted([col, rel["column"]]))
                        rel_key_details = {k: v for k, v in rel.items() if k not in ["column", "type"]}
                        rel_key = json.dumps({"cols": cols_in_pair, "details": rel_key_details, "type": rel.get("type")}, sort_keys=True)
                   # For asymmetric relationships (FD), use source -> target
                   elif rel.get("type") == "functional_dependency":
                        rel_key = json.dumps({"source": col, "target": rel["column"], "type": rel.get("type")}, sort_keys=True)
                   else:
                        # Fallback for other types with 'column'
                        rel_key = json.dumps({"col": col, "rel": rel}, sort_keys=True)

              elif rel.get("type") == "formula" and "source_columns" in rel and "formula" in rel:
                   # For formulas, use target and sorted source columns
                   rel_key = json.dumps({"target": col, "sources": sorted(rel["source_columns"]), "formula": rel.get("formula"), "type": rel.get("type")}, sort_keys=True)
              else:
                   # Fallback for any other relationship structure
                   rel_key = json.dumps({"col": col, "rel": rel}, sort_keys=True)


              if rel_key not in all_relationships:
                  all_relationships[rel_key] = {
                      "Relationship Type": rel.get("type"),
                      "Original Data": True,
                      "Synthetic Data": False,
                      "Details": rel # Store original details
                  }
                  # Add specific details for display later
                  if rel.get("type") == "pearson_correlation":
                       all_relationships[rel_key]["Original Correlation"] = rel.get("correlation")
                  elif rel.get("type") == "value_relationship":
                       all_relationships[rel_key]["Original Relationship"] = rel.get("relationship")
                  elif rel.get("type") == "formula":
                       all_relationships[rel_key]["Original Formula"] = rel.get("formula")


    # Add synthetic relationships (from detect_relationships_in_synthetic_data)
    for col, rel_list in synthetic_relationships.items():
         for rel in rel_list:
              if rel.get("type") in exclude_types: continue

              # Create a unique key for each relationship (matching the original logic)
              if rel.get("type") in ["one_to_one", "pearson_correlation", "value_relationship", "functional_dependency"] and "column" in rel:
                   if rel.get("type") in ["one_to_one", "pearson_correlation", "value_relationship"]:
                        cols_in_pair = tuple(sorted([col, rel["column"]]))
                        rel_key_details = {k: v for k, v in rel.items() if k not in ["column", "type"]}
                        rel_key = json.dumps({"cols": cols_in_pair, "details": rel_key_details, "type": rel.get("type")}, sort_keys=True)
                   elif rel.get("type") == "functional_dependency":
                        rel_key = json.dumps({"source": col, "target": rel["column"], "type": rel.get("type")}, sort_keys=True)
                   else:
                        rel_key = json.dumps({"col": col, "rel": rel}, sort_keys=True)

              elif rel.get("type") == "formula" and "source_columns" in rel and "formula" in rel:
                   rel_key = json.dumps({"target": col, "sources": sorted(rel["source_columns"]), "formula": rel.get("formula"), "type": rel.get("type")}, sort_keys=True)
              else:
                   rel_key = json.dumps({"col": col, "rel": rel}, sort_keys=True)


              if rel_key not in all_relationships:
                  # If a relationship is in synthetic but not original, mark original as False
                  all_relationships[rel_key] = {
                      "Relationship Type": rel.get("type"),
                      "Original Data": False,
                      "Synthetic Data": True,
                      "Details": rel # Store synthetic details
                  }
                  # Add specific details for display later
                  if rel.get("type") == "pearson_correlation":
                       all_relationships[rel_key]["Synthetic Correlation"] = rel.get("correlation")
                  elif rel.get("type") == "value_relationship":
                       all_relationships[rel_key]["Synthetic Relationship"] = rel.get("relationship")
                  elif rel.get("type") == "formula":
                       all_relationships[rel_key]["Synthetic Formula"] = rel.get("formula")

              else:
                  # If the relationship is in both, update synthetic status and values
                  all_relationships[rel_key]["Synthetic Data"] = True
                  if rel.get("type") == "pearson_correlation":
                       all_relationships[rel_key]["Synthetic Correlation"] = rel.get("correlation")
                  elif rel.get("type") == "value_relationship":
                       all_relationships[rel_key]["Synthetic Relationship"] = rel.get("relationship")
                  elif rel.get("type") == "formula":
                       all_relationships[rel_key]["Synthetic Formula"] = rel.get("formula")


    comparison_data = []
    for rel_key, status in all_relationships.items():
        row = {
            "Relationship Type": status.get("Relationship Type", "Unknown"),
            "Original Data": status["Original Data"],
            "Synthetic Data": status["Synthetic Data"]
        }

        # Extract columns and details based on the relationship type from the key or stored details
        try:
             parsed_key = json.loads(rel_key)
             rel_type = status.get("Relationship Type", "Unknown") # Use type from status if available

             if rel_type == "pearson_correlation":
                  row["Column 1"] = parsed_key.get("cols", ["N/A", "N/A"])[0]
                  row["Column 2"] = parsed_key.get("cols", ["N/A", "N/A"])[1]
                  row["Original Value"] = status.get("Original Correlation", "N/A")
                  row["Synthetic Value"] = status.get("Synthetic Correlation", "N/A")
                  row["Details"] = f"{row.get('Column 1', 'N/A')} - {row.get('Column 2', 'N/A')}"

             elif rel_type == "functional_dependency":
                  row["Source Column"] = parsed_key.get("source", "N/A")
                  row["Target Column"] = parsed_key.get("target", "N/A")
                  row["Original Value"] = "Exists" if status["Original Data"] else "Not Detected"
                  row["Synthetic Value"] = "Exists" if status["Synthetic Data"] else "Not Detected"
                  row["Details"] = f"{row.get('Source Column', 'N/A')} -> {row.get('Target Column', 'N/A')}"

             elif rel_type == "one_to_one":
                  row["Column 1"] = parsed_key.get("cols", ["N/A", "N/A"])[0]
                  row["Column 2"] = parsed_key.get("cols", ["N/A", "N/A"])[1]
                  row["Original Value"] = "Exists" if status["Original Data"] else "Not Detected"
                  row["Synthetic Value"] = "Exists" if status["Synthetic Data"] else "Not Detected"
                  row["Details"] = f"{row.get('Column 1', 'N/A')} <-> {row.get('Column 2', 'N/A')}"

             elif rel_type == "value_relationship":
                  row["Column 1"] = parsed_key.get("cols", ["N/A", "N/A"])[0]
                  row["Column 2"] = parsed_key.get("cols", ["N/A", "N/A"])[1]
                  row["Relationship"] = parsed_key.get("details", {}).get("relationship", "N/A") # Get relationship from key details
                  row["Original Value"] = status.get("Original Relationship", "N/A")
                  row["Synthetic Value"] = status.get("Synthetic Relationship", "N/A")
                  row["Details"] = f"{row.get('Column 1', 'N/A')} {row.get('Relationship', 'N/A')} {row.get('Column 2', 'N/A')}"

             elif rel_type == "formula":
                  row["Target Column"] = parsed_key.get("target", "N/A")
                  row["Source Columns"] = ", ".join(parsed_key.get("sources", []))
                  row["Formula"] = parsed_key.get("formula", "N/A")
                  row["Original Value"] = "Exists" if status["Original Data"] else "Not Detected"
                  row["Synthetic Value"] = "Exists" if status["Synthetic Data"] else "Not Detected"
                  row["Details"] = f"{row.get('Target Column', 'N/A')} = {row.get('Formula', 'N/A')}"

             else:
                 # Fallback for unknown types - try to get some info from details
                 details = status.get("Details", {})
                 row["Details"] = json.dumps(details)
                 row["Original Value"] = "Exists" if status["Original Data"] else "Not Detected"
                 row["Synthetic Value"] = "Exists" if status["Synthetic Data"] else "Not Detected"


        except Exception as e:
             print(f"Error parsing relationship key or details for comparison: {e}. Key: {rel_key}. Status: {status}. Using default N/A.")
             row["Column 1"], row["Column 2"], row["Source Column"], row["Target Column"], row["Relationship"], row["Source Columns"], row["Formula"] = "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"
             row["Original Value"] = "Exists" if status["Original Data"] else "Not Detected"
             row["Synthetic Value"] = "Exists" if status["Synthetic Data"] else "Not Detected"
             row["Details"] = json.dumps(status.get("Details", {}))


        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)
    ordered_columns = [
        "Relationship Type", "Details", "Original Data", "Synthetic Data",
        "Column 1", "Column 2", "Original Correlation", "Synthetic Correlation",
        "Source Column", "Target Column", "Original Relationship", "Synthetic Relationship",
        "Source Columns", "Original Formula", "Synthetic Formula"
    ]
    # Filter columns to only include those present in the comparison_df
    comparison_df = comparison_df.reindex(columns=[col for col in ordered_columns if col in comparison_df.columns])

    print("Relationship comparison complete.")
    return comparison_df

def save_data(data, csv_file_path):
    """Saves data to a CSV file."""
    try:
        data.to_csv(csv_file_path, index=False)
        print(f"Data successfully saved to {csv_file_path}")
    except Exception as e:
        print(f"Error saving data to {csv_file_path}: {e}")

def main():
    """Main function to orchestrate the synthetic data generation process."""
    print("--- Starting synthetic data generation process (Faker + Post-processing) ---")

    original_data = load_data(INPUT_CSV)
    if original_data is None:
        print("Failed to load input data. Exiting.")
        return

    schema = load_schema(SCHEMA_JSON)
    if schema is None:
        print("Failed to load schema. Exiting.")
        return

    # Load relationships from the top-level 'relationships' key
    original_relationships = schema.get("relationships", {})
    # Load column information from the top-level 'columns' key
    column_schema_info = schema.get("columns", {})

    print(f"Loaded relationships from schema's 'relationships' key for {len(original_relationships)} columns.")
    print(f"Loaded column information from schema's 'columns' key for {len(column_schema_info)} columns.")


    # Generate initial synthetic data using Faker (including custom regex)
    # Pass the column schema info to the generation function
    synthetic_data = generate_initial_data_with_faker(schema, NUM_SYNTHETIC_ROWS, original_data)
    if synthetic_data is None:
        print("Failed to generate initial synthetic data. Exiting.")
        return

    # Apply Post-processing for relationships and constraints
    # Pass the full schema (containing both 'columns' and 'relationships')
    synthetic_data = enforce_primary_key_uniqueness(synthetic_data, schema, original_data)

    # Pass the full schema (containing both 'columns' and 'relationships') to the application functions
    synthetic_data = apply_one_to_one_relationships(synthetic_data, original_data, schema, schema) # Pass schema for relationships and column info
    synthetic_data = apply_functional_dependencies(synthetic_data, original_data, schema, schema) # Pass schema for relationships and column info
    synthetic_data = apply_inequalities(synthetic_data, schema, schema) # Pass schema for relationships and column info
    synthetic_data = apply_formulas(synthetic_data, schema, schema) # Pass schema for relationships and column info
    synthetic_data = apply_range_constraints(synthetic_data, schema) # Pass schema for column info


    save_data(synthetic_data, OUTPUT_SYNTHETIC_CSV)

    print("\nDetecting relationships in the generated synthetic data...")
    synthetic_relationships_all = detect_relationships_in_synthetic_data(synthetic_data)
    print(f"Detected {len(synthetic_relationships_all)} columns with relationships (including Pearson) in synthetic data.")

    # Pass the original relationships (from schema) and synthetic relationships (detected) to compare
    relationship_comparison_df = compare_relationships(schema, synthetic_relationships_all) # Pass the full schema for original relationships structure
    save_data(relationship_comparison_df, OUTPUT_RELATIONSHIP_COMPARISON_CSV)

    print("\n--- Synthetic data generation process finished ---")

if __name__ == "__main__":
    main()
