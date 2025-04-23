import pandas as pd
import json
import os
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta, date # Import date specifically for Faker date args
import re # Import re for potential regex-based Faker generation or validation
from itertools import combinations # Import combinations for formula detection in synthetic data
import inspect # Import inspect to check function signatures
import warnings
import string # Import string for character sets

# Configuration
INPUT_CSV = "customer_data.csv" # Original data for mappings and distributions
SCHEMA_JSON = "enhanced_schema.json" # Schema with column info and relationships
OUTPUT_SYNTHETIC_CSV = "synthetic_customer_data_faker.csv" # Output file for synthetic data
OUTPUT_RELATIONSHIP_COMPARISON_CSV = "relationship_comparison.csv" # Output file for relationship comparison
NUM_SYNTHETIC_ROWS = 10000 # Number of synthetic rows to generate
FORMULA_TOLERANCE = 1e-6 # Tolerance for floating-point comparisons in formula detection
INEQUALITY_CONSISTENCY_THRESHOLD = 0.999 # Threshold for checking inequality consistency in synthetic data

# Initialize Faker
# Use the same locale as in the schema generation script
fake = Faker(['vi_VN'])
Faker.seed(42) # Use a fixed seed for reproducibility

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

def generate_initial_data_with_faker(schema, num_rows, original_df=None):
    """
    Generates initial synthetic data using Faker based on schema and original data stats.
    """
    print(f"\nGenerating {num_rows} initial rows using Faker...")
    synthetic_data = {}
    column_infos = schema.get("columns", {})

    # Create a list of columns to maintain order
    columns = list(column_infos.keys())

    for col_name in columns:
        col_info = column_infos.get(col_name, {})
        faker_provider_name = col_info.get("faker_provider")
        faker_args = col_info.get("faker_args", {})
        data_type = col_info.get("data_type")
        stats = col_info.get("stats", {})
        null_percentage = col_info.get("null_percentage", 0)

        generated_values = []

        # Get the Faker provider method
        faker_method = getattr(fake, faker_provider_name, None) if faker_provider_name else None

        if faker_method:
            print(f"  Generating data for column '{col_name}' using Faker provider '{faker_provider_name}'...")
            # Check if the faker method accepts the arguments specified in faker_args
            # This is a basic check, complex args might still cause issues
            try:
                 # Inspect the method signature to see what arguments it accepts
                 sig = inspect.signature(faker_method)
                 valid_args = {arg: faker_args[arg] for arg in faker_args if arg in sig.parameters}
                 # Add 'locale' if the method accepts it and it's not already in args
                 if 'locale' in sig.parameters and 'locale' not in valid_args:
                      valid_args['locale'] = fake.locale # Use the locale the Faker instance was initialized with

                 # Generate values using the Faker method with validated args
                 for _ in range(num_rows):
                     try:
                         value = faker_method(**valid_args)
                         generated_values.append(value)
                     except Exception as e:
                         # If Faker generation fails for a row, generate a placeholder or sample from original
                         # print(f"    Warning: Faker generation failed for row in '{col_name}': {e}. Using placeholder.") # Suppress per-row error
                         # Attempt to sample a value from original data if available
                         if original_df is not None and col_name in original_df.columns and not original_df[col_name].empty:
                             sample_value = random.choice(original_df[col_name].dropna().tolist())
                             generated_values.append(sample_value)
                         else:
                             # Fallback to a type-appropriate placeholder if sampling isn't possible
                             if data_type in ["integer", "float", "numerical"]:
                                 generated_values.append(0)
                             elif data_type == "datetime":
                                 generated_values.append(datetime.now())
                             else:
                                 generated_values.append("ERROR_GENERATING")

            except Exception as e:
                 print(f"  Warning: Could not call Faker provider '{faker_provider_name}' for '{col_name}' with args {faker_args}: {e}. Falling back to sampling/placeholders.")
                 # Fallback if method inspection or initial call fails
                 if original_df is not None and col_name in original_df.columns and not original_df[col_name].empty:
                      original_values = original_df[col_name].dropna().tolist()
                      # --- FIX: Add check for empty original_values before sampling ---
                      if original_values:
                           generated_values = random.choices(original_values, k=num_rows)
                      else:
                           print(f"    Warning: Original data for '{col_name}' is empty or contains only nulls. Cannot sample. Generating placeholders.")
                           # Fallback to type-appropriate placeholders if no original data or column
                           for _ in range(num_rows):
                               if data_type in ["integer", "float", "numerical"]:
                                   generated_values.append(0)
                               elif data_type == "datetime":
                                   generated_values.append(datetime.now())
                               else:
                                   generated_values.append("PLACEHOLDER")
                      # --- End FIX ---
                 else:
                      print(f"  No Faker provider or original data available for '{col_name}'. Generating placeholders.")
                      # Fallback to type-appropriate placeholders if no original data or column
                      for _ in range(num_rows):
                          if data_type in ["integer", "float", "numerical"]:
                              generated_values.append(0)
                          elif data_type == "datetime":
                              generated_values.append(datetime.now())
                          else:
                              generated_values.append("PLACEHOLDER")


        elif original_df is not None and col_name in original_df.columns and not original_df[col_name].empty:
            print(f"  No Faker provider specified or found for '{col_name}'. Sampling from original data distribution.")
            # If no Faker provider, sample from the original data distribution
            original_values = original_df[col_name].dropna().tolist()
            # --- FIX: Add check for empty original_values before sampling ---
            if original_values:
                 generated_values = random.choices(original_values, k=num_rows)
            else:
                 print(f"    Warning: Original data for '{col_name}' is empty or contains only nulls. Cannot sample. Generating placeholders.")
                 # Fallback to type-appropriate placeholders if no original data or column
                 for _ in range(num_rows):
                     if data_type in ["integer", "float", "numerical"]:
                         generated_values.append(0)
                     elif data_type == "datetime":
                         generated_values.append(datetime.now())
                     else:
                         generated_values.append("PLACEHOLDER")
            # --- End FIX ---
        else:
            print(f"  No Faker provider or original data available for '{col_name}'. Generating placeholders.")
            # If no Faker provider and no original data, generate placeholders based on detected type
            for _ in range(num_rows):
                if data_type in ["integer", "float", "numerical"]:
                    generated_values.append(0)
                elif data_type == "datetime":
                    generated_values.append(datetime.now())
                else:
                    generated_values.append("PLACEHOLDER")


        # Apply nulls based on the detected null percentage
        num_nulls_to_add = int(num_rows * (null_percentage / 100))
        null_indices = random.sample(range(num_rows), min(num_nulls_to_add, num_rows))
        for idx in null_indices:
            generated_values[idx] = np.nan # Use numpy.nan for nulls in pandas

        synthetic_data[col_name] = generated_values

    # Create DataFrame, ensuring column order matches original or schema
    synthetic_df = pd.DataFrame(synthetic_data, columns=columns)
    print("Initial data generation complete.")
    return synthetic_df

def apply_functional_dependencies(synthetic_df, original_df, functional_dependencies):
    """
    Applies functional dependencies as a post-processing step.
    """
    print("\nApplying functional dependencies...")

    # Iterate through the functional dependencies identified in the schema
    for source_col, rel_list in functional_dependencies.items():
        # Filter for relationships of type functional_dependency
        fd_relationships = [rel for rel in rel_list if rel.get("type") == "functional_dependency"]

        for rel in fd_relationships:
            target_col = rel.get("column")

            if source_col in synthetic_df.columns and target_col in synthetic_df.columns and \
               source_col in original_df.columns and target_col in original_df.columns:

                print(f"  Enforcing functional dependency: {source_col} -> {target_col}")

                try:
                    # 1. Create mapping from original data
                    # Drop rows with NaN in source or target, drop duplicates based on source_col, keep first mapping
                    original_mapping = original_df.dropna(subset=[source_col, target_col]).drop_duplicates(subset=[source_col], keep='first').set_index(source_col)[target_col].to_dict()
                    # print(f"    Original mapping created for {source_col} -> {target_col}: {original_mapping}") # Debug print

                    # 2. Apply the mapping to the synthetic DataFrame
                    # Use .map() on the synthetic source column. Values not in original_mapping will become NaN.
                    mapped_series = synthetic_df[source_col].map(original_mapping)
                    # print(f"    Mapped series created for {target_col}.") # Debug print

                    # 3. Update the target column in the synthetic DataFrame
                    # Only update rows where the mapping was successful (not NaN after map)
                    # This preserves original synthetic values for source values not seen in original data
                    mapped_mask = mapped_series.notna()

                    if mapped_mask.any():
                         # Use .loc for assignment to avoid SettingWithCopyWarning
                         synthetic_df.loc[mapped_mask, target_col] = mapped_series[mapped_mask]
                         print(f"    Updated {mapped_mask.sum()} values in '{target_col}' based on '{source_col}' mapping.")
                    else:
                         print(f"    Warning: No values updated in '{target_col}': No successful mappings found for '{source_col}' values in synthetic data.")

                    # Handle synthetic source values that were NOT in the original data's source column
                    # These will have NaN in the mapped_series. We need to fill these NaNs.
                    unmapped_mask = mapped_series.isna() & synthetic_df[source_col].notna() # Synthetic source is not NaN, but mapping resulted in NaN

                    if unmapped_mask.any():
                         print(f"    Handling {unmapped_mask.sum()} synthetic '{source_col}' values not found in original data mapping.")
                         # Get the list of possible target values from the original data
                         original_target_values = original_df[target_col].dropna().unique().tolist()

                         if original_target_values:
                             # For each unmapped synthetic source value, sample a target value from the original distribution
                             # Avoid iterating row by row if possible for performance
                             # Create a series of sampled values matching the index of unmapped rows
                             num_unmapped = unmapped_mask.sum()
                             # Ensure sampling handles potential data types (e.g., convert to appropriate type after sampling if needed)
                             sampled_values = random.choices(original_target_values, k=num_unmapped)
                             synthetic_df.loc[unmapped_mask, target_col] = sampled_values
                             print(f"    Sampled and assigned {num_unmapped} values to '{target_col}' for new synthetic '{source_col}' values.")
                         else:
                             print(f"    Warning: No non-null values in original target column '{target_col}' to sample from. Unmapped synthetic '{source_col}' values will remain as generated by Faker (potentially incorrect).")


                except Exception as e:
                    print(f"    Error applying functional dependency {source_col} -> {target_col}: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"  Skipping functional dependency {source_col} -> {target_col}: Source or target column not found in synthetic or original data.")

    print("Finished applying functional dependencies.")
    return synthetic_df

def apply_inequalities(synthetic_df, inequalities):
    """
    Applies inequality relationships (<=, >=) as a post-processing step.
    This is a simple enforcement that might need refinement based on data types.
    """
    print("\nApplying inequality constraints...")

    # Iterate through the inequality relationships identified in the schema
    for col1, rel_list in inequalities.items():
        # Filter for relationships of type value_relationship (inequalities)
        inequality_relationships = [rel for rel in rel_list if rel.get("type") == "value_relationship" and rel.get("relationship") in ['<=', '>=', 'less_than_or_equal_to', 'greater_than_or_equal_to']]

        for rel in inequality_relationships:
            col2 = rel.get("column")
            relation = rel.get("relationship")
            normalized_relation = '<=' if relation in ['<=', 'less_than_or_equal_to'] else '>='

            if col1 in synthetic_df.columns and col2 in synthetic_df.columns:
                 print(f"  Enforcing inequality: {col1} {normalized_relation} {col2}")

                 try:
                     # Ensure columns are numeric for comparison, coerce errors to NaN
                     series1 = pd.to_numeric(synthetic_df[col1], errors='coerce')
                     series2 = pd.to_numeric(synthetic_df[col2], errors='coerce')

                     # Identify rows where the inequality is violated and neither value is NaN
                     if normalized_relation == '<=':
                         violation_mask = (series1 > series2) & series1.notna() & series2.notna()
                     else: # normalized_relation == '>='
                         violation_mask = (series1 < series2) & series1.notna() & series2.notna()

                     if violation_mask.any():
                         print(f"    Found {violation_mask.sum()} violations.")
                         # Simple correction: For violations, adjust one of the values.
                         # We'll adjust the first column (col1) to satisfy the inequality.
                         # This might not be the statistically ideal way, but it enforces the rule.
                         if normalized_relation == '<=':
                             # If col1 > col2, set col1 = col2 (or col2 + small_epsilon for floats)
                             # For simplicity, let's set col1 = col2 for both int and float, assuming types are compatible or will be handled downstream
                             synthetic_df.loc[violation_mask, col1] = series2[violation_mask]
                         else: # normalized_relation == '>='
                             # If col1 < col2, set col1 = col2 (or col2 - small_epsilon for floats)
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

def apply_formulas(synthetic_df, formulas):
    """
    Applies formula relationships (A = B + C) as a post-processing step.
    """
    print("\nApplying formula constraints (A = B + C)...")

    # Iterate through the formula relationships identified in the schema
    for target_col, rel_list in formulas.items():
        # Filter for relationships of type formula
        formula_relationships = [rel for rel in rel_list if rel.get("type") == "formula"]

        for rel in formula_relationships:
            source_cols = rel.get("source_columns")
            formula_str = rel.get("formula") # e.g., "col_b + col_c"

            # Ensure target and all source columns exist in the synthetic DataFrame
            if target_col in synthetic_df.columns and all(src_col in synthetic_df.columns for src_col in source_cols):
                 print(f"  Enforcing formula: {target_col} = {formula_str}")

                 try:
                     # Ensure columns are numeric, coerce errors to NaN
                     target_series = pd.to_numeric(synthetic_df[target_col], errors='coerce')
                     source_series_list = [pd.to_numeric(synthetic_df[src_col], errors='coerce') for src_col in source_cols]

                     # Calculate the expected value based on the formula (assuming simple addition for now)
                     if len(source_series_list) == 2: # Only handle A = B + C for now
                          expected_value = source_series_list[0] + source_series_list[1]
                          # Identify rows where the formula is violated and none of the values are NaN
                          violation_mask = ~np.isclose(target_series, expected_value, atol=FORMULA_TOLERANCE, equal_nan=True) & \
                                           target_series.notna() & source_series_list[0].notna() & source_series_list[1].notna()

                          if violation_mask.any():
                              print(f"    Found {violation_mask.sum()} violations.")
                              # Correction: Set the target column value to the calculated expected value
                              synthetic_df.loc[violation_mask, target_col] = expected_value[violation_mask]
                              print(f"    Corrected {violation_mask.sum()} values in '{target_col}'.")
                         # else: # Handle other formula complexities if needed later
                             # print(f"    Warning: Formula '{formula_str}' for '{target_col}' is not a simple A=B+C format. Skipping enforcement.")

                 except Exception as e:
                     print(f"    Error applying formula {target_col} = {formula_str}: {e}")
                     import traceback
                     traceback.print_exc()
            else:
                print(f"  Skipping formula {target_col} = {formula_str}: Target or source columns not found in synthetic data.")

    print("Finished applying formula constraints.")
    return synthetic_df


# Re-use the detection logic from the schema generation script
def detect_relationships_in_synthetic_data(df):
    """
    Detects column relationships (Pearson, basic functional dependency, value relationships, simple formulas)
     in the synthetic data and returns them as a dictionary.
     This is a duplicate of the function in the schema script for comparison purposes.
    """
    relationships = {} # Initialize relationships as a dictionary
    RELATIONSHIP_CONSISTENCY_THRESHOLD = 0.999 # Define threshold for value relationship consistency (e.g., 99.9%)
    CORRELATION_THRESHOLD = 0.7 # Threshold for reporting correlations (use same as schema script)
    FORMULA_TOLERANCE = 1e-6 # Tolerance for floating-point comparisons (use same as schema script)


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
                     print(f"Detected Pearson correlations above threshold {CORRELATION_THRESHOLD} in synthetic data.")
                else:
                     pass # print("Not enough valid numeric columns after cleaning for Pearson correlation calculation in synthetic data.")

            except Exception as pearson_e:
                 print(f"Error during Pearson correlation calculation for reporting in synthetic data: {pearson_e}")


        # --- Functional Dependency (Simple Check: one value maps to only one other value) ---
        all_cols = df.columns.tolist() # Check dependency across all columns
        print("\nChecking for functional dependencies in synthetic data...")
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
                                                     print(f"Detected potential functional dependency in synthetic data: {col1} -> {col2}")
                    except Exception as dep_e:
                        pass


        # --- Basic Value Relationships (e.g., col1 <= col2, col1 >= col2) ---
        # Check for consistent inequality relationships between numeric columns
        print("\nChecking for basic value relationships (e.g., <=, >=) in synthetic data...")
        if len(numeric_cols) >= 2:
             for col1 in numeric_cols:
                 for col2 in numeric_cols:
                     if col1 != col2:
                         try:
                             df_filtered = df[[col1, col2]].dropna()
                             if not df_filtered.empty:
                                 num_rows_checked = len(df_filtered)

                                 # Check col1 <= col2
                                 if num_rows_checked > 0 and (pd.to_numeric(df_filtered[col1], errors='coerce') <= pd.to_numeric(df_filtered[col2], errors='coerce')).sum() / num_rows_checked >= INEQUALITY_CONSISTENCY_THRESHOLD:
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
                                              print(f"Detected value relationship in synthetic data: {col1} <= {col2} (holds for >= {INEQUALITY_CONSISTENCY_THRESHOLD:.1%} of data)")
                                     # No need to add B >= A explicitly if A <= B is added

                                 # Check col1 >= col2
                                 if num_rows_checked > 0 and (pd.to_numeric(df_filtered[col1], errors='coerce') >= pd.to_numeric(df_filtered[col2], errors='coerce')).sum() / num_rows_checked >= INEQUALITY_CONSISTENCY_THRESHOLD:
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
                                              print(f"Detected value relationship in synthetic data: {col1} >= {col2} (holds for >= {INEQUALITY_CONSISTENCY_THRESHOLD:.1%} of data)")
                                     # No need to add B <= A explicitly if A >= B is added


                         except Exception as rel_e:
                             pass


        # --- Simple Formula Relationships (A = B + C) ---
        print("\nChecking for simple formula relationships (A = B + C) in synthetic data...")
        if len(numeric_cols) >= 3:
             # Iterate through all combinations of 3 unique numeric columns
             for col_a, col_b, col_c in combinations(numeric_cols, 3):
                 try:
                     # Drop rows where any of the three columns are NaN
                     df_filtered = df[[col_a, col_b, col_c]].dropna()
                     if not df_filtered.empty:
                         num_rows_checked = len(df_filtered)

                         # Ensure columns are numeric
                         series_a = pd.to_numeric(df_filtered[col_a], errors='coerce')
                         series_b = pd.to_numeric(df_filtered[col_b], errors='coerce')
                         series_c = pd.to_numeric(df_filtered[col_c], errors='coerce')

                         # Check if col_a = col_b + col_c
                         if num_rows_checked > 0 and np.allclose(series_a, series_b + series_c, atol=FORMULA_TOLERANCE, equal_nan=True):
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
                                  print(f"Detected formula relationship in synthetic data: {col_a} = {formula}")

                         # Check if col_b = col_a + col_c
                         if num_rows_checked > 0 and np.allclose(series_b, series_a + series_c, atol=FORMULA_TOLERANCE, equal_nan=True):
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
                                  print(f"Detected formula relationship in synthetic data: {col_b} = {formula}")

                         # Check if col_c = col_a + col_b
                         if num_rows_checked > 0 and np.allclose(series_c, series_a + series_b, atol=FORMULA_TOLERANCE, equal_nan=True):
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
                                  print(f"Detected formula relationship in synthetic data: {col_c} = {formula}")

                 except Exception as formula_e:
                     pass # Suppress errors during formula check


    except Exception as e:
        print(f"Error detecting relationships for reporting in synthetic data: {str(e)}")
        import traceback
        traceback.print_exc()

    # Clean up empty relationship lists (columns with no detected relationships)
    relationships = {col: rels for col, rels in relationships.items() if rels}

    return relationships

def compare_relationships(original_relationships, synthetic_relationships):
    """
    Compares relationships detected in original and synthetic data and returns a DataFrame.
    """
    print("\nComparing relationships...")
    comparison_data = []

    # Collect all unique relationships from both original and synthetic data
    all_relationships = {}

    # Add original relationships
    for col, rel_list in original_relationships.items():
        for rel in rel_list:
            # Create a unique key for each relationship for comparison
            rel_key = json.dumps(rel, sort_keys=True) # Use JSON string as a unique key
            if rel_key not in all_relationships:
                all_relationships[rel_key] = {"original": True, "synthetic": False, "details": rel}
            else:
                all_relationships[rel_key]["original"] = True # Mark as present in original

    # Add synthetic relationships
    for col, rel_list in synthetic_relationships.items():
        for rel in rel_list:
            rel_key = json.dumps(rel, sort_keys=True)
            if rel_key not in all_relationships:
                # This case should ideally not happen if synthetic relationships perfectly match original,
                # but include it for robustness if synthetic data introduces new patterns.
                all_relationships[rel_key] = {"original": False, "synthetic": True, "details": rel}
            else:
                all_relationships[rel_key]["synthetic"] = True # Mark as present in synthetic

    # Prepare data for the comparison DataFrame
    for rel_key, status in all_relationships.items():
        rel_details = status["details"]
        row = {
            "Relationship Type": rel_details.get("type"),
            "Original Data": status["original"],
            "Synthetic Data": status["synthetic"]
        }

        # Add specific details based on relationship type
        if rel_details.get("type") == "pearson_correlation":
            row["Column 1"] = rel_details.get("column") # Source column in schema dict
            # Find the other column in the relationship details
            other_col = None
            # Pearson correlation is listed for both columns, find the one that's not the key
            for original_col, original_rel_list in original_relationships.items():
                 for original_rel in original_rel_list:
                      if original_rel.get("type") == "pearson_correlation" and original_rel.get("column") == rel_details.get("column") and original_col != rel_details.get("column"):
                           other_col = original_col
                           break
                 if other_col: break # Found the other column

            row["Column 2"] = other_col if other_col else rel_details.get("column") # Fallback if not found
            row["Correlation Value"] = rel_details.get("correlation")
            row["Details"] = f"{row['Column 1']} - {row['Column 2']}" # Simplified detail
        elif rel_details.get("type") == "functional_dependency":
            row["Source Column"] = rel_details.get("column") # Source column in schema dict
            row["Target Column"] = rel_details.get("target_column") # Target column in schema dict
            # Functional dependency is listed as source -> target, so column is source, target_column is target
            # Let's correct this based on how detect_relationships stores it:
            # detect_relationships stores {source_col: [{"column": target_col, ...}]}
            # So, rel_details.get("column") is the target column in the list item
            # The key in the relationships dictionary is the source column.
            source_col_key = None
            for original_col, original_rel_list in original_relationships.items():
                 for original_rel in original_rel_list:
                      if original_rel.get("type") == "functional_dependency" and original_rel.get("column") == rel_details.get("column"):
                           source_col_key = original_col
                           break
                 if source_col_key: break

            row["Source Column"] = source_col_key if source_col_key else "N/A"
            row["Target Column"] = rel_details.get("column") # This is the target column in the list item
            row["Details"] = f"{row['Source Column']} -> {row['Target Column']}"
        elif rel_details.get("type") == "value_relationship":
             row["Column 1"] = rel_details.get("column") # Source column in schema dict
             row["Column 2"] = rel_details.get("target_column") # Target column in schema dict
             row["Relationship"] = rel_details.get("relationship") # <= or >=
             # Correcting based on detect_relationships storage: {col1: [{"column": col2, "relationship": "<=", ...}]}
             source_col_key = None
             for original_col, original_rel_list in original_relationships.items():
                  for original_rel in original_rel_list:
                       if original_rel.get("type") == "value_relationship" and original_rel.get("column") == rel_details.get("column") and original_rel.get("relationship") == rel_details.get("relationship"):
                            source_col_key = original_col
                            break
                  if source_col_key: break

             row["Column 1"] = source_col_key if source_col_key else "N/A"
             row["Column 2"] = rel_details.get("column") # This is col2 in the list item
             row["Relationship"] = rel_details.get("relationship")
             row["Details"] = f"{row['Column 1']} {row['Relationship']} {row['Column 2']}"
        elif rel_details.get("type") == "formula":
            row["Target Column"] = rel_details.get("target_column")
            row["Source Columns"] = ", ".join(rel_details.get("source_columns", []))
            row["Formula"] = rel_details.get("formula")
            row["Details"] = f"{row['Target Column']} = {row['Formula']}"
        else:
            row["Details"] = json.dumps(rel_details) # Fallback for unknown types

        comparison_data.append(row)

    # Create DataFrame
    comparison_df = pd.DataFrame(comparison_data)

    # Define column order for better readability
    ordered_columns = [
        "Relationship Type",
        "Details",
        "Original Data",
        "Synthetic Data",
        "Column 1", # For Pearson, Value
        "Column 2", # For Pearson, Value
        "Correlation Value", # For Pearson
        "Source Column", # For Functional Dependency
        "Target Column", # For Functional Dependency, Formula
        "Relationship", # For Value
        "Source Columns", # For Formula
        "Formula" # For Formula
    ]
    # Select and reorder columns, dropping ones that don't exist in this comparison
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
    """Main function to generate synthetic data using Faker and post-processing."""
    print("Starting synthetic data generation process (Faker + Post-processing).")

    # 1. Load data and schema
    original_data = load_data(INPUT_CSV)
    if original_data is None:
        print("Failed to load input data. Exiting.")
        return

    schema = load_schema(SCHEMA_JSON)
    if schema is None:
        print("Failed to load schema. Exiting.")
        return

    # Extract relationships from the schema (these are the relationships in the original data)
    original_relationships = schema.get("relationships", {})
    print(f"Loaded {len(original_relationships)} columns with detected relationships from schema.")

    # 2. Generate initial synthetic data using Faker
    synthetic_data = generate_initial_data_with_faker(schema, NUM_SYNTHETIC_ROWS, original_data)
    if synthetic_data is None:
        print("Failed to generate initial synthetic data. Exiting.")
        return

    # 3. Apply Post-processing for relationships
    # Functional Dependencies
    functional_dependencies_to_apply = {
        col_name: [rel for rel in rel_list if rel.get("type") == "functional_dependency"]
        for col_name, rel_list in original_relationships.items()
    }
    functional_dependencies_to_apply = {
        col: rels for col, rels in functional_dependencies_to_apply.items() if rels
    }
    if functional_dependencies_to_apply:
         synthetic_data = apply_functional_dependencies(synthetic_data, original_data, functional_dependencies_to_apply)
    else:
         print("No functional dependencies found in schema to apply.")

    # Inequalities (Value Relationships)
    inequalities_to_apply = {
        col_name: [rel for rel in rel_list if rel.get("type") == "value_relationship"]
        for col_name, rel_list in original_relationships.items()
    }
    inequalities_to_apply = {
        col: rels for col, rels in inequalities_to_apply.items() if rels
    }
    if inequalities_to_apply:
        synthetic_data = apply_inequalities(synthetic_data, inequalities_to_apply)
    else:
        print("No inequality relationships found in schema to apply.")

    # Formulas (Additive)
    formulas_to_apply = {
        col_name: [rel for rel in rel_list if rel.get("type") == "formula"]
        for col_name, rel_list in original_relationships.items()
    }
    formulas_to_apply = {
        col: rels for col, rels in formulas_to_apply.items() if rels
    }
    if formulas_to_apply:
         synthetic_data = apply_formulas(synthetic_data, formulas_to_apply)
    else:
         print("No formula relationships found in schema to apply.")


    # Pearson correlations are not strictly enforced in this post-processing,
    # but the base generation using Faker and the enforcement of other relationships
    # might indirectly influence correlations.

    # 4. Save synthetic data
    save_data(synthetic_data, OUTPUT_SYNTHETIC_CSV)

    # 5. Detect relationships in synthetic data for comparison
    print("\nDetecting relationships in the generated synthetic data...")
    synthetic_relationships = detect_relationships_in_synthetic_data(synthetic_data)
    print(f"Detected {len(synthetic_relationships)} columns with relationships in synthetic data.")


    # 6. Compare relationships and save comparison to CSV
    relationship_comparison_df = compare_relationships(original_relationships, synthetic_relationships)
    save_data(relationship_comparison_df, OUTPUT_RELATIONSHIP_COMPARISON_CSV)


    print("\nSynthetic data generation process finished.")

if __name__ == "__main__":
    main()
