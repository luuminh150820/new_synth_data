import pandas as pd
import json
import numpy as np
from faker import Faker
import os
import re
from collections import defaultdict
import random
from datetime import datetime, timedelta

class SyntheticDataGenerator:
    def __init__(self, schema_path, csv_path, num_rows=1000):
        """
        Initialize the synthetic data generator.
        
        Args:
            schema_path (str): Path to the enhanced schema JSON file
            csv_path (str): Path to the original CSV data file
            num_rows (int): Number of synthetic rows to generate
        """
        self.schema_path = schema_path
        self.csv_path = csv_path
        self.NUM_ROWS = num_rows
        self.faker = Faker()
        
        # Load schema and original data
        self.schema = self._load_json(schema_path)
        self.original_data = self._load_csv(csv_path)
        
        # Initialize variables
        self.relationships = {}
        self.fd_relationships = {}
        self.o2o_relationships = {}
        self.value_relationships = {}
        self.temporal_relationships = {}
        self.column_unique_counts = {}
        self.synthetic_data = {}
        self.column_unique_values = {}
        
        # Extract column order from original data
        self.column_order = list(self.original_data.columns)
        
        print(f"Initialized generator with {len(self.schema['columns'])} columns")
        print(f"Original data shape: {self.original_data.shape}")
        print(f"Target synthetic rows: {self.NUM_ROWS}")
        
    def _load_json(self, file_path):
        """Load and parse JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise Exception(f"Failed to load schema from {file_path}: {str(e)}")
            
    def _load_csv(self, file_path):
        """Load CSV file into DataFrame"""
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise Exception(f"Failed to load data from {file_path}: {str(e)}")
            
    def step0_map_relationships(self):
        """
        Step 0: Sort through relationships and create mappings.
        
        Returns:
            dict: Organized relationships by type
        """
        print("\n--- STEP 0: Mapping Relationships ---")
        
        # Process relationships from schema
        for col_name, col_info in self.schema['columns'].items():
            post_processing_rules = col_info.get('post_processing_rules', [])
            
            # Store unique counts for later use
            self.column_unique_counts[col_name] = col_info['stats'].get('unique_count', 0)
            
            for rule in post_processing_rules:
                related_col = rule.get('column')
                rel_type = rule.get('type')
                
                # Skip if missing information
                if not related_col or not rel_type:
                    continue
                
                # Store in appropriate relationship container
                if rel_type == 'functional_dependency':
                    if col_name not in self.fd_relationships:
                        self.fd_relationships[col_name] = []
                    self.fd_relationships[col_name].append(related_col)
                    print(f"  FD: {col_name} -> {related_col}")
                    
                elif rel_type == 'one_to_one':
                    if col_name not in self.o2o_relationships:
                        self.o2o_relationships[col_name] = []
                    self.o2o_relationships[col_name].append(related_col)
                    print(f"  O2O: {col_name} <-> {related_col}")
                    
                elif rel_type == 'value_relationship':
                    relationship = rule.get('relationship')
                    if not relationship:
                        continue
                    
                    key = f"{col_name}_{related_col}"
                    self.value_relationships[key] = {
                        'source': col_name,
                        'target': related_col,
                        'relationship': relationship
                    }
                    print(f"  Value: {col_name} {relationship} {related_col}")
                    
                elif rel_type == 'temporal_relationship':
                    relationship = rule.get('relationship')
                    if not relationship:
                        continue
                    
                    key = f"{col_name}_{related_col}"
                    self.temporal_relationships[key] = {
                        'source': col_name,
                        'target': related_col,
                        'relationship': relationship
                    }
                    print(f"  Temporal: {col_name} {relationship} {related_col}")
        
        # Combine all relationships
        self.relationships = {
            'fd': self.fd_relationships,
            'o2o': self.o2o_relationships,
            'value': self.value_relationships,
            'temporal': self.temporal_relationships
        }
        
        # Print summary
        print(f"  Found {len(self.fd_relationships)} columns with FD relationships")
        print(f"  Found {len(self.o2o_relationships)} columns with O2O relationships")
        print(f"  Found {len(self.value_relationships)} value relationships")
        print(f"  Found {len(self.temporal_relationships)} temporal relationships")
        
        return self.relationships
    
    def _get_faker_provider(self, column_name):
        """Get the appropriate Faker provider for a column"""
        col_info = self.schema['columns'].get(column_name, {})
        provider_name = col_info.get('faker_provider')
        faker_args = col_info.get('faker_args', {})
        
        if not provider_name:
            return lambda: None

        if provider_name == 'random_int':
            provider_name = 'random_number'  
        if provider_name == 'random_float':
            provider_name = 'pyfloat' 
        if provider_name == 'date_time_between_dates':
            provider_name = 'date_time_this_century'
        # Handle unique providers
        is_unique = False
        if provider_name.startswith('unique.'):
            is_unique = True
            provider_name = provider_name[7:]  # Remove 'unique.' prefix
            
        if provider_name == 'regexify':
            pattern = faker_args.get('pattern')
            if pattern is None:
                print(f"Warning: 'regexify' provider specified for {column_name} but no 'pattern' found in faker_args. Returning None provider.")
                return lambda: None # Return a provider that gives None if no pattern
            print(f"  Using custom regex generator for {column_name} with pattern: {pattern}")
            return lambda: generate_from_regex_pattern(pattern) # Return function that calls your custom regex generator



        try:
            # Get the faker method
            if '.' in provider_name:
                # Handle nested providers like 'person.name'
                provider_parts = provider_name.split('.')
                provider_obj = self.faker
                for part in provider_parts:
                    provider_obj = getattr(provider_obj, part)
                provider_method = provider_obj
            else:
                provider_method = getattr(self.faker, provider_name)
                
            # Create a function that calls the provider with the right args
            def provider_func():
                try:
                    return provider_method(**faker_args)
                except Exception as e:
                    print(f"Error using provider {provider_name} for {column_name}: {str(e)}")
                    # Fallback based on data type
                    data_type = col_info.get('data_type', 'text')
                    if data_type == 'integer':
                        return random.randint(0, 1000)
                    elif data_type == 'float':
                        return random.random() * 1000
                    elif data_type == 'datetime':
                        return self.faker.date_time_this_decade()
                    else:
                        return f"unknown_{column_name}_{random.randint(1, 1000)}"
            
            return provider_func
            
        except Exception as e:
            print(f"Failed to get provider for {column_name} ({provider_name}): {str(e)}")
            # Return a simple default provider
            return lambda: f"error_{column_name}_{random.randint(1, 1000)}"
            
    def _generate_unique_values(self, column_name, target_count):
        """Generate unique values for a column"""
        provider = self._get_faker_provider(column_name)
        unique_values = set()
        
        # Cap at 10,000 attempts to prevent infinite loops
        max_attempts = 10000
        attempts = 0
        
        while len(unique_values) < target_count and attempts < max_attempts:
            value = provider()
            unique_values.add(value)
            attempts += 1
            
        if attempts >= max_attempts and len(unique_values) < target_count:
            print(f"  Warning: Could only generate {len(unique_values)}/{target_count} unique values for {column_name}")
            
        return list(unique_values)
    
    def step1_generate_initial_uniques(self):
        """
        Step 1: Generate initial unique values for each column.
        
        Returns:
            dict: Column unique values
        """
        print("\n--- STEP 1: Generating Initial Unique Values ---")
        
        # Calculate scaling factor for columns with >20 unique values
        scaling_factor = len(self.original_data) / self.NUM_ROWS
        print("SCALING FACTOR__________________________________________________")
        print(scaling_factor)

        for column_name in self.schema['columns']:
            # Get unique count from schema
            unique_count = self.column_unique_counts.get(column_name, 0)
            
            # Skip if no unique values needed
            if unique_count == 0:
                self.column_unique_values[column_name] = []
                continue
                
            # Determine target unique count
            is_categorical = unique_count <= 20
            if is_categorical:
                target_count = min(unique_count,self.NUM_ROWS)
            else:
                # Scale unique count based on target row count
                target_count = max(1, int(unique_count / scaling_factor))
                
            print(f"  Generating {target_count} unique values for {column_name} (original unique: {unique_count})")
            
            # Generate unique values
            unique_values = self._generate_unique_values(column_name, target_count)
            self.column_unique_values[column_name] = unique_values
            
            print(f"  Generated {len(unique_values)} unique values for {column_name}")
            
        # Summary
        print(f"  Generated unique values for {len(self.column_unique_values)} columns")
        return self.column_unique_values
    
    def save_synthetic_data(self, output_path):
        """
        Save synthetic data to CSV file.
        
        Args:
            output_path (str): Path to save the output CSV
        """
        # For now, just save the unique values since we haven't generated full data yet
        result = {}
        
        # Create a DataFrame with column unique values
        for column_name, unique_values in self.column_unique_values.items():
            # Pad with None to match NUM_ROWS
            padded_values = unique_values + [None] * (self.NUM_ROWS - len(unique_values))
            result[column_name] = padded_values[:self.NUM_ROWS]
            
        # Convert to DataFrame
        result_df = pd.DataFrame(result)
        
        # Reorder columns to match original data
        ordered_cols = [col for col in self.column_order if col in result_df.columns]
        result_df = result_df[ordered_cols]
        
        # Save to CSV
        result_df.to_csv(output_path, index=False)
        print(f"\nSaved synthetic data to {output_path}")
        print(f"Shape: {result_df.shape}")
        
        return result_df

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
                  print(f"Warning: Trailing '\\\\' in regex pattern: {pattern}. Ignoring.")
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

def main():
    """Main function to run the synthetic data generator"""
    #import argparse
    
    #parser = argparse.ArgumentParser(description='Generate synthetic data based on schema')
    #parser.add_argument('--schema', required=True, help='Path to schema JSON file')
    #parser.add_argument('--csv', required=True, help='Path to original CSV file')
    #parser.add_argument('--output', default='synthetic_data.csv', help='Output CSV path')
    #parser.add_argument('--rows', type=int, default=1000, help='Number of synthetic rows to generate')

    schema = 'enhanced_schema.json'
    csv = 'customer_data.csv'
    #args = parser.parse_args()
    rows = 2
    output = 'synth.csv'
    
    # Initialize generator
    print("lets start")
    generator = SyntheticDataGenerator(schema, csv, rows)
    
    # Run steps
    generator.step0_map_relationships()
    generator.step1_generate_initial_uniques()
    
    # Save results
    generator.save_synthetic_data(output)
    
if __name__ == "__main__":
    main()