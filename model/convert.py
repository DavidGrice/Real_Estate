import glob
import pandas as pd
import os
import openpyxl

# # Get a list of all .json files in the current directory and subdirectories
# json_files = glob.glob('**/*.json', recursive=True)

# # Define the name of the output CSV file
# output_file = 'output.csv'

# # Loop through the list of files
# for file in json_files:
#     # Read the JSON file into a DataFrame
#     df = pd.read_json(file)

#     # Check if the CSV file already exists
#     if os.path.isfile(output_file):
#         # If the file exists, do not write the header
#         df.to_csv(output_file, mode='a', index=False, header=False)
#     else:
#         # If the file does not exist, write the header
#         df.to_csv(output_file, mode='a', index=False)

output_file = 'realtorData.csv'
# # Read the CSV file into a DataFrame
df = pd.read_csv(output_file)

# Assuming the addresses are in a column named 'address' in the DataFrame df
df['zipcode'] = df['address3'].str.extract(r'(\d{5})')

# Write the DataFrame to an Excel file
with pd.ExcelWriter('output.xlsx') as writer:
    df.to_excel(writer, index=False, sheet_name='Sheet1')

    # Write the zipcode to its own sheet
    df['zipcode'].to_excel(writer, index=False, sheet_name='Zipcodes')