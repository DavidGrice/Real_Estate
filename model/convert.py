import glob
import pandas as pd
import os
import openpyxl

# json_files = glob.glob('**/*.json', recursive=True)
# output_file = 'output.csv'
# for file in json_files:
#     df = pd.read_json(file)
#     if os.path.isfile(output_file):
#         df.to_csv(output_file, mode='a', index=False, header=False)
#     else:
#         df.to_csv(output_file, mode='a', index=False)

output_file = 'realtorData.csv'

df = pd.read_csv(output_file)

df['zipcode'] = df['address3'].str.extract(r'(\d{5})')

with pd.ExcelWriter('output.xlsx') as writer:
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    df['zipcode'].to_excel(writer, index=False, sheet_name='Zipcodes')
