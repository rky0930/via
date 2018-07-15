#-*- coding:utf-8 -*-

import pandas as pd
from yaml import load
import json

# read
df = pd.read_csv('train/annotations.csv')

# main dict
doc = {}

# Iterate through dataframe
df_iterator = df.iterrows()
for index, row in df_iterator:

    # Get data from each row
    filename     = row['#filename']
    file_size    = row['file_size']
    region_count = row['region_count']
    region_id    = row['region_id']
    file_attributes         = load(row['file_attributes'])
    region_shape_attributes = load(row['region_shape_attributes'])
    region_attributes       = load(row['region_attributes'])
    
    image_id = filename+str(file_size)
    
    # Put the data into main dict(doc)
    doc[image_id] = {}
    doc[image_id]['fileref'] = ""
    doc[image_id]['size'] = file_size
    doc[image_id]['filename'] = filename
    doc[image_id]['base64_img_data'] = ""
    doc[image_id]['file_attributes'] = file_attributes
    doc[image_id]['regions'] = dict()

    # If there is region,
    if region_count > 0:     
        doc[image_id]['regions'][region_id] = {
            "shape_attributes": region_shape_attributes,
            "region_attributes": region_attributes
        }
            
        # If there is more than one region,
        for _ in range(1, region_count):
            sub_index, sub_row = df_iterator.next()
            
            sub_filename  = sub_row['#filename']
            sub_file_size = sub_row['file_size']
            sub_region_id = sub_row['region_id']
            
            sub_image_id  = sub_filename+str(sub_file_size)

            # image_id and region_id check
            if image_id == sub_image_id and region_id != sub_region_id: 
                
                sub_region_shape_attributes = load(sub_row['region_shape_attributes'])
                sub_region_attributes       = load(sub_row['region_attributes'])

                doc[image_id]['regions'][sub_region_id] = {
                    "shape_attributes": sub_region_shape_attributes,
                    "region_attributes": sub_region_attributes
                }
    
            else: # Error check
                print('Error: sub_image_id != image_id')    

# Save to file
output_filename = 'result.json'
with open(output_filename, 'w') as fp:
    json.dump(doc, fp, ensure_ascii=False)
