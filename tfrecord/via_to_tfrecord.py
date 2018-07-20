# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os

import pandas as pd
import yaml
import json

import numpy as np
import PIL.Image
import tensorflow as tf

from utils import dataset_util
from utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('image_dir', 'JPEGImages', 'Image directory for raw dataset')
flags.DEFINE_string('annotation_file', 'via_region_data.json', 'Annotation file for raw dataset')
flags.DEFINE_string('output', 'output', 'Path to output TFRecord ex) tmp: output file = tmp_train, tmp_val')
flags.DEFINE_float('train_val_ratio', '1.0', 'Training & Validation set ratio. ex) 0.9 => training:validation = 9:1')
flags.DEFINE_string('label_map_file', 'label_map.pbtxt', 'Path to label map proto')
flags.DEFINE_string('category_name', 'display_name', 'Category name in annotation file. ex) display_name, name')
FLAGS = flags.FLAGS

def dict_to_tf_example(data,
                       image_dir,
                       label_map_dict):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: Annoation data of single image
    image_dir: Path to root directory holding images
    label_map_dict: A map from string label names to integers ids.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  filename = data['filename'].encode('utf8')
  full_path = os.path.join(image_dir, filename) 
  with tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()


  image_width, image_height = image.size
  image_width = int(image_width)
  image_height= int(image_height)

  ## regions
  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  
  for region_id, attribute in data['regions'].iteritems():
    try:
      shape = attribute['shape_attributes']
      region_x = shape['x']
      region_y = shape['y']
      region_xmax = region_x + shape['width']
      region_ymax = region_y + shape['height']
    
      xmin.append(region_x / image_width)
      ymin.append(region_y / image_height)
      xmax.append(region_xmax / image_width)
      ymax.append(region_ymax / image_height)

      region_attributes = attribute['region_attributes']  
      category_name = FLAGS.category_name
      #classes_text.append(region_attributes[category_name])
      classes.append(label_map_dict[region_attributes[category_name].encode('utf8')])
    except KeyError as e:
      if len(attribute['region_attributes']) == 0: 
        print("No categery(image: %s, region: %s)" % (filename, region_id))
      else: 
        print(e)
    
  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(image_height),
      'image/width': dataset_util.int64_feature(image_width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      #'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return example


def main(_):

  image_dir       = FLAGS.image_dir
  annotation_path = FLAGS.annotation_file
  annotation_fid  = open(annotation_path)

  output_dir  = os.path.dirname (FLAGS.output)
  output_name = os.path.basename(FLAGS.output)
  output_name, output_ext  = os.path.splitext(output_name)

  with open(annotation_path) as annotation_fid:

    annotations     = yaml.safe_load(annotation_fid)
    annotation_size = len(annotations)
    
    # Shuffle
    annotation_keys = annotations.keys()
    np.random.shuffle(annotation_keys)

    # Train:Validation
    training_size = int(annotation_size * FLAGS.train_val_ratio)
    training_data    = annotation_keys[:training_size]
    validataion_data = annotation_keys[training_size:]

    use_display_name = True if FLAGS.category_name == 'display_name' else False
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_file, use_display_name=use_display_name) 
          
    dataset = [
      ('_train', training_data),
      ('_val', validataion_data),
    ]

    for suffix, data_list in dataset:
      if not data_list: # Data is empty
        continue

      output = os.path.join(output_dir, output_name + suffix + output_ext)
      with tf.python_io.TFRecordWriter(output) as writer:
        for idx, key in enumerate(data_list):

          # Print Status
          if idx % 100 == 0:
            print('On image %d of %d'% (idx, training_size))

          tf_example = dict_to_tf_example(annotations[key], image_dir, label_map_dict)
          writer.write(tf_example.SerializeToString())
    print("Success. ")
if __name__ == '__main__':
  tf.app.run()
