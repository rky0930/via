# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert raw PASCAL dataset to TFRecord for object_detection.

Example usage:
python via_json_create_tf_record.py --image_dir=tmp/JPEGImages \
  --annotation_file=tmp/via_region_data_coco.json \
  --label_map_file=$HOME/workspace/git_rky0930/models/research/object_detection/data/mscoco_label_map.pbtxt \
  --output=tmp/coco_overfit.record \
  --t_v_ratio=1

"""
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

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('image_dir', 'JPEGImages', 'Image directory for raw dataset.')
flags.DEFINE_string('annotation_file', 'via_region_data.json', 'Annotation file for raw dataset.')
flags.DEFINE_string('output', 'output', 'Path to output TFRecord ex) tmp: output file = tmp_t, tmp_v')
flags.DEFINE_float('t_v_ratio', '0.9', 'Training & Validation set ratio. ex) 0.9 => training:validation = 9:1')
flags.DEFINE_string('label_map_file', 'label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_boolean('use_display_name', False, 'whether to use the label map items\' display names as keys.')
flags.DEFINE_string('region_attributes_key', 'id', 'Key value of region_attribute for annotation file.')
FLAGS = flags.FLAGS

def dict_to_tf_example(data,
                       image_dir,
                       label_map_dict):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    image_dir: Path to root directory holding PASCAL dataset
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
    r_key = FLAGS.region_attributes_key
    classes_text.append(region_attributes[r_key].encode('utf8'))
    classes.append(label_map_dict[region_attributes[r_key]])
  
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
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
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

    annotations     = json.load(annotation_fid)
    annotation_size = len(annotations)
    
    # Shuffle
    annotation_keys = annotations.keys()
    np.random.shuffle(annotation_keys)

    # Train:Validation (T:V)
    training_size = int(annotation_size * FLAGS.t_v_ratio)
    training_data    = annotation_keys[:training_size]
    validataion_data = annotation_keys[training_size:]

    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_file, use_display_name=FLAGS.use_display_name) 
          
    dataset = [
      ('_t', training_data),
      ('_v', validataion_data),
    ]

    for suffix, data_list in dataset:
      if not data_list: # Data is empty
        continue

      output = os.path.join(output_dir, output_name + suffix + output_ext)
      print(output)
      with tf.python_io.TFRecordWriter(output) as writer:
        for idx, key in enumerate(data_list):

          # Print Status
          if idx % 100 == 0:
            print('On image %d of %d'% (idx, training_size))
            
          tf_example = dict_to_tf_example(annotations[key], image_dir, label_map_dict)
          writer.write(tf_example.SerializeToString())

if __name__ == '__main__':
  tf.app.run()
