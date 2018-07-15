import tensorflow as tf
import json
import yaml
import os
from string import maketrans

"""
To run: 
  python tfrecord_to_via.py \ 
        --tfrecord_file=<<record_file_path>> \
        --img_dir=<<img_dir_path>> \
        --annotation_file=<<annotation_file_path>> \
        --label_map_file=mscoco_label_map.pbtxt \
        --max_read=10

"""

flags = tf.app.flags
flags.DEFINE_string('tfrecord_file', None, 'rocord file path.')
flags.DEFINE_string('img_dir', 'img_dir', 'Image file output path')
flags.DEFINE_string('annotation_file', 'result.json', 'Annotation file output path. Default is result.json.')
flags.DEFINE_string('label_map_file', None, 'label_mapfile path.')
flags.DEFINE_string('region_shape', 'rect', 'Via Region shape. Default is "rect".')
flags.DEFINE_integer('max_read', None, 'Maximum number of image to extract')
flags.DEFINE_boolean('with_label', True, 'Extract with labels')
FLAGS = flags.FLAGS

doc = {}
def convert_to_height_width(ymin, xmin, ymax, xmax):
  height = abs(ymax - ymin)
  width = abs(xmax - xmin)
  return height, width

def get_tfrecords_feature_list(tfrecord_file):
    ptr = 0
    record_iterator = tf.python_io.tf_record_iterator(path=tfrecord_file)

    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        return example.features.feature.keys()

    return []

def label_map_pbtxt_to_dict(label_map_path):

  # Read <<label_map.pb.txt>>
  label_map = None
  with open(label_map_path, "r") as label_map_fid:
    label_map = label_map_fid.read()

  # Parsing to dict
  tbl = maketrans('}', ' ')
  label_map = label_map.translate(tbl)

  items = label_map.split('item {')
  tbl = maketrans('\n', ',')
  doc = []
  default_attribute = {}
  for item in items:
    try:
      item = item.strip()
      item = item.translate(tbl)
      item_json= yaml.load('{'+item+'}')
      name_key = 'display_name' if 'display_name' in item_json.keys() else 'name'
      default_attribute[item_json['id']] = item_json[name_key]
    except KeyError:
      continue
  
  print('<<INSERT_VIA_DEFAULT_ATTRIBUTE>>')
  print('{ "id": '),
  print([display_name for display_name in default_attribute.values()]),
  print('}')
  return default_attribute
  

def read_tfrecords(tfrecord_file, label_map, max_read, with_label=True):

  # Read tfrecord file
  record_iterator = tf.python_io.tf_record_iterator(path=tfrecord_file)
  for idx, string_record in enumerate(record_iterator):

    example = tf.train.Example()
    example.ParseFromString(string_record)

    img = example.features.feature['image/encoded'].bytes_list.value[0]
    img_size = len(img)
    ymins = example.features.feature['image/object/bbox/ymin'].float_list.value
    xmins = example.features.feature['image/object/bbox/xmin'].float_list.value
    ymaxes = example.features.feature['image/object/bbox/ymax'].float_list.value
    xmaxes = example.features.feature['image/object/bbox/xmax'].float_list.value
    img_width = example.features.feature['image/width'].int64_list.value[0]
    img_height = example.features.feature['image/height'].int64_list.value[0]
    img_format = example.features.feature['image/format'].bytes_list.value
    img_name = '{}.{}'.format(idx, img_format[0])

    # For file attibute 
    file_attributes = {}
    if with_label:
      labels = (example.features.feature['image/object/class/label'].int64_list.value)
      if len(xmins) == 0 and len(labels) == 1:
        file_attributes['id'] = labels[0]

    # Write data to Via annotation form(json) 
    image_id = img_name+str(img_size)
    doc[image_id] = {}
    doc[image_id]['fileref'] = ""
    doc[image_id]['size'] = img_size
    doc[image_id]['filename'] = img_name
    doc[image_id]['base64_img_data'] = ""
    doc[image_id]['regions'] = dict()
    doc[image_id]['file_attributes'] = file_attributes

    region_id = 0
    for ymin, xmin, ymax, xmax, label in zip(ymins, xmins, ymaxes, xmaxes, labels):
      height, width = convert_to_height_width(ymin, xmin, ymax, xmax)
      doc[image_id]['regions'][region_id] = {
        "shape_attributes": {
        'name': FLAGS.region_shape,
        'y': ymin * img_height,
        'x': xmin * img_width,
        'height': height * img_height,
        'width': width * img_width,
        },
        "region_attributes": {
          ## Display id as name. 
          'id': label_map[label]
        }
      }
      region_id += 1
  
    # Write image. If the dir not exist, create dir
    img_path = os.path.join(FLAGS.img_dir, img_name)
    if not os.path.exists(os.path.dirname(img_path)):
      try:
          os.makedirs(os.path.dirname(img_path))
      except OSError as exc: # Guard against race condition
          if exc.errno != errno.EEXIST:
              raise

    with open(img_path, 'wb') as f_img:
      f_img.write(img)

    # Max Read
    if max_read is not None and idx > max_read:
        break

def main(_):

  if FLAGS.tfrecord_file is None:
    raise ValueError("Record file is None.")
  if FLAGS.label_map_file is None:
    raise ValueError("label_map.pbtxt path is None.")
  
  # Check tfrecord Feature 
  print('-- Check tfrecord Feature --')
  tfrecord_features = get_tfrecords_feature_list(FLAGS.tfrecord_file)
  tfrecord_features.sort()
  for tfrecord_feature in tfrecord_features:
    print(tfrecord_feature)
  print('- - - -')

  # Read Label Map
  label_map = label_map_pbtxt_to_dict(FLAGS.label_map_file)

  # Read annotation
  read_tfrecords(FLAGS.tfrecord_file, label_map, FLAGS.max_read, with_label=FLAGS.with_label)
  # Save to file
  with open(FLAGS.annotation_file, 'w') as annotation_fid:
    json.dump(doc, annotation_fid, ensure_ascii=False)


if __name__ == "__main__":
    tf.app.run()
