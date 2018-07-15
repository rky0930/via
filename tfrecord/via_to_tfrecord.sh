python via_to_tfrecord.py \
  --image_dir=example/JPEGImages \
  --annotation_file=example/via_region_data.json \
  --label_map_file=example/spaceX_label_map.pbtxt \
  --output=example/spaceX.record \
  --t_v_ratio=1
