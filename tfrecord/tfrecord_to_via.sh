python tfrecord_to_via.py \
       --tfrecord_file=example/spaceX_train.record \
       --image_dir=example/from_tfrecord/JPEGImages \
       --annotation_file=example/from_tfrecord/via_annotation.json \
       --label_map_file=example/spaceX_label_map.pbtxt
