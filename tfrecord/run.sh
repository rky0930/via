python tfrecord_to_via.py --tfrecord_file=/home/yoon/Downloads/yolo/testC.record \
       --img_dir=image \
       --annotation_file=via_annotation_a.json \
       --label_map_file=/home/yoon/Downloads/yolo/detection.pbtxt \
       --max_read=10

python via_json_create_tf_record.py --image_dir=tmp/JPEGImages \
  --annotation_file=tmp/via_region_data_coco.json \
  --label_map_file=$HOME/workspace/git_rky0930/models/research/object_detection/data/mscoco_label_map.pbtxt \
  --output=tmp/coco_overfit.record \
  --t_v_ratio=1

python via_tfrecord.py \
  --tfrecord_file= \
  --image_dir= \
  --label_map_file= \
  --annotation_file= \