export DATA_DIR=/home/yoon/workspace/dataset/mscoco

python tfrecord_to_via.py --tfrecord_file=${DATA_DIR}/mscoco_trainval.record \
       --img_dir=mscoco_trainval/images \
       --annotation_file=mscoco_trainval/via_annotation.json \
       --label_map_file=${TF_MODELS}/object_detection/data/mscoco_label_map.pbtxt

python tfrecord_to_via.py --tfrecord_file=${DATA_DIR}/Pascal_VOC/pascal_07_val.record \
       --img_dir=image \
       --annotation_file=via_annotation.json \
       --label_map_file=${TF_MODELS}/object_detection/data/pascal_label_map.pbtxt \
       --max_read=10

python tfrecord_to_via.py --tfrecord_file=/home/yoon/Downloads/yolo/testC.record \
       --img_dir=image \
       --annotation_file=via_annotation_a.json \
       --label_map_file=/home/yoon/Downloads/yolo/detection.pbtxt \
       --max_read=10
