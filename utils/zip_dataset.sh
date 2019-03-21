#!/bin/bash
# Usage: ./zip_dataset.sh <dataset_path>
# For a given dataset with name $1, make .zip files with only useful files: metadata.textproto

DATASET_DIR=$1
DATASET_NAME=$(basename $DATASET_DIR)
cd $DATASET_DIR/
zip -r --exclude=*__pycache__* --exclude=*.DS_Store* --exclude=*__MACOSX* --exclude=*.yaml --exclude=*.csv $DATASET_NAME.data.zip $DATASET_NAME.data;
zip $DATASET_NAME.solution.zip $DATASET_NAME.solution
