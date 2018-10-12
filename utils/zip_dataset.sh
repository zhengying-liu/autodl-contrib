#!/bin/bash
# Usage: ./zip_dataset.sh <dataset_name>
# For a given dataset with name $1, make .zip files with only useful files: metadata.textproto

INPUT_DIR=$(pwd)/../formatted_datasets
DATASET_DIR=$INPUT_DIR/$1
DATA_DIR=$DATASET_DIR/$1.data
cd $DATASET_DIR/
zip -r --exclude=*__pycache__* --exclude=*.DS_Store* --exclude=*__MACOSX* --exclude=*.yaml --exclude=*.csv $1.data.zip $1.data;
zip $1.solution.zip $1.solution
