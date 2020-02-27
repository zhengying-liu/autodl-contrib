STARTING_KIT_DIR = '/Users/evariste/projects/autodl/codalab_competition_bundle/AutoDL_starting_kit'  # TO BE REPLACED
from google.protobuf import text_format
from tensorflow import gfile
import argparse
import sys
import time
sys.path.append(STARTING_KIT_DIR)
from AutoDL_ingestion_program.data_pb2 import DataSpecification

def test_metadata_textproto(path_to_textproto):
    metadata_ = DataSpecification()
    begin = time.time()
    print("Begin reading metadata.textproto file at {}..."\
            .format(path_to_textproto))
    with gfile.GFile(path_to_textproto, "r") as f:
        text_format.Merge(f.read(), metadata_)
    end = time.time()
    print("Successfully read metadata file with DataSpecification")
    print("Time used: {} seconds".format(end - begin))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test if a metadata file is valid.')
    parser.add_argument('-p', '--path_to_textproto', type=str,
                        help='Path to metadata.textproto')
    args = parser.parse_args()
    path_to_textproto = args.path_to_textproto
    test_metadata_textproto(path_to_textproto)
