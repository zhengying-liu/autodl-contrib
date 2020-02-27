# Format utils contains functions shared by format_image, format_video, etc.

import os
import glob
import hashlib
import numpy as np
import pandas as pd
import tarfile
import zipfile
from sklearn.utils import shuffle

def get_labels_df(dataset_dir, shuffling=True):
  """ Read labels.csv and return DataFrame
  """
  if not os.path.isdir(dataset_dir):
    raise IOError("{} is not a directory!".format(dataset_dir))
  labels_csv_files = [file for file in glob.glob(os.path.join(dataset_dir, '*labels*.csv'))]
  if len(labels_csv_files) > 1:
    raise ValueError("Ambiguous label file! Several of them found: {}".format(labels_csv_files))
  elif len(labels_csv_files) < 1:
    raise ValueError("No label file found! The name of this file should follow the glob pattern `*labels*.csv` (e.g. monkeys_labels_file_format.csv).")
  else:
    labels_csv_file = labels_csv_files[0]
  labels_df = pd.read_csv(labels_csv_file)
  if shuffling:
    labels_df = shuffle(labels_df, random_state=42)
  return labels_df

def get_merged_df(labels_df, train_size=0.8):
  """Do train/test split (if needed) by generating random number in [0,1]."""
  merged_df = labels_df.copy()
  if 'subset' not in labels_df:
      np.random.seed(42)
      def get_subset(u):
        if u < train_size:
          return 'train'
        else:
          return 'test'
      merged_df['subset'] = merged_df.apply(lambda x: get_subset(np.random.rand()), axis=1)
  return merged_df

def get_labels(labels, confidence_pairs=False):
  """Parse label confidence pairs into two lists of labels and confidence.

  Args:
    labels: string, of form `2 0.0001 9 0.48776 0 1.0`." or "2 9 0"
    confidence_pairs: True if labels are confidence pairs.
  """
  if isinstance(labels, str):
      l_split = labels.split(' ')
  else:
      l_split = [labels]
  if confidence_pairs:
      labels = [int(x) for i, x in enumerate(l_split) if i%2 == 0]
      confidences = [float(x) for i, x in enumerate(l_split) if i%2 == 1]
  else:
      labels = [int(x) for x in l_split if x==x] # x==x to remove NaN values
      confidences = [1 for _ in labels]
  return labels, confidences

def get_all_classes(merged_df):
  if 'LabelConfidencePairs' in list(merged_df):
      label_confidence_pairs = merged_df['LabelConfidencePairs']
      confidence_pairs = True
  elif 'Labels' in list(merged_df):
      label_confidence_pairs = merged_df['Labels']
      confidence_pairs = False
  else:
      raise Exception('No labels found, please check labels.csv file.')

  labels_sets = label_confidence_pairs.apply(lambda x: set(get_labels(x, confidence_pairs=confidence_pairs)[0]))
  all_classes = set()
  for labels_set in labels_sets:
    all_classes = all_classes.union(labels_set)
  return all_classes

# Functions below are taken from:
#   https://github.com/pytorch/vision/blob/07cbb46aba8569f0fac95667d57421391e6d36e9/torchvision/datasets/utils.py

def _is_tarxz(filename):
    return filename.endswith(".tar.xz")


def _is_tar(filename):
    return filename.endswith(".tar")


def _is_targz(filename):
    return filename.endswith(".tar.gz")


def _is_tgz(filename):
    return filename.endswith(".tgz")


def _is_gzip(filename):
    return filename.endswith(".gz") and not filename.endswith(".tar.gz")


def _is_zip(filename):
    return filename.endswith(".zip")


def extract_archive(from_path, to_path=None, remove_finished=False):
    if to_path is None:
        to_path = os.path.dirname(from_path)

    if _is_tar(from_path):
        with tarfile.open(from_path, 'r') as tar:
            tar.extractall(path=to_path)
    elif _is_targz(from_path) or _is_tgz(from_path):
        with tarfile.open(from_path, 'r:gz') as tar:
            tar.extractall(path=to_path)
    elif _is_tarxz(from_path) and PY3:
        # .tar.xz archive only supported in Python 3.x
        with tarfile.open(from_path, 'r:xz') as tar:
            tar.extractall(path=to_path)
    elif _is_gzip(from_path):
        to_path = os.path.join(to_path, os.path.splitext(os.path.basename(from_path))[0])
        with open(to_path, "wb") as out_f, gzip.GzipFile(from_path) as zip_f:
            out_f.write(zip_f.read())
    elif _is_zip(from_path):
        with zipfile.ZipFile(from_path, 'r') as z:
            z.extractall(to_path)
    else:
        raise ValueError("Extraction of {} not supported".format(from_path))

    if remove_finished:
        os.remove(from_path)


def calculate_md5(fpath, chunk_size=1024 * 1024):
    md5 = hashlib.md5()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5.update(chunk)
    return md5.hexdigest()


def check_md5(fpath, md5, **kwargs):
    return md5 == calculate_md5(fpath, **kwargs)


def check_integrity(fpath, md5=None):
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)


def download_and_extract_archive(url, download_root, extract_root=None, filename=None,
                                 md5=None, remove_finished=False):
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if not filename:
        filename = os.path.basename(url)

    download_url(url, download_root, filename, md5)

    archive = os.path.join(download_root, filename)
    print("Extracting {} to {}".format(archive, extract_root))
    extract_archive(archive, extract_root, remove_finished)


def download_url(url, root, filename=None, md5=None):
    """Download a file from a url and place it in root.
    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    from six.moves import urllib

    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    os.makedirs(root, exist_ok=True)

    # check if file is already present locally
    if check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:   # download the file
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(
                url, fpath
            )
        except (urllib.error.URLError, IOError) as e:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(
                    url, fpath,
                    reporthook=gen_bar_updater()
                )
            else:
                raise e
        # check integrity of downloaded file
        if not check_integrity(fpath, md5):
            raise RuntimeError("File not found or corrupted.")
