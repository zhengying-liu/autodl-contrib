# Author: Zhengying Liu
# Creation Date: 3 Dec 2019
"""Given two files of Python packages (results of `pip list`), show their
differences.
"""

from pprint import pprint
import argparse
import sys

def read_requirements_file(filepath, sep='='):
  """Given a requirements.txt file, create a dictionary of package_name:version
  pairs.
  """
  pkg_ver = {}
  with open(filepath, 'r') as f:
    line = f.readline().rstrip()
    while line:
      package_name = line.split(sep)[0]
      version = line.split(sep)[-1]
      pkg_ver[package_name] = version
      line = f.readline().rstrip()
  return pkg_ver


def compare(dict1, dict2):
  """Return 3 dict showing the differences of the 2 dict."""
  in1_not_in2 = {k:v for k,v in dict1.items() if k not in dict2}
  in2_not_in1 = {k:v for k,v in dict2.items() if k not in dict1}
  diff_val = {}
  for k in dict1:
    if k in dict2:
      v1 = dict1[k]
      v2 = dict2[k]
      if v1 != v2:
        diff_val[k] = (v1, v2)
  return in1_not_in2, in2_not_in1, diff_val


def compare_requirements(filepath1, filepath2, sep='='):
  dict1 = read_requirements_file(filepath1, sep=sep)
  dict2 = read_requirements_file(filepath2, sep=sep)
  in1_not_in2, in2_not_in1, diff_val = compare(dict1, dict2)
  print("\nPackages in {} but not in {}:".format(filepath1, filepath2))
  pprint(in1_not_in2)
  print("\nPackages in {} but not in {}:".format(filepath2, filepath1))
  pprint(in2_not_in1)
  print("\nPackages in both requirements files but have different versions:")
  pprint(diff_val)


def test_read_requirements_file():
  filepath = 'autodl_requirements.txt'
  pkg_ver = read_requirements_file(filepath, sep=' ')
  print(pkg_ver)


def test_compare_requirements():
  filepath1 = 'autodl_requirements.txt'
  filepath2 = 'autonlp_requirements.txt'
  filepath3 = 'autospeech_requirements.txt'
  compare_requirements(filepath1, filepath2, sep=' ')
  compare_requirements(filepath1, filepath3, sep=' ')


def main(*argv):
  test_read_requirements_file()
  test_compare_requirements()


if __name__ == '__main__':
  main(sys.argv[1:])
