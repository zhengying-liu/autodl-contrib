# Author: Zhengying Liu
# Date: 28 Aug, 2018
# Description: basic Python code to deal with WAV files

from scipy.io import wavfile

def main():
  filename = "example.wav"
  fs, data = wavfile.read(filename)
  print(fs)

if __name__ == '__main__':
  main()
