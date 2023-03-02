import requests
import os
from os.path import exists


def download_file(url, filepath):
    if exists(filepath) == False:
        f = requests.get(url)
        open(filepath, 'wb').write(f.content)
        f.close()
    else:
        print('A file with the name {} already exists. It has not been downloaded again. To start the download rename the existing file.'.format(filepath))

def download_cicy_raw():
    file_name = 'rawdata.txt'
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{file_name}')
    download_file('http://www-thphys.physics.ox.ac.uk/projects/CalabiYau/cicylist/cicylist.txt', filepath)


if __name__ == '__main__':
    download_cicy_raw()
