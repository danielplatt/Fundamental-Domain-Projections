import requests
from os.path import exists

def download_file(url, filename):
    if exists(filename) == False:
        f = requests.get(url)
        open(filename, 'wb').write(f.content)
        f.close()
    else:
        print('A file with the name {} already exists. It has not been downloaded again. To start the download rename the existing file.'.format(filename))

def download_cicy_raw():
    download_file('http://www-thphys.physics.ox.ac.uk/projects/CalabiYau/cicylist/cicylist.txt','rawdata.txt')

if __name__ == '__main__':
    download_cicy_raw()