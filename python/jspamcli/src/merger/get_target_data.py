"""get_target_data.py
Author : Jackson Cole
Affil  : Middle Tennessee State University
Purpose: This module scrapes the https://data.galaxyzoo.org/ webpage
         for links to downloadable target data, which it can then
         download and unzip. It also takes a switcher argument which
         specifies the case to be used (interactive or batch). If the
         case is interactive, 'target_given' is just a dummy value.
         Otherwise, it is the SDSSID of the target.

Notes  : This really does not belong in the data_tools package, but
         it's the best place for it right now, so here it lives.
"""

from os import mkdir, path, remove, system
import gzip

from bs4 import BeautifulSoup as bs
import requests


def get_target_data(dest_path, switcher, target_given = 900):

    _base = 'https://data.galaxyzoo.org/'
    _URL = _base + 'mergers.html'
    r = requests.get(_URL)

    soup = bs(r.text, 'lxml')
    paths_for_url = []
    target_names = []

    for link in [link for link in soup.findAll(name = 'a')
        if 'parameter_files' in str(link)]:
        link = str(link)
        path_for_url = link[link.find('galaxy-'):link.find('.txt.gz') +
                len('.txt.gz')]
        target_name = link[link.find('r_files/') + len('r_files/'):
                link.find('_combined')]
        paths_for_url.append(path_for_url)
        target_names.append(target_name)

    if (switcher == 0):
        print('Below will appear the list of available targets.')
        input('Please press [ENTER].')
        print('\nNumber\tTarget Name')
        for i, target in enumerate(target_names):
            print('{number}:\t{target}'.format(number = i + 1,\
                    target = target))

        target_wanted = \
                input('Please enter the number corresponding to a target to'
                'download the associated data.\n'
                'Enter "ALL" to download and upzip all targets to input'
                'directory.\n> ')

    elif (switcher == 1):
        target_wanted = target_names.index(target_given)


    if (target_wanted != 'ALL'):
        target_to_download = [int(target_wanted) - 1]
        target_name = target_names[int(target_wanted) - 1]
    elif (target_wanted == 'ALL'):
        target_to_download = list(range(len(paths_for_url)))
        target_name = target_names

    if path.isdir(dest_path) == False:
        mkdir(dest_path)

    for target_number, target in zip(target_to_download, target_name):
        target_gz = dest_path + '/{}.txt.gz'.format(target)
        url = _base + paths_for_url[target_number]
        #system('wget -qO ' + target_gz + ' ' + url)
        r = requests.get(url, stream=True)
        # the below section was found at
        # https://stackoverflow.com/questions/16694907/how-to-download-
        # large-file-in-python-with-requests-py
        with open(target_gz, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

        with gzip.open(target_gz, 'rb') as infile:
            with open(target_gz[:-3], 'wb') as outfile:
                for line in infile:
                    outfile.write(line)

        remove(target_gz)
