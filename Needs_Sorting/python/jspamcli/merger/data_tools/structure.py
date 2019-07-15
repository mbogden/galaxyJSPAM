"""structure.py
Author : Jackson Cole
Affil  : Middle Tennessee State University
Purpose: This is a general purpose directory structure creator. It takes
         a list of path elements as an argument.
"""

from os import mkdir, path


class Structure:
    def __init__(self, path_list):
        """Constructor for this class:
            This class takes a list as an argument that contains the
            directory structure that should be created.
            The path list should have the form
            path_list = ['root', 'child_1', 'child_2', ...]
        """
        self.paths = self.get_structure_strings(path_list)
        self.full_path = './'+'/'.join(i for i in path_list) + '/'
        self.name = self.paths[1]


    def get_structure_strings(self, path_list):
        """Return the paths for each individual level in the directory
        structure.
        """
        paths = []
        for i in range(len(path_list)):
            paths.append('./' + '/'.join(j for j in path_list[:i+1]))

        return paths


    def create(self):
        """Actually create the each of the directories in the
        appropriate structure.
        """
        for i, path_name in enumerate(self.paths):
            if path.isdir(path_name) == False:
                for j in self.paths[i:]:
                    mkdir(j)
