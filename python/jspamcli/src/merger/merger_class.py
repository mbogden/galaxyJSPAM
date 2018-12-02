"""merger_class.py
Author : Jackson Cole
Affil  : Middle Tennessee State University
Purpose: The MergerRun class is useful for encapsulating all data
         and methods that are commonly used when interacting with a
         merger's data.

         There are general size and dimensions classes at the bottom as
         well. They do what they do.
"""

import os
from sys import argv

import glob
import imageio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy import misc
import scipy.ndimage as ndimage

from data_tools import Structure
from galaxy import Galaxy


class MergerRun:
    def __init__(self, path_to_info_file, n1_particles, n2_particles,
            run_number, init_run_string):
        """Constructor for Merger class:
        """
        self.info = self.make_info_dict(path_to_info_file)
        self.name            = self.info['name']
        self.height          = Size(int(self.info['imageheight']),
                                        None)
        self.width           = Size(int(self.info['imagewidth']),
                                        None)
        self.dimensions      = Dimensions(self.height, self.width)
        self.filename        = self.name + '.txt'
        self.humanscores_filename = self.name + '.humanscores.txt'
        self.run             = run_number
        self.init            = init_run_string
        self.structure_created = False
        self.target_dirs     = self.setup_structure()
        self.primary         = Galaxy(
                name      = self.info['pname'],
                particles = n1_particles,
                ra        = self.info['pra'],
                dec       = self.info['pdec'],
                xc        = self.info['pxc'],
                yc        = self.info['pyc'],)


        self.secondary         = Galaxy(
                name      = self.info['sname'],
                particles = n1_particles,
                ra        = self.info['sra'],
                dec       = self.info['sdec'],
                xc        = self.info['sxc'],
                yc        = self.info['syc'],)

        self.all_point_data = []

        # None-type attibutes upon initialization
        self.scores        = None


    def make_info_dict(self, path_to_info_file):
        entries = []
        with open(path_to_info_file, 'r') as f:
            for line in f:
                if not (line.startswith('#') or line == '\n'):
                    line = line.rstrip().split('=')
                    entries.append("'{}': '{}'".format(line[0], line[1]))


        info = eval('{' + ', '.join(entries) + '}')
        return info


    def setup_structure(self):
        if self.structure_created == False:
            path_list = ['output',
                    self.name,
                    'run'+str(self.run + 1).zfill(4),
                    ]
            target_dir = Structure(path_list)
        else:
            print('Directory structure already created. Continuing...')
            target_dir = Structure(path_list)

        return target_dir


    def existing_structure(self):
        if os.path.exists(self.target_dir.full_path):
            self.structure_created = True
        else:
            self.structure_created = False


    def create(self):
        """This really just calls the structure create method. I'm sure
        there is a better way to do this.
        """
        self.target_dirs.create()


    def get_scores(self):
        scores = []
        path_to_target_file = './input/' + self.filename
        if os.path.exists(path_to_target_file):
            with open(path_to_target_file, 'r') as f:
                line = f.readline()
                while len(line.split('\t')[0].split(',')) != 1:
                    score_info = line.split('\t')[0].split(',')[1:4]
                    scores.append(score_info)
                    line = f.readline()

        else:
            print(path_to_target_file + ' does not exist.')

        self.scores = scores


    def write_scores(self):
        path_to = './output/' + self.name + '/'
        if os.path.exists(path_to + self.humanscores_filename):
            pass

        else:
            with open(path_to + self.humanscores_filename, 'w') as f:
                for score_list in self.scores:
                    f.write(','.join(score_list) + '\n')


    def load_data(self, filename):
        data_file = open(filename, 'r')
        point_data = np.empty(shape=[len(data_file.readlines()), 6], dtype=float)
        data_file.seek(0)
        for i, line in enumerate(data_file):
            for j, value in enumerate(line.split()):
                point_data[i, j] = float(value)

        data_file.close()

        return point_data


    def fill_all_data(self):
        list_of_files = []
        for (dirpath, dirnames, filenames) in os.walk('.'):
            filenames = [x for x in filenames if x.startswith('a_') and\
                    x.endswith(tuple(str(i) for i in range(0,1000000)))]

            filenames.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
            for filename in filenames:
                #if filename.startswith('a_') and \
                #        filename.endswith(tuple(str(i)
                #            for i in range(0,1000000))):
                    list_of_files.append(filename)

        for filename in list_of_files:
            self.all_point_data.append(self.load_data(filename))


    def plotting_2d(self, point_data):
        plot_list = ['x', 'y']
        #plot_list = [coord.strip() for coord in plot_list.split(',')]
        x = point_data[:, 0]
        y = point_data[:, 1]
        z = point_data[:, 2]
        fig = plt.figure()
        rect = fig.patch
        rect.set_facecolor('black')
        ax = fig.gca()
        plt.xlim(-4,4)
        plt.ylim(-4,4)
        # ax = fig.add_subplot(111, projection='3d')
        ax.scatter(eval(plot_list[0]), eval(plot_list[1]), s=1, marker='.', c='white', edgecolor='None')
        ax.set_frame_on(False)
        plt.axis('off')
        return fig


    def plotting_3d_for_gif(self, point_data):
        x = point_data[:, 0]
        y = point_data[:, 1]
        z = point_data[:, 2]
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_xlim3d(-2, 2)
        ax.set_ylim3d(-2, 2)
        ax.set_zlim3d(-2, 2)
        # ax = fig.add_subplot(111, projection='3d')

        ax.scatter(x, y, z, s=1.5, marker='.', edgecolor='None')
        plt.xticks([],  [])
        plt.yticks([],  [])
        plt.title(self.name)


    def make_gif(self, dimensions = 2):
        self.fill_all_data()
        gif_filename = "{sdssid}.{run}.{n1}.{n2}.{dim}D".format(
                sdssid = self.name,
                run    = str(self.run + 1).zfill(4),
                n1     = self.primary.particles,
                n2     = self.secondary.particles,
                dim    = dimensions)
        temp_structure = Structure(['.','images',gif_filename])
        temp_structure.create()

        for i, data_list in enumerate(self.all_point_data):
            if dimensions == 2:
                fig = self.plotting_2d(data_list)
                plt.savefig('./images/{}/img{}.png'.format(gif_filename,
                    str(i).zfill(3)), facecolor=fig.get_facecolor(),
                    edgecolor='none', dpi=300, pad_inches=0)

            elif dimensions == 3:
                self.plotting_3d_for_gif(data_list)
                plt.savefig('./images/{}/img{}.png'.format(gif_filename,
                    str(i).zfill(3)), dpi=300, bbox_inches='tight')

        plt.close()
        images = []
        for root,_,image_files in os.walk('./images/{}'.format(gif_filename)):
            image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
            for image in image_files:
                image = os.path.join(root, image)
                images.append(imageio.imread(image))
        file_path_name = self.target_dirs.paths[1] + '/{}.gif'.format(gif_filename)
        imageio.mimwrite(file_path_name, images)
        os.system("rm a_*")


class Size:
    def __init__(self, image_dim, actual_dim):
        """Constructor for a general size glass
        """
        self.image = image_dim
        self.real  = actual_dim


class Dimensions:
    def __init__(self, height, width):
        """Constructor for Dimensions class
        """
        self.image = [height.image, width.image]
        self.real  = [height.real, width.real]
