#!/usr/bin/env python3
from merger import MergerRun, data_tools
from data_tools import Structure
from os import walk, mkdir
from os.path import isdir, exists
from sys import argv
import numpy as np
import scipy.ndimage as ndimage
from scipy import misc
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import glob
import imageio


def load_data(filename):
    print(filename)
    data_file = open(filename, 'r')
    point_data = np.empty(shape=[len(data_file.readlines()), 6], dtype=float)
    data_file.seek(0)
    for i, line in enumerate(data_file):
        for j, value in enumerate(line.split()):
            point_data[i, j] = float(value)

    data_file.close()

    return point_data


def load_all_data():
    list_of_files = []
    for (dirpath, dirnames, filenames) in walk('.'):
        for filename in filenames:
            if filename.startswith('a_') and filename.endswith(tuple(str(i) for i in range(0,1000000))):
                list_of_files.append(filename)

    all_data = []
    for filename in list_of_files:
        all_data.append(load_data(filename))

    return all_data

def make_image_files():
    if exists('./images') == False:
        mkdir('./images')
        mkdir('./images/gif')
    else:
        if exists('./images/gif') == False:
            mkdir('./images/gif')

def plotting_2d(point_data, filename_wo_txt):
    plot_list = input('Enter coordinates to be plotted in 2D separated by a comma:\n> ')
    plot_list = [coord.strip() for coord in plot_list.split(',')]
    x = point_data[:, 0]
    y = point_data[:, 1]
    z = point_data[:, 2]
    fig = plt.figure()
    rect = fig.patch
    rect.set_facecolor('black')
    ax = fig.gca()
    # plt.xlim(-8,8)
    # plt.ylim(-8,8)
    # ax = fig.add_subplot(111, projection='3d')
    ax.scatter(eval(plot_list[0]), eval(plot_list[1]), s=0.5, marker='.', c='white', edgecolor='None')
    ax.set_frame_on(False)
    plt.axis('off')
    plt.savefig('{}_2d.png'.format(filename_wo_txt), facecolor=fig.get_facecolor(), edgecolor='none', dpi=1000, pad_inches=0)

    plt.close()


def gaussian_blur():
    # smoothed = np.empty(shape=np.shape(data))
    image_data = imageio.imread('a.101_2d.png')
    # misc.imsave('a.101_2d.png', image)
    gaussian_smoothed = ndimage.gaussian_filter(image_data, 8)
    median_smoothed = ndimage.median_filter(image_data, 8)
    imageio.imwrite('gaussian_smoothed.png', gaussian_smoothed)
    imageio.imwrite('median_smoothed.png', median_smoothed)
    #plt.imshow(image)
    #plt.show()

#gaussian_blur()

def plotting_3d(point_data, filename_wo_txt):
    x = point_data[:, 0]
    y = point_data[:, 1]
    z = point_data[:, 2]
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlim3d(-8, 8)
    ax.set_ylim3d(-8, 8)
    ax.set_zlim3d(-8, 8)
    # ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=0.1, marker='.', edgecolor='None')
    plt.xticks([],  [])
    plt.yticks([],  [])
    plt.savefig('{}_3d.png'.format(filename_wo_txt), dpi=1000)
    plt.close()

def plotting_3d_for_gif(point_data, filename_wo_txt):
    x = point_data[:, 0]
    y = point_data[:, 1]
    z = point_data[:, 2]
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlim3d(-8, 8)
    ax.set_ylim3d(-8, 8)
    ax.set_zlim3d(-8, 8)
    # ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, s=1, marker='.', edgecolor='None')
    plt.xticks([],  [])
    plt.yticks([],  [])


def determine_plot_types():
        while True:
            try:
                plot_type = int(input('Enter the number corresponding to your choice of plotting:\n(1) 2D\n(2) 3D\n(3) both\n> '))
                if plot_type in [1, 2, 3]:
                    break
                elif plot_type == 5:
                    break
                else:
                    print('ERROR: Enter valid choice')
            except ValueError:
                print('ERROR: Invalid input')

        gif_answer = input('Do you want to create a 3D gif comprised of each time step? (y/n)\n> ')

        return plot_type, gif_answer


def make_a_gif(filename_wo_txt, all_data):
    gif_filename = ''.format(filename_wo_txt)
    for i, data_list in enumerate(all_data):
        plotting_3d_for_gif(data_list, filename_wo_txt)
        plt.savefig('images/{}/img{}.png'.format(gif_filename, str(i).zfill(3)), bbox_inches='tight')
    plt.close()
    images = [imageio.imread(image) for image in walk('./images/{}/*.png'.format(gif_filename))]
    file_path_name = 'images/{}.gif'.format(gif_filename)
    imageio.mimsave(file_path_name, images)


def main(argv):
    yes = frozenset(['yes', 'y', 'ye', 'ys'])
    if len(argv) > 1:
        if argv[1] != ('gaussian'):
            filename = argv[1]
            filename_wo_txt = filename[:len(filename)]
            make_image_files()


            plot_type, gif_answer = determine_plot_types()
            if plot_type == 1:
                point_data = load_data(filename)
                plotting_2d(point_data, filename_wo_txt)
                gaussian_blur()
            elif plot_type == 2:
                point_data = load_data(filename)
                plotting_3d(point_data, filename_wo_txt)
            elif plot_type == 3:
                point_data = load_data(filename)
                plotting_2d(point_data, filename_wo_txt)
                plotting_3d(point_data, filename_wo_txt)
            else:
                gaussian_blur()

            if gif_answer in yes:
                all_data = load_all_data()
                make_a_gif(filename_wo_txt, all_data)
        else:
            gaussian_blur()

    else:
        print("ERROR: plotting_testing takes 1 argument. None given. Run again while giving the name of the data file at the time step you wish to plot.")


if __name__ == '__main__':
    main(argv)
