#!/usr/bin/env python3
"""jspamcli.py
Author : Jackson Cole
Affil  : Middle Tennessee State University
Purpose: This program's purpose is to script the interaction with the
         original Fortran90 basic_run program to allow for multiple runs
         to be processed.

Notes  : This script requires Python3.
"""

from sys import argv
from os import system, path, mkdir, chdir, rename, walk, listdir

from numpy import genfromtxt
from multiprocessing import Pool, cpu_count

from merger import MergerRun, data_tools, get_target_data
from data_tools import Structure


def main(argv):
    """This program accepts command line arguments, and they are dealt
    with here.
    """
    if argv[1] == '-i':
        """This handles interactive (-i) processing.
        """
        yes = ['yes', 'y']
        if (str(input('Do you need to download any input files? (yes/no)\n> ')).lower() in yes):
            get_target_data('input', 0, 'dummy')

        name, filename = get_input_file_INTERACTIVE()

        n1_particles = int(input('Enter number of particles for galaxy 1:\n> '))
        n2_particles = int(input('Enter number of particles for galaxy 2:\n> '))

        first_run, last_run = how_many_runs()

        init_value_strings, run_list = get_init(
                filename, first_run, last_run)

        path_to_info_file = './targets/' + name + '/' + name + '.info.txt'

        mergers = []
        for init_run_string, run in zip(init_value_strings, run_list):
            mergers.append(MergerRun(path_to_info_file, n1_particles,
                n2_particles, run, init_run_string))

        for merger in mergers:
            get_runs_scores_and_wipe(merger)

    elif argv[1] == '-bi':
        """This handles batch processing... interactively (-bi).
        """
        if path.isdir('./batch_run_files/') == False:
            mkdir('./batch_run_files')
            with open('./batch_run_files/sample.txt', 'w') as f:
                f.write('#target,n1_particles,n2_particles,first_run,last_run\n')
                f.write('587722984435351614,500,500,100,100\n')

            print('A batch_run_file has been created with a sample run file '\
                    'inside.\n You must create a batch run file to do batch '\
                    'processing.')

        else:
            print('The following batch processing files are available:')

            for file_name in [f for f in listdir('./batch_run_files/')]:
                if path.isfile('./batch_run_files/'+file_name) == True and \
                        file_name != 'sample.txt':
                    print(file_name)

            choice = input('Please copy and paste the path of the appropriate '\
                    'batch run file:\n> ')

            lines = []
            with open('./batch_run_files/' + choice, 'r') as f:
                for line in f:
                    if line.startswith('#') == False:
                        lines.append(line.rstrip().split(','))

            for line in lines:
                name = line[0]
                n1_particles = int(line[1])
                n2_particles = int(line[2])
                first_run    = int(line[3]) - 1
                last_run     = int(line[4]) - 1

                get_target_data('input', 1, name)

                filename = './input/' + name + '.txt'
                init_value_strings, run_list = get_init(
                        filename, first_run, last_run)

                path_to_info_file = './targets/' + name + '/' + name + \
                        '.info.txt'

                mergers = []
                for init_run_string, run in zip(init_value_strings, run_list):
                    mergers.append(MergerRun(path_to_info_file, n1_particles,
                        n2_particles, run, init_run_string))

                for merger in mergers:
                    get_runs_scores_and_wipe(merger)


    elif argv[1] == '-b':
        """This handles full batch processing (-b).
        """
        if len(argv) < 3:
            print('No batch_run_files specified...')
        else:
            for batch_run_file in argv[2:]:
                lines = []
                with open('./batch_run_files/' + batch_run_file, 'r') as f:
                    for line in f:
                        if line.startswith('#') == False:
                            lines.append(line.rstrip().split(','))

                for line in lines:
                    name = line[0]
                    n1_particles = int(line[1])
                    n2_particles = int(line[2])
                    first_run    = int(line[3]) - 1
                    last_run     = int(line[4]) - 1

                    get_target_data('input', 1, name)

                    filename = './input/' + name + '.txt'
                    init_value_strings, run_list = get_init(
                            filename, first_run, last_run)

                    path_to_info_file = './targets/' + name + '/' + name + \
                            '.info.txt'

                    mergers = []
                    for init_run_string, run in zip(init_value_strings, run_list):
                        mergers.append(MergerRun(path_to_info_file, n1_particles,
                            n2_particles, run, init_run_string))

                    for merger in mergers:
                        get_runs_scores_and_wipe(merger)


    elif argv[1] == '-bm':
        """This handles batch processing across multiple cores. While
        the actual computation does not get parallelized in this code,
        multiple processes are created and are mapped to the number of
        cores requested.
        """
        if len(argv) < 3:
            print('No batch_run_files specified...')
        else:
            # --------------------------------------------------------------- #
            # The third CL arg is the number of cores requested to be
            ## used. If the number of cores requested is greater than
            ## half of the total number of cores on the machine, nothing
            ## is executed.
            cores_requested = int(argv[2])
            if cores_requested > cpu_count() / 2:
                print("ERROR: Per my rules, you shall not use more "
                        "than HALF of available CPUs.")
                print("There are " + cpu_count() / 2 + "cores "
                        "available.")
            # --------------------------------------------------------------- #

            else:
                for batch_run_file in argv[3:]:
                    lines = []
                    with open('./batch_run_files/' + batch_run_file, 'r') as f:
                        for line in f:
                            if line.startswith('#') == False:
                                lines.append(line.rstrip().split(','))

                    mergers_master = []
                    for line in lines:
                        name = line[0]
                        n1_particles = int(line[1])
                        n2_particles = int(line[2])
                        first_run    = int(line[3]) - 1
                        last_run     = int(line[4]) - 1
                        if len(line) == 6:
                            if line[5] == "ALL":
                                do_all = True
                        else:
                            do_all = False


                        get_target_data('input', 1, name)

                        filename = './input/' + name + '.txt'
                        init_value_strings, run_list = get_init(
                        filename, first_run, last_run, do_all)

                        path_to_info_file = './targets/' + name + '/' + name + \
                        '.info.txt'

                        mergers = []
                        for init_run_string, run in zip(init_value_strings, run_list):
                            mergers.append(MergerRun(path_to_info_file, n1_particles,
                            n2_particles, run, init_run_string))

                        mergers_master.append(mergers)

                # ----------------------------------------------------------- #
                # This code snippet is likely a waste of memory, and
                # could just be implemented in the last for loop, but I
                ## may incorporate another line of logic here.
                all_mergers = []
                for mergers in mergers_master:
                    for merger in mergers:
                        all_mergers.append(merger)
                # ----------------------------------------------------------- #

                # ----------------------------------------------------------- #
                # Creating the processes to map to multiple cores, then
                # immediately getting the result
                with Pool(processes = cores_requested) as pool:
                    result = pool.map_async(get_runs_scores_and_wipe_unpack,
                            zip(all_mergers, range(len(all_mergers))))
                    result.get()
                # ----------------------------------------------------------- #

    elif argv[1] == '-g':
        print("WELCOME TO THE GALAXY MERGER GIF CREATION TOOL")
        name, filename = get_input_file_INTERACTIVE()

        n1_particles = int(input('Enter number of particles for galaxy 1:\n> '))
        n2_particles = int(input('Enter number of particles for galaxy 2:\n> '))

        dimensions = int(input("2D or 3D? (2/3):\n> "))

        first_run, last_run = how_many_runs()

        init_value_strings, run_list = get_init(
                filename, first_run, last_run)

        path_to_info_file = './targets/' + name + '/' + name + '.info.txt'

        mergers = []
        for init_run_string, run in zip(init_value_strings, run_list):
            mergers.append(MergerRun(path_to_info_file, n1_particles,
                n2_particles, run, init_run_string))

        for merger in mergers:
            giffy(merger, dimensions)

    else:
        print('Option not recognized')


    print('Exiting...')


def get_input_file_INTERACTIVE():
    print('These are the available input files: ')
    for file_name in [f for f in listdir('./input/')
            if path.isfile('./input/' + f)]:
                print(file_name)

    name_input = input('Copy and paste the name of a file in the input '\
                 'directory you want to work with:\n> ')
    name = name_input[:-len('.txt')]
    filename = './input/' + name + '.txt'

    print('Target File Selected: {}'.format(path.splitext(filename)[0]))

    return name, filename


def how_many_runs():
    """Determine how many runs to do, return the numbers corresponding
    to the first and last runs. first_run and last_run will be first_run
    + 1 if only one run is wanted.
    """
    how_many = 1000
    while how_many != 1 and how_many != 2:
        how_many = int(input('Would you like to run one run or multiple '\
                'runs? (1),(2):\n(1) one\n(2) multiple\n> '))
        if how_many != 1 and how_many != 2:
            print('ERROR: Invalid input')

    if how_many == 1:
        first_run = int(input('Enter number of run:\n> '))
        last_run = first_run
    elif how_many == 2:
        first_run = int(input('Enter number of first_run:\n> '))
        last_run = int(input('Enter number of last_run:\n> '))

    return first_run, last_run


def get_init(filename, first_run, last_run, do_all = False):
    """Get the initial value strings from the target file
    """
    init_value_strings = []
    if not do_all:
        for i, line in enumerate(open(filename, 'r')):
            if i in range(first_run - 1, last_run):
                init_value_strings.append(line.split('\t')[1].strip())

    else:
        first_run = 0
        for i, line in enumerate(open(filename, 'r')):
            init_value_strings.append(line.split('\t')[1].strip())

        last_run = i

    run_list = range(first_run - 1, last_run)

    return init_value_strings, run_list


def setup_target_structure(filename, run_number, n1_particles, n2_particles):
    """This function sets up the target structure using the Structure
    module. The lines leading up to "target_dirs = Structure(paths)"
    setup the "paths" list, which is the only argument that the
    Structure module needs.
    """
    root_name = 'output'
    run_number = str(run_number + 1).zfill(4)
    print("Processing: " + run_number)
    name = filename[len('./input/'):-len('.txt')]

    paths = [root_name, name, 'run{}'.format(run_number)]
    target_dirs = Structure(paths)

    return target_dirs


def wipe(merger, distinguisher = 1):
    """The current configuration of this program outputs ALL runfiles
    in the root directory where run_multiple is called. This function
    uses the Structure module in the data_tools package to create the
    directory structure needed for organizing the data effectively.

    The runfiles are the relocated to the appropriate directory in the
    structure and are renamed with the following naming convention:
        {sdssid}.(i, f).{number of particles in galaxy 1}
        .{num of particles in galaxy 2}.txt

    This function should also read in the citizen science scores from
    the target run file... there's probably a better place to do this.
    """
    merger.create()
    merger.get_scores()
    merger.write_scores()

    root_name = merger.target_dirs.paths[0]
    name = merger.name
    run_number = str(merger.run + 1).zfill(4)

    for file_name in [f for f in listdir('.') if path.isfile(f)]:
        if (file_name.startswith("a_"+str(distinguisher))):
            extension = path.splitext(file_name)[1][1:]
            rename(file_name, (merger.target_dirs.full_path +
                    '{sdssid}.{run_no}.{ext}.{n1}.{n2}.txt'.format(
                        root   = root_name,
                        sdssid = name,
                        run_no = run_number,
                        ext    = ('i', 'f')[extension == '101'],
                        n1     = merger.primary.particles,
                        n2     = merger.secondary.particles,)))


def get_runs_scores_and_wipe(merger, distinguisher = 1):
    """This function with a too-long name actually calls the Fortran
    simulation via basic_run.
    """
    system('./basic_run -m {} -n1 {} -n2 {} {}'
            .format(distinguisher,
                    merger.primary.particles,
                    merger.secondary.particles,
                    merger.init))
    wipe(merger, distinguisher)


def get_runs_scores_and_wipe_unpack(zipped):
    """This function just unzips zipped arguments for
    get_runs_scores_and_wipe. This was the only implementation that allowed
    calling a multi-arg function with map_async using only one
    arg.
    """
    get_runs_scores_and_wipe(zipped[0], zipped[1])


def giffy(merger, dimensions, distinguisher = 1):
    system('./basic_run -g DUMMY_ARG -m {} -n1 {} -n2 {} {}'
            .format(distinguisher,
                    merger.primary.particles,
                    merger.secondary.particles,
                    merger.init))

    merger.create()
    merger.make_gif(dimensions)


if (__name__ == '__main__'):
    if len(argv) > 1:
        main(argv)

    else:
        print('jspamcli accepts the following command line'\
                ' options:\n')
        print('    -i  : run interactively\n'\
              '    -bi : batch process (interactively...)\n'\
              '    -b  : batch process\n'\
              '    -bm : batch process on multiple cores\n'\
              '    -g  : GIF Creation Tool\n')

