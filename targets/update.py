"""update
Author : Jackson Cole
Affil  : Middle Tennessee State University
Purpose: This script exists solely to update targets_done.txt,
         all_targets.txt, and remaining.txt. These files will help in
         figuring out which targets have not yet been processed.

Notes  : This script should be made executable, as it calls specifically
         python3 in the hashbang.
"""
from os import listdir, path

def main():
    done = [x for x in listdir(".") if path.isdir(x)]

    with open('targets_done.txt','w') as f:
        for d in done:
            f.write(d + '\n')

    all_targets = []
    with open('all_targets.txt', 'r') as f:
        for line in f:
            all_targets.append(line.rstrip())

    had_problems = []
    with open('had_problems.txt', 'r') as f:
        for line in f:
            if not line.startswith("#"):
                had_problems.append(line.rstrip())

    remaining = \
    set(all_targets).difference(done).difference(had_problems)
    with open('remaining.txt','w') as f:
        for t in remaining:
            f.write(t + '\n')

if __name__ == "__main__":
    main()
