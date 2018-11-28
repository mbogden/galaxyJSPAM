"""reduce.py
Author : Jackson Cole
Affil  : Middle Tennessee State University
Purpose: This program reduces the size of the downloadable input files
         to contain ONLY the scored runs (66395)
"""

import os


def main():
    count = 0
    for x in [f for f in os.listdir(".") if (os.path.isfile(f) and
            os.path.splitext(f)[1] == ".txt")]:
        lines = []
        with open(x, 'r') as f:
            lines = f.readlines()  # Somewhat inefficient, but it'll do
            try:
                last_score = lines.index("#rejected\n") - 1
            except:
                last_score = len(lines)

        with open(x, 'w') as f:
            for line in lines[0:last_score]:
                f.write(line)

if __name__ == "__main__":
    main()
