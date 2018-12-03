#/bin/sh

make
ln -s python/jspamcli/jspamcli.py .
# Uncomment the following line if you're on a mac... or if you're me and can't
# get it working
export MPLBACKEND="module://tkinter"
