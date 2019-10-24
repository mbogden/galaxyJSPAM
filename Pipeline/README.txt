////To install Qt on Linux
//First, download qt
wget http://download.qt.io/official_releases/qt/5.13/5.13.1/qt-opensource-linux-x64-5.13.1.run

//Then, run this to install it automatically
//Note: This installs a lot more than is needed, as I have not prioritized finding the minimal set needed
//Feel free to exclude --verbose, it's very verbose
//If you exclude it, some inconseqential warning messages may occur
//This will install Qt to a directory in the current directory named "Qt"
./qt-unified-linux-x64-3.1.1-online.run --platform minimal --verbose --script install.qs

//Then, while in the directory with the galaxyGUI sources
<Path to just mentioned "Qt" folder>/5.13.1/gcc_64/bin/qmake
make -j 2

//If all goes well, you can now run galaxyGUI 
//GUI
./galaxyGUI
//No GUI
./galaxyGUI -platform minimal /galaxyZooModel_file_or_folder /imageParameterFile /SDSS_targetInfoFolder <WorkDir>
