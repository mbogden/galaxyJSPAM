#include "galaxymainwindow.h"

#include <iostream>
#include <QDebug>
#include <QApplication>

bool argvContains(int argc, char** argv, QStringList check)
{
    for (int i = 0; i < argc; i++)
    {
        QString s = argv[i];
        for (QString c: check)
        if (check.contains(s))
            return true;
    }

    return false;
}

int main(int argc, char *argv[])
{
    std::cout << "To run with GUI, do not specify a '-platform' option.\n"
                 "To run without GUI, use '-platform minimal'.\n"
                 "After the addition (or not) of the platform flag, the first arg is the SDSS models dir\n"
                 "the second the image parameter file, and the third the sdss data dir.\n"
                 "Usage: ./galaxyGUI <-platform minimal> /file1 /file2 /file3\n";
    QCoreApplication::setApplicationVersion("alpha-0.3");

    if (!argvContains(argc, argv, {"-platform"}))
    {
        std::cout << "Running with GUI\n";
        QApplication a(argc, argv);

        galaxyMainWindow w;
        w.show();
        return a.exec();
    }
    else
    {
        QApplication a(argc, argv);

        galaxyMainWindow w;
        QString imF;

        w.GUI = false;
        QCoreApplication::processEvents();
        std::cout << "Running in headless mode\n";
        if (argc != 4)
        {
            std::cout << "Wrong # of arguments specified\n";
        }

        QString modelDir = argv[1];
        QString imageFile = argv[2];
        QString dataDir = argv[3];

        std::cout << "SDSS Models dir: " + modelDir.toStdString() + "\n";
        std::cout << "Image parameter file: " + imageFile.toStdString() + "\n";
        std::cout << "SDSS Data dir: " + dataDir.toStdString() + "\n";
        w.runManually(modelDir, imageFile, dataDir);
        return 0;
    }
}
