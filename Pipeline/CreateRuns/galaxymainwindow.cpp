#include "galaxymainwindow.h"
#include "ui_galaxymainwindow.h"
#include <QFile>
#include <QDir>
#include <QDirIterator>
#include <QFileDialog>
#include <QMessageBox>
#include <QRegularExpression>
#include <QDebug>
#include <QInputDialog>
#include <iostream>

galaxyMainWindow::galaxyMainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::galaxyMainWindow)
{
    ui->setupUi(this);
    QDir d;
    ui->workDirLabel->setText("Working dir: " + d.currentPath());

    //Dev ease of use
    QDir d2;
    d2.setPath("Input_Data");

    if (d2.exists())
    {
        //Set image dir, if it exists
        if (d2.exists("image_parameters/param_v3_default.txt"))
        {
            IMAGE_PARAM_FILE = d2.absoluteFilePath("image_parameters/param_v3_default.txt");
            ui->imageDirLabel->setText("Image parameter file: " + IMAGE_PARAM_FILE);
        }

        //Set zoo_models dir if it exists
        if (d2.exists("zoo_models"))
        {
            SDSS_MODELS_DIR = d2.absoluteFilePath("zoo_models");
            ui->sdssFilesLabel->setText("Galaxy zoo models folder: " + SDSS_MODELS_DIR);
        }

        //Set data dir if it exists
        if (d2.exists("targets"))
        {
            SDSS_DATA_DIR = d2.absoluteFilePath("targets");
            ui->dataDirLabel->setText("SDSS target information folder: " + SDSS_DATA_DIR);
        }
    }
    else
    {
        //Setup is handled by calling function in main.cpp for headless use
    }
}

galaxyMainWindow::~galaxyMainWindow()
{
    delete ui;
}

//Allows the user to select a single file to prepare or dir of them
void galaxyMainWindow::on_selectFilePushButton_clicked()
{
    bool ok;
    QString res = QInputDialog::getItem(this, "Single file or directory?", "Please select an existing dir of files or file", {"File", "Dir"}, 0, false, &ok);

    if (!ok)
    {
        QMessageBox::information(this, "Error", "Please select something");
        return;
    }

    if (res == "File")
    {
        QFileInfo fi(QFileDialog::getOpenFileName(this, "Select input file(s)", OURDIR.currentPath()));

        if (fi.exists())
        {
            ui->sdssFilesLabel->setText("Galaxy zoo model file: " + fi.absoluteFilePath());
            SDSS_MODELS_DIR = fi.absoluteFilePath();
        }
        else
            QMessageBox::information(this, "Erorr", "Selected file " + fi.absoluteFilePath() + " does not exist");
    }
    else
    {
        QDir d(QFileDialog::getExistingDirectory(this, "Select input folder", OURDIR.currentPath()));

        if (d.exists())
        {
            ui->sdssFilesLabel->setText("Galaxy zoo models folder: " + d.absolutePath());
            SDSS_MODELS_DIR = d.absolutePath();
        }
        else
            QMessageBox::information(this, "Erorr", "Selected dir " + d.path() + " does not exist");
    }
}

void galaxyMainWindow::on_selectImageDirPushButton_clicked()
{
    QFileInfo f(QFileDialog::getOpenFileName(this, "Select image parameter file", OURDIR.currentPath()));

    if (f.exists())
    {
        ui->imageDirLabel->setText("Image parameter file: " + f.absoluteFilePath());
        IMAGE_PARAM_FILE = f.absoluteFilePath();
    }
    else
        QMessageBox::information(this, "Erorr", "Selected image parameter file " + f.absoluteFilePath() + " does not exist");
}

void galaxyMainWindow::on_dataDirPushButton_clicked()
{
    QDir d(QFileDialog::getExistingDirectory(this, "Select SDSS target information folder", OURDIR.currentPath()));

    if (d.exists())
    {
        ui->dataDirLabel->setText("SDSS target information folder: " + d.absolutePath());
        SDSS_DATA_DIR = d.absolutePath();
    }
    else
        QMessageBox::information(this, "Erorr", "Selected folder " + d.path() + " does not exist");
}

void galaxyMainWindow::on_selectWorkDirPushButton_clicked()
{
    QDir d = QFileDialog::getExistingDirectory(this, "Please select the working directory", d.currentPath());
    if (d.exists())
    {
        OURDIR = d;
        ui->workDirLabel->setText("Working dir: " + d.absolutePath());
    }
}

//Action for the gui go button
void galaxyMainWindow::on_runSetupPushButton_clicked()
{
    QString modelsDir = SDSS_MODELS_DIR;
    QFile f(modelsDir);
    if (!f.exists())
    {
        if (GUI)
            QMessageBox::information(this, "Error", "Current zoo models file/dir does not exist");
        else
            qDebug() << "Current galaxy zoo models file/dir: " + SDSS_MODELS_DIR + " does not exist";
        return;
    }

    QFileInfo id(IMAGE_PARAM_FILE);
    if (!id.exists())
    {
        if (GUI)
            QMessageBox::information(this, "Error", "Current image parameter file does not exist");
        else
            qDebug() << "Current image parameter file: " + IMAGE_PARAM_FILE + " does not exist";
        return;
    }

    QString dataDir = SDSS_DATA_DIR;
    QDir dd(dataDir);
    if (!dd.exists())
    {
        if (GUI)
            QMessageBox::information(this, "Error", "Current sdss data dir does not exist");
        else
            qDebug() << "Current SDSS target information folder: " + SDSS_DATA_DIR + " does not exist";
        return;
    }

    QString setupDir = OURDIR.absolutePath();

    int generation = 0;
    int count = 0;
    createModelDirectory(setupDir, modelsDir, generation, count);

    f.close();

    std::cout << "Initilized " + QString::number(count).toStdString() + " simulation directories in:\n" + setupDir.toStdString() + "\n";
}

//Generates the directory tree
bool galaxyMainWindow::createModelDirectory(const QString& dirIn, const QString& inFile, const int& generation, int& count)
{
    count = 0;
    QDir wd(dirIn);
    QFile inF(inFile);

    //Check if work dir exists
    if (!wd.exists())
    {
        QMessageBox::information(this, "Error", "Work dir: " + dirIn + " does not exist");
        return false;
    }

    //Check if info file exists
    if (!inF.exists())
    {
        QMessageBox::information(this, "Error", "Input file: " + inFile + " does not exist");
        return false;
    }

    //Check if we can make the sdss # dir
    QFileInfo fi(inF);

    //Check if inF was a directory, if so loop over the files there instead of just one
    if (fi.isDir())
    {
        QDirIterator di(fi.filePath());
        //Only do sdss files that have a corresponding file in the dat dir
        //createRunsFromFile will not make a run dir for lines without scoring info
        while (di.hasNext())
        {
           QFileInfo lfi(di.next());

           if (lfi.fileName().contains(QRegularExpression("[0-9]+\\.txt")))
           {
               QFileInfo datMatch(SDSS_DATA_DIR + "/" + lfi.fileName().replace(".txt", ""));

               if (datMatch.exists())
               {
#ifdef DEBUG
                   qDebug() << "Found matching dat for sdss file " + lfi.fileName();
#endif
                   bool b = createRunsFromFile(lfi, wd, generation);
                   if (!b)
                   {
                       return false;
                   }
                   else
                   {
                       count++;
                   }
               }
            }
        }
    }
    else
    {
        count = 1;
        return createRunsFromFile(fi, wd, generation);
    }

    return true;
}

//Generates the run directories from an input file
bool galaxyMainWindow::createRunsFromFile(const QFileInfo& fi, const QDir& inDir, int generation)
{
    QString inFile = fi.filePath();
    QFile inF(inFile);
    QDir wd = inDir;
    QString sdss = fi.fileName().replace(".txt", "");
    if (!wd.mkdir(sdss))
    {
        if (GUI)
            QMessageBox::information(this, "Error", "Could not make dir: " + wd.path() + "/" + fi.fileName().replace(".txt", ""));
        return false;
    }
    wd.setPath(wd.path() + "/" + sdss);

    //Make the parameter dir & associated actions
    QString paraDirName = "sdssParameters";
    if (!wd.mkdir(paraDirName))
    {
        if (GUI)
            QMessageBox::information(this, "Error", "Could not make dir: " + wd.path() + paraDirName);
        return false;
    }
    else
    {
        QDir pd(wd.absoluteFilePath(paraDirName));
        //Copy the image parameter file to this new dir
        QFileInfo imf(IMAGE_PARAM_FILE);
        if (!QFile::copy(IMAGE_PARAM_FILE, pd.absoluteFilePath(imf.fileName())))
                qDebug() << "Failed to copy " + IMAGE_PARAM_FILE + " to " + pd.absolutePath();

        //Make parameters.txt
        QString ptxt;
        QStringList imageData = getImageDataFromDir(sdss);
        ptxt += "\n###  Target Image Data  ###\n";
        ptxt += "target_zoo.png " + imageData.value(1) + " " + imageData.value(2) + "\n";
        ptxt += "primary_luminosity " + imageData.value(3) + "\n";
        ptxt += "secondary_luminosity " + imageData.value(4) + "\n";

        ptxt += "\n###  Model Image Parameters  ###\n";

        ptxt += "default " + imf.fileName() + "\n";

        QFile pf(pd.absoluteFilePath("parameters.txt"));
        if (!pf.open(QIODevice::WriteOnly))
        {
            qDebug() << "Could not open for writing: " + pd.absoluteFilePath("parameters.txt");
            return false;
        }
        pf.write(ptxt.toUtf8());
        pf.close();

        //Copy imageData[0], the .png file, to target_zoo.png
        QFile::copy(imageData[0], pd.absoluteFilePath("target_zoo.png"));
    }

    //Check if we can make the generation dir
    QString genDirName = "gen" + padToN(generation, 3);
    if (!wd.mkdir(genDirName))
    {
        if (GUI)
            QMessageBox::information(this, "Error", "Could not make dir: " + wd.path() + genDirName);
        return false;
    }

    //Open info file
    if (!inF.open(QIODevice::ReadOnly))
    {
        if (GUI)
            QMessageBox::information(this, "Error", "Could not open file: " + inFile);
        return false;
    }

    //Now, make some stuff
    QStringList fileContents = QString(inF.readAll()).split("\n", QString::SkipEmptyParts);

    inF.close();

    wd.setPath(wd.path() + "/"+ genDirName);
    int i = 0;
    for (QString line: fileContents)
    {        
        //Make the info.txt information
        QString toWrite = infoFileFromLine(sdss, padToN(i, 5), line, padToN(generation, 3));
        //Skip the iteration if no scoring information is detected
        if (toWrite == "")
        {
            continue;
        }

        QString iPadded = padToN(i, 5);
        //Check if we can make the run dir
        QString folderName = "/run_" + iPadded;
        if (!wd.mkdir(wd.path() + folderName))
        {
            if (GUI)
                QMessageBox::information(this, "Error", "Could not make dir: " + folderName);
            return false;
        }

        //Make other required dirs
        if (!wd.mkdir(wd.path() + folderName +"/particle_files"))
        {
            if (GUI)
                QMessageBox::information(this, "Error", "Could not make dir: " + wd.path() + folderName + "/particle_files");
            return false;
        }
        if (!wd.mkdir(wd.path() + folderName + "/model_images"))
        {
            if (GUI)
                QMessageBox::information(this, "Error", "Could not make dir: " + wd.path() + folderName + "/model_images");
            return false;
        }
        if (!wd.mkdir(wd.path() + folderName + "/misc_images"))
        {
            if (GUI)
                QMessageBox::information(this, "Error", "Could not make dir: " + wd.path() + folderName + "/misc_images");
            return false;
        }

        //Finally, write our info file
        QFile dirInfoFile(wd.path() + "/" + folderName + "/info.txt");
        if (!dirInfoFile.open(QIODevice::WriteOnly))
        {
            if (GUI)
                QMessageBox::information(this, "Error", "Could not open file: " + wd.path() + "/" + folderName + "/info.txt");
            return false;
        }
        INFOFILESWRITTEN++;

        QString rName = folderName.remove(0, 1);

        dirInfoFile.write(toWrite.toUtf8());
        dirInfoFile.close();

        i++;
    }

    std::cout << "Initilized " + QString::number(i).toStdString() + " runs for sdss " + sdss.toStdString() + "\n";

    return true;
}

//Creates the initial info.txt contents
//Return "" if failed
QString galaxyMainWindow::infoFileFromLine(const QString& sdss, const QString& run, const QString& sdssFileLine, const QString& generation)
{
    QString ret ="###  Model Data  ###\n";
    QStringList parts = sdssFileLine.split(QRegularExpression("\t"));
    QStringList descParts = parts[1].split(",");
    QStringList scoreParts = parts[0].split(",");

    //Reject models with no scores, only use complete information
    if (scoreParts.length() != 4)
    {
#ifdef DEBUG
        qDebug() << "DEBUG: model data line may be incomplete, left of tab is detected as wrong: " + sdssFileLine;
#endif
        return "";
    }

    ret += "sdss_name " + sdss + "\n";
    ret += "generation " + generation + "\n";
    ret += "run_number " + run + "\n";
    ret += "model_data " + parts.value(1) + "\n";
    ret += "human_score " + scoreParts.value(1) + "\n";
    ret += "wins/total " + scoreParts.value(2) + "/" + scoreParts.value(3)  + "\n";

    return ret;
}

//Searches in data dir for sdss folder and finds the png, meta, and pair files inside
//Returns empty list if nothing found
//If found, returns
QStringList galaxyMainWindow::getImageDataFromDir(const QString& sdss)
{
    QStringList ret;
    QStringList nameFilter(sdss);
    QDir directory(SDSS_DATA_DIR);
    directory.setFilter(QDir::Dirs);
    QStringList txtFilesAndDirectories = directory.entryList(nameFilter);

    //Few directories actually exist
    if (txtFilesAndDirectories.length() != 0)
    {
        QFileInfo meta (SDSS_DATA_DIR + "/" + sdss + "/sdss" + sdss + ".meta" );
        if (!meta.exists())
        {
#ifdef DEBUG
            qDebug() << "No sdss" + sdss +".meta file ";
#endif
            return ret;
        }
        QFileInfo pair (SDSS_DATA_DIR + "/" + sdss + "/sdss" + sdss + ".pair" );
        if (!pair.exists())
        {
#ifdef DEBUG
            qDebug() << "No sdss" + sdss +".pair file ";
#endif
            return ret;
        }
        QFileInfo png (SDSS_DATA_DIR + "/" + sdss + "/sdss" + sdss + ".png" );
        if (!png.exists())
        {
#ifdef DEBUG
            qDebug() << "No sdss" + sdss +".png file ";
#endif
            return ret;
        }

        QFile mF(meta.absoluteFilePath());
        QFile pF(pair.absoluteFilePath());

        if (!mF.open(QIODevice::ReadOnly))
        {
#ifdef DEBUG
            qDebug() << "Could not open sdss" + sdss +".meta file ";
#endif
            return ret;
        }
        if (!pF.open(QIODevice::ReadOnly))
        {
#ifdef DEBUG
            qDebug() << "Could not open sdss" + sdss +".pair file ";
#endif
            return ret;
        }
        QStringList mContents = QString(mF.readAll()).split("\n", QString::SkipEmptyParts);
        QStringList pContents = QString(pF.readAll()).split("\n", QString::SkipEmptyParts);
        std::pair<double, double> pc1; //primary_center_1
        std::pair<double, double> sc2; //secondary_center_2
        QString pl; //primary_luminosity
        QString sl; //secondary_luminosity

        bool ok = true;
        //Iterate meta file
        for (QString s: mContents)
        {
            if (s.contains("px="))
            {
                QStringList line = s.split("=");
                pc1.first = line[1].toDouble(&ok);
                if (!ok)
                {
#ifdef DEBUG
                    qDebug() << "Could not convert " + line[1] + " to double";
#endif
                    return ret;
                }
            }

            if (s.contains("py="))
            {
                QStringList line = s.split("=");
                pc1.second = line[1].toDouble(&ok);
                if (!ok)
                {
#ifdef DEBUG
                    qDebug() << "Could not convert " + line[1] + " to double";
#endif
                    return ret;
                }
            }

            if (s.contains("sx="))
            {
                QStringList line = s.split("=");
                sc2.first = line[1].toDouble(&ok);
                if (!ok)
                {
#ifdef DEBUG
                    qDebug() << "Could not convert " + line[1] + " to double";
#endif
                    return ret;
                }
            }

            if (s.contains("sy="))
            {
                QStringList line = s.split("=");
                sc2.second = line[1].toDouble(&ok);
                if (!ok)
                {
#ifdef DEBUG
                    qDebug() << "Could not convert " + line[1] + " to double";
#endif
                    return ret;
                }
            }
        }

        for (QString s: pContents)
        {
            if (s.contains("primaryLuminosity="))
            {
                QStringList line = s.split("=");
                pl = QString::number(line[1].replace(";", "").toDouble(&ok));
                if (!ok)
                {
#ifdef DEBUG
                    qDebug() << "Could not convert " + line[1] + " to double";
#endif
                    return ret;
                }
            }

            if (s.contains("secondaryLuminosity="))
            {
                QStringList line = s.split("=");
                sl = QString::number(line[1].replace(";", "").toDouble(&ok));
                if (!ok)
                {
#ifdef DEBUG
                    qDebug() << "Could not convert " + line[1] + " to double";
#endif
                    return ret;
                }
            }
        }

        ret.append(png.absoluteFilePath());
        ret.append(QString::number(pc1.first) + " "+ QString::number(pc1.second));
        ret.append(QString::number(sc2.first) + " "+ QString::number(sc2.second));
        ret.append(pl);
        ret.append(sl);

        mF.close();
        pF.close();

#ifdef DEBUG
        qDebug() << "Successfully retreived image data for sdss " + sdss;
#endif
        return ret;
    }
    else
    {
#ifdef DEBUG
        qDebug() << "DEBUG: No " + sdss + " dir found in " + SDSS_DATA_DIR;
#endif
        return ret;
    }
}

void galaxyMainWindow::runManually(QString model, QString image, QString data)
{
    SDSS_MODELS_DIR = model;
    IMAGE_PARAM_FILE = image;
    SDSS_DATA_DIR = data;

    on_runSetupPushButton_clicked();
}

//Call before you run manually, else whats the point
bool galaxyMainWindow::setWorkDir(const QString& in)
{
    QDir d(in);

    if (!d.exists())
    {
        qDebug() << "Work dir folder: " + in + " does not exist.";
        return false;
    }

    OURDIR = d;
    return true;
}
