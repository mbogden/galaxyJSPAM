#ifndef GALAXYMAINWINDOW_H
#define GALAXYMAINWINDOW_H

#include <QMainWindow>
#include <QDir>

QT_BEGIN_NAMESPACE
namespace Ui { class galaxyMainWindow; }
QT_END_NAMESPACE

class galaxyMainWindow : public QMainWindow
{
    Q_OBJECT

public:
    galaxyMainWindow(QWidget *parent = nullptr);
    ~galaxyMainWindow();

    void runManually(QString model, QString image, QString data);
    bool GUI = true;
public slots:
    //GUI methods
    void on_runSetupPushButton_clicked();

private slots:
    //Main methods to create data
    bool createModelDirectory(const QString &dirIn, const QString &inFile, const int &generation);
    QString infoFileFromLine(const QString &sdss, const QString &run, const QString& sdssFileLine, const QString &generation);
    bool createRunsFromFile(const QFileInfo& fi, const QDir& inDir, int generation);
    QStringList getImageDataFromDir(const QString& sdss);

    //Helper methods
    //  Pads with 0's to n size
    //  padTo(7, 5)  -> 00007
    //  padTo(90, 3) -> 090
    QString padToN(int in, int n)
    {
        QString ret = QString::number(in);
        QString app;

        for (int i = 0; i < (n - ret.length()); i++)
            app = "0" + app;

        ret = app + ret;
        return ret;
    }

    //GUI methods
    void on_selectFilePushButton_clicked();
    void on_selectWorkDirPushButton_clicked();
    void on_selectImageDirPushButton_clicked();

private:
    Ui::galaxyMainWindow *ui;
    QDir OURDIR;
    QString IMAGE_PARAM_FILE;
    QString SDSS_MODELS_DIR;
    QString SDSS_DATA_DIR;
    int INFOFILESWRITTEN = 0;
};
#endif // GALAXYMAINWINDOW_H
