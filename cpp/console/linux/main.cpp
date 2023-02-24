#include <QtWidgets/QApplication>
#include <QObject>
#include <QThread>
#include <opencv2/opencv.hpp>

#include "trtengine.h"
#include "argparse.h"

using namespace std;

bool isnum(string s)
{
    stringstream sin(s);
    double t;
    char p;
    if(!(sin >> t))
        return false;
    if(sin >> p)
        return false;
    else
        return true;
}

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    auto args = argsutil::argparser("EdgeYOLO TensorRT argument parser.");

    args.add_argument<string>("json", "json config")
        .add_argument<string>("source", "video source")
        .add_option<double>("-c", "--conf", "confidence threshold, default is 0.25", 0.25)
        .add_option<double>("-n", "--nms", "NMS threshold, default is 0.5", 0.5)
        .add_option<bool>("-s", "--no-label", "do not draw labels. default is false", false)
        .add_option<bool>("-l", "--loop", "loop playback. default is false", false)
        .add_help_option()
        .parse(argc, argv);


//    QString json_path = QString("E:/code/python/project/edgeyolo/output/export/pth2trt/edgeyolo_tiny_coco/640x640_batch16_int8.json");
//    QString source = QString("E:/videos/test.avi");

    QString json_path = QString::fromStdString(args.get_argument_string("json"));
    QString source = QString::fromStdString(args.get_argument_string("source"));

    cv::VideoCapture cap;
    if (isnum(source.toStdString()))
        cap.open(source.toInt());
    else
        cap.open(source.toStdString());

    cout<<json_path.toStdString()<<endl;

    cudaSetDevice(0);
    auto detector = TensorRTEngine();
    cout<<detector.load_engine(json_path).toStdString()<<endl;
    if (!detector.engine_loaded)
        return -1;

    detector.set_conf_threshold(args.get_option_double("--conf"));
    detector.set_nms_threshold(args.get_option_double("--nms"));

    Detector *d = new Detector(cap, detector);

    d->draw_label = !args.get_option_bool("--no-label");
    d->loop = args.get_option_bool("--loop");

    d->start();
    
    cout<<"start"<<endl;
    
    while (d->isRunning()) {
        if (d->img_show.empty()) continue;
        break;
    }
    
    while (true){
        while (d->isRunning()){
            //if (d->img_show.empty()) continue;
            if(!d->updated) continue;
            d->updated = false;
            cv::imshow("test", d->img_show);
            if (cv::waitKey(10)==27){
                d->loop = false;
                break;
            }
        }
        if (!d->loop)
            break;
        else
            d->cap.open(source.toStdString());
    }
    cout<<"end"<<endl;

    //delete d;
//    return app.exec();
    return 0;
}
