//To compile: g++ -o open_anotations open_anotations.cpp `pkg-config --cflags --libs oepncv` -l sqlite3

#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sqlite3.h>
#include <string>
#include <typeinfo>

using namespace cv;
using namespace std;

//Classe de cada caixa a ser desenhada
class Box
{
    private:
        int object_id, frame_number,
            x_top_left, y_top_left,
            x_bottom_right, y_bottom_right;

    public:
        Box(int, int, int, int, int, int);
        Box();
        void print_data();
        int get_frame();
        int get_x_top_left();
        int get_y_top_left();
        int get_x_bottom_right();
        int get_y_bottom_right();
};


Box::Box (int id, int frame, int x_tl, int y_tl, int x_br, int y_br)
{
    object_id = id;
    frame_number = frame;
    x_top_left = x_tl;
    y_top_left = y_tl;
    x_bottom_right = x_br;
    y_bottom_right = y_br;
}

Box::Box ()
{
    object_id = 0;
    frame_number = 0;
    x_top_left = 0;
    y_top_left = 0;
    x_bottom_right = 0;
    y_bottom_right = 0;

}

void Box::print_data()
{
    cout << "object_id = " << object_id << endl;
    cout << "frame = " << frame_number << endl;
    cout << "x_top_left = " << x_top_left << endl;
    cout << "y_top_left = " << y_top_left << endl;
    cout << "x_bottom_right = " << x_bottom_right << endl;
    cout << "y_bottom_right = " << y_bottom_right << "\n" << endl;
}

int Box::get_frame(){
    return frame_number;
}

int Box::get_x_top_left(){
    return x_top_left;
}

int Box::get_y_top_left(){
    return y_top_left;
}

int Box::get_x_bottom_right(){
    return x_bottom_right;
}

int Box::get_y_bottom_right(){
    return y_bottom_right;
}

Vector<Box> open_database(char*, char*);
Vector<Box> get_frame_boxes(Vector<Box>, int);
static int callback(void *NotUsed, int argc, char **argv, char **azColName);
int playVideo(char*, Vector<Box>);
Mat draw_box(Mat , Box);

int main(int argc, char** argv)
{
    char* db_path;
    char* video_path;
    Vector<Box> boxes;

    video_path = "../Datasets/UrbanTracker\ Dataset\ -\ Single\ Camera/Rouen/rouen_video.avi";
    db_path = "../Datasets/UrbanTracker\ Dataset\ -\ Single\ Camera/Rouen/rouen_annotations/rouen_gt.sqlite";

    boxes = open_database(db_path, "bounding_boxes");
    playVideo(video_path, boxes);
    return 0;
}

//Função que roda o video
int playVideo(char* filename, Vector<Box> boxes)
{
    Vector<Box> squares;
    VideoCapture cap(filename);
    if(!cap.isOpened())
        return -1;

    int frame_number = 0;
    namedWindow("opa", 1);

    for(;;)
    {
        Mat frame;
        cap >> frame;
        squares = get_frame_boxes(boxes, frame_number);

        for (Vector<Box>::iterator it = squares.begin(); it != squares.end(); it++){
            frame = draw_box(frame, *it);
        }

        imshow("opa", frame);
        cout << "Frame: " << frame_number << endl;
        frame_number++;

        if(waitKey(30)>=0) break;
    }
    return 0;
}


Vector<Box> get_frame_boxes(Vector<Box> box_list, int frame_number)
{
    Vector<Box> frame_boxes;
    for (Vector<Box>::iterator it = box_list.begin() ; it != box_list.end(); it++){
        if(it->get_frame() == frame_number)
        {
            frame_boxes.push_back(*it);
        }
    }
    return frame_boxes;
}

Mat draw_box(Mat frame , Box box)
{
    Point top_left = Point(box.get_x_top_left(), box.get_y_top_left());
    Point bottom_right = Point(box.get_x_bottom_right(), box.get_y_bottom_right());


    rectangle(frame, top_left, bottom_right, Scalar(0, 255, 0), 2);
    return frame;
}

//Função que abre a base de dados
Vector<Box> open_database(char* filename, char* table_name)
{
    sqlite3 *db;
    char *zErrMsg = 0;
    int rc;
    Vector<Box> boxes_list;

    string sql_operation ("SELECT * FROM ");


    rc = sqlite3_open(filename, &db);

    if( rc ){
        cout << stderr << "Can't open database:" << sqlite3_errmsg(db) << endl;
        sqlite3_close(db);
    }

    sql_operation = sql_operation + table_name;

    //Realizar a leitura dos dados, aqui é referenciado o callback
    rc = sqlite3_exec(db, sql_operation.c_str(), callback, (void*)&boxes_list, &zErrMsg);


    if( rc!=SQLITE_OK ){
        cout << stderr << "SQL error:" << zErrMsg << endl;
        sqlite3_free(zErrMsg);
    }
    sqlite3_close(db);
    return boxes_list;
}


static int callback(void *param, int argc, char **argv, char **azColName)
{
    //Criando uma box com os dados obtidos no banco de dados
    Box mybox (atoi(argv[0]), atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
    Vector<Box> *box_list = (Vector<Box>*) param;
    box_list->push_back(mybox);

    return 0;
}

