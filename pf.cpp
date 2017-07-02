#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/legacy/legacy.hpp>

#include <math.h>
#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>

//--------------------- Classes ---------------------
class MovObj{
  private:
    int h, w;
    int area;
    cv::Point2f square_origin;
    cv::Point2f center;
    cv::Point2f vel;
    cv::Mat template_obj;

  public:
    MovObj(cv::Point2f square_origin_,int h_, int w_, int area_){
      //center = center_;
      //square_origin = square_origin_;
      h = h_;
      w = w_;
      area = area_;
      square_origin=square_origin_;
      center.x=((square_origin.x)+(square_origin.x+w))/2;
      center.y=((square_origin.y)+(square_origin.y+h))/2;
    }
    //~MovObj();

    int get_height(){
      return(h);
    }

    int get_width(){
      return(w);
    }

    int get_area(){
      return(area);
    }

    cv::Point2f get_origin(){
      return(square_origin);
    }
};

class Opt_flow {
  private:
    cv::VideoCapture video;
    cv::String filename;
    cv::Mat frame;
    cv::Mat opt_flow,flow_mask,cont_mask;
    cv::Mat prev_frame,next_frame;
    cv::Mat x_vals,y_vals;
    std::vector<float> vel_x;
    std::vector<float> vel_y;
    float std_error;
    cv::Vec3b frame_intensity,next_frame_intensity;
    float errors;
    int counter;
    CvSize win_size;
    int step,interval_pixels,tam_vel;
    int kernel_size;
    int threshold,ratio;
    int pyr_scale, levels, winsize, iterations, poly_n, poly_sigma;
    std::vector<cv::Vec4i> hierarchy;
    std::vector<std::vector<cv::Point> > contours;
    std::vector<MovObj> mov_objects;


  public:

    Opt_flow(cv::VideoCapture video_ , cv::String filename_) {
      cv::namedWindow("Video");
      cv::namedWindow("OPT_FLOW");
      cv::namedWindow("Contours");
      cv::namedWindow("Trackbars");
      //cv::namedWindow("OPT_FLOW-X");
      //cv::namedWindow("OPT_FLOW-Y");
      video=video_;
      filename=filename_;
      std_error=80;
      counter=0;
      std_error=(std_error/100);
      win_size.height=3;
      win_size.width=3;
      kernel_size=3;
      interval_pixels=2;
      tam_vel=1;
      step=5;
      threshold=2;
      ratio=1;
      pyr_scale = 5; //Will be divided for 10
      levels = 3;
      winsize = 15;
      iterations = 3;
      poly_n = 5;
      poly_sigma = 1.2;


      video.open(filename);
    }

    ~Opt_flow(){
      cv::destroyWindow("Video");
      //cv::destroyWindow("OPT_FLOW-X");
      //cv::destroyWindow("OPT_FLOW-Y");
      cv::destroyWindow("OPT_FLOW");
      cv::destroyWindow("Contours");
      cv::destroyWindow("Trackbars");
    }

    void play(){
      int i=0;
      while (char(cv::waitKey(1))!='q'&&video.isOpened()){
        video >> frame;

        if (i==0){
          cv::cvtColor(frame,prev_frame,cv::COLOR_BGR2GRAY);
          cv::GaussianBlur(prev_frame,prev_frame,cv::Size(kernel_size,kernel_size) ,0,0,cv::BORDER_DEFAULT);
        }

        if(i!=0&&i%step==0){
          //getchar();
          cv::cvtColor(frame,next_frame,cv::COLOR_BGR2GRAY);
          cv::GaussianBlur(next_frame,next_frame,cv::Size(kernel_size,kernel_size) ,0,0,cv::BORDER_DEFAULT);

          //std::cout << "PREV_FRAME:" << prev_frame.size() << '|' << prev_frame.channels() << '\n';
          //std::cout << "NEXT_FRAME:" << next_frame.size() << '|' << next_frame.channels() << '\n';

          cv::calcOpticalFlowFarneback(prev_frame, next_frame, opt_flow,
                                      pyr_scale/10.0, levels, winsize,
                                      iterations, poly_n, poly_sigma, 0);
          //draw_flow(opt_flow,frame);

          get_xvals(opt_flow);
          get_yvals(opt_flow);
          generate_flowmask(x_vals,y_vals);
          cv::GaussianBlur(flow_mask,flow_mask,cv::Size(kernel_size,kernel_size) ,0,0,cv::BORDER_DEFAULT);
          cv::threshold(flow_mask,flow_mask,threshold, threshold*ratio,cv::THRESH_BINARY);

          flow_mask.convertTo(cont_mask,CV_8U);

          cv::findContours(cont_mask,contours, hierarchy, CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
          get_contours(flow_mask);
          create_trackbars();

          show_frame();
          //print_moving_objects_vector();
          //getchar();

          prev_frame=next_frame.clone();
        }

        std::cout << "i = " << i << '\n';
        i++;
      }
    }

    void show_frame(){
      cv::imshow("Video",frame);
      cv::imshow("OPT_FLOW",flow_mask);
      cv::imshow("Contours",cont_mask);
    }

    void create_trackbars(){
      cv::createTrackbar("Pyramid Scale", "Trackbars", &pyr_scale, 9);
      cv::createTrackbar("Levels", "Trackbars", &levels, 10);
      cv::createTrackbar("Window Size", "Trackbars", &winsize, 50);
      cv::createTrackbar("Iterations", "Trackbars", &iterations, 10);
      cv::createTrackbar("Pixel Neighborhood Size", "Trackbars", &poly_n, 10);
    }

    void background_extr(){
      int i,j,aux_error;

      for(i=0;i<(frame.rows)-1;i++){
        for(j=0;j<(frame.cols)-1;j++){
          counter++;

          frame_intensity = frame.at<uint>(i,j);
          next_frame_intensity = next_frame.at<uint>(i,j);

          if (next_frame_intensity.val[0]==0){
            next_frame_intensity.val[0]=1;
          }
          errors=frame_intensity.val[0]/next_frame_intensity.val[0];

          if(errors<1-std_error||errors>1+std_error){
            opt_flow.at<uint>(i,j)=255;
            //std::cout << "MUDOU O PIXEL" << '\n';
          }
        }
      }
      //std::cout << "TROCOU FRAME" << '\n';

    }

    void blackfy(cv::Mat input){
      for(int i=0;i<input.rows;i++){
        for(int j=0;j<input.cols;j++){
          input.at<uint>(i,j)=0;
          //std::cout << "(" << i << "," << j << ")" << '\n';
        }
      }
    }

    void draw_flow(cv::Mat input, cv::Mat output){
      for (int y = 0; y < output.rows; y += interval_pixels) {
          for (int x = 0; x < output.cols; x += interval_pixels)
          {
              const cv::Point2f flowatxy = input.at<cv::Point2f>(y, x) * tam_vel;
              cv::line(output, cv::Point(x, y), cv::Point(cvRound(x + flowatxy.x), cvRound(y + flowatxy.y)), cv::Scalar(255, 0, 0));
              cv::circle(output, cv::Point(x, y), 1, cv::Scalar(0, 0, 0), -1);
          }
      }
    }

    void get_xvals(cv::Mat input){
      x_vals=cv::Mat::zeros(input.rows,input.cols, CV_32F);

      for (int i=0;i<input.rows;i++){
        for(int j=0;j<input.cols;j++){
          //std::cout << "X_VALS:" << x_vals.size() << '|' << x_vals.channels() << '|' << x_vals.type() << '\n';

          x_vals.at<float>(i,j)=input.at<cv::Point2f>(i,j).x;
        }
      }
    }

    void get_yvals(cv::Mat input){
      y_vals=cv::Mat::zeros(input.rows,input.cols, CV_32F);

      for (int i=0;i<input.rows;i++){
        for(int j=0;j<input.cols;j++){
          //std::cout << "Y_VALS:" << y_vals.size() << '|' << y_vals.channels() << '|' << y_vals.type() << '\n';

          y_vals.at<float>(i,j)=input.at<cv::Point2f>(i,j).y;
        }
      }
    }

    void generate_flowmask(cv::Mat x_matrix,cv::Mat y_matrix) {

      flow_mask=cv::Mat::zeros(x_matrix.rows,x_matrix.cols, CV_32F);

      for (int i=0;i<x_matrix.rows;i++){
        for(int j=0;j<x_matrix.cols;j++){
          flow_mask.at<float>(i,j)=magnitude(x_matrix.at<float>(i,j), y_matrix.at<float>(i,j));
        }
      }
    }

    float magnitude(float a, float b){
      float mag;

      mag=sqrt((a*a)+(b*b));

      return(mag);
    }

    void get_contours(cv::Mat input){
      std::vector<std::vector<cv::Point> > contours_poly(contours.size());
      std::vector<cv::Rect> boundRect(contours.size());

      for(int i=0;i<contours.size();i++){
        cv::approxPolyDP(cv::Mat(contours[i]),contours_poly[i],3,true );
        boundRect[i] = cv::boundingRect(cv::Mat(contours_poly[i]));
        draw_rectangles(boundRect);
        get_mov_obj(boundRect);
     }
    }

    void draw_rectangles(std::vector<cv::Rect> boundRect){
      for(int i=0; i< contours.size(); i++ ){
        rectangle(frame, boundRect[i].tl(), boundRect[i].br(), cv::Scalar(0,0,255), 2, 8, 0 );
      }
    }

    void get_mov_obj(std::vector<cv::Rect> boundRect){
      cv::Point2f origin_point_aux;
      for (int i=0;i<contours.size();i++){
        origin_point_aux.x=boundRect[i].x;
        origin_point_aux.y=boundRect[i].y;
        MovObj aux(origin_point_aux,boundRect[i].height,boundRect[i].width,boundRect[i].area());
        mov_objects.push_back(aux);
        /*std::cout << "Object i=" << i << ':' << '\n';
        std::cout << "Height:" << mov_objects[i].get_height() << '\n';
        std::cout << "Width:" << mov_objects[i].get_width() << '\n';
        getchar();*/
      }
    }

    void print_moving_objects_vector(){
      for(int i=0;i<mov_objects.size();i++){
        std::cout << "i=" << i << '\n';
        std::cout << "Height:" << mov_objects[i].get_height() << '\n';
        std::cout << "Width:" << mov_objects[i].get_width() << '\n';
        std::cout << "Area:" << mov_objects[i].get_area() << '\n';
        std::cout << "Origin Point:" << mov_objects[i].get_origin() << '\n';
        std::cout << "----------------------------------" << '\n';
      }
    }
};


//--------------------- Global Variables ---------------------

//--------------------- Main Function ---------------------

int main(int argc, char *argv[]){
  cv::String filename;
  cv::VideoCapture video;

  if (argc<=1){
    std::cout << "\n\nThere was no input video to run.\nPress ENTER to quit.\n>>";
    getchar();
    return(-1);
  }

  filename=argv[1];

  Opt_flow opt_flow(video,filename);
  opt_flow.play();

  return(0);
}
