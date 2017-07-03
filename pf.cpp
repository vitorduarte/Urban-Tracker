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
    int height, width;
    int area;
    cv::Point2f square_origin, square_end;
    float error;
    int height_cmp, width_cmp;
    int area_cmp;
    cv::Point2f square_origin_cmp;
    cv::Point2f center;
    cv::Point2f vel;
    cv::Mat template_obj;
    int label;

  public:
    MovObj(cv::Point2f square_origin_,int h_, int w_, int area_,int label_){

      label=label_;
      error=10;
      error=error/100;
      height = h_;
      width = w_;
      area = area_;
      square_origin=square_origin_;
      square_end.x = (square_origin.x+width);
      square_end.y = (square_origin.y+height);
      height_cmp = height*sqrt(1+error);
      width_cmp = width*sqrt(1+error);
      area_cmp = height_cmp*width_cmp;
      square_origin_cmp.x=square_origin.x-((square_origin.x+width_cmp)-(square_origin.x+width))/2;
      square_origin_cmp.y=square_origin.y-((square_origin.y+height_cmp)-(square_origin.y+height))/2;
      center.x=((square_origin.x)+(square_origin.x+width))/2;
      center.y=((square_origin.y)+(square_origin.y+height))/2;
    }
    //~MovObj();

    int get_height(){
      return(height);
    }

    int get_height_cmp(){
      return(height_cmp);
    }

    int get_width(){
      return(width);
    }

    int get_width_cmp(){
      return(width_cmp);
    }

    int get_area(){
      return(area);
    }

    cv::Point2f get_origin(){
      return(square_origin);
    }
    cv::Point2f get_end(){
      return(square_end);
    }
    cv::Point2f get_origin_cmp(){
      return(square_origin_cmp);
    }

    cv::Point2f get_center(){
      return(center);
    }

    int get_label(){
      return(label);
    }

    void set_label(int label_){
      label=label_;
    }

    cv::Mat get_template(cv::Mat source){
      if(get_width() != 0 && get_height() != 0){
        cv::Rect cropped_rectangle = cv::Rect(square_origin.x, square_origin.y,
                                            get_width(), get_height());
        template_obj = source(cropped_rectangle);
        return(template_obj);
      }
      cv::Rect cropped_rectangle = cv::Rect(square_origin.x, square_origin.y,
                                            get_width(), get_height());
      template_obj = source(cropped_rectangle);

      return(template_obj);
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
    CvSize win_size;
    int step,interval_pixels,tam_vel;
    int kernel_size;
    int threshold,ratio;
    int pyr_scale, levels, winsize, iterations, poly_n, poly_sigma;
    std::vector<cv::Vec4i> hierarchy;
    std::vector<std::vector<cv::Point> > contours;
    std::vector<MovObj> mov_objects_prev;
    std::vector<MovObj> mov_objects_next;


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

          mov_objects_prev=mov_objects_next;

          mov_objects_next.clear();

          cv::cvtColor(frame,next_frame,cv::COLOR_BGR2GRAY);
          cv::GaussianBlur(next_frame,next_frame,cv::Size(kernel_size,kernel_size) ,0,0,cv::BORDER_DEFAULT);

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

          if (mov_objects_prev.size()!=0 && mov_objects_next.size()!=0){
            compare_mov_obj();
          }

          show_frame();

          prev_frame=next_frame.clone();
        }

        std::cout << "i = " << i << '\n';
        i++;
      }
    }

    void show_frame(){
      cv::imshow("Video",frame);
      cv::imshow("OPT_FLOW",flow_mask);
      //cv::imshow("Contours",cont_mask);
    }

    void create_trackbars(){
      cv::createTrackbar("Pyramid Scale", "Trackbars", &pyr_scale, 9);
      cv::createTrackbar("Levels", "Trackbars", &levels, 10);
      cv::createTrackbar("Window Size", "Trackbars", &winsize, 50);
      cv::createTrackbar("Iterations", "Trackbars", &iterations, 10);
      cv::createTrackbar("Pixel Neighborhood Size", "Trackbars", &poly_n, 10);
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
      int flag=0;
      std::vector<std::vector<cv::Point> > contours_poly(contours.size());
      std::vector<cv::Rect> boundRect(contours.size());

      if (mov_objects_prev.size()==0) {
        flag=1;
      }

      for(int i=0;i<contours.size();i++){
        cv::approxPolyDP(cv::Mat(contours[i]),contours_poly[i],3,true );
        boundRect[i] = cv::boundingRect(cv::Mat(contours_poly[i]));
      }
      get_mov_obj(boundRect,flag);
      draw_rectangles(mov_objects_next);
    }

    void draw_rectangles(std::vector<MovObj> mov_objects){
      for(int i=0; i< mov_objects.size(); i++ ){
        if(mov_objects[i].get_label() != 0){
          rectangle(frame, mov_objects[i].get_origin(),mov_objects[i].get_end(),
                    cv::Scalar(0,0,255), 2, 8, 0 );
        }
      }
    }

    void get_mov_obj(std::vector<cv::Rect> boundRect,int flag){
      cv::Point2f origin_point_aux;
      for (int i=0;i<contours.size();i++){
        origin_point_aux.x=boundRect[i].x;
        origin_point_aux.y=boundRect[i].y;
        MovObj aux(origin_point_aux,boundRect[i].height,boundRect[i].width,boundRect[i].area(),i);
        mov_objects_next.push_back(aux);
      }
    }

    void compare_mov_obj(){
      cv::Point2f next_aux,prev_aux;
      int height, width;

      for(int i;i<mov_objects_next.size();i++){

        next_aux=mov_objects_next[i].get_center();

        for(int j;j<mov_objects_prev.size();j++){

          prev_aux=mov_objects_prev[i].get_origin_cmp();
          height=mov_objects_prev[i].get_height_cmp();
          width=mov_objects_prev[i].get_width_cmp();

          if(next_aux.x>=prev_aux.x&&next_aux.x<=(prev_aux.x+width)){
            if(next_aux.y>=prev_aux.y&&next_aux.y<=(prev_aux.y+height)){
              mov_objects_next[i].set_label(mov_objects_prev[j].get_label());
            }
            else{
              mov_objects_next[i].set_label(-1);
            }
          }
        }
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
