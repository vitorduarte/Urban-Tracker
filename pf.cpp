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

#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>

//--------------------- Classes ---------------------

class Opt_flow {
  private:
    cv::VideoCapture video;
    cv::String filename;
    cv::Mat frame;
    cv::Mat mov_mat,black_mat;
    cv::Mat next_frame;
    std::vector<float> vel_x;
    std::vector<float> vel_y;
    float std_error;
    cv::Vec3b frame_intensity,next_frame_intensity;
    float errors;
    int counter;
    CvSize win_size;
    int passo,interval_pixels,tam_vel;
    int kernel_size;


  public:

    Opt_flow(cv::VideoCapture video_ , cv::String filename_) {
      cv::namedWindow("Video");
      cv::namedWindow("Movement");
      video=video_;
      filename=filename_;
      std_error=80;
      counter=0;
      std_error=(std_error/100);
      passo=1;
      win_size.height=3;
      win_size.width=3;
      interval_pixels=5;
      tam_vel=1;
      kernel_size=3;

      video.open(filename);
    }

    ~Opt_flow(){
      cv::destroyWindow("Video");
      cv::destroyWindow("Movement");
    }

    void play(){
      int i=0;
      while (char(cv::waitKey(1))!='q'&&video.isOpened()){
        video >> next_frame;
        cv::cvtColor(next_frame,next_frame,cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(next_frame,next_frame,cv::Size(kernel_size,kernel_size) ,0,0,cv::BORDER_DEFAULT);

        if(i==0){
          mov_mat=next_frame.clone();
          blackfy();
          black_mat=mov_mat.clone();
        }

        if(i!=0&&i%passo==0){
          mov_mat=black_mat.clone();
          //std::cout << '\n';
          //std::cout << "FRAME: \n" << frame.size() << "|" << frame.channels() << '\n';
          //std::cout << "NEXT_FRAME: \n" << next_frame.size() << "|" << next_frame.channels() << '\n';
          //getchar();

          cv::calcOpticalFlowFarneback(frame, next_frame, mov_mat, .4, 1, 12, 2, 8, 1.2, 0);
          draw_flow();

          //if (i%passo==0) {
            show_frame();
          //}

        }

        frame=next_frame.clone();
        i++;
      }
    }

    void show_frame(){
      cv::imshow("Video",frame);
      //v::imshow("Movement",mov_mat);
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
            mov_mat.at<uint>(i,j)=255;
            //std::cout << "MUDOU O PIXEL" << '\n';
          }
        }
      }
      //std::cout << "TROCOU FRAME" << '\n';

    }

    void blackfy(){
      for(int i=0;i<mov_mat.rows;i++){
        for(int j=0;j<mov_mat.cols;j++){
          mov_mat.at<uint>(i,j)=0;
          //std::cout << "(" << i << "," << j << ")" << '\n';
        }
      }
    }

    void draw_flow(){
      for (int y = 0; y < frame.rows; y += interval_pixels) {
          for (int x = 0; x < frame.cols; x += interval_pixels)
          {
              // get the flow from y, x position * 3 for better visibility
              const cv::Point2f flowatxy = mov_mat.at<cv::Point2f>(y, x) * tam_vel;
              // draw line at flow direction
              cv::line(frame, cv::Point(x, y), cv::Point(cvRound(x + flowatxy.x), cvRound(y + flowatxy.y)), cv::Scalar(255, 0, 0));
              // draw initial point
              cv::circle(frame, cv::Point(x, y), 1, cv::Scalar(0, 0, 0), -1);
          }
      }
    }
};

class Tracker {


};


//--------------------- Global Variables ---------------------

//--------------------- Main Function ---------------------

int main(int argc, char *argv[]){
  cv::String filename;
  cv::VideoCapture video;
  filename=argv[1];


  Opt_flow opt_flow(video,filename);

  opt_flow.play();

  return(0);
}
