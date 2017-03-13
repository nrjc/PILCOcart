// Program to test delay of webcam

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;

#include <armadillo>
using namespace arma;

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cv.h>
#include <highgui.h>
#include <time.h>
#include <sys/time.h>
#include <fstream>
#include <ctime>
#include "boost/thread.hpp"

extern "C" {
#include "s626drv.h"
#include "App626.h"
#include "s626mod.h"
#include "s626core.h"
#include "s626api.h"
}

#include <pthread.h>
using namespace std;

// Some definitions used in setting up the sensor
#define RANGE_10V   0x00    // Range code for ADC ±10V range.
#define RANGE_5V    0x10    // Range code for ADC ±5V range.
#define EOPL      0x80    // ADC end-of-poll-list marker.
#define CHANMASK    0x0F    // ADC channel number mask.

#define MAX_U      10
//#define LENP2    65
#define LENP2      136.6
#define SIDE_LIMIT 1125 //1250
#define RAND_SIDE_LIMIT 500

float calc_expected_pos(float sens_pos, float sens_vel, float dt){
  return sens_pos - sens_vel * dt;
}

//TODO: initialise + infer the actual length scales from initial rod length
void app(DWORD board)
{;}


void applyController(int frame_interval, float dt, float sens_pos, float sens_vel, int &signal, bool &return_mid){
  static int call_count = 0;
  float expected_sens_pos = calc_expected_pos(sens_pos, sens_vel, dt);
  bool failsafe = (expected_sens_pos>RAND_SIDE_LIMIT || expected_sens_pos<-RAND_SIDE_LIMIT);
  if (failsafe || call_count%frame_interval == 0)
    {
      float r = (float) 1.0;
      signal = (int) ((r-0.5)*20000);
      if(failsafe){
        if(signal * sens_pos >0)
          signal *=-1;
      }
      cout << "Sens pos is:" << sens_pos << endl;
      cout << "Input force is:" << signal << endl;
      S626_WriteDAC(0,0, signal);
    }
  call_count = call_count + 1;
}

using namespace std;

IplImage* GetColourImage(IplImage* imgHSV, int h_low, int s_low, int v_low, int h_high, int s_high, int v_high){
       IplImage* imgThresh=cvCreateImage(cvGetSize(imgHSV),IPL_DEPTH_8U, 1);
       cvInRangeS(imgHSV, cvScalar(h_low,s_low,v_low), cvScalar(h_high,s_high,v_high), imgThresh);
       return imgThresh;
}
//                hl    sl   vl  hh  sh   vh
#define GREENTHRESH 50, 130, 80, 90, 256, 256
#define ORANGETHRESH 10, 150, 200, 20, 256, 256
#define REDTHRESH 320, 130, 80, 360, 256, 256


void getCameraFrame(IplImage** &img, CvCapture** &capture, int* &latest_frame_number/*, int* &processed_frame_number*/){
  while(1){
    while(!cvGrabFrame((*capture)));
    (*img) = cvRetrieveFrame((*capture));
    (*latest_frame_number)++;
  }
}

void processImage(IplImage* frame, float* posX, float* posY, int h_low, int s_low, int v_low, int h_high, int s_high, int v_high){
    // Holds the Red thresholded image (green = white, rest = black)
    IplImage* imgThresh = GetColourImage(frame, h_low, s_low, v_low, h_high, s_high, v_high);
    CvScalar c = cvAvg(imgThresh);
    
    cout << "Value of scalar"<<c.val[0]<<","<<c.val[1]<<","<<c.val[2]<<","<<c.val[3]<<endl;
    //cout << endl << "Area of green part = " << greenarea << endl;

    (*posX) = c.val[0];
    (*posY) = 0;

    cvReleaseImage(&imgThresh);
}



int main()
{
  float curr_time = 0; // A float for the current time in s

  // Initialise connection to board
  printf("Open\n");
  S626_OpenBoard( 0, 0, app, 1 );
  printf("Done\n");

  // Creating files for data output
  ofstream state_data;
  state_data.open("state.txt");

  // Set up the sensor reader
  BYTE        poll_list[16];    // List of items to be digitized.
  WORD        databuf[16];    // Buffer to receive digitized data.

  // Populate the poll list.
  poll_list[0]  =  0 | RANGE_10V;     // Chan 0, ±10V range.
  poll_list[1]  =  1 | RANGE_10V;     // Chan 1, ±10V range.
  poll_list[2]  =  2 | RANGE_10V;     // Chan 2, ±10V range.
  poll_list[3]  =  3 | RANGE_10V;     // Chan 3, ±10V range.
  poll_list[4]  =  4 | RANGE_10V;     // Chan 4, ±10V range.
  poll_list[5]  =  5 | RANGE_10V;     // Chan 5, ±10V range.
  poll_list[6]  =  6 | RANGE_10V;     // Chan 6, ±10V range.
  poll_list[7]  =  7 | RANGE_10V;     // Chan 7, ±10V range.
  poll_list[8]  =  8 | RANGE_5V;      // Chan 8,  ±5V range.
  poll_list[9]  =  9 | RANGE_5V;      // Chan 9,  ±5V range.
  poll_list[10] = 10 | RANGE_5V;      // Chan 10, ±5V range.
  poll_list[11] = 11 | RANGE_5V;      // Chan 11, ±5V range.
  poll_list[12] = 12 | RANGE_5V;      // Chan 12, ±5V range.
  poll_list[13] = 13 | RANGE_5V;      // Chan 13, ±5V range.
  poll_list[14] = 14 | RANGE_5V;      // Chan 14, ±5V range.
  poll_list[15] = 15 | RANGE_5V | EOPL; // Chan 15, ±5V range, mark as list end.

  // Prepare for A/D conversions by passing the poll list to the driver.
  S626_ResetADC( 0, poll_list );

  // Initialising accurate timers
  struct timespec t1, t2;

  // Initialise saving of previous angle
  float pp_cos = -1; //Usually I will start the pendulum at bottom position and cos is -1 there
  float pp_sin = 0;
  float last_cos = -1; //Usually I will start the pendulum at bottom position and cos is -1 there
  float last_sin = 0;


  // Boolean used when storing first position
  bool do_once = 1;
  bool initiating_angles = 1;
  bool return_mid = 0;
  bool end_return = 0;
  float startLocation;
  float startTime;
  float dt = 0;
  int latest_frame_number = 0;
  int processed_frame_number = 0;


  // Initialise the initial control input
  int signal = 0;
  float p_signal = 0;
  float pp_signal = 0;
  float ppp_signal = 0;

  // Initialize capturing live feed from the camera
  CvCapture* capture = 0;
  capture = cvCaptureFromCAM(CV_CAP_ANY);
  int width = 640;
  int height = 480;
  cvSetCaptureProperty( capture, CV_CAP_PROP_FRAME_WIDTH, width );
  cvSetCaptureProperty( capture, CV_CAP_PROP_FRAME_HEIGHT, height );
  //cvSetCaptureProperty( capture, CV_CAP_PROP_FPS , 60);

  // Initialise saving the video
    CvVideoWriter *writer = cvCreateVideoWriter("./test.avi",
                  CV_FOURCC('D','I','V','X'),
                  30,
                  cvSize(width, height)
                  );
  
  
  // Couldn't get a device? Throw an error and quit
  if(!capture)
  {
    printf("Could not initialize capturing...\n");
    return -1;
  }
  // The windows we'll be using
  //cvNamedWindow("video");

  // This image holds the "scribble" data...
  // the tracked positions of the colour
  IplImage* imgScribble = NULL;


  int syncro_counter = 0; // Variable to make each iteration start at a specific time, not used atm
  //float dt_syncro = 0.08;

  // Start the clock
  clock_gettime(CLOCK_MONOTONIC,  &t1);
  // Initialise the variable that holds the previous time
  float last_time = 0;

  // Set upper coordinate of the frame
  int upperc = 0;

  // Will hold a frame captured from the camera
  IplImage* img = 0;

  // Start separate thread that captures images continuously
  boost::thread CapturingImages(getCameraFrame, &img, &capture, &latest_frame_number);
  
  // Set motor to 0
  S626_WriteDAC (0, 0, 0);
  // Allow camera to run for a bit
  dt = 0;
  int initiation_time = 2;
  last_time = 0;
  while(dt < initiation_time){
    clock_gettime(CLOCK_MONOTONIC,  &t2);
    curr_time = (t2.tv_sec - t1.tv_sec) + (double) (t2.tv_nsec - t1.tv_nsec) * 1e-9;
    dt = curr_time - last_time;
    }
  cout << "Dt is:" << dt << endl;
  bool start = true;

  
  // An infinite loop
  while(true)
  {
    // Will hold a frame captured from the camera
    //IplImage* img = 0;
    if(do_once) {
      clock_gettime(CLOCK_MONOTONIC,  &t2);
      last_time = (t2.tv_sec - t1.tv_sec) + (double) (t2.tv_nsec - t1.tv_nsec) * 1e-9;
    }
    if (!start && curr_time > (initiation_time+0.5)) start = 1; // Run for 0.5 sec before applying control and storing pos
    // Initialise dt 
    //dt = 0;

    // Wait for a frame to appear
    while(latest_frame_number - processed_frame_number < 2) {cout<<"";}


    IplImage* video = cvCreateImage(cvGetSize(img), 8, 3);
    cvCopy(img,video);
    IplImage* frame = cvCreateImage(cvGetSize(video), 8, 3);


    processed_frame_number = latest_frame_number;

    // Finds time
    clock_gettime(CLOCK_MONOTONIC,  &t2);
    curr_time = (t2.tv_sec - t1.tv_sec) + (double) (t2.tv_nsec - t1.tv_nsec) * 1e-9;

    // Finds time step between frames
    dt = curr_time - last_time;

    // Clear the scribble on each iteration, so I dont get clutter
    cvSet(imgScribble, cvScalar(0,0,0));

    // Convert tp HSV colour space, note that the output image will look weird
    cvCvtColor(img, frame, CV_BGR2HSV);


    // If we couldn't grab a frame quit
    if(!frame)
      break;

    // If this is the first frame, we need to initialize it
    if(imgScribble == NULL)
    {
      imgScribble = cvCreateImage(cvGetSize(img), 8, 3);
    }


    static float redposX = 0;
    static float redposY = 0;

    float redlastX = redposX;
    float redlastY = redposY;

    processImage(frame, &redposX, &redposY, REDTHRESH);
    cout << "Brightness is: " << redposX << endl;


    if(do_once) {
      //startLocation = orangeposX;
      startLocation = width/2;
    }

    if(do_once){
      do_once = 0;
    }
    dt = 0.0333;

    
    // Acquire readings from sensor
    S626_ReadADC (0, databuf);
    float sens_pos = (short)databuf[0]*(-0.08);
    float sens_vel = (short)databuf[1]*(-0.43);
      
      bool applycontrollerbuffer = true;

    // Applies a control signal every second time step
    if(start){
      applyController(5,dt,0,0,signal,applycontrollerbuffer);
    }
    

    last_time = curr_time;

    //cout << "Current time is " << time << "s" << endl;
    

    // Add the scribbling image and the frame... and we get a combination of the two
    cvAdd(video, imgScribble, video);
    

    // Write to videosens_vel = (short)databuf[1]*(-0.43);
    cvWriteFrame(writer, video);

    S626_ReadADC (0, databuf);
      
    //EXIT AND PRINT CURRENT TIME IF CHANGE IN POS IS DETECTED.
    if (redposX!=redlastX) {
        S626_WriteDAC (0, 0, 0);
        printf("The time taken to detect movement is:%f",curr_time-initiation_time-0.5);
        break;
    }

    if(start) syncro_counter++;
    if (curr_time > 12.5 + initiation_time) {S626_WriteDAC (0, 0, 0); break;}
  }

  // We're done using the camera. Other applications can now use it
  cvReleaseCapture(&capture);
  cvReleaseVideoWriter(&writer);
  return 0;
}
