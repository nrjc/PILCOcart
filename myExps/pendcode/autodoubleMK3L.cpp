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
#define SIDE_LIMIT 1125 //1250 //DAY!
//#define SIDE_LIMIT 850 // NIGHT!
#define RAND_SIDE_LIMIT 500

float calc_expected_pos(float sens_pos, float sens_vel, float dt){
  return sens_pos - sens_vel * dt;
}

template<class Matrix>
void print_matrix(Matrix matrix) {
    matrix.print(std::cout);
}

//provide explicit instantiations of the template function for 
//every matrix type you use somewhere in your program.
template void print_matrix<arma::mat>(arma::mat matrix);
template void print_matrix<arma::cx_mat>(arma::cx_mat matrix);

//TODO: initialise + infer the actual length scales from initial rod length
void app(DWORD board)
{;}

int CURRT=0; // GLOBAL VARIABLE

void applyController(float cart_pos, float cart_vel, float pen_angle, float dtangle, float p1, float p2, float p3, float p4, int &signal)
{
  float dist = cart_pos; // Distance from a central location
  int control_signal = p1*8000*pen_angle + p2*8000*dtangle - p3*8000*cart_vel + p4*dist*8000;

  if (control_signal > 5500) control_signal = 10000; // Upper cut off on demand signal
  if (control_signal < -5500) control_signal = -10000;

  S626_WriteDAC (0, 0, control_signal);
  signal = control_signal;
}

void applySimpleRandomControl(int frame_interval, float dt, float sens_pos, float sens_vel, int &signal, bool &return_mid){
  static int call_count = 0;
  float expected_sens_pos = calc_expected_pos(sens_pos, sens_vel, dt);
  bool failsafe = (expected_sens_pos>RAND_SIDE_LIMIT || expected_sens_pos<-RAND_SIDE_LIMIT);
  if (failsafe || call_count%frame_interval == 0)
    {
      float r = (float)(rand()%1000)/1000;
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

void applylearnedControllerL(mat weights, mat biases, mat state, int &signal, bool &return_mid)
{
    mat evaluation=weights.row(CURRT)*state+biases(CURRT);

    //Pass through squashing function
    evaluation = (9*sin(evaluation) + sin(3*evaluation))/8;
    signal = (int) (evaluation(0,0)*10000);

    
    S626_WriteDAC(0, 0, signal);

    CURRT=(CURRT+1)%biases.n_elem;
}

void applylearnedController(mat centers, mat weights, mat W, mat state, int &signal, bool &return_mid)
{

    W = diagmat(exp(-2*W));
    mat ones;
    ones.ones(50, 1); // 100 weights
    centers = centers - ones*state.t(); // Makes each row into a difference

    mat evaluation = centers%centers;
    evaluation = evaluation*W;
    evaluation = sum(evaluation, 1);
    evaluation = exp(-0.5*evaluation);
    evaluation = weights.t()*evaluation;

    //Pass through squashing function
    evaluation = (9*sin(evaluation) + sin(3*evaluation))/8;
    signal = (int) (evaluation(0,0)*10000);

    
    S626_WriteDAC(0, 0, signal);

}

using namespace std;

IplImage* GetColourImage(IplImage* imgHSV, int h_low, int s_low, int v_low, int h_high, int s_high, int v_high){
       IplImage* imgThresh=cvCreateImage(cvGetSize(imgHSV),IPL_DEPTH_8U, 1);
       cvInRangeS(imgHSV, cvScalar(h_low,s_low,v_low), cvScalar(h_high,s_high,v_high), imgThresh);
       return imgThresh;
}
//                hl    sl   vl  hh  sh   vh
//#define GREENTHRESH 50, 130, 80, 90, 255, 255
#define YELLOWTHRESH 50, 130, 80, 90, 255, 255 //end of project
#define ORANGETHRESH 10, 150, 200, 20, 255, 255 //end of project
//#define YELLOWTHRESH 25, 100, 150, 45, 255, 255
//#define GREENTHRESH 25, 100, 150, 45, 255, 255 //end of project
#define GREENTHRESH 150, 100, 150, 250, 255, 255


void getCameraFrame(IplImage** &img, CvCapture** &capture, int* &latest_frame_number/*, int* &processed_frame_number*/){
  while(1){
    while(!cvGrabFrame((*capture)));
    (*img) = cvRetrieveFrame((*capture));
    (*latest_frame_number)++;
  }
}

void processImage(IplImage* frame, float* posX, float* posY, int h_low, int s_low, int v_low, int h_high, int s_high, int v_high){
    // Holds the Green thresholded image (green = white, rest = black)
  IplImage* imgThresh = GetColourImage(frame, h_low, s_low, v_low, h_high, s_high, v_high);

    // Calculate the moments to estimate the position
    CvMoments *moments = (CvMoments*)malloc(sizeof(CvMoments));
    cvMoments(imgThresh, moments, 1);

    // The actual moment values
    double moment10 = cvGetSpatialMoment(moments, 1, 0);
    double moment01 = cvGetSpatialMoment(moments, 0, 1);
    double area = cvGetCentralMoment(moments, 0, 0);

    //cout << endl << "Area of green part = " << greenarea << endl;

    (*posX) = moment10/area;
    (*posY) = moment01/area;

    cvReleaseImage(&imgThresh);
    delete moments;
}



int main()
{
  float curr_time = 0; // A float for the current time in s

  // Initialise connection to board
  printf("Open\n");
  S626_OpenBoard( 0, 0, app, 1 );
  printf("Done\n");

  //Create initiate matrices and load variables for controller
  mat weights, biases, state;
  weights.load("weights.txt", raw_ascii);
  biases.load("biases.txt", raw_ascii);
  state.zeros(18,1);
  bool init_rollout=false;
  if(weights.n_rows == 0){
    init_rollout=true;
  }

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

  float pp_cos_2 = -1; //Usually I will start the pendulum at bottom position and cos is -1 there
  float pp_sin_2 = 0;
  float last_cos_2 = -1; //Usually I will start the pendulum at bottom position and cos is -1 there
  float last_sin_2 = 0;


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
  capture = cvCaptureFromCAM(0);
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
  cvNamedWindow("video");

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
  bool start = 0;

  // In case of random trial
  srand ( time(0) ); rand();

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

    
    static float greenposX = 0;
    static float greenposY = 0;

    float greenlastX = greenposX;
    float greenlastY = greenposY;

    static float orangeposX = 0;
    static float orangeposY = 0;

    float orangelastX = orangeposX;
    float orangelastY = orangeposY;

    static float yellowposX = 0;
    static float yellowposY = 0;

    float yellowlastX = yellowposX;
    float yellowlastY = yellowposY;


    processImage(frame, &greenposX, &greenposY, GREENTHRESH);     
    processImage(frame, &orangeposX, &orangeposY, ORANGETHRESH);
    processImage(frame, &yellowposX, &yellowposY, YELLOWTHRESH);

    if(do_once) {
      //startLocation = orangeposX;
      startLocation = width/2;
      orangelastX = orangeposX;
      orangelastY = orangeposY;
    }

    // Save previous position
    static float cart_pos = 0;
    static float p_pos = 0;
    static float pp_pos = 0;

    pp_pos = p_pos;
    p_pos = cart_pos;
    
    // Finds time step
    // Calculate position and velocity of cart
    float cart_velocity =  (orangeposX - orangelastX)/dt;
    cart_pos = orangeposX - startLocation; // Possibly change to 0 in the initialised valued
    // Calculate angle(actually just need ratio), then use dt to get the angular velocity
    // Need angle as well as a variable
    float a = greenposX - orangeposX;
    float b = -(greenposY - orangeposY); //'-' sign because coordinates of image are flipped
    float a_2 = yellowposX - greenposX; // a for second link
    float b_2 = -yellowposY + greenposY;

    float sin_angle = - a/sqrt(a*a + b*b); //'-' sign because we measure anti-clockwise
    float cos_angle = b/sqrt(a*a + b*b);
    float sin_angle_2 = - a_2/sqrt(a_2*a_2 + b_2*b_2); //'-' sign since measure anti-clockwise
    float cos_angle_2 = b_2/sqrt(a_2*a_2 + b_2*b_2);

    // Save the pendulum length for normalising purposes if wanted
    static float angle = 3.1416;
    
    static float pp_angle = 3.1416;
    static float p_angle = 3.1416;
    pp_angle = p_angle;
    p_angle = angle;
    angle = angle + asin(sin_angle*last_cos - last_sin*cos_angle);

    static float angle_2 = 3.1416;
    static float pp_angle_2 = 3.1416;
    static float p_angle_2 = 3.1416;
    pp_angle_2 = p_angle_2;
    p_angle_2 = angle_2;
    angle_2 = angle_2 + asin(sin_angle_2*last_cos_2 - last_sin_2*cos_angle_2);

    float anglev;
    if(do_once){
      do_once = 0;
    }
    dt = 0.0333;

    anglev = asin(sin_angle*last_cos - last_sin*cos_angle)/dt;

    ppp_signal = pp_signal;
    pp_signal = p_signal;
    p_signal = MAX_U*signal/10000.0; //Divide by 1000 to make it the same as the optimiser uses

    state(0) = ppp_signal;
    state(1) = pp_pos/LENP2;
    state(2) = pp_signal;
    state(3) = p_pos/LENP2;
    state(4) = p_signal;
    state(5) = cart_pos/LENP2;
    state(6) = pp_sin;
    state(7) = pp_cos;
    state(8) = pp_sin_2;
    state(9) = pp_cos_2;
    state(10) = last_sin;
    state(11) = last_cos;
    state(12) = last_sin_2;
    state(13) = last_cos_2;
    state(14) = sin_angle;
    state(15) = cos_angle;
    state(16) = sin_angle_2;
    state(17) = cos_angle_2;

    pp_cos = last_cos;
    pp_sin = last_sin;
    last_cos = cos_angle;
    last_sin = sin_angle;

    pp_cos_2 = last_cos_2;
    pp_sin_2 = last_sin_2;    
    last_cos_2 = cos_angle_2;
    last_sin_2 = sin_angle_2;
    
    // Acquire readings from sensor
    S626_ReadADC (0, databuf);
    float sens_pos = (short)databuf[0]*(-0.08);
    float sens_vel = (short)databuf[1]*(-0.43);

    // Applies a control signal every second time step
    if(start){
      if (!end_return){
        if(!(syncro_counter%1)){
          if(init_rollout)
            applySimpleRandomControl(5,dt,sens_pos,sens_vel,signal,return_mid);
          else
            applylearnedControllerL(weights, biases, state, signal, return_mid);
            //applylearnedController(centers, weights, W, state, signal, return_mid);
        }
      }
      else
      applyController((orangeposX-width/2), cart_velocity, -sin_angle, -anglev, 0, 0, -0.0005, 0.005, signal);
    }
    
    if (!end_return && start){
      state_data << state(0) << " " << state(1) << " " << pp_angle << " " << pp_angle_2 << " ";
      state_data << state(2) << " " << state(3) << " " << p_angle << " "  << p_angle_2 << " ";
      state_data << state(4) << " " << state(5) << " " << angle << " "  << angle_2 << " ";
      state_data << (float)MAX_U*signal/10000.0 << endl;
    }
    
    last_time = curr_time;

    //cout << "Current time is " << time << "s" << endl;
    
    // We want to draw a line only if its a valid position
    if(orangelastX>0 && orangelastY>0 && orangeposX>0 && orangeposY>0)
    {
      // Draw a yellow line from the previous point to the current point
      cvLine(imgScribble, cvPoint(orangeposX-10, orangeposY+upperc), cvPoint(orangeposX+10, orangeposY+upperc), cvScalar(255,0,255), 2);
      cvLine(imgScribble, cvPoint(orangeposX, orangeposY+upperc-10), cvPoint(orangeposX, orangeposY+upperc+10), cvScalar(255,0,255), 2);

    }

    if(greenlastX>0 && greenlastY>0 && greenposX>0 && greenposY>0)
    {
      // Draw a yellow line from the previous point to the current point
      cvLine(imgScribble, cvPoint(greenposX-10, greenposY+upperc), cvPoint(greenposX+10, greenposY+upperc), cvScalar(255,0,255), 2);
      cvLine(imgScribble, cvPoint(greenposX, greenposY+upperc+10), cvPoint(greenposX, greenposY+upperc-10), cvScalar(255,0,255), 2);
    }

    if(yellowlastX>0 && yellowlastY>0 && yellowposX>0 && yellowposY>0)
    {
      // Draw a yellow line from the previous point to the current point
      cvLine(imgScribble, cvPoint(yellowposX-10, yellowposY+upperc), cvPoint(yellowposX+10, yellowposY+upperc), cvScalar(255,0,255), 2);
      cvLine(imgScribble, cvPoint(yellowposX, yellowposY+upperc+10), cvPoint(yellowposX, yellowposY+upperc-10), cvScalar(255,0,255), 2);
    }

    // Add the scribbling image and the frame... and we get a combination of the two
    cvAdd(video, imgScribble, video);
    

    // Write to videosens_vel = (short)databuf[1]*(-0.43);
    cvWriteFrame(writer, video);

    S626_ReadADC (0, databuf);
    sens_pos = (short)databuf[0]*(-0.08);
    sens_vel = (short)databuf[1]*(-0.43);

    float sens_pos_pred = calc_expected_pos(sens_pos, sens_vel, dt);

    if (curr_time > 3.1 + initiation_time && !end_return || (sens_pos_pred<-SIDE_LIMIT || sens_pos_pred>SIDE_LIMIT))
      {
        S626_WriteDAC (0, 0, 0);
        state_data.close();
        end_return = 1;
      }
    if(start) syncro_counter++;
    if (curr_time > 12.5 + initiation_time) {S626_WriteDAC (0, 0, 0); break;}
  }

  // We're done using the camera. Other applications can now use it
  cvReleaseCapture(&capture);
  cvReleaseVideoWriter(&writer);
  return 0;
}