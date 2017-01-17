package sample;


import java.io.ByteArrayInputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;


public class Controller {
    @FXML
    private Button button;
    @FXML
    private Button brighten;
    @FXML
    private CheckBox grayscale;
    @FXML
    private ImageView histogram;
    @FXML
    private ImageView currentFrame;

    // Brightness change that is required for a register
    private static long brightnessDelta = 10;
    // a timer for acquiring the video stream
    private ScheduledExecutorService timer;
    // the OpenCV object that realizes the video capture
    private VideoCapture capture;
    // a flag to change the button behavior
    private boolean cameraActive;
    //Old Frame Store
    private Mat oldFrame;
    //Initial Time
    private long initialTime;

    /**
     * Initialize method, automatically called by @{link FXMLLoader}
     */
    public void initialize()
    {
        this.capture = new VideoCapture();
        this.cameraActive = false;
    }

    @FXML
    protected void bright()
    {
        try {

            Process process1 = Runtime.getRuntime().exec("brightness 0");
            Thread.sleep(5000);
            this.initialTime = System.currentTimeMillis();
            process1 = Runtime.getRuntime().exec("brightness 1");
        }
        catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    /**
     * The action triggered by pushing the button on the GUI
     */
    @FXML
    protected void startCamera()
    {
        this.currentFrame.setFitWidth(600);
        this.currentFrame.setPreserveRatio(true);

        if (!this.cameraActive)
        {
            this.capture.open(0);

            if (this.capture.isOpened())
            {
                this.cameraActive = true;
                Runnable frameGrabber = new Runnable() {
                    @Override
                    public void run()
                    {
                        Image imageToShow = grabFrame();
                        currentFrame.setImage(imageToShow);
                    }
                };

                this.timer = Executors.newSingleThreadScheduledExecutor();

                this.timer.scheduleAtFixedRate(frameGrabber, 0, 33, TimeUnit.MILLISECONDS);

                this.button.setText("Stop Camera");
            }
            else
            {
                System.err.println("Impossible to open the camera connection...");
            }
        }
        else
        {
            this.cameraActive = false;
            this.button.setText("Start Camera");
            try
            {
                this.timer.shutdown();
                this.timer.awaitTermination(33, TimeUnit.MILLISECONDS);
            }
            catch (InterruptedException e)
            {
                System.err.println("Exception in stopping the frame capture, trying to release the camera now... " + e);
            }
            this.capture.release();
            this.currentFrame.setImage(null);
        }
    }

    /**
     * Get a frame from the opened video stream (if any)
     *
     * @return the {@link Image} to show
     */
    private Image grabFrame()
    {
        long curTime = this.initialTime;
        Image imageToShow = null;
        Mat frame = new Mat();

        if (this.capture.isOpened())
        {
            try
            {
                this.capture.read(frame);

                if (!frame.empty())
                {
                    if (grayscale.isSelected())
                    {
                        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_BGR2GRAY);
                    }

                    this.showHistogram(frame, grayscale.isSelected());
                    imageToShow = mat2Image(frame);
                    long outputTime = detectSigBrightnessChange(this.oldFrame,frame);
                    if (outputTime>0)
                    {
                        System.out.println(outputTime-curTime);
                    }
                    this.oldFrame = frame;
                }

            }
            catch (Exception e)
            {
                // log the error
                System.err.println("Exception during the frame elaboration: " + e);
            }
        }

        return imageToShow;
    }

    private long detectSigBrightnessChange(Mat oldFrame, Mat newFrame)
    {
        Mat result = new Mat();
        if (oldFrame==null)
        {
            return -1;
        }
        Core.subtract(newFrame,oldFrame,result);
        if (Math.abs(Core.mean(result).val[0])>this.brightnessDelta)
        {
            return System.currentTimeMillis();
        }
        return -1;
    }

    /**
     * Compute and show the histogram for the given {@link Mat} image
     *
     * @param frame
     *            the {@link Mat} image for which compute the histogram
     * @param gray
     *            is a grayscale image?
     */
    private void showHistogram(Mat frame, boolean gray)
    {
        // split the frames in multiple images
        List<Mat> images = new ArrayList<Mat>();
        Core.split(frame, images);

        // set the number of bins at 256
        MatOfInt histSize = new MatOfInt(256);
        // only one channel
        MatOfInt channels = new MatOfInt(0);
        // set the ranges
        MatOfFloat histRange = new MatOfFloat(0, 256);

        // compute the histograms for the B, G and R components
        Mat hist_b = new Mat();
        Mat hist_g = new Mat();
        Mat hist_r = new Mat();

        // B component or gray image
        Imgproc.calcHist(images.subList(0, 1), channels, new Mat(), hist_b, histSize, histRange, false);

        // G and R components (if the image is not in gray scale)
        if (!gray)
        {
            Imgproc.calcHist(images.subList(1, 2), channels, new Mat(), hist_g, histSize, histRange, false);
            Imgproc.calcHist(images.subList(2, 3), channels, new Mat(), hist_r, histSize, histRange, false);
        }

        // draw the histogram
        int hist_w = 150; // width of the histogram image
        int hist_h = 150; // height of the histogram image
        int bin_w = (int) Math.round(hist_w / histSize.get(0, 0)[0]);

        Mat histImage = new Mat(hist_h, hist_w, CvType.CV_8UC3, new Scalar(0, 0, 0));
        // normalize the result to [0, histImage.rows()]
        Core.normalize(hist_b, hist_b, 0, histImage.rows(), Core.NORM_MINMAX, -1, new Mat());

        // for G and R components
        if (!gray)
        {
            Core.normalize(hist_g, hist_g, 0, histImage.rows(), Core.NORM_MINMAX, -1, new Mat());
            Core.normalize(hist_r, hist_r, 0, histImage.rows(), Core.NORM_MINMAX, -1, new Mat());
        }

        // effectively draw the histogram(s)
        for (int i = 1; i < histSize.get(0, 0)[0]; i++)
        {
            // B component or gray image
            Imgproc.line(histImage, new Point(bin_w * (i - 1), hist_h - Math.round(hist_b.get(i - 1, 0)[0])),
                    new Point(bin_w * (i), hist_h - Math.round(hist_b.get(i, 0)[0])), new Scalar(255, 0, 0), 2, 8, 0);
            // G and R components (if the image is not in gray scale)
            if (!gray)
            {
                Imgproc.line(histImage, new Point(bin_w * (i - 1), hist_h - Math.round(hist_g.get(i - 1, 0)[0])),
                        new Point(bin_w * (i), hist_h - Math.round(hist_g.get(i, 0)[0])), new Scalar(0, 255, 0), 2, 8,
                        0);
                Imgproc.line(histImage, new Point(bin_w * (i - 1), hist_h - Math.round(hist_r.get(i - 1, 0)[0])),
                        new Point(bin_w * (i), hist_h - Math.round(hist_r.get(i, 0)[0])), new Scalar(0, 0, 255), 2, 8,
                        0);
            }
        }

        // display the histogram...
        Image histImg = mat2Image(histImage);
        this.histogram.setImage(histImg);

    }

    /**
     * Convert a Mat object (OpenCV) in the corresponding Image for JavaFX
     *
     * @param frame
     *            the {@link Mat} representing the current frame
     * @return the {@link Image} to show
     */
    private Image mat2Image(Mat frame)
    {
        // create a temporary buffer
        MatOfByte buffer = new MatOfByte();
        // encode the frame in the buffer, according to the PNG format
        Imgcodecs.imencode(".png", frame, buffer);
        // build and return an Image created from the image encoded in the
        // buffer
        return new Image(new ByteArrayInputStream(buffer.toArray()));
    }

}
