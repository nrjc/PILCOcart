package sample;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.layout.BorderPane;
import javafx.stage.Stage;
import org.opencv.core.Core;

public class Main extends Application {
    @Override
    public void start(Stage primaryStage)
    {
        try
        {
            // load the FXML resource
            FXMLLoader loader = new FXMLLoader(getClass().getResource("sample.fxml"));
            // store the root element so that the controllers can use it
            BorderPane rootElement = (BorderPane) loader.load();
            // create and style a scene
            Scene scene = new Scene(rootElement, 800, 600);
            scene.getStylesheets().add(getClass().getResource("application.css").toExternalForm());
            // create the stage with the given title and the previously created
            // scene
            primaryStage.setTitle("Video processing");
            primaryStage.setScene(scene);
            // show the GUI
            primaryStage.show();

        }
        catch (Exception e)
        {
            e.printStackTrace();
        }
    }

    public static void main(String[] args)
    {
        // load the native OpenCV library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        launch(args);
    }
}
