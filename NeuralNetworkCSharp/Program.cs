
using System.Diagnostics;
using NeuralNetworkCSharp;

class Program
{
    public static void Main(string[] args)
    {
        int inputLayer = 784;
        int hiddenLayer = 15;
        int outputLayer = 10;
        List<int> networkSize = new List<int>(){ inputLayer, hiddenLayer, outputLayer };
        
        Network network = new(networkSize);
        
        ImageProcessing imageProcessing = new ImageProcessing();
        var imagesBytes = imageProcessing.SingleImageProcessing("C:/Jun Local Things/Playground/Neural Network C#/mnist_png/train/1/711.png");
        var imagesArray = imagesBytes.ToList<double>();
        Console.WriteLine($"Images size: {imagesBytes.Length}");
        
        // List<double> input = new List<double>() {0.02, 0.04} ;
        List<double> result = network.FeedForward(imagesArray);
        foreach (double value in result)
        {
            Console.WriteLine(value);
        }
        
        // Load training folder
        // NOTE : DOES using double[] is a type of Unmanaged Memory?
        var watch = Stopwatch.StartNew();
        var trainingDirectory = "C:/Jun Local Things/Playground/Neural Network C#/mnist_png/train";
        (List<string> imagePaths, List<int[]> labelEncoding) = network.LoadDataset(trainingDirectory);
        List<double[]> trainingImages = imageProcessing.BatchImageProcessing(imagePaths);
        watch.Stop();
        
        var elapsedS = watch.ElapsedMilliseconds/1000f;
        Console.WriteLine("Finished loading training data....");
        Console.WriteLine($"Dataset count : {imagePaths.Count}");
        Console.WriteLine($"Loading training data took {elapsedS} seconds.");
        
    }
}