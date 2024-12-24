
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
        
        ImageProcessing loader = new ImageProcessing();
        var imagesBytes = loader.ImagetoByteArray("C:/Jun Local Things/Playground/Neural Network C#/mnist_png/train/1/711.png");
        var imagesArray = imagesBytes.ToList<double>();
        Console.WriteLine($"Images size: {imagesBytes.Length}");
        
        // List<double> input = new List<double>() {0.02, 0.04} ;
        List<double> result = network.FeedForward(imagesArray);
        foreach (double value in result)
        {
            Console.WriteLine(value);
        }
        
    }
}