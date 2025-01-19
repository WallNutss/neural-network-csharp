using NeuralNetworkCSharp.Core;

namespace NeuralNetworkCSharp;

static class Program
{
    public static void Main(string[] args)
    {
        const int inputLayer = 784;
        const int hiddenLayer = 20;
        const int outputLayer = 10;
        List<int> networkSize = new List<int>(){ inputLayer, hiddenLayer, outputLayer };
        
        Network network = new(networkSize);
        
        // Load training folder
        var trainingDirectory = "C:/Jun Local Things/Playground/Neural Network C#/mnist_png/train";
        
        // Start training process
        // Batch size selection? Perhaps see this reference -> https://arxiv.org/abs/1206.5533 
        network.Train(trainingDirectory, 10, 32, 0.1);
        
        // Save the model result
        network.SaveModel("./model.wes");
        
        // Test the network model
        string testImage = "C:/Jun Local Things/Playground/Neural Network C#/mnist_png/valid/2/35.png";
        List<double> prediction = network.Fit(testImage);
        foreach (var p in prediction)
        {
            Console.WriteLine(p);
        }
    }
}