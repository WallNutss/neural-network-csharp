using NeuralNetworkCSharp.Core;

namespace NeuralNetworkCSharp;

static class Program
{
    public static void Main(string[] args)
    {
        const int inputLayer = 784;
        const int hiddenLayer = 15;
        const int outputLayer = 10;
        List<int> networkSize = new List<int>(){ inputLayer, hiddenLayer, outputLayer };
        
        Network network = new(networkSize);
        
        // Load training folder
        var trainingDirectory = "C:/Jun Local Things/Playground/Neural Network C#/mnist_png/train";
        
        // Start training process
        network.Train(trainingDirectory, 10, 1000, 0.01);
        
        // Save the model result
        network.SaveModel("./model.wes");
        
        // Test the network model

    }
}