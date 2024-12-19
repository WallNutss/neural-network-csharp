
using NeuralNetworkCSharp;

class Program
{
    public static void Main(string[] args)
    {
        int inputLayer = 5;
        int hiddenLayer = 15;
        int outputLayer = 4;
        List<int> networkSize = new List<int>(){ inputLayer, hiddenLayer, outputLayer };
        
        Network network = new(networkSize);
        
        List<double> input = new List<double>() {1,2,3,5,6} ;
        List<double> result = network.FeedForward(input);
        foreach (double value in result)
        {
            Console.WriteLine(value);
        }
        
    }
}