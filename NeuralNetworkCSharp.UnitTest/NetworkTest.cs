using NeuralNetworkCSharp.Core;

namespace NeuralNetworkCSharp.UnitTest;

public class NetworkTest
{
    [Fact]
    public void FeedForwardCalculation_ShouldReturnCorrectValue_WithWeightsAndBiasSetToOneAndNetwork3x2x1()
    {
        // Initialize the network
        const int inputLayer = 3;
        const int secondLayer = 2;
        const int outputLayer = 1; 
        List<int> networkSize = new List<int>(){ inputLayer, secondLayer, outputLayer }; 
        
        Network network = new Network(networkSize);
        
        // Initialize the weight and bias
        int totalWeightsNetwork = 0;
        for (int i = 0; i < networkSize.Count - 1; i++)
        {
            totalWeightsNetwork += networkSize[i] * networkSize[i+1];
        }
        int totalBiasNetwork = 0;
        for (int i = 1; i < networkSize.Count; i++)
        {
            totalBiasNetwork += networkSize[i];
        }
        List<double> weights = Enumerable.Repeat(1.0, totalWeightsNetwork).ToList();
        List<double> biases = Enumerable.Repeat(1.0, totalBiasNetwork).ToList();
        
        // Input network
        var inputNetwork = new List<double>(){1,1,1};
        
        // Import the weight and bias
        network.ImportWeights(weights);
        network.ImportBiases(biases);
        
        List<double> expected = new() { 0.9509223 };
        var tolerance= 0.00001;
        
        var output = network.UpdateMiniBatch(inputNetwork, expected);
        Assert.All(output, (o, index) =>
        {
            var expectedResult = expected[index];
            Assert.InRange(o, expectedResult - tolerance, expectedResult + tolerance);
        });
    }
    
    [Fact]
    public void Backpropagation_WeightsShouldReturnCorrectValue_WithNetwork2x2x2()
    {
        // Number from this reference -> https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
        // Initialize the network
        const int inputLayer = 2;
        const int secondLayer = 2;
        const int outputLayer = 2; 
        List<int> networkSize = new List<int>(){ inputLayer, secondLayer, outputLayer }; 
        
        Network network = new Network(networkSize);
        
        // Initialize the weight and bias
        List<double> weights = new List<double>() { 0.15, 0.20, 0.25, 0.30, 0.40, 0.45, 0.50, 0.55 };
        List<double> biases = new List<double>() { 0.35, 0.35, 0.60, 0.60 };
        
        // Input network
        var inputNetwork = new List<double>(){ 0.05, 0.10 };
        
        // Import the weight and bias
        network.ImportWeights(weights);
        network.ImportBiases(biases);
        
        List<double> expected = new() { 0.75136507, 0.772928465 };
        List<double> outputTrueLabels = new() { 0.01, 0.99 };
        var tolerance= 0.000001;
        List<double> expectedWeights = new()
        {
            0.149780716, 0.19956143, 0.24975114, 0.29950229,
            0.35891648, 0.408666186, 0.511301270, 0.561370121
        };
        
        var outputFeedForward = network.UpdateMiniBatch(inputNetwork, outputTrueLabels, 0.5);
        
        // Check if the feedforward result is correct
        Assert.All(outputFeedForward, (o, index) =>
        {
            var expectedResult = expected[index];
            Assert.InRange(o, expectedResult - tolerance, expectedResult + tolerance);
        });
        
        // Get the weights of the network and check the result with tolerance
        List<double> actualWeights = network.ExportWeights();
        Assert.All(actualWeights, (w, index) =>
        {
            var expectedWeight = expectedWeights[index];
            Assert.InRange(w, expectedWeight - tolerance, expectedWeight + tolerance);
        });

    }
}