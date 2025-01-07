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
        network.ImportBias(biases);
        
        List<double> expected = new() { 0.9509223 };
        var tolerance= 0.00001;
        
        var output = network.FeedForward(inputNetwork, expected);
        Assert.All(output, (o, index) =>
        {
            var expectedResult = expected[index];
            Assert.InRange(o, expectedResult - tolerance, expectedResult + tolerance);
        });
    }
    
    [Fact]
    public void FeedForwardCalculation_ShouldReturnCorrectValue_WithWeightsAndBiasSetToOneAndNetwork2x3x2()
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
        network.ImportBias(biases);
        
        List<double> expected = new() { 0.75136507, 0.772928465 };
        List<double> outputTrueLabels = new() { 0.01, 0.99 };
        var tolerance= 0.00001;
        
        var output = network.FeedForward(inputNetwork, outputTrueLabels);
        Assert.All(output, (o, index) =>
        {
            var expectedResult = expected[index];
            Assert.InRange(o, expectedResult - tolerance, expectedResult + tolerance);
        });

    }
}