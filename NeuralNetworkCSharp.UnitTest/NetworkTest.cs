namespace NeuralNetworkCSharp.UnitTest;

public class NetworkTest
{
    private readonly Network _network;
    const int inputLayer = 3;
    const int secondLayer = 2;
    const int outputLayer = 1; 
    List<int> networkSize = new List<int>(){ inputLayer, secondLayer, outputLayer }; 

    public NetworkTest()
    {
        _network = new Network(networkSize);
    }
    [Fact]
    public void FeedForwardCalculation_ShouldReturnCorrectValue_WithWeightsAndBiasSetToOne()
    {
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
        
        // Create the network
        var inputNetwork = new List<double>(){1,1,1};
        
        // Import the weight and bias
        _network.ImportWeights(weights);
        _network.ImportBias(biases);
        
        var expected = 0.9509223;
        var tolerance= 0.00001;
        
        var output = _network.FeedForward(inputNetwork);
        Assert.InRange(output.First(), expected - tolerance, expected + tolerance);
    }
}