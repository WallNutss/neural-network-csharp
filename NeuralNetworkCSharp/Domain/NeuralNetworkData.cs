namespace NeuralNetworkCSharp.Domain;

public class NeuralNetworkData
{
    public List<List<double>> Biases { get; set; } = new List<List<double>>();
    public List<List<List<double>>> Weights { get; set; } = new List<List<List<double>>>();
}