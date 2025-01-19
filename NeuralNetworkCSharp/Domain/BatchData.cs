namespace NeuralNetworkCSharp.Domain;

public class BatchData
{
    public List<string> Images { get; set; }
    public List<double[]> Labels { get; set; }
}