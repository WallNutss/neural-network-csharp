namespace NeuralNetworkCSharp.Domain;

public class GradientParameters
{
    public List<List<List<double>>> NablaWeights { get; set; }
    public List<List<double>> NablaBiases { get; set; }
}