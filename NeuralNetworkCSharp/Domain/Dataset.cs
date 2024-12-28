namespace NeuralNetworkCSharp.Domain;

public class Dataset
{
    public List<string> TrainingDatasetImages { get; set; }
    public List<double[]> TrainingDatasetLabels { get; set; }
    public List<string> TestingDatasetImages { get; set; }
    public List<int[]> TestingDatasetLabels { get; set; }
}