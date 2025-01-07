namespace NeuralNetworkCSharp.Interface;

public interface INetwork
{
    void Train(string trainingFilePath, int epochs, int batchSize, double learningRate);
    void Test();
    List<double> FeedForward(List<double> dataInput, List<double> dataLabel, double learningRate);
    void Backpropagation(List<List<double>> activations, List<List<double>> zs, List<double> dataPrediction, List<double> dataLabel, double learningRate);
    void ImportWeights(List<double> importedWeights);
    void ImportBiases(List<double> importedBiases);
    List<double> ExportWeights();
    List<double> ExportBiases();
}