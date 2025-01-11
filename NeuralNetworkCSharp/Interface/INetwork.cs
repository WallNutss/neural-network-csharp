namespace NeuralNetworkCSharp.Interface;

public interface INetwork
{
    void Train(string trainingFilePath, int epochs, int batchSize, double learningRate);
    void Test();
    List<double> Fit(string filePath);
    List<double> UpdateMiniBatch(List<double> dataInput, List<double> dataLabel, double learningRate);
    List<double> ForwardPass(List<double> input, out List<List<double>> activations, out List<List<double>> zs);
    void Backpropagation(List<List<double>> activations, List<List<double>> zs, List<double> dataPrediction, List<double> dataLabel, double learningRate);
    void ImportWeights(List<double> importedWeights);
    void ImportBiases(List<double> importedBiases);
    List<double> ExportWeights();
    List<double> ExportBiases();
}