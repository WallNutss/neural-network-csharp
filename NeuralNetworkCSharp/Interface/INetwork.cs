using NeuralNetworkCSharp.Domain;
using NeuralNetworkCSharp.Enum;

namespace NeuralNetworkCSharp.Interface;

public interface INetwork
{
    void Train(string trainingFilePath, int epochs, int batchSize, double learningRate, BackpropagationMethod method);
    void Test();
    List<double> Fit(string filePath);
    double UpdateMiniBatch(BatchData batchData, double learningRate, BackpropagationMethod method);
    List<double> ForwardPass(List<double> input, out List<List<double>> activations, out List<List<double>> zs);
    GradientParameters Backpropagation(List<List<double>> activations, List<List<double>> zs, List<double> dataPrediction, List<double> dataLabel, double learningRate, BackpropagationMethod method);
    void ImportWeights(List<double> importedWeights);
    void ImportBiases(List<double> importedBiases);
    List<double> ExportWeights();
    List<double> ExportBiases();
    void SaveModel(string modelPath);
    void LoadModel(string modelPath);
}