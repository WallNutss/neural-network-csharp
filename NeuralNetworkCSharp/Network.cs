using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetworkCSharp;

public class Network
{
    private int _numLayers;
    private List<int> _sizes;
    private List<List<double>> _biases { get; set; }
    private List<List<List<double>>> _weights { get; set; }
    
    public Network(List<int> sizes)
    {
        _sizes = sizes;
        _numLayers = sizes.Count;
        _biases = GenerateInitialBiases(_sizes);
        _weights = GenerateWeights(_sizes);
    }

    public List<double> ApplySigmoid(List<double> input)
    {
        return input.Select(SigmoidKernelFunction).ToList();
    }
    public List<double> FeedForward(List<double> input)
    {
        if(input.Count != _sizes[0]) 
            throw new Exception($"The number of inputs must match the number of input percepton. Current Input Percepton {_sizes[0]}");

        List<double> a = new List<double>(input);
        
        // Iterate in each layer
        for (int i = 0; i < _biases.Count ; i++)
        {
            List<double> biasCurrentLayer = _biases[i];
            List<List<double>> weightCurrentLayer = _weights[i];
            
            List<double> inputValuesCurrentLayer = new List<double>();
            
            // Calculate the value of the activations layer for this current layer stage
            for (int j = 0; j < weightCurrentLayer.Count; j++)
            {
                var weightVector = Vector<double>.Build.Dense(weightCurrentLayer[j].ToArray());
                var inputVector = Vector<double>.Build.Dense(a.ToArray());

                // Calculate the dot product of the weights and activations
                double value = weightVector.DotProduct(inputVector) + biasCurrentLayer[j];
                
                inputValuesCurrentLayer.Add(SigmoidKernelFunction(value));
            }
            
            a = inputValuesCurrentLayer;
        }

        return a;
    }
    private List<List<double>> GenerateInitialBiases(List<int> sizes)
    {
        List<List<double>> biases = new List<List<double>>();
        for (int i = 1; i < _numLayers; i++)
        {
            List<double> layerBiases = new List<double>();
            var normalDist = new Normal(0, 1);

            for (int j = 0; j < sizes[i]; j++)
            {
                layerBiases.Add(normalDist.Sample());
            }
            biases.Add(layerBiases);
        }
        return biases;
    }
    private List<List<List<double>>> GenerateWeights(List<int> sizes)
    {
        List<List<List<double>>> weights = new ();
        for (int i = 1; i < _numLayers; i++)
        {
            List<List<double>> layerWeights = new List<List<double>>();

            for (int j = 0; j < sizes[i]; j++)
            {
                List<double> neuronWeights = new List<double>();
                var normalDist = new Normal(0, 1);

                for (int k = 0; k < sizes[i - 1]; k++)
                {
                    neuronWeights.Add(normalDist.Sample());
                }
                layerWeights.Add(neuronWeights);
            }
            weights.Add(layerWeights);
        }
        return weights;
    }
    private double SigmoidKernelFunction(double x)
    {
        return 1 / (1 + Math.Exp(-x));
    }
}