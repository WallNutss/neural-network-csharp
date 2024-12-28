using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using NeuralNetworkCSharp.Domain;

namespace NeuralNetworkCSharp;

public class Network
{
    private int NumLayers {get; set;}
    private List<int> Sizes { get; set; }
    private List<List<double>> Biases { get; set; }
    private List<List<List<double>>> Weights { get; set; }
    private DatasetLoader _datasetLoader;
    private ImageProcessing _imageProcessing;
    
    public Network(List<int> sizes)
    {
        Sizes = sizes;
        NumLayers = sizes.Count;
        Biases = GenerateInitialBiases();
        Weights = GenerateWeights();
        _datasetLoader = new DatasetLoader();
        _imageProcessing = new ImageProcessing();
    }
    
    /// <summary>
    /// Apply Sigmoid Kernel Function into list of input(double)
    /// </summary>
    public List<double> ApplySigmoid(List<double> input)
    {
        return input.Select(SigmoidKernelFunction).ToList();
    }
    
    /// <summary>
    /// FeedForward calculation from input layer to output layer in batches
    /// </summary>
    public List<double> FeedForward(List<double> input)
    {
        if(input.Count != Sizes[0]) 
            throw new Exception($"The number of inputs must match the number of input percepton. Current Input Percepton {Sizes[0]}");

        List<double> a = new List<double>(input);
        
        // Iterate in each layer
        for (int i = 0; i < Biases.Count ; i++)
        {
            List<double> biasCurrentLayer = Biases[i];
            List<List<double>> weightCurrentLayer = Weights[i];
            
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
    
    /// <summary>
    /// Generate initial random normal distribution of biases on all layer
    /// </summary>
    private List<List<double>> GenerateInitialBiases()
    {
        List<List<double>> biases = new List<List<double>>();
        for (int i = 1; i < NumLayers; i++)
        {
            List<double> layerBiases = new List<double>();
            var normalDist = new Normal(0, 1);

            for (int j = 0; j < Sizes[i]; j++)
            {
                layerBiases.Add(normalDist.Sample());
            }
            biases.Add(layerBiases);
        }
        return biases;
    }
    
    /// <summary>
    /// Generate initial random normal distribution of weights on all layer
    /// </summary>
    private List<List<List<double>>> GenerateWeights()
    {
        List<List<List<double>>> weights = new ();
        for (int i = 1; i < NumLayers; i++)
        {
            List<List<double>> layerWeights = new List<List<double>>();

            for (int j = 0; j < Sizes[i]; j++)
            {
                List<double> neuronWeights = new List<double>();
                var normalDist = new Normal(0, 1);

                for (int k = 0; k < Sizes[i - 1]; k++)
                {
                    neuronWeights.Add(normalDist.Sample());
                }
                layerWeights.Add(neuronWeights);
            }
            weights.Add(layerWeights);
        }
        return weights;
    }
    
    /// <summary>
    /// General Sigmoid Kernel Function
    /// </summary>
    private double SigmoidKernelFunction(double x)
    {
        return 1 / (1 + Math.Exp(-x));
    }
    
    // TODO : Backpropagation Algorithm (Learning / Updating the Weight and Biases)
    
    // TODO : Training Loops Mechanism

    private List<double> CalculateError(List<double> feedForwardInput, List<double> labelEncodingInput)
    {
        List<double> errors = new List<double>();
        if (feedForwardInput.Count != labelEncodingInput.Count) 
            throw new Exception("The number of label encoding input must match the number of label encoding input");
        
        for (int i = 0; i < feedForwardInput.Count; i++)
        {
            errors.Add(feedForwardInput[i] - labelEncodingInput[i]);
        }
        return errors;
    }
    
    public void Train(string trainingDirectory, int epochs, int batchSize)
    {
        // Load the training dataset and get their label based on one-hot encoding
        Dataset dataset = _datasetLoader.LoadTrainingDataset(trainingDirectory);
        
        // Start training processs
        for (int i = 1; i <= epochs; i++)
        {
            Console.WriteLine($"Epoch #{i}/{epochs}....");
            
            // Get the shuffle version of dataset
            (List<string> trainingShuffleDataset, List<double[]> trainingShuffleLabel) =
                _datasetLoader.ShuffleDataset(dataset.TrainingDatasetImages, dataset.TrainingDatasetLabels, batchSize);
            
            List<List<double>> errorBatchResult = new();
            // Get the feed forward result in each shuffle data
            for (int j = 0; j < trainingShuffleDataset.Count; j++)
            {
                double[] imageBytesForm = _imageProcessing.SingleImageProcessing(trainingShuffleDataset[j]);
                var imagesArrayForm = imageBytesForm.ToList<double>();
                List<double> trainingLocalBatchPerImageResult = FeedForward(imagesArrayForm);
                
                // TODO : Get the error between network result and the expected one-hot encoding result [Partially Done]
                List<double> errorImageResult =
                    CalculateError(trainingLocalBatchPerImageResult, trainingShuffleLabel[j].ToList());

            }
            // TODO : Update the weight and Biases
        }
    }
    
    // TODO : Predict class (Single Prediction)
}
