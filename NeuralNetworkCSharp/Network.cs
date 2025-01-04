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
            throw new Exception($"The number of inputs must match the number of input perceptron. Current Input perceptron {Sizes[0]}");
        
        List<List<double>> activations = new List<List<double>>();
        List<List<double>> zs = new List<List<double>>();
        
        List<double> activation = new List<double>(input);
        // Iterate in each layer
        for (int i = 0; i < Biases.Count ; i++)
        {
            List<double> biasCurrentLayer = Biases[i];
            List<List<double>> weightCurrentLayer = Weights[i];
            
            List<double> inputCurrentLayer = new List<double>();
            List<double> zCurrentLayer = new List<double>();
            
            // Calculate the value of the activations layer for this current layer stage for each neuron
            for (int j = 0; j < biasCurrentLayer.Count; j++)
            {
                var weightVector = Vector<double>.Build.Dense(weightCurrentLayer[j].ToArray());
                var inputVector = Vector<double>.Build.Dense(activation.ToArray());

                // Calculate the dot product of the weights and activations
                double z = weightVector.DotProduct(inputVector) + biasCurrentLayer[j];
                double activationValue = SigmoidKernelFunction(z);
                
                inputCurrentLayer.Add(activationValue);
                zCurrentLayer.Add(z);
            }
            
            activation = inputCurrentLayer;
            zs.Add(zCurrentLayer);
            activations.Add(activation);
        }

        return activation;
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
    /// Importing Custom Weights based on the Network Layer and their individual neuron inputs
    /// Follow the scheme of index 0 start from the left, with the first input of the current L-Layer the first
    /// Follow the second index 1 of their second weight on that neuron
    /// </summary>
    public void ImportWeights(List<double> weights)
    {
        int totalWeightsNetwork = 0;
        for (int i = 0; i < Sizes.Count - 1; i++)
        {
            totalWeightsNetwork += Sizes[i] * Sizes[i+1];
        }
        if(weights.Count != totalWeightsNetwork)
            throw new InvalidOperationException("The number of imported weights must be the same as the number of network weights");

        int index = 0;
        for (int i = 0; i < Weights.Count; i++)
        {
            for (int j = 0; j < Weights[i].Count; j++)
            {
                for (int k = 0; k < Weights[i][j].Count; k++)
                {
                    Weights[i][j][k] = weights[index];
                    index++;
                }
            }
        }
        
    }
    
    /// <summary>
    /// Importing Custom Bias based on the Network Layer and their individual neuron inputs
    /// </summary>
    public void ImportBias(List<double> biases)
    {
        int totalBiasNetwork = 0;
        for (int i = 1; i < Sizes.Count; i++)
        {
            totalBiasNetwork += Sizes[i];
        }
        if(biases.Count != totalBiasNetwork)
            throw new InvalidOperationException("The number of imported bias must be the same as the number of network weights");

        int index = 0;
        for (int i = 0; i < Weights.Count; i++)
        {
            for (int j = 0; j < Weights[i].Count; j++)
            {
                Biases[i][j] = biases[index];
                index++;
            }
        }
    }
    
    
    /// <summary>
    /// General Sigmoid Kernel Function
    /// </summary>
    private double SigmoidKernelFunction(double x)
    {
        return 1 / (1 + Math.Exp(-x));
    }
    
    /// <summary>
    /// General Sigmoid Kernel Function Derivative
    /// </summary>
    private double DerivativeSigmoidKernelFunction(double x)
    {
        return SigmoidKernelFunction(x) * (1 - SigmoidKernelFunction(x));
    }
    
    // TODO : Backpropagation Algorithm (Learning / Updating the Weight and Biases)
    private void Back(double loss)
    {
        int efectiveLayer = NumLayers - 1;
    }
    
    // TODO : Training Loops Mechanism

    public double CalculateCrossEntropyLoss(List<double> feedForwardPredicted, List<double> trueLabels)
    {
        double loss = 0.0;
        if (feedForwardPredicted.Count != trueLabels.Count)
            throw new InvalidOperationException("The number of predicted labels must match the number of true labels");
    
        for (int i = 0; i < feedForwardPredicted.Count; i++)
        {
            loss += trueLabels[i] * Math.Log(feedForwardPredicted[i]);
        }
        return -loss;
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
            List<(List<string>, List<double[]>)> batches =
                _datasetLoader.ShuffleDataset(dataset.TrainingDatasetImages, dataset.TrainingDatasetLabels, batchSize);

            // Processing the entire dataset but limit it by one batch at a time
            for (int j = 0; j < batches.Count; j++)
            {
                Console.WriteLine($"Epoch #{i}/{epochs} | Batch {j + 1}/{batches.Count}....");
                List<double> errorMiniBatch = new();
                // Processing mini-batch
                for (int k = 0; k < batches[j].Item1.Count; k++)
                {
                    // Item1 is the images
                    // Item2 is the labels
                    // Idk how I can change their name, please help me
                    double[] imageProcessing = _imageProcessing.SingleImageProcessing(batches[j].Item1[k]);
                    List<double> imageInput = imageProcessing.ToList();
                    List<double> prediction = FeedForward(imageInput);
                    
                    // TODO : Get the error between network result and the expected one-hot encoding result
                    double error =
                        CalculateCrossEntropyLoss(prediction, batches[j].Item2[k].ToList());
                    errorMiniBatch.Add(error);
                }
                double averageMiniBatchLoss = errorMiniBatch.Average();
                Console.WriteLine($"Mini Batch average loss is {averageMiniBatchLoss}");
            }
            
            // TODO : Update the weight and Biases
            // StochasticGradientDescent(averageLoss);
            
        }
    }
    
    // TODO : Predict class (Single Prediction)
}
