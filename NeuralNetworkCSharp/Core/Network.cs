using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using NeuralNetworkCSharp.Domain;
using NeuralNetworkCSharp.Interface;

namespace NeuralNetworkCSharp.Core;

public class Network : INetwork
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
        Weights = GenerateInitialWeights();
        _datasetLoader = new DatasetLoader();
        _imageProcessing = new ImageProcessing();
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
    private List<List<List<double>>> GenerateInitialWeights()
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
    public void ImportBiases(List<double> biases)
    {
        int totalBiasNetwork = 0;
        for (int i = 1; i < Sizes.Count; i++)
        {
            totalBiasNetwork += Sizes[i];
        }
        if(biases.Count != totalBiasNetwork)
            throw new InvalidOperationException("The number of imported bias must be the same as the number of network weights");

        int index = 0;
        for (int i = 0; i < Biases.Count; i++)
        {
            for (int j = 0; j < Biases[i].Count; j++)
            {
                Biases[i][j] = biases[index];
                index++;
            }
        }
    }
    
    // TODO : Export weight and biases
    /// <summary>
    /// Exporting Weights
    /// </summary>
    public List<double> ExportWeights()
    {
        List<double> exportWeights = new();
        for (int i = 0; i < Weights.Count; i++)
        {
            for (int j = 0; j < Weights[i].Count; j++)
            {
                for (int k = 0; k < Weights[i][j].Count; k++)
                {
                    exportWeights.Add(Weights[i][j][k]);
                }
            }
        }
        return exportWeights;
    }
    
    /// <summary>
    /// Exporting Biases
    /// </summary>
    public List<double> ExportBiases()
    {
        List<double> exportBiases = new();
        for (int i = 0; i < Biases.Count; i++)
        {
            for (int j = 0; j < Biases[i].Count; j++)
            {
                exportBiases.Add(Biases[i][j]);
            }
        }
        return exportBiases;
    }
    
    // TODO : Alogrithm to Update Weight and Bias from Nabla from the BackPropagation result
    /// <summary>
    /// Update the weights
    /// </summary>
    private void UpdateWeights(List<List<List<double>>> nablaWeights, double learningRate)
    {
        if (Weights[0].Count != nablaWeights[0].Count)
            throw new InvalidOperationException("The dimension of weights must be the same as the dimension of the nabla weights network");
        
        for (int i = 0; i < Weights.Count; i++)
        {
            for (int j = 0; j < Weights[i].Count; j++)
            {
                if (Weights[i][j].Count != nablaWeights[i][j].Count)
                    throw new InvalidOperationException($"The dimension of the intermitten layer of W-Layer[{i}][{j}] does not match the number of network weights");
                
                for (int k = 0; k < Weights[i][j].Count; k++)
                {
                    Weights[i][j][k] += - learningRate * nablaWeights[i][j][k];
                }
            }
        }
    }
    
    /// <summary>
    /// Update the biases
    /// </summary>
    private void UpdateBiases(List<List<double>> nablaBiases, double learningRate)
    {
        if (Biases[0].Count != nablaBiases[0].Count)
            throw new InvalidOperationException("The dimension of biases must be the same as the dimension of the nabla biases network");
        
        for (int i = 0; i < Biases.Count; i++)
        {
            for (int j = 0; j < Biases[i].Count; j++)
            {
                if (Biases[i].Count != nablaBiases[i].Count)
                    throw new InvalidOperationException($"The dimension of the intermitten layer of B-Layer[{i}][{j}] does not match the number of network biases");
                
                Biases[i][j] += -learningRate * nablaBiases[i][j];
            }
        }
    }
    /// <summary>
    /// Apply Sigmoid Kernel Function into list of input(double)
    /// </summary>
    public List<double> ApplySigmoid(List<double> input)
    {
        return input.Select(SigmoidKernelFunction).ToList();
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

    /// <summary>
    /// Calculate the losses (labels - prediction) using Cross-Entropy Loss
    /// </summary>
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

    /// <summary>
    /// Calculate the losses (labels - prediction) using Mean Square Error
    /// </summary>
    public double CalculateMeanSquareErrorLoss(List<double> feedForwardPredicted, List<double> trueLabels)
    {
        double loss = 0.0;
        if (feedForwardPredicted.Count != trueLabels.Count)
            throw new InvalidOperationException("The number of predicted labels must match the number of true labels");
    
        for (int i = 0; i < feedForwardPredicted.Count; i++)
        {
            loss += Math.Pow(trueLabels[i] - feedForwardPredicted[i], 2);
        }
        return loss/feedForwardPredicted.Count;
    }
    
    /// <summary>
    /// FeedForward calculation from input layer to output layer in batches
    /// </summary>
    public List<double> FeedForward(List<double> input, List<double> outputLabels, double learningRate=0.01)
    {
        if(input.Count != Sizes[0]) 
            throw new Exception($"The number of inputs must match the number of input perceptron. Current Input perceptron {Sizes[0]}");
        
        List<List<double>> activations = new List<List<double>>();
        List<List<double>> zs = new List<List<double>>();
        
        List<double> activation = new List<double>(input);
        activations.Add(activation);
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
            activations.Add(activation);
            zs.Add(zCurrentLayer);
        }
        
        // TODO : Do Learning Algorithm here
        Backpropagation(activations, zs, activation, outputLabels, learningRate);

        return activation;
    }
    
    // TODO : Backpropagation Algorithm (Learning / Updating the Weight and Biases)
    /// <summary>
    /// Backpropagation calculation from output layer back to input layer
    /// </summary>
    public void Backpropagation(List<List<double>> activations, List<List<double>> zs, List<double> prediction, List<double> target, double learningRate)
    {
        int intermittenLayer = NumLayers - 2;
        List<List<List<double>>> nablaWeights = new();
        List<List<double>> nablaBiases = new();
        
        List<List<double>> dLdzs = new();

        // Start the weight and bias update using gradient descent
        for (int i = intermittenLayer; i >= 0; i--)
        {
            List<List<double>> nablaWeightsEachIntermittenLayer = new();
            List<double> nablaBiasesEachIntermittenLayer = new();
            List<double> dldzEachIntermittenLayer = new();
            
            for (int j = 0; j < Weights[i].Count; j++)
            {
                // Updating the first intermitten layer, its special because its the first chain to update the weights
                if (i == intermittenLayer)
                {
                    // Update the Weights
                    var dlda = prediction[j] - target[j];
                    var dadz = DerivativeSigmoidKernelFunction(zs[i][j]);
                    var dzdw = Vector<double>.Build.Dense(activations[i].ToArray());

                    var dldz = dlda * dadz;
                    var dldw = dldz * dzdw;
                    
                    // Update the Biases
                    var dzdb = 1.0; // Because derivative of zL = wL.AL-1 + bL respective to b is 1 so....
                    var dldb = dldz * dzdb;
                    
                    dldzEachIntermittenLayer.Add(dldz);
                    nablaWeightsEachIntermittenLayer.Add(dldw.ToList());
                    nablaBiasesEachIntermittenLayer.Add(dldb);
                }
                // The rest of the intermitten layer, if there is four layer {input, hidden, hidden, output}, then this
                // will be the input-hidden, hidden-hidden
                else
                {
                    // Update Weights
                    // Iterate in each loop of previous dldz layer intermitten and get total of that to get the dL/dA[each neuron]
                    List<double> dzda1 = new();
                    foreach (var each in Weights[i + 1])
                    {
                        dzda1.Add(each[j]);
                    }
                    var dldzPreviousLayer = Vector<double>.Build.Dense(dLdzs[0].ToArray());
                    var dzda1Vector = Vector<double>.Build.Dense(dzda1.ToArray());
                    
                    // Get the current dLdz by dL/dA[each neuron] * dA[Each Neuron]/dz[Each Neuron]
                    var dadz = DerivativeSigmoidKernelFunction(zs[i][j]);
                    var dldz = dldzPreviousLayer.DotProduct(dzda1Vector) * dadz;
                    
                    // Get the nabla of dLdw by simply now dLdz * dzdw
                    var dzdw = Vector<double>.Build.Dense(activations[i].ToArray());
                    var dldw = dldz * dzdw;
                    
                    // Update Biases
                    var dzdb = 1.0; // Because derivative of zL = wL.AL-1 + bL respective to b is 1 so....
                    var dldb = dldz * dzdb;
                    
                    dldzEachIntermittenLayer.Add(dldz);
                    nablaWeightsEachIntermittenLayer.Add(dldw.ToList());
                    nablaBiasesEachIntermittenLayer.Add(dldb);
                }
            }
            nablaWeights.Insert(0, nablaWeightsEachIntermittenLayer);
            nablaBiases.Insert(0, nablaBiasesEachIntermittenLayer);
            dLdzs.Insert(0, dldzEachIntermittenLayer);
        }
        
        // Update the weight and biases
        UpdateWeights(nablaWeights, learningRate);
        UpdateBiases(nablaBiases, learningRate);
    }
    
    // TODO : Training Loops Mechanism
    /// <summary>
    /// Training Pipeline
    /// </summary>
    public void Train(string trainingDirectory, int epochs, int batchSize, double learningRate)
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
                    List<double> imageLabels = batches[j].Item2[k].ToList();
                    List<double> prediction = FeedForward(imageInput, imageLabels, learningRate);
                    
                    // TODO : Get the error between network result and the expected one-hot encoding result
                    double error =
                        CalculateMeanSquareErrorLoss(prediction, imageLabels);
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

    // TODO : Add function to test the result of the network
    /// <summary>
    /// Testing Pipeline
    /// </summary>
    public void Test()
    {
        
    }
}
