using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using NeuralNetworkCSharp.Domain;
using NeuralNetworkCSharp.Helper;
using NeuralNetworkCSharp.Interface;
using Newtonsoft.Json;

namespace NeuralNetworkCSharp.Core;

public class Network : INetwork
{
    private int NumLayers {get; set;}
    private List<int> Sizes { get; set; }
    private List<List<double>> Biases { get; set; }
    private List<List<List<double>>> Weights { get; set; }
    private readonly DatasetLoader _datasetLoader;
    private readonly ImageProcessing _imageProcessing;
    
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
    /// <returns>List of biases of each neuron in their each layer</returns>
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
    /// <returns>List of weights of each neuron in their each layer</returns>
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
    /// <param name="weights"></param>
    /// <returns>Loading weights into network</returns>
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
    /// <param name="biases"></param>
    /// <returns>Loading biases into network</returns>
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
    
    /// <summary>
    /// Exporting Weights
    /// </summary>
    /// <returns>Export the weights of the network into a list of doubles</returns>
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
    /// <returns>Export the biases of the network into a list of doubles</returns>
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
    
    /// <summary>
    /// Update the weights
    /// </summary>
    /// <param name="nablaWeights">Gradient of weights over the loss function</param>
    /// <param name="learningRate"></param>
    /// <returns>Update the weights of the network</returns>
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
    /// <param name="nablaBiases">Gradient of bias over the loss function</param>
    /// <param name="learningRate"></param>
    /// <returns>Update the biases of the network</returns>
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
    /// Saves the network's weights and biases to a .wes file.
    /// .wes mean in Indonesia in intention is "ah sudahlah", but in here means "weight and biases"
    /// </summary>
    /// <param name="modelPath">The model path you want to save with its actual name with .wes file on the end</param>
    /// <returns>Save the weight and model into .wes file on the specific model path</returns>
    public void SaveModel(string modelPath)
    {
        // Check if the file extension is .wes
        if (Path.GetExtension(modelPath).ToLower() != ".wes")
        {
            throw new ArgumentException("The file must have a .wes extension.");
        }
        
        var data = new NeuralNetworkData
        {
            Weights = Weights,
            Biases = Biases
        };

        // Serialize the data to JSON
        string jsonData = JsonConvert.SerializeObject(data, Formatting.Indented);

        // Save to file
        File.WriteAllText(modelPath, jsonData);
    }

    /// <summary>
    /// Load model from .wes file
    /// .wes mean in Indonesia in intention is "ah sudahlah", but in here means "weight and biases"
    /// </summary>
    /// <param name="modelPath">The model path with .wes file on the end</param>
    /// <returns>Load the weight and biases model from .wes file</returns>
    public void LoadModel(string modelPath)
    {
        // Check if the file extension is .wes
        if (Path.GetExtension(modelPath).ToLower() != ".wes")
        {
            throw new ArgumentException("The file must have a .wes extension.");
        }
        
        // Read the file contents
        string jsonData = File.ReadAllText(modelPath);

        // Deserialize the data from JSON
        var neuralNetworkData = JsonConvert.DeserializeObject<NeuralNetworkData>(jsonData);
        
        if ( neuralNetworkData == null )
            throw new Exception("Neural Network Data does not exist. Please check the file.");
        
        Biases = neuralNetworkData.Biases;
        Weights = neuralNetworkData.Weights;
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
    /// <returns>Result of sigmoid function</returns>
    private double SigmoidKernelFunction(double x)
    {
        return 1 / (1 + Math.Exp(-x));
    }
    
    /// <summary>
    /// General Sigmoid Kernel Function Derivative
    /// </summary>
    /// <param name="x">a double value to get the result of the derivative</param>
    /// <returns>Result of derivative sigmoid function</returns>
    private double DerivativeSigmoidKernelFunction(double x)
    {
        return SigmoidKernelFunction(x) * (1 - SigmoidKernelFunction(x));
    }
    
    /// <summary>
    /// General Sigmoid Kernel Function Derivative for array inpputs
    /// </summary>
    /// <param name="inputs">a double value to get the result of the derivative</param>
    /// <returns>Result of derivative sigmoid function</returns>
    private List<double> DerivativeSigmoidKernelFunction(List<double> inputs)
    {
        return inputs.Select(DerivativeSigmoidKernelFunction).ToList();
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
    /// Network calculation learning frocm input layer to output layer in batches (with learning and backpropagation)
    /// </summary>
    /// <param name="input">the input data</param>
    /// <param name="outputLabels">the labels value of the data</param>
    /// <param name="learningRate">learning rate - the value of how much gradient descent will step</param>
    /// <returns>Result prediction of the output layer</returns>
    public List<double> UpdateMiniBatch(List<double> input, List<double> outputLabels, double learningRate=0.01)
    {
        if(input.Count != Sizes[0]) 
            throw new Exception($"The number of inputs must match the number of input perceptron. Current Input perceptron {Sizes[0]}");
        
        // Run forward pass and get activations
        List<List<double>> activations;
        List<List<double>> zs;
        List<double> outputActivation = ForwardPass(input, out activations, out zs);
        
        // Do backpropagation and update the weights and biases of the network
        Backpropagation(activations, zs, outputActivation, outputLabels, learningRate);

        return outputActivation;
    }
    
    /// <summary>
    /// Forward pass through the network layers (common logic for both Inference and UpdateMiniBatch).
    /// </summary>
    /// <param name="input">Input the go through the network, must match with the amount of input neuron layer</param>
    /// <param name="activations">out an activation of each layer</param>
    /// <param name="zs">out calculation of neuron in each layer</param>
    /// <returns>Result prediction of the output layer</returns>
    public List<double> ForwardPass(List<double> input, out List<List<double>> activations, out List<List<double>> zs)
    {
        if(input.Count != Sizes[0]) 
            throw new Exception($"The number of inputs must match the number of input perceptron. Current Input perceptron {Sizes[0]}");
        
        activations = new List<List<double>>();
        zs = new List<List<double>>();
    
        List<double> activation = new List<double>(input);
        activations.Add(activation);
    
        // Iterate in each layer
        for (int i = 0; i < Biases.Count; i++)
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
    
        return activation;
    }
    
    /// <summary>
    /// Backpropagation calculation from output layer back to input layer
    /// </summary>
    /// <param name="activations">activations value of each layer</param>
    /// <param name="zs">neuron calculation of each layer</param>
    /// <param name="prediction">prediction value from forward pass</param>
    /// <param name="target">label value that will be contested with the prediction value of forward pass</param>
    /// <param name="learningRate">learning rate - the value of how much gradient descent will step</param>
    /// <returns>nabla weights and biases - the change of rate of each one of weights and biases on the network</returns>
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
            
            // // TODO : Instead iterating like this, how about using Hadamard Product to get the result of the nabla's?
            if (i == intermittenLayer)
            {
                // Update the weights
                var predictionVector = Vector<double>.Build.Dense(prediction.ToArray());
                var targetVector = Vector<double>.Build.Dense(target.ToArray());
                
                var dLdA = predictionVector - targetVector;
                var dAdz = Vector<double>.Build.Dense(DerivativeSigmoidKernelFunction(zs[i]).ToArray());
                var dzdw = Vector<double>.Build.Dense(activations[i].ToArray());
                
                // Get the dLdZ of this intermitten layer
                var dLdz = dLdA.PointwiseMultiply(dAdz);
                
                var dLdw = dLdz.ToColumnMatrix() * dzdw.ToRowMatrix();
                
                // Update the biases
                var dzdb = 1.0; // Because derivative of zL = wL.AL-1 + bL respective to b is 1 so....
                var dLdb = dLdz * dzdb;
                
                nablaWeights.Insert(0, MatrixConverter.ConvertMatrixToListOfLists(dLdw));
                nablaBiases.Insert(0, dLdb.ToList());
                dLdzs.Insert(0, dLdz.ToList());
            }
            // The rest of the intermitten layer, if there is four layer {input, hidden, hidden, output}, then this
            // will be the input-hidden, hidden-hidden
            else
            {
                // Update Weights
                var dLdzPreviousLayer = Vector<double>.Build.Dense(dLdzs[0].ToArray());
                var dzdA = MatrixConverter.ConvertListOfListsToMatrix(Weights[i + 1]).Transpose();
                var dAdz = Vector<double>.Build.Dense(DerivativeSigmoidKernelFunction(zs[i]).ToArray());
                var dzdw = Vector<double>.Build.Dense(activations[i].ToArray());
                
                // Get the dLdZ of this intermitten layer
                var dLdz = Vector<double>.Build.Dense(dzdA.RowCount);

                // Perform the element-wise multiplication for each row and sum
                for (int k = 0; k < dzdA.RowCount; k++)
                    dLdz[i] = dLdzPreviousLayer.PointwiseMultiply(dzdA.Row(k)).Sum();  // Element-wise multiplication + sum for the row
                
                // Get the nabla of dLdw by simply now dLdz * dzdw
                var dLdw = dLdz.ToColumnMatrix() * dzdw.ToRowMatrix();
                
                // Update Biases
                var dzdb = 1.0; // Because derivative of zL = wL.AL-1 + bL respective to b is 1 so....
                var dLdb = dLdz * dzdb;
                
                nablaWeights.Insert(0, MatrixConverter.ConvertMatrixToListOfLists(dLdw));
                nablaBiases.Insert(0, dLdb.ToList());
                dLdzs.Insert(0, dLdz.ToList());
            }
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
                    List<double> prediction = UpdateMiniBatch(imageInput, imageLabels, learningRate);
                    
                    // TODO : Get the error between network result and the expected one-hot encoding result
                    double error =
                        CalculateMeanSquareErrorLoss(prediction, imageLabels);
                    errorMiniBatch.Add(error);
                }
                double averageMiniBatchLoss = errorMiniBatch.Average();
                Console.WriteLine($"Mini Batch average loss is {averageMiniBatchLoss}");
            }
            
            // TODO : Update the weight and Biases based on batches?
            // StochasticGradientDescent(averageLoss);
            
        }
    }
    
    /// <summary>
    /// Inference Pipeline
    /// </summary>
    /// <param name="input">the input data</param>
    /// <returns>Prediction result of a file</returns> 
    public List<double> Inference(List<double> input)
    {
        List<double> prediction = ForwardPass(input, out _, out _);
        return prediction;
    }
    
    /// <summary>
    /// Predict the result based on current network or should I say Inference
    /// </summary>
    public List<double> Fit(string filePath)
    {
        double[] imageProcessing = _imageProcessing.SingleImageProcessing(filePath);
        List<double> imageInput = imageProcessing.ToList();
        List<double> prediction = Inference(imageInput);
        return prediction;
    }

    // TODO : Add function to test the result of the network
    /// <summary>
    /// Testing Pipeline
    /// </summary>
    public void Test()
    {
        
    }
}
