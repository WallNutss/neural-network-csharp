using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using NeuralNetworkCSharp.Domain;
using NeuralNetworkCSharp.Enum;
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
    /// Generate initial value of nabla biases in their each layer
    /// </summary>
    /// <param name="x">The value you want to set on the nabla biases</param>
    /// <returns>List of nabla biases of each neuron in their each layer</returns>
    private List<List<double>> GenerateInitialNablaBiases(double x)
    {
        List<List<double>> nablaBiases = new List<List<double>>();
        for (int i = 1; i < NumLayers; i++)
        {
            List<double> layerBiases = new List<double>();

            for (int j = 0; j < Sizes[i]; j++)
                layerBiases.Add(x);
            
            nablaBiases.Add(layerBiases);
        }
        return nablaBiases;
    } 
    
    /// <summary>
    /// Generate initial value of nabla weights in their each layer
    /// </summary>
    /// <param name="x">The value you want to set on the nabla weights</param>
    /// <returns>List of nabla weights of each neuron in their each layer</returns>
    private List<List<List<double>>> GenerateInitialNablaWeights(double x)
    {
        List<List<List<double>>> nablaWeights = new ();
        for (int i = 1; i < NumLayers; i++)
        {
            List<List<double>> layerWeights = new List<List<double>>();

            for (int j = 0; j < Sizes[i]; j++)
            {
                List<double> neuronWeights = new List<double>();
                for (int k = 0; k < Sizes[i - 1]; k++)
                    neuronWeights.Add(x);
                
                layerWeights.Add(neuronWeights);
            }
            nablaWeights.Add(layerWeights);
        }
        return nablaWeights;
    }
    
    /// <summary>
    /// Importing Custom Weights based on the Network Layer and their individual neuron inputs
    /// Follow the scheme of index 0 start from the left, with the first input of the current L-Layer the first
    /// Follow the second index 1 of their second weight on that neuron
    /// </summary>
    /// <param name="weights">List of the weight in form of array
    /// The total of arrays must match to the number of the network weights in all layer</param>
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
    /// <param name="biases">List of the biases network in form of array
    /// The total of arrays must match to the number of the network biases in all layer</param>
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
    /// Enumerate the gradients parameter of the network from each data backpropagation result on the batch data
    /// </summary>
    /// <param name="batchGradientParameters">The accumulated gradients parameter value of the batch</param>
    /// <param name="dataGradientParameters">The gradients parameter value that was got from the backpropagation result</param>
    /// <returns>The add up value of nabla weights and nabla biases of the batch result</returns>
    public GradientParameters AccumulatedGradients(GradientParameters batchGradientParameters, GradientParameters dataGradientParameters)
    {
        GradientParameters accumulatedGradientParameters = new GradientParameters()
        {
            NablaBiases = AccumulateNablaBiases(batchGradientParameters.NablaBiases, dataGradientParameters.NablaBiases),
            NablaWeights = AccumulateNablaWeights(batchGradientParameters.NablaWeights, dataGradientParameters.NablaWeights)
        };
        return accumulatedGradientParameters;
    }
    
    /// <summary>
    /// Enumerate the biases parameter of the network from each data backpropagation result on the batch data
    /// </summary>
    /// <param name="batchNablaBiases">The accumulated biases parameter value of the batch</param>
    /// <param name="nablaBiases">The biases parameter value that was got from the backpropagation result</param>
    /// <returns>The add up value of nabla biases of the batch result</returns>
    private List<List<double>> AccumulateNablaBiases(List<List<double>> batchNablaBiases, List<List<double>> nablaBiases)
    {
        if (batchNablaBiases[0].Count != nablaBiases[0].Count)
            throw new InvalidOperationException("The dimension of nabla biases from the batch must be the same as the dimension of the nabla biases on the data");
        
        for (int i = 0; i < batchNablaBiases.Count; i++)
        {
            for (int j = 0; j < batchNablaBiases[i].Count; j++)
            {
                if (batchNablaBiases[i].Count != nablaBiases[i].Count)
                    throw new InvalidOperationException($"The dimension of the intermitten layer of B-Layer[{i}][{j}] does not match the number of network biases");
                
                batchNablaBiases[i][j] += nablaBiases[i][j];
            }
        }
        return batchNablaBiases;
    }
    
    /// <summary>
    /// Enumerate the weights parameter of the network from each data backpropagation result on the batch data
    /// </summary>
    /// <param name="batchNablaWeights">The accumulated weights parameter value of the batch</param>
    /// <param name="nablaWeights">The weights parameter value that was got from the backpropagation result</param>
    /// <returns>The add up value of nabla weights of the batch result</returns>
    private List<List<List<double>>> AccumulateNablaWeights(List<List<List<double>>> batchNablaWeights, List<List<List<double>>> nablaWeights)
    {
        if (batchNablaWeights[0].Count != nablaWeights[0].Count)
            throw new InvalidOperationException("The dimension of weights must be the same as the dimension of the nabla weights network");
        
        for (int i = 0; i < batchNablaWeights.Count; i++)
        {
            for (int j = 0; j < batchNablaWeights[i].Count; j++)
            {
                if (batchNablaWeights[i][j].Count != nablaWeights[i][j].Count)
                    throw new InvalidOperationException($"The dimension of the intermitten layer of W-Layer[{i}][{j}] does not match the number of network weights");
                
                for (int k = 0; k < batchNablaWeights[i][j].Count; k++)
                {
                    batchNablaWeights[i][j][k] += nablaWeights[i][j][k];
                }
            }
        }
        return batchNablaWeights;
    }
    
    /// <summary>
    /// Update the weights
    /// </summary>
    /// <param name="nablaWeights">Gradient of weights over the loss function</param>
    /// <param name="batchSize">batch size</param>
    /// <param name="learningRate">learning rate</param>
    /// <returns>Update the weights of the network</returns>
    public void UpdateWeights(List<List<List<double>>> nablaWeights, int batchSize, double learningRate)
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
                    Weights[i][j][k] += -(learningRate/batchSize) * nablaWeights[i][j][k];
                }
            }
        }
    }
  
    /// <summary>
    /// Update the biases
    /// </summary>
    /// <param name="nablaBiases">Gradient of bias over the loss function</param>
    /// <param name="batchSize">batch size</param>
    /// <param name="learningRate">learning rate</param>
    /// <returns>Update the biases of the network</returns>
    public void UpdateBiases(List<List<double>> nablaBiases, int batchSize, double learningRate)
    {
        if (Biases[0].Count != nablaBiases[0].Count)
            throw new InvalidOperationException("The dimension of biases must be the same as the dimension of the nabla biases network");
        
        for (int i = 0; i < Biases.Count; i++)
        {
            for (int j = 0; j < Biases[i].Count; j++)
            {
                if (Biases[i].Count != nablaBiases[i].Count)
                    throw new InvalidOperationException($"The dimension of the intermitten layer of B-Layer[{i}][{j}] does not match the number of network biases");
                
                Biases[i][j] += -(learningRate/batchSize) * nablaBiases[i][j];
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
    /// <param name="batch">the input data</param>
    /// <param name="learningRate">learning rate - the value of how much gradient descent will step</param>
    /// <param name="method">for calculating the weights and bias update, prefer to calculate it straight in each layer at the same time (Matrix Calculation) or one at-a-time in each layer in every neuron</param>
    /// <returns>Result prediction of the output layer</returns>
    public double UpdateMiniBatch(BatchData batch,
        double learningRate=0.01, BackpropagationMethod method = BackpropagationMethod.Forloop)
    {
        // Get the batch size
        if (batch.Images.Count != batch.Labels.Count)
            throw new InvalidOperationException("The number of image batches must match the number of images labels!");
        int batchSize = batch.Images.Count;
        
        // Initialize the gradient for this batch to store the accumulated gradients that should be update weight and biases
        GradientParameters batchGradientParameter = new GradientParameters()
        {
            NablaBiases = GenerateInitialNablaBiases(0),
            NablaWeights = GenerateInitialNablaWeights(0),
        };
        
        // Initialize the error of the batch for reporting
        List<double> errorMiniBatch = new();
        
        for (int k = 0; k < batchSize; k++)
        {
            double[] imageProcessing = _imageProcessing.SingleImageProcessing(batch.Images[k]);
            List<double> imageInput = imageProcessing.ToList();
            List<double> imageLabels = batch.Labels[k].ToList();
            
            if(imageInput.Count != Sizes[0]) 
                throw new Exception($"The number of inputs must match the number of input perceptron. Current Input perceptron {Sizes[0]}");
            
            // Run forward pass and get activations and it's error
            List<List<double>> activations;
            List<List<double>> zs;
            List<double> outputActivation = ForwardPass(imageInput, out activations, out zs);
            double error = CalculateMeanSquareErrorLoss(outputActivation, imageLabels);
            errorMiniBatch.Add(error);
            
            // Do backpropagation and get its nabla weight and biases weight and biases
            GradientParameters gradientParametersData =  Backpropagation(activations, zs, outputActivation, imageLabels, learningRate, method);
            
            // Accumulate the gradients result of the weight and biases of this batch
            batchGradientParameter = AccumulatedGradients(batchGradientParameter, gradientParametersData);
        }
        double averageBatchError = errorMiniBatch.Average();
        
        UpdateWeights(batchGradientParameter.NablaWeights, batchSize, learningRate);
        UpdateBiases(batchGradientParameter.NablaBiases, batchSize, learningRate);
        
        return averageBatchError;
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
    /// <param name="method">for calculating the weights and bias update, prefer to calculate it straight in each layer at the same time (Matrix Calculation) or one at-a-time in each layer in every neuron</param>
    /// <returns>nabla weights and biases - the change of rate of each one of weights and biases on the network</returns>
    public GradientParameters Backpropagation(List<List<double>> activations, List<List<double>> zs, 
        List<double> prediction, List<double> target, double learningRate, BackpropagationMethod method = BackpropagationMethod.Forloop)
    {
        int intermittenLayer = NumLayers - 2;
        
        List<List<List<double>>> nablaWeights = new();
        List<List<double>> nablaBiases = new();
        List<List<double>> dLdzs = new();

        if (method == BackpropagationMethod.Forloop)
        {
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
                        // Instead of this, why not transpose the weight and times it using hadamard product?
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
        }
        else if (method == BackpropagationMethod.MatrixMultiplication)
        {
            // Start the weight and bias update using gradient descent with matrix multiplication (hadamard product)
            for (int i = intermittenLayer; i >= 0; i--)
            {
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
                    var dLdA = Vector<double>.Build.Dense(dzdA.RowCount);

                    // Perform the element-wise multiplication for each row and sum
                    for (int k = 0; k < dzdA.RowCount; k++)
                        dLdA[k] = dLdzPreviousLayer.PointwiseMultiply(dzdA.Row(k)).Sum();
                    
                    var dLdz = dLdA.PointwiseMultiply(dAdz);
                    
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
        }
        
        GradientParameters gradientParameters = new()
        {
            NablaBiases = nablaBiases,
            NablaWeights = nablaWeights
        };
        return gradientParameters;
    }
    
    
    // TODO : Training Loops Mechanism
    /// <summary>
    /// Training Pipeline
    /// </summary>
    public void Train(string trainingDirectory, int epochs, 
        int batchSize, double learningRate, BackpropagationMethod method = BackpropagationMethod.Forloop)
    {
        // Load the training dataset and get their label based on one-hot encoding
        Dataset dataset = _datasetLoader.LoadTrainingDataset(trainingDirectory);
        
        // Start training process
        for (int i = 1; i <= epochs; i++)
        {
            Console.WriteLine($"Epoch #{i}/{epochs}....");
            // Get the shuffle version of dataset
            List<BatchData> batches =
                _datasetLoader.ShuffleDataset(dataset.TrainingDatasetImages, dataset.TrainingDatasetLabels, batchSize);

            // Processing the entire dataset but limit it by one batch at a time
            for (int j = 0; j < batches.Count; j++)
            {
                Console.WriteLine($"Epoch #{i}/{epochs} | Batch {j + 1}/{batches.Count}....");
                
                // Processing mini-batch or update mini-batch, whether it's the batch size is 1 or more than one
                double averageMiniBatchLoss = UpdateMiniBatch(batches[j], learningRate, method);
                
                Console.WriteLine($"Mini Batch average loss is {averageMiniBatchLoss}");
            }
            
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
