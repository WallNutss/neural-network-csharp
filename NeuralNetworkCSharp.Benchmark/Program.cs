using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using NeuralNetworkCSharp.Core;
using NeuralNetworkCSharp.Enum;

class Program
{
    static void Main(){
        BenchmarkRunner.Run<NeuralNetworkBenchmark>();
    }
}

[MemoryDiagnoser]
public class NeuralNetworkBenchmark
{
    private string _trainingDirectory;
    private int _epochs;
    private int _batchSize;
    private double _learningRate;
    
    private StringWriter _stringWriter;
    private TextWriter _originalConsoleOut;

    [GlobalSetup]
    public void Setup()
    {
        _trainingDirectory = "C:/Jun Local Things/Playground/Neural Network C#/mnist_png/train";
        _epochs = 5;
        _batchSize = 1000;
        _learningRate = 0.01;
        
        // Store the original Console output
        _originalConsoleOut = Console.Out;
        
        // Redirect Console output to a StringWriter (which will capture output)
        _stringWriter = new StringWriter();
        Console.SetOut(_stringWriter);
    }
    
    [GlobalCleanup]
    public void Cleanup()
    {
        // Restore the original Console output
        Console.SetOut(_originalConsoleOut);
        
        // process the captured output, e.g., save to a file or analyze it.
        string capturedOutput = _stringWriter.ToString();
    }

    // Benchmark a method for training with for-loop calculation
    [Benchmark]
    public void BenchmarkTrainUsingForLoopCalculation()
    {
        const int inputLayer = 784;
        const int hiddenLayer = 15;
        const int outputLayer = 10;
        List<int> networkSize = new List<int>() { inputLayer, hiddenLayer, outputLayer };
        
        var network = new Network(networkSize);
        network.Train(_trainingDirectory, _epochs, _batchSize, _learningRate);
    }

    // Benchmark a method for training with matrix-based calculation
    [Benchmark]
    public void BenchmarkTrainUsingMatrixCalculation()
    {
        const int inputLayer = 784;
        const int hiddenLayer = 15;
        const int outputLayer = 10;
        List<int> networkSize = new List<int>() { inputLayer, hiddenLayer, outputLayer };
        
        var network = new Network(networkSize);
        network.Train(_trainingDirectory, _epochs, _batchSize, _learningRate, BackpropagationMethod.MatrixMultiplication);
    }
}