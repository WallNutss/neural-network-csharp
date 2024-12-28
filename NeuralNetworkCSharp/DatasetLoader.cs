using NeuralNetworkCSharp.Domain;

namespace NeuralNetworkCSharp;

public class DatasetLoader
{
    private static Random _rng;
    public DatasetLoader(){
        _rng = new Random();
    }
    
    // TODO : Getting the images based on training directory [Done, will delete this later]
    public Dataset LoadTrainingDataset(string trainingPath)
    {
        if(!Directory.Exists(trainingPath))
            throw new DirectoryNotFoundException();
        
        // Initialize the images and their own decoding
        List<string> imagesPath = new ();
        List<double[]> labels = new();
        
        // Get all the images on that training directory
        imagesPath.AddRange(Directory.GetFiles(trainingPath, "*.png", SearchOption.AllDirectories));
        
        // Get the labels name based on the subfolder directory
        List<string> labelsName = new List<string>();
        labelsName.AddRange(Directory.GetDirectories(trainingPath).Select(dir => Path.GetFileName(dir)!));
        
        // Get the one-hot encoding of the images label based on their location on that folder
        foreach (var imagePath in imagesPath)
        {
            var labelTag = new DirectoryInfo(Path.GetDirectoryName(imagePath)!).Name;
            double[] encoding = LabelEncoding(labelsName, labelTag);
            labels.Add(encoding);
        }

        var dataset = new Dataset()
        {
            TrainingDatasetImages = imagesPath,
            TrainingDatasetLabels = labels
        };
        return dataset;
    }
    
    // TODO : Getting the label encoding [Done, will delete this later]
    private double[] LabelEncoding(List<string> labels, string label)
    {
        double[] labelEncoding = new double[labels.Count];
        for (int i = 0; i < labelEncoding.Length; i++)
        {
            if (labels[i] == label)
                labelEncoding[i] = 1;
            else
                labelEncoding[i] = 0;   
        }
        return labelEncoding;
    }
    
    // TODO : Making randomize the order of images and their labels [Done, will delete this later]
    public (List<string>, List<double[]>) ShuffleDataset(List<string> images, List<double[]> labels, int batchSize = 100)
    {
        int count = images.Count;
        List<string> localBatchimagesPath = new();
        List<double[]> localBatchlabels = new();
        
        // Based on this Fisher-Yates Shuffle from this stackoverlflow forum
        // https://stackoverflow.com/questions/273313/randomize-a-listt
        
        // NOTE : Perhaps not shuffle, but more like permutation?
        List<int> permutationIndex = Permutation(count, batchSize);

        foreach (var p in permutationIndex)
        {
            localBatchimagesPath.Add(images[p]);
            localBatchlabels.Add(labels[p]);
        }
        return (localBatchimagesPath, localBatchlabels);
    }

    private List<int> Permutation(int size, int batchSize)
    {
        List<int> permutation = new();
        while (batchSize > 0)
        {
            int k = _rng.Next(0, size);
            if(permutation.Contains(k)) continue;
            
            permutation.Add(k);
            batchSize--;
        }
        return permutation;
    }
}

