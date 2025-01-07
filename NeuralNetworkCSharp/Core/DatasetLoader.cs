using NeuralNetworkCSharp.Domain;

namespace NeuralNetworkCSharp.Core;

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
    public List<(List<string> images, List<double[]> labels)> ShuffleDataset(List<string> images, List<double[]> labels, int batchSize = 100)
    {
        int count = images.Count;
        int batchSet = BatchSet(batchSize, count);
        (List<string> shuffledImages, List<double[]> shuffledLabels) = Shuffle(images, labels);
        List<(List<string> images, List<double[]> labels)> batches = new();
        
        int index = 0;
        for (int i = 0; i < batchSet; i++)
        {
            List<string> miniBatchImages = new();
            List<double[]> miniBatchLabels = new();

            for (int j = 0 ; j < batchSize && index < count ; j++)
            {
                miniBatchImages.Add(shuffledImages[index]);
                miniBatchLabels.Add(shuffledLabels[index]);
                index++;
            }
            batches.Add((miniBatchImages, miniBatchLabels));
        }
        return batches;
    }
    
    private (List<string>, List<double[]>) Shuffle(List<string> listImages, List<double[]> listLabels)
    {
        // Based on this Fisher-Yates Shuffle from this stackoverlflow forum
        // https://stackoverflow.com/questions/273313/randomize-a-listt
        
        Random rng = new Random();
        int n = listImages.Count;
        while (n > 0)
        {
            n--;
            int k = rng.Next(n + 1);
            (listImages[k], listImages[n]) = (listImages[n], listImages[k]);
            (listLabels[k], listLabels[n]) = (listLabels[n], listLabels[k]);
        }
        return (listImages, listLabels);
    }

    private int BatchSet(int batchSize, int totalImage)
    {
        int remainder = totalImage % batchSize;
        int result = totalImage / batchSize;
        if (remainder == 0) return result;
        return result + 1;
    }
}

