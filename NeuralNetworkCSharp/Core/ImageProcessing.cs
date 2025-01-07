using SkiaSharp;

namespace NeuralNetworkCSharp.Core;

public class ImageProcessing
{
    // TODO : Get the bytes array based on the training dataset [Done, will delete this later]
    public List<double[]> BatchImageProcessing(List<string> imagePaths)
    {
        List<double[]> batchImage = new List<double[]>();
        foreach (var imagePath in imagePaths)
        {
            batchImage.Add(SingleImageProcessing(imagePath));
        }
        return batchImage;
    }
    public double[] SingleImageProcessing(string imagePath)
    {
        double[] image = new double[728];
        
        // Decode the .jpeg/png data
        using (var skImage = SKImage.FromEncodedData(imagePath))
        {
            // Get the bitmap
            using (var bitmap = SKBitmap.FromImage(skImage))
            {
                // Ensure the image is 28x28 pixels
                if (bitmap.Width != 28 || bitmap.Height != 28)
                {
                    throw new InvalidOperationException("Each image must be 28x28 pixels.");
                }
                
                // Convert the image to a 1D byte array (flattened)
                image = GetFlattenedImage(bitmap);
            }
        }
        
        return image;
    }
    private static double[] GetFlattenedImage(SKBitmap bitmap)
    {
        int width = bitmap.Width;
        int height = bitmap.Height;
        int pixelCount = width * height;

        double[] result = new double[pixelCount];

        int index = 0;
        // Loop through the pixels (grayscale, so just take one channel)
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                var color = bitmap.GetPixel(x, y);
                // Convert to grayscale using the standard formula: 0.299 * R + 0.587 * G + 0.114 * B
                double grayValue = (byte)(0.299 * color.Red + 0.587 * color.Green + 0.114 * color.Blue);
                // Copy the grayscale pixels
                result[index++] = grayValue;
            }
        }

        return result;
    }
}