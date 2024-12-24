using System.Drawing;

namespace NeuralNetworkCSharp;

public class ImageProcessing
{
    public double[] ImagetoByteArray(string imagePath)
    {
        double[] image = new double[imagePath.Length];
        
        using (Bitmap bitmap = new Bitmap(imagePath))
        {
            // Ensure the image is 28x28 pixels
            if (bitmap.Width != 28 || bitmap.Height != 28)
            {
                throw new InvalidOperationException("Each image must be 28x28 pixels.");
            }

            // Convert the image to a 1D byte array (flattened)
            image = GetFlattenedImage(bitmap);
        }
        return image;
    }
    private static double[] GetFlattenedImage(Bitmap bitmap)
    {
        int width = bitmap.Width;
        int height = bitmap.Height;
        int pixelCount = width * height;

        double[] result = new double[728];  // We want 728 values (28x28 = 784, excluding last 56 pixels)

        int index = 0;

        // Loop through the pixels (grayscale, so just take one channel)
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                Color pixelColor = bitmap.GetPixel(x, y);
                // Convert to grayscale using the standard formula: 0.299 * R + 0.587 * G + 0.114 * B
                double grayValue = (byte)(0.299 * pixelColor.R + 0.587 * pixelColor.G + 0.114 * pixelColor.B);

                if (index < 728)  // Only copy the first 728 pixels
                {
                    result[index++] = grayValue;
                }
            }
        }

        return result;
    }
}