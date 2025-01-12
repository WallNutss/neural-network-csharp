using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetworkCSharp.Helper;

public static class MatrixConverter
{
    public static Matrix<double> ConvertListOfListsToMatrix(List<List<double>> listOfLists)
    {
        // Convert List<List<double>> to a jagged array (double[][])
        double[][] jaggedArray = listOfLists.Select(row => row.ToArray()).ToArray();

        // Create a double[,] array (multidimensional array) from the jagged array
        int rows = jaggedArray.Length;
        int cols = jaggedArray[0].Length;
        double[,] array = new double[rows, cols];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                array[i, j] = jaggedArray[i][j];
            }
        }

        // Create a Matrix<double> from the double[,] array
        return Matrix<double>.Build.DenseOfArray(array);
    }

    public static List<List<double>> ConvertMatrixToListOfLists(Matrix<double> matrix)
    {
        // Initialize the list of lists
        var listOfLists = new List<List<double>>();

        // Iterate over the rows of the matrix
        for (int i = 0; i < matrix.RowCount; i++)
        {
            var rowList = new List<double>();
            for (int j = 0; j < matrix.ColumnCount; j++)
            {
                rowList.Add(matrix[i, j]);
            }
            listOfLists.Add(rowList);
        }
        return listOfLists;
    }
    
}