```

BenchmarkDotNet v0.14.0, Windows 11 (10.0.22631.4602/23H2/2023Update/SunValley3)
13th Gen Intel Core i7-1360P, 1 CPU, 16 logical and 12 physical cores
.NET SDK 8.0.404
  [Host]     : .NET 8.0.11 (8.0.1124.51707), X64 RyuJIT AVX2
  DefaultJob : .NET 8.0.11 (8.0.1124.51707), X64 RyuJIT AVX2


```
| Method                                | Mean    | Error   | StdDev  | Gen0          | Gen1         | Gen2         | Allocated |
|-------------------------------------- |--------:|--------:|--------:|--------------:|-------------:|-------------:|----------:|
| BenchmarkTrainUsingForLoopCalculation | 28.84 s | 0.502 s | 0.493 s | 16652000.0000 |    6000.0000 |    2000.0000 | 145.84 GB |
| BenchmarkTrainUsingMatrixCalculation  | 82.44 s | 1.487 s | 1.391 s | 18905000.0000 | 5265000.0000 | 1799000.0000 | 175.78 GB |
