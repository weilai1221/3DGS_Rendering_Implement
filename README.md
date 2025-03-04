# 3DGS_Rendering_Implement
This repository implements the 3DGS rendering engine by translating the original graphdeco-inria/gaussian-splatting C++/CUDA code to Python and PyTorch. It modularizes the rendering process for clarity and easier maintenance.


Diff-gaussian-rasterization Workflow

```
Python (`_C.rasterize_gaussians()`)
    ├──> C++ (`RasterizeGaussiansCUDA()`)  [rasterize_points.cu]
    │     ├──> `CudaRasterizer::Rasterizer::forward()` [rasterizer_impl.cu]
    │     │     ├──> `FORWARD::preprocess()` [forward.cu] → 計算投影 & Covariance
    │     │     ├──> Tile Sorting (`cub::DeviceRadixSort::SortPairs()`) [rasterizer_impl.cu]
    │     │     ├──> `FORWARD::render()` [forward.cu] → Alpha Blending & 最終影像輸出
    │     │
    │     └──> **回傳 `out_color`, `depth` 給 Python**
```


trace preprocessCUDA
1. 
// Initialize radius and touched tiles to 0. If this isn't changed, this Gaussian will not be processed further.

2.
// Perform near culling, quit if outside.

3.
// Transform point by projecting

4.
// If 3D covariance matrix is precomputed, use it, otherwise compute from scaling and rotation parameters. 

5.
// Compute 2D screen-space covariance matrix

6.
// Invert covariance (EWA algorithm)

7.
// Compute extent in screen space (by finding eigenvalues of
// 2D covariance matrix). Use extent to compute a bounding rectangle
// of screen-space tiles that this Gaussian overlaps with. Quit if
// rectangle covers 0 tiles. 


8.
// If colors have been precomputed, use them, otherwise convert
// spherical harmonics coefficients to RGB color.

9.
// Store some useful helper data for the next steps.

10.
// Inverse 2D covariance and opacity neatly pack into one float4


