# GPU-based SPH Fluid Simulation

This project implements a Smoothed Particle Hydrodynamics (SPH) fluid simulation using GPU acceleration. 

## Features

- GPU-accelerated fluid simulation using Compute Shaders.
- Support for simulating various fluid behaviors such as gravity, pressure, viscosity, and interactions.
- Spatial Grid Hashing for efficient neighbour search
- GPU-based sorting algorithm (Bitonic Sort)

## Usage

1. Attach the SPHFluidCompute script to a GameObject in your Unity scene.
2. Assign a Compute Shader, particle mesh, and material to the script.
3. Set simulation parameters such as the number of particles, particle radius, smoothing radius, etc.
4. Optionally, customize the simulation behavior by modifying the Compute Shader kernels and parameters.
5. Run the simulation in the Unity editor or during gameplay.

## Some comparison for performance gained on GPU

 - 150 fps with 4096 particles on CPU with multi threading. (AMD Ryzen 5950x)
 - 400 fps with 131072 particles on GPU with parallel processing on cuda cores. (Nvidia RTX 3070)

 Thats 2.5x more fps with 32x more particles.

Both the implementations were using the same GPU Instanced materials, and optimisations such as
spatial grid hashing for both cpu and gpu variant of the sim.

## Requirements

- Unity 3D (compatible with version 2022.3 and above)
- GPU that supports Compute Shaders

![GIF Showing simulation running on GPU with 131k particles at 400 fps](https://github.com/mohitkarangiya/SPHFluidSim/blob/main/GIF/test.gif?raw=true)
Simulation running on GPU with 131k particles at 400 fps