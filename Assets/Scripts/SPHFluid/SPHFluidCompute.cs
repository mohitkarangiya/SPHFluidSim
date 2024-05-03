using System;
using UnityEngine;

public class SPHFluidCompute : MonoBehaviour
{
    [SerializeField]ComputeShader computeShader;

    [SerializeField]Mesh particleMesh;
    [SerializeField]Material particleMat;
    [SerializeField]BoxCollider2D col;

    [Space(5)]
    [SerializeField]int numOfParticles = 1; //the program only works with 2^n number of particles for now
    [SerializeField]float radius = 0.5f;
    [SerializeField]float smoothingRadius = 0.5f;
    [SerializeField]float spacing = 0f;
    [SerializeField]float gravity = -9.8f;
    [SerializeField]float lookAheadFactor = -9.8f;
    bool applyGravity = true;
    [SerializeField]float viscosityStrength = 0.1f;
    [SerializeField][Range(0f,1f)]float collisionDamping = 1;

    [Space(10)]
    [SerializeField]float targetDensity;
    [SerializeField]float pressureMultiplier;
    [SerializeField]float nearPressureMultiplier;

    [Space(10)]
    [SerializeField]float maxVisualVelocity;
    [SerializeField]float maxVisualDensity;
    [SerializeField]Gradient colormap;
    [SerializeField]Texture2D colorMapTex;
    [SerializeField]int colorMapResolution;

    [Space(10)]
    public int iterations = 5;
    public bool simulate;
    public bool drawGizmos;
    public bool fixedTimeStep;
    [Range(0f,1f)]public float timeScale;

    [Space(10)]
    [SerializeField]float mouseInteractionRadius = 1f;
    [SerializeField]float mouseInteractionStrength = 5f;

    const float PI = Mathf.PI;

    // Compute Buffers vars
    ComputeBuffer velocityBuffer;
    ComputeBuffer positionBuffer;
    ComputeBuffer predictedPositionBuffer;
    ComputeBuffer densityBuffer;
    ComputeBuffer spatialLookupBuffer;
    ComputeBuffer startIndicesBuffer;
    Vector2 boundsSize;

    // Precomputed kernel volume vars
    float smoothingKernelVolume;
    float spikyKernelPow2Volume;
    float spikyKernelPow2DerivativeVolume;
    float spikyKernelPow3Volume;
    float spikyKernelPow3DerivativeVolume;

    // Compute Shader Kernel Indices
    int applyGravitykernel;
    int calculatePredictedPositionsKernel;
    int updateSpatialLookupArrayKernel;
    int bitonicSortKernel;
    int updateStartIndicesArrayKernel;
    int calculateDensitiesKernel;
    int calculatePressureForceKernel;
    int calculateViscosityForceKernel;
    int interactionForceKernel;
    int updatePositionsKernel;
    int resolveCollisionsKernel;

    void Start()
    {
        //Initializing Compute Buffers
        velocityBuffer = new ComputeBuffer(numOfParticles,sizeof(float)*2);
        positionBuffer = new ComputeBuffer(numOfParticles,sizeof(float)*2);
        predictedPositionBuffer = new ComputeBuffer(numOfParticles,sizeof(float)*2);
        densityBuffer = new ComputeBuffer(numOfParticles,sizeof(float)*2);
        spatialLookupBuffer = new ComputeBuffer(numOfParticles,sizeof(uint)*2);
        startIndicesBuffer = new ComputeBuffer(numOfParticles,sizeof(uint));

        //Initializing Particles
        InitialiseParticles();
        //Initializing Volume var for smoothing kernels
        UpdateVolumes();
        //Generating texture from color map
        GenerateColorMap();

        //Passing compute buffers to particle mat
        particleMat.SetBuffer("velocity",velocityBuffer);
        particleMat.SetBuffer("position",positionBuffer);


        //Initialising kernel index vars
        applyGravitykernel = computeShader.FindKernel("ApplyGravity");
        calculatePredictedPositionsKernel = computeShader.FindKernel("CalculatePredictedPositions");
        updateSpatialLookupArrayKernel = computeShader.FindKernel("UpdateSpatialLookupArray");
        bitonicSortKernel = computeShader.FindKernel("BitonicSort");
        updateStartIndicesArrayKernel = computeShader.FindKernel("UpdateStartIndicesArray");
        calculateDensitiesKernel = computeShader.FindKernel("CalculateDensities");
        calculatePressureForceKernel = computeShader.FindKernel("CalculatePressureForce");
        calculateViscosityForceKernel = computeShader.FindKernel("CalculateViscosityForce");
        interactionForceKernel = computeShader.FindKernel("ApplyInteractionForce");
        updatePositionsKernel = computeShader.FindKernel("UpdatePositions");
        resolveCollisionsKernel = computeShader.FindKernel("ResolveCollisions");

        //Setting up compute buffers for different kernels
        SetBuffer(computeShader,positionBuffer,"positions",calculatePredictedPositionsKernel,interactionForceKernel,updatePositionsKernel,resolveCollisionsKernel);
        SetBuffer(computeShader,predictedPositionBuffer,"predictedPositions",calculatePredictedPositionsKernel,updateSpatialLookupArrayKernel,interactionForceKernel,calculateDensitiesKernel,calculatePressureForceKernel,calculateViscosityForceKernel);
        SetBuffer(computeShader,velocityBuffer,"velocities",applyGravitykernel,calculatePredictedPositionsKernel,calculatePressureForceKernel,calculateViscosityForceKernel,interactionForceKernel,updatePositionsKernel,resolveCollisionsKernel);
        SetBuffer(computeShader,densityBuffer,"densities",calculatePressureForceKernel,calculateDensitiesKernel);
        SetBuffer(computeShader,spatialLookupBuffer,"spatialLookup",updateSpatialLookupArrayKernel,bitonicSortKernel,updateStartIndicesArrayKernel,calculateDensitiesKernel,calculatePressureForceKernel,calculateViscosityForceKernel);
        SetBuffer(computeShader,startIndicesBuffer,"startIndices",updateSpatialLookupArrayKernel,bitonicSortKernel,updateStartIndicesArrayKernel,calculateDensitiesKernel,calculatePressureForceKernel,calculateViscosityForceKernel);

        computeShader.SetInt("numOfParticles",numOfParticles);
    }

    void UpdateVolumes()
    {
        // Computes and updates variables req. for smoothing kernel calculations
        smoothingKernelVolume = PI * Mathf.Pow(smoothingRadius, 8) / 4f;
        spikyKernelPow2Volume = PI * Mathf.Pow(smoothingRadius,4) / 6f;
        spikyKernelPow2DerivativeVolume = PI * Mathf.Pow(smoothingRadius,4)/12f;
        spikyKernelPow3Volume = PI * Mathf.Pow(smoothingRadius, 5) / 10f;
        spikyKernelPow3DerivativeVolume =  PI * Mathf.Pow(smoothingRadius, 5)/30f;

        computeShader.SetFloat("smoothingKernelVolume",smoothingKernelVolume);
        computeShader.SetFloat("spikyKernelPow2Volume",spikyKernelPow2Volume);
        computeShader.SetFloat("spikyKernelPow2DerivativeVolume",spikyKernelPow2DerivativeVolume);
        computeShader.SetFloat("spikyKernelPow3Volume",spikyKernelPow3Volume);
        computeShader.SetFloat("spikyKernelPow3DerivativeVolume",spikyKernelPow3DerivativeVolume);
    }

    void FixedUpdate()
    {
        if(fixedTimeStep)Simulate(Time.fixedDeltaTime);
    }

    private void Update()
    {
        if(Input.GetKeyDown(KeyCode.G))
            ToggleGravity();

        if(Input.GetKeyDown(KeyCode.R))
            InitialiseParticles();

        if(!fixedTimeStep && Time.frameCount>10)
            Simulate(Time.deltaTime);

        DrawParticles();

        if(Input.GetKeyDown(KeyCode.F))
            fixedTimeStep = !fixedTimeStep;
    }

    void Simulate(float deltaTime)
    {
        // Runs sim in multiple iterations
        float dt = deltaTime/iterations * timeScale;
        for(int i = 0; i< iterations; ++i)
            SimulationStep(dt);
    }

    void SimulationStep(float deltaTime)
    {
        // Executes one simulation step for given deltaTime timeframe
        if(!simulate)return;
        boundsSize = col.bounds.size;

        // Passing required parameteres to gpu
        computeShader.SetFloat("deltaTime",deltaTime);
        computeShader.SetFloat("gravity",gravity);
        computeShader.SetFloat("targetDensity",targetDensity);
        computeShader.SetFloat("smoothingRadius",smoothingRadius);
        computeShader.SetFloat("lookAheadFactor",lookAheadFactor);
        computeShader.SetVector("boundsSize",boundsSize);
        computeShader.SetFloat("collisionDamping",collisionDamping);
        computeShader.SetFloat("pressureMultiplier",pressureMultiplier);
        computeShader.SetFloat("nearPressureMultiplier",nearPressureMultiplier);
        computeShader.SetFloat("viscosityStrength",viscosityStrength);

        // Calculating thread group size.
        int threadGroups = Mathf.CeilToInt(numOfParticles/128f);

        // Gravity kernel
        if(applyGravity)
            computeShader.Dispatch(applyGravitykernel,threadGroups,1,1);

        // Predicted Position kernel
        computeShader.Dispatch(calculatePredictedPositionsKernel,threadGroups,1,1);
        // Updating Spatial Lookup Array
        UpdateSpatialLookup(threadGroups);
        // Calculating Densities
        computeShader.Dispatch(calculateDensitiesKernel,threadGroups,1,1);
        // Calculating Pressure
        computeShader.Dispatch(calculatePressureForceKernel,threadGroups,1,1);
        // Calculating Viscosity
        computeShader.Dispatch(calculateViscosityForceKernel,threadGroups,1,1);

        if(Input.GetMouseButton(0) || Input.GetMouseButton(1))
        {
            // Apply Interaction Forces
            computeShader.SetFloat("interactionStrength",Input.GetMouseButton(0)?mouseInteractionStrength:-mouseInteractionStrength);
            computeShader.SetVector("interactionPos",(Vector2)MouseToWorldPos());
            computeShader.SetFloat("interactionRadius",mouseInteractionRadius);
            computeShader.Dispatch(interactionForceKernel,threadGroups,1,1);
        }

        // Update Positions
        computeShader.Dispatch(updatePositionsKernel,threadGroups,1,1);
        // Resolve Boundary Collisions
        computeShader.Dispatch(resolveCollisionsKernel,threadGroups,1,1);
    }

    void DrawParticles()
    {   
        // Draws Particles with GPU Instancing
        particleMat.SetFloat("maxvelocity",maxVisualVelocity);
        // particleMat.SetFloat("maxdensity",maxVisualDensity);
        particleMat.SetFloat("radius",radius*2);
        Graphics.DrawMeshInstancedProcedural(particleMesh,0,particleMat,col.bounds,numOfParticles,null,UnityEngine.Rendering.ShadowCastingMode.Off,false);
    }

    void UpdateSpatialLookup(int threadGroups)
    {
        // Update Spatial Lookup Array
        computeShader.Dispatch(updateSpatialLookupArrayKernel,threadGroups,1,1);
        Sort();
        computeShader.Dispatch(updateStartIndicesArrayKernel,threadGroups,1,1);
    }

    void Sort()
    {
        // Bitonic Sort on GPU
        // Recursive bitonic sequence creation, starting from size 2 until size
        // is equal to numofparticles
        int nextPowerof2 = Mathf.NextPowerOfTwo(numOfParticles);
        int numThreadGroups = Mathf.Max(nextPowerof2/2/128,1);

        for (int cbs = 2; cbs <= nextPowerof2; cbs *= 2)
        {
            computeShader.SetInt("cbs",cbs);
            for(int cd = cbs/2;cd>0;cd/=2)
            {
                computeShader.SetInt("cd",cd);
                computeShader.Dispatch(bitonicSortKernel,numThreadGroups,1,1);
            }
        }
    }

    private void InitialiseParticles()
    {
        // Initialises particle positions and velocities
        // and passes the data to related compute buffers
        Vector2[] positions = new Vector2[numOfParticles];
        Vector2[] predictedPositions = new Vector2[numOfParticles];
        Vector2[] velocities = new Vector2[numOfParticles];

        int particlePerRow = (int)Mathf.Sqrt(numOfParticles);
        int particlePerCol = Mathf.CeilToInt(numOfParticles/particlePerRow);
        float spawngap = radius*2 + spacing;

        float x=0,y=0;
        for(int i = 0; i < numOfParticles;i++)
        {
            x = (i%particlePerRow - (particlePerRow-1)/2f)*spawngap;
            y = (i/particlePerRow - (particlePerCol-1)/2f)*spawngap;
            positions[i] = predictedPositions[i] = new Vector2(x,y);
            velocities[i] = Vector2.zero;
        }

        positionBuffer.SetData(positions);
        predictedPositionBuffer.SetData(predictedPositions);
        velocityBuffer.SetData(velocities);
    }

    private void ToggleGravity()
    {
        //Toggles Gravity;
        applyGravity = !applyGravity;
    }

    void GenerateColorMap()
    {
        // Creates a texture from color gradient
        if(colorMapTex == null)
        {
            colorMapTex = new Texture2D(colorMapResolution, 1)
            {
                wrapMode = TextureWrapMode.Clamp,
            };
        }

        Color[] pixels = new Color[colorMapResolution];
        for(int i = 0; i<colorMapResolution; i++)
            pixels[i] = colormap.Evaluate(Mathf.InverseLerp(0,colorMapResolution-1,i));
        colorMapTex.SetPixels(pixels);
        colorMapTex.Apply();
        particleMat.SetTexture("ColourMap",colorMapTex);
    }

    Vector3 MouseToWorldPos()
    {
        // Returns WorldSpace Position from mouse position
        return Camera.main.ScreenToWorldPoint(new Vector3(Input.mousePosition.x,Input.mousePosition.y,-Camera.main.transform.position.z));
    }

    private void OnValidate()
    {
        if(numOfParticles<0)numOfParticles = 0;
        if(iterations<1)iterations = 1;
        GenerateColorMap();
        UpdateVolumes();
    }

    private void OnDrawGizmos()
    {
        if(!drawGizmos) return;
        if(Input.GetMouseButton(0) || Input.GetMouseButton(1))
        {
            Gizmos.color = Color.blue;
            Vector3 samplePos = MouseToWorldPos();
            Gizmos.DrawWireSphere(samplePos,mouseInteractionRadius);
        }
    }

    private void OnDestroy()
    {
        // Releasing Compute Buffers
        velocityBuffer.Release();
        positionBuffer.Release();
        densityBuffer.Release();
        predictedPositionBuffer.Release();
        spatialLookupBuffer.Release();
        startIndicesBuffer.Release();
    }

    void SetBuffer(ComputeShader compute,ComputeBuffer buffer,string nameId,params int[] kernels)
    {
        // Helper function to set buffers to kernels
        foreach(int kernel in kernels)
            compute.SetBuffer(kernel,nameId,buffer);
    } 
}