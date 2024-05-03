using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;

public class SPHFluid : MonoBehaviour
{
    [SerializeField]Mesh particleMesh;
    [SerializeField]Material particleMat;
    [SerializeField]BoxCollider2D col;

    [Space(5)]
    [SerializeField]int numOfParticles = 1;
    [SerializeField]float radius = 0.5f;
    [SerializeField]float smoothingRadius = 0.5f;
    [SerializeField]float spacing = 0f;
    [SerializeField]float gravity = -9.8f;
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
    public bool useSpatialHashing;
    public bool simulate;
    public bool drawGizmos;
    public bool fixedTimeStep;
    public bool diplayPredictedPositions;
    [Range(0f,1f)]public float timeScale;

    [Space(10)]
    [SerializeField]float mouseInteractionRadius = 1f;
    [SerializeField]float mouseInteractionStrength = 5f;

    Vector2[] positions;
    Vector2[] predictedPositions;
    Vector2[] velocities;
    float[] densities;
    float[] nearDensities;
    RenderParams rp;

    const float PI = Mathf.PI;
    // const float mass = 1;

    ComputeBuffer velocityBuffer;
    ComputeBuffer positionBuffer;
    ComputeBuffer densityBuffer;
    Vector3 boundsSize;

    static float smoothingKernelVolume;
    static float spikyKernelPow2Volume;
    static float spikyKernelPow2DerivativeVolume;
    static float spikyKernelPow3Volume;
    static float spikyKernelPow3DerivativeVolume;

    void Start()
    {
        positions = new Vector2[numOfParticles];
        predictedPositions = new Vector2[numOfParticles];
        velocities = new Vector2[numOfParticles];
        densities = new float[numOfParticles];
        nearDensities = new float[numOfParticles];
        spatialLookup = new Entry[numOfParticles];
        startIndices = new uint[numOfParticles];

        int particlePerRow = (int)Mathf.Sqrt(numOfParticles);
        int particlePerCol = Mathf.CeilToInt(numOfParticles/particlePerRow);
        float spawngap = radius*2 + spacing;

        float x=0,y=0;

        velocityBuffer = new ComputeBuffer(numOfParticles,sizeof(float)*2);
        positionBuffer = new ComputeBuffer(numOfParticles,sizeof(float)*2);
        densityBuffer = new ComputeBuffer(numOfParticles,sizeof(float));
        
        smoothingKernelVolume = PI * Mathf.Pow(smoothingRadius, 8) / 4f;
        spikyKernelPow2Volume = PI * Mathf.Pow(smoothingRadius,4) / 6f;
        spikyKernelPow2DerivativeVolume = PI * Mathf.Pow(smoothingRadius,4)/12f;
        spikyKernelPow3Volume = PI * Mathf.Pow(smoothingRadius, 5) / 10f;
        spikyKernelPow3DerivativeVolume =  PI * Mathf.Pow(smoothingRadius, 5)/30f;

        for(int i = 0; i < numOfParticles;i++)
        {
            x = (i%particlePerRow - (particlePerRow-1)/2f)*spawngap;
            y = (i/particlePerRow - (particlePerCol-1)/2f)*spawngap;
            positions[i] = predictedPositions[i] = new Vector2(x,y);
        }

        rp = new RenderParams()
        {
            material = particleMat,
            receiveShadows = false,
            worldBounds = col.bounds
        };

        GenerateColorMap();
    }

    void OnGUI()
    {
        Vector2 mouseWorldPos = MouseToWorldPos();
        (int x,int y) = PositionToCellCoord(mouseWorldPos,smoothingRadius);

        GUI.Label(new Rect(10, 10, 100, 20), CalculateDensity(mouseWorldPos).ToString());
        GUI.Label(new Rect(10, 30, 100, 20), GetHashKey(GetCellHash(x,y)).ToString());
        GUI.Label(new Rect(10, 50, 100, 20), CalculateViscosityForce(0).ToString());
        
    }

    void FixedUpdate()
    {
        if(fixedTimeStep)SimulationStep(Time.fixedDeltaTime*timeScale);
    }

    private void Update()
    {
        if(!fixedTimeStep && Time.frameCount>10)SimulationStep(Time.deltaTime*timeScale);
        DrawParticles();
    }

    void SimulationStep(float deltaTime)
    {
        if(!simulate)return;
        boundsSize = col.bounds.size;
        Parallel.For(0,numOfParticles,(i)=>
        {
            // Apply Gravity Force
            velocities[i] += deltaTime * gravity * Vector2.up;
            // Calculate predicted positions
            predictedPositions[i] = positions[i] + 1/120f * velocities[i];
        });

        if(useSpatialHashing)
            UpdateSpatialLookup();

        Parallel.For(0,numOfParticles,(i)=>
        {
            // Calculate densities based on positions
            (densities[i],nearDensities[i]) = CalculateDensity(predictedPositions[i]);
        });

        Parallel.For(0,numOfParticles,(i)=>
        {
            // Apply Pressure Forces
            Vector2 pressureForce = CalculatePressureForce(i);
            Vector2 pressureAcceleration = pressureForce/densities[i]; // a = f/m , but here each particle is considered tiny units volume, and density[i], gives the mass in the smoothing radius or the unity volume, thus m = densities[i] 
            velocities[i] += deltaTime * pressureAcceleration;
        });

        Parallel.For(0,numOfParticles,(i)=>
        {
            // Apply Pressure Forces
            Vector2 viscosityForce = CalculateViscosityForce(i);
            velocities[i] += deltaTime * viscosityForce;
        });

        if(Input.GetMouseButton(0) || Input.GetMouseButton(1))
        {
            Vector2 samplePos = MouseToWorldPos();
            float strength = Input.GetMouseButton(0)?mouseInteractionStrength:-mouseInteractionStrength;

            Parallel.For(0,numOfParticles,(i)=>
            {
                Vector2 force = InteractionForce(samplePos,mouseInteractionRadius,strength,i);
                Vector2 acceleration = force; // a = f/m , but here each particle is considered tiny units volume, and density[i], gives the mass in the smoothing radius or the unity volume, thus m = densities[i] 
                velocities[i] += deltaTime * acceleration;
            });
        }

        Parallel.For(0,numOfParticles,(i)=>
        {
            // Apply new positions based on new velocity
            positions[i] += deltaTime * velocities[i];
            // Resolve Collisions
            ResolveCollisions(ref positions[i],ref velocities[i]);
        });
    }

    void DrawParticles()
    {
        Quaternion rot = Quaternion.identity;
        Vector3 scaleVec = 2 * radius * Vector3.one;

        velocityBuffer.SetData(velocities);
        positionBuffer.SetData(diplayPredictedPositions?predictedPositions:positions);
        densityBuffer.SetData(densities);
        particleMat.SetFloat("maxvelocity",maxVisualVelocity);
        particleMat.SetFloat("maxdensity",maxVisualDensity);
        particleMat.SetFloat("radius",radius*2);
        particleMat.SetBuffer("velocity",velocityBuffer);
        // particleMat.SetBuffer("density",densityBuffer);
        particleMat.SetBuffer("position",positionBuffer);
        Graphics.DrawMeshInstancedProcedural(particleMesh,0,particleMat,col.bounds,numOfParticles,null,UnityEngine.Rendering.ShadowCastingMode.Off,false);
    }

    void ResolveCollisions(ref Vector2 position, ref Vector2 velocity)
    {
        Vector2 halfBounddSize = boundsSize/2 - radius*Vector3.one;

        if (Mathf.Abs(position.x) > halfBounddSize.x)
        {
            position.x = halfBounddSize.x * Mathf.Sign(position.x);
            velocity.x *= -collisionDamping;
        }
        if (Mathf.Abs(position.y) > halfBounddSize.y)
        {
            position.y = halfBounddSize.y * Mathf.Sign(position.y);
            velocity.y *= -collisionDamping;
        }       
    }

    Vector2 InteractionForce(Vector2 inputPos,float interactionRadius,float strength,int particleIdx)
    {
        Vector2 force = Vector2.zero;

        float sqrdist = (predictedPositions[particleIdx] - inputPos).sqrMagnitude;
        if (sqrdist>interactionRadius*interactionRadius)
            return force;

        float dist = Mathf.Sqrt(sqrdist);
        Vector2 dir = (predictedPositions[particleIdx] - inputPos).normalized;
        float t = 1 - dist/interactionRadius;
        force += t*(strength*dir - velocities[particleIdx]);

        return force;
    }

    static float SmoothKernel(float dist, float radius)
    {
        if(dist>=radius)return 0;
        float val = radius * radius - dist * dist;
        return val*val*val/smoothingKernelVolume;
    }

    static float SpikyKernelPow2(float radius,float dist)
    {
        // CAN BE OPTIMISED
        if (dist>=radius) return 0;
        return (radius - dist) * (radius - dist) / spikyKernelPow2Volume;
    }

    static float SpikyKernelPow2Derivative(float radius,float dist)
    {
        if (dist>=radius) return 0;
        return (dist-radius)/spikyKernelPow2DerivativeVolume;    
    }

    static float SpikyKernelPow3(float radius,float dist)
    {
        // CAN BE OPTIMISED
        if (dist>=radius) return 0;
        float v = radius - dist;
        return v * v * v / spikyKernelPow3Volume;
    }

    static float SpikyKernelPow3Derivative(float radius,float dist)
    {
        if (dist>=radius) return 0;
        float v = radius - dist;
        return -v * v / spikyKernelPow3DerivativeVolume;    
    }

    (float,float) CalculateDensity(Vector2 samplePoint)
    {
        float density = 0, neardensity = 0;//0.0001f;
        float dist;
        if (useSpatialHashing)
        {
            foreach(uint particleidx in ParticleWithinRadius(samplePoint))
            {
                dist = Vector2.Distance(predictedPositions[particleidx],samplePoint);
                density += SpikyKernelPow2(smoothingRadius,dist);
                neardensity += SpikyKernelPow3(smoothingRadius,dist);
            }
        }
        else
        {
            foreach(var pos in predictedPositions)
            {
                dist = Vector2.Distance(pos,samplePoint);
                density += SpikyKernelPow2(smoothingRadius,dist);
                neardensity += SpikyKernelPow3(smoothingRadius,dist);
            }
        }

        return (density,neardensity);
    }

    Vector2 CalculateViscosityForce(int particleIdx)
    {
        Vector2 viscocityForce = Vector2.zero;
        Vector2 position = predictedPositions[particleIdx];

        foreach(int idx in ParticleWithinRadius(position))
        {
            if(particleIdx==idx)continue;
            float dst = Vector2.Distance(position,predictedPositions[idx]);
            float influence = SmoothKernel(dst,smoothingRadius);
            viscocityForce += (velocities[idx] - velocities[particleIdx]) * influence;
        }

        return viscocityForce*viscosityStrength;
    }

    Vector2 CalculatePressureForce(int particleIdx)
    {
        // Any property A[x,y] = Sum of influence of all the properties of particles at that point.
        // A[x,y] = Sum (i->[0 to numofparticles]){ A[i] * mass[i] * density(i) * smoothingfunction( dist((x,y),position[i])) ) }
        Vector2 pressureForce = Vector2.zero;

        if (useSpatialHashing)
        {
            foreach(uint otherparticleidx in ParticleWithinRadius(predictedPositions[particleIdx]))
            {
                if(otherparticleidx == particleIdx)continue;

                Vector2 offset = predictedPositions[particleIdx] - predictedPositions[otherparticleidx];
                float dist = offset.magnitude;
                Vector2 dir = dist == 0? Vector2.right : offset/dist;
                float density = densities[otherparticleidx];
                float nearDensity = nearDensities[otherparticleidx];
                float sharedPressure = CalculateSharedPressure(density,densities[particleIdx]);
                float sharedNearPressure = CalculateSharedNearPressure(density,densities[particleIdx]);
                pressureForce += -sharedPressure * (SpikyKernelPow2Derivative(smoothingRadius,dist)/density) * dir; // * mass
                pressureForce += -sharedNearPressure * (SpikyKernelPow3Derivative(smoothingRadius,dist)/nearDensity) * dir;
            }
        }
        else
        {
            for(int otherparticleidx = 0; otherparticleidx<numOfParticles ; otherparticleidx++)
            {
                if(otherparticleidx == particleIdx)continue;

                Vector2 offset = predictedPositions[particleIdx] - predictedPositions[otherparticleidx];
                float dist = offset.magnitude;
                Vector2 dir = dist == 0? Vector2.right : offset/dist;
                float density = densities[otherparticleidx];
                float nearDensity = nearDensities[otherparticleidx];
                float sharedPressure = CalculateSharedPressure(density,densities[particleIdx]);
                float sharedNearPressure = CalculateSharedNearPressure(nearDensity,densities[particleIdx]);
                pressureForce += -sharedPressure * (SpikyKernelPow2Derivative(smoothingRadius,dist)/density) * dir; // * mass
                pressureForce += -sharedNearPressure * (SpikyKernelPow3Derivative(smoothingRadius,dist)/nearDensity) * dir;
            }
        }

        return pressureForce;
    }

    float ConvertDensityToPressure(float density)
    {
        float densityError = density - targetDensity;
        float pressure = densityError * pressureMultiplier;
        return pressure;
    }

    float ConvertDensityToNearPressure(float nearDensity)
    {
        return nearDensity * nearPressureMultiplier;
    }

    float CalculateSharedPressure(float densityA, float densityB)
    {
        float pressureA = ConvertDensityToPressure(densityA);
        float pressureB = ConvertDensityToPressure(densityB);
        return (pressureA + pressureB) / 2;
    }

    float CalculateSharedNearPressure(float densityA, float densityB)
    {
        float pressureA = ConvertDensityToNearPressure(densityA);
        float pressureB = ConvertDensityToNearPressure(densityB);
        return (pressureA + pressureB) / 2;
    }

    [System.Serializable]
    struct Entry
    {
        public uint idx;
        public uint cellKey;
    }

    Entry[] spatialLookup;
    uint[] startIndices;

    void UpdateSpatialLookup()
    {
        Parallel.For(0,numOfParticles,i=>
        {
            (int cellX,int cellY) = PositionToCellCoord(predictedPositions[i],smoothingRadius);
            uint hash = GetCellHash(cellX,cellY);
            uint key = GetHashKey(hash);
            spatialLookup[i].idx = (uint)i;
            spatialLookup[i].cellKey = key;
            startIndices[i] = uint.MaxValue;
        });

        Array.Sort(spatialLookup,(entry1,entry2)=>{return (int)entry1.cellKey-(int)entry2.cellKey;});

        Parallel.For(0,numOfParticles,i=>
        {
            uint key = spatialLookup[i].cellKey;
            uint prevkey = i==0 ? uint.MaxValue : spatialLookup[i-1].cellKey;

            if(key!=prevkey)
            {
                startIndices[key] = (uint)i;
            }
        });
    }

    uint GetCellHash(int cellX,int cellY)
    {
        uint a = (uint)cellX * 15823;
        uint b = (uint)cellY * 9737333;
        return a + b;
    }

    uint GetHashKey(uint hash)
    {
        return (uint)(hash%numOfParticles);
    }

    (int x,int y) PositionToCellCoord(Vector2 pos,float radius)
    {
        int cellX = (int)(pos.x/radius);
        int cellY = (int)(pos.y/radius);
        return (cellX,cellY);
    }

    IEnumerable<uint> ParticleWithinRadius(Vector2 samplePoint)
    {
        (int cellX,int cellY) = PositionToCellCoord(samplePoint,smoothingRadius);
        float sqrradius = smoothingRadius * smoothingRadius;
        for(int x=-1;x<=1;x++)
        {
            for(int y=-1;y<=1;y++)
            {
                uint key = GetHashKey(GetCellHash(cellX+x,cellY+y));
                uint cellstartIdx = startIndices[key];
                while(cellstartIdx<numOfParticles && spatialLookup[cellstartIdx].cellKey==key)
                {
                    uint particleIdx = spatialLookup[cellstartIdx].idx;
                    float sqrdist = Vector2.SqrMagnitude(predictedPositions[particleIdx]-samplePoint);

                    if(sqrdist<=sqrradius)
                        yield return particleIdx;

                    cellstartIdx++;
                }
            }
        }
    }

    IEnumerable<uint> ParticleWithinAdjacentCells(Vector2 samplePoint)
    {
        (int cellX,int cellY) = PositionToCellCoord(samplePoint,smoothingRadius);
        for(int x=-1;x<=1;x++)
        {
            for(int y=-1;y<=1;y++)
            {
                uint key = GetHashKey(GetCellHash(cellX+x,cellY+y));
                uint cellstartIdx = startIndices[key];
                while(cellstartIdx<numOfParticles && spatialLookup[cellstartIdx].cellKey==key)
                {
                    uint particleIdx = spatialLookup[cellstartIdx].idx;
                    yield return particleIdx;
                    cellstartIdx++;
                }
            }
        }
    }

    void GenerateColorMap()
    {
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
        return Camera.main.ScreenToWorldPoint(new Vector3(Input.mousePosition.x,Input.mousePosition.y,-Camera.main.transform.position.z));
    }

    private void OnValidate()
    {
        if(numOfParticles<0)numOfParticles = 0;
        GenerateColorMap();
        smoothingKernelVolume = PI * Mathf.Pow(smoothingRadius, 8) / 4f;
        spikyKernelPow2Volume = PI * Mathf.Pow(smoothingRadius,4) / 6f;
        spikyKernelPow2DerivativeVolume = PI * Mathf.Pow(smoothingRadius,4)/12f;
        spikyKernelPow3Volume = PI * Mathf.Pow(smoothingRadius, 5) / 10f;
        spikyKernelPow3DerivativeVolume =  PI * Mathf.Pow(smoothingRadius, 5)/30f;
    }

    private void OnDrawGizmos()
    {
        if(drawGizmos && positions!=null && positions.Length>0)
        {
            Gizmos.color = Color.red;
            Gizmos.DrawWireSphere(positions[0],smoothingRadius);
            Gizmos.color = Color.white;
            if(useSpatialHashing)
                foreach(uint idx in ParticleWithinAdjacentCells(positions[0]))
                {
                    Gizmos.DrawWireSphere(positions[idx],radius);
                }
            
            if(Input.GetMouseButton(0) || Input.GetMouseButton(1))
            {
                Gizmos.color = Color.blue;
                Vector3 samplePos = MouseToWorldPos();
                Gizmos.DrawWireSphere(samplePos,mouseInteractionRadius);
            }
        }
    }

    private void OnDestroy()
    {
        velocityBuffer.Release();
        positionBuffer.Release();
        densityBuffer.Release();
    }
}