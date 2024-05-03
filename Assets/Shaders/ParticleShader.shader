Shader "ParticleShader"
{
    Properties
    {
        [NoScaleOffset]_MainTex ("Texture", 2D) = "white" {}
    }
    
    SubShader
    {
        Tags { "RenderType"="Transparent" "Queue"="Transparent" }
        LOD 100
        
        Pass
        {
            Blend SrcAlpha OneMinusSrcAlpha
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_instancing

            #include "UnityCG.cginc"
            
            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };
            
            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
                float4 col : COLOR;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            StructuredBuffer<float2> position;
            StructuredBuffer<float2> velocity;
            StructuredBuffer<float> density;
            float maxvelocity;
            float maxdensity;
            float radius;
            sampler2D _MainTex;
            Texture2D<float4> ColourMap;
            SamplerState linear_clamp_sampler;
            float4 _MainTex_ST;
            
            v2f vert(appdata v,uint id:SV_InstanceID)
            {
                v2f o;
                float4 worldPos = mul(unity_ObjectToWorld,v.vertex*radius);
                worldPos.xy += position[id];
                v.vertex = mul(unity_WorldToObject,worldPos);
                UNITY_SETUP_INSTANCE_ID(v);
                UNITY_TRANSFER_INSTANCE_ID(v, o);
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                // float2 coord = float2 ( saturate((velocity[id].x * velocity[id].x + velocity[id].y * velocity[id].y)/(maxvelocity*maxvelocity)) ,0.5);
                // float2 coord = float2(saturate( density[id] / maxdensity), 0.5);
                float2 coord = float2(saturate( length(velocity[id]/maxvelocity)), 0.5);
                o.col = ColourMap.SampleLevel(linear_clamp_sampler,coord,0);
                return o;
            }
            
            fixed4 frag(v2f i) : SV_Target
            {
                fixed4 col = tex2D(_MainTex, i.uv) * i.col;
                return col;
            }
            ENDCG
        }
    }
}
