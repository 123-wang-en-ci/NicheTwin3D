Shader "Custom/SurfaceExpressionShader"
{
    Properties
    {
        _EmissionStrength ("Emission Strength", Range(0, 10)) = 2.0
        _Glossiness ("Smoothness", Range(0, 1)) = 0.8
        _PosOffset ("Position Offset", Vector) = (0,0,0,0)
    }

    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Geometry" "RenderPipeline" = "UniversalPipeline" }
        LOD 200

        Pass
        {
            Name "ForwardLit"
            Tags { "LightMode" = "UniversalForward" }
            
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"

            // Compute Buffers from IDW Interpolation
            StructuredBuffer<float> _GridHeights;
            StructuredBuffer<float4> _GridColors;
            
            float _HeightMultiplier;
            int _GridResolution;

            CBUFFER_START(UnityPerMaterial)
                float _EmissionStrength;
                float _Glossiness;
                float3 _PosOffset;
            CBUFFER_END

            struct Attributes
            {
                float3 positionOS : POSITION;
                uint vertexID : SV_VertexID;
                uint instanceID : SV_InstanceID;
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float3 positionWS : TEXCOORD0;
                float3 normalWS   : TEXCOORD1;
                float4 color      : COLOR;
            };

            Varyings vert(Attributes input)
            {
                Varyings output;
                
                uint id = input.vertexID;
                uint inst = input.instanceID;
                
                int flatIndex = inst * (_GridResolution * _GridResolution) + id;
                
                // Fetch evaluated height and color for this grid point
                float expr = _GridHeights[flatIndex];
                float4 cellColor = _GridColors[flatIndex];
                
                float3 worldPos = input.positionOS;
                worldPos.y = (expr * _HeightMultiplier) + 0.1; // Extrude up
                
                // Calculate smooth normals using central differences
                int x = id % _GridResolution;
                int y = id / _GridResolution;
                
                int xL = max(x - 1, 0);
                int xR = min(x + 1, _GridResolution - 1);
                int yD = max(y - 1, 0);
                int yU = min(y + 1, _GridResolution - 1);
                
                int offset = inst * (_GridResolution * _GridResolution);
                
                float hL = _GridHeights[offset + y * _GridResolution + xL] * _HeightMultiplier;
                float hR = _GridHeights[offset + y * _GridResolution + xR] * _HeightMultiplier;
                float hD = _GridHeights[offset + yD * _GridResolution + x] * _HeightMultiplier;
                float hU = _GridHeights[offset + yU * _GridResolution + x] * _HeightMultiplier;
                
                // The grid spacing in world space (approximate using normalized scale for normals)
                float stepSize = 1.0; 
                float3 dX = float3(stepSize * 2.0, hR - hL, 0);
                float3 dZ = float3(0, hU - hD, stepSize * 2.0);
                
                float3 normalOS = normalize(cross(dZ, dX));
                
                output.positionCS = TransformWorldToHClip(worldPos + _PosOffset);
                output.positionWS = worldPos + _PosOffset;
                output.normalWS = TransformObjectToWorldNormal(normalOS);
                output.color = cellColor;
                
                return output;
            }

            half4 frag(Varyings input) : SV_Target
            {
                // Core feature: clip transparent empty spaces linking different cell types!
                clip(input.color.a - 0.5);
                
                // Simple lighting model for a modern glossy look
                Light mainLight = GetMainLight();
                float3 normalWS = normalize(input.normalWS);
                
                // Diffuse
                float NdotL = saturate(dot(normalWS, mainLight.direction));
                float3 diffuse = mainLight.color * NdotL * 0.8;
                
                // Ambient (soft rim or sky light)
                float3 ambient = float3(0.2, 0.25, 0.35) + (normalWS.y * 0.1); 
                
                // Specular Highlights (Plastic/Glossy feel)
                float3 viewDir = normalize(_WorldSpaceCameraPos - input.positionWS);
                float3 halfVector = normalize(mainLight.direction + viewDir);
                float NdotH = saturate(dot(normalWS, halfVector));
                float specular = pow(NdotH, exp2(_Glossiness * 10.0 + 1.0)) * 0.5;
                
                // Final composition
                float4 col = input.color;
                
                // Mix the base heat color with diffuse, add specular, and a bit of emissive glow
                float3 finalRGB = col.rgb * (diffuse + ambient) + specular * mainLight.color;
                finalRGB += col.rgb * _EmissionStrength * 0.2; // Subtle self-illumination
                
                return half4(finalRGB, 1.0);
            }
            ENDHLSL
        }
    }
    FallBack "Hidden/Universal Render Pipeline/FallbackError"
}
