Shader "Custom/InstancedPointShader"
{
    Properties
    {
        _BaseColor ("Color", Color) = (1,1,1,1)
        _EmissionStrength ("Emission Strength", Range(0, 10)) = 2.0
        _GlobalScale ("Global Scale", Float) = 1.0
    }

    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Geometry" "RenderPipeline" = "UniversalPipeline" }
        LOD 100

        Pass
        {
            Name "ForwardLit"
            Tags { "LightMode" = "UniversalForward" }
            
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_instancing
            
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            StructuredBuffer<float3> _CellPositions;
            StructuredBuffer<float> _CellScales;
            StructuredBuffer<float4> _CellColors;
            StructuredBuffer<float> _CellExpressions;

            CBUFFER_START(UnityPerMaterial)
                float4 _BaseColor;
                float _EmissionStrength;
                float _GlobalScale;
            CBUFFER_END

            struct Attributes
            {
                float4 positionOS : POSITION;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float4 color : TEXCOORD0;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            Varyings vert(Attributes input, uint instanceID : SV_InstanceID)
            {
                Varyings output;
                
                UNITY_SETUP_INSTANCE_ID(input);
                UNITY_TRANSFER_INSTANCE_ID(input, output);

                float3 worldPos = _CellPositions[instanceID];
                float scale = _CellScales[instanceID] * _GlobalScale;
                float4 cellColor = _CellColors[instanceID];

                float3 scaledPosition = input.positionOS.xyz * scale;
                float3 worldPosition = worldPos + scaledPosition;
                
                output.positionCS = TransformWorldToHClip(worldPosition);
                output.color = cellColor;
                
                return output;
            }

            half4 frag(Varyings input) : SV_Target
            {
                UNITY_SETUP_INSTANCE_ID(input);
                
                float4 col = input.color;
                col.rgb *= _EmissionStrength;
                col.a = 1.0;
                
                return col;
            }
            ENDHLSL
        }
    }
    FallBack "Hidden/Universal Render Pipeline/FallbackError"
}
