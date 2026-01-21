using UnityEngine;
using System.Collections.Generic;
using System.IO;
using System.Globalization;
using System.Runtime.InteropServices;

public class GPURenderer : MonoBehaviour
{
    [Header("Rendering Resources")]
    public Mesh cellMesh;
    public Material cellMaterial;
    public Gradient colorGradient;
    
    [Header("Scaling Parameters")]
    public float positionScale = 0.5f;
    public float heightMultiplier = 1.0f;
    public float baseScale = 5.0f;
    public float emissionIntensity = 2.0f;
    
    [Header("Color System")]
    public int typeColorCount = 45;
    public Color[] typeColors;
    public float saturation = 0.8f;
    public float brightness = 0.9f;

    public struct CellDataGPU
    {
        public Vector3 position;
        public float scale;
        public Vector4 color;
        public float expression;

        public int cellIdHash;
    }

    // GPU Buffers
    private ComputeBuffer positionBuffer;
    private ComputeBuffer scaleBuffer;
    private ComputeBuffer colorBuffer;
    private ComputeBuffer expressionBuffer;
    private ComputeBuffer argsBuffer;
    
    private uint[] args = new uint[5] { 0, 0, 0, 0, 0 };
    private int cellCount = 0;
    private Bounds bounds;

    private List<CellDataGPU> cellDataList = new List<CellDataGPU>();
    private Dictionary<string, int> cellIdToIndexMap = new Dictionary<string, int>();
    private List<string> cellIdList = new List<string>(); 

    public enum ViewMode
    {
        Expression,
        CellType,
        AI_Annotation,
        TissueRegion,
        ZeroShot
    }
    
    public ViewMode currentViewMode = ViewMode.Expression;
    
    void Awake()
    {
        GenerateTypeColors();
        bounds = new Bounds(Vector3.zero, new Vector3(10000, 10000, 10000));
    }

    void Update()
    {
        if (cellCount > 0 && cellMaterial != null && cellMesh != null)
        {
            Graphics.DrawMeshInstancedIndirect(
                cellMesh,
                0,
                cellMaterial,
                bounds,
                argsBuffer
            );
        }
    }

    void OnDisable()
    {
        ReleaseBuffers();
    }

    void OnDestroy()
    {
        ReleaseBuffers();
    }

    void ReleaseBuffers()
    {
        if (positionBuffer != null) positionBuffer.Release();
        if (scaleBuffer != null) scaleBuffer.Release();
        if (colorBuffer != null) colorBuffer.Release();
        if (expressionBuffer != null) expressionBuffer.Release();
        if (argsBuffer != null) argsBuffer.Release();
        
        positionBuffer = null;
        scaleBuffer = null;
        colorBuffer = null;
        expressionBuffer = null;
        argsBuffer = null;
    }

    public void GenerateTypeColors()
    {
        typeColors = new Color[typeColorCount];
        for (int i = 0; i < typeColorCount; i++)
        {
            float hue = (float)i / typeColorCount;
            typeColors[i] = Color.HSVToRGB(hue, saturation, brightness);
        }
        Debug.Log($"[GPU Renderer] Generated {typeColorCount} type colors");
    }

    public void InitializeData(List<CellDataGPU> data, Dictionary<string, int> idToIndexMap, List<string> idList)
    {
        this.cellDataList = new List<CellDataGPU>(data);
        this.cellCount = data.Count;
        this.cellIdToIndexMap = new Dictionary<string, int>(idToIndexMap);
        this.cellIdList = new List<string>(idList);
        
        InitializeBuffers();
        UpdateAllBuffers();
        
        Debug.Log($"[GPU Renderer] Initialized {cellCount} cells");
    }

    void InitializeBuffers()
    {
        ReleaseBuffers();
        
        if (cellCount == 0) return;

        positionBuffer = new ComputeBuffer(cellCount, sizeof(float) * 3);
        scaleBuffer = new ComputeBuffer(cellCount, sizeof(float));
        colorBuffer = new ComputeBuffer(cellCount, sizeof(float) * 4);
        expressionBuffer = new ComputeBuffer(cellCount, sizeof(float));

        argsBuffer = new ComputeBuffer(1, args.Length * sizeof(uint), ComputeBufferType.IndirectArguments);
        
        args[0] = (uint)cellMesh.GetIndexCount(0);
        args[1] = (uint)cellCount;
        args[2] = (uint)cellMesh.GetIndexStart(0);
        args[3] = (uint)cellMesh.GetBaseVertex(0);
        args[4] = 0;
        argsBuffer.SetData(args);
        cellMaterial.SetBuffer("_CellPositions", positionBuffer);
        cellMaterial.SetBuffer("_CellScales", scaleBuffer);
        cellMaterial.SetBuffer("_CellColors", colorBuffer);
        cellMaterial.SetBuffer("_CellExpressions", expressionBuffer);
        cellMaterial.SetFloat("_GlobalScale", baseScale);
        cellMaterial.SetFloat("_EmissionStrength", emissionIntensity);
    }

    void UpdateAllBuffers()
    {
        if (cellCount == 0) return;
        
        Vector3[] positions = new Vector3[cellCount];
        float[] scales = new float[cellCount];
        Vector4[] colors = new Vector4[cellCount];
        float[] expressions = new float[cellCount];
        
        for (int i = 0; i < cellCount; i++)
        {
            positions[i] = cellDataList[i].position;
            scales[i] = cellDataList[i].scale;
            colors[i] = cellDataList[i].color;
            expressions[i] = cellDataList[i].expression;
        }
        
        positionBuffer.SetData(positions);
        scaleBuffer.SetData(scales);
        colorBuffer.SetData(colors);
        expressionBuffer.SetData(expressions);
    }

    public void UpdateCellVisual(string cellId, Vector3 position, Color color, float scale, float expression)
    {
        if (!cellIdToIndexMap.ContainsKey(cellId)) return;
        
        int index = cellIdToIndexMap[cellId];
        CellDataGPU cell = cellDataList[index];
        
        cell.position = position;
        cell.color = new Vector4(color.r, color.g, color.b, color.a);
        cell.scale = scale;
        cell.expression = expression;
        
        cellDataList[index] = cell;
        
        positionBuffer.SetData(new Vector3[] { position }, 0, index, 1);
        scaleBuffer.SetData(new float[] { scale }, 0, index, 1);
        colorBuffer.SetData(new Vector4[] { new Vector4(color.r, color.g, color.b, color.a) }, 0, index, 1);
        expressionBuffer.SetData(new float[] { expression }, 0, index, 1);
    }

    public void UpdateCellsVisual(List<string> cellIds, List<Vector3> positions, List<Color> colors, List<float> scales, List<float> expressions)
    {
        for (int i = 0; i < cellIds.Count && i < cellDataList.Count; i++)
        {
            if (cellIdToIndexMap.ContainsKey(cellIds[i]))
            {
                int index = cellIdToIndexMap[cellIds[i]];
                CellDataGPU cell = cellDataList[index];
                
                if (positions != null && i < positions.Count)
                    cell.position = positions[i];
                if (colors != null && i < colors.Count)
                    cell.color = new Vector4(colors[i].r, colors[i].g, colors[i].b, colors[i].a);
                if (scales != null && i < scales.Count)
                    cell.scale = scales[i];
                if (expressions != null && i < expressions.Count)
                    cell.expression = expressions[i];
                
                cellDataList[index] = cell;
            }
        }
        
        UpdateAllBuffers();
    }

    public void RefreshAllCells(ViewMode mode, Dictionary<string, float> expressionMap, 
        Dictionary<string, int> typeMap, Dictionary<string, int> aiPredictionMap,
        Dictionary<string, int> zeroShotClusterMap, Dictionary<int, Color> zeroShotColorMap,
        int highlightedTypeID,
        int highlightedRegionID, 
        Dictionary<string, int> regionMap)
    {
        currentViewMode = mode;
        
        for (int i = 0; i < cellDataList.Count; i++)
        {
            string cellId = cellIdList[i];
            CellDataGPU cell = cellDataList[i];
            
            float targetValue = 0f;
            Color baseColor = Color.white;
            float scale = 0.5f;
            
            switch (mode)
            {
                case ViewMode.Expression:
                    if (expressionMap != null && expressionMap.ContainsKey(cellId))
                    {
                        float expr = expressionMap[cellId];
                        targetValue = expr;
                        baseColor = colorGradient.Evaluate(expr);
                        scale = 0.5f + expr;
                    }
                    break;
                    
                case ViewMode.CellType:
                    targetValue = 1.0f;
                    if (typeMap != null && typeMap.ContainsKey(cellId))
                    {
                        int typeId = typeMap[cellId];
                        int safeId = Mathf.Clamp(typeId, 0, typeColors.Length - 1);
                        baseColor = typeColors[safeId];
                    }
                    scale = 0.5f;
                    break;
                    
                case ViewMode.AI_Annotation:
                    targetValue = 0.5f;
                    if (aiPredictionMap != null && aiPredictionMap.ContainsKey(cellId))
                    {
                        int predId = aiPredictionMap[cellId];
                        if (highlightedTypeID == -1 || predId == highlightedTypeID)
                        {
                            int safeId = Mathf.Clamp(predId, 0, typeColors.Length - 1);
                            baseColor = typeColors[safeId];
                            scale = 0.8f;
                        }
                        else
                        {
                            scale = 0.0f;
                        }
                    }
                    break;
                    
                case ViewMode.ZeroShot:
                    targetValue = 0.5f;
                    scale = 0.7f;
                    if (zeroShotClusterMap != null && zeroShotClusterMap.ContainsKey(cellId))
                    {
                        int cId = zeroShotClusterMap[cellId];
                        if (zeroShotColorMap != null && zeroShotColorMap.ContainsKey(cId))
                        {
                            baseColor = zeroShotColorMap[cId];
                        }
                        else
                        {
                            int safeId = Mathf.Clamp(cId, 0, typeColors.Length - 1);
                            baseColor = typeColors[safeId];
                        }
                    }
                    else
                    {
                        baseColor = Color.gray;
                        scale = 0.3f;
                    }
                    break;

                case ViewMode.TissueRegion:
                    targetValue = 0.5f;
                    scale = 0.5f; 

                    if (regionMap != null && regionMap.ContainsKey(cellId))
                    {
                        int regionId = regionMap[cellId];

                        if (highlightedRegionID == -1 || regionId == highlightedRegionID)
                        {
                            int safeId = Mathf.Clamp(regionId, 0, typeColors.Length - 1);
                            baseColor = typeColors[safeId];
                            scale = (highlightedRegionID != -1) ? 0.8f : 0.5f;
                        }
                        else
                        {
                            scale = 0.0f;
                        }
                    }
                    else
                    {
                        scale = 0.0f;
                    }
                    break;
            }
            Vector3 newPos = new Vector3(cell.position.x, targetValue * heightMultiplier, cell.position.z);
            
            cell.position = newPos;
            cell.color = new Vector4(baseColor.r, baseColor.g, baseColor.b, 1.0f);
            cell.scale = scale;
            
            if (expressionMap != null && expressionMap.ContainsKey(cellId))
            {
                cell.expression = expressionMap[cellId];
            }
            
            cellDataList[i] = cell;
        }
        
        UpdateAllBuffers();
    }
    public bool TryGetCellIndex(string cellId, out int index)
    {
        return cellIdToIndexMap.TryGetValue(cellId, out index);
    }

    public bool TryGetCellData(int index, out CellDataGPU cellData)
    {
        if (index >= 0 && index < cellDataList.Count)
        {
            cellData = cellDataList[index];
            return true;
        }
        cellData = default(CellDataGPU);
        return false;
    }

    public string GetCellId(int index)
    {
        if (index >= 0 && index < cellIdList.Count)
            return cellIdList[index];
        return null;
    }

    public void UpdateColorsForRegions(List<int> regionIds, Color[] palette)
    {
        if (regionIds == null || cellDataList == null || cellDataList.Count == 0) return;
        
        for (int i = 0; i < cellDataList.Count && i < regionIds.Count; i++)
        {
            int rId = regionIds[i];
            Color c = palette[rId % palette.Length];
            
            CellDataGPU cell = cellDataList[i];
            cell.color = new Vector4(c.r, c.g, c.b, 1.0f);
            cellDataList[i] = cell;
        }
        
        UpdateAllBuffers();
    }

    public void SetMaterialParameters(float globalScale, float emission)
    {
        if (cellMaterial != null)
        {
            cellMaterial.SetFloat("_GlobalScale", globalScale);
            cellMaterial.SetFloat("_EmissionStrength", emission);
            baseScale = globalScale;
            emissionIntensity = emission;
        }
    }
}
