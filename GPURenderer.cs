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
    public Material surfaceMaterial; 
    public ComputeShader idwComputeShader; 
    public int gridResolution = 200;      
    public float surfaceSmoothingRadius = 5.0f; 
    public bool showSurfaceMode = false;   // Whether the interpolation surface mode is enabled
    public int maxSurfaceInstances = 45;   // Determine how many independent cell type surfaces can be generated
    public Gradient colorGradient;
    
    [Header("Scaling Parameters")]
    public float positionScale = 0.5f;
    public float heightMultiplier = 1.0f;
    public float baseScale = 5.0f;
    public float emissionIntensity = 1.0f; // Reduce the emission intensity to prevent color deviation caused by HDR tone mapping (red turns pink, green turns yellow)
    public float surfaceEmissionIntensity = 1.0f; // Dedicated brightness for the surface
    
    [Header("Color System")]
    public int typeColorCount = 45;
    public Color[] typeColors;
    [HideInInspector]
    public Color[] regionColors;
    public float saturation = 0.8f;
    public float brightness = 0.9f;

    [Header("Imputation System")]
    public Color imputedCellColor = Color.green; // Interpolate the exclusive fluorescent color of deficient cells
    public bool enableImputedColorOverride = true; // Whether to turn on the cut-off dyeing switch

    [Header("Comparison System")]
    public Camera targetCamera = null;

    // GPU data structure
    public struct CellDataGPU
    {
        public Vector3 position;
        public float scale;
        public Vector4 color;
        public float expression;
        
        // Pass cell true type for surface separation
        public int typeId;
    }

    // GPU Buffers
    private ComputeBuffer positionBuffer;
    private ComputeBuffer scaleBuffer;
    private ComputeBuffer colorBuffer;
    private ComputeBuffer expressionBuffer;
    private ComputeBuffer typeIdBuffer;
    private ComputeBuffer argsBuffer;
    
    private uint[] args = new uint[5] { 0, 0, 0, 0, 0 };
    private int cellCount = 0;
    private Bounds bounds;
    
    // Surface Mesh Data
    private Mesh surfaceTopologyMesh;
    private ComputeBuffer gridHeightsBuffer;
    private ComputeBuffer gridColorsBuffer;
    private float gridMinX, gridMaxX, gridMinZ, gridMaxZ;
    
    // data storage
    private List<CellDataGPU> cellDataList = new List<CellDataGPU>();
    private Dictionary<string, int> cellIdToIndexMap = new Dictionary<string, int>();
    private List<string> cellIdList = new List<string>(); 

    // View mode enumeration
    public enum ViewMode
    {
        Expression,
        CellType,
        AI_Annotation,
        TissueRegion,
        ZeroShot
    }
    
    // View mode related
    public ViewMode currentViewMode = ViewMode.Expression;
    
    void Awake()
    {
        GenerateTypeColors();
        bounds = new Bounds(Vector3.zero, new Vector3(10000, 10000, 10000));
    }

    void OnValidate()
    {
        if (cellMaterial != null)
        {
            cellMaterial.SetFloat("_GlobalScale", baseScale);
            cellMaterial.SetFloat("_EmissionStrength", emissionIntensity);
        }
        if (surfaceMaterial != null)
        {
            surfaceMaterial.SetFloat("_EmissionStrength", surfaceEmissionIntensity);
            surfaceMaterial.SetFloat("_HeightMultiplier", heightMultiplier);
        }
    }

    void Update()
    {
        if (cellCount == 0) return;

        bool canShowSurface = (currentViewMode == ViewMode.Expression || currentViewMode == ViewMode.TissueRegion || currentViewMode == ViewMode.AI_Annotation || currentViewMode == ViewMode.ZeroShot);
        if (canShowSurface && showSurfaceMode && surfaceMaterial != null && surfaceTopologyMesh != null)
        {
            Graphics.DrawMeshInstancedProcedural(
                surfaceTopologyMesh, 
                0, 
                surfaceMaterial, 
                bounds, 
                maxSurfaceInstances, 
                null, 
                UnityEngine.Rendering.ShadowCastingMode.Off, 
                false,
                0,
                targetCamera
            );
        }
        // On the contrary, render the sphere point cloud
        else if (cellMaterial != null && cellMesh != null)
        {
            Graphics.DrawMeshInstancedIndirect(
                cellMesh,
                0,
                cellMaterial,
                bounds,
                argsBuffer,
                0,
                null,
                UnityEngine.Rendering.ShadowCastingMode.On,
                true,
                0,
                targetCamera
            );
        }
    }

    void OnEnable()
    {
        if (cellCount > 0 && positionBuffer == null)
        {
            InitializeBuffers();
            UpdateAllBuffers();
            if (showSurfaceMode)
            {
                ComputeSurfaceInterpolation();
            }
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
        if (typeIdBuffer != null) typeIdBuffer.Release();
        if (argsBuffer != null) argsBuffer.Release();
        
        if (gridHeightsBuffer != null) gridHeightsBuffer.Release();
        if (gridColorsBuffer != null) gridColorsBuffer.Release();
        
        positionBuffer = null;
        scaleBuffer = null;
        colorBuffer = null;
        expressionBuffer = null;
        typeIdBuffer = null;
        argsBuffer = null;
        gridHeightsBuffer = null;
        gridColorsBuffer = null;
    }

    public void GenerateTypeColors()
    {
        typeColors = new Color[typeColorCount];
        regionColors = new Color[typeColorCount];
        
        // Use the golden ratio conjugate to jump significantly around the hue circle
        float goldenRatioConjugate = 0.618033988749895f;
        
        for (int i = 0; i < typeColorCount; i++)
        {
            float hue = (i * goldenRatioConjugate) % 1.0f;
            typeColors[i] = Color.HSVToRGB(hue, saturation, brightness);
            
            float regionHue = ((i * goldenRatioConjugate) + 0.5f) % 1.0f;
            regionColors[i] = Color.HSVToRGB(regionHue, Mathf.Clamp01(saturation + 0.15f), Mathf.Clamp01(brightness - 0.1f));
        }
        Debug.Log($"[GPU Renderer] Generated {typeColorCount} type & region distinct colors using Golden Ratio");
    }

    public void InitializeData(List<CellDataGPU> data, Dictionary<string, int> idToIndexMap, List<string> idList)
    {
        this.cellDataList = new List<CellDataGPU>(data);
        this.cellCount = data.Count;
        this.cellIdToIndexMap = new Dictionary<string, int>(idToIndexMap);
        this.cellIdList = new List<string>(idList);
        
        CalculateBounds();
        InitializeBuffers();
        UpdateAllBuffers();
        BuildSurfaceTopologyMesh();
        
        Debug.Log($"[GPU Renderer] Initialized {cellCount} cells");
    }

    public List<CellDataGPU> GetCellDataList() { return cellDataList; }
    public Dictionary<string, int> GetCellIdToIndexMap() { return cellIdToIndexMap; }
    public List<string> GetCellIdList() { return cellIdList; }
    public int GetCellCount() { return cellCount; }

    public void CopyStateFrom(GPURenderer source)
    {
        if (source == null) return;
        this.cellDataList = new List<CellDataGPU>(source.GetCellDataList());
        this.cellCount = source.GetCellCount();
        this.cellIdToIndexMap = new Dictionary<string, int>(source.GetCellIdToIndexMap());
        this.cellIdList = new List<string>(source.GetCellIdList());
        this.currentViewMode = source.currentViewMode;
        this.showSurfaceMode = source.showSurfaceMode;
        
        CalculateBounds();
        InitializeBuffers();
        UpdateAllBuffers();
        BuildSurfaceTopologyMesh();
        if (showSurfaceMode)
        {
            ComputeSurfaceInterpolation();
        }
    }

    void CalculateBounds()
    {
        if (cellCount == 0) return;
        gridMinX = cellDataList[0].position.x;
        gridMaxX = cellDataList[0].position.x;
        gridMinZ = cellDataList[0].position.z;
        gridMaxZ = cellDataList[0].position.z;
        foreach (var cell in cellDataList)
        {
            if (cell.position.x < gridMinX) gridMinX = cell.position.x;
            if (cell.position.x > gridMaxX) gridMaxX = cell.position.x;
            if (cell.position.z < gridMinZ) gridMinZ = cell.position.z;
            if (cell.position.z > gridMaxZ) gridMaxZ = cell.position.z;
        }
    }

    void BuildSurfaceTopologyMesh()
    {
        surfaceTopologyMesh = new Mesh();
        surfaceTopologyMesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
        
        int numVertices = gridResolution * gridResolution;
        Vector3[] vertices = new Vector3[numVertices];
        Vector2[] uvs = new Vector2[numVertices];
        
        // Build vertices covering the spatial bounds
        for (int y = 0; y < gridResolution; y++)
        {
            float tZ = (float)y / (gridResolution - 1);
            float posZ = Mathf.Lerp(gridMinZ, gridMaxZ, tZ);
            
            for (int x = 0; x < gridResolution; x++)
            {
                float tX = (float)x / (gridResolution - 1);
                float posX = Mathf.Lerp(gridMinX, gridMaxX, tX);
                
                int index = y * gridResolution + x;
                vertices[index] = new Vector3(posX, 0, posZ);
                uvs[index] = new Vector2(tX, tZ);
            }
        }
        
        // Build connected triangles
        int[] triangles = new int[(gridResolution - 1) * (gridResolution - 1) * 6];
        int t = 0;
        for (int y = 0; y < gridResolution - 1; y++)
        {
            for (int x = 0; x < gridResolution - 1; x++)
            {
                int index = y * gridResolution + x;
                triangles[t++] = index;
                triangles[t++] = index + gridResolution;
                triangles[t++] = index + 1;
                
                triangles[t++] = index + 1;
                triangles[t++] = index + gridResolution;
                triangles[t++] = index + gridResolution + 1;
            }
        }
        
        surfaceTopologyMesh.vertices = vertices;
        surfaceTopologyMesh.uv = uvs;
        surfaceTopologyMesh.triangles = triangles;
        surfaceTopologyMesh.RecalculateNormals(); 
        surfaceTopologyMesh.bounds = bounds; // Prevent culling
        
        Debug.Log($"[GPU Renderer] Built Surface Grid Mesh with {numVertices} vertices.");
    }

    void InitializeBuffers()
    {
        ReleaseBuffers();
        
        if (cellCount == 0) return;
        
        // Create individual property buffers
        positionBuffer = new ComputeBuffer(cellCount, sizeof(float) * 3);
        scaleBuffer = new ComputeBuffer(cellCount, sizeof(float));
        colorBuffer = new ComputeBuffer(cellCount, sizeof(float) * 4);
        expressionBuffer = new ComputeBuffer(cellCount, sizeof(float));
        typeIdBuffer = new ComputeBuffer(cellCount, sizeof(int));
        
        gridHeightsBuffer = new ComputeBuffer(gridResolution * gridResolution * maxSurfaceInstances, sizeof(float));
        gridColorsBuffer = new ComputeBuffer(gridResolution * gridResolution * maxSurfaceInstances, sizeof(float) * 4);
 
        argsBuffer = new ComputeBuffer(1, args.Length * sizeof(uint), ComputeBufferType.IndirectArguments);
        
        args[0] = (uint)cellMesh.GetIndexCount(0);
        args[1] = (uint)cellCount;
        args[2] = (uint)cellMesh.GetIndexStart(0);
        args[3] = (uint)cellMesh.GetBaseVertex(0);
        args[4] = 0;
        argsBuffer.SetData(args);

        if (cellMaterial != null)
        {
            cellMaterial.SetBuffer("_CellPositions", positionBuffer);
            cellMaterial.SetBuffer("_CellScales", scaleBuffer);
            cellMaterial.SetBuffer("_CellColors", colorBuffer);
            cellMaterial.SetBuffer("_CellExpressions", expressionBuffer);
            cellMaterial.SetFloat("_GlobalScale", baseScale);
            cellMaterial.SetFloat("_EmissionStrength", emissionIntensity);
        }

        if (surfaceMaterial != null)
        {
            surfaceMaterial.SetBuffer("_GridHeights", gridHeightsBuffer);
            surfaceMaterial.SetBuffer("_GridColors", gridColorsBuffer);
            surfaceMaterial.SetInt("_GridResolution", gridResolution);
            surfaceMaterial.SetFloat("_HeightMultiplier", heightMultiplier);
            surfaceMaterial.SetFloat("_EmissionStrength", surfaceEmissionIntensity);
        }
    }

    public void ComputeSurfaceInterpolation()
    {
        if (idwComputeShader == null || gridHeightsBuffer == null || cellCount == 0) return;
        
        int kernel = idwComputeShader.FindKernel("ComputeGridSurface");
        idwComputeShader.SetBuffer(kernel, "_CellPositions", positionBuffer);
        idwComputeShader.SetBuffer(kernel, "_CellExpressions", expressionBuffer);
        idwComputeShader.SetBuffer(kernel, "_CellColors", colorBuffer);
        idwComputeShader.SetBuffer(kernel, "_CellTypeIds", typeIdBuffer);
        idwComputeShader.SetBuffer(kernel, "_CellScales", scaleBuffer);
        
        idwComputeShader.SetBuffer(kernel, "_GridHeights", gridHeightsBuffer);
        idwComputeShader.SetBuffer(kernel, "_GridColors", gridColorsBuffer);
        
        idwComputeShader.SetInt("_CellCount", cellCount);
        idwComputeShader.SetFloat("_GridMinX", gridMinX);
        idwComputeShader.SetFloat("_GridMaxX", gridMaxX);
        idwComputeShader.SetFloat("_GridMinZ", gridMinZ);
        idwComputeShader.SetFloat("_GridMaxZ", gridMaxZ);
        idwComputeShader.SetInt("_GridResolution", gridResolution);
        idwComputeShader.SetFloat("_SmoothingRadius", surfaceSmoothingRadius);
        idwComputeShader.SetInt("_MaxTypes", maxSurfaceInstances);
        idwComputeShader.SetInt("_ViewMode", (int)currentViewMode);
        

        int threadGroups = Mathf.CeilToInt(gridResolution / 8.0f);
        idwComputeShader.Dispatch(kernel, threadGroups, threadGroups, maxSurfaceInstances);
    }

    void UpdateAllBuffers()
    {
        if (cellCount == 0) return;
        
        Vector3[] positions = new Vector3[cellCount];
        float[] scales = new float[cellCount];
        Vector4[] colors = new Vector4[cellCount];
        float[] expressions = new float[cellCount];
        int[] typeIds = new int[cellCount];
        
        for (int i = 0; i < cellCount; i++)
        {
            positions[i] = cellDataList[i].position;
            scales[i] = cellDataList[i].scale;
            colors[i] = cellDataList[i].color;
            expressions[i] = cellDataList[i].expression;
            typeIds[i] = cellDataList[i].typeId;
        }
        
        positionBuffer.SetData(positions);
        scaleBuffer.SetData(scales);
        colorBuffer.SetData(colors);
        expressionBuffer.SetData(expressions);
        typeIdBuffer.SetData(typeIds);
    }

    // Update visual effects of individual cells
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
        
        // Update the corresponding Buffer data
        positionBuffer.SetData(new Vector3[] { position }, 0, index, 1);
        scaleBuffer.SetData(new float[] { scale }, 0, index, 1);
        colorBuffer.SetData(new Vector4[] { new Vector4(color.r, color.g, color.b, color.a) }, 0, index, 1);
        expressionBuffer.SetData(new float[] { expression }, 0, index, 1);
        // typeId updates are typically static, but can add it if needed
    }

    // Batch update cell visual effects
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

    // Update all cells based on view mode
    public void RefreshAllCells(ViewMode mode, Dictionary<string, float> expressionMap, 
        Dictionary<string, int> typeMap, Dictionary<string, int> aiPredictionMap,
        Dictionary<string, int> zeroShotClusterMap, Dictionary<int, Color> zeroShotColorMap,
        int highlightedTypeID,
        int highlightedRegionID, 
        Dictionary<string, int> regionMap,
        bool isImputation = false,
        Dictionary<string, bool> isImputedMap = null)
    {
        currentViewMode = mode;
        bool allowSurfacePersist = (mode == ViewMode.Expression && isImputation) 
            || mode == ViewMode.TissueRegion 
            || mode == ViewMode.AI_Annotation 
            || mode == ViewMode.ZeroShot;
        if (!allowSurfacePersist)
        {
            showSurfaceMode = false;
        }
        
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
                        targetValue = expressionMap[cellId];
                        cell.expression = targetValue;
                    }
                    else targetValue = cell.expression; // fallback
                    
                    if (isImputation && enableImputedColorOverride && isImputedMap != null && isImputedMap.ContainsKey(cellId) && isImputedMap[cellId])
                    {
                        baseColor = Color.Lerp(Color.black, imputedCellColor, targetValue);
                    }
                    else
                    {
                        baseColor = colorGradient.Evaluate(targetValue);
                    }
                    scale = 0.5f;
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
                            int safeId = Mathf.Clamp(regionId, 0, regionColors.Length - 1);
                            baseColor = regionColors[safeId];
                            scale = 0.8f;
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

            if (showSurfaceMode)
            {
                newPos.y += 0.1f;
            }
            
            cell.position = newPos;
            cell.color = new Vector4(baseColor.r, baseColor.g, baseColor.b, 1.0f);
            cell.scale = scale;
            
            if (expressionMap != null && expressionMap.ContainsKey(cellId))
            {
                cell.expression = expressionMap[cellId];
            }
            
            if (mode == ViewMode.TissueRegion && regionMap != null && regionMap.ContainsKey(cellId))
            {
                cell.typeId = regionMap[cellId];
            }
            else if (mode == ViewMode.AI_Annotation && aiPredictionMap != null && aiPredictionMap.ContainsKey(cellId))
            {
                cell.typeId = aiPredictionMap[cellId];
            }
            else if (mode == ViewMode.ZeroShot && zeroShotClusterMap != null && zeroShotClusterMap.ContainsKey(cellId))
            {
                cell.typeId = zeroShotClusterMap[cellId];
            }
            else if (typeMap != null && typeMap.ContainsKey(cellId))
            {
                cell.typeId = typeMap[cellId]; 
            }
            
            cellDataList[i] = cell;
        }
        
        UpdateAllBuffers();

        if (showSurfaceMode)
        {
            ComputeSurfaceInterpolation();
        }
    }

    // Get the index corresponding to the cell ID
    public bool TryGetCellIndex(string cellId, out int index)
    {
        return cellIdToIndexMap.TryGetValue(cellId, out index);
    }

    // Get cell data
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

    //     Obtain cell ID
    public string GetCellId(int index)
    {
        if (index >= 0 && index < cellIdList.Count)
            return cellIdList[index];
        return null;
    }

    // Update area color
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

    // Set global parameters
    public void SetMaterialParameters(float globalScale, float emission)
    {
        if (cellMaterial != null)
        {
            cellMaterial.SetFloat("_GlobalScale", globalScale);
            cellMaterial.SetFloat("_EmissionStrength", emission);
            baseScale = globalScale;
            emissionIntensity = emission;
        }

        if (surfaceMaterial != null)
        {
            surfaceMaterial.SetFloat("_EmissionStrength", surfaceEmissionIntensity);
        }
    }

    // Independently set the basic size for UI Slider to call
    public void SetBaseScale(float scale)
    {
        SetMaterialParameters(scale, emissionIntensity);
    }

    // Independently set the height magnification for UI Slider to call
    public void SetHeightMultiplier(float newMultiplier)
    {
        heightMultiplier = newMultiplier;

        if (surfaceMaterial != null)
        {
            surfaceMaterial.SetFloat("_HeightMultiplier", heightMultiplier);
        }

        if (cellCount == 0) return;

        // Recalculate the Y-axis position of all cells
        for (int i = 0; i < cellDataList.Count; i++)
        {
            CellDataGPU cell = cellDataList[i];
            float targetValue = 0f;
            
            switch (currentViewMode)
            {
                case ViewMode.Expression:
                    targetValue = cell.expression;
                    break;
                case ViewMode.CellType:
                    targetValue = 1.0f;
                    break;
                case ViewMode.AI_Annotation:
                case ViewMode.ZeroShot:
                case ViewMode.TissueRegion:
                    targetValue = 0.5f;
                    break;
            }
            
            float newY = targetValue * heightMultiplier;
            if (showSurfaceMode)
            {
                newY += 0.1f;
            }
            
            cell.position = new Vector3(cell.position.x, newY, cell.position.z);
            cellDataList[i] = cell;
        }
        
        UpdateAllBuffers();

        if (showSurfaceMode)
        {
            ComputeSurfaceInterpolation();
        }
    }
}
