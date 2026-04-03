using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System.Globalization;
using UnityEngine.Networking;
using UnityEngine.UI;
using TMPro;

/// <summary>
/// GPU Instancing version of DataLoader 
/// 
/// /// Uses GPU rendering instead of traditional GameObject methods, greatly improving performance
/// </summary>
public class DataLoaderGPU : MonoBehaviour
{
    [Header("Data Settings")]
    public string csvFileName = "unity_cell_data.csv";

    // Render properties mapped to GPURenderer.cs

    [Header("GPU Rendering Components")]
    public GPURenderer gpuRenderer;
    public CellProxyManager proxyManager;

    [Header("Legend Panel")]
    public GameObject legendPanel;
    public GameObject legendItemPrefab;
    public Transform legendContent;
    public GameObject toggleSurfaceButtonObj; // UI button to toggle render mode

    [Header("Region Filter")]
    public TMP_Dropdown regionDropdown;
    private Dictionary<string, int> regionMap = new Dictionary<string, int>(); 
    public int highlightedRegionID = -1;
    private List<string> currentRegionNames = new List<string>();
    private List<int> savedRegionIds = new List<int>();

    //     Use GPURenderer's ViewMode enumeration to maintain consistency
    public GPURenderer.ViewMode currentMode = GPURenderer.ViewMode.Expression;
    private bool currentIsImputation = false;

    private Dictionary<string, CellData> currentDataMap = new Dictionary<string, CellData>();
    private Dictionary<string, float> originalExpressionMap = new Dictionary<string, float>();
    private Dictionary<string, bool> isImputedMap = new Dictionary<string, bool>();
    private Dictionary<string, int> aiPredictionMap = new Dictionary<string, int>();
    private Dictionary<string, int> zeroShotClusterMap = new Dictionary<string, int>();
    private Dictionary<int, Color> zeroShotColorMap = new Dictionary<int, Color>();

    public int highlightedTypeID = -1;
    public List<string> annotationLegend = new List<string>();
    private List<GameObject> legendItems = new List<GameObject>();

    struct CellData
    {
        public string id;
        public float x;
        public float y;
        public float expression;
        public int typeId;
        public string typeName;
    }

    // Color configurations mapped to GPURenderer.cs

    void Awake()
    {
        // Make sure the GPU Renderer exists
        if (gpuRenderer == null)
        {
            gpuRenderer = FindObjectOfType<GPURenderer>();
            if (gpuRenderer == null)
            {
                GameObject gpuObj = new GameObject("GPU_Renderer");
                gpuRenderer = gpuObj.AddComponent<GPURenderer>();
            }
        }
        
        // Make sure that Proxy Manager exists
        if (proxyManager == null)
        {
            GameObject proxyObj = new GameObject("Proxy_Manager");
            proxyManager = proxyObj.AddComponent<CellProxyManager>();
        }
    }

    // Color generation logic moved to GPURenderer.cs

    void Start()
    {
        string filePath = Path.Combine(Application.streamingAssetsPath, csvFileName);
        if (File.Exists(filePath))
        {
            List<CellData> dataList = ParseCSV(filePath);
            InitializeCells(dataList);
        }
        else
        {
            Debug.LogError("Cannot find CSV file: " + filePath);
        }

        if (legendPanel != null)
            legendPanel.SetActive(false);
            
        if (toggleSurfaceButtonObj != null)
            toggleSurfaceButtonObj.SetActive(false);
    }

    List<CellData> ParseCSV(string path)
    {
        List<CellData> list = new List<CellData>();
        string[] lines = File.ReadAllLines(path);

        for (int i = 1; i < lines.Length; i++)
        {
            string line = lines[i];
            if (string.IsNullOrEmpty(line)) continue;
            string[] values = line.Split(',');
            if (values.Length < 6) continue;

            CellData data = new CellData();
            try
            {
                data.id = values[0];
                data.x = float.Parse(values[1], CultureInfo.InvariantCulture);
                data.y = float.Parse(values[2], CultureInfo.InvariantCulture);
                data.expression = float.Parse(values[4], CultureInfo.InvariantCulture);
                data.typeId = int.Parse(values[5]);
                if (values.Length > 6) data.typeName = values[6];

                list.Add(data);

                if (!currentDataMap.ContainsKey(data.id))
                {
                    currentDataMap.Add(data.id, data);
                }
            }
            catch (System.Exception e) { Debug.LogWarning(e.Message); }
        }
        return list;
    }

    void InitializeCells(List<CellData> cells)
    {
        List<GPURenderer.CellDataGPU> gpuDataList = new List<GPURenderer.CellDataGPU>();
        Dictionary<string, int> idToIndexMap = new Dictionary<string, int>();
        List<string> cellIdList = new List<string>();
        List<Vector3> initialPositions = new List<Vector3>();

        for (int i = 0; i < cells.Count; i++)
        {
            var cell = cells[i];
            
            // Calculate the initial position
            Vector3 pos = new Vector3(
                cell.x * gpuRenderer.positionScale,
                cell.expression * gpuRenderer.heightMultiplier,
                cell.y * gpuRenderer.positionScale
            );
            
            // Calculate color and scale
            Color baseColor = gpuRenderer.colorGradient.Evaluate(cell.expression);
            float scale = 0.5f;
            
            GPURenderer.CellDataGPU gpuData = new GPURenderer.CellDataGPU
            {
                position = pos,
                scale = scale,
                color = new Vector4(baseColor.r, baseColor.g, baseColor.b, 1.0f),
                expression = cell.expression,
                typeId = cell.typeId
            };
            
            gpuDataList.Add(gpuData);
            idToIndexMap[cell.id] = i;
            cellIdList.Add(cell.id);
            initialPositions.Add(pos);
        }

        //Initialize the GPU Renderer
        if (gpuRenderer != null)
        {
            gpuRenderer.InitializeData(gpuDataList, idToIndexMap, cellIdList);
        }

        // Initialize the agent Collider system
        if (proxyManager != null)
        {
            proxyManager.InitializeProxies(gpuRenderer, cellIdList, initialPositions);
        }

        Debug.Log($"[DataLoader GPU] Initialized {cells.Count} cells using GPU Instancing");
    }

    void UpdateObjectVisuals(string cellId, bool markProxyUpdate = true)
    {
        if (!currentDataMap.ContainsKey(cellId)) return;
        
        CellData cell = currentDataMap[cellId];
        float targetValue = 0f;
        Color baseColor = Color.white;
        float scale = 0.5f;
        
        switch (currentMode)
        {
            case GPURenderer.ViewMode.Expression:
                targetValue = cell.expression;
                // Data traceability: If there is an open coverage and the cells are high-expression cells interpolated from very low expression (<=0.01).
                if (currentIsImputation && gpuRenderer != null && gpuRenderer.enableImputedColorOverride && isImputedMap.ContainsKey(cell.id) && isImputedMap[cell.id])
                {
                    // Force dyeing to a specified monochrome gradient, preserving intensity mapping
                    baseColor = Color.Lerp(Color.black, gpuRenderer.imputedCellColor, cell.expression);
                }
                else
                {
                    // The original ecology is gradient with a standard red and blue thermometer
                    baseColor = gpuRenderer.colorGradient.Evaluate(cell.expression);
                }
                scale = 0.5f;
                break;
                
            case GPURenderer.ViewMode.CellType:
                targetValue = 1.0f;
                int safeId = Mathf.Clamp(cell.typeId, 0, gpuRenderer.typeColors.Length - 1);
                baseColor = gpuRenderer.typeColors[safeId];
                scale = 0.5f;
                break;
                
            case GPURenderer.ViewMode.AI_Annotation:
                targetValue = 0.5f;
                int predId = 0;
                if (aiPredictionMap.ContainsKey(cell.id))
                {
                    predId = aiPredictionMap[cell.id];
                }

                if (highlightedTypeID == -1 || predId == highlightedTypeID)
                {
                    int safeId2 = Mathf.Clamp(predId, 0, gpuRenderer.typeColors.Length - 1);
                    baseColor = gpuRenderer.typeColors[safeId2];
                    scale = 0.8f;
                }
                else
                {
                    scale = 0.0f;
                }
                break;
                
            case GPURenderer.ViewMode.ZeroShot:
                targetValue = 0.5f;
                scale = 0.7f;

                if (zeroShotClusterMap.ContainsKey(cell.id))
                {
                    int cId = zeroShotClusterMap[cell.id];
                    if (zeroShotColorMap.ContainsKey(cId))
                    {
                        baseColor = zeroShotColorMap[cId];
                    }
                    else
                    {
                        int safeId3 = Mathf.Clamp(cId, 0, gpuRenderer.typeColors.Length - 1);
                        baseColor = gpuRenderer.typeColors[safeId3];
                    }
                }
                else
                {
                    baseColor = Color.gray;
                    scale = 0.3f;
                }
                break;
                
            case GPURenderer.ViewMode.TissueRegion:
                targetValue = 0.5f;
                scale = 0.8f;
                // The region color is handled in ApplyRegionSegmentation
                break;
        }

        Vector3 targetPos = new Vector3(
            cell.x * gpuRenderer.positionScale,
            targetValue * gpuRenderer.heightMultiplier,
            cell.y * gpuRenderer.positionScale
        );

        // The region color is handled in ApplyRegionSegmentation
        if (gpuRenderer != null)
        {
            gpuRenderer.UpdateCellVisual(cellId, targetPos, baseColor, scale, cell.expression);
        }

        // Flag needs to update agent location
        if (markProxyUpdate && proxyManager != null)
        {
            proxyManager.MarkForUpdate();
        }
    }

    [System.Serializable]
    public class UpdateData { public string id; public float new_expr; }

    [System.Serializable]
    public class ServerResponse
    {
        public string status;
        public string message;
        public UpdateData[] updates;
    }

    public void UpdateVisuals(string jsonResponse)
    {
        ServerResponse response = JsonUtility.FromJson<ServerResponse>(jsonResponse);
        if (response == null || response.updates == null) return;

        bool wasImputation = currentIsImputation;
        if (!string.IsNullOrEmpty(response.message))
        {
            currentIsImputation = response.message.Contains("Imputation") || response.message.Contains("Denoise");
        }
        
        // If the query is new (non-imputed), reset the traceability cache
        if (!currentIsImputation)
        {
            originalExpressionMap.Clear();
            isImputedMap.Clear();
        }

        foreach (var update in response.updates)
        {
            if (currentDataMap.ContainsKey(update.id))
            {
                CellData data = currentDataMap[update.id];
                
                if (!currentIsImputation)
                {
                    // Preserve first-hand wild sequencing concentrations
                    originalExpressionMap[update.id] = update.new_expr;
                    isImputedMap[update.id] = false;
                }
                else
                {
                    //  Imputation update: Compare the native concentration
                    float origExpr = originalExpressionMap.ContainsKey(update.id) ? originalExpressionMap[update.id] : 0f;
                    //  Definition: Cells that are native to less than 0.01 and exceed 0.01 after imputation are judged to be cells transformed by pure algorithms!
                    isImputedMap[update.id] = (origExpr <= 0.01f && update.new_expr > 0.01f);
                }
                
                data.expression = update.new_expr;
                currentDataMap[update.id] = data;

                UpdateObjectVisuals(update.id, false);
            }
        }
        
        // Updated graphics card rendering mode
        if (gpuRenderer != null && currentMode == GPURenderer.ViewMode.Expression)
        {
            // If it is false, it will be reset back to false to display the ball again
            gpuRenderer.showSurfaceMode = currentIsImputation;
            
            // If true, in addition to displaying the surface, the interpolation mesh needs to be updated
            if (currentIsImputation)
            {
                gpuRenderer.ComputeSurfaceInterpolation();
            }
            
            if (toggleSurfaceButtonObj != null)
            {
                toggleSurfaceButtonObj.SetActive(currentIsImputation);
            }
        }

        // Bulk update agent locations
        if (proxyManager != null)
        {
            proxyManager.UpdateProxiesImmediate();
        }
    }

    [System.Serializable]
    public class AnnotationUpdate { public string id; public int pred_id; }
    [System.Serializable]
    public class AnnotationResponse { public string status; public string[] legend; public AnnotationUpdate[] updates; }

    public void ApplyAnnotationData(string jsonResponse)
    {
        AnnotationResponse res = JsonUtility.FromJson<AnnotationResponse>(jsonResponse);
        if (res.status != "success") return;

        annotationLegend.Clear();
        annotationLegend.AddRange(res.legend);

        foreach (var update in res.updates)
        {
            if (aiPredictionMap.ContainsKey(update.id))
                aiPredictionMap[update.id] = update.pred_id;
            else
                aiPredictionMap.Add(update.id, update.pred_id);
        }

        currentMode = GPURenderer.ViewMode.AI_Annotation;
        highlightedTypeID = -1;
        
        if (toggleSurfaceButtonObj != null)
            toggleSurfaceButtonObj.SetActive(true);
        if (gpuRenderer != null)
            gpuRenderer.showSurfaceMode = false;
        
        RefreshAllCells();

        StartCoroutine(FetchAnnotationLegend((success) =>
        {
            if (!success) Debug.LogError("Failed to fetch annotation legend");
        }));
    }

    public void ApplyZeroShotClustering(List<ClusterLegendItem> legend, List<ClusterUpdateItem> updates)
    {
        zeroShotColorMap.Clear();
        LegendItem[] uiLegendItems = new LegendItem[legend.Count];

        for (int i = 0; i < legend.Count; i++)
        {
            var item = legend[i];
            Color color;
            if (ColorUtility.TryParseHtmlString(item.color, out color))
            {
                color.a = 1.0f;
                if (!zeroShotColorMap.ContainsKey(item.id))
                    zeroShotColorMap.Add(item.id, color);
            }
            else
            {
                if (!zeroShotColorMap.ContainsKey(item.id))
                    zeroShotColorMap.Add(item.id, gpuRenderer.typeColors[item.id % gpuRenderer.typeColors.Length]);
            }
            uiLegendItems[i] = new LegendItem { id = item.id, name = item.name };
        }

        zeroShotClusterMap.Clear();
        foreach (var update in updates)
        {
            zeroShotClusterMap[update.id] = update.cluster_id;
        }

        currentMode = GPURenderer.ViewMode.ZeroShot;
        
        if (toggleSurfaceButtonObj != null)
            toggleSurfaceButtonObj.SetActive(true);
        if (gpuRenderer != null)
            gpuRenderer.showSurfaceMode = false;
        
        RefreshAllCells();
        CreateLegendPanel(uiLegendItems, zeroShotColorMap);
    }

    public void SwitchMode(int modeIndex)
    {
        GPURenderer.ViewMode oldMode = currentMode;
        currentMode = (GPURenderer.ViewMode)modeIndex;

        if ((oldMode == GPURenderer.ViewMode.AI_Annotation || oldMode == GPURenderer.ViewMode.ZeroShot || oldMode == GPURenderer.ViewMode.TissueRegion) &&
            (currentMode != GPURenderer.ViewMode.AI_Annotation && currentMode != GPURenderer.ViewMode.ZeroShot && currentMode != GPURenderer.ViewMode.TissueRegion))
        {
            ClearLegendPanel();
        }
        else if (currentMode == GPURenderer.ViewMode.AI_Annotation)
        {
            if (annotationLegend.Count > 0)
                StartCoroutine(FetchAnnotationLegend(null));
        }
        
        if (toggleSurfaceButtonObj != null)
        {
            bool showToggle = (currentMode == GPURenderer.ViewMode.Expression && currentIsImputation) 
                || currentMode == GPURenderer.ViewMode.TissueRegion
                || currentMode == GPURenderer.ViewMode.AI_Annotation
                || currentMode == GPURenderer.ViewMode.ZeroShot;
            toggleSurfaceButtonObj.SetActive(showToggle);
        }

        RefreshAllCells();
    }

    public void ToggleViewMode() { int nextMode = (currentMode == GPURenderer.ViewMode.Expression) ? 1 : 0; SwitchMode(nextMode); }

    public void ToggleSurfaceMode()
    {
        bool isAllowed = (currentMode == GPURenderer.ViewMode.Expression && currentIsImputation) 
            || currentMode == GPURenderer.ViewMode.TissueRegion
            || currentMode == GPURenderer.ViewMode.AI_Annotation
            || currentMode == GPURenderer.ViewMode.ZeroShot;
        if (isAllowed)
        {
            if (gpuRenderer != null)
            {
                gpuRenderer.showSurfaceMode = !gpuRenderer.showSurfaceMode;
                RefreshAllCells();
                if (gpuRenderer.showSurfaceMode)
                {
                    gpuRenderer.ComputeSurfaceInterpolation();
                }
            }
        }
    }

    void RefreshAllCells()
    {
        // Build data mappings
        Dictionary<string, float> expressionMap = new Dictionary<string, float>();
        Dictionary<string, int> typeMap = new Dictionary<string, int>();

        foreach (var kvp in currentDataMap)
        {
            expressionMap[kvp.Key] = kvp.Value.expression;
            typeMap[kvp.Key] = kvp.Value.typeId;
        }

        // Use the GPU Renderer's batch update method
        if (gpuRenderer != null)
        {
            gpuRenderer.RefreshAllCells(
                (GPURenderer.ViewMode)currentMode,
                expressionMap,
                typeMap,
                aiPredictionMap,
                zeroShotClusterMap,
                zeroShotColorMap,
                highlightedTypeID,
                highlightedRegionID, // Make sure this variable is updated correctly in FilterRegions
                this.regionMap,     
                currentIsImputation,
                this.isImputedMap
            );
        }

        // Update agent location
        if (proxyManager != null)
        {
            proxyManager.UpdateProxiesImmediate();
        }
    }
    public bool GetCellDetails(string id, out string typeName, out Vector2 pos, out float expr)
    {
        if (currentDataMap.ContainsKey(id))
        {
            CellData data = currentDataMap[id];
            typeName = string.IsNullOrEmpty(data.typeName) ? "Unknown" : data.typeName;
            pos = new Vector2(data.x, data.y);
            expr = data.expression;
            return true;
        }
        typeName = "Unknown"; pos = Vector2.zero; expr = 0;
        return false;
    }

    public float GetAverageExpression()
    {
        if (currentDataMap.Count == 0) return 0;
        float sum = 0;
        foreach (var kvp in currentDataMap) sum += kvp.Value.expression;
        return sum / currentDataMap.Count;
    }

    [System.Serializable]
    public class LegendItem
    {
        public int id;
        public string name;
    }

    [System.Serializable]
    public class LegendResponse
    {
        public string status;
        public LegendItem[] legend;
    }

    public IEnumerator FetchAnnotationLegend(System.Action<bool> onComplete)
    {
        UnityWebRequest request = UnityWebRequest.Get("http://localhost:8000/annotation_legend");
        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            string jsonRespone = request.downloadHandler.text;
            ProcessLegendData(jsonRespone);
            if (onComplete != null) onComplete(true);
        }
        else
        {
            Debug.LogError("Failed to fetch legend: " + request.error);
            if (onComplete != null) onComplete(false);
        }
    }

    private void ProcessLegendData(string jsonRespone)
    {
        var response = JsonUtility.FromJson<LegendResponse>(jsonRespone);
        if (response.status == "success")
        {
            CreateLegendPanel(response.legend);
        }
    }

    private void CreateLegendPanel(LegendItem[] legendData, Dictionary<int, Color> overrideColors = null)
    {
        ClearLegendPanel();

        if (legendPanel != null)
            legendPanel.SetActive(true);

        foreach (var item in legendData)
        {
            if (legendItemPrefab != null && legendContent != null)
            {
                GameObject legendItemObj = Instantiate(legendItemPrefab, legendContent);
                legendItems.Add(legendItemObj);

                Image colorBox = null;
                Transform colorBoxTransform = legendItemObj.transform.Find("ColorBox");
                if (colorBoxTransform != null) colorBox = colorBoxTransform.GetComponent<Image>();
                else colorBox = legendItemObj.GetComponentInChildren<Image>();

                TMPro.TMP_Text tmpTextLabel = legendItemObj.GetComponentInChildren<TMPro.TMP_Text>();
                Text uiTextLabel = legendItemObj.GetComponentInChildren<Text>();

                Color finalColor = Color.white;
                if (overrideColors != null && overrideColors.ContainsKey(item.id))
                {
                    finalColor = overrideColors[item.id];
                }
                else if (item.id < gpuRenderer.typeColors.Length)
                {
                    if (currentMode == GPURenderer.ViewMode.TissueRegion && gpuRenderer.regionColors != null)
                        finalColor = gpuRenderer.regionColors[item.id];
                    else
                        finalColor = gpuRenderer.typeColors[item.id];
                }

                if (finalColor.a <= 0f) finalColor.a = 1f;
                if (colorBox != null) colorBox.color = finalColor;

                if (tmpTextLabel != null) tmpTextLabel.text = item.name;
                else if (uiTextLabel != null) uiTextLabel.text = item.name;
            }
        }
        Canvas.ForceUpdateCanvases();
        if (legendContent.TryGetComponent<VerticalLayoutGroup>(out var layout))
        {
            layout.enabled = false;
            layout.enabled = true;
        }
    }

    private void ClearLegendPanel()
    {
        foreach (var item in legendItems)
        {
            if (item != null) DestroyImmediate(item);
        }
        legendItems.Clear();

        if (legendPanel != null) legendPanel.SetActive(false);
    }

    public void ApplyRegionSegmentation(List<int> regionIds, List<string> regionNames)
    {
        currentMode = GPURenderer.ViewMode.TissueRegion;
        Debug.Log($"[DataLoader GPU] Applying region segmentation: {regionIds.Count} regions");

        // 1. Empty and repopulate the class member variable regionMap
        this.regionMap.Clear();
        List<string> cellIdList = new List<string>(currentDataMap.Keys);

        for (int i = 0; i < cellIdList.Count && i < regionIds.Count; i++)
        {
            this.regionMap[cellIdList[i]] = regionIds[i];
        }

        // 2.  Initialize Dropdown
        currentRegionNames = regionNames;
        savedRegionIds = regionIds;
        InitRegionDropdown(regionNames);

        // 3.Show all by default (-1)
        highlightedRegionID = -1;

        if (regionNames != null && regionNames.Count > 0)
        {
            LegendItem[] legendData = new LegendItem[regionNames.Count];
            for (int i = 0; i < regionNames.Count; i++)
            {
                // Create legend data with IDs of 0, 1, 2:
                legendData[i] = new LegendItem { id = i, name = regionNames[i] };
            }
            // Call the Create Legend panel
            CreateLegendPanel(legendData);
        }
        else
        {
            // If there is no name, only ID, try to generate a generic legend as well
            int maxId = 0;
            foreach (var id in savedRegionIds) if (id > maxId) maxId = id;

            LegendItem[] legendData = new LegendItem[maxId + 1];
            for (int i = 0; i <= maxId; i++)
            {
                legendData[i] = new LegendItem { id = i, name = $"Region {i}" };
            }
            CreateLegendPanel(legendData);
        }

        if (toggleSurfaceButtonObj != null)
        {
            toggleSurfaceButtonObj.SetActive(true);
            if (gpuRenderer != null) gpuRenderer.showSurfaceMode = false; // Spheres are displayed by default
        }

        // 4. Refresh the view
        RefreshAllCells();
    }

    private void InitRegionDropdown(List<string> names)
    {
        if (regionDropdown == null) return;
        regionDropdown.ClearOptions();
        List<string> options = new List<string> { "Show All" };

        // If the backend has a name, use the name, and if not, use Region 0, Region 1...
        if (names != null && names.Count > 0)
        {
            options.AddRange(names);
        }
        else
        {
            // If names are empty, the ID name is automatically generated
            int maxId = 0;
            foreach (var id in savedRegionIds) if (id > maxId) maxId = id;
            for (int i = 0; i <= maxId; i++) options.Add($"Region {i}");
        }

        regionDropdown.AddOptions(options);
        regionDropdown.value = 0; // Show All is selected by default
        regionDropdown.onValueChanged.RemoveAllListeners();
        regionDropdown.onValueChanged.AddListener(OnDropdownValueChanged);
    }


    private void OnDropdownValueChanged(int index)
    {
        // index 0 is "Show All" (-1) 
        // index 1 is "Region 0" (0) 
        // so targetId = index - 1
        int targetRegionId = index - 1;

        FilterRegions(targetRegionId);
    }

    public void FilterRegions(int targetRegionId)
    {
        // 1. Update the Highlight ID status
        this.highlightedRegionID = targetRegionId;

        // 2. If it is not currently in regional mode, force toggle
        if (currentMode != GPURenderer.ViewMode.TissueRegion)
        {
            currentMode = GPURenderer.ViewMode.TissueRegion;
        }

        Debug.Log($"[DataLoader GPU] 切换区域显示: {(targetRegionId == -1 ? "ALL" : targetRegionId.ToString())}");

        // 3.RefreshAllCells is called to let the GPU handle the visibility
        RefreshAllCells();
    }
}
