using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;
using TMPro;

public class CellComparisonManager : MonoBehaviour
{
    public static CellComparisonManager Instance { get; private set; }

    [Header("Settings")]
    public Vector3 offset = new Vector3(800f, 0f, 0f);
    public bool isComparisonMode = false;

    [Header("References")]
    public GPURenderer mainRenderer;
    public Camera mainCamera;
    public DataLoaderGPU dataLoader;

    [Header("UI Reference")]
    public Button compareButton;

    [Header("Label UI References")]
    public GameObject beforeLabelUI;
    public GameObject afterLabelUI;

    private GPURenderer beforeRenderer;
    private Camera beforeCamera;

    // Snapshot Cache
    private GPURenderer.CellDataGPU[] snapshotCellData;
    private GPURenderer.ViewMode snapshotViewMode;
    private bool snapshotShowSurfaceMode;

    void Awake()
    {
        if (Instance == null)
        {
            Instance = this;
            DontDestroyOnLoad(gameObject);
        }
        else
        {
            Destroy(gameObject);
        }
    }

    void Start()
    {
        if (mainRenderer == null) mainRenderer = FindObjectOfType<GPURenderer>();
        if (mainCamera == null)
        {
            InteractionManager interact = FindObjectOfType<InteractionManager>();
            if (interact != null) mainCamera = interact.mainCamera;
        }
        if (mainCamera == null) mainCamera = Camera.main;
        if (dataLoader == null) dataLoader = FindObjectOfType<DataLoaderGPU>();

        if (compareButton != null)
        {
            compareButton.onClick.RemoveAllListeners();
            compareButton.onClick.AddListener(ToggleComparisonMode);
            UpdateButtonText();
        }

        // Hide labels initially
        if (beforeLabelUI != null) beforeLabelUI.SetActive(false);
        if (afterLabelUI != null) afterLabelUI.SetActive(false);
    }

    void OnEnable()
    {
        if (LocalizationManager.Instance != null)
            LocalizationManager.Instance.OnLanguageChanged += UpdateButtonText;
    }

    void OnDisable()
    {
        if (LocalizationManager.Instance != null)
            LocalizationManager.Instance.OnLanguageChanged -= UpdateButtonText;
    }

    public void UpdateButtonText()
    {
        if (compareButton != null)
        {
            string key = isComparisonMode ? "BTN_COMPARE_ON" : "BTN_COMPARE_OFF";
            string textVal = LocalizationManager.Instance != null ? LocalizationManager.Instance.GetText(key) : (isComparisonMode ? "Compare: ON" : "Compare: OFF");

            TextMeshProUGUI tmpText = compareButton.GetComponentInChildren<TextMeshProUGUI>();
            if (tmpText != null) tmpText.text = textVal;
            else
            {
                Text uiText = compareButton.GetComponentInChildren<Text>();
                if (uiText != null) uiText.text = textVal;
            }
        }
    }

    public void SaveBeforeStateSnapshot()
    {
        if (mainRenderer == null) return;
        
        var list = mainRenderer.GetCellDataList();
        if (list == null || list.Count == 0) return;

        // Take a copy of current cell data list
        snapshotCellData = list.ToArray();
        snapshotViewMode = mainRenderer.currentViewMode;
        snapshotShowSurfaceMode = mainRenderer.showSurfaceMode;
        
        Debug.Log($"[ComparisonManager] Snapshotted 'before' state with {snapshotCellData.Length} cells.");

        // If comparison mode is currently active, apply the snapshot to beforeRenderer immediately
        if (isComparisonMode && beforeRenderer != null)
        {
            ApplySnapshotToBeforeRenderer();
        }
    }

    private void ApplySnapshotToBeforeRenderer()
    {
        if (beforeRenderer == null || snapshotCellData == null) return;

        // Initialize/copy snapshot data to beforeRenderer
        beforeRenderer.InitializeData(
            new List<GPURenderer.CellDataGPU>(snapshotCellData),
            mainRenderer.GetCellIdToIndexMap(),
            mainRenderer.GetCellIdList()
        );
        beforeRenderer.currentViewMode = snapshotViewMode;
        beforeRenderer.showSurfaceMode = snapshotShowSurfaceMode;
        
        if (snapshotShowSurfaceMode)
        {
            beforeRenderer.ComputeSurfaceInterpolation();
        }
    }

    public void ToggleComparisonMode()
    {
        isComparisonMode = !isComparisonMode;
        
        // Update Compare Button Text
        UpdateButtonText();

        if (isComparisonMode)
        {
            EnableComparison();
        }
        else
        {
            DisableComparison();
        }

        if (UIManager.Instance != null)
        {
            UIManager.Instance.ShowSystemMessage(
                isComparisonMode ? "Comparison Mode Enabled (Left: Before, Right: After)" : "Comparison Mode Disabled", 
                false
            );
        }
    }

    private void EnsureComparisonLabelsExist()
    {
        if (beforeLabelUI != null && afterLabelUI != null) return;

        Canvas mainCanvas = FindObjectOfType<Canvas>();
        if (mainCanvas == null) return;

        bool isChinese = LocalizationManager.Instance != null && 
                         LocalizationManager.Instance.currentLanguage == Language.Chinese;

        // 1. Create Left Label (Initial State / 原始基座状态)
        if (beforeLabelUI == null)
        {
            GameObject labelObj = new GameObject("Before_ComparisonLabel_UI");
            labelObj.transform.SetParent(mainCanvas.transform, false);
            
            RectTransform rect = labelObj.AddComponent<RectTransform>();
            rect.anchorMin = new Vector2(0.02f, 0.93f);
            rect.anchorMax = new Vector2(0.24f, 0.98f);
            rect.pivot = new Vector2(0f, 1f);
            rect.anchoredPosition = Vector2.zero;

            Image bg = labelObj.AddComponent<Image>();
            bg.color = new Color(0.05f, 0.10f, 0.18f, 0.85f);

            GameObject textObj = new GameObject("Text");
            textObj.transform.SetParent(labelObj.transform, false);
            RectTransform textRect = textObj.AddComponent<RectTransform>();
            textRect.anchorMin = Vector2.zero;
            textRect.anchorMax = Vector2.one;
            textRect.sizeDelta = Vector2.zero;

            TextMeshProUGUI tmp = textObj.AddComponent<TextMeshProUGUI>();
            tmp.text = isChinese ? "<b><color=#00FFCC>◀ 原始基座状态 (Initial State)</color></b>" : "<b><color=#00FFCC>◀ Initial State</color></b>";
            tmp.alignment = TextAlignmentOptions.Center;
            tmp.fontSize = 16;

            beforeLabelUI = labelObj;
        }

        // 2. Create Right Label (Forecast Result / AI预测结果)
        if (afterLabelUI == null)
        {
            GameObject labelObj = new GameObject("After_ComparisonLabel_UI");
            labelObj.transform.SetParent(mainCanvas.transform, false);
            
            RectTransform rect = labelObj.AddComponent<RectTransform>();
            rect.anchorMin = new Vector2(0.52f, 0.93f);
            rect.anchorMax = new Vector2(0.74f, 0.98f);
            rect.pivot = new Vector2(0f, 1f);
            rect.anchoredPosition = Vector2.zero;

            Image bg = labelObj.AddComponent<Image>();
            bg.color = new Color(0.05f, 0.10f, 0.18f, 0.85f);

            GameObject textObj = new GameObject("Text");
            textObj.transform.SetParent(labelObj.transform, false);
            RectTransform textRect = textObj.AddComponent<RectTransform>();
            textRect.anchorMin = Vector2.zero;
            textRect.anchorMax = Vector2.one;
            textRect.sizeDelta = Vector2.zero;

            TextMeshProUGUI tmp = textObj.AddComponent<TextMeshProUGUI>();
            tmp.text = isChinese ? "<b><color=#FFCC00>AI 预测结果 (Forecast Result) ▶</color></b>" : "<b><color=#FFCC00>Forecast Result ▶</color></b>";
            tmp.alignment = TextAlignmentOptions.Center;
            tmp.fontSize = 16;

            afterLabelUI = labelObj;
        }
    }

    void EnableComparison()
    {
        if (mainRenderer == null) return;

        // Ensure dual-screen labels exist
        EnsureComparisonLabelsExist();

        // Snapshot current state if none exists yet
        if (snapshotCellData == null)
        {
            SaveBeforeStateSnapshot();
        }

        // 1. Create duplicate GPURenderer at offset
        if (beforeRenderer == null)
        {
            GameObject beforeRendererObj = new GameObject("Before_Renderer");
            beforeRendererObj.transform.position = offset;
            beforeRendererObj.transform.rotation = Quaternion.identity;
            beforeRenderer = beforeRendererObj.AddComponent<GPURenderer>();
            
            // Set fields from main renderer
            beforeRenderer.cellMesh = mainRenderer.cellMesh;
            beforeRenderer.cellMaterial = Instantiate(mainRenderer.cellMaterial);
            if (mainRenderer.surfaceMaterial != null)
                beforeRenderer.surfaceMaterial = Instantiate(mainRenderer.surfaceMaterial);
            
            beforeRenderer.idwComputeShader = mainRenderer.idwComputeShader;
            beforeRenderer.gridResolution = mainRenderer.gridResolution;
            beforeRenderer.surfaceSmoothingRadius = mainRenderer.surfaceSmoothingRadius;
            beforeRenderer.maxSurfaceInstances = mainRenderer.maxSurfaceInstances;
            beforeRenderer.colorGradient = mainRenderer.colorGradient;
            beforeRenderer.positionScale = mainRenderer.positionScale;
            beforeRenderer.heightMultiplier = mainRenderer.heightMultiplier;
            beforeRenderer.baseScale = mainRenderer.baseScale;
            beforeRenderer.emissionIntensity = mainRenderer.emissionIntensity;
            beforeRenderer.surfaceEmissionIntensity = mainRenderer.surfaceEmissionIntensity;
            beforeRenderer.typeColorCount = mainRenderer.typeColorCount;
            beforeRenderer.typeColors = mainRenderer.typeColors;
            beforeRenderer.regionColors = mainRenderer.regionColors;
            beforeRenderer.saturation = mainRenderer.saturation;
            beforeRenderer.brightness = mainRenderer.brightness;
            beforeRenderer.imputedCellColor = mainRenderer.imputedCellColor;
            beforeRenderer.enableImputedColorOverride = mainRenderer.enableImputedColorOverride;

            // Apply position offset vector in the material instances
            beforeRenderer.cellMaterial.SetVector("_PosOffset", offset);
            if (beforeRenderer.surfaceMaterial != null)
                beforeRenderer.surfaceMaterial.SetVector("_PosOffset", offset);
        }

        ApplySnapshotToBeforeRenderer();
        beforeRenderer.gameObject.SetActive(true);

        // 2. Create duplicate camera
        if (beforeCamera == null)
        {
            beforeCamera = Instantiate(mainCamera);
            beforeCamera.name = "Before_Camera";

            // Remove controllers and listeners to avoid input & audio conflicts
            if (beforeCamera.TryGetComponent<UnityTemplateProjects.SimpleCameraController>(out var controller))
                Destroy(controller);
            if (beforeCamera.TryGetComponent<CameraOrbit>(out var orbit))
                Destroy(orbit);
            if (beforeCamera.TryGetComponent<AudioListener>(out var listener))
                Destroy(listener);
        }

        // Viewport rect split screen (Left: Before, Right: After)
        mainCamera.rect = new Rect(0.5f, 0f, 0.5f, 1f);
        beforeCamera.rect = new Rect(0f, 0f, 0.5f, 1f);
        beforeCamera.gameObject.SetActive(true);

        // Target cameras so they only render for their respective viewport
        mainRenderer.targetCamera = mainCamera;
        beforeRenderer.targetCamera = beforeCamera;

        // Show split screen labels
        if (beforeLabelUI != null) beforeLabelUI.SetActive(true);
        if (afterLabelUI != null) afterLabelUI.SetActive(true);

        SyncCameras();
    }

    void DisableComparison()
    {
        // Reset target cameras so mainRenderer draws for all cameras again
        if (mainRenderer != null)
        {
            mainRenderer.targetCamera = null;
        }
        if (beforeRenderer != null)
        {
            beforeRenderer.targetCamera = null;
        }

        // Reset viewport rect of the main camera
        if (mainCamera != null)
        {
            mainCamera.rect = new Rect(0f, 0f, 1f, 1f);
        }

        // Disable comparison camera
        if (beforeCamera != null)
        {
            beforeCamera.gameObject.SetActive(false);
        }

        // Disable comparison renderer
        if (beforeRenderer != null)
        {
            beforeRenderer.gameObject.SetActive(false);
        }

        // Hide split screen labels
        if (beforeLabelUI != null) beforeLabelUI.SetActive(false);
        if (afterLabelUI != null) afterLabelUI.SetActive(false);
    }

    void LateUpdate()
    {
        if (isComparisonMode && beforeCamera != null && mainCamera != null)
        {
            SyncCameras();
        }
    }

    void SyncCameras()
    {
        // Mirror all camera coordinates plus the spatial rendering offset
        beforeCamera.transform.position = mainCamera.transform.position + offset;
        beforeCamera.transform.rotation = mainCamera.transform.rotation;
        beforeCamera.orthographic = mainCamera.orthographic;
        beforeCamera.orthographicSize = mainCamera.orthographicSize;
        beforeCamera.fieldOfView = mainCamera.fieldOfView;
    }

    void OnDestroy()
    {
        if (beforeCamera != null) Destroy(beforeCamera.gameObject);
        if (beforeRenderer != null) Destroy(beforeRenderer.gameObject);
    }
}
