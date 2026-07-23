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

    public void UpdateButtonText()
    {
        if (compareButton != null)
        {
            TextMeshProUGUI tmpText = compareButton.GetComponentInChildren<TextMeshProUGUI>();
            if (tmpText != null) tmpText.text = isComparisonMode ? "Compare: ON" : "Compare: OFF";
            else
            {
                Text uiText = compareButton.GetComponentInChildren<Text>();
                if (uiText != null) uiText.text = isComparisonMode ? "Compare: ON" : "Compare: OFF";
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

    void EnableComparison()
    {
        if (mainRenderer == null) return;

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
