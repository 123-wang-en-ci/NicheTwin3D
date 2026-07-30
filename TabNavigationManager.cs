using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;

public class TabNavigationManager : MonoBehaviour
{
    public static TabNavigationManager Instance { get; private set; }

    public enum TabType
    {
        GeneImputation,
        CellAnnotation,
        RegionSegmentation,
        ZeroShotClustering,
        Settings,
        Help
    }

    [Header("Left Navigation Buttons (左侧 6 大导航按钮)")]
    public Button btnGeneImputation;
    public Button btnCellAnnotation;
    public Button btnRegionSegment;
    public Button btnClustering;
    public Button btnSettings;
    public Button btnHelp;

    [Header("Right Dynamic Content Slots (右侧放置各功能群面板的槽位)")]
    [Tooltip("Right side parent content slot container (右侧用于放置并呈现各功能按钮群的动态槽位)")]
    public Transform rightContentSlot;

    [Tooltip("1. Gene Imputation Sub-Panel (右侧：基因表达插补功能群面板槽)")]
    public GameObject panelGeneImputation;

    [Tooltip("2. Cell Type Annotation Sub-Panel (右侧：细胞类型注释功能群面板槽)")]
    public GameObject panelCellAnnotation;

    [Tooltip("3. Region Segmentation Sub-Panel (右侧：区域组织语义分割功能群面板槽)")]
    public GameObject panelRegionSegment;

    [Tooltip("4. Zero-Shot Clustering Sub-Panel (右侧：零样本聚类功能群面板槽)")]
    public GameObject panelClustering;

    [Tooltip("5. Settings Sub-Panel (右侧：设置功能群面板槽，包含中英文切换等)")]
    public GameObject panelSettings;

    [Tooltip("6. Help Sub-Panel (右侧：帮助文档面板槽)")]
    public GameObject panelHelp;

    [Header("Shared Common Controls (公共通用工具栏槽位 - 如截图F12、重置视角等)")]
    [Tooltip("Shared utility panel containing Screenshot(F12), Reset View, etc. (常驻共享工具栏，无需在每个面板重复创建)")]
    public GameObject sharedCommonPanel;

    [Header("Button Colors / Visual Style (按钮激活状态颜色配置)")]
    public Color normalButtonColor = new Color(0.15f, 0.20f, 0.28f, 0.9f);
    public Color activeButtonColor = new Color(0.00f, 0.85f, 0.75f, 1.0f); // Cyan highlight #00FFCC
    public Color normalTextColor = new Color(0.85f, 0.85f, 0.85f, 1.0f);
    public Color activeTextColor = new Color(0.05f, 0.10f, 0.15f, 1.0f);

    private Dictionary<TabType, Button> tabButtons = new Dictionary<TabType, Button>();
    private Dictionary<TabType, GameObject> tabPanels = new Dictionary<TabType, GameObject>();

    private TabType currentTab = TabType.GeneImputation;

    void Awake()
    {
        if (Instance == null)
        {
            Instance = this;
        }
        else
        {
            Destroy(gameObject);
        }
    }

    void Start()
    {
        RegisterTabs();
        BindButtonEvents();

        // Default select Gene Imputation on startup
        SelectTab(TabType.GeneImputation);
    }

    private void RegisterTabs()
    {
        tabButtons[TabType.GeneImputation] = btnGeneImputation;
        tabButtons[TabType.CellAnnotation] = btnCellAnnotation;
        tabButtons[TabType.RegionSegmentation] = btnRegionSegment;
        tabButtons[TabType.ZeroShotClustering] = btnClustering;
        tabButtons[TabType.Settings] = btnSettings;
        tabButtons[TabType.Help] = btnHelp;

        tabPanels[TabType.GeneImputation] = panelGeneImputation;
        tabPanels[TabType.CellAnnotation] = panelCellAnnotation;
        tabPanels[TabType.RegionSegmentation] = panelRegionSegment;
        tabPanels[TabType.ZeroShotClustering] = panelClustering;
        tabPanels[TabType.Settings] = panelSettings;
        tabPanels[TabType.Help] = panelHelp;
    }

    private void BindButtonEvents()
    {
        if (btnGeneImputation != null) btnGeneImputation.onClick.AddListener(() => SelectTab(TabType.GeneImputation));
        if (btnCellAnnotation != null) btnCellAnnotation.onClick.AddListener(() => SelectTab(TabType.CellAnnotation));
        if (btnRegionSegment != null) btnRegionSegment.onClick.AddListener(() => SelectTab(TabType.RegionSegmentation));
        if (btnClustering != null) btnClustering.onClick.AddListener(() => SelectTab(TabType.ZeroShotClustering));
        if (btnSettings != null) btnSettings.onClick.AddListener(() => SelectTab(TabType.Settings));
        if (btnHelp != null) btnHelp.onClick.AddListener(() => SelectTab(TabType.Help));
    }

    public void SelectTab(TabType selectedTab)
    {
        // 1. Trigger automatic Reset View / Clear Data when switching between the 4 main analysis tabs
        bool isNewAnalysisTab = (selectedTab == TabType.GeneImputation ||
                                 selectedTab == TabType.CellAnnotation ||
                                 selectedTab == TabType.RegionSegmentation ||
                                 selectedTab == TabType.ZeroShotClustering);

        if (isNewAnalysisTab && selectedTab != currentTab)
        {
            InteractionManager interaction = FindObjectOfType<InteractionManager>();
            if (interaction != null)
            {
                interaction.RequestClearData();
            }
        }

        currentTab = selectedTab;

        // 2. Hide all right sub-panels and show only the selected one in the rightContentSlot
        foreach (var kvp in tabPanels)
        {
            if (kvp.Value != null)
            {
                bool shouldShow = (kvp.Key == selectedTab);
                kvp.Value.SetActive(shouldShow);

                // Ensure active panel is parented to rightContentSlot if assigned
                if (shouldShow && rightContentSlot != null && kvp.Value.transform.parent != rightContentSlot)
                {
                    kvp.Value.transform.SetParent(rightContentSlot, false);
                }
            }
        }

        // 3. Control visibility of sharedCommonPanel (Screenshot F12, Reset View, etc.)
        // Show sharedCommonPanel ONLY for analysis tabs; hide on Settings and Help tabs
        if (sharedCommonPanel != null)
        {
            sharedCommonPanel.SetActive(isNewAnalysisTab);
        }

        // Special handling for Help button tab: trigger HelpManager if assigned
        if (selectedTab == TabType.Help)
        {
            if (HelpManager.Instance != null)
            {
                HelpManager.Instance.OpenHelp();
            }
        }

        // 2. Update visual styles of left navigation buttons
        foreach (var kvp in tabButtons)
        {
            if (kvp.Value != null)
            {
                bool isActive = (kvp.Key == selectedTab);
                Image btnImg = kvp.Value.GetComponent<Image>();
                if (btnImg != null)
                {
                    btnImg.color = isActive ? activeButtonColor : normalButtonColor;
                }

                // Update text color
                var tmpText = kvp.Value.GetComponentInChildren<TMPro.TextMeshProUGUI>();
                if (tmpText != null)
                {
                    tmpText.color = isActive ? activeTextColor : normalTextColor;
                }
                else
                {
                    var stdText = kvp.Value.GetComponentInChildren<Text>();
                    if (stdText != null)
                    {
                        stdText.color = isActive ? activeTextColor : normalTextColor;
                    }
                }
            }
        }

        // Refresh canvas layout
        Canvas.ForceUpdateCanvases();
    }
}
