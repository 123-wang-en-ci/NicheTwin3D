using System;
using System.Collections.Generic;
using UnityEngine;

public enum Language
{
    English,
    Chinese
}

public class LocalizationManager : MonoBehaviour
{
    public static LocalizationManager Instance { get; private set; }

    [Header("Language Settings")]
    public Language currentLanguage = Language.English;

    // Action event fired when language changes
    public event Action OnLanguageChanged;

    private Dictionary<string, string> englishDict = new Dictionary<string, string>();
    private Dictionary<string, string> chineseDict = new Dictionary<string, string>();

    void Awake()
    {
        if (Instance == null)
        {
            Instance = this;
            DontDestroyOnLoad(gameObject);
            InitializeTranslations();
            LoadSavedLanguage();
        }
        else
        {
            Destroy(gameObject);
        }
    }

    private void LoadSavedLanguage()
    {
        int savedLang = PlayerPrefs.GetInt("SelectedLanguage", (int)Language.English);
        currentLanguage = (Language)savedLang;
    }

    void Start()
    {
        OnLanguageChanged?.Invoke();
    }

    public void ToggleLanguage()
    {
        currentLanguage = (currentLanguage == Language.English) ? Language.Chinese : Language.English;
        PlayerPrefs.SetInt("SelectedLanguage", (int)currentLanguage);
        PlayerPrefs.Save();
        
        OnLanguageChanged?.Invoke();
        Debug.Log($"[LocalizationManager] Switched language to: {currentLanguage}");
    }

    public void SetLanguage(Language language)
    {
        if (currentLanguage != language)
        {
            currentLanguage = language;
            PlayerPrefs.SetInt("SelectedLanguage", (int)currentLanguage);
            PlayerPrefs.Save();
            
            OnLanguageChanged?.Invoke();
        }
    }

    public string GetText(string key)
    {
        if (string.IsNullOrEmpty(key)) return "";

        var dict = (currentLanguage == Language.Chinese) ? chineseDict : englishDict;
        if (dict.TryGetValue(key, out string translated))
        {
            return translated;
        }

        // Fallback to key if not found
        return key;
    }

    private void InitializeTranslations()
    {
        // ------------------ GENERAL & BUTTONS ------------------
        AddTranslation("BTN_LANG", "EN / CN", "中文 / 英文");
        AddTranslation("BTN_SEARCH", "Search", "搜索");
        AddTranslation("BTN_RESET", "Previous View", "重置视角");
        AddTranslation("BTN_IMPUTE", "GeneImputation", "基因表达插补");
        AddTranslation("BTN_SAVE", "Save", "保存");
        AddTranslation("BTN_SCREENSHOT", "Screenshot (F12)", "屏幕截图 (F12)");
        AddTranslation("BTN_COMPARE_OFF", "Compare: OFF", "对比模式: 关");
        AddTranslation("BTN_COMPARE_ON", "Compare: ON", "对比模式: 开");
        AddTranslation("BTN_SURFACE_OFF", "Surface: OFF", "曲面化: 关");
        AddTranslation("BTN_SURFACE_ON", "Surface: ON", "曲面化: 开");
        AddTranslation("BTN_CLUSTER", "ZeroShotClustering", "零样本聚类");
        AddTranslation("BTN_ANNOTATE", "CellTypeAnnotation", "细胞类型注释");
        AddTranslation("BTN_SEGMENT", "RegionSegmentation", "区域组织分割");
        AddTranslation("BTN_SELECT_CELLTYPE", "Select a type of cell", "选择一种细胞类型");
        AddTranslation("BTN_SELECT_AREA", "Select a single area", "选择一种区域");
        AddTranslation("BTN_ENTER_GENE", "Enter Gene...", "输入基因...");
        AddTranslation("BTN_LEIDEN_RESOLUTION", "Leiden Resolution Input", "Leiden分辨率输入...");
        AddTranslation("BTN_HELP", "Help (F1)", "帮助 (F1)");

        // ------------------ LEFT TAB NAVIGATION ------------------
        AddTranslation("TAB_IMPUTE", "Gene Imputation", "基因表达插补");
        AddTranslation("TAB_ANNOTATE", "Cell Type Annotation", "细胞类型注释");
        AddTranslation("TAB_SEGMENT", "Region Segmentation", "区域组织分割");
        AddTranslation("TAB_CLUSTER", "Zero-Shot Cluster", "零样本聚类");
        AddTranslation("TAB_SETTINGS", "Settings", "设置");
        AddTranslation("TAB_HELP", "Help", "帮助");

        // ------------------ SINGLE CELL DETAILS ------------------
        AddTranslation("TITLE_SINGLE_CELL", ":: SINGLE CELL ANALYSIS ::", ":: 单细胞表达分析 ::");
        AddTranslation("LABEL_ID_REF", "ID Ref:", "细胞 ID:");
        AddTranslation("LABEL_CELL_TYPE", "Cell Type:", "细胞类型:");
        AddTranslation("LABEL_SPATIAL_COORDS", "Spatial Coords (um):", "空间坐标 (um):");
        AddTranslation("LABEL_GENE_EXPR", "Gene Expression:", "基因表达量:");
        AddTranslation("LABEL_DEV", "Dev:", "离散偏差:");
        AddTranslation("LABEL_VS_AVG", "vs Avg", "对比均值");

        // ------------------ COMPARE LABELS ------------------
        AddTranslation("LABEL_BEFORE_STATE", "InitailState", "初始状态");
        AddTranslation("LABEL_AFTER_STATE", "ForecastResult", "预测结果");

        // ------------------ CLUSTERING & PARAMS ------------------
        AddTranslation("LABEL_BASE_SCALE", "CellScale:", "细胞大小:");
        AddTranslation("LABEL_HEIGHT_MULT", "CellHeight:", "高度倍率:");
        AddTranslation("LABEL_LEIDEN_RES", "Leiden Resolution", "Leiden 分辨率:");

        // ------------------ SYSTEM MESSAGES ------------------
        AddTranslation("MSG_PLEASE_SEARCH_FIRST", "Please search a specific gene first.", "请先搜索特定基因。");
        AddTranslation("MSG_NO_DATA_TO_SAVE", "No gene data to save.", "没有可保存的基因数据。");
        AddTranslation("MSG_COMPARE_ENABLED", "Comparison Mode Enabled (Left: Before, Right: After)", "双屏对比已开启 (左侧: 原始状态, 右侧: 预测结果)");
        AddTranslation("MSG_COMPARE_DISABLED", "Comparison Mode Disabled", "双屏对比已关闭");
        AddTranslation("MSG_RESET_SUCCESS", "Reset Successful", "重置成功");
        AddTranslation("MSG_RESET_FAILED", "Reset Failed", "重置失败");
        AddTranslation("MSG_RUNNING_LEIDEN", "Running Leiden Clustering...", "正在运行 Leiden 零样本聚类...");
        AddTranslation("MSG_CLUSTERING_SUCCESS", "Clustering Complete", "零样本聚类完成");
        AddTranslation("MSG_SCREENSHOT_SAVED", "Screenshot Saved to Gallery", "截图已保存至相册");
    }

    private void AddTranslation(string key, string en, string cn)
    {
        englishDict[key] = en;
        chineseDict[key] = cn;
    }
}
