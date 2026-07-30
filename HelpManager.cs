using UnityEngine;
using TMPro;
using UnityEngine.UI;

public class HelpManager : MonoBehaviour
{
    public static HelpManager Instance { get; private set; }

    [Header("Help UI Panel References")]
    public GameObject helpPanel;
    public TextMeshProUGUI helpTitleText;
    public TextMeshProUGUI helpContentText;
    public Button closeButton;

    [Header("Toggle Button (Optional)")]
    public Button helpButton;

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
        if (closeButton != null)
        {
            closeButton.onClick.RemoveAllListeners();
            closeButton.onClick.AddListener(CloseHelp);
        }

        if (helpButton != null)
        {
            helpButton.onClick.RemoveAllListeners();
            helpButton.onClick.AddListener(ToggleHelp);
        }

        if (helpPanel != null)
        {
            helpPanel.SetActive(false);
        }

        if (LocalizationManager.Instance != null)
        {
            LocalizationManager.Instance.OnLanguageChanged += RefreshHelpContent;
        }
    }

    void OnDestroy()
    {
        if (LocalizationManager.Instance != null)
        {
            LocalizationManager.Instance.OnLanguageChanged -= RefreshHelpContent;
        }
    }

    void Update()
    {
        // Toggle Help Panel using F1 or H keys
        if (Input.GetKeyDown(KeyCode.F1) || Input.GetKeyDown(KeyCode.H))
        {
            ToggleHelp();
        }

        // Close using ESC if panel is active
        if (helpPanel != null && helpPanel.activeSelf && Input.GetKeyDown(KeyCode.Escape))
        {
            CloseHelp();
        }
    }

    public void ToggleHelp()
    {
        if (helpPanel == null) return;
        bool newState = !helpPanel.activeSelf;
        helpPanel.SetActive(newState);

        if (newState)
        {
            RefreshHelpContent();
        }
    }

    public void OpenHelp()
    {
        if (helpPanel == null) return;
        helpPanel.SetActive(true);
        RefreshHelpContent();
    }

    public void CloseHelp()
    {
        if (helpPanel == null) return;
        helpPanel.SetActive(false);
    }

    public void RefreshHelpContent()
    {
        if (helpPanel == null || !helpPanel.activeSelf) return;

        bool isChinese = LocalizationManager.Instance != null && LocalizationManager.Instance.currentLanguage == Language.Chinese;

        if (helpTitleText != null)
        {
            helpTitleText.text = isChinese ? ":: NicheTwin3D 使用指南与快捷键 ::" : ":: NicheTwin3D HELP & USER GUIDE ::";
        }

        if (helpContentText != null)
        {
            if (isChinese)
            {
                helpContentText.text = GetChineseHelpText();
            }
            else
            {
                helpContentText.text = GetEnglishHelpText();
            }

            // Force TextMeshPro to calculate layout based on current width
            helpContentText.ForceMeshUpdate();

            RectTransform textRect = helpContentText.rectTransform;
            Transform parentTransform = helpContentText.transform.parent;

            // Dynamically calculate required height based on container width
            if (parentTransform != null && parentTransform.TryGetComponent<RectTransform>(out var parentRect))
            {
                float currentWidth = parentRect.rect.width;
                if (currentWidth > 50f)
                {
                    textRect.SetSizeWithCurrentAnchors(RectTransform.Axis.Horizontal, currentWidth);
                }

                // Force TMP to recompute preferred height for new width
                float neededHeight = helpContentText.preferredHeight + 40f;

                // Expand parent Content RectTransform height so ScrollRect never clips text
                parentRect.SetSizeWithCurrentAnchors(RectTransform.Axis.Vertical, neededHeight);
                textRect.SetSizeWithCurrentAnchors(RectTransform.Axis.Vertical, neededHeight);
            }
            else
            {
                float neededHeight = helpContentText.preferredHeight + 40f;
                textRect.SetSizeWithCurrentAnchors(RectTransform.Axis.Vertical, neededHeight);
            }

            Canvas.ForceUpdateCanvases();
        }
    }

    private string GetChineseHelpText()
    {
        return 
            "<b><color=#00FFCC>【1. 3D 视角操作说明】</color></b>\n" +
            "  • <b>按住鼠标右键拖拽</b>：围绕当前焦点 360° 旋转 3D 细胞云视角。\n" +
            "  • <b>按键盘 W/A/S/D/Q/E 键</b>：自由平移相机镜头（W前, S后, A左, D右, E上, Q下）。\n" +
            "  • <b>直接滚动鼠标滚轮</b>：推近或拉远视角放大倍率（支持无缝深入细胞云内部）。\n" +
            "  • <b>鼠标左键点击任意细胞</b>：右下角面板实时显示该细胞ID、细胞类型、空间坐标及基因表达量。\n\n" +

            "<b><color=#00FFCC>【2. 分析功能与界面按钮对照步骤】</color></b>\n" +
            "  • <b>基因表达插补全流程</b>：\n" +
            "     1. 在 <b>[输入基因...]</b> 框中输入目标基因名称（或从右侧下拉菜单直接选择），点击 <b>[搜索]</b> 按钮渲染原始散点表达；\n" +
            "     2. 点击 <b>[基因表达插补]</b> 按钮，启动后端NicheFormer大模型推断插补缺失基因表达数据；\n" +
            "     3. 插补完成后，点击 <b>[曲面化: 关]</b> 按钮在散点表达与连续 3D 活性表达曲面模式之间切换；\n" +
            "     4. 点击 <b>[保存]</b> 按钮即可导出并保存插补后的基因表达数据。\n\n" +

            "  • <b>双屏分屏对比模式</b>：\n" +
            "     点击 <b>[对比模式: 开]</b> 按钮切换双分屏展示（左侧屏幕展示原始基座状态，右侧屏幕展示AI预测/插补结果），方便直接对比验证。\n\n" +

            "  • <b>零样本 Leiden 社区聚类</b>：\n" +
            "     在 <b>[Leiden分辨率输入...]</b> 框中输入聚类分辨率数值（如 0.5，或直接从下拉框选择预设分辨率），点击 <b>[零样本聚类]</b> 按钮，算法将自动对细胞云进行亚群聚类分类并着色。\n\n" +

            "  • <b>细胞类型注释功能</b>：\n" +
            "     首先点击 <b>[细胞类型注释]</b> 按钮调用后端模型进行自动类型预测；待运行完成后，在 <b>[选择一种细胞类型]</b> 下拉框中选中指定细胞类别，3D 视图将自动高亮着色该细胞群落。\n\n" +

            "  • <b>区域组织语义分割</b>：\n" +
            "     首先点击 <b>[区域组织分割]</b> 按钮触发组织结构分割算法；待运行完成后，在 <b>[选择一种区域]</b> 下拉框中选择特定解剖区域，系统将自动高亮并隔离显示该组织语义区域。\n\n" +

            "<b><color=#00FFCC>【3. 快捷键与实用按钮】</color></b>\n" +
            "  • <b>按 F12 键 / 点击 [屏幕截图 (F12)] 按钮</b>：一键截取屏蔽环境 UI 的纯净科研高清 3D 渲染无损大图。\n" +
            "  • <b>点击 [重置视角] 按钮</b>：快速将相机视角恢复至初始对焦中心与默认位置。\n" +
            "  • <b>点击 [中文 / 英文] 按钮</b>：一键全界面中英文双语即时无缝切换。\n" +
            "  • <b>按 F1 键 / H 键 / ESC 键</b>：随时唤起或关闭本帮助文档窗口。";
    }

    private string GetEnglishHelpText()
    {
        return 
            "<b><color=#00FFCC>[1. 3D Navigation & Camera Controls]</color></b>\n" +
            "  • <b>Right Mouse Drag</b>: Rotate the 3D cell cloud 360° around the target pivot.\n" +
            "  • <b>WASD / QE Keys</b>: Pan camera freely (W: forward, S: back, A: left, D: right, E: up, Q: down).\n" +
            "  • <b>Mouse Scroll Wheel</b>: Zoom in / Zoom out view distance (supports zooming deep inside point clouds).\n" +
            "  • <b>Left Click Cell</b>: Inspect single-cell ID, cell type, spatial coords, and gene expression on bottom-right panel.\n\n" +

            "<b><color=#00FFCC>[2. Analysis Workflow & Button Mapping]</color></b>\n" +
            "  • <b>Gene Imputation Workflow</b>:\n" +
            "     1. Type target gene name in <b>[Enter Gene...]</b> input box (or select from dropdown), then click <b>[Search]</b> button to render raw expression;\n" +
            "     2. Click <b>[GeneImputation]</b> button to run NicheFormer model to impute missing expression data;\n" +
            "     3. After completion, click <b>[Surface: OFF]</b> button to toggle between point cloud and continuous 3D surface mode;\n" +
            "     4. Click <b>[Save]</b> button to export and save the imputed gene dataset.\n\n" +

            "  • <b>Dual-Screen Comparison Mode</b>:\n" +
            "     Click <b>[Compare: ON]</b> button to toggle split-screen view (Left: Baseline state, Right: AI Prediction result) for visual comparison.\n\n" +

            "  • <b>Zero-Shot Leiden Clustering</b>:\n" +
            "     Enter resolution value in <b>[Leiden Resolution Input]</b> box (e.g., 0.5, or select from dropdown), then click <b>[ZeroShotClustering]</b> button for automatic unsupervised cell subpopulation discovery.\n\n" +

            "  • <b>Cell Type Annotation</b>:\n" +
            "     First click <b>[CellTypeAnnotation]</b> button to run model prediction; once complete, choose a category from <b>[Select a type of cell]</b> dropdown to highlight that subpopulation.\n\n" +

            "  • <b>Region Tissue Segmentation</b>:\n" +
            "     First click <b>[RegionSegmentation]</b> button to execute spatial domain algorithm; once complete, choose a region from <b>[Select a single area]</b> dropdown to isolate and display that tissue domain.\n\n" +

            "<b><color=#00FFCC>[3. Hotkeys & Utility Buttons]</color></b>\n" +
            "  • <b>F12 Key / [Screenshot (F12)] Button</b>: One-click clean high-res 3D screenshot without UI overlays for publications.\n" +
            "  • <b>[Previous View] Button</b>: Reset camera orientation and focus pivot.\n" +
            "  • <b>[EN / CN] Button</b>: Instant bilingual UI toggle for English and Chinese.\n" +
            "  • <b>F1 / H / ESC Keys</b>: Open or close this Help & User Guide window anytime.";
    }
}
