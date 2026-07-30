using UnityEngine;
using TMPro;
using UnityEngine.UI; 

public class UIManager : MonoBehaviour
{
    [Header("UI components")]
    public TextMeshProUGUI infoTitleText;
    public TextMeshProUGUI infoBodyText;

    [Header("System message component")]
    public GameObject messagePanel; 
    public TextMeshProUGUI messageText; 
    public Image messageBg; 

    public static UIManager Instance;


    void Awake()
    {
        Instance = this;
    }

 
    // Last clicked cell cache for real-time localization refresh
    private bool hasActiveCellDetails = false;
    private string lastId;
    private string lastCellType;
    private Vector2 lastCoordinates;
    private float lastExpression;
    private float lastAvgExpression;

    void OnEnable()
    {
        if (LocalizationManager.Instance != null)
            LocalizationManager.Instance.OnLanguageChanged += RefreshCellDetailsOnLanguageChange;
    }

    void OnDisable()
    {
        if (LocalizationManager.Instance != null)
            LocalizationManager.Instance.OnLanguageChanged -= RefreshCellDetailsOnLanguageChange;
    }

    void Start()
    {
        if (LocalizationManager.Instance != null)
            LocalizationManager.Instance.OnLanguageChanged += RefreshCellDetailsOnLanguageChange;
    }

    private void RefreshCellDetailsOnLanguageChange()
    {
        if (hasActiveCellDetails)
        {
            ShowCellDetails(lastId, lastCellType, lastCoordinates, lastExpression, lastAvgExpression);
        }
    }

    public void ShowCellDetails(string id, string cellType, Vector2 coordinates, float expression, float avgExpression)
    {
        // Cache parameters
        hasActiveCellDetails = true;
        lastId = id;
        lastCellType = cellType;
        lastCoordinates = coordinates;
        lastExpression = expression;
        lastAvgExpression = avgExpression;

        string title = LocalizationManager.Instance != null ? LocalizationManager.Instance.GetText("TITLE_SINGLE_CELL") : ":: SINGLE CELL ANALYSIS ::";
        infoTitleText.text = title;

        string labelId = LocalizationManager.Instance != null ? LocalizationManager.Instance.GetText("LABEL_ID_REF") : "ID Ref:";
        string labelType = LocalizationManager.Instance != null ? LocalizationManager.Instance.GetText("LABEL_CELL_TYPE") : "Cell Type:";
        string labelCoords = LocalizationManager.Instance != null ? LocalizationManager.Instance.GetText("LABEL_SPATIAL_COORDS") : "Spatial Coords (um):";
        string labelExpr = LocalizationManager.Instance != null ? LocalizationManager.Instance.GetText("LABEL_GENE_EXPR") : "Gene Expression:";
        string labelDev = LocalizationManager.Instance != null ? LocalizationManager.Instance.GetText("LABEL_DEV") : "Dev:";
        string labelVsAvg = LocalizationManager.Instance != null ? LocalizationManager.Instance.GetText("LABEL_VS_AVG") : "vs Avg";

        string content = "";
        content += $"<color=#FFFFFF>{labelId}</color>\n";
        content += $"  <b><color=#FFFFFF>{id}</color></b>\n";

        content += $"<color=#FFFFFF>{labelType}</color>\n";
        content += $"  <b><color=#00FF00>{cellType}</color></b>\n";

        content += $"<color=#FFFFFF>{labelCoords}</color>\n";
        content += $"  X: <b>{coordinates.x:F2}</b>  Y: <b>{coordinates.y:F2}</b>\n";

        content += $"<color=#FFFFFF>{labelExpr}</color>\n";
        
        string exprColor = expression > 0.5f ? "#FF4444" : "#4444FF";
        content += $"  Value: <b><color={exprColor}>{expression:F4}</color></b>\n";
        
        float deviation = ((expression - avgExpression) / avgExpression) * 100f;
        string sign = deviation >= 0 ? "+" : "";
        content += $"  {labelDev}   <size=80%>{sign}{deviation:F1}% {labelVsAvg}</size>";

        infoBodyText.text = content;
    }
    // Display system prompt messages in the center of the screen
    public void ShowSystemMessage(string msg, bool isError, bool autoHide = true)
    {
        if (messagePanel == null) return;

        // 1. Set content
        messagePanel.SetActive(true);
        messageText.text = msg;

        // 2. Change color according to the status (for example: red background for errors, green or black background for success)
        if (messageBg != null)
        {
            if (isError)
                messageBg.color = new Color(0.8f, 0.2f, 0.2f, 0.9f); // Red warning
            else
                messageBg.color = new Color(0.1f, 0.1f, 0.1f, 0.8f); 
        }


        CancelInvoke("HideSystemMessage"); 
        
        // 3. Automatic disappearance
        if (autoHide)
        {
            Invoke("HideSystemMessage", 3.0f);
        }
    }

    public void HideSystemMessage()
    {
        if (messagePanel != null)
            messagePanel.SetActive(false);
    }
}