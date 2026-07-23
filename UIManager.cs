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

 
    public void ShowCellDetails(string id, string cellType, Vector2 coordinates, float expression, float avgExpression)
    {

        infoTitleText.text = ":: SINGLE  CELL  ANALYSIS ::";

        string content = "";

        content += $"<color=#FFFFFF>ID Ref:</color>\n";
        content += $"  <b><color=#FFFFFF>{id}</color></b>\n";

        content += $"<color=#FFFFFF>Cell Type:</color>\n";
        content += $"  <b><color=#00FF00>{cellType}</color></b>\n"; // Green highlight

        content += $"<color=#FFFFFF>Spatial Coords (um):</color>\n";
        content += $"  X: <b>{coordinates.x:F2}</b>  Y: <b>{coordinates.y:F2}</b>\n"; // F2 Keep two decimal places

        content += $"<color=#FFFFFF>Gene Expression:</color>\n";
        
        // Set color according to expression level (Red=High, Blue=Low)
        string exprColor = expression > 0.5f ? "#FF4444" : "#4444FF";
        content += $"  Value: <b><color={exprColor}>{expression:F4}</color></b>\n"; // F4 Keep four decimal places to show precision
        
        // Calculate deviation percentage
        float deviation = ((expression - avgExpression) / avgExpression) * 100f;
        string sign = deviation >= 0 ? "+" : "";
        content += $"  Dev:   <size=80%>{sign}{deviation:F1}% vs Avg</size>";

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