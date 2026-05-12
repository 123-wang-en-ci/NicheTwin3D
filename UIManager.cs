using UnityEngine;
using TMPro;
using UnityEngine.UI; // Reference Image component

public class UIManager : MonoBehaviour
{
    [Header("UI Component")]
    public TextMeshProUGUI infoTitleText;
    public TextMeshProUGUI infoBodyText;

    [Header("System Message Component")]
    public GameObject messagePanel; // Drag into SystemMessagePanel
    public TextMeshProUGUI messageText; // Drag in Txt_Message
    public Image messageBg; // Drag into SystemMessagePanel (to change the background color)

    public static UIManager Instance;


    void Awake()
    {
        Instance = this;
    }

    // Rigorous display method: receive specific data fields instead of a bunch of messy strings
    public void ShowCellDetails(string id, string cellType, Vector2 coordinates, float expression, float avgExpression)
    {
        // 1. Set title
        infoTitleText.text = ":: SINGLE  CELL  ANALYSIS ::";

// 2. Format content (use Rich Text)
        // <color=#888888> is the gray label, <b> is the bold value
        string content = "";

        content += $"<color=#FFFFFF>ID Ref:</color>\n";
        content += $"  <b><color=#FFFFFF>{id}</color></b>\n";

        content += $"<color=#FFFFFF>Cell Type:</color>\n";
        content += $" <b><color=#00FF00>{cellType}</color></b>\n"; // Green highlight type

        content += $"<color=#FFFFFF>Spatial Coords (um):</color>\n";
        content += $"

        content += $"<color=#FFFFFF>Gene Expression:</color>\n";
        
        //Change color according to expression level (red=high, blue=low)
        string exprColor = expression > 0.5f ? "#FF4444" : "#4444FF";
        content += $" Value: <b><color={exprColor}>{expression:F4}</color></b>\n"; // F4 retains four decimal places to reflect precision
        
        // Calculate deviation percentage
        float deviation = ((expression - avgExpression) / avgExpression) * 100f;
        string sign = deviation >= 0 ? "+" : "";
        content += $"  Dev:   <size=80%>{sign}{deviation:F1}% vs Avg</size>";

        infoBodyText.text = content;
    }
    // Method to display system prompt messages in the center of the screen
    public void ShowSystemMessage(string msg, bool isError)
    {
        if (messagePanel == null) return;

        // 1. Set content
        messagePanel.SetActive(true);
        messageText.text = msg;

        // 2. Change the color according to the status (for example: red background for errors, green or black background for success)
        if (messageBg != null)
        {
            if (isError)
                messageBg.color = new Color(0.8f, 0.2f, 0.2f, 0.9f); // Red warning
            else
                messageBg.color = new Color(0.1f, 0.1f, 0.1f, 0.8f); // Black prompt
        }

        // 3. Automatically disappear (after 3 seconds)
        CancelInvoke("HideSystemMessage"); // If the previous one has not disappeared, cancel it first to prevent flickering
        Invoke("HideSystemMessage", 3.0f);
    }

    void HideSystemMessage()
    {
        if (messagePanel != null)
            messagePanel.SetActive(false);
    }
}