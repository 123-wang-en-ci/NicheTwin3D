using UnityEngine;
using TMPro;
using UnityEngine.UI; 

public class UIManager : MonoBehaviour
{

    public TextMeshProUGUI infoTitleText;
    public TextMeshProUGUI infoBodyText;


    public GameObject messagePanel; 
    public TextMeshProUGUI messageText; 
    public Image messageBg; // 

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
        content += $"  <b><color=#00FF00>{cellType}</color></b>\n"; 

        content += $"<color=#FFFFFF>Spatial Coords (um):</color>\n";
        content += $"  X: <b>{coordinates.x:F2}</b>  Y: <b>{coordinates.y:F2}</b>\n"; 

        content += $"<color=#FFFFFF>Gene Expression:</color>\n";
        
   
        string exprColor = expression > 0.5f ? "#FF4444" : "#4444FF";
        content += $"  Value: <b><color={exprColor}>{expression:F4}</color></b>\n"; 

        float deviation = ((expression - avgExpression) / avgExpression) * 100f;
        string sign = deviation >= 0 ? "+" : "";
        content += $"  Dev:   <size=80%>{sign}{deviation:F1}% vs Avg</size>";

        infoBodyText.text = content;
    }

    public void ShowSystemMessage(string msg, bool isError)
    {
        if (messagePanel == null) return;

        messagePanel.SetActive(true);
        messageText.text = msg;

       
        if (messageBg != null)
        {
            if (isError)
                messageBg.color = new Color(0.8f, 0.2f, 0.2f, 0.9f);
            else
                messageBg.color = new Color(0.1f, 0.1f, 0.1f, 0.8f); 
        }

        CancelInvoke("HideSystemMessage"); 
        Invoke("HideSystemMessage", 3.0f);
    }

    void HideSystemMessage()
    {
        if (messagePanel != null)
            messagePanel.SetActive(false);
    }
}