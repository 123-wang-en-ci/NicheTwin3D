using UnityEngine;
using TMPro;

public class UI_GeneSearch : MonoBehaviour
{
    public TMP_InputField inputField;
    public InteractionManager interactionManager;

    public void OnSearchClicked()
    {
        string geneName = "";
        if (inputField != null) geneName = inputField.text.Trim();

        if (!string.IsNullOrEmpty(geneName))
        {
            Debug.Log($"[UI] �û�������������: {geneName}");
            interactionManager.RequestGeneSwitch(geneName);
        }
        else
        {
            Debug.LogWarning("[UI] �����Ϊ�գ�");
        }
    }

 
    // ---------------------------------------------------------
    public void OnPreviousViewClicked()
    {
        Debug.Log("[UI] ���󷵻�Ĭ����ͼ (View Only, �����Ŷ�)");

        if (inputField != null) inputField.text = "";

        interactionManager.RequestGeneSwitch("RESET");
    }
}