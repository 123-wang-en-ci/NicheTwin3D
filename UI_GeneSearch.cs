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
            Debug.Log($"[UI] User request to search for genes: {geneName}");
            interactionManager.RequestGeneSwitch(geneName);
        }
        else
        {
            Debug.LogWarning("[UI] The input box is empty!");
        }
    }

    public void OnPreviousViewClicked()
    {
        Debug.Log("[UI] Request to return to default view ");

        if (inputField != null) inputField.text = "";

        interactionManager.RequestGeneSwitch("RESET");
    }
}