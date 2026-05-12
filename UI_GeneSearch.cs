using UnityEngine;
using TMPro;

public class UI_GeneSearch : MonoBehaviour
{
    public TMP_InputField inputField;
    public InteractionManager interactionManager;

    // Bind to the "Search" button
    public void OnSearchClicked()
    {
        string geneName = "";
        if (inputField != null) geneName = inputField.text.Trim();

        if (!string.IsNullOrEmpty(geneName))
        {
            Debug.Log($"[UI] User requested to search for genes: {geneName}");
            interactionManager.RequestGeneSwitch(geneName);
        }
        else
        {
            Debug.LogWarning("[UI] input box is empty!");
        }
    }

    public void OnPreviousViewClicked()
    {
        Debug.Log("[UI] Request to return to default view (View Only, retain disturbance)");
        // Clear the input box to let the user know that the specific gene is not currently being searched.
        if (inputField != null) inputField.text = "";

        //Send RESET signal
        interactionManager.RequestGeneSwitch("RESET");
    }
}