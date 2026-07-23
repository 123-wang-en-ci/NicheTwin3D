using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Networking;
using System.Collections;
using System.Collections.Generic;
using TMPro;

[System.Serializable]
public class SearchGenesResponse
{
    public string status;
    public List<string> results;
}

public class GeneSearchAutoComplete : MonoBehaviour
{
    [Header("UI Components")]
    [Tooltip("Drag the TMP_InputField here.")]
    public TMP_InputField inputField;

    [Tooltip("Drag the ScrollView suggestion panel here.")]
    public GameObject suggestionPanel;

    [Tooltip("Drag the parent Content node of the ScrollView here.")]
    public Transform suggestionContainer;

    [Tooltip("Prefab for suggestion items (must contain a Button and a Text/TMP_Text component).")]
    public GameObject suggestionItemPrefab;

    [Header("Scripts")]
    [Tooltip("Optional: Drag the UI_GeneSearch script here to trigger auto-search on selection.")]
    public UI_GeneSearch uiGeneSearch;

    [Header("Settings")]
    public string searchUrl = "http://127.0.0.1:8000/search_genes?query=";
    public float debounceTime = 0.2f;

    private Coroutine searchCoroutine;
    private bool isSelecting = false; // Flag to prevent triggering search requests when programmatically setting text

    void Start()
    {
        if (inputField == null)
        {
            inputField = GetComponent<TMP_InputField>();
        }

        if (inputField != null)
        {
            inputField.onValueChanged.AddListener(OnInputChanged);
        }
        else
        {
            Debug.LogError("[AutoComplete] InputField is not assigned!");
        }

        if (suggestionPanel != null)
        {
            suggestionPanel.SetActive(false);
        }
    }

    void OnInputChanged(string text)
    {
        if (isSelecting) return;

        if (searchCoroutine != null)
        {
            StopCoroutine(searchCoroutine);
        }

        if (string.IsNullOrEmpty(text) || string.IsNullOrEmpty(text.Trim()))
        {
            HideSuggestions();
            return;
        }

        searchCoroutine = StartCoroutine(SearchGenesDebounced(text));
    }

    IEnumerator SearchGenesDebounced(string query)
    {
        yield return new WaitForSeconds(debounceTime);

        string escapedQuery = UnityWebRequest.EscapeURL(query.Trim());
        string url = searchUrl + escapedQuery;

        using (UnityWebRequest webRequest = UnityWebRequest.Get(url))
        {
            yield return webRequest.SendWebRequest();

            if (webRequest.result == UnityWebRequest.Result.Success)
            {
                string jsonText = webRequest.downloadHandler.text;
                SearchGenesResponse response = JsonUtility.FromJson<SearchGenesResponse>(jsonText);

                if (response != null && response.status == "success" && response.results != null && response.results.Count > 0)
                {
                    UpdateSuggestions(response.results);
                }
                else
                {
                    HideSuggestions();
                }
            }
            else
            {
                Debug.LogWarning("[AutoComplete] Search request failed: " + webRequest.error);
                HideSuggestions();
            }
        }
    }

    void UpdateSuggestions(List<string> results)
    {
        if (suggestionPanel == null || suggestionContainer == null || suggestionItemPrefab == null) return;

        // Clear existing suggestions
        foreach (Transform child in suggestionContainer)
        {
            Destroy(child.gameObject);
        }

        // Instantiate matching suggestions
        foreach (string matchedGene in results)
        {
            GameObject item = Instantiate(suggestionItemPrefab, suggestionContainer);
            
            // Try to set text using TMP_Text first, then fallback to standard Text
            TMP_Text tmpText = item.GetComponentInChildren<TMP_Text>();
            if (tmpText != null)
            {
                tmpText.text = matchedGene;
            }
            else
            {
                Text stdText = item.GetComponentInChildren<Text>();
                if (stdText != null)
                {
                    stdText.text = matchedGene;
                }
            }

            // Bind click event
            Button btn = item.GetComponent<Button>();
            if (btn != null)
            {
                string localGeneName = matchedGene; // Capture variable for closure
                btn.onClick.AddListener(() => OnSuggestionSelected(localGeneName));
            }
        }

        suggestionPanel.SetActive(true);
    }

    void OnSuggestionSelected(string geneName)
    {
        isSelecting = true;
        if (inputField != null)
        {
            inputField.text = geneName;
        }
        isSelecting = false;

        HideSuggestions();
    }

    public void HideSuggestions()
    {
        if (suggestionPanel != null)
        {
            suggestionPanel.SetActive(false);
        }
    }

    // Hide suggestions panel when clicking outside or input field loses focus (optional)
    void OnDisable()
    {
        HideSuggestions();
    }
}
