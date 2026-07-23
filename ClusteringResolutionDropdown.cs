using UnityEngine;
using UnityEngine.UI;
using UnityEngine.EventSystems;
using System.Collections;
using System.Collections.Generic;
using TMPro;

public class ClusteringResolutionDropdown : MonoBehaviour, IPointerClickHandler, ISelectHandler
{
    [Header("UI Components")]
    [Tooltip("Drag the TMP_InputField for cluster resolution (K-value) here.")]
    public TMP_InputField inputField;

    [Tooltip("Drag the Dropdown list container panel here.")]
    public GameObject dropdownPanel;

    [Tooltip("Drag the 5 Option Buttons here.")]
    public List<Button> optionButtons = new List<Button>();

    void Start()
    {
        if (inputField == null)
        {
            inputField = GetComponent<TMP_InputField>();
        }

        if (dropdownPanel != null)
        {
            dropdownPanel.SetActive(false);
        }

        // Bind onClick events for each option button
        for (int i = 0; i < optionButtons.Count; i++)
        {
            if (optionButtons[i] != null)
            {
                int index = i; // Local copy for closure
                optionButtons[index].onClick.AddListener(() => OnOptionClicked(optionButtons[index]));
            }
        }
    }

    // Triggered when clicked by pointer
    public void OnPointerClick(PointerEventData eventData)
    {
        ShowDropdown();
    }

    // Triggered when selected via keyboard/navigation
    public void OnSelect(BaseEventData eventData)
    {
        ShowDropdown();
    }

    public void ShowDropdown()
    {
        if (dropdownPanel != null)
        {
            dropdownPanel.SetActive(true);
            dropdownPanel.transform.SetAsLastSibling(); 
        }
    }

    public void HideDropdown()
    {
        if (dropdownPanel != null)
        {
            dropdownPanel.SetActive(false);
        }
    }

    void OnOptionClicked(Button clickedButton)
    {
        // Try getting text from TextMeshPro, fallback to standard Text
        TMP_Text tmpText = clickedButton.GetComponentInChildren<TMP_Text>();
        string value = "";

        if (tmpText != null)
        {
            value = tmpText.text;
        }
        else
        {
            Text stdText = clickedButton.GetComponentInChildren<Text>();
            if (stdText != null)
            {
                value = stdText.text;
            }
        }

        if (!string.IsNullOrEmpty(value) && inputField != null)
        {
            inputField.text = value.Trim();
        }

        HideDropdown();
    }

    void Update()
    {
        // Close the dropdown when clicking outside both the input field and the dropdown panel
        if (dropdownPanel != null && dropdownPanel.activeSelf && Input.GetMouseButtonDown(0))
        {
            RectTransform inputRect = inputField.GetComponent<RectTransform>();
            RectTransform panelRect = dropdownPanel.GetComponent<RectTransform>();

            if (inputRect != null && panelRect != null)
            {
                if (!RectTransformUtility.RectangleContainsScreenPoint(inputRect, Input.mousePosition) &&
                    !RectTransformUtility.RectangleContainsScreenPoint(panelRect, Input.mousePosition))
                {
                    // Delay slightly to prevent intercepting button clicks
                    StartCoroutine(HideDropdownDelayed());
                }
            }
        }
    }

    IEnumerator HideDropdownDelayed()
    {
        yield return new WaitForSeconds(0.15f);
        HideDropdown();
    }

    void OnDisable()
    {
        HideDropdown();
    }
}
