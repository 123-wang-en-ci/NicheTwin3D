using UnityEngine;
using TMPro;
using UnityEngine.UI;

public class LocalizedText : MonoBehaviour
{
    [Header("Localization Key")]
    public string localizationKey;

    private TextMeshProUGUI tmpText;
    private Text uiText;
    private bool isSubscribed = false;

    void Awake()
    {
        tmpText = GetComponentInChildren<TextMeshProUGUI>(true);
        uiText = GetComponentInChildren<Text>(true);
    }

    void OnEnable()
    {
        TrySubscribe();
        UpdateText();
    }

    void OnDisable()
    {
        Unsubscribe();
    }

    void Start()
    {
        TrySubscribe();
        UpdateText();
    }

    private void TrySubscribe()
    {
        if (!isSubscribed && LocalizationManager.Instance != null)
        {
            LocalizationManager.Instance.OnLanguageChanged += UpdateText;
            isSubscribed = true;
        }
    }

    private void Unsubscribe()
    {
        if (isSubscribed && LocalizationManager.Instance != null)
        {
            LocalizationManager.Instance.OnLanguageChanged -= UpdateText;
            isSubscribed = false;
        }
    }

    public void UpdateText()
    {
        TrySubscribe();

        if (string.IsNullOrEmpty(localizationKey)) return;
        if (LocalizationManager.Instance == null) return;

        if (tmpText == null && uiText == null)
        {
            tmpText = GetComponentInChildren<TextMeshProUGUI>(true);
            uiText = GetComponentInChildren<Text>(true);
        }

        string translated = LocalizationManager.Instance.GetText(localizationKey);

        if (tmpText != null)
        {
            tmpText.text = translated;
        }
        else if (uiText != null)
        {
            uiText.text = translated;
        }
    }
}
