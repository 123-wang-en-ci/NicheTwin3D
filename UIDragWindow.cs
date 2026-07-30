using UnityEngine;
using UnityEngine.EventSystems;

public class UIDragWindow : MonoBehaviour, IPointerDownHandler, IDragHandler
{
    [Header("Target Window to Move")]
    [Tooltip("Drag the parent HelpPanel RectTransform here. If left empty, it will automatically find the parent window.")]
    public RectTransform targetWindow;

    private Canvas canvas;

    void Awake()
    {
        // If targetWindow is empty or accidentally set to TitleBar itself, find parent window
        if (targetWindow == null || targetWindow == GetComponent<RectTransform>())
        {
            if (transform.parent != null)
            {
                targetWindow = transform.parent.GetComponent<RectTransform>();
            }
            else
            {
                targetWindow = GetComponent<RectTransform>();
            }
        }

        // Find parent Canvas for proper scaling
        canvas = GetComponentInParent<Canvas>();
    }

    public void OnPointerDown(PointerEventData eventData)
    {
        // Bring window to front when clicked
        if (targetWindow != null)
        {
            targetWindow.SetAsLastSibling();
        }
    }

    public void OnDrag(PointerEventData eventData)
    {
        if (targetWindow == null) return;

        float scaleFactor = (canvas != null && canvas.scaleFactor > 0) ? canvas.scaleFactor : 1.0f;
        targetWindow.anchoredPosition += eventData.delta / scaleFactor;
    }
}
