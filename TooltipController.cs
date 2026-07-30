using UnityEngine;
using TMPro; 

public class TooltipController : MonoBehaviour
{
    [Header("UI component reference")]
    public GameObject tooltipObj;       
    public TextMeshProUGUI idText;     
    public RectTransform canvasRect;   

    [Header("set up")]
    public Vector2 offset = new Vector2(15f, -15f); 

    void Start()
    {
        // Make sure to hide when the game starts
        if (tooltipObj != null)
        {
            tooltipObj.SetActive(false);

            // Ensure tooltip NEVER blocks mouse raycasts, UI pointer events, or camera zoom/rotate
            CanvasGroup cg = tooltipObj.GetComponent<CanvasGroup>();
            if (cg == null) cg = tooltipObj.AddComponent<CanvasGroup>();
            cg.blocksRaycasts = false;
            cg.interactable = false;

            foreach (var graphic in tooltipObj.GetComponentsInChildren<UnityEngine.UI.Graphic>(true))
            {
                graphic.raycastTarget = false;
            }
        }
    }

    void Update()
    {
        // 1. If mouse is hovering over any UI panel (LeftNavPanel, RightContentSlot, buttons, etc.), hide cell ID tooltip
        if (UnityEngine.EventSystems.EventSystem.current != null && UnityEngine.EventSystems.EventSystem.current.IsPointerOverGameObject())
        {
            if (tooltipObj != null && tooltipObj.activeSelf)
            {
                tooltipObj.SetActive(false);
            }
            return;
        }

        // 2. If comparison mode is active and mouse is on the left half, hide tooltip and return
        CellComparisonManager comp = FindObjectOfType<CellComparisonManager>();
        if (comp != null && comp.isComparisonMode && Input.mousePosition.x < Screen.width * 0.5f)
        {
            if (tooltipObj != null && tooltipObj.activeSelf)
            {
                tooltipObj.SetActive(false);
            }
            return;
        }

        Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
        RaycastHit hit;

        if (Physics.Raycast(ray, out hit))
        {
            if (tooltipObj != null && !tooltipObj.activeSelf)
            {
                tooltipObj.SetActive(true);
            }

            if (idText != null)
            {
                idText.text = hit.transform.name;
            }

            Vector2 localPoint;
            RectTransformUtility.ScreenPointToLocalPointInRectangle(
                canvasRect,
                Input.mousePosition,
                null, 
                out localPoint
            );

            // set position + offset
            tooltipObj.transform.localPosition = localPoint + offset;
        }
        else
        {
            if (tooltipObj != null && tooltipObj.activeSelf)
            {
                tooltipObj.SetActive(false);
            }
        }
    }
}