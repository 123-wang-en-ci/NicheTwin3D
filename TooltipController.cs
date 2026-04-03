using UnityEngine;
using TMPro; 

public class TooltipController : MonoBehaviour
{
    [Header("UI 组件引用")]
    public GameObject tooltipObj;       
    public TextMeshProUGUI idText;      
    public RectTransform canvasRect;   

    [Header("设置")]
    public Vector2 offset = new Vector2(15f, -15f); // Mouse offset to prevent blocking the mouse

    void Start()
    {
        // Make sure to hide at the beginning of the game
        if (tooltipObj != null) tooltipObj.SetActive(false);
    }

    void Update()
    {
        //  Emission ray detection
        Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
        RaycastHit hit;

        // Be sure to give your cells Prefab plus Collider for detection!
        if (Physics.Raycast(ray, out hit))
        {
            //  If you hit an object (show Tooltip)
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