using UnityEngine;
using TMPro;

public class TooltipController : MonoBehaviour
{

    public GameObject tooltipObj;       
    public TextMeshProUGUI idText;      
    public RectTransform canvasRect;    

 
    public Vector2 offset = new Vector2(15f, -15f); 

    void Start()
    {

        if (tooltipObj != null) tooltipObj.SetActive(false);
    }

    void Update()
    {

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