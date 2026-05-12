using UnityEngine;
using TMPro; // Reference TextMeshPro

public class TooltipController : MonoBehaviour
{
    [Header("UI component reference")]
    public GameObject tooltipObj; // Drag in the Tooltip object
public TextMeshProUGUI idText; // Drag in the Txt_ID object
    public RectTransform canvasRect; // Drag into the Canvas object

    [Header("Settings")]
    public Vector2 offset = new Vector2(15f, -15f); //Mouse offset to prevent blocking the mouse

    void Start()
    {
        // Make sure to hide when the game starts
        if (tooltipObj != null) tooltipObj.SetActive(false);
    }

    void Update()
    {
        // 1. Emission ray detection
        Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
        RaycastHit hit;

        // Note: You must add Collider to your cell Prefab to detect it!
        if (Physics.Raycast(ray, out hit))
        {
            // 2. If an object is hit (display Tooltip)
            if (tooltipObj != null && !tooltipObj.activeSelf)
            {
                tooltipObj.SetActive(true);
            }

            // 3. Update text (read object name as ID)
            if (idText != null)
            {
                idText.text = hit.transform.name;
            }

            // 4. Let the UI follow the mouse movement
            //Convert the mouse coordinates on the screen to the coordinates inside the Canvas
            Vector2 localPoint;
            RectTransformUtility.ScreenPointToLocalPointInRectangle(
                canvasRect,
                Input.mousePosition,
                null, // If Canvas is in Overlay mode, fill in null here
                out localPoint
            );

            //Set position + offset
            tooltipObj.transform.localPosition = localPoint + offset;
        }
        else
        {
            // 5. If no object is hit (hide Tooltip)
            if (tooltipObj != null && tooltipObj.activeSelf)
            {
                tooltipObj.SetActive(false);
            }
        }
    }
}