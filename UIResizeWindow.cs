using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

public class UIResizeWindow : MonoBehaviour, IPointerDownHandler, IDragHandler
{
    public enum ResizeEdge
    {
        AutoDetect,
        Left,
        Right,
        Top,
        Bottom,
        TopLeft,
        TopRight,
        BottomLeft,
        BottomRight
    }

    [Header("Resize Configuration")]
    public ResizeEdge resizeEdge = ResizeEdge.AutoDetect;
    public RectTransform targetWindow;

    [Header("Size Constraints")]
    public Vector2 minSize = new Vector2(350f, 250f);
    public Vector2 maxSize = new Vector2(1400f, 900f);

    [Header("Edge Hitbox Thickness (px)")]
    public float edgeThickness = 10f;

    private Canvas canvas;

    void Awake()
    {
        canvas = GetComponentInParent<Canvas>();

        // If this is a child edge handle with an assigned edge, do not auto-generate sub-handles
        if (resizeEdge != ResizeEdge.AutoDetect)
        {
            return;
        }

        // Only the main window itself should auto-generate handles
        if (targetWindow == null)
        {
            if (transform.parent != null && transform.parent.GetComponent<Canvas>() == null)
            {
                targetWindow = transform.parent.GetComponent<RectTransform>();
            }
            else
            {
                targetWindow = GetComponent<RectTransform>();
            }
        }

        // Only auto-generate borders if we are attached to the target window itself
        if (resizeEdge == ResizeEdge.AutoDetect && targetWindow == GetComponent<RectTransform>())
        {
            CreateAutoBorderHandles();
        }
    }

    private void CreateAutoBorderHandles()
    {
        // 4 Edges
        CreateEdgeHandle("Border_L", ResizeEdge.Left, new Vector2(0, 0), new Vector2(0, 1), new Vector2(0.5f, 0.5f), new Vector2(edgeThickness, 0));
        CreateEdgeHandle("Border_R", ResizeEdge.Right, new Vector2(1, 0), new Vector2(1, 1), new Vector2(0.5f, 0.5f), new Vector2(edgeThickness, 0));
        CreateEdgeHandle("Border_T", ResizeEdge.Top, new Vector2(0, 1), new Vector2(1, 1), new Vector2(0.5f, 0.5f), new Vector2(0, edgeThickness));
        CreateEdgeHandle("Border_B", ResizeEdge.Bottom, new Vector2(0, 0), new Vector2(1, 0), new Vector2(0.5f, 0.5f), new Vector2(0, edgeThickness));

        // 4 Corners
        float cornerSize = edgeThickness * 1.5f;
        CreateEdgeHandle("Corner_TL", ResizeEdge.TopLeft, new Vector2(0, 1), new Vector2(0, 1), new Vector2(0.5f, 0.5f), new Vector2(cornerSize, cornerSize));
        CreateEdgeHandle("Corner_TR", ResizeEdge.TopRight, new Vector2(1, 1), new Vector2(1, 1), new Vector2(0.5f, 0.5f), new Vector2(cornerSize, cornerSize));
        CreateEdgeHandle("Corner_BL", ResizeEdge.BottomLeft, new Vector2(0, 0), new Vector2(0, 0), new Vector2(0.5f, 0.5f), new Vector2(cornerSize, cornerSize));
        CreateEdgeHandle("Corner_BR", ResizeEdge.BottomRight, new Vector2(1, 0), new Vector2(1, 0), new Vector2(0.5f, 0.5f), new Vector2(cornerSize, cornerSize));
    }

    private void CreateEdgeHandle(string handleName, ResizeEdge edge, Vector2 anchorMin, Vector2 anchorMax, Vector2 pivot, Vector2 size)
    {
        if (transform.Find(handleName) != null) return;

        GameObject handleObj = new GameObject(handleName, typeof(RectTransform), typeof(Image));
        handleObj.transform.SetParent(transform, false);

        Image img = handleObj.GetComponent<Image>();
        img.color = new Color(0, 0, 0, 0); // Transparent raycast hitbox
        img.raycastTarget = true;

        RectTransform rt = handleObj.GetComponent<RectTransform>();
        rt.anchorMin = anchorMin;
        rt.anchorMax = anchorMax;
        rt.pivot = pivot;
        rt.anchoredPosition = Vector2.zero;
        rt.sizeDelta = size;

        UIResizeWindow resizer = handleObj.AddComponent<UIResizeWindow>();
        resizer.resizeEdge = edge;
        resizer.targetWindow = targetWindow;
        resizer.minSize = minSize;
        resizer.maxSize = maxSize;
    }

    public void OnPointerDown(PointerEventData eventData)
    {
        if (targetWindow != null)
        {
            targetWindow.SetAsLastSibling();
        }
    }

    public void OnDrag(PointerEventData eventData)
    {
        if (targetWindow == null) return;

        float scaleFactor = (canvas != null && canvas.scaleFactor > 0) ? canvas.scaleFactor : 1.0f;
        Vector2 delta = eventData.delta / scaleFactor;

        float currentWidth = targetWindow.rect.width;
        float currentHeight = targetWindow.rect.height;

        float deltaWidth = 0f;
        float deltaHeight = 0f;

        // Horizontal resize calculation
        if (resizeEdge == ResizeEdge.Right || resizeEdge == ResizeEdge.TopRight || resizeEdge == ResizeEdge.BottomRight)
        {
            float newW = Mathf.Clamp(currentWidth + delta.x, minSize.x, maxSize.x);
            deltaWidth = newW - currentWidth;
        }
        else if (resizeEdge == ResizeEdge.Left || resizeEdge == ResizeEdge.TopLeft || resizeEdge == ResizeEdge.BottomLeft)
        {
            float newW = Mathf.Clamp(currentWidth - delta.x, minSize.x, maxSize.x);
            deltaWidth = newW - currentWidth;
        }

        // Vertical resize calculation
        if (resizeEdge == ResizeEdge.Top || resizeEdge == ResizeEdge.TopLeft || resizeEdge == ResizeEdge.TopRight)
        {
            float newH = Mathf.Clamp(currentHeight + delta.y, minSize.y, maxSize.y);
            deltaHeight = newH - currentHeight;
        }
        else if (resizeEdge == ResizeEdge.Bottom || resizeEdge == ResizeEdge.BottomLeft || resizeEdge == ResizeEdge.BottomRight)
        {
            float newH = Mathf.Clamp(currentHeight - delta.y, minSize.y, maxSize.y);
            deltaHeight = newH - currentHeight;
        }

        if (Mathf.Approximately(deltaWidth, 0f) && Mathf.Approximately(deltaHeight, 0f)) return;

        float finalWidth = currentWidth + deltaWidth;
        float finalHeight = currentHeight + deltaHeight;

        // Safely set width and height
        targetWindow.SetSizeWithCurrentAnchors(RectTransform.Axis.Horizontal, finalWidth);
        targetWindow.SetSizeWithCurrentAnchors(RectTransform.Axis.Vertical, finalHeight);

        // Calculate anchored position offset to lock opposite edges in place while dragging
        Vector2 pivot = targetWindow.pivot;
        Vector2 posOffset = Vector2.zero;

        if (resizeEdge == ResizeEdge.Right || resizeEdge == ResizeEdge.TopRight || resizeEdge == ResizeEdge.BottomRight)
        {
            posOffset.x = deltaWidth * (1.0f - pivot.x);
        }
        else if (resizeEdge == ResizeEdge.Left || resizeEdge == ResizeEdge.TopLeft || resizeEdge == ResizeEdge.BottomLeft)
        {
            posOffset.x = -deltaWidth * pivot.x;
        }

        if (resizeEdge == ResizeEdge.Top || resizeEdge == ResizeEdge.TopLeft || resizeEdge == ResizeEdge.TopRight)
        {
            posOffset.y = deltaHeight * (1.0f - pivot.y);
        }
        else if (resizeEdge == ResizeEdge.Bottom || resizeEdge == ResizeEdge.BottomLeft || resizeEdge == ResizeEdge.BottomRight)
        {
            posOffset.y = -deltaHeight * pivot.y;
        }

        targetWindow.anchoredPosition += posOffset;

        // Force canvas & layout update
        Canvas.ForceUpdateCanvases();
        if (HelpManager.Instance != null && HelpManager.Instance.helpPanel != null && HelpManager.Instance.helpPanel.activeSelf)
        {
            HelpManager.Instance.RefreshHelpContent();
        }
    }
}
