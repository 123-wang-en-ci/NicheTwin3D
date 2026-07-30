using UnityEngine;

public class CameraOrbit : MonoBehaviour
{
    public Transform target; 
    public float distance = 50.0f;
    public float xSpeed = 120.0f;
    public float ySpeed = 120.0f;

    [Header("Zoom Settings")]
    public float zoomSpeed = 300.0f;
    public float minDistance = 2.0f;
    public float maxDistance = 500.0f;

    private float x = 0.0f;
    private float y = 0.0f;

    void Start()
    {
        Vector3 angles = transform.eulerAngles;
        x = angles.y;
        y = angles.x;
        if (y > 180f) y -= 360f;

        // If there is no target, create a target point directly in front of the camera's initial position
        if (target == null)
        {
            GameObject t = new GameObject("CamTarget");
            // Maintain the camera's exact starting position and rotation from the Unity Scene
            t.transform.position = transform.position + transform.forward * distance; 
            target = t.transform;
        }
        else
        {
            // If target is assigned, calculate actual distance so camera doesn't snap
            distance = Vector3.Distance(transform.position, target.position);
        }
    }

    void LateUpdate()
    {
        bool isPointerOverUI = IsPointerOverInteractiveUI();

        // 1. Right click down: sync rotation angles and target position
        if (target && Input.GetMouseButtonDown(1) && !isPointerOverUI)
        {
            Vector3 angles = transform.eulerAngles;
            x = angles.y;
            y = angles.x;
            if (y > 180f) y -= 360f;
            target.position = transform.position + transform.forward * distance;
        }

        // 2. Mouse scroll wheel zoom (works directly at any time without holding right-click)
        float scroll = Input.GetAxis("Mouse ScrollWheel");
        if (Mathf.Abs(scroll) > 0.0001f && !isPointerOverUI)
        {
            float zoomDelta = scroll * zoomSpeed;
            
            // Advance target position forward when zooming in closer than minDistance
            if (zoomDelta > 0 && (distance - zoomDelta) < minDistance)
            {
                float excess = zoomDelta - (distance - minDistance);
                distance = minDistance;
                if (target != null)
                {
                    target.position += transform.forward * excess;
                }
            }
            else
            {
                distance -= zoomDelta;
                distance = Mathf.Clamp(distance, minDistance, maxDistance);
            }
        }

        // 3. Right mouse drag rotation
        if (target && Input.GetMouseButton(1) && !isPointerOverUI)
        {
            x += Input.GetAxis("Mouse X") * xSpeed * 0.02f;
            y -= Input.GetAxis("Mouse Y") * ySpeed * 0.02f;
            y = Mathf.Clamp(y, -89f, 89f);
        }

        // 4. Check if WASD keys are being pressed to move the camera
        bool isWASDPressed = Input.GetKey(KeyCode.W) || Input.GetKey(KeyCode.A) || 
                             Input.GetKey(KeyCode.S) || Input.GetKey(KeyCode.D) || 
                             Input.GetKey(KeyCode.Q) || Input.GetKey(KeyCode.E);

        if (target)
        {
            if (isWASDPressed)
            {
                // WASD is moving camera position in Update(), so sync target position AND rotation angles (x, y)
                target.position = transform.position + transform.forward * distance;
                Vector3 angles = transform.eulerAngles;
                x = angles.y;
                y = angles.x;
                if (y > 180f) y -= 360f;
            }
            else
            {
                // Continuously maintain camera position & rotation in LateUpdate to prevent position rollbacks
                Quaternion rotation = Quaternion.Euler(y, x, 0);
                Vector3 position = rotation * new Vector3(0.0f, 0.0f, -distance) + target.position;

                transform.rotation = rotation;
                transform.position = position;
            }
        }
    }

    private bool IsPointerOverInteractiveUI()
    {
        var eventSystem = UnityEngine.EventSystems.EventSystem.current;
        if (eventSystem == null) return false;

        if (!eventSystem.IsPointerOverGameObject()) return false;

        var pointerEventData = new UnityEngine.EventSystems.PointerEventData(eventSystem)
        {
            position = Input.mousePosition
        };

        var results = new System.Collections.Generic.List<UnityEngine.EventSystems.RaycastResult>();
        eventSystem.RaycastAll(pointerEventData, results);

        foreach (var result in results)
        {
            if (result.gameObject != null)
            {
                string objName = result.gameObject.name.ToLower();
                if (objName.Contains("tooltip") || objName.Contains("hover"))
                    continue;

                // Check if it's a real interactive UI element (Button, ScrollRect, InputField, Dropdown)
                if (result.gameObject.GetComponentInParent<UnityEngine.UI.Selectable>() != null ||
                    result.gameObject.GetComponentInParent<UnityEngine.UI.ScrollRect>() != null ||
                    result.gameObject.GetComponentInParent<TMPro.TMP_InputField>() != null)
                {
                    return true;
                }
            }
        }

        return false;
    }
}