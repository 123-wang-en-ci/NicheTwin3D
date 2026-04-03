using UnityEngine;

public class CameraOrbit : MonoBehaviour
{
    public Transform target; // Drag in Cell_Container (create an empty object and place it in the center of the cell)
    public float distance = 50.0f;
    public float xSpeed = 120.0f;
    public float ySpeed = 120.0f;

    private float x = 0.0f;
    private float y = 0.0f;

    void Start()
    {
        Vector3 angles = transform.eulerAngles;
        x = angles.y;
        y = angles.x;

        // If there is no goal, create a temporary center point
        if (target == null)
        {
            GameObject t = new GameObject("CamTarget");
            t.transform.position = new Vector3(50, 0, 50); 
            target = t.transform;
        }
    }

    void LateUpdate()
    {
        // Hold down the right mouse button to rotate
        if (target && Input.GetMouseButton(1))
        {
            x += Input.GetAxis("Mouse X") * xSpeed * 0.02f;
            y -= Input.GetAxis("Mouse Y") * ySpeed * 0.02f;

            Quaternion rotation = Quaternion.Euler(y, x, 0);
            Vector3 position = rotation * new Vector3(0.0f, 0.0f, -distance) + target.position;

            transform.rotation = rotation;
            transform.position = position;
        }

        // Scroll wheel zoom
        distance -= Input.GetAxis("Mouse ScrollWheel") * 10f;
    }
}