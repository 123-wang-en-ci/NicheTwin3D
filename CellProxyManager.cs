using UnityEngine;
using System.Collections.Generic;

/// <summary>
/// Agent Collider Manager - Interaction Detection for GPU Instancing Rendering 
/// Create lightweight invisible GameObject, only for Physics Raycast detection
/// </summary>
public class CellProxyManager : MonoBehaviour
{
    [Header("Proxy Settings")]
    public GameObject proxyPrefab; // Lightweight Prefab with Collider (not visible)
    public float proxyUpdateInterval = 0.1f; // Agent location update interval (seconds)
    
    private Dictionary<string, GameObject> proxyMap = new Dictionary<string, GameObject>();
    private GPURenderer gpuRenderer;
    private float lastUpdateTime = 0f;
    private bool needsUpdate = false;
    
    void Start()
    {
        if (proxyPrefab == null)
        {
            // If no Prefab is provided, create a simple Sphere Collider proxy
            CreateDefaultProxyPrefab();
        }
    }
    
    void Update()
    {
        // Regularly update proxy locations (reduce CPU overhead)
        if (Time.time - lastUpdateTime >= proxyUpdateInterval && needsUpdate)
        {
            UpdateProxyPositions();
            lastUpdateTime = Time.time;
            needsUpdate = false;
        }
    }
    
    void CreateDefaultProxyPrefab()
    {
        GameObject defaultPrefab = new GameObject("ProxyCollider");
        SphereCollider collider = defaultPrefab.AddComponent<SphereCollider>();
        collider.radius = 0.5f;
        
        // Set to Invisible
        MeshRenderer mr = defaultPrefab.AddComponent<MeshRenderer>();
        if (mr != null)
        {
            mr.enabled = false; // Disable rendering
        }
        
        proxyPrefab = defaultPrefab;
    }
    
    /// <summary>
    /// Initializing the Agent System
    /// </summary>
    public void InitializeProxies(GPURenderer renderer, List<string> cellIds, List<Vector3> initialPositions)
    {
        gpuRenderer = renderer;
        
        // Clean up old proxies
        ClearAllProxies();
        
        // Create new agent
        GameObject proxyContainer = new GameObject("ProxyContainer");
        proxyContainer.transform.SetParent(transform);
        
        for (int i = 0; i < cellIds.Count && i < initialPositions.Count; i++)
        {
            string cellId = cellIds[i];
            Vector3 pos = initialPositions[i];
            
            GameObject proxy = Instantiate(proxyPrefab, proxyContainer.transform);
            proxy.name = cellId; // Use the cell ID as the name for easy identification
            proxy.transform.position = pos;
            proxy.SetActive(true);
            
            proxyMap[cellId] = proxy;
        }
        
        needsUpdate = true;
        Debug.Log($"[Proxy Manager] Initialized {proxyMap.Count} proxy colliders");
    }
    
    /// <summary>
    /// Updates the position of all agents (obtained from GPU Renderer)
    /// </summary>
    public void UpdateProxyPositions()
    {
        if (gpuRenderer == null) return;
        
        int updatedCount = 0;
        foreach (var kvp in proxyMap)
        {
            string cellId = kvp.Key;
            GameObject proxy = kvp.Value;
            
            // Get the location from the GPU Renderer
            int index;
            if (gpuRenderer.TryGetCellIndex(cellId, out index))
            {
                GPURenderer.CellDataGPU cellData;
                if (gpuRenderer.TryGetCellData(index, out cellData))
                {
                    proxy.transform.position = cellData.position;
                    updatedCount++;
                }
            }
        }
        
        // Debug.Log($"[Proxy Manager] Updated {updatedCount} proxy positions");
    }
    
    /// <summary>
    /// The tag requires an update to the agent location
    /// </summary>
    public void MarkForUpdate()
    {
        needsUpdate = true;
    }
    
    /// <summary>
    /// Update proxy locations instantly (no waiting for intervals)
    /// </summary>
    public void UpdateProxiesImmediate()
    {
        UpdateProxyPositions();
        lastUpdateTime = Time.time;
    }
    
    /// <summary>
    /// Update individual agent locations
    /// </summary>
    public void UpdateProxyPosition(string cellId, Vector3 position)
    {
        if (proxyMap.ContainsKey(cellId))
        {
            proxyMap[cellId].transform.position = position;
        }
    }
    
    /// <summary>
    /// Set up proxy visibility (for filtering)
    /// </summary>
    public void SetProxyVisibility(string cellId, bool visible)
    {
        if (proxyMap.ContainsKey(cellId))
        {
            proxyMap[cellId].SetActive(visible);
        }
    }
    
    /// <summary>
    /// Set up proxy visibility in bulk
    /// </summary>
    public void SetProxiesVisibility(List<string> cellIds, bool visible)
    {
        foreach (string cellId in cellIds)
        {
            SetProxyVisibility(cellId, visible);
        }
    }
    
    /// <summary>
    /// Set up all agent visibility
    /// </summary>
    public void SetAllProxiesVisibility(bool visible)
    {
        foreach (var kvp in proxyMap)
        {
            kvp.Value.SetActive(visible);
        }
    }
    
    /// <summary>
    /// Clean up all proxies
    /// </summary>
    public void ClearAllProxies()
    {
        foreach (var kvp in proxyMap)
        {
            if (kvp.Value != null)
            {
                DestroyImmediate(kvp.Value);
            }
        }
        proxyMap.Clear();
        
        // Clean the container
        Transform container = transform.Find("ProxyContainer");
        if (container != null)
        {
            DestroyImmediate(container.gameObject);
        }
    }
    
    /// <summary>
    /// Get proxy GameObject (for external access)
    /// </summary>
    public GameObject GetProxy(string cellId)
    {
        if (proxyMap.ContainsKey(cellId))
            return proxyMap[cellId];
        return null;
    }
    
    /// <summary>
    /// Check if there is a proxy
    /// </summary>
    public bool HasProxy(string cellId)
    {
        return proxyMap.ContainsKey(cellId);
    }
    
    void OnDestroy()
    {
        ClearAllProxies();
    }
}
