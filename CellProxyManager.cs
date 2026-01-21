using UnityEngine;
using System.Collections.Generic;

/// <summary>
/// Proxy Collider Manager - Interaction detection for GPU Instancing rendering
/// Create a lightweight invisible GameObject for Physics Raycast detection only
/// </summary>
public class CellProxyManager : MonoBehaviour
{
    [Header("Proxy Settings")]
    public GameObject proxyPrefab; 
    public float proxyUpdateInterval = 0.1f; 
    
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
        
        MeshRenderer mr = defaultPrefab.AddComponent<MeshRenderer>();
        if (mr != null)
        {
            mr.enabled = false;
        }
        
        proxyPrefab = defaultPrefab;
    }
    
    public void InitializeProxies(GPURenderer renderer, List<string> cellIds, List<Vector3> initialPositions)
    {
        gpuRenderer = renderer;

        ClearAllProxies();
        
        GameObject proxyContainer = new GameObject("ProxyContainer");
        proxyContainer.transform.SetParent(transform);
        
        for (int i = 0; i < cellIds.Count && i < initialPositions.Count; i++)
        {
            string cellId = cellIds[i];
            Vector3 pos = initialPositions[i];
            
            GameObject proxy = Instantiate(proxyPrefab, proxyContainer.transform);
            proxy.name = cellId;
            proxy.transform.position = pos;
            proxy.SetActive(true);
            
            proxyMap[cellId] = proxy;
        }
        
        needsUpdate = true;
        Debug.Log($"[Proxy Manager] Initialized {proxyMap.Count} proxy colliders");
    }
    
    /// <summary>
    /// Update the location of all proxies (get from GPU Renderer)
    /// </summary>
    public void UpdateProxyPositions()
    {
        if (gpuRenderer == null) return;
        
        int updatedCount = 0;
        foreach (var kvp in proxyMap)
        {
            string cellId = kvp.Key;
            GameObject proxy = kvp.Value;

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
    
    public void MarkForUpdate()
    {
        needsUpdate = true;
    }

    public void UpdateProxiesImmediate()
    {
        UpdateProxyPositions();
        lastUpdateTime = Time.time;
    }

    public void UpdateProxyPosition(string cellId, Vector3 position)
    {
        if (proxyMap.ContainsKey(cellId))
        {
            proxyMap[cellId].transform.position = position;
        }
    }

    public void SetProxyVisibility(string cellId, bool visible)
    {
        if (proxyMap.ContainsKey(cellId))
        {
            proxyMap[cellId].SetActive(visible);
        }
    }

    public void SetProxiesVisibility(List<string> cellIds, bool visible)
    {
        foreach (string cellId in cellIds)
        {
            SetProxyVisibility(cellId, visible);
        }
    }

    public void SetAllProxiesVisibility(bool visible)
    {
        foreach (var kvp in proxyMap)
        {
            kvp.Value.SetActive(visible);
        }
    }

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

        Transform container = transform.Find("ProxyContainer");
        if (container != null)
        {
            DestroyImmediate(container.gameObject);
        }
    }
    
    public GameObject GetProxy(string cellId)
    {
        if (proxyMap.ContainsKey(cellId))
            return proxyMap[cellId];
        return null;
    }

    public bool HasProxy(string cellId)
    {
        return proxyMap.ContainsKey(cellId);
    }
    
    void OnDestroy()
    {
        ClearAllProxies();
    }
}
