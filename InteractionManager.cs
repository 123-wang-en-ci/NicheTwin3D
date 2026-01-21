using UnityEngine;
using UnityEngine.Networking; 
using System.Collections;
using System.Text;
using UnityEngine.UI;
using System.Collections.Generic;

[System.Serializable]
public class PerturbRequest
{
    public string target_id;
    public string perturb_type; 
    public string target_gene; 
}

[System.Serializable]
public class GeneRequest
{
    public string gene_name;
    public bool use_imputation; 
}

[System.Serializable]
public class RegionResponse
{
    public string status;
    public List<int> regions;
    public List<string> names;
}
public class InteractionManager : MonoBehaviour
{
    public string serverUrl = "http://127.0.0.1:8000/perturb"; 
    [Tooltip("Drag DataLoaderGPU here. (GPU Instancing version)")]
    public DataLoaderGPU dataLoader;
    public Camera mainCamera;    

    public enum InteractionMode { Inspect, Perturb }
    public InteractionMode currentMode = InteractionMode.Inspect;

    public Image btnInspectImg;
    public Image btnPerturbImg;
    public Color activeColor = Color.green;
    public Color inactiveColor = Color.white;

    public TMPro.TMP_InputField perturbGeneInput;
    public Toggle toggleKO;

    private string lastSearchedGene = "RESET";

    public TMPro.TMP_Dropdown typeDropdown; 

    public TMPro.TMP_InputField clusterCountInput;  

    private bool HasDataLoader => dataLoader != null;

    private bool TryGetCellDetails(string id, out string typeName, out Vector2 pos, out float expr)
    {
        if (dataLoader != null) return dataLoader.GetCellDetails(id, out typeName, out pos, out expr);
        typeName = "Unknown"; pos = Vector2.zero; expr = 0;
        return false;
    }

    private float GetAverageExpression()
    {
        if (dataLoader != null) return dataLoader.GetAverageExpression();
        return 0f;
    }

    private void UpdateVisuals(string jsonString)
    {
        if (dataLoader != null) dataLoader.UpdateVisuals(jsonString);
    }

    private void ApplyAnnotationData(string jsonString)
    {
        if (dataLoader != null) dataLoader.ApplyAnnotationData(jsonString);
    }

    private IReadOnlyList<string> GetAnnotationLegend()
    {
        if (dataLoader != null) return dataLoader.annotationLegend;
        return System.Array.Empty<string>();
    }

    private void SetHighlightedType(int typeId)
    {
        if (dataLoader != null) dataLoader.highlightedTypeID = typeId;
    }

    private void SwitchModeToExpressionIfNeeded()
    {
        if (dataLoader != null)
        {
            if (dataLoader.currentMode != GPURenderer.ViewMode.Expression)
                dataLoader.SwitchMode((int)GPURenderer.ViewMode.Expression);
        }
    }

    private void SwitchModeToAIAnnotation()
    {
        if (dataLoader != null) dataLoader.SwitchMode((int)GPURenderer.ViewMode.AI_Annotation);
    }
    public void SetInspectMode()
    {
        currentMode = InteractionMode.Inspect;
        UpdateButtonVisuals();
    }

    public void SetPerturbMode()
    {
        currentMode = InteractionMode.Perturb;
        UpdateButtonVisuals();
    }

    public void RequestImputation()
    {
        if (string.IsNullOrEmpty(lastSearchedGene) || lastSearchedGene == "RESET")
        {
            if (UIManager.Instance != null)
                UIManager.Instance.ShowSystemMessage("Please search a specific gene first.", true);
            return;
        }

        Debug.Log($"[UI] Requesting Single Gene Imputation for: {lastSearchedGene}");
        StartCoroutine(SendGeneSwitchRequest(lastSearchedGene, true));
    }

    public void RequestSaveImputation()
    {
        if (string.IsNullOrEmpty(lastSearchedGene) || lastSearchedGene == "RESET")
        {
            if (UIManager.Instance != null)
                UIManager.Instance.ShowSystemMessage("No gene data to save.", true);
            return;
        }
        StartCoroutine(SendSaveImputationRequest());
    }

    IEnumerator SendSaveImputationRequest()
    {

        UnityWebRequest request = new UnityWebRequest(serverUrl.Replace("/perturb", "/save_imputation"), "POST");
        request.downloadHandler = new DownloadHandlerBuffer();
        GeneRequest req = new GeneRequest { gene_name = lastSearchedGene, use_imputation = true };
        byte[] bodyRaw = Encoding.UTF8.GetBytes(JsonUtility.ToJson(req));

        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.SetRequestHeader("Content-Type", "application/json");

        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {

            string jsonString = request.downloadHandler.text;

            DataLoaderGPU.ServerResponse response = JsonUtility.FromJson<DataLoaderGPU.ServerResponse>(jsonString);

            string msg = string.IsNullOrEmpty(response.message) ? "Imputation Saved!" : response.message;
            bool isError = response.status != "success";

            if (UIManager.Instance != null)
                UIManager.Instance.ShowSystemMessage(msg, isError);
        }
        else
        {
            if (UIManager.Instance != null)
                UIManager.Instance.ShowSystemMessage("Save Failed: " + request.error, true);
        }
    }

    IEnumerator SendGeneSwitchRequest(string geneName, bool doImpute)
    {
        GeneRequest req = new GeneRequest
        {
            gene_name = geneName,
            use_imputation = doImpute
        };

        string json = JsonUtility.ToJson(req);

        UnityWebRequest request = new UnityWebRequest("http://127.0.0.1:8000/switch_gene", "POST");
        byte[] bodyRaw = Encoding.UTF8.GetBytes(json);
        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.downloadHandler = new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");

        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            string jsonString = request.downloadHandler.text;
            DataLoaderGPU.ServerResponse response = JsonUtility.FromJson<DataLoaderGPU.ServerResponse>(jsonString);

            if (!string.IsNullOrEmpty(response.message))
            {
                bool isError = (response.status != "success");
                if (UIManager.Instance != null) UIManager.Instance.ShowSystemMessage(response.message, isError);
            }

            if (response.updates != null && response.updates.Length > 0)
            {
                lastSearchedGene = geneName;
                if (HasDataLoader)
                {
                    UpdateVisuals(jsonString);
                    SwitchModeToExpressionIfNeeded();
                }
            }
        }
        else
        {
            if (UIManager.Instance != null) UIManager.Instance.ShowSystemMessage("Request Failed", true);
        }
    }

    public void RequestGeneSwitch(string geneName)
    {

        StartCoroutine(SendGeneSwitchRequest(geneName, false));
    }

    public void RequestDisableImputation()
    {
        if (!string.IsNullOrEmpty(lastSearchedGene))
        {
            StartCoroutine(SendGeneSwitchRequest(lastSearchedGene, false));
            if (UIManager.Instance != null) UIManager.Instance.ShowSystemMessage("Showing Raw Data.", false);
        }
    }

    IEnumerator SendPerturbRequest(string id)
    {
        string pType = toggleKO.isOn ? "KO" : "OE";
        string pGene = "";
        if (perturbGeneInput != null && !string.IsNullOrEmpty(perturbGeneInput.text) && !string.IsNullOrWhiteSpace(perturbGeneInput.text))
        {
            pGene = perturbGeneInput.text.Trim();
        }
        else
        {
            string errorMsg = "Input Error: Please enter a Gene Symbol (e.g. NPHS1).";
            if (UIManager.Instance != null) UIManager.Instance.ShowSystemMessage(errorMsg, true);
            yield break;
        }

        PerturbRequest req = new PerturbRequest
        {
            target_id = id,
            perturb_type = pType,
            target_gene = pGene
        };

        string json = JsonUtility.ToJson(req);

        UnityWebRequest request = new UnityWebRequest(serverUrl, "POST");
        byte[] bodyRaw = Encoding.UTF8.GetBytes(json);
        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.downloadHandler = new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");

        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            string jsonString = request.downloadHandler.text;
            DataLoaderGPU.ServerResponse response = JsonUtility.FromJson<DataLoaderGPU.ServerResponse>(jsonString);

            if (!string.IsNullOrEmpty(response.message))
            {
                bool isError = (response.status != "success");
                if (UIManager.Instance != null) UIManager.Instance.ShowSystemMessage(response.message, isError);
            }

            if (response.updates != null && response.updates.Length > 0)
            {
                if (HasDataLoader) UpdateVisuals(jsonString);
            }
        }
        else
        {
            if (UIManager.Instance != null) UIManager.Instance.ShowSystemMessage("Server Connection Failed", true);
        }
    }

    public void RequestManualSave()
    {
        StartCoroutine(SendSaveRequest());
    }

    IEnumerator SendSaveRequest()
    {
        UnityWebRequest request = new UnityWebRequest(serverUrl.Replace("/perturb", "/save_manual"), "POST");
        request.downloadHandler = new DownloadHandlerBuffer();
        byte[] bodyRaw = Encoding.UTF8.GetBytes("{}");
        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.SetRequestHeader("Content-Type", "application/json");

        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            string jsonString = request.downloadHandler.text;
            DataLoaderGPU.ServerResponse response = JsonUtility.FromJson<DataLoaderGPU.ServerResponse>(jsonString);
            string msg = string.IsNullOrEmpty(response.message) ? "Snapshot Saved" : response.message;

            if (UIManager.Instance != null)
                UIManager.Instance.ShowSystemMessage(msg, false);
        }
        else
        {
            if (UIManager.Instance != null)
                UIManager.Instance.ShowSystemMessage("Save Failed: " + request.error, true);
        }
    }

    public void RequestClearData()
    {
        StartCoroutine(SendClearRequest());
    }

    IEnumerator SendClearRequest()
    {
        UnityWebRequest request = new UnityWebRequest(serverUrl.Replace("/perturb", "/clear_perturbation"), "POST");
        request.downloadHandler = new DownloadHandlerBuffer();
        byte[] bodyRaw = Encoding.UTF8.GetBytes("{}");
        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.SetRequestHeader("Content-Type", "application/json");

        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            if (HasDataLoader) UpdateVisuals(request.downloadHandler.text);
            SetInspectMode();
            lastSearchedGene = "RESET";

            if (UIManager.Instance != null)
                UIManager.Instance.ShowSystemMessage("Reset Successful", false);
        }
        else
        {
            if (UIManager.Instance != null)
                UIManager.Instance.ShowSystemMessage("Reset Failed", true);
        }
    }

    void UpdateButtonVisuals()
    {
        if (btnInspectImg != null && btnPerturbImg != null)
        {
            btnInspectImg.color = (currentMode == InteractionMode.Inspect) ? activeColor : inactiveColor;
            btnPerturbImg.color = (currentMode == InteractionMode.Perturb) ? activeColor : inactiveColor;
        }
    }

    void Start()
    {
        if (mainCamera == null) mainCamera = Camera.main;
        UpdateButtonVisuals();
    }

    void Update()
    {
        if (Input.GetMouseButtonDown(0))
        {
            HandleClick();
        }
    }

    void HandleClick()
    {
        Ray ray = mainCamera.ScreenPointToRay(Input.mousePosition);
        RaycastHit hit;

        if (Physics.Raycast(ray, out hit))
        {
            string clickedId = hit.transform.name;

            string typeName;
            Vector2 pos;
            float currentExpr;
            if (TryGetCellDetails(clickedId, out typeName, out pos, out currentExpr))
            {
                float avgExpr = GetAverageExpression();
                if (UIManager.Instance != null)
                    UIManager.Instance.ShowCellDetails(clickedId, typeName, pos, currentExpr, avgExpr);
                if (DashboardManager.Instance != null)
                    DashboardManager.Instance.UpdateChart(currentExpr, avgExpr);
            }

            if (currentMode == InteractionMode.Perturb)
            {
                StartCoroutine(SendPerturbRequest(clickedId));
            }
        }
    }

    public void RequestAnnotation()
    {
        StartCoroutine(SendAnnotationRequest());
    }

    IEnumerator SendAnnotationRequest()
    {
        if (UIManager.Instance != null)
            UIManager.Instance.ShowSystemMessage("AI Predicting Cell Types...", false);

        UnityWebRequest request = new UnityWebRequest(serverUrl.Replace("/perturb", "/get_annotation"), "POST");
        request.downloadHandler = new DownloadHandlerBuffer();
        request.uploadHandler = new UploadHandlerRaw(Encoding.UTF8.GetBytes("{}"));
        request.SetRequestHeader("Content-Type", "application/json");

        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {

            if (HasDataLoader) ApplyAnnotationData(request.downloadHandler.text);

  
            if (typeDropdown != null)
            {
                typeDropdown.gameObject.SetActive(true); 
                typeDropdown.ClearOptions();

     
                System.Collections.Generic.List<string> options = new System.Collections.Generic.List<string>();
                options.Add("Show All Types");
                options.AddRange(GetAnnotationLegend()); 
                typeDropdown.AddOptions(options);
                typeDropdown.value = 0; // 

     
                typeDropdown.onValueChanged.RemoveAllListeners();
                typeDropdown.onValueChanged.AddListener(OnTypeDropdownChanged);
            }

            if (UIManager.Instance != null)
                UIManager.Instance.ShowSystemMessage("Annotation Complete!", false);
        }
        else
        {
            Debug.LogError(request.error);
            if (UIManager.Instance != null)
                UIManager.Instance.ShowSystemMessage("Annotation Failed: " + request.error, true);
        }
    }

    public void OnTypeDropdownChanged(int index)
    {

        int typeId = index - 1;

        Debug.Log($"�л���������: {typeId}");
        if (HasDataLoader)
        {
            SetHighlightedType(typeId);
            SwitchModeToAIAnnotation(); 
        }
    }

    public void RequestSaveAnnotation()
    {
        StartCoroutine(SendSaveAnnotationRequest());
    }

    IEnumerator SendSaveAnnotationRequest()
    {
        //save_annotation
        UnityWebRequest request = new UnityWebRequest(serverUrl.Replace("/perturb", "/save_annotation"), "POST");
        request.downloadHandler = new DownloadHandlerBuffer();

        byte[] bodyRaw = Encoding.UTF8.GetBytes("{}");
        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.SetRequestHeader("Content-Type", "application/json");

        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            string jsonString = request.downloadHandler.text;
            DataLoaderGPU.ServerResponse response = JsonUtility.FromJson<DataLoaderGPU.ServerResponse>(jsonString);

            string msg = string.IsNullOrEmpty(response.message) ? "Annotation Saved!" : response.message;
            bool isError = response.status != "success";

            if (UIManager.Instance != null)
                UIManager.Instance.ShowSystemMessage(msg, isError);
        }
        else
        {
            if (UIManager.Instance != null)
                UIManager.Instance.ShowSystemMessage("Save Failed: " + request.error, true);
        }
    }

    public void RequestRegionSegmentation()
    {
        StartCoroutine(GetRegionRoutine());
    }
    IEnumerator GetRegionRoutine()
    {
        string url = "http://127.0.0.1:8000/get_tissue_regions";

        UnityWebRequest request = UnityWebRequest.PostWwwForm(url, "");
        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            string json = request.downloadHandler.text;
       
            RegionResponse res = JsonUtility.FromJson<RegionResponse>(json);

            if (res.status == "success")
            {
                Debug.Log(": " + (res.regions != null ? res.regions.Count.ToString() : "null"));
                if (dataLoader != null) dataLoader.ApplyRegionSegmentation(res.regions, res.names);
            }
        }
        else
        {
            Debug.LogError(": " + request.error);
        }
    }

    public void OnSaveRegionBtnClick()
    {
        StartCoroutine(SaveRegionDataRoutine());
    }

    IEnumerator SaveRegionDataRoutine()
    {
        string url = "http://127.0.0.1:8000/save_tissue_regions";

        using (UnityWebRequest request = new UnityWebRequest(url, "POST"))
        {
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");

            yield return request.SendWebRequest();

            if (request.result == UnityWebRequest.Result.Success)
            {
                var res = JsonUtility.FromJson<CommonResponse>(request.downloadHandler.text);
                if (UIManager.Instance != null)
                    UIManager.Instance.ShowSystemMessage("Save to" + res.message, false);
                Debug.Log($"<color=green></color> {res.message}");

               
            }
            else
            {
                Debug.LogError($": {request.error}");
            }
        }
        

        
    }

    public void RequestZeroShotClustering()
    {
        int k = 10;

        if (clusterCountInput != null && !string.IsNullOrEmpty(clusterCountInput.text))
        {

            if (!int.TryParse(clusterCountInput.text, out k))
            {
                k = 10; 
               
            }
        }

        k = Mathf.Clamp(k, 2, 50);

        StartCoroutine(SendClusteringRequest(k));
    }

    IEnumerator SendClusteringRequest(int k)
    {
        if (UIManager.Instance != null)
            UIManager.Instance.ShowSystemMessage($"Running K-Means Clustering (K={k})...", false);


        ClusteringRequest req = new ClusteringRequest { n_clusters = k };
        string json = JsonUtility.ToJson(req);


        string url = serverUrl.Replace("/perturb", "/zero_shot_cluster");
        UnityWebRequest request = new UnityWebRequest(url, "POST");
        byte[] bodyRaw = Encoding.UTF8.GetBytes(json);

        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.downloadHandler = new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");

        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            string responseJson = request.downloadHandler.text;
            Debug.Log("Clustering Response: " + responseJson); 


            ClusteringResponse response = JsonUtility.FromJson<ClusteringResponse>(responseJson);

            if (response.status == "success")
            {
                Debug.Log($"[Unity]  {response.legend.Count} ");

                if (dataLoader != null)
                {
                    

                    if (response.clusters != null && response.clusters.Count > 0)
                    {

                        if (response.updates != null)
                        {
                            if (dataLoader != null) dataLoader.ApplyZeroShotClustering(response.legend, response.updates);
                        }
                        else
                        {
                            Debug.LogError("Clustering Error: No updates found.");
                        }
                    }
                    else if (response.updates != null)
                    {
                        if (dataLoader != null) dataLoader.ApplyZeroShotClustering(response.legend, response.updates);
                    }
                }

                if (UIManager.Instance != null)
                    UIManager.Instance.ShowSystemMessage(response.message, false);
            }
            else
            {
                if (UIManager.Instance != null)
                    UIManager.Instance.ShowSystemMessage("Clustering Error: " + response.message, true);
            }
        }
        else
        {
            Debug.LogError(request.error);
            if (UIManager.Instance != null)
                UIManager.Instance.ShowSystemMessage("Network Error: " + request.error, true);
        }
    }


    public void RequestSaveClustering()
    {
        StartCoroutine(SendSaveClusteringRequest());
    }

    IEnumerator SendSaveClusteringRequest()
    {
        if (UIManager.Instance != null)
            UIManager.Instance.ShowSystemMessage("Saving clustering results...", false);

        // save_zero_shot
        string url = serverUrl.Replace("/perturb", "/save_zero_shot");


        UnityWebRequest request = new UnityWebRequest(url, "POST");
        request.downloadHandler = new DownloadHandlerBuffer();

        byte[] bodyRaw = Encoding.UTF8.GetBytes("{}");
        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.SetRequestHeader("Content-Type", "application/json");

        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            CommonResponse res = JsonUtility.FromJson<CommonResponse>(request.downloadHandler.text);
            string msg = string.IsNullOrEmpty(res.message) ? "Clustering Results Saved!" : res.message;

            if (UIManager.Instance != null)
                UIManager.Instance.ShowSystemMessage(msg, false);
        }
        else
        {
            if (UIManager.Instance != null)
                UIManager.Instance.ShowSystemMessage("Save Failed: " + request.error, true);
        }
    }

}

[System.Serializable]
public class CommonResponse
{
    public string status;
    public string message;
}


[System.Serializable]
public class ClusteringRequest
{
    public int n_clusters;      
}

[System.Serializable]
public class ClusterLegendItem
{
    public int id;
    public string name;
    public string color; 
}


[System.Serializable]
public class ClusterUpdateItem
{
    public string id;
    public int cluster_id;
}

[System.Serializable]
public class ClusteringResponse
{
    public string status;
    public string message;
    public List<ClusterLegendItem> legend;
    
    public List<ClusterUpdateItem> updates;


    public List<int> clusters;
}