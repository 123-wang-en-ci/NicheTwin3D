using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class UI_GPURendererParams : MonoBehaviour
{
    [Header("Dependencies")]
    public GPURenderer gpuRenderer;

    [Header("Base Scale Settings")]
    public Slider baseScaleSlider;
    public TextMeshProUGUI baseScaleText;
    public float minBaseScale = 0.5f;
    public float maxBaseScale = 15.0f;

    [Header("Height Multiplier Settings")]
    public Slider heightMultiplierSlider;
    public TextMeshProUGUI heightMultiplierText;
    public float minHeightMultiplier = 0.1f;
    public float maxHeightMultiplier = 5.0f;

    void Start()
    {
        if (gpuRenderer == null)
        {
            gpuRenderer = FindObjectOfType<GPURenderer>();
        }

        if (gpuRenderer == null)
        {
            Debug.LogError("[UI_GPURendererParams] GPURenderer component not found!");
            return;
        }

        //Initialize BaseScale Slider
        if (baseScaleSlider != null)
        {
            baseScaleSlider.minValue = minBaseScale;
            baseScaleSlider.maxValue = maxBaseScale;
            baseScaleSlider.value = gpuRenderer.baseScale;
            
            //Add listener
            baseScaleSlider.onValueChanged.AddListener(OnBaseScaleChanged);
            
            //Initial display
            UpdateBaseScaleText(gpuRenderer.baseScale);
        }

        //Initialize HeightMultiplier Slider
        if (heightMultiplierSlider != null)
        {
            heightMultiplierSlider.minValue = minHeightMultiplier;
            heightMultiplierSlider.maxValue = maxHeightMultiplier;
            heightMultiplierSlider.value = gpuRenderer.heightMultiplier;
            
            //Add listener
            heightMultiplierSlider.onValueChanged.AddListener(OnHeightMultiplierChanged);
            
            //Initial display
            UpdateHeightMultiplierText(gpuRenderer.heightMultiplier);
        }
    }

    private void OnBaseScaleChanged(float val)
    {
        if (gpuRenderer != null)
        {
            gpuRenderer.SetBaseScale(val);
            UpdateBaseScaleText(val);
        }
    }

    private void OnHeightMultiplierChanged(float val)
    {
        if (gpuRenderer != null)
        {
            gpuRenderer.SetHeightMultiplier(val);
            UpdateHeightMultiplierText(val);
        }
    }

    private void UpdateBaseScaleText(float val)
    {
        if (baseScaleText != null)
        {
            baseScaleText.text = $"{val:F0}";
        }
    }

    private void UpdateHeightMultiplierText(float val)
    {
        if (heightMultiplierText != null)
        {
            heightMultiplierText.text = $"{val:F0}";
        }
    }
}
