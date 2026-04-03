using UnityEngine;
using UnityEngine.UI; 
using TMPro;
using System.Linq; 

public class DashboardManager : MonoBehaviour
{
    [Header("UI")]
    public RectTransform barCurrent;
    public RectTransform barAverage;
    public TextMeshProUGUI valCurrentText;
    public TextMeshProUGUI valAverageText;

    [Header("Settings")]
    public float maxHeight = 200f; //  Maximum height of the column

    // Singleton is easy to call
    public static DashboardManager Instance;

    void Awake() { Instance = this; }

    // Update the chart
    public void UpdateChart(float currentVal, float allCellsAverage)
    {
        // 1. Set text
        valCurrentText.text = currentVal.ToString("F2");
        valAverageText.text = allCellsAverage.ToString("F2");

        // 2. Set the height of the column (assuming the maximum value is 1.0) 
        // Simple animation effects can be set directly with Mathf.Lerp, here
        barCurrent.sizeDelta = new Vector2(barCurrent.sizeDelta.x, currentVal * maxHeight);
        barAverage.sizeDelta = new Vector2(barAverage.sizeDelta.x, allCellsAverage * maxHeight);
    }
}