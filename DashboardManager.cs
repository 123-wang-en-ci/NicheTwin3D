using UnityEngine;
using UnityEngine.UI; // Reference UI
using TMPro;
using System.Linq; // used to calculate the average

public class DashboardManager : MonoBehaviour
{
    [Header("UI Component")]
    public RectTransform barCurrent; //Drag into the red column
    public RectTransform barAverage; // Drag into the gray column
    public TextMeshProUGUI valCurrentText;
    public TextMeshProUGUI valAverageText;

    [Header("Settings")]
    public float maxHeight = 200f; // Maximum height of column

    // Singleton is easy to call
    public static DashboardManager Instance;

    void Awake() { Instance = this; }

    //Update chart
    public void UpdateChart(float currentVal, float allCellsAverage)
    {
        // 1. Set text
        valCurrentText.text = currentVal.ToString("F2");
        valAverageText.text = allCellsAverage.ToString("F2");

        // 2. Set the column height (assuming the maximum value is 1.0)
        // Simple animation effects can be used Mathf.Lerp, set directly here
        barCurrent.sizeDelta = new Vector2(barCurrent.sizeDelta.x, currentVal * maxHeight);
        barAverage.sizeDelta = new Vector2(barAverage.sizeDelta.x, allCellsAverage * maxHeight);
    }
}