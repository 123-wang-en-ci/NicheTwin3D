using System;
using System.Collections.Generic;
using UnityEngine;

public static class DelaunayTriangulator
{
    public struct Triangle
    {
        public int p1;
        public int p2;
        public int p3;

        public Triangle(int point1, int point2, int point3)
        {
            p1 = point1;
            p2 = point2;
            p3 = point3;
        }

        public bool HasVertex(int vertex)
        {
            return p1 == vertex || p2 == vertex || p3 == vertex;
        }

        public bool SharesEdgeWith(int pA, int pB)
        {
            return (p1 == pA && p2 == pB) || (p2 == pA && p1 == pB) ||
                   (p2 == pA && p3 == pB) || (p3 == pA && p2 == pB) ||
                   (p3 == pA && p1 == pB) || (p1 == pA && p3 == pB);
        }
    }

    private struct Edge
    {
        public int p1;
        public int p2;

        public Edge(int point1, int point2)
        {
            p1 = point1;
            p2 = point2;
        }

        public bool Equals(Edge other)
        {
            return (p1 == other.p1 && p2 == other.p2) || (p1 == other.p2 && p2 == other.p1);
        }
    }

    // Returns a list of triangle vertex indices (3 integers per triangle) based on 2D coordinates
    // We only use X and Z coordinates for 2D triangulation of the plane
    public static List<int> Triangulate(List<Vector3> points)
    {
        List<int> trianglesIndices = new List<int>();
        if (points == null || points.Count < 3) return trianglesIndices;

        int pointCount = points.Count;

        // Find min/max X and Z coordinates to create the super-triangle
        float minX = points[0].x, minY = points[0].z, maxX = minX, maxY = minY;
        for (int i = 1; i < pointCount; i++)
        {
            if (points[i].x < minX) minX = points[i].x;
            if (points[i].z < minY) minY = points[i].z;
            if (points[i].x > maxX) maxX = points[i].x;
            if (points[i].z > maxY) maxY = points[i].z;
        }

        float dx = maxX - minX;
        float dy = maxY - minY;
        float deltaMax = Mathf.Max(dx, dy);
        float midX = (minX + maxX) / 2f;
        float midY = (minY + maxY) / 2f;

        // Add Super Triangle vertices (indices: pointCount, pointCount+1, pointCount+2)
        Vector3 st1 = new Vector3(midX - 20 * deltaMax, 0, midY - deltaMax);
        Vector3 st2 = new Vector3(midX, 0, midY + 20 * deltaMax);
        Vector3 st3 = new Vector3(midX + 20 * deltaMax, 0, midY - deltaMax);
        
        List<Vector3> allPoints = new List<Vector3>(points);
        allPoints.Add(st1);
        allPoints.Add(st2);
        allPoints.Add(st3);

        List<Triangle> triangles = new List<Triangle>
        {
            new Triangle(pointCount, pointCount + 1, pointCount + 2)
        };

        // Add points sequentially to triangulation
        for (int i = 0; i < pointCount; i++)
        {
            Vector3 pt = allPoints[i];
            List<Triangle> badTriangles = new List<Triangle>();
            HashSet<Triangle> badTrianglesSet = new HashSet<Triangle>();

            // Find all triangles that are no longer valid due to the insertion
            for (int t = 0; t < triangles.Count; t++)
            {
                if (InCircumcircle(pt, allPoints[triangles[t].p1], allPoints[triangles[t].p2], allPoints[triangles[t].p3]))
                {
                    badTriangles.Add(triangles[t]);
                    badTrianglesSet.Add(triangles[t]);
                }
            }

            List<Edge> polygon = new List<Edge>();

            // Find the boundary of the polygonal hole
            foreach (var badTri in badTriangles)
            {
                Edge e1 = new Edge(badTri.p1, badTri.p2);
                Edge e2 = new Edge(badTri.p2, badTri.p3);
                Edge e3 = new Edge(badTri.p3, badTri.p1);

                if (!SharedEdge(e1, badTriangles, badTri)) polygon.Add(e1);
                if (!SharedEdge(e2, badTriangles, badTri)) polygon.Add(e2);
                if (!SharedEdge(e3, badTriangles, badTri)) polygon.Add(e3);
            }

            // Remove bad triangles from triangulation
            triangles.RemoveAll(t => badTrianglesSet.Contains(t));

            // Re-triangulate the polygonal hole
            foreach (var edge in polygon)
            {
                triangles.Add(new Triangle(edge.p1, edge.p2, i));
            }
        }

        // Clean up triangles that share vertices with the super-triangle
        for (int i = triangles.Count - 1; i >= 0; i--)
        {
            Triangle t = triangles[i];
            if (t.HasVertex(pointCount) || t.HasVertex(pointCount + 1) || t.HasVertex(pointCount + 2))
            {
                continue;
            }

            // Also remove triangles with dangerously long edges (artifact of concave hull)
            float dist1 = Vector3.Distance(allPoints[t.p1], allPoints[t.p2]);
            float dist2 = Vector3.Distance(allPoints[t.p2], allPoints[t.p3]);
            float dist3 = Vector3.Distance(allPoints[t.p3], allPoints[t.p1]);
            
            // Adjust threshold depending on your data spacing if needed.
            float threshold = deltaMax * 0.05f; // heuristic to cut outer boundary stretch
            
            if (dist1 < threshold && dist2 < threshold && dist3 < threshold)
            {
                trianglesIndices.Add(t.p1);
                // Winding order check (Unity uses clockwise)
                if (IsClockwise(allPoints[t.p1], allPoints[t.p2], allPoints[t.p3]))
                {
                    trianglesIndices.Add(t.p2);
                    trianglesIndices.Add(t.p3);
                }
                else
                {
                    trianglesIndices.Add(t.p3);
                    trianglesIndices.Add(t.p2);
                }
            }
        }

        return trianglesIndices;
    }

    private static bool IsClockwise(Vector3 a, Vector3 b, Vector3 c)
    {
        return (b.x - a.x) * (c.z - a.z) - (b.z - a.z) * (c.x - a.x) > 0;
    }

    private static bool SharedEdge(Edge edge, List<Triangle> badTriangles, Triangle exclude)
    {
        foreach (var t in badTriangles)
        {
            if (t.Equals(exclude)) continue;
            if (t.SharesEdgeWith(edge.p1, edge.p2)) return true;
        }
        return false;
    }

    // Check if point p is inside the circumcircle of triangle (a, b, c) using 2D coordinates (X, Z)
    private static bool InCircumcircle(Vector3 p, Vector3 a, Vector3 b, Vector3 c)
    {
        float ax_ = a.x - p.x;
        float ay_ = a.z - p.z;
        float bx_ = b.x - p.x;
        float by_ = b.z - p.z;
        float cx_ = c.x - p.x;
        float cy_ = c.z - p.z;

        float det = (ax_ * ax_ + ay_ * ay_) * (bx_ * cy_ - cx_ * by_) -
                    (bx_ * bx_ + by_ * by_) * (ax_ * cy_ - cx_ * ay_) +
                    (cx_ * cx_ + cy_ * cy_) * (ax_ * by_ - bx_ * ay_);

        // Ensure CCW orientation of triangle points
        float orient = (b.x - a.x) * (c.z - a.z) - (b.z - a.z) * (c.x - a.x);
        if (orient < 0) det = -det;

        return det > 0;
    }
}
