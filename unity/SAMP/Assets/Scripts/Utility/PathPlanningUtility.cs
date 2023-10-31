using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.AI;


public class PathPlanningUtility
{
    public NavMeshPath Path = null;
    public Vector3[] WayPoints = null;
    public bool FinalTargetReached = false;
    public int CurrTarget = 0;
    public Vector3[] MilestonesPosition = null;
    public Vector3[] MilestonesForward = null;

    public PathPlanningUtility()
    {
        Path = new NavMeshPath();
        MilestonesPosition = new Vector3[0];
        MilestonesForward = new Vector3[0];
    }

    public Vector3[] ComputePath(Vector3 pointA, Vector3 pointB)
    {
        Debug.Log("Compute a path!");
        ResetPath();
        //Find closest point
        NavMeshHit ClosestToA;
        if (!NavMesh.SamplePosition(pointA, out ClosestToA, 1.0f, 1))
            Debug.Log("No close point found for Point A");

        //Find closest point
        NavMeshHit ClosestToB;
        if (!NavMesh.SamplePosition(pointB, out ClosestToB, 1.0f, 1))
            Debug.Log("No close point found for Point B");

        if (!NavMesh.CalculatePath(ClosestToA.position, ClosestToB.position, 1, Path))
        {
            Debug.Log("Invalid path! Directly connect the start and the goal!");
            WayPoints = new Vector3[2];
            WayPoints[0] = pointA;
            WayPoints[1] = pointB;
        }
        else
        {
            WayPoints = new Vector3[Path.corners.Length];
            WayPoints[0] = pointA;
            WayPoints[WayPoints.Length - 1] = pointB;
            for (int i = 1; i < Path.corners.Length - 1; i++)
            {
                WayPoints[i] = Path.corners[i];
            }

            WayPoints = PreProcessWayPoints();
        }
        return WayPoints;
    }

    public Vector3[] PreProcessWayPoints()
    {
        List<Vector3> PrunedWayPoints = new List<Vector3>();
        PrunedWayPoints.Add(WayPoints[0]);
        for (int i = 1; i < WayPoints.Length - 1; i++)
        {
            // Get rid of points which are too close to each other 
            if (Vector3.Distance(WayPoints[i], WayPoints[WayPoints.Length - 1]) > 0.5f)
                PrunedWayPoints.Add(WayPoints[i]);
        }
        PrunedWayPoints.Add(WayPoints[WayPoints.Length - 1]);
        WayPoints = PrunedWayPoints.ToArray();
        return WayPoints;
    }

    public Vector3 GetNextMove(Vector3 root, Vector3 currForward, Vector3 finalTarget)
    {
        float thresh = 0.3f;

        if (FinalTargetReached || CurrTarget == (WayPoints.Length - 1))
        {
            FinalTargetReached = true;
            Debug.Log("Returning Final Target");
            return finalTarget;
        }
        else
        {
            var nextWayPoint = WayPoints[CurrTarget];
            if (Vector3.Distance(root, nextWayPoint) <= thresh)
            {
                CurrTarget += 1;
            }

            Vector3 ToCurrTarget = nextWayPoint - root;
            Vector3 move = Vector3.zero;

            if (Vector3.Angle(currForward, ToCurrTarget) < 45)
            {
                move.z += 1f; //Move forward
            }
            return move;
        }
    }

    public Vector3 GetFinalMove(Vector3 root, Vector3 currForward)
    {
        var targetPoint = WayPoints[WayPoints.Length - 1];
        Vector3 ToCurrTarget = targetPoint - root;
        Vector3 move = Vector3.zero;

        if (Vector3.Angle(currForward, ToCurrTarget) < 45)
        {
            move.z += ToCurrTarget.magnitude > 1f ? 1f : ToCurrTarget.magnitude;
        }
        return move;
    }

    public float GetNextTurn(Vector3 root, Vector3 currForward)
    {
        float thresh = 0.5f;
        var nextWayPoint = WayPoints[CurrTarget];
        if (Vector3.Distance(root, nextWayPoint) <= thresh)
        {
            CurrTarget += 1;
        }
        Vector3 ToCurrTarget = nextWayPoint - root;

        float turn = 0f;
        if (Vector3.SignedAngle(currForward, ToCurrTarget, Vector3.up) > 10)
        {
            turn += 90f;
        }
        else if (Vector3.SignedAngle(currForward, ToCurrTarget, Vector3.up) < -10)
        {
            turn -= 90f;
        }
        return turn;
    }

    public float GetFinalTurn(Vector3 root, Vector3 currForward)
    {
        var nextWayPoint = WayPoints[WayPoints.Length - 1];
        Vector3 ToCurrTarget = nextWayPoint - root;

        float turn = 0f;
        if (Vector3.SignedAngle(currForward, ToCurrTarget, Vector3.up) > 10)
        {
            turn += 90f;
        }
        else if (Vector3.SignedAngle(currForward, ToCurrTarget, Vector3.up) < -10)
        {
            turn -= 90f;
        }
        return turn;
    }

    public float Distance2Goal(Vector3 root)
    {
        Vector3 TargetGround = WayPoints[WayPoints.Length - 1];
        TargetGround.y = 0f;
        return Vector3.Distance(root, TargetGround);
    }

    public void DrawPath()
    {
        if (Path.corners.Length > 0)
        {
            UltiDraw.Begin();
            for (int i = 0; i < WayPoints.Length; i++)
            {
                UltiDraw.DrawSphere(WayPoints[i], Quaternion.identity, 0.2f, UltiDraw.Black);
                if (i < WayPoints.Length - 1)
                { UltiDraw.DrawLine(WayPoints[i], WayPoints[i + 1], 0.03f, UltiDraw.Red); }
            }
            UltiDraw.End();
        }
    }

    public void DrawPath(Color c)
    {
        if (Path.corners.Length > 0)
        {
            UltiDraw.Begin();
            for (int i = 0; i < WayPoints.Length; i++)
            {
                UltiDraw.DrawSphere(WayPoints[i], Quaternion.identity, 0.2f, UltiDraw.Black);
                if (i < WayPoints.Length - 1)
                { UltiDraw.DrawLine(WayPoints[i], WayPoints[i + 1], 0.03f, c); }
            }
            UltiDraw.End();
        }
    }

    public void ResetPath()
    {
        Path.ClearCorners();
        WayPoints = null;
        CurrTarget = 0;
        FinalTargetReached = false;
	    MilestonesPosition = new Vector3[0];
        MilestonesForward = new Vector3[0];
    }

    public Vector3 ResetPoint(Vector3 querypoint, Vector3 Goal)
    {
        NavMeshHit closest;
        if (!NavMesh.SamplePosition(querypoint, out closest, 1.0f, 1))
        {
            Debug.Log("No close point found for Point query");
            return querypoint;
        }
        if (Vector3.Distance(closest.position, querypoint) > 0.001)
        {
            Vector3 delta_pos = querypoint - Goal;
            return delta_pos * 1.5f + Goal;
        }
        //if (Vector3.Distance(closest.position, Goal) < 0.5)
        //{
        //    return Goal;
        //}
        return closest.position;
    }

    public void GenerateMilestones(Vector3 start_forward, Vector3 end_forward, float distance_interval=1.5f)
    {
        Vector3[] NewWayPoints = new Vector3[WayPoints.Length];
        for (int i =0; i < WayPoints.Length; i++) {
            NewWayPoints[i] = WayPoints[i];
            NewWayPoints[i].y = 0f;
	}
        MilestonesPosition = new Vector3[0];
        MilestonesForward = new Vector3[0];
        for (int i = 1; i < NewWayPoints.Length; i++)
        {
            ArrayExtensions.Add(ref MilestonesPosition, NewWayPoints[i - 1]);
            float distance = Vector3.Distance(NewWayPoints[i - 1], NewWayPoints[i]);
            int N = (int)(distance/ distance_interval);
            Vector3 delta_pos = (NewWayPoints[i] - NewWayPoints[i - 1]) / (N + 1);
            for (int j = 1; j < N + 1; j++)
            {
                Vector3 point = NewWayPoints[i - 1] + delta_pos * j;
                ArrayExtensions.Add(ref MilestonesPosition, point);
            }
        }
        ArrayExtensions.Add(ref MilestonesPosition, NewWayPoints[NewWayPoints.Length - 1]);

        ArrayExtensions.Add(ref MilestonesForward, start_forward);
        for (int i = 1; i < MilestonesPosition.Length - 1; i++)
        {
            Vector3 point_forward = (MilestonesPosition[i] - MilestonesPosition[i - 1]).normalized;
            ArrayExtensions.Add(ref MilestonesForward, point_forward);
        }
        ArrayExtensions.Add(ref MilestonesForward, end_forward);

        ArrayExtensions.Add(ref MilestonesPosition, NewWayPoints[NewWayPoints.Length - 1]);
        ArrayExtensions.Add(ref MilestonesForward, end_forward);
    }

    public void PostProcessMilestones(float ratio)
    {
        List<Vector3> PurnedMilestonesPos = new List<Vector3>();
        List<Vector3> PurnedMilestonesDir = new List<Vector3>();
        PurnedMilestonesPos.Add(MilestonesPosition[0]);
        PurnedMilestonesDir.Add(MilestonesForward[0]);
        //PurnedMilestonesPos.Add(MilestonesPosition[0]);
        //PurnedMilestonesDir.Add(MilestonesForward[0]);
        if (Vector3.Distance(MilestonesPosition[1], MilestonesPosition[0]) > 0f)
        {
            PurnedMilestonesPos.Add(Vector3.Slerp(MilestonesPosition[0], MilestonesPosition[1], ratio));
            PurnedMilestonesDir.Add(Vector3.Slerp(MilestonesForward[0], MilestonesForward[1], ratio));
        }
            PurnedMilestonesPos.Add(Vector3.Slerp(PurnedMilestonesPos[1], MilestonesPosition[2], 1 - ratio));
            PurnedMilestonesDir.Add(Vector3.Slerp(PurnedMilestonesDir[1], MilestonesForward[2], 1 - ratio));

        for (int i = 2; i < MilestonesPosition.Length - 1; i++)
        {
            // Get rid of points which are too close to each other 
            if (Vector3.Distance(MilestonesPosition[i], PurnedMilestonesPos[PurnedMilestonesPos.Count - 1]) > 0.5f) { 
                PurnedMilestonesPos.Add(MilestonesPosition[i]);
                PurnedMilestonesDir.Add(MilestonesForward[i]);
	    
	    }
        }
        PurnedMilestonesPos.Add(MilestonesPosition[MilestonesPosition.Length - 1]);
        PurnedMilestonesDir.Add(MilestonesForward[MilestonesForward.Length - 1]);
        MilestonesPosition = PurnedMilestonesPos.ToArray();
        MilestonesForward = PurnedMilestonesDir.ToArray();
    }

    public void DrawMilestones()
    {
        if (MilestonesPosition.Length > 0)
        {
            UltiDraw.Begin();
            for (int i = 0; i < MilestonesPosition.Length; i++)
            {
                UltiDraw.DrawSphere(MilestonesPosition[i], Quaternion.identity, 0.2f, UltiDraw.Blue);
                if (i < MilestonesPosition.Length - 1)
                { UltiDraw.DrawLine(MilestonesPosition[i], MilestonesPosition[i + 1], 0.03f, UltiDraw.Yellow); }
                //Directions
                UltiDraw.DrawLine(MilestonesPosition[i], MilestonesPosition[i] + 0.25f * MilestonesForward[i], 0.025f, 0f, UltiDraw.Purple.Transparent(0.75f));
            }
            UltiDraw.End();
        }
    }
}
