using UnityEngine;
using System.Collections;
using UnityEditor;
using System;
using System.Collections.Generic;
using UnityEditor.AI;


public class PosTest : SAMPTest
{

    // Use this for initialization
    void Start()
    {
        LoadAllFiles();
    }

    // Update is called once per frame
    void Update()
    {
        if (Test_i < 1)
        {
            TestSequence();
        }
    }

    public override void TestSequence()
    {
        if (editor.GetData().GetName() != TestSequences[Test_i] || !editor.GetData().GetScene().isLoaded)
        {
            Debug.Log($"Start Load Scene: {TestSequences[Test_i]}");
            StartCoroutine(LoadSequence(Files[Test_i]));
        }
        if (!Animation.GetIsinteracting() && editor.GetData().GetName() == TestSequences[Test_i] && editor.GetData().GetScene().isLoaded)
        {
            //editor.ChangeObject(MotionNum - GetSampledNum());
            editor.SetAllObjectInactive();
            ReBuildNavMesh();
            Debug.Log($"Start Motion {MotionNum}!");
            StartCoroutine(Animation.InteractWithObject(actions[Test_i], GetEndPoint(), GetStartPoint(), GetStartPose(), GetStartVelocity()));
        }
    }

    public Vector3 SamplePos()
    {
        float d = UnityEngine.Random.Range(1f, 10f);
        float angle = UnityEngine.Random.Range(-(float)Math.PI, (float)Math.PI);
        Vector3 pos = Vector3.zero;
        pos.x = d * (float)Math.Sin(angle);
        pos.z = d * (float)Math.Cos(angle);
        return pos;
    }


    public virtual Matrix4x4 GetEndPoint()
    {
        Sequence seq = editor.GetData().GetUnrolledSequence();
        float end = editor.FloorToTargetTime(editor.GetData().GetFrame(seq.End).Timestamp);
        editor.LoadFrame(end);
        Matrix4x4 EndPoint = editor.GetActor().GetRoot().GetWorldMatrix(true);
        Vector3 SampledPos = SamplePos();
        EndPoint.m03 = SampledPos.x;
        EndPoint.m13 = SampledPos.y;
        EndPoint.m23 = SampledPos.z;
        return EndPoint;
    }

    public virtual Matrix4x4 GetStartPoint()
    {
        Sequence seq = editor.GetData().GetUnrolledSequence();
        float start = editor.FloorToTargetTime(editor.GetData().GetFrame(seq.Start).Timestamp);
        editor.LoadFrame(start);
        Matrix4x4 StartPoint = editor.GetActor().GetRoot().GetWorldMatrix(true);
        Vector3 SampledPos = SamplePos();
        StartPoint.m03 = SampledPos.x;
        StartPoint.m13 = SampledPos.y;
        StartPoint.m23 = SampledPos.z;
        return StartPoint;
    }

    public virtual Matrix4x4[] GetStartPose()
    {
        Sequence seq = editor.GetData().GetUnrolledSequence();
        float start = editor.FloorToTargetTime(editor.GetData().GetFrame(seq.Start).Timestamp);
        editor.LoadFrame(start);
        editor.SetAllObjectInactive();
        return editor.GetActor().GetBoneTransformations();
    }

    public virtual Vector3[] GetStartVelocity()
    {
        Sequence seq = editor.GetData().GetUnrolledSequence();
        float start = editor.FloorToTargetTime(editor.GetData().GetFrame(seq.Start).Timestamp);
        editor.LoadFrame(start);
        editor.SetAllObjectInactive();
        return editor.GetActor().GetBoneVelocities();
    }
}
