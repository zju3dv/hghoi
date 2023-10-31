using UnityEngine;
using System.Collections;
using UnityEditor;
using System;
using System.Collections.Generic;
using UnityEditor.AI;


public class SAMPTest : MonoBehaviour
{
    public SAMP_Demo Animation = null;
    public MotionEditor editor = null;
    public string act = "Sit";
    public int Test_i = 0;
    public int SampleNum = 1;
    public int MotionNum = 0;
    public int MaxFrames = 1800;
    public int StaticFrames = 120;

    protected string[] actions = {"Sit", "Sit", "Sit", "Liedown", "Sit", "Sit"};
    public string[] TestSequences = { "armchair019", "chair_mo019", "highstool019", "lie_down_19", "sofa019", "table019"};
    public int[] ObjectsNum = { 3, 3, 3, 3, 3, 3};

    public List<MotionData> Files = new List<MotionData>();

    // Use this for initialization
    void Start()
    {
        LoadAllFiles();
    }

    // Update is called once per frame
    void Update()
    {
        if (Test_i < editor.Files.Length) 
	    { 
            TestSequence();
	    }
    }

    public void ReBuildNavMesh()
    {
        NavMeshBuilder.BuildNavMesh();
    }

    public IEnumerator LoadSequence(MotionData file)
    {
        EditorUtility.UnloadUnusedAssetsImmediate();
        Resources.UnloadUnusedAssets();
        GC.Collect();
        editor.LoadData(file);
        Debug.Log("File: " + editor.GetData().GetName());
        while (!editor.GetData().GetScene().isLoaded)
        {
            Debug.Log("Waiting for scene to be loaded.");
            yield return new WaitForSeconds(0f);
        }
    }

    public virtual void TestSequence() 
    { 
        if (editor.GetData().GetName() != TestSequences[Test_i] || !editor.GetData().GetScene().isLoaded)
		{ 
            Debug.Log($"Start Load Scene: {TestSequences[Test_i]}");
	        StartCoroutine(LoadSequence(Files[Test_i]));
	    }
        if (!Animation.GetIsinteracting() && editor.GetData().GetName() == TestSequences[Test_i] && editor.GetData().GetScene().isLoaded) 
	    {
            editor.ChangeObject(MotionNum - GetSampledNum());
            ReBuildNavMesh();
            Debug.Log($"Start Motion {MotionNum}!");
            StartCoroutine(Animation.InteractWithObject(actions[Test_i], GetEndPoint(), GetStartPoint(), GetStartPose(), GetStartVelocity()));
        }
    }

    public void SetTesti()
    {
        int n = 0;
        for (int i = 0; i < Test_i + 1; i++)
        {
            n += ObjectsNum[i] * SampleNum;
        }
        if (MotionNum >= n)
        {
            Test_i++;
        }
    }

    public int GetSampledNum()
    {
        int n = 0;
        for (int i = 0; i < Test_i; i++)
        {
            n += ObjectsNum[i] * SampleNum;
        }
        return n;
    }

    public int GetTotalNum()
    { 
        int n = 0;
        for (int i = 0; i < ObjectsNum.Length; i++)
        {
            n += ObjectsNum[i] * SampleNum;
        }
        return n;
    }

    public void LoadAllFiles()
    {   
        Files = new List<MotionData>();
        for (int j = 0; j < TestSequences.Length; j ++) {
            for (int i = 0; i < editor.Files.Length; i++)
            {
                if (editor.Files[i].name == TestSequences[j])
                {
                    Files.Add(editor.Files[i]);
                    break;
                }
            }
	    }
    }

    public virtual Matrix4x4 GetEndPoint() 
    { 
        Sequence seq = editor.GetData().GetUnrolledSequence();
        float end = editor.FloorToTargetTime(editor.GetData().GetFrame(seq.End).Timestamp);
        editor.LoadFrame(end);
        Matrix4x4 EndPoint = editor.GetActor().GetRoot().GetWorldMatrix(true);
        return EndPoint;
    }

    public virtual Matrix4x4 GetStartPoint() 
    { 
        Sequence seq = editor.GetData().GetUnrolledSequence();
        float start = editor.FloorToTargetTime(editor.GetData().GetFrame(seq.Start).Timestamp);
        editor.LoadFrame(start);
        Matrix4x4 StartPoint = editor.GetActor().GetRoot().GetWorldMatrix(true);
        return StartPoint;
    }

    public virtual Matrix4x4[] GetStartPose() 
    { 
        Sequence seq = editor.GetData().GetUnrolledSequence();
        float start = editor.FloorToTargetTime(editor.GetData().GetFrame(seq.Start).Timestamp);
        editor.LoadFrame(start);
        return editor.GetActor().GetBoneTransformations();
    }

    public virtual Vector3[] GetStartVelocity()
    {
        Sequence seq = editor.GetData().GetUnrolledSequence();
        float start = editor.FloorToTargetTime(editor.GetData().GetFrame(seq.Start).Timestamp);
        editor.LoadFrame(start);
        return editor.GetActor().GetBoneVelocities();
    }
}
