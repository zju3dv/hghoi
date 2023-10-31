using System.Collections;
using System.Collections.Generic;
using UnityEngine;


public class ReceiveFrames
{

    public FramePose[] Data = new FramePose[0];
    public FramePose[] KeyPose = new FramePose[0];
    public int[] key_t = new int[0];
    public int Pivot = 0;
    public int Dim = 0;

    public ReceiveFrames(int datadim)
    {
        Dim = datadim;
    }

    public int GetFramesNum
    {
        get { return Data.Length; }
    }

    public int GetDataLength
    {
        get
        {
            if (Data.Length > 0)
            { return Data.Length * Data[0].GetDataLength; }
            else
            { return Data.Length * (247 + 2048); }
        }
    }

    public int GetKeyPivot()
    {
        int key_i = 0;
        for (int i = 0; i < key_t.Length; i++)
        {
            if (Pivot < key_t[i])
            {
                key_i = i - 1;
                break;
            }
        }
        return key_i;

    }

    public FramePose GetPose()
    {
        return Data[Pivot];
    }
    public void SetKey(float[] data, float[] T)
    {
        KeyPose = new FramePose[T.Length];
        int k = 0;
        for (int i = 0; i < T.Length; i++)
        {
            float[] pose = new float[Dim];
            for (int j = 0; j < pose.Length; j++)
            {
                pose[j] = data[k];
                k++;
            }
            KeyPose[i] = new FramePose(pose);
            ArrayExtensions.Add(ref key_t, (int)T[i]);
        }
    }
    public void SetData(float[] data, int T)
    {
        Data = new FramePose[T];
        int k = 0;
        for (int i = 0; i < T; i++)
        {
            float[] pose = new float[Dim];
            for (int j = 0; j < pose.Length; j++)
            {
                pose[j] = data[k];
                k++;
            }
            Data[i] = new FramePose(pose);
        }
    }

    public class FramePose
    {
        public float[] data;
        public int Pivot = -1;

        public FramePose(float[] pose)
        {
            data = pose;
        }

        public int GetDataLength
        {
            get { return data.Length; }
        }

        public float Read()
        {
            Pivot += 1;
            return data[Pivot];
        }

        public Vector3 ReadVector3()
        {
            return new Vector3(Read(), Read(), Read());
        }

    }
}