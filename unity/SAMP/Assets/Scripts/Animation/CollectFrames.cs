using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CollectFrames
{
    public int DataDim;
    public FrameData[] Data = new FrameData[0];
    public Matrix4x4[] Traj = new Matrix4x4[0];

    public CollectFrames(int datadim)
    {
        DataDim = datadim;
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
            { return Data.Length * (DataDim); }
        }
    }

    public Matrix4x4 GetTraj(int pivot)
    {
        return Traj[pivot];
    }

    public float[] GetData()
    {
        float[] expand_data = new float[GetDataLength];
        int k = 0;
        for (int i = 0; i < GetFramesNum; i++)
        {
            for (int j = 0; j < Data[i].GetDataLength; j++)
            {
                expand_data[k] = Data[i].data[j];
                k++;
            }
        }
        return expand_data;
    }

    public void Add(FrameData data)
    {
        ArrayExtensions.Add(ref Data, data);
    }

    public void Add(Matrix4x4 traj)
    {
        ArrayExtensions.Add(ref Traj, traj);
    }

    public class FrameData
    {
        public float[] data = new float[0];
        public int PastKeys;
        public int FutureKeys;
        public int Resolution;
        public int Pivot = -1;

        public FrameData(int pastkeys, int futurekeys, int resolution, int datadim)
        {
            data = new float[datadim];
            PastKeys = pastkeys;
            FutureKeys = futurekeys;
            Resolution = resolution;
        }

        public int GetDataLength
        {
            get { return data.Length; }
        }

        public void Feed(float value)
        {
            Pivot += 1;
            data[Pivot] = value;
        }

        public void Feed(float[] values)
        {
            for (int i = 0; i < values.Length; i++)
            {
                Feed(values[i]);
            }
        }

        public void Feed(Vector3 vector)
        {
            Feed(vector.x);
            Feed(vector.y);
            Feed(vector.z);
        }

        public void FeedXZ(Vector3 vector)
        {
            Feed(vector.x);
            Feed(vector.z);
        }
    }
}