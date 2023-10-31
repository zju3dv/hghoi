using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SendData
{
    public int DataDim;
    public float[] Data;
    public int Pivot = -1;

    public SendData(int datadim)
    {
        DataDim = datadim;
        Data = new float[datadim];
    }

    public int GetDataLength
    {
        get { return Data.Length; }
    }

    public void Feed(float value)
    {
        Pivot += 1;
        Data[Pivot] = value;
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