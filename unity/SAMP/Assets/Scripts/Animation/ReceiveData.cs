using System.Collections;
using System.Collections.Generic;
using UnityEngine;


public class ReceiveData
{

    public FrameData[] Data = new FrameData[0];
    public int[] Data_t = new int[0];
    public int Pivot = 0;
    public int Dim = 0;

    public ReceiveData(int datadim)
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
            { return Data.Length * Dim; }
        }
    }

    public FrameData ReadDataPrevioust(int t) 
    {
        for (int i = 0; i < Data_t.Length; i++) 
	    { 
            if (Data_t[i] > t) 
	        {
                return i > 0 ? Data[i - 1] : Data[0];
	        }
	    }
        return Data[Data_t.Length - 1];
    }

    public FrameData ReadDataAftert(int t)
    {
        for (int i = 0; i < Data_t.Length; i++) 
	    { 
            if (Data_t[i] > t) 
	        {
                return Data[i];
	        }
	    }
        return Data[Data_t.Length - 1];
    }


    public FrameData ReadData()
    {
        if (Pivot >= Data.Length)
        {
            Pivot = Data.Length - 1;
        }

        return Data[Pivot];
    }

    public FrameData GetData() 
    {
        Pivot += 1;
        if (Pivot >= Data.Length)
        {
            Pivot = Data.Length;
        }
        return Data[Pivot - 1];
    }

    public FrameData GetData(int pivot)
    {
        return Data[pivot];
    }

    public int[] GetT() 
    {
        return Data_t;
    }

    public void SetT(int[] whole_T) 
    {
        if (whole_T.Length != Data.Length) { return; }
        Data_t = new int[whole_T.Length];

        for (int i = 0; i < whole_T.Length; i++) 
	    {
            Data_t[i] = whole_T[i];
	    }
    }

    public void SetT(float[] whole_T) 
    {
        if (whole_T.Length != Data.Length) { return; }
        Data_t = new int[whole_T.Length];

        for (int i = 0; i < whole_T.Length; i++) 
	    {
            Data_t[i] = (int)whole_T[i];
	    }
    }

    public void SetData(float[] whole_data, float[] data_t) 
    { 
        int l = whole_data.Length / Dim;
        Data = new FrameData[l];
        Data_t = new int[l];
        int k = 0;
        for (int i = 0; i < l; i++)
        {
            float[] data = new float[Dim];
            for (int j = 0; j < data.Length; j++)
            {
                data[j] = whole_data[k];
                k++;
            }
            Data[i] = new FrameData(data);
            Data_t[i] = (int)data_t[i];
        }
    }

    public void SetData(float[] whole_data, int[] data_t)
    {
        int l = whole_data.Length / Dim;
        Data = new FrameData[l];
        Data_t = new int[l];
        int k = 0;
        for (int i = 0; i < l; i++)
        {
            float[] data = new float[Dim];
            for (int j = 0; j < data.Length; j++)
            {
                data[j] = whole_data[k];
                k++;
            }
            Data[i] = new FrameData(data);
            Data_t[i] = data_t[i];
        }
    }

    public void SetData(float[] whole_data) 
    { 
        int l = whole_data.Length / Dim;
        Data = new FrameData[l];
        Data_t = new int[l];
        int k = 0;
        for (int i = 0; i < l; i++)
        {
            float[] data = new float[Dim];
            for (int j = 0; j < data.Length; j++)
            {
                data[j] = whole_data[k];
                k++;
            }
            Data[i] = new FrameData(data);
            Data_t[i] = i;
        }
    }
      
    public void SetData(float[] whole_data, int l)
    {
        Data = new FrameData[l];
        Data_t = new int[l];
        int k = 0;
        for (int i = 0; i < l; i++)
        {
            float[] data = new float[Dim];
            for (int j = 0; j < data.Length; j++)
            {
                data[j] = whole_data[k];
                k++;
            }
            Data[i] = new FrameData(data);
            Data_t[i] = i;
        }
    }

    public void SetData_i(float[] onedata, int i)
    {
        FrameData new_data = new FrameData(onedata);
        Data[i] = new_data;
    }

    //For path
    public void DrawMilestone(Matrix4x4 root)
    {
        UltiDraw.Begin();
        for (int i = 0; i < GetFramesNum; i++)
        {
            Vector3 pos = GetData(i).ReadXZ().GetRelativePositionFrom(root);
            GetData(i).Reset();
            UltiDraw.DrawSphere(pos, Quaternion.identity, 0.2f, UltiDraw.Purple);
            if (i < GetFramesNum - 1)
            {
                Vector3 pos_ = GetData(i + 1).ReadXZ().GetRelativePositionFrom(root);
                GetData(i + 1).Reset();

                //UltiDraw.DrawLine(pos, pos_, 0.03f, UltiDraw.Red);
            }
        }
        UltiDraw.End();
    }

    public void DrawSubMilestone(Matrix4x4 root, int[] NoDraw)
    {
        UltiDraw.Begin();
        for (int i = 0; i < GetFramesNum; i += 1)
        {
            bool nodraw_flag = false;
            for (int j = 0; j < NoDraw.Length; j++)
            {
                if (i == NoDraw[j])
                {
                    nodraw_flag = true;
                    break;
                }
            }
            if (nodraw_flag)
            {
                continue;
            }

            Vector3 pos = GetData(i).ReadXZ().GetRelativePositionFrom(root);
            GetData(i).Reset();
            UltiDraw.DrawSphere(pos, Quaternion.identity, 0.2f, UltiDraw.Purple);
        }

        UltiDraw.End();
    }

    public void DrawPath(Matrix4x4 root)
    {
        UltiDraw.Begin();
        for (int i = 0; i < GetFramesNum; i += 5)
        {
            Vector3 pos = GetData(i).ReadXZ().GetRelativePositionFrom(root);
            GetData(i).Reset();
            UltiDraw.DrawSphere(pos, Quaternion.identity, 0.025f, UltiDraw.Black);
            if (i < GetFramesNum - 5)
            {
                Vector3 pos_ = GetData(i + 5).ReadXZ().GetRelativePositionFrom(root);
                GetData(i + 5).Reset();

                UltiDraw.DrawLine(pos, pos_, 0.025f, 0f, UltiDraw.Green.Transparent(0.75f));
            }
        }
        UltiDraw.End();
    }



    public void Reset()
    {
        Pivot = 0;
    }

    public class FrameData
    {
        public float[] data;
        public int Pivot = -1;

        public FrameData(float[] receive_data)
        {
            data = receive_data;
        }

        public int GetDataLength
        {
            get { return data.Length; }
        }

        public float Read()
        {
            if (Pivot == data.Length - 1)
            {
                Pivot = -1;
            }
            Pivot += 1;
            return data[Pivot];
        }

        public Vector3 ReadXZ()
        {
            return new Vector3(Read(), 0f, Read());
        }

        public Vector3 ReadVector3()
        {
            return new Vector3(Read(), Read(), Read());
        }

        public float[] Read(int count)
        {
            float[] values = new float[count];
            for (int i = 0; i < count; i++)
            {
                values[i] = Read();
            }
            return values;
        }

        public float[] Read(int start, int end)
        {
            float[] values = new float[end - start];
            for (int i = 0; i < (end - start); i++)
            {
                values[i] = data[start + i];
            }
            return values;
        }

        public void Reset()
        {
            Pivot = -1;
        }
    }
}
