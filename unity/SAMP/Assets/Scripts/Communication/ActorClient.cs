using System.Collections;
using System;
using System.Text;
using System.Net;
using System.Net.Sockets;
using UnityEngine;

namespace Assets.Scripts.Communication
{
    public class ActorClient : Client
    {
        private int FailTimes = 0;

        // Use this for initialization
        void Start()
        {

        }

        // Update is called once per frame
        void Update()
        {

        }

        public float[] receivedata()
        {
            // receive length
            byte[] bytes = new byte[4];
            // return bytes length used
            int IndUsedBytes = ClientSocket.Receive(bytes);
            float[] data_length = new float[1];
            Buffer.BlockCopy(bytes, 0, data_length, 0, IndUsedBytes);
            int length = (int)data_length[0];

            // receive data
            float[] data_received = new float[length];
            int UsedBytesAccumulated = 0;
            int next_bytes_length = length * 4 > 1024 ? 1024 : length * 4;
            while (true)
            {
                byte[] data_bytes = new byte[next_bytes_length];
                IndUsedBytes = ClientSocket.Receive(data_bytes);
                Buffer.BlockCopy(data_bytes, 0, data_received, UsedBytesAccumulated, IndUsedBytes);
                UsedBytesAccumulated += IndUsedBytes;
                if (UsedBytesAccumulated == length * 4)
                {
                    break;
                }
                next_bytes_length = length * 4 - UsedBytesAccumulated > 1024 ? 1024 : length * 4 - UsedBytesAccumulated;
            }

            Debug.Log($"Successfully receive {length} data!");
            return data_received;
        }

        public ReceiveFrames SendAndReceive(CollectFrames data, int receive_dim)
        {
            if (BuildClient())
            {
                FailTimes = 0;
                var byteArrayLength = new byte[4];
                int[] ArrayLength = new int[1];
                ArrayLength[0] = data.GetDataLength;
                Buffer.BlockCopy(ArrayLength, 0, byteArrayLength, 0, 4);
                ClientSocket.Send(byteArrayLength);

                // Convert float array to byte array
                var byteArray = new byte[data.GetDataLength * 4];
                Buffer.BlockCopy(data.GetData(), 0, byteArray, 0, byteArray.Length);

                // Send
                ClientSocket.Send(byteArray);

                float[] keypose = receivedata();
                float[] keypose_t = receivedata();

                float[] pose = receivedata();
                int T = data.GetFramesNum;
                ReceiveFrames receive_frames = new ReceiveFrames(receive_dim);
                receive_frames.SetKey(keypose, keypose_t);
                receive_frames.SetData(pose, T);

                ClientSocket.Close();


                return receive_frames;
            }
            FailTimes += 1;
            return null;
        }

        public int GetFailedTimes()
        {
            return FailTimes;
        }
    }
}