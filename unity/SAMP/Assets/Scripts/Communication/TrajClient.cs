using System.Collections;
using System;
using System.Text;
using System.Net;
using System.Net.Sockets;
using UnityEngine;

namespace Assets.Scripts.Communication
{
    public class TrajClient : Client
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

        public float[] SendAndReceive(float[] data)
        {
            if (BuildClient() && (data != null))
            {
                FailTimes = 0;
                var byteArrayLength = new byte[4];
                int[] ArrayLength = new int[1];
                ArrayLength[0] = data.Length;
                Buffer.BlockCopy(ArrayLength, 0, byteArrayLength, 0, 4);
                ClientSocket.Send(byteArrayLength);

                // Convert float array to byte array
                var byteArray = new byte[data.Length * 4];
                Buffer.BlockCopy(data, 0, byteArray, 0, byteArray.Length);

                // Send
                ClientSocket.Send(byteArray);

                float[] pose_float = receivedata();
                receivedata(); // for omega

                return pose_float;
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