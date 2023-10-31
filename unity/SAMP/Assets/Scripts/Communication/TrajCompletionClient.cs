using System.Collections;
using System;
using System.Text;
using System.Net;
using System.Net.Sockets;
using UnityEngine;

namespace Assets.Scripts.Communication
{
    public class TrajCompletionClient : Client
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

        public ReceiveData[] SendAndReceiveLandmark(float[] data, int receive_dim, int port)
        {
            if (BuildClient(port) && (data != null))
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

                float[] landmark_t = receivedata();
                float[] landmark_float = receivedata();
                float[] landmark_state_float = receivedata();

                ReceiveData landmark = new ReceiveData(4);
                ReceiveData landmark_state = new ReceiveData(receive_dim);
                landmark_state.SetData(landmark_state_float, landmark_t);
                landmark.SetData(landmark_float, landmark_t);
                ReceiveData[] all_data = new ReceiveData[2];
                all_data[0] = landmark;
                all_data[1] = landmark_state;

                ClientSocket.Close();

                return all_data;
            }
            FailTimes += 1;
            return null;
        }

        public ReceiveData[] SendAndReceiveTraj(float[] data, int receive_dim, int port)
        {
            if (BuildClient(port) && (data != null))
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

                float[] traj_float = receivedata();
                float[] traj_state_float = receivedata();

                ReceiveData traj = new ReceiveData(4);
                ReceiveData traj_state = new ReceiveData(receive_dim);
                traj.SetData(traj_float);
                traj_state.SetData(traj_state_float);
                ReceiveData[] all_data = new ReceiveData[2];
                all_data[0] = traj;
                all_data[1] = traj_state;

                ClientSocket.Close();

                return all_data;
            }
            FailTimes += 1;
            return null;
        }

        public ReceiveData SendAndReceivePose(float[] data, int receive_dim, int port)
        {
            if (BuildClient(port) && (data != null))
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

                ReceiveData pose = new ReceiveData(receive_dim);
                pose.SetData(pose_float);
                ClientSocket.Close();

                return pose;
            }
            FailTimes += 1;
            return null;
        }

        public ReceiveData[] SendAndReceivePoseContact(float[] data, int receive_dim, int port)
        {
            if (BuildClient(port) && (data != null))
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
                float[] contact_float = receivedata();

                ReceiveData pose = new ReceiveData(receive_dim);
                pose.SetData(pose_float);
                ReceiveData contact = new ReceiveData(5);
                contact.SetData(contact_float);
                ReceiveData[] all_data = new ReceiveData[2];
                all_data[0] = pose;
                all_data[1] = contact;
                ClientSocket.Close();

                return all_data;
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
