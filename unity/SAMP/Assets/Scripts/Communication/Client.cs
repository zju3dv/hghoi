using System.Collections;
using System;
using System.Text;
using System.Net;
using System.Net.Sockets;
using UnityEngine;

namespace Assets.Scripts.Communication
{
    public class Client : MonoBehaviour
    {
        public string Server = "10.76.2.110";
        public int port = 3457;

        protected Socket ClientSocket = null;
        protected IPEndPoint ServerIpEndPoint;

        // Use this for initialization
        void Start()
        {
            float[] data = new float[10];
            data[0] = 1f;
        }

        // Update is called once per frame
        void Update()
        {
        }

        public bool BuildClient(int out_port)
        {
            ClientSocket = new Socket(SocketType.Stream, ProtocolType.Tcp);
            ServerIpEndPoint = new IPEndPoint(IPAddress.Parse(Server), out_port);
            try
            {
                ClientSocket.Connect(ServerIpEndPoint);
            }
            catch (SocketException e)
            {
                Debug.Log(e);
                Debug.Log($"Failed to connect to server {Server}:{out_port}");
                return false;
            }
            if (!ClientSocket.Connected)
            {
                Debug.Log($"Failed to connect to server {Server}:{out_port}");
                return false;
            }
            //Debug.Log($"Connect to server {Server}:{port}");
            return true;
        }

        public bool BuildClient()
        {
            ClientSocket = new Socket(SocketType.Stream, ProtocolType.Tcp);
            ServerIpEndPoint = new IPEndPoint(IPAddress.Parse(Server), port);
            try
            {
                ClientSocket.Connect(ServerIpEndPoint);
            }
            catch (SocketException e)
            {
                Debug.Log(e);
                Debug.Log($"Failed to connect to server {Server}:{port}");
                return false;
            }
            if (!ClientSocket.Connected)
            {
                Debug.Log($"Failed to connect to server {Server}:{port}");
                return false;
            }
            //Debug.Log($"Connect to server {Server}:{port}");
            return true;
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
    }
}