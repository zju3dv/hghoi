using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using DeepLearning;
using Assets.Scripts.Communication;

public class System_Demo : SAMP_Demo
{
    public TrajCompletionClient client;
    public Actor KeyPoseActor1;
    public Actor KeyPoseActor2;
    public Actor GoalPoseActor;

    public SendData goalposesensor;

    public ReceiveData GoalPoseData;
    public ReceiveData GoalContactData;

    public bool ReceiveData = false;
    public bool Playing = false;
    public bool ShowEnvsmall = false;
    public bool ShowEnvbig = false;
    public bool ShowMilestone = false;

    public int GoalposePort = 3417;
    public int GoalposeSensorDims = 2048 + 5;


    public int GoalSensorDims = 5288;
    public int LandmarkSensorDims = 2497;
    public int StaticposeSensorDims = 2485;
    public int TrajSensorDims = 2749;
    public int TrajDims = 122;
    public int PoseDims = 264;

    public int LandmarkPort = 3457;
    public int TrajPort = 3447;
    public int StaticPosePort = 3467;
    public int TrajPosePort = 3497;

    public SendData goalsensor;
    public SendData[] landmarksensor;
    public SendData[] landmarkposesensor;
    public SendData[] trajsensor;
    public SendData startpose;
    public SendData idlepose;

    public ReceiveData TrajData;
    public ReceiveData TrajStateData;
    public ReceiveData LandMarkData;
    public ReceiveData LandMarkStateData;
    public ReceiveData StaticPoseData;
    public ReceiveData TrajPoseData;

    public UI_Traj UI;

    protected Matrix4x4 StartRoot;
    protected Matrix4x4 WholeGoal;
    protected Matrix4x4 WholeGoalGround;
    protected Matrix4x4 Idlegoal;

    protected CylinderMap Environment_small;
    protected CylinderMap Environment_big;

    protected override void Setup()
    {
        base.Setup();
        idlepose = new SendData(PoseDims);
        FeedPose(Actor, ref idlepose, RootSeries.Transformations[TimeSeries.Pivot]);
        Idlegoal = Matrix4x4.identity;
        Environment_small = new CylinderMap(2f, 5, 5, true);
        Environment_big = new CylinderMap(4f, 5, 5, true);

    }

    public virtual void ClearSensor()
    {
        goalposesensor = new SendData(GoalposeSensorDims);
        GoalPoseData = new ReceiveData(PoseDims);
        GoalContactData = new ReceiveData(5);
        goalsensor = new SendData(GoalSensorDims);
        landmarksensor = new SendData[0];
        landmarkposesensor = new SendData[0];
        trajsensor = new SendData[0];
        startpose = new SendData(PoseDims);
        TrajData = new ReceiveData(4);
        TrajStateData = new ReceiveData(TrajDims);
        LandMarkData = new ReceiveData(4);
        LandMarkStateData = new ReceiveData(TrajDims);
        StaticPoseData = new ReceiveData(PoseDims);
        TrajPoseData = new ReceiveData(PoseDims);
    }

    public virtual void ClearTransform()
    {
        StartRoot = Matrix4x4.zero;
        WholeGoal = Matrix4x4.zero;
        WholeGoalGround = Matrix4x4.zero;
    }

    public virtual void InteractionSense()
    {
        //Sense goal geometry
        Interaction interaction;
        interaction = Controller.ProjectionInteraction != null ? Controller.ProjectionInteraction : Controller.GetClosestInteraction(transform);

        Vector3 TargetPosition;
        Vector3 TargetDirection;
        if (interaction != null)
        {
            if (UseGoalNet)
            {
                Debug.Log("GoalNet!");
                goalNet.PredictGoal(interaction, "Hips_Pred");
                TargetPosition = interaction.GetPredContact("Hips_Pred").GetPosition();
                TargetDirection = interaction.GetPredContact("Hips_Pred").GetForward();
            }
            else
            {
                TargetPosition = interaction.GetContact("Hips").GetPosition();
                TargetDirection = interaction.GetContact("Hips").GetForward();
            }

            WholeGoal = Matrix4x4.TRS(TargetPosition, Quaternion.LookRotation(TargetDirection, Vector3.up), Vector3.one);
            TargetPosition.y = 0f;
            WholeGoalGround = Matrix4x4.TRS(TargetPosition, Quaternion.LookRotation(TargetDirection, Vector3.up), Vector3.one);
        }
        else { Debug.Log("No interaction, wrong results!"); }

        Geometry.Setup(Geometry.Resolution);
        Geometry.Sense(interaction.GetCenter(), LayerMask.GetMask("Interaction"), interaction.GetExtents(), InteractionSmoothing);
        Environment_small.Sense(WholeGoalGround, LayerMask.GetMask("Interaction"));
        Environment_big.Sense(WholeGoalGround, LayerMask.GetMask("Interaction"));
    }

    public virtual void SampleGoal()
    {
        StartRoot = RootSeries.GetTransformation(TimeSeries.Pivot);
        if (SavePrediction)
        {
            WholeGoal = Idlegoal;
            WholeGoalGround = WholeGoal;
        }
        else
        {
            Vector3 pos = Vector3.zero;
            pos.x = 2f;
            pos.z = 3f;
            pos = pos.GetRelativePositionFrom(RootSeries.GetTransformation(TimeSeries.Pivot));
            WholeGoal = Matrix4x4.TRS(pos, Quaternion.LookRotation(WholeGoal.GetForward(), Vector3.up), Vector3.one);
            WholeGoalGround = WholeGoal;
        }
    }

    public float[] GetActionArray(string goalaction)
    {
        float[] goalaction_array = { 0, 0, 0, 0, 0 };
        switch (goalaction)
        {
            case "Idle":
                goalaction_array[0] = 1f;
                break;
            case "Walk":
                goalaction_array[1] = 1f;
                break;
            case "Run":
                goalaction_array[2] = 1f;
                break;
            case "Sit":
                goalaction_array[3] = 1f;
                break;
            case "Liedown":
                goalaction_array[4] = 1f;
                break;
        }

        return goalaction_array;
    }

    public virtual void GoalposeSense(string goalaction)
    {
        goalposesensor.Feed(GetActionArray(goalaction));

        //Get Root
        Matrix4x4 root = RootSeries.Transformations[TimeSeries.Pivot];
        if (StartRoot == Matrix4x4.zero)
        {
            StartRoot = root;
        }
        if (goalaction == "Sit" || goalaction == "Liedown")
        {
            InteractionSense();
        }
        goalposesensor.Feed(Environment_small.Occupancies);
        goalposesensor.Feed(Environment_big.Occupancies);
        for (int i = 0; i < Geometry.Points.Length; i++)
        {
            goalposesensor.Feed(Geometry.References[i].GetRelativePositionTo(root));
            goalposesensor.Feed(Geometry.Occupancies[i]);
        }
    }

    public virtual void GoalSense(string goalaction)
    {
        FeedPose(Actor, ref goalsensor, RootSeries.GetTransformation(TimeSeries.Pivot));

        if (goalaction == "Idle")
        {
            SampleGoal();
            goalsensor.Feed(idlepose.Data);
            GoalPoseData.SetData(idlepose.Data);
        }
        else
        {
            goalsensor.Feed(GoalPoseData.ReadData().data);
        }
        Matrix4x4 root = StartRoot;

        //Input Geometry
        for (int i = 0; i < Geometry.Points.Length; i++)
        {
            goalsensor.Feed(Geometry.References[i].GetRelativePositionTo(root));
            goalsensor.Feed(Geometry.Occupancies[i]);
        }

        goalsensor.Feed(root.GetPosition());
        goalsensor.Feed(root.GetForward());
        goalsensor.Feed(root.GetUp());
        goalsensor.Feed(root.GetRight());


        Matrix4x4 goal = WholeGoalGround;
        goalsensor.Feed(goal.GetPosition());
        goalsensor.Feed(goal.GetForward());
        goalsensor.Feed(goal.GetUp());
        goalsensor.Feed(goal.GetRight());

        //Input Geometry
        for (int i = 0; i < Geometry.Points.Length; i++)
        {
            goalsensor.Feed(Geometry.References[i].GetRelativePositionTo(goal));
            goalsensor.Feed(Geometry.Occupancies[i]);
        }

        //Input Root env
        Environment_big.Sense(root, LayerMask.GetMask("Interaction"));
        goalsensor.Feed(Environment_big.Occupancies);
        //Input Goal env
        Environment_big.Sense(WholeGoalGround, LayerMask.GetMask("Interaction"));
        goalsensor.Feed(Environment_big.Occupancies);

        goalsensor.Feed(StyleSeries.Values[TimeSeries.Pivot]);
        goalsensor.Feed(GetActionArray(goalaction));
    }

    public virtual void LandmarkSense()
    {
        int T = LandMarkData.GetFramesNum;
        Matrix4x4[] landmark = new Matrix4x4[T];
        landmarksensor = new SendData[T];
        for (int i = 0; i < T; i++)
        {
            landmarksensor[i] = new SendData(LandmarkSensorDims);
            landmarksensor[i].Feed(LandMarkStateData.GetData(i).data);
            Vector3 pos = LandMarkData.GetData(i).ReadXZ();
            Vector3 dir = LandMarkData.GetData(i).ReadXZ();
            Matrix4x4 subgoal = Matrix4x4.TRS(pos, Quaternion.LookRotation(dir, Vector3.up), Vector3.one);
            landmark[i] = subgoal;
            subgoal = subgoal.GetRelativeTransformationFrom(StartRoot);
            for (int j = 0; j < Geometry.Points.Length; j++)
            {
                landmarksensor[i].Feed(Geometry.References[j].GetRelativePositionTo(subgoal));
                landmarksensor[i].Feed(Geometry.Occupancies[j]);
            }
            landmarksensor[i].Feed(subgoal.GetPosition());
            landmarksensor[i].Feed(subgoal.GetForward());
            landmarksensor[i].Feed(subgoal.GetUp());
            landmarksensor[i].Feed(subgoal.GetRight());

            Environment_big.Sense(subgoal, LayerMask.GetMask("Interaction"));
            landmarksensor[i].Feed(Environment_big.Occupancies);
        }
    }

    public virtual void LandmarkPoseSense()
    {
        int T = LandMarkData.GetFramesNum;
        landmarkposesensor = new SendData[T];
        for (int i = 0; i < T; i++)
        {
            landmarkposesensor[i] = new SendData(StaticposeSensorDims);

            int landmark_i = LandMarkStateData.Data_t[i] - 1;
            landmarkposesensor[i].Feed(TrajStateData.GetData(landmark_i).data);

            Vector3 pos = LandMarkData.GetData(i).ReadXZ();
            Vector3 dir = LandMarkData.GetData(i).ReadXZ();
            Matrix4x4 subgoal = Matrix4x4.TRS(pos, Quaternion.LookRotation(dir, Vector3.up), Vector3.one);
            subgoal = subgoal.GetRelativeTransformationFrom(StartRoot);
            for (int j = 0; j < Geometry.Points.Length; j++)
            {
                landmarkposesensor[i].Feed(Geometry.References[j].GetRelativePositionTo(subgoal));
                landmarkposesensor[i].Feed(Geometry.Occupancies[j]);
            }
            Environment_big.Sense(subgoal, LayerMask.GetMask("Interaction"));
            landmarkposesensor[i].Feed(Environment_big.Occupancies);

        }

        FeedPose(Actor, ref startpose, StartRoot);
    }

    public virtual void TrajSense()
    {
        int T = TrajData.GetFramesNum;
        trajsensor = new SendData[T];
        for (int i = 0; i < T; i++)
        {
            trajsensor[i] = new SendData(TrajSensorDims);
            trajsensor[i].Feed(StaticPoseData.ReadDataPrevioust(i + 1).data);
            trajsensor[i].Feed(TrajStateData.GetData(i).data);
            Vector3 pos = TrajData.GetData(i).ReadXZ();
            Vector3 dir = TrajData.GetData(i).ReadXZ();
            Matrix4x4 subgoal = Matrix4x4.TRS(pos, Quaternion.LookRotation(dir, Vector3.up), Vector3.one);
            subgoal = subgoal.GetRelativeTransformationFrom(StartRoot);
            for (int j = 0; j < Geometry.Points.Length; j++)
            {
                trajsensor[i].Feed(Geometry.References[j].GetRelativePositionTo(subgoal));
                trajsensor[i].Feed(Geometry.Occupancies[j]);
            }
            Environment_big.Sense(subgoal, LayerMask.GetMask("Interaction"));
            trajsensor[i].Feed(Environment_big.Occupancies);
        }
    }

    public float[] SendData2Float(SendData data)
    {
        return data.Data;
    }

    public float[] SendData2Float(SendData data, SendData[] dataarray)
    {
        int T = data.GetDataLength + dataarray.Length * dataarray[0].GetDataLength;
        float[] float_data = new float[T];
        for (int i = 0; i < data.GetDataLength; i++)
        {
            float_data[i] = data.Data[i];
        }
        for (int i = 0; i < dataarray.Length; i++)
        {
            for (int j = 0; j < dataarray[i].GetDataLength; j++)
            {
                float_data[data.GetDataLength + i * dataarray[i].GetDataLength + j] = dataarray[i].Data[j];
            }
        }
        return float_data;
    }


    public float[] SendData2Float(SendData[] data)
    {
        int T = data.Length * data[0].GetDataLength;
        float[] float_data = new float[T];
        for (int i = 0; i < data.Length; i++)
        {
            for (int j = 0; j < data[i].GetDataLength; j++)
            {
                float_data[i * data[i].GetDataLength + j] = data[i].Data[j];
            }
        }
        return float_data;
    }

    public float[] SendData2Float(SendData data, ReceiveData data_rec, SendData[] dataarray)
    {
        int T = data_rec.ReadData().GetDataLength + data.GetDataLength + dataarray.Length * dataarray[0].GetDataLength;
        float[] float_data = new float[T];
        int ind = 0;

        for (int i = 0; i < data.GetDataLength; i++)
        {
            float_data[ind] = data.Data[i];
            ind++;
        }
        for (int i = 0; i < data_rec.ReadData().GetDataLength; i++)
        {
            float_data[ind] = data_rec.ReadData().data[i];
            ind++;
        }

        for (int i = 0; i < dataarray.Length; i++)
        {
            for (int j = 0; j < dataarray[i].GetDataLength; j++)
            {
                float_data[ind] = dataarray[i].Data[j];
                ind++;
            }
        }
        return float_data;
    }

    public virtual void ReceiveGoalPose()
    {
        float[] goalposedata_float = SendData2Float(goalposesensor);
        ReceiveData[] goalpose;
        goalpose = client.SendAndReceivePoseContact(goalposedata_float, PoseDims, GoalposePort);
        GoalPoseData = goalpose[0];
        GoalContactData = goalpose[1];

        SendData new_goalpose = new SendData(PoseDims);
        AssignPose(ref GoalPoseActor, GoalPoseData.ReadData(), WholeGoalGround, true);
        Transform pelvis = GoalPoseActor.FindBone("pelvis").Transform;
        Vector3 pelvisPos = pelvis.transform.position;
        pelvisPos.y = (WholeGoal.GetPosition().y + pelvisPos.y) / 2;
        pelvis.position = pelvisPos;
        PostprocessStaticPose(ref GoalPoseActor, GoalContactData.ReadData().data);
        FeedPose(GoalPoseActor, ref new_goalpose, WholeGoalGround);
        GoalPoseData.SetData(new_goalpose.Data);
    }

    public virtual void ReceiveLandmark()
    {
        float[] goaldata_float = SendData2Float(goalsensor);
        ReceiveData[] receive_landmark = client.SendAndReceiveLandmark(goaldata_float, TrajDims, LandmarkPort);
        LandMarkData = receive_landmark[0];
        LandMarkStateData = receive_landmark[1];
    }

    public virtual void ReceiveTraj()
    {
        float[] landmarkdata_float = SendData2Float(landmarksensor);
        ReceiveData[] receive_traj = client.SendAndReceiveTraj(landmarkdata_float, TrajDims, TrajPort);
        TrajData = receive_traj[0];
        TrajStateData = receive_traj[1];
    }

    public virtual void ReceiveLandmarkPose()
    {
        float[] landmarkpose_float = SendData2Float(startpose, GoalPoseData, landmarkposesensor);
        StaticPoseData = client.SendAndReceivePose(landmarkpose_float, PoseDims, StaticPosePort);
        StaticPoseData.SetT(LandMarkData.GetT());
        Matrix4x4 root = KeyPoseActor1.transform.GetWorldMatrix();

        for (int i = 0; i < StaticPoseData.GetFramesNum; i++)
        {
            // Postprocess to obtain valid pose from smpl instead of predictions
            AssignPose(ref KeyPoseActor1, StaticPoseData.GetData(i), root, true);
            int t = StaticPoseData.GetT()[i];
            float[] contacts = TrajStateData.GetData(t - 1).Read(117, 122);
            for (int j = 0; j < contacts.Length; j++)
            {
                contacts[j] = Mathf.Clamp(contacts[j], 0f, 1f);
            }
            //PostprocessStaticPose(ref KeyPoseActor1, contacts);

            SendData new_staticpose = new SendData(PoseDims);
            FeedPose(KeyPoseActor1, ref new_staticpose, root);
            StaticPoseData.SetData_i(new_staticpose.Data, i);
        }
    }

    public virtual void ReceiveTrajPose()
    {
        float[] traj_float = SendData2Float(trajsensor);
        TrajPoseData = client.SendAndReceivePose(traj_float, PoseDims, TrajPosePort);
    }

    public virtual int PredictInteraction(string goalaction)
    {
        Debug.Log($"Predict with action {goalaction}");
        ClearSensor();
        if (goalaction == "Sit" || goalaction == "Liedown")
        {
            ClearTransform();
        }
        GoalposeSense(goalaction);
        ReceiveGoalPose();
        GoalSense(goalaction);
        ReceiveLandmark();
        LandmarkSense();
        ReceiveTraj();
        LandmarkPoseSense();
        ReceiveLandmarkPose();
        TrajSense();
        ReceiveTrajPose();
        UI.ResetSlider();
        ReceiveData = true;
        return TrajData.GetFramesNum;
    }

    public void ReadGoalPose()
    {
        AssignPose(ref GoalPoseActor, GoalPoseData.ReadData(), WholeGoalGround);
        //AssignPose(ref GoalPoseActor, GoalPoseData.ReadData(), GoalPoseActor.transform.GetWorldMatrix());
    }

    public void ReadTraj()
    {
        Vector3 root_pos = TrajData.ReadData().ReadXZ();
        Vector3 root_dir = TrajData.ReadData().ReadXZ();
        Matrix4x4 root = Matrix4x4.TRS(root_pos, Quaternion.LookRotation(root_dir, Vector3.up), Vector3.one);
        root = root.GetRelativeTransformationFrom(StartRoot);
        transform.position = root.GetPosition();
        transform.rotation = root.GetRotation();
    }

    public virtual void ReadTrajState()
    {
        int pivot = TrajData.Pivot;
        Vector3 root_pos = TrajData.ReadData().ReadXZ();
        Vector3 root_dir = TrajData.ReadData().ReadXZ();
        Matrix4x4 root = Matrix4x4.TRS(root_pos, Quaternion.LookRotation(root_dir, Vector3.up), Vector3.one);
        root = root.GetRelativeTransformationFrom(StartRoot);

        TrajStateData.Pivot = pivot;
        for (int i = 0; i < TimeSeries.KeyCount; i++)
        {
            TimeSeries.Sample sample = TimeSeries.GetKey(i);
            Vector3 pos = TrajStateData.ReadData().ReadXZ().GetRelativePositionFrom(root);
            Vector3 dir = TrajStateData.ReadData().ReadXZ().normalized.GetRelativeDirectionFrom(root);
            float[] styles = TrajStateData.ReadData().Read(StyleSeries.Styles.Length);

            RootSeries.SetPosition(sample.Index, pos);
            RootSeries.SetDirection(sample.Index, dir);
            //M: edit future only
            for (int j = 0; j < styles.Length; j++)
            {
                styles[j] = Mathf.Clamp(styles[j], 0f, 1f);
            }
            StyleSeries.Values[sample.Index] = styles;
        }
        float[] contacts = TrajStateData.ReadData().Read(ContactSeries.Bones.Length);
        for (int i = 0; i < contacts.Length; i++)
        {
            contacts[i] = Mathf.Clamp(contacts[i], 0f, 1f);
        }

        for (int i = 0; i < TimeSeries.Pivot; i++)
        {
            for (int j = 0; j < ContactSeries.Bones.Length; j++)
            {
                ContactSeries.Values[i][j] = ContactSeries.Values[i + 1][j];
            }
        }
        ContactSeries.Values[TimeSeries.Pivot] = contacts;

        for (int i = 0; i < TimeSeries.KeyCount; i++)
        {
            TimeSeries.Sample sample = TimeSeries.GetKey(i);
            GoalSeries.Transformations[sample.Index] = WholeGoal;
        }
    }

    public void ReadLandMark()
    {
        int pivot = TrajData.Pivot;
        Vector3 pos1 = LandMarkData.ReadDataPrevioust(pivot).ReadXZ();
        Vector3 dir1 = LandMarkData.ReadDataPrevioust(pivot).ReadXZ();
        Vector3 pos2 = LandMarkData.ReadDataAftert(pivot).ReadXZ();
        Vector3 dir2 = LandMarkData.ReadDataAftert(pivot).ReadXZ();

        Matrix4x4 root1 = Matrix4x4.TRS(pos1, Quaternion.LookRotation(dir1, Vector3.up), Vector3.one);
        Matrix4x4 root2 = Matrix4x4.TRS(pos2, Quaternion.LookRotation(dir2, Vector3.up), Vector3.one);
        root1 = root1.GetRelativeTransformationFrom(StartRoot);
        root2 = root2.GetRelativeTransformationFrom(StartRoot);
        KeyPoseActor1.transform.position = root1.GetPosition();
        KeyPoseActor1.transform.rotation = root1.GetRotation();
        KeyPoseActor2.transform.position = root2.GetPosition();
        KeyPoseActor2.transform.rotation = root2.GetRotation();
    }

    public void ReadLandMarkStaticPose()
    {
        int pivot = TrajData.Pivot;
        Vector3 pos1 = LandMarkData.ReadDataPrevioust(pivot).ReadXZ();
        Vector3 dir1 = LandMarkData.ReadDataPrevioust(pivot).ReadXZ();
        Vector3 pos2 = LandMarkData.ReadDataAftert(pivot).ReadXZ();
        Vector3 dir2 = LandMarkData.ReadDataAftert(pivot).ReadXZ();

        Matrix4x4 root1 = Matrix4x4.TRS(pos1, Quaternion.LookRotation(dir1, Vector3.up), Vector3.one);
        Matrix4x4 root2 = Matrix4x4.TRS(pos2, Quaternion.LookRotation(dir2, Vector3.up), Vector3.one);
        root1 = root1.GetRelativeTransformationFrom(StartRoot);
        root2 = root2.GetRelativeTransformationFrom(StartRoot);

        //root1 = KeyPoseActor1.transform.GetWorldMatrix();
        //root2 = KeyPoseActor2.transform.GetWorldMatrix();

        AssignPose(ref KeyPoseActor1, StaticPoseData.ReadDataPrevioust(pivot), root1);
        AssignPose(ref KeyPoseActor2, StaticPoseData.ReadDataAftert(pivot), root2);
    }

    public virtual void ReadPose()
    {
        Vector3 root_pos = TrajData.ReadData().ReadXZ();
        Vector3 root_dir = TrajData.ReadData().ReadXZ();
        TrajData.ReadData().Reset();
        Matrix4x4 root = Matrix4x4.TRS(root_pos, Quaternion.LookRotation(root_dir, Vector3.up), Vector3.one);
        root = root.GetRelativeTransformationFrom(StartRoot);

        //Get Root
        int pivot = TrajData.Pivot;
        TrajPoseData.Pivot = pivot;

        AssignPose(ref Actor, TrajPoseData.ReadData(), root);
        //PostprocessStaticPose(ref Actor, ContactSeries.Values[TimeSeries.Pivot]);


    }

    public void AssignPose(ref Actor actor, ReceiveData.FrameData framepose, Matrix4x4 root, bool onlyrot = false)
    {
        //Read Posture
        Vector3[] positions = new Vector3[actor.Bones.Length];
        Vector3[] forwards = new Vector3[actor.Bones.Length];
        Vector3[] upwards = new Vector3[actor.Bones.Length];
        Vector3[] velocities = new Vector3[actor.Bones.Length];
        for (int i = 0; i < actor.Bones.Length; i++)
        {
            Vector3 position = framepose.ReadVector3().GetRelativePositionFrom(root);
            Vector3 forward = framepose.ReadVector3().normalized.GetRelativeDirectionFrom(root);
            Vector3 upward = framepose.ReadVector3().normalized.GetRelativeDirectionFrom(root);
            Vector3 velocity = framepose.ReadVector3().GetRelativeDirectionFrom(root);
            positions[i] = position;
            forwards[i] = forward;
            upwards[i] = upward;
            velocities[i] = velocity;
        }
        framepose.Reset();

        for (int i = 0; i < actor.Bones.Length; i++)
        {

            if (onlyrot)
            {
                actor.Bones[0].Transform.position = positions[0];
            }
            else
            {
                actor.Bones[i].Transform.position = positions[i];
            }
            actor.Bones[i].Transform.rotation = Quaternion.LookRotation(forwards[i], upwards[i]);
            actor.Bones[i].Velocity = velocities[i];
            actor.Bones[i].ApplyLength();
        }
    }

    protected void FeedPose(Matrix4x4[] Poseture, Vector3[] Velocites, Matrix4x4 root, ref SendData send_data)
    {
        for (int i = 0; i < Poseture.Length; i++)
        {
            send_data.Feed(Poseture[i].GetPosition().GetRelativePositionTo(root));
            send_data.Feed(Poseture[i].GetForward().GetRelativeDirectionTo(root));
            send_data.Feed(Poseture[i].GetUp().GetRelativeDirectionTo(root));
            send_data.Feed(Velocites[i].GetRelativeDirectionTo(root));
        }
    }

    protected void FeedPose(Actor actor, ref SendData send_data, Matrix4x4 root)
    {
        for (int i = 0; i < actor.Bones.Length; i++)
        {
            send_data.Feed(actor.Bones[i].Transform.GetWorldMatrix().GetPosition().GetRelativePositionTo(root));
            send_data.Feed(actor.Bones[i].Transform.GetWorldMatrix().GetForward().GetRelativeDirectionTo(root));
            send_data.Feed(actor.Bones[i].Transform.GetWorldMatrix().GetUp().GetRelativeDirectionTo(root));
            send_data.Feed(actor.Bones[i].Velocity.GetRelativeDirectionTo(root));
        }
    }

    protected void PostprocessStaticPose(ref Actor actor, float[] contacts)
    {

        //RightFootIK = UltimateIK.BuildModel(actor.FindTransform("right_hip"), actor.GetBoneTransforms(ContactSeries.Bones[3]));
        //LeftFootIK = UltimateIK.BuildModel(actor.FindTransform("left_hip"), actor.GetBoneTransforms(ContactSeries.Bones[4]));
        //Matrix4x4 rightFoot = actor.GetBoneTransformation(ContactSeries.Bones[3]);
        //Matrix4x4 leftFoot = actor.GetBoneTransformation(ContactSeries.Bones[4]);
        //RightFootIK.Objectives[0].SetTarget(rightFoot.GetPosition(), 1f - contacts[3]);
        //RightFootIK.Objectives[0].SetTarget(rightFoot.GetRotation());
        //LeftFootIK.Objectives[0].SetTarget(leftFoot.GetPosition(), 1f - contacts[4]);
        //LeftFootIK.Objectives[0].SetTarget(leftFoot.GetRotation());
        //RightFootIK.Solve();
        //LeftFootIK.Solve();

        Transform rightToe = actor.FindBone("right_foot").Transform;
        Vector3 rightPos = rightToe.transform.position;
        float deltay_right = Mathf.Max(rightPos.y, 0.02f) - rightPos.y;
        //rightPos.y = Mathf.Max(rightPos.y, 0.02f);
        //rightToe.position = rightPos;

        Transform leftToe = actor.FindBone("left_foot").Transform;
        Vector3 leftPos = leftToe.transform.position;
        float deltay_left = Mathf.Max(leftPos.y, 0.02f) - leftPos.y;
        //leftPos.y = Mathf.Max(leftPos.y, 0.02f);
        //leftToe.position = leftPos;

        Transform pelvis = actor.FindBone("pelvis").Transform;
        Vector3 pelvisPos = pelvis.transform.position;
        pelvisPos.y += (deltay_left + deltay_right) / 2;
        pelvis.position = pelvisPos;
    }

    public void ReadEnv()
    {
        Vector3 root_pos = TrajData.ReadData().ReadXZ();
        Vector3 root_dir = TrajData.ReadData().ReadXZ();
        Matrix4x4 root = Matrix4x4.TRS(root_pos, Quaternion.LookRotation(root_dir, Vector3.up), Vector3.one);
        root = root.GetRelativeTransformationFrom(StartRoot);

        Environment_big.Sense(root, LayerMask.GetMask("Interaction"));
    }

    protected override void OnRenderObjectDerived()
    {
        base.OnRenderObjectDerived();
        if (ShowEnvsmall)
        {
            Environment_small.Draw(UltiDraw.Mustard.Transparent(0.25f));
        }
        if (ShowEnvbig)
        {
            Environment_big.Draw(UltiDraw.Mustard.Transparent(0.25f));
        }
        if (ReceiveData)
        {
            LandMarkData.DrawMilestone(StartRoot);
        }
    }

    protected override void Read()
    {
        if (ReceiveData)
        {
            ReadGoalPose();
            ReadTraj();
            ReadLandMark();
            ReadTrajState();
            ReadLandMarkStaticPose();
            ReadPose();
            ReadEnv();
            WriteData();
        }
        else
        {
            base.Read();
        }
    }

    public override IEnumerator InteractWithObject(string act, Matrix4x4 End, Matrix4x4 Start, Matrix4x4[] Poseture, Vector3[] Velocities)
    {
        reinit_funcs(Start, Poseture, Velocities);
        IsInteracting = true;
        yield return new WaitForSeconds(2f);
        reinit_funcs(Start, Poseture, Velocities);
        idlepose = new SendData(PoseDims);
        FeedPose(Poseture, Velocities, Start, ref idlepose);
        Idlegoal = End;
        int FramesNum = PredictInteraction(act);
        Playing = true;
        SaveStatus = true;
        while (TrajData.Pivot != FramesNum - 1)
        {
            yield return new WaitForSeconds(0);
        }
        Playing = false;
        yield return new WaitForSeconds(3);
        FramesNum = PredictInteraction("Idle");
        Playing = true;
        while (TrajData.Pivot != FramesNum - 1)
        {
            yield return new WaitForSeconds(0);
        }
        Playing = false;
        SaveStatus = false;
        ReceiveData = false;
        TestScript.MotionNum++;
        TestScript.SetTesti();
        IsInteracting = false;
    }
}

