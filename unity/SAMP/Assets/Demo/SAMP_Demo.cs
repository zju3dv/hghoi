using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using DeepLearning;

public class SAMP_Demo : NeuralAnimation
{
    public string[] InteractionObjectsNames = { };
    public int MultiInteractionFrame = 0;
    public bool ShowBiDirectional = true;
    public bool ShowRoot = true;
    public bool ShowGoal = true;
    public bool OnlyPoint = false;
    public bool ShowCurrent = true;
    public bool ShowContacts = false;
    public bool ShowInteraction = true;
    public bool ShowGUI = true;

    protected Controller Controller;
    protected TimeSeries TimeSeries;
    protected TimeSeries.Root RootSeries;
    protected TimeSeries.Style StyleSeries;
    protected TimeSeries.Goal GoalSeries;
    protected TimeSeries.Contact ContactSeries;
    protected CuboidMap Geometry;

    protected Vector3[] PosePrediction;
    protected Matrix4x4[] RootPrediction;
    protected Matrix4x4[] GoalPrediction;

    protected float[] Signals = new float[0];
    protected float UserControl = 0f;
    protected float NetworkControl = 0f;

    public float InteractionSmoothing = 0.9f;
    protected bool IsInteracting = false;

    //Path Planning
    public bool UsePathPlanning = true;
    public bool ShowPath = true;
    protected PathPlanningUtility PathPlanning = null;

    //GoalNet
    public bool UseGoalNet = true;
    public GoalNet goalNet;

    protected UltimateIK.Model RightFootIK, LeftFootIK;

    public bool SavePrediction = false;
    public SAMPTest TestScript;
    protected bool SaveStatus = false;
    protected Data X;
    protected StreamWriter S;

    public int InteractionFrames = 0;
    public int TotalFrames = 0;

    public Controller GetController()
    {
        return Controller;
    }

    public TimeSeries GetTimeSeries()
    {
        return TimeSeries;
    }

    public void reinit_funcs(Matrix4x4 root, Matrix4x4[] Posture, Vector3[] Velocities)
    {
        for (int i = 0; i < Actor.Bones.Length; i++)
        {
            Actor.Bones[i].Velocity = Velocities[i];
            Actor.Bones[i].Transform.position = Posture[i].GetPosition();
            Actor.Bones[i].Transform.rotation = Posture[i].GetRotation();
            Actor.Bones[i].ApplyLength();
        }

        Geometry = new CuboidMap(new Vector3Int(8, 8, 8));

        TimeSeries = new TimeSeries(6, 6, 1f, 1f, 5);
        RootSeries = new TimeSeries.Root(TimeSeries);
        StyleSeries = new TimeSeries.Style(TimeSeries, "Idle", "Walk", "Run", "Sit", "Liedown");
        GoalSeries = new TimeSeries.Goal(TimeSeries, "Idle", "Walk", "Run", "Sit", "Liedown");
        ContactSeries = new TimeSeries.Contact(TimeSeries, "right_hip", "right_wrist", "left_wrist", "right_ankle", "left_ankle");

        for (int i = 0; i < TimeSeries.Samples.Length; i++)
        {
            RootSeries.Transformations[i] = root;
            if (StyleSeries.Styles.Length > 0)
            {
                StyleSeries.Values[i][0] = 1f;
            }
            if (GoalSeries.Actions.Length > 0)
            {
                GoalSeries.Values[i][0] = 1f;
            }
            GoalSeries.Transformations[i] = root;
            Geometry.Pivot = root;
            Geometry.References[i] = root.GetPosition();
        }

        PosePrediction = new Vector3[Actor.Bones.Length];
        RootPrediction = new Matrix4x4[7];
        GoalPrediction = new Matrix4x4[7];

        //PathPlanning
        if (UsePathPlanning) { PathPlanning = new PathPlanningUtility(); }

        RightFootIK = UltimateIK.BuildModel(Actor.FindTransform("right_hip"), Actor.GetBoneTransforms(ContactSeries.Bones[3]));
        LeftFootIK = UltimateIK.BuildModel(Actor.FindTransform("left_hip"), Actor.GetBoneTransforms(ContactSeries.Bones[4]));
        transform.position = RootSeries.GetPosition(TimeSeries.Pivot);
        transform.rotation = RootSeries.GetRotation(TimeSeries.Pivot);
    }


    public void reinit_funcs()
    {
        Geometry = new CuboidMap(new Vector3Int(8, 8, 8));

        TimeSeries = new TimeSeries(6, 6, 1f, 1f, 5);
        RootSeries = new TimeSeries.Root(TimeSeries);
        StyleSeries = new TimeSeries.Style(TimeSeries, "Idle", "Walk", "Run", "Sit", "Liedown");
        GoalSeries = new TimeSeries.Goal(TimeSeries, "Idle", "Walk", "Run", "Sit", "Liedown");
        ContactSeries = new TimeSeries.Contact(TimeSeries, "right_hip", "right_wrist", "left_wrist", "right_ankle", "left_ankle");

        for (int i = 0; i < TimeSeries.Samples.Length; i++)
        {
            RootSeries.Transformations[i] = transform.GetWorldMatrix(true);
            if (StyleSeries.Styles.Length > 0)
            {
                StyleSeries.Values[i][0] = 1f;
            }
            if (GoalSeries.Actions.Length > 0)
            {
                GoalSeries.Values[i][0] = 1f;
            }
            GoalSeries.Transformations[i] = transform.GetWorldMatrix(true);
            Geometry.Pivot = transform.GetWorldMatrix(true);
            Geometry.References[i] = transform.position;
        }

        PosePrediction = new Vector3[Actor.Bones.Length];
        RootPrediction = new Matrix4x4[7];
        GoalPrediction = new Matrix4x4[7];

        //PathPlanning
        if (UsePathPlanning) { PathPlanning = new PathPlanningUtility(); }

        RightFootIK = UltimateIK.BuildModel(Actor.FindTransform("right_hip"), Actor.GetBoneTransforms(ContactSeries.Bones[3]));
        LeftFootIK = UltimateIK.BuildModel(Actor.FindTransform("left_hip"), Actor.GetBoneTransforms(ContactSeries.Bones[4]));
    }

    public bool GetIsinteracting()
    {
        return IsInteracting;
    }

    protected override void Setup()
    {

        Controller = new Controller();
        Controller.Signal idle = Controller.AddSignal("Idle");
        idle.Default = true;
        idle.Velocity = 0f;
        idle.AddKey(KeyCode.W, false);
        idle.AddKey(KeyCode.A, false);
        idle.AddKey(KeyCode.S, false);
        idle.AddKey(KeyCode.D, false);
        idle.AddKey(KeyCode.Q, false);
        idle.AddKey(KeyCode.E, false);
        idle.AddKey(KeyCode.V, true);
        idle.UserControl = 0.25f;
        idle.NetworkControl = 0.1f;

        Controller.Signal walk = Controller.AddSignal("Walk");
        walk.AddKey(KeyCode.W, true);
        walk.AddKey(KeyCode.A, true);
        walk.AddKey(KeyCode.S, true);
        walk.AddKey(KeyCode.D, true);
        walk.AddKey(KeyCode.Q, true);
        walk.AddKey(KeyCode.E, true);
        walk.AddKey(KeyCode.LeftShift, false);
        walk.AddKey(KeyCode.C, false);
        walk.Velocity = 1f;
        walk.UserControl = 0.25f;
        walk.NetworkControl = 0.25f;

        Controller.Signal run = Controller.AddSignal("Run");
        run.AddKey(KeyCode.LeftShift, true);
        run.Velocity = 3f;
        run.UserControl = 0.25f;
        run.NetworkControl = 0.25f;

        Controller.Signal sit = Controller.AddSignal("Sit");
        sit.AddKey(KeyCode.C, true);
        sit.Velocity = 0f;
        sit.UserControl = 0.25f;
        sit.NetworkControl = 0f;

        Controller.Signal liedown = Controller.AddSignal("Liedown");
        liedown.AddKey(KeyCode.F, true);
        liedown.Velocity = 0f;
        liedown.UserControl = 0.25f;
        liedown.NetworkControl = 0f;

        Geometry = new CuboidMap(new Vector3Int(8, 8, 8));

        TimeSeries = new TimeSeries(6, 6, 1f, 1f, 5);
        RootSeries = new TimeSeries.Root(TimeSeries);
        StyleSeries = new TimeSeries.Style(TimeSeries, "Idle", "Walk", "Run", "Sit", "Liedown");
        GoalSeries = new TimeSeries.Goal(TimeSeries, "Idle", "Walk", "Run", "Sit", "Liedown");
        ContactSeries = new TimeSeries.Contact(TimeSeries, "right_hip", "right_wrist", "left_wrist", "right_foot", "left_foot");

        for (int i = 0; i < TimeSeries.Samples.Length; i++)
        {
            RootSeries.Transformations[i] = transform.GetWorldMatrix(true);
            if (StyleSeries.Styles.Length > 0)
            {
                StyleSeries.Values[i][0] = 1f;
            }
            if (GoalSeries.Actions.Length > 0)
            {
                GoalSeries.Values[i][0] = 1f;
            }
            GoalSeries.Transformations[i] = transform.GetWorldMatrix(true);
            Geometry.Pivot = transform.GetWorldMatrix(true);
            Geometry.References[i] = transform.position;
        }

        PosePrediction = new Vector3[Actor.Bones.Length];
        RootPrediction = new Matrix4x4[7];
        GoalPrediction = new Matrix4x4[7];

        //PathPlanning
        if (UsePathPlanning) { PathPlanning = new PathPlanningUtility(); }
        if (SavePrediction)
        {
            S = CreateFile("Sequences");
            X = new Data(CreateFile("Pred"), CreateFile("PredLabels"));
        }
    }

    protected override void Feed()
    {
        Controller.Update();

        //Get Root
        Matrix4x4 root = RootSeries.Transformations[TimeSeries.Pivot];

        Signals = Controller.PoolSignals();
        //Control Cycle

        UserControl = Controller.PoolUserControl(Signals);
        NetworkControl = Controller.PoolNetworkControl(Signals);

        if (IsInteracting)
        {
            //Do nothing because coroutines have control.
        }
        else if (Controller.QuerySignal("Sit") || Controller.QuerySignal("Liedown"))
        {
            StartCoroutine(SitLieDown());
        }
        else
        {
            Default();
        }

        //Input Bone Positions / Velocities
        for (int i = 0; i < Actor.Bones.Length; i++)
        {
            NeuralNetwork.Feed(Actor.Bones[i].Transform.position.GetRelativePositionTo(root));
            NeuralNetwork.Feed(Actor.Bones[i].Transform.forward.GetRelativeDirectionTo(root));
            NeuralNetwork.Feed(Actor.Bones[i].Transform.up.GetRelativeDirectionTo(root));
            NeuralNetwork.Feed(Actor.Bones[i].Velocity.GetRelativeDirectionTo(root));
        }

        //Input Inverse Bone Positions
        for (int i = 0; i < Actor.Bones.Length; i++)
        {
            NeuralNetwork.Feed(Actor.Bones[i].Transform.position.GetRelativePositionTo(RootSeries.Transformations.Last()));
        }

        //Input Trajectory Positions / Directions / Velocities / Styles
        for (int i = 0; i < TimeSeries.KeyCount; i++)
        {
            TimeSeries.Sample sample = TimeSeries.GetKey(i);
            NeuralNetwork.FeedXZ(RootSeries.GetPosition(sample.Index).GetRelativePositionTo(root));
            NeuralNetwork.FeedXZ(RootSeries.GetDirection(sample.Index).GetRelativeDirectionTo(root));
            NeuralNetwork.Feed(StyleSeries.Values[sample.Index]);
        }

        //Input Contacts
        NeuralNetwork.Feed(ContactSeries.Values[TimeSeries.Pivot]);

        //Input Inverse Trajectory Positions 
        for (int i = 0; i < TimeSeries.KeyCount; i++)
        {
            TimeSeries.Sample sample = TimeSeries.GetKey(i);
            NeuralNetwork.FeedXZ(RootSeries.GetPosition(sample.Index).GetRelativePositionTo(GoalSeries.Transformations[TimeSeries.Pivot]));
            NeuralNetwork.FeedXZ(RootSeries.GetDirection(sample.Index).GetRelativeDirectionTo(GoalSeries.Transformations[TimeSeries.Pivot]));
        }

        //Input Goals
        for (int i = 0; i < TimeSeries.KeyCount; i++)
        {
            TimeSeries.Sample sample = TimeSeries.GetKey(i);
            NeuralNetwork.Feed(GoalSeries.Transformations[sample.Index].GetPosition().GetRelativePositionTo(root));
            NeuralNetwork.Feed(GoalSeries.Transformations[sample.Index].GetForward().GetRelativeDirectionTo(root));
            NeuralNetwork.Feed(GoalSeries.Values[sample.Index]);
        }

        //Input Geometry
        for (int i = 0; i < Geometry.Points.Length; i++)
        {
            NeuralNetwork.Feed(Geometry.References[i].GetRelativePositionTo(root));
            NeuralNetwork.Feed(Geometry.Occupancies[i]);
        }

    }

    protected override void Read()
    {

        //Update Past State
        for (int i = 0; i < TimeSeries.Pivot; i++)
        {
            TimeSeries.Sample sample = TimeSeries.Samples[i];
            RootSeries.SetPosition(i, RootSeries.GetPosition(i + 1));
            RootSeries.SetDirection(i, RootSeries.GetDirection(i + 1));
            for (int j = 0; j < StyleSeries.Styles.Length; j++)
            {
                StyleSeries.Values[i][j] = StyleSeries.Values[i + 1][j];
            }
            for (int j = 0; j < ContactSeries.Bones.Length; j++)
            {
                ContactSeries.Values[i][j] = ContactSeries.Values[i + 1][j];
            }
            GoalSeries.Transformations[i] = GoalSeries.Transformations[i + 1];
            for (int j = 0; j < GoalSeries.Actions.Length; j++)
            {
                GoalSeries.Values[i][j] = GoalSeries.Values[i + 1][j];
            }
        }

        //Get Root
        Matrix4x4 root = RootSeries.Transformations[TimeSeries.Pivot];

        //Read Posture
        Vector3[] positions = new Vector3[Actor.Bones.Length];
        Vector3[] forwards = new Vector3[Actor.Bones.Length];
        Vector3[] upwards = new Vector3[Actor.Bones.Length];
        Vector3[] velocities = new Vector3[Actor.Bones.Length];
        for (int i = 0; i < Actor.Bones.Length; i++)
        {
            Vector3 position = NeuralNetwork.ReadVector3().GetRelativePositionFrom(root);
            Vector3 forward = NeuralNetwork.ReadVector3().normalized.GetRelativeDirectionFrom(root);
            Vector3 upward = NeuralNetwork.ReadVector3().normalized.GetRelativeDirectionFrom(root);
            Vector3 velocity = NeuralNetwork.ReadVector3().GetRelativeDirectionFrom(root);
            positions[i] = Vector3.Lerp(Actor.Bones[i].Transform.position + velocity / GetFramerate(), position, 0.5f);
            forwards[i] = forward;
            upwards[i] = upward;
            velocities[i] = velocity;
        }

        //Read Inverse Pose
        for (int i = 0; i < Actor.Bones.Length; i++)
        {
            PosePrediction[i] = NeuralNetwork.ReadVector3().GetRelativePositionFrom(RootSeries.Transformations.Last());
            //velocities[i] = Vector3.Lerp(velocities[i], GetFramerate() * (PosePrediction[i] - Actor.Bones[i].Transform.position), 1f / GetFramerate());
        }

        //Read Future Trajectory
        for (int i = 0; i < TimeSeries.KeyCount; i++)
        {
            TimeSeries.Sample sample = TimeSeries.GetKey(i);
            Vector3 pos = NeuralNetwork.ReadXZ().GetRelativePositionFrom(root);
            Vector3 dir = NeuralNetwork.ReadXZ().normalized.GetRelativeDirectionFrom(root);
            float[] styles = NeuralNetwork.Read(StyleSeries.Styles.Length);

            //M: edit future only
            if (i >= TimeSeries.PivotKey)
            {
                RootSeries.SetPosition(sample.Index, pos);
                RootSeries.SetDirection(sample.Index, dir);
                for (int j = 0; j < styles.Length; j++)
                {
                    styles[j] = Mathf.Clamp(styles[j], 0f, 1f);
                    Vector3 goal = GoalSeries.GetPosition(TimeSeries.Pivot).GetRelativePositionTo(root);
                    if (goal.magnitude > 1)
                    {
                        //if (j == 1)
                        //{
                        //    styles[j] = 1f;
                        //}
                        //else
                        //{
                        //    styles[j] = 0f;
                        //}
                    }
                }
                StyleSeries.Values[sample.Index] = styles;
                if (i >= 6)
                { RootPrediction[i - 6] = Matrix4x4.TRS(pos, Quaternion.LookRotation(dir, Vector3.up), Vector3.one); }

            }

        }

        //Read Future Contacts
        float[] contacts = NeuralNetwork.Read(ContactSeries.Bones.Length);
        for (int i = 0; i < contacts.Length; i++)
        {
            contacts[i] = Mathf.Clamp(contacts[i], 0f, 1f);
        }
        ContactSeries.Values[TimeSeries.Pivot] = contacts;

        //Read Inverse Trajectory
        for (int i = 0; i < TimeSeries.KeyCount; i++)
        {
            TimeSeries.Sample sample = TimeSeries.GetKey(i);
            Matrix4x4 goal = GoalSeries.Transformations[TimeSeries.Pivot];
            goal[1, 3] = 0f;
            Vector3 pos = NeuralNetwork.ReadXZ().GetRelativePositionFrom(goal);
            Vector3 dir = NeuralNetwork.ReadXZ().normalized.GetRelativeDirectionFrom(goal);
            if (i > TimeSeries.PivotKey)
            {
                Matrix4x4 pivot = RootSeries.Transformations[sample.Index];
                pivot[1, 3] = 0f;
                Matrix4x4 reference = GoalSeries.Transformations[sample.Index];
                reference[1, 3] = 0f;
                float distance = Vector3.Distance(pivot.GetPosition(), reference.GetPosition());
                float weight = Mathf.Pow((float)(i - 6) / 7f, (distance * distance) * 100.0f);
                RootSeries.SetPosition(sample.Index, Vector3.Lerp(RootSeries.GetPosition(sample.Index), pos, weight));
                RootSeries.SetDirection(sample.Index, Vector3.Slerp(RootSeries.GetDirection(sample.Index), dir, weight));
            }
            if (i >= TimeSeries.PivotKey)
            { GoalPrediction[i - 6] = Matrix4x4.TRS(pos, Quaternion.LookRotation(dir, Vector3.up), Vector3.one); }

        }

        //Read and Correct Goals
        for (int i = 0; i < TimeSeries.KeyCount; i++)
        {
            float weight = TimeSeries.GetWeight1byN1(TimeSeries.GetKey(i).Index, 2f);
            TimeSeries.Sample sample = TimeSeries.GetKey(i);
            Vector3 pos = NeuralNetwork.ReadVector3().GetRelativePositionFrom(root);
            Vector3 dir = NeuralNetwork.ReadVector3().normalized.GetRelativeDirectionFrom(root);
            float[] actions = NeuralNetwork.Read(GoalSeries.Actions.Length);
            for (int j = 0; j < actions.Length; j++)
            {
                actions[j] = Mathf.Clamp(actions[j], 0f, 1f);
            }
            GoalSeries.Transformations[sample.Index] = Utility.Interpolate(GoalSeries.Transformations[sample.Index], Matrix4x4.TRS(pos, Quaternion.LookRotation(dir, Vector3.up), Vector3.one), weight * NetworkControl);
            GoalSeries.Values[sample.Index] = Utility.Interpolate(GoalSeries.Values[sample.Index], actions, weight * NetworkControl);
        }

        //Interpolate Current to Future Trajectory
        for (int i = 0; i < TimeSeries.Samples.Length; i++)
        {
            float weight = (float)(i % TimeSeries.Resolution) / TimeSeries.Resolution;
            TimeSeries.Sample sample = TimeSeries.Samples[i];
            TimeSeries.Sample prevSample = TimeSeries.GetPreviousKey(i);
            TimeSeries.Sample nextSample = TimeSeries.GetNextKey(i);
            RootSeries.SetPosition(sample.Index, Vector3.Lerp(RootSeries.GetPosition(prevSample.Index), RootSeries.GetPosition(nextSample.Index), weight));
            RootSeries.SetDirection(sample.Index, Vector3.Slerp(RootSeries.GetDirection(prevSample.Index), RootSeries.GetDirection(nextSample.Index), weight));
            GoalSeries.Transformations[sample.Index] = Utility.Interpolate(GoalSeries.Transformations[prevSample.Index], GoalSeries.Transformations[nextSample.Index], weight);
            for (int j = 0; j < StyleSeries.Styles.Length; j++)
            {
                StyleSeries.Values[i][j] = Mathf.Lerp(StyleSeries.Values[prevSample.Index][j], StyleSeries.Values[nextSample.Index][j], weight);
            }
            for (int j = 0; j < GoalSeries.Actions.Length; j++)
            {
                GoalSeries.Values[i][j] = Mathf.Lerp(GoalSeries.Values[prevSample.Index][j], GoalSeries.Values[nextSample.Index][j], weight);
            }
        }

        //Assign Posture
        transform.position = RootSeries.GetPosition(TimeSeries.Pivot);
        transform.rotation = RootSeries.GetRotation(TimeSeries.Pivot);
        for (int i = 0; i < Actor.Bones.Length; i++)
        {
            Actor.Bones[i].Velocity = velocities[i];
            Actor.Bones[i].Transform.position = positions[i];
            Actor.Bones[i].Transform.rotation = Quaternion.LookRotation(forwards[i], upwards[i]);
            Actor.Bones[i].ApplyLength();
        }
        WriteData();
    }

    public virtual void WriteData()
    {
        TotalFrames++;
        if (SavePrediction)
        {
            int n = (TestScript.Test_i + 1) * 1000 + TestScript.MotionNum - TestScript.GetSampledNum() + 1;
            if (SaveStatus)
            {
                Matrix4x4 root = RootSeries.Transformations[TimeSeries.Pivot];
                S.WriteLine(n.ToString());
                for (int i = 0; i < Actor.Bones.Length; i++)
                {
                    X.Feed(Actor.Bones[i].Transform.position.GetRelativePositionTo(root), "Bone" + (i + 1) + Actor.Bones[i].GetName() + "Position");
                    X.Feed(Actor.Bones[i].Transform.rotation.GetForward().GetRelativeDirectionTo(root), "Bone" + (i + 1) + Actor.Bones[i].GetName() + "Forward");
                    X.Feed(Actor.Bones[i].Transform.rotation.GetUp().GetRelativeDirectionTo(root), "Bone" + (i + 1) + Actor.Bones[i].GetName() + "Up");
                    X.Feed(Actor.Bones[i].Velocity.GetRelativeDirectionTo(root), "Bone" + (i + 1) + Actor.Bones[i].GetName() + "Velocity");
                }
                for (int i = 0; i < StyleSeries.Styles.Length; i++)
                {
                    X.Feed(StyleSeries.GetStyle(TimeSeries.Pivot, StyleSeries.Styles[i]), "Style-" + StyleSeries.Styles[i]);
                }
                X.Feed(transform.position, "RootPosition");
                X.Feed(transform.rotation.GetForward(), "RootForward");

                Interaction interaction;
                interaction = Controller.GetClosestInteraction(transform);
                X.Feed(interaction.GetContact("Hips").GetPosition(), "GoalPostion");
                X.Feed(interaction.GetContact("Hips").GetForward(), "GoalForward");
                X.Store();
            }
            else if (TestScript.MotionNum == TestScript.GetTotalNum())
            {
                if (!X.Finished)
                {
                    X.Finish();
                    S.Close();
                    Debug.Log("Finish Writing!");
                }
            }
        }
    }

    protected void Default()
    {
        if (Controller.ProjectionActive)
        {
            ApplyStaticGoal(Controller.Projection.point, Vector3.ProjectOnPlane(Controller.Projection.point - transform.position, Vector3.up).normalized, Signals);
        }
        else
        {
            ApplyDynamicGoal(
            RootSeries.Transformations[TimeSeries.Pivot],
            Controller.QueryMove(KeyCode.W, KeyCode.S, KeyCode.A, KeyCode.D, Signals),
            Controller.QueryTurn(KeyCode.Q, KeyCode.E, 90f),
            Signals
            );
        }
        Geometry.Setup(Geometry.Resolution);
        Geometry.Sense(RootSeries.Transformations[TimeSeries.Pivot], LayerMask.GetMask("Interaction"), Vector3.zero, InteractionSmoothing);

    }

    protected virtual IEnumerator SitLieDown()
    {
        string act = string.Empty;
        if (Controller.QuerySignal("Sit"))
        {
            act = "Sit";
        }
        else if (Controller.QuerySignal("Liedown"))
        {
            act = "Liedown";
        }
        Controller.Signal signal = Controller.GetSignal(act);
        Interaction interaction;
        interaction = Controller.ProjectionInteraction != null ? Controller.ProjectionInteraction : Controller.GetClosestInteraction(transform);
        float threshold = 0.25f;

        if (interaction != null)
        {
            Controller.ActiveInteraction = interaction;
            IsInteracting = true;
            Vector3 TargetPosition;
            Vector3 TargetDirection;

            if (UseGoalNet)
            {
                goalNet.PredictGoal(interaction, "Hips_Pred");
                TargetPosition = interaction.GetPredContact("Hips_Pred").GetPosition();
                TargetDirection = interaction.GetPredContact("Hips_Pred").GetForward();

            }
            else
            {
                TargetPosition = interaction.GetContact("Hips").GetPosition();
                TargetDirection = interaction.GetContact("Hips").GetForward();
            }



            if (UsePathPlanning)
            {
                Vector3 root = RootSeries.GetPosition(TimeSeries.Pivot);

                if (PathPlanning.Path == null || PathPlanning.Path.corners.Length == 0)
                    PathPlanning.ComputePath(root, TargetPosition);
                PathPlanning.GenerateMilestones(RootSeries.GetTransformation(TimeSeries.Pivot).GetForward(), TargetDirection);

                while (signal.Query())
                {
                    PathPlanning.DrawPath();
                    root = RootSeries.GetPosition(TimeSeries.Pivot);
                    Vector3 nextMove = Vector3.zero;
                    float nextTurn = 0f;

                    if (!PathPlanning.FinalTargetReached)
                    {
                        nextMove = PathPlanning.GetNextMove(root, RootSeries.GetDirection(TimeSeries.Pivot), TargetPosition);
                        nextTurn = PathPlanning.GetNextTurn(root, RootSeries.GetDirection(TimeSeries.Pivot));
                    }

                    if (PathPlanning.FinalTargetReached)
                    {
                        // Execute final goal, sit or liedown
                        ApplyStaticGoal(TargetPosition, TargetDirection, Signals);
                    }
                    else
                    {
                        // Walk towards the next intermediate goal
                        float[] walkSignals = { 0f, 1f, 0f, 0f, 0f };
                        ApplyDynamicGoal(RootSeries.Transformations[TimeSeries.Pivot],
                                            nextMove,
                                            nextTurn,
                                            walkSignals
                                        );
                    }


                    Geometry.Setup(Geometry.Resolution);
                    Geometry.Sense(interaction.GetCenter(), LayerMask.GetMask("Interaction"), interaction.GetExtents(), InteractionSmoothing);

                    yield return new WaitForSeconds(0f);
                }

            }
            else
            {
                while (signal.Query())
                {
                    ApplyStaticGoal(TargetPosition, TargetDirection, Signals);
                    Geometry.Setup(Geometry.Resolution);
                    Geometry.Sense(interaction.GetCenter(), LayerMask.GetMask("Interaction"), interaction.GetExtents(), InteractionSmoothing);

                    yield return new WaitForSeconds(0f);
                }
            }

            if (UsePathPlanning)
            {
                // Resent path after execution has been completed
                PathPlanning.ResetPath();
            }
            while (StyleSeries.GetStyle(TimeSeries.Pivot, act) > threshold)
            {
                ApplyDynamicGoal(
                    RootSeries.Transformations[TimeSeries.Pivot],
                    Controller.QueryMove(KeyCode.W, KeyCode.S, KeyCode.A, KeyCode.D, Signals),
                    Controller.QueryTurn(KeyCode.Q, KeyCode.E, 90f),
                    Signals
                );
                Geometry.Setup(Geometry.Resolution);
                Geometry.Sense(interaction.GetCenter(), LayerMask.GetMask("Interaction"), interaction.GetExtents(), InteractionSmoothing);
                yield return new WaitForSeconds(0f);
            }
            IsInteracting = false;
            Controller.ActiveInteraction = null;
        }
    }

    public virtual IEnumerator InteractWithObject(string act, Matrix4x4 End, Matrix4x4 Start, Matrix4x4[] Poseture, Vector3[] Velocities)
    {

        // act = "Sit" or "Liedown"
        reinit_funcs(Start, Poseture, Velocities);
        IsInteracting = true;
        yield return new WaitForSeconds(2f);
        Vector3 EndPosition = End.GetPosition();
        Vector3 EndDirection = End.GetForward();
        reinit_funcs(Start, Poseture, Velocities);
        Debug.Log("Start moving!");
        SaveStatus = true;
        Controller.Signal signal = Controller.GetSignal(act);
        signal.AutoRun = true;
        TotalFrames = 0;
        Interaction interaction;
        interaction = Controller.ProjectionInteraction != null ? Controller.ProjectionInteraction : Controller.GetClosestInteraction(transform);

        if (interaction != null)
        {
            Controller.ActiveInteraction = interaction;
            Vector3 TargetPosition;
            Vector3 TargetDirection;

            if (UseGoalNet)
            {
                goalNet.PredictGoal(interaction, "Hips_Pred");
                TargetPosition = interaction.GetPredContact("Hips_Pred").GetPosition();
                TargetDirection = interaction.GetPredContact("Hips_Pred").GetForward();
            }
            else
            {
                TargetPosition = interaction.GetContact("Hips").GetPosition();
                TargetDirection = interaction.GetContact("Hips").GetForward();
            }

            Vector3 root = RootSeries.GetPosition(TimeSeries.Pivot);

            PathPlanning.ComputePath(root, TargetPosition);

            while (signal.Query())
            {
                PathPlanning.DrawPath();
                root = RootSeries.GetPosition(TimeSeries.Pivot);
                Vector3 nextMove = Vector3.zero;
                float nextTurn = 0f;

                if (!PathPlanning.FinalTargetReached)
                {
                    nextMove = PathPlanning.GetNextMove(root, RootSeries.GetDirection(TimeSeries.Pivot), TargetPosition);
                    nextTurn = PathPlanning.GetNextTurn(root, RootSeries.GetDirection(TimeSeries.Pivot));
                }

                if (TotalFrames > TestScript.MaxFrames) { break; }

                if (PathPlanning.FinalTargetReached)
                {
                    // Execute final goal, sit or liedown
                    ApplyStaticGoal(TargetPosition, TargetDirection, Signals);
                    if (PathPlanning.Distance2Goal(RootSeries.GetPosition(TimeSeries.Pivot)) < 0.3)
                    {
                        InteractionFrames += 1;
                        yield return new WaitForFixedUpdate();
                        if (InteractionFrames > TestScript.StaticFrames)
                        {
                            InteractionFrames = 0;
                            break;
                        }
                    }
                }
                else
                {
                    // Walk towards the next intermediate goal
                    float[] walkSignals = { 0f, 1f, 0f, 0f, 0f };
                    ApplyDynamicGoal(RootSeries.Transformations[TimeSeries.Pivot],
                                        nextMove,
                                        nextTurn,
                                        walkSignals
                                    );
                }

                Geometry.Setup(Geometry.Resolution);
                Geometry.Sense(interaction.GetCenter(), LayerMask.GetMask("Interaction"), interaction.GetExtents(), InteractionSmoothing);

                if (TotalFrames > TestScript.MaxFrames) { break; }

                yield return new WaitForSeconds(0f);
            }

            root = RootSeries.GetPosition(TimeSeries.Pivot);

            TargetPosition = EndPosition;
            TargetDirection = EndDirection;

            PathPlanning.ComputePath(root, TargetPosition);

            while (signal.Query())
            {
                PathPlanning.DrawPath();
                root = RootSeries.GetPosition(TimeSeries.Pivot);
                Vector3 nextMove;
                float nextTurn;

                if (!PathPlanning.FinalTargetReached)
                {
                    nextMove = PathPlanning.GetNextMove(root, RootSeries.GetDirection(TimeSeries.Pivot), TargetPosition);
                    nextTurn = PathPlanning.GetNextTurn(root, RootSeries.GetDirection(TimeSeries.Pivot));
                }
                else
                {
                    nextMove = PathPlanning.GetFinalMove(root, RootSeries.GetDirection(TimeSeries.Pivot));
                    nextTurn = PathPlanning.GetFinalTurn(root, RootSeries.GetDirection(TimeSeries.Pivot));
                }

                // Walk towards the next intermediate goal
                float[] MovingSignals = { 0f, 0f, 0f, 0f, 0f };
                if (PathPlanning.Distance2Goal(RootSeries.GetPosition(TimeSeries.Pivot)) < 0.2)
                {
                    MovingSignals[0] = 1f;
                    InteractionFrames++;
                    yield return new WaitForFixedUpdate();
                    nextTurn = 0f;
                    if (InteractionFrames > 10)
                    {
                        InteractionFrames = 0;
                        break;
                    }
                }
                else
                {
                    MovingSignals[1] = 1f;
                }
                ApplyDynamicGoal(RootSeries.Transformations[TimeSeries.Pivot],
                                        nextMove,
                                        nextTurn,
                                        MovingSignals
                                    );

                Geometry.Setup(Geometry.Resolution);
                Geometry.Sense(interaction.GetCenter(), LayerMask.GetMask("Interaction"), interaction.GetExtents(), InteractionSmoothing);
                if (TotalFrames > TestScript.MaxFrames) { break; }

                yield return new WaitForSeconds(0f);
            }
        }
        SaveStatus = false;
        TestScript.MotionNum++;
        TestScript.SetTesti();
        signal.AutoRun = false;
        IsInteracting = false;
    }


    protected override void OnGUIDerived()
    {
        if (!ShowGUI)
        {
            return;
        }
        if (ShowGoal)
        {
            GoalSeries.GUI(OnlyPoint);
        }
        if (ShowCurrent)
        {
            StyleSeries.GUI();
        }
        if (ShowContacts)
        {
            ContactSeries.GUI();
        }
    }

    protected override void OnRenderObjectDerived()
    {
        Controller.Draw();

        if (ShowRoot)
        {
            RootSeries.Draw();
        }
        if (ShowGoal)
        {
            GoalSeries.Draw(OnlyPoint);
        }
        if (ShowCurrent)
        {
            StyleSeries.Draw();
        }
        if (ShowContacts)
        {
            ContactSeries.Draw();
        }
        if (ShowInteraction)
        {
            Geometry.Draw(UltiDraw.Cyan.Transparent(0.25f));
        }
        if (ShowPath && PathPlanning != null)
        {
            PathPlanning.DrawPath();
            PathPlanning.DrawMilestones();
        }

        if (ShowBiDirectional)
        {
            UltiDraw.Begin();
            for (int i = 0; i < PosePrediction.Length; i++)
            {
                UltiDraw.DrawSphere(PosePrediction[i], Quaternion.identity, 0.05f, UltiDraw.Magenta);
            }
            for (int i = 0; i < RootPrediction.Length; i++)
            {
                UltiDraw.DrawCircle(RootPrediction[i].GetPosition(), 0.05f, UltiDraw.DarkRed.Darken(0.5f));
                UltiDraw.DrawArrow(RootPrediction[i].GetPosition(), RootPrediction[i].GetPosition() + 0.1f * RootPrediction[i].GetForward(), 0f, 0f, 0.025f, UltiDraw.DarkRed);
                if (i < RootPrediction.Length - 1)
                {
                    UltiDraw.DrawLine(RootPrediction[i].GetPosition(), RootPrediction[i + 1].GetPosition(), UltiDraw.Black);
                }
            }
            for (int i = 0; i < GoalPrediction.Length; i++)
            {
                UltiDraw.DrawCircle(GoalPrediction[i].GetPosition(), 0.05f, UltiDraw.DarkGreen.Darken(0.5f));
                UltiDraw.DrawArrow(GoalPrediction[i].GetPosition(), GoalPrediction[i].GetPosition() + 0.1f * GoalPrediction[i].GetForward(), 0f, 0f, 0.025f, UltiDraw.DarkGreen);
                if (i < GoalPrediction.Length - 1)
                {
                    UltiDraw.DrawLine(GoalPrediction[i].GetPosition(), GoalPrediction[i + 1].GetPosition(), UltiDraw.Black);
                }
            }
            UltiDraw.End();
        }
    }


    protected virtual void ApplyStaticGoal(Vector3 position, Vector3 direction, float[] actions)
    {
        //Transformations
        for (int i = 0; i < TimeSeries.Samples.Length; i++)
        {
            float weight = TimeSeries.GetWeight1byN1(i, 2f);
            float positionBlending = weight * UserControl;
            float directionBlending = weight * UserControl;
            Matrix4x4Extensions.SetPosition(ref GoalSeries.Transformations[i], Vector3.Lerp(GoalSeries.Transformations[i].GetPosition(), position, positionBlending));
            Matrix4x4Extensions.SetRotation(ref GoalSeries.Transformations[i], Quaternion.LookRotation(Vector3.Slerp(GoalSeries.Transformations[i].GetForward(), direction, directionBlending), Vector3.up));
        }

        //Actions
        for (int i = TimeSeries.Pivot; i < TimeSeries.Samples.Length; i++)
        {
            float w = (float)(i - TimeSeries.Pivot) / (float)(TimeSeries.FutureSampleCount);
            w = Utility.Normalise(w, 0f, 1f, 1f / TimeSeries.FutureKeyCount, 1f);
            for (int j = 0; j < GoalSeries.Actions.Length; j++)
            {
                float weight = GoalSeries.Values[i][j];
                weight = 2f * (0.5f - Mathf.Abs(weight - 0.5f));
                weight = Utility.Normalise(weight, 0f, 1f, UserControl, 1f - UserControl);
                if (actions[j] != GoalSeries.Values[i][j])
                {
                    GoalSeries.Values[i][j] = Mathf.Lerp(
                        GoalSeries.Values[i][j],
                        Mathf.Clamp(GoalSeries.Values[i][j] + weight * UserControl * Mathf.Sign(actions[j] - GoalSeries.Values[i][j]), 0f, 1f),
                        w);
                }
            }
        }
    }

    protected virtual void ApplyDynamicGoal(Matrix4x4 root, Vector3 move, float turn, float[] actions)
    {
        //Transformations
        Vector3[] positions_blend = new Vector3[TimeSeries.Samples.Length];
        Vector3[] directions_blend = new Vector3[TimeSeries.Samples.Length];
        float time = 2f;
        for (int i = 0; i < TimeSeries.Samples.Length; i++)
        {
            float weight = TimeSeries.GetWeight1byN1(i, 0.5f);
            float bias_pos = 1.0f - Mathf.Pow(1.0f - weight, 0.75f);
            float bias_dir = 1.0f - Mathf.Pow(1.0f - weight, 0.75f);
            directions_blend[i] = Quaternion.AngleAxis(bias_dir * turn, Vector3.up) * Vector3.ProjectOnPlane(root.GetForward(), Vector3.up).normalized;
            if (i == 0)
            {
                positions_blend[i] = root.GetPosition() +
                    Vector3.Lerp(
                    GoalSeries.Transformations[i + 1].GetPosition() - GoalSeries.Transformations[i].GetPosition(),
                    time / (TimeSeries.Samples.Length - 1f) * (Quaternion.LookRotation(directions_blend[i], Vector3.up) * move),
                    bias_pos
                    );
            }
            else
            {
                positions_blend[i] = positions_blend[i - 1] +
                    Vector3.Lerp(
                    GoalSeries.Transformations[i].GetPosition() - GoalSeries.Transformations[i - 1].GetPosition(),
                    time / (TimeSeries.Samples.Length - 1f) * (Quaternion.LookRotation(directions_blend[i], Vector3.up) * move),
                    bias_pos
                    );
            }
        }
        for (int i = 0; i < TimeSeries.Samples.Length; i++)
        {
            Matrix4x4Extensions.SetPosition(ref GoalSeries.Transformations[i], Vector3.Lerp(GoalSeries.Transformations[i].GetPosition(), positions_blend[i], UserControl));
            Matrix4x4Extensions.SetRotation(ref GoalSeries.Transformations[i], Quaternion.Slerp(GoalSeries.Transformations[i].GetRotation(), Quaternion.LookRotation(directions_blend[i], Vector3.up), UserControl));
        }

        //Actions
        for (int i = TimeSeries.Pivot; i < TimeSeries.Samples.Length; i++)
        {
            float w = (float)(i - TimeSeries.Pivot) / (float)(TimeSeries.FutureSampleCount);
            w = Utility.Normalise(w, 0f, 1f, 1f / TimeSeries.FutureKeyCount, 1f);
            for (int j = 0; j < GoalSeries.Actions.Length; j++)
            {
                float weight = GoalSeries.Values[i][j];
                weight = 2f * (0.5f - Mathf.Abs(weight - 0.5f));
                weight = Utility.Normalise(weight, 0f, 1f, UserControl, 1f - UserControl);
                if (actions[j] != GoalSeries.Values[i][j])
                {
                    GoalSeries.Values[i][j] = Mathf.Lerp(
                        GoalSeries.Values[i][j],
                        Mathf.Clamp(GoalSeries.Values[i][j] + weight * UserControl * Mathf.Sign(actions[j] - GoalSeries.Values[i][j]), 0f, 1f),
                        w);
                }
            }
        }
    }


    protected override void Postprocess()
    {
        Matrix4x4 rightFoot = Actor.GetBoneTransformation(ContactSeries.Bones[3]);
        Matrix4x4 leftFoot = Actor.GetBoneTransformation(ContactSeries.Bones[4]);
        RightFootIK.Objectives[0].SetTarget(rightFoot.GetPosition(), 1f - ContactSeries.Values[TimeSeries.Pivot][3]);
        RightFootIK.Objectives[0].SetTarget(rightFoot.GetRotation());
        LeftFootIK.Objectives[0].SetTarget(leftFoot.GetPosition(), 1f - ContactSeries.Values[TimeSeries.Pivot][4]);
        LeftFootIK.Objectives[0].SetTarget(leftFoot.GetRotation());
        RightFootIK.Solve();
        LeftFootIK.Solve();

        Transform rightToe = Actor.FindBone("right_foot").Transform;
        Vector3 rightPos = rightToe.transform.position;
        rightPos.y = Mathf.Max(rightPos.y, 0.02f);
        rightToe.position = rightPos;

        Transform leftToe = Actor.FindBone("left_foot").Transform;
        Vector3 leftPos = leftToe.transform.position;
        leftPos.y = Mathf.Max(leftPos.y, 0.02f);
        leftToe.position = leftPos;
    }

    public StreamWriter CreateFile(string name)
    {
        string filename = string.Empty;
        string folder = Application.dataPath + "/../../Prediction/";
        if (!File.Exists(folder + name + ".txt"))
        {
            filename = folder + name;
        }
        else
        {
            int i = 1;
            while (File.Exists(folder + name + " (" + i + ").txt"))
            {
                i += 1;
            }
            filename = folder + name + " (" + i + ")";
        }
        return File.CreateText(filename + ".txt");
    }
    ///////

    public class Data
    {
        public StreamWriter File, Labels;

        public RunningStatistics[] Statistics = null;

        private Queue<float[]> Buffer = new Queue<float[]>();
        private Task Writer = null;

        private float[] Values = new float[0];
        private string[] Names = new string[0];
        //private float[] Weights = new float[0];
        private int Dim = 0;

        public bool Finished = false;
        private bool Setup = false;

        public Data(StreamWriter file, StreamWriter labels)
        {
            File = file;
            Labels = labels;
            Writer = Task.Factory.StartNew(() => WriteData());
        }

        public void Feed(float value, string name, float weight = 1f)
        {
            if (!Setup)
            {
                ArrayExtensions.Add(ref Values, value);
                ArrayExtensions.Add(ref Names, name);
                //ArrayExtensions.Add(ref Weights, weight);
            }
            else
            {
                Dim += 1;
                Values[Dim - 1] = value;
            }
        }

        public void Feed(float[] values, string name, float weight = 1f)
        {
            for (int i = 0; i < values.Length; i++)
            {
                Feed(values[i], name + (i + 1), weight);
            }
        }

        public void Feed(bool[] values, string name, float weight = 1f)
        {
            for (int i = 0; i < values.Length; i++)
            {
                Feed(values[i] ? 1f : 0f, name + (i + 1), weight);
            }
        }

        public void Feed(float[,] values, string name, float weight = 1f)
        {
            for (int i = 0; i < values.GetLength(0); i++)
            {
                for (int j = 0; j < values.GetLength(1); j++)
                {
                    Feed(values[i, j], name + (i * values.GetLength(1) + j + 1), weight);
                }
            }
        }

        public void Feed(bool[,] values, string name, float weight = 1f)
        {
            for (int i = 0; i < values.GetLength(0); i++)
            {
                for (int j = 0; j < values.GetLength(1); j++)
                {
                    Feed(values[i, j] ? 1f : 0f, name + (i * values.GetLength(1) + j + 1), weight);
                }
            }
        }

        public void Feed(Vector2 value, string name, float weight = 1f)
        {
            Feed(value.x, name + "X", weight);
            Feed(value.y, name + "Y", weight);
        }

        public void Feed(Vector3 value, string name, float weight = 1f)
        {
            Feed(value.x, name + "X", weight);
            Feed(value.y, name + "Y", weight);
            Feed(value.z, name + "Z", weight);
        }

        public void FeedXY(Vector3 value, string name, float weight = 1f)
        {
            Feed(value.x, name + "X", weight);
            Feed(value.y, name + "Y", weight);
        }

        public void FeedXZ(Vector3 value, string name, float weight = 1f)
        {
            Feed(value.x, name + "X", weight);
            Feed(value.z, name + "Z", weight);
        }

        public void FeedYZ(Vector3 value, string name, float weight = 1f)
        {
            Feed(value.y, name + "Y", weight);
            Feed(value.z, name + "Z", weight);
        }

        private void WriteData()
        {
            while (!Finished || Buffer.Count > 0)
            {
                if (Buffer.Count > 0)
                {
                    float[] item;
                    lock (Buffer)
                    {
                        item = Buffer.Dequeue();
                    }

                    //Write to File
                    File.WriteLine(String.Join(" ", Array.ConvertAll(item, x => x.ToString("F5"))));
                }
                else
                {
                    Thread.Sleep(1);
                }

            }


        }

        public void Store()
        {
            if (!Setup)
            {
                Setup = true;
                //Write Labels
                for (int i = 0; i < Names.Length; i++)
                {
                    Labels.WriteLine("[" + i + "]" + " " + Names[i]);
                }
                Labels.Close();
            }
            //Enqueue Sample
            float[] item = (float[])Values.Clone();
            lock (Buffer)
            {
                Buffer.Enqueue(item);
            }

            //Reset Running Index
            Dim = 0;
        }

        public void Finish()
        {
            Finished = true;

            Task.WaitAll(Writer);

            File.Close();

        }
    }
}