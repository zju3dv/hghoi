﻿#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System;
using System.Threading;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using System.Threading.Tasks;

public class MotionNetExporter : EditorWindow
{

    public static EditorWindow Window;
    public static Vector2 Scroll;

    public string Directory = string.Empty;
    public float Framerate = 30;
    public int BatchSize = 10;

    public List<LabelGroup> Actions = new List<LabelGroup>();
    public List<LabelGroup> Styles = new List<LabelGroup>();

    public bool ShowFiles = true;
    public List<MotionData> Files = new List<MotionData>();
    public List<bool> Export = new List<bool>();

    public MotionEditor Editor = null;

    private int Index = -1;
    private float Progress = 0f;
    private float Performance = 0f;

    public bool WriteMirror = false;
    public bool LoadActiveOnly = true;

    private int Start = 0;
    private int End = 0;

    private static bool Exporting = false;
    private static string Separator = " ";
    private static string Accuracy = "F5";

    public bool Train = true;
    public string[] TestSequences = { "armchair019", "chair_mo019", "highstool019", "reebokstep019", "sofa019", "table019", "lie_down_19" };



    [MenuItem("AI4Animation/MotionNet Exporter")]
    static void Init()
    {
        Window = EditorWindow.GetWindow(typeof(MotionNetExporter));
        Scroll = Vector3.zero;
    }

    public void OnInspectorUpdate()
    {
        Repaint();
    }

    void OnGUI()
    {
        Scroll = EditorGUILayout.BeginScrollView(Scroll);

        Editor = GameObject.FindObjectOfType<MotionEditor>();

        if (Editor == null)
        {
            Utility.SetGUIColor(UltiDraw.Black);
            using (new EditorGUILayout.VerticalScope("Box"))
            {
                Utility.ResetGUIColor();
                Utility.SetGUIColor(UltiDraw.Grey);
                using (new EditorGUILayout.VerticalScope("Box"))
                {
                    Utility.ResetGUIColor();
                    Utility.SetGUIColor(UltiDraw.Orange);
                    using (new EditorGUILayout.VerticalScope("Box"))
                    {
                        Utility.ResetGUIColor();
                        EditorGUILayout.LabelField("Exporter");
                    }
                    EditorGUILayout.LabelField("No Motion Editor found in scene.");
                }
            }
        }
        else
        {
            Utility.SetGUIColor(UltiDraw.Black);
            using (new EditorGUILayout.VerticalScope("Box"))
            {
                Utility.ResetGUIColor();

                Utility.SetGUIColor(UltiDraw.Grey);
                using (new EditorGUILayout.VerticalScope("Box"))
                {
                    Utility.ResetGUIColor();

                    Utility.SetGUIColor(UltiDraw.Orange);
                    using (new EditorGUILayout.VerticalScope("Box"))
                    {
                        Utility.ResetGUIColor();
                        EditorGUILayout.LabelField("Exporter");
                    }

                    Framerate = EditorGUILayout.FloatField("Framerate", Framerate);
                    BatchSize = Mathf.Max(1, EditorGUILayout.IntField("Batch Size", BatchSize));
                    WriteMirror = EditorGUILayout.Toggle("Write Mirror", WriteMirror);
                    LoadActiveOnly = EditorGUILayout.Toggle("Load Active Only", LoadActiveOnly);

                    Utility.SetGUIColor(UltiDraw.White);
                    using (new EditorGUILayout.VerticalScope("Box"))
                    {
                        Utility.ResetGUIColor();
                        EditorGUILayout.LabelField("Export Path: " + GetExportPath());
                    }

                    Utility.SetGUIColor(UltiDraw.LightGrey);
                    using (new EditorGUILayout.VerticalScope("Box"))
                    {
                        Utility.ResetGUIColor();
                        Utility.SetGUIColor(UltiDraw.Cyan);
                        using (new EditorGUILayout.VerticalScope("Box"))
                        {
                            Utility.ResetGUIColor();
                            EditorGUILayout.LabelField("Files" + " [" + Files.Count + "]");
                        }

                        Train = EditorGUILayout.Toggle("Train", Train);
                        ShowFiles = EditorGUILayout.Toggle("Show Files", ShowFiles);
                        if (Files.Count == 0)
                        {
                            EditorGUILayout.LabelField("No files found.");
                        }
                        else
                        {
                            if (ShowFiles)
                            {
                                EditorGUILayout.BeginHorizontal();
                                if (Utility.GUIButton("Export All", UltiDraw.DarkGrey, UltiDraw.White))
                                {
                                    for (int i = 0; i < Export.Count; i++)
                                    {
                                        if (Files[i].Export)
                                        {
                                            Export[i] = Files[i].Export;
                                        }
                                    }
                                }
                                if (Utility.GUIButton("Export None", UltiDraw.DarkGrey, UltiDraw.White))
                                {
                                    for (int i = 0; i < Export.Count; i++)
                                    {
                                        Export[i] = false;
                                    }
                                }
                                EditorGUILayout.EndHorizontal();
                                EditorGUILayout.BeginHorizontal();
                                Start = EditorGUILayout.IntField("Start", Start);
                                End = EditorGUILayout.IntField("End", End);
                                if (Utility.GUIButton("Toggle", UltiDraw.DarkGrey, UltiDraw.White))
                                {
                                    for (int i = Start - 1; i <= End - 1; i++)
                                    {
                                        if (Files[i].Export)
                                        {
                                            Export[i] = !Export[i];
                                        }
                                    }
                                }
                                EditorGUILayout.EndHorizontal();
                                for (int i = 0; i < Files.Count; i++)
                                {
                                    Utility.SetGUIColor(Index == i ? UltiDraw.Cyan : Export[i] ? UltiDraw.Gold : UltiDraw.White);
                                    using (new EditorGUILayout.VerticalScope("Box"))
                                    {
                                        Utility.ResetGUIColor();
                                        EditorGUILayout.BeginHorizontal();
                                        EditorGUILayout.LabelField((i + 1) + " - " + Files[i].GetName(), GUILayout.Width(200f));
                                        if (Files[i].Export)
                                        {
                                            EditorGUILayout.BeginVertical();
                                            if (Files[i].Export)
                                            {
                                                string info = " Scene - ";
                                                if (Files[i].Symmetric)
                                                {
                                                    info += "[Default / Mirror]";
                                                }
                                                else
                                                {
                                                    info += "[Default]";
                                                }
                                                EditorGUILayout.LabelField(info, GUILayout.Width(200f));
                                            }
                                            EditorGUILayout.EndVertical();
                                            GUILayout.FlexibleSpace();
                                            if (Utility.GUIButton("O", Export[i] ? UltiDraw.DarkGreen : UltiDraw.DarkRed, UltiDraw.White, 50f))
                                            {
                                                Export[i] = !Export[i];
                                            }
                                        }
                                        EditorGUILayout.EndHorizontal();
                                    }
                                }
                            }
                        }
                    }

                    Utility.SetGUIColor(UltiDraw.LightGrey);
                    using (new EditorGUILayout.VerticalScope("Box"))
                    {
                        Utility.ResetGUIColor();
                        Utility.SetGUIColor(UltiDraw.Cyan);
                        using (new EditorGUILayout.VerticalScope("Box"))
                        {
                            Utility.ResetGUIColor();
                            EditorGUILayout.LabelField("Actions" + " [" + Actions.Count + "]");
                        }
                        if (Actions.Count == 0)
                        {
                            EditorGUILayout.LabelField("No actions found.");
                        }
                        else
                        {
                            for (int i = 0; i < Actions.Count; i++)
                            {
                                Utility.SetGUIColor(UltiDraw.Grey);
                                using (new EditorGUILayout.VerticalScope("Box"))
                                {
                                    Utility.ResetGUIColor();
                                    EditorGUILayout.BeginHorizontal();
                                    EditorGUILayout.LabelField("Group " + (i + 1));
                                    if (Utility.GUIButton("X", UltiDraw.DarkRed, UltiDraw.White, 20f, 20f))
                                    {
                                        Actions.RemoveAt(i);
                                        EditorGUIUtility.ExitGUI();
                                    }
                                    EditorGUILayout.EndHorizontal();
                                    for (int j = 0; j < Actions[i].Labels.Length; j++)
                                    {
                                        Actions[i].Labels[j] = EditorGUILayout.TextField(Actions[i].Labels[j]);
                                    }
                                    EditorGUILayout.BeginHorizontal();
                                    if (Utility.GUIButton("+", UltiDraw.DarkGrey, UltiDraw.White))
                                    {
                                        ArrayExtensions.Expand(ref Actions[i].Labels);
                                    }
                                    if (Utility.GUIButton("-", UltiDraw.DarkGrey, UltiDraw.White))
                                    {
                                        ArrayExtensions.Shrink(ref Actions[i].Labels);
                                    }
                                    EditorGUILayout.EndHorizontal();
                                }
                            }
                        }
                    }

                    Utility.SetGUIColor(UltiDraw.LightGrey);
                    using (new EditorGUILayout.VerticalScope("Box"))
                    {
                        Utility.ResetGUIColor();
                        Utility.SetGUIColor(UltiDraw.Cyan);
                        using (new EditorGUILayout.VerticalScope("Box"))
                        {
                            Utility.ResetGUIColor();
                            EditorGUILayout.LabelField("Styles" + " [" + Styles.Count + "]");
                        }
                        if (Styles.Count == 0)
                        {
                            EditorGUILayout.LabelField("No styles found.");
                        }
                        else
                        {
                            for (int i = 0; i < Styles.Count; i++)
                            {
                                Utility.SetGUIColor(UltiDraw.Grey);
                                using (new EditorGUILayout.VerticalScope("Box"))
                                {
                                    Utility.ResetGUIColor();
                                    EditorGUILayout.BeginHorizontal();
                                    EditorGUILayout.LabelField("Group " + (i + 1));
                                    if (Utility.GUIButton("X", UltiDraw.DarkRed, UltiDraw.White, 20f, 20f))
                                    {
                                        Styles.RemoveAt(i);
                                        EditorGUIUtility.ExitGUI();
                                    }
                                    EditorGUILayout.EndHorizontal();
                                    for (int j = 0; j < Styles[i].Labels.Length; j++)
                                    {
                                        Styles[i].Labels[j] = EditorGUILayout.TextField(Styles[i].Labels[j]);
                                    }
                                    EditorGUILayout.BeginHorizontal();
                                    if (Utility.GUIButton("+", UltiDraw.DarkGrey, UltiDraw.White))
                                    {
                                        ArrayExtensions.Expand(ref Styles[i].Labels);
                                    }
                                    if (Utility.GUIButton("-", UltiDraw.DarkGrey, UltiDraw.White))
                                    {
                                        ArrayExtensions.Shrink(ref Styles[i].Labels);
                                    }
                                    EditorGUILayout.EndHorizontal();
                                }
                            }
                        }
                    }

                    if (!Exporting)
                    {
                        if (Utility.GUIButton("Reload", UltiDraw.DarkGrey, UltiDraw.White))
                        {
                            Load();
                        }
                        if (Utility.GUIButton("Export Data", UltiDraw.DarkGrey, UltiDraw.White))
                        {
                            this.StartCoroutine(ExportDataSIGGRAPHAsia());
                        }
                    }
                    else
                    {
                        EditorGUILayout.LabelField("File: " + Editor.GetData().GetName());

                        EditorGUI.DrawRect(new Rect(EditorGUILayout.GetControlRect().x, EditorGUILayout.GetControlRect().y, Progress * EditorGUILayout.GetControlRect().width, 25f), UltiDraw.Green.Transparent(0.75f));

                        EditorGUILayout.LabelField("Frames Per Second: " + Performance.ToString("F3"));

                        if (Utility.GUIButton("Stop", UltiDraw.DarkRed, UltiDraw.White))
                        {
                            Exporting = false;
                        }
                    }
                }
            }
        }

        EditorGUILayout.EndScrollView();
    }

    public void Load()
    {
        if (Editor != null)
        {
            Actions = new List<LabelGroup>();
            Styles = new List<LabelGroup>();

            Files = new List<MotionData>();
            Export = new List<bool>();
            for (int i = 0; i < Editor.Files.Length; i++)
            {
                if (Train)
                {
                    if (Array.Exists(TestSequences, element => element == Editor.Files[i].name))
                    {
                        continue;
                    }
                    else
                    {
                        Files.Add(Editor.Files[i]);
                    }

                }
                else
                {
                    if (Array.Exists(TestSequences, element => element == Editor.Files[i].name))
                    {
                        Files.Add(Editor.Files[i]);
                    }
                    else
                    {
                        continue;
                    }

                }
                if (Editor.Files[i].Export || !LoadActiveOnly)
                {
                    Export.Add(true);
                    if (Editor.Files[i].GetModule(Module.ID.Goal) != null)
                    {
                        GoalModule module = (GoalModule)Editor.Files[i].GetModule(Module.ID.Goal);
                        for (int j = 0; j < module.Functions.Length; j++)
                        {
                            if (Actions.Find(x => ArrayExtensions.Contains(ref x.Labels, module.Functions[j].Name)) == null)
                            {
                                Actions.Add(new LabelGroup(module.Functions[j].Name));
                            }
                        }
                    }
                    if (Editor.Files[i].GetModule(Module.ID.Style) != null)
                    {
                        StyleModule module = (StyleModule)Editor.Files[i].GetModule(Module.ID.Style);
                        for (int j = 0; j < module.Functions.Length; j++)
                        {
                            if (Styles.Find(x => ArrayExtensions.Contains(ref x.Labels, module.Functions[j].Name)) == null)
                            {
                                Styles.Add(new LabelGroup(module.Functions[j].Name));
                            }
                        }
                    }
                }
                else
                {
                    Export.Add(false);
                }
            }
        }
    }

    private StreamWriter CreateFile(string name)
    {
        string filename = string.Empty;
        string folder = Application.dataPath + "/../../Export/";
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

    public class Data
    {
        public StreamWriter File, Norm, Labels;

        public RunningStatistics[] Statistics = null;

        private Queue<float[]> Buffer = new Queue<float[]>();
        private Task Writer = null;

        private float[] Values = new float[0];
        private string[] Names = new string[0];
        private float[] Weights = new float[0];
        private int Dim = 0;

        private bool Finished = false;
        private bool Setup = false;

        public Data(StreamWriter file, StreamWriter norm, StreamWriter labels)
        {
            File = file;
            Norm = norm;
            Labels = labels;
            Writer = Task.Factory.StartNew(() => WriteData());
        }

        public void Feed(float value, string name, float weight = 1f)
        {
            if (!Setup)
            {
                ArrayExtensions.Add(ref Values, value);
                ArrayExtensions.Add(ref Names, name);
                ArrayExtensions.Add(ref Weights, weight);
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
            while (Exporting && (!Finished || Buffer.Count > 0))
            {
                if (Buffer.Count > 0)
                {
                    float[] item;
                    lock (Buffer)
                    {
                        item = Buffer.Dequeue();
                    }
                    //Update Mean and Std
                    for (int i = 0; i < item.Length; i++)
                    {
                        Statistics[i].Add(item[i]);
                    }
                    //Write to File
                    File.WriteLine(String.Join(Separator, Array.ConvertAll(item, x => x.ToString(Accuracy))));
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
                //Setup Mean and Std
                Statistics = new RunningStatistics[Values.Length];
                for (int i = 0; i < Statistics.Length; i++)
                {
                    Statistics[i] = new RunningStatistics();
                }

                //Write Labels
                for (int i = 0; i < Names.Length; i++)
                {
                    Labels.WriteLine("[" + i + "]" + " " + Names[i]);
                }
                Labels.Close();

                Setup = true;
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

            if (Setup)
            {
                //Write Mean
                float[] mean = new float[Statistics.Length];
                for (int i = 0; i < mean.Length; i++)
                {
                    mean[i] = Statistics[i].Mean();
                }
                Norm.WriteLine(String.Join(Separator, Array.ConvertAll(mean, x => x.ToString(Accuracy))));

                //Write Std
                float[] std = new float[Statistics.Length];
                for (int i = 0; i < std.Length; i++)
                {
                    std[i] = Statistics[i].Std();
                }
                Norm.WriteLine(String.Join(Separator, Array.ConvertAll(std, x => x.ToString(Accuracy))));
            }

            Norm.Close();
        }
    }

    [Serializable]
    public class LabelGroup
    {

        public string[] Labels;

        private int[] Indices;

        public LabelGroup(params string[] labels)
        {
            Labels = labels;
        }

        public string GetID()
        {
            string id = string.Empty;
            for (int i = 0; i < Labels.Length; i++)
            {
                id += Labels[i];
            }
            return id;
        }

        public void Setup(string[] references)
        {
            List<int> indices = new List<int>();
            for (int i = 0; i < references.Length; i++)
            {
                if (ArrayExtensions.Contains(ref Labels, references[i]))
                {
                    indices.Add(i);
                }
            }
            Indices = indices.ToArray();
        }

        public float Filter(float[] values)
        {
            float value = 0f;
            for (int i = 0; i < Indices.Length; i++)
            {
                value += values[Indices[i]];
            }
            if (value > 1f)
            {
                Debug.Log("Value larger than expected.");
            }
            return value;
        }

    }

    private string GetExportPath()
    {
        string path = Application.dataPath;
        path = path.Substring(0, path.LastIndexOf("/"));
        path = path.Substring(0, path.LastIndexOf("/"));
        path += "/Export";
        return path;
    }

    private void Precompute()
    {
        //Precomputations
        for (int j = 0; j < Actions.Count; j++)
        {
            Actions[j].Setup(((GoalModule)Editor.GetData().GetModule(Module.ID.Goal)).GetNames());
        }
        for (int j = 0; j < Styles.Count; j++)
        {
            Styles[j].Setup(((StyleModule)Editor.GetData().GetModule(Module.ID.Style)).GetNames());
        }

        if (
            (((GoalModule)Editor.GetData().GetModule(Module.ID.Goal)).GetGoalFunction("Sit") != null && ((ContactModule)Editor.GetData().GetModule(Module.ID.Contact)).EditMotion == false)
            ||
            (((StyleModule)Editor.GetData().GetModule(Module.ID.Style)).GetStyleFunction("Climb") != null && ((ContactModule)Editor.GetData().GetModule(Module.ID.Contact)).EditMotion == false)
            ||
            (Editor.GetData().name.Contains("Shelf") && ((ContactModule)Editor.GetData().GetModule(Module.ID.Contact)).EditMotion == false)
        )
        {
            Debug.LogError("No editing in file " + Editor.GetData().name + "!");
        }
    }

    private void WriteWorldData(ref Data World, InputSIGGRAPHAsia current)
    {

        World.Feed(current.Root.GetPosition(), "Root-Position");
        World.Feed(current.Root.GetForward(), "Root-Forward");
        World.Feed(current.Root.GetUp(), "Root-Up");
        World.Feed(current.Root.GetRight(), "Root-Right");

        World.Feed(current.Goal.GetPosition(), "Goal-Position");
        World.Feed(current.Goal.GetForward(), "Goal-Forward");
        World.Feed(current.Goal.GetUp(), "Goal-Up");
        World.Feed(current.Goal.GetRight(), "Goal-Right");

        //Interaction Geometry related to Goal
        if (current.GoalInteraction != null)
        {
            for (int k = 0; k < current.GoalInteraction.Points.Length; k++)
            {
                World.Feed(current.GoalInteraction.References[k].GetRelativePositionTo(current.Goal_Ground), "InteractionPosition" + (k + 1));
                World.Feed(current.GoalInteraction.Occupancies[k], "InteractionOccupancy" + (k + 1));
            }
        }
        else
        {
            for (int k = 0; k < current.Interaction.Points.Length; k++)
            {
                World.Feed(0f, "InteractionPosition" + (k + 1) + "X");
                World.Feed(0f, "InteractionPosition" + (k + 1) + "Y");
                World.Feed(0f, "InteractionPosition" + (k + 1) + "Z");
                World.Feed(0f, "InteractionOccupancy" + (k + 1));
            }
        }
    }

    private void WriteInputData(ref Data X, InputSIGGRAPHAsia current)
    {
        //Input
        //Auto-Regressive Posture
        for (int k = 0; k < current.Posture.Length; k++)
        {
            X.Feed(current.Posture[k].GetPosition().GetRelativePositionTo(current.Root), "Bone" + (k + 1) + Editor.GetActor().Bones[k].GetName() + "Position");
            X.Feed(current.Posture[k].GetForward().GetRelativeDirectionTo(current.Root), "Bone" + (k + 1) + Editor.GetActor().Bones[k].GetName() + "Forward");
            X.Feed(current.Posture[k].GetUp().GetRelativeDirectionTo(current.Root), "Bone" + (k + 1) + Editor.GetActor().Bones[k].GetName() + "Up");
            X.Feed(current.Velocities[k].GetRelativeDirectionTo(current.Root), "Bone" + (k + 1) + Editor.GetActor().Bones[k].GetName() + "Velocity");
        }

        //Inverse Posture
        for (int k = 0; k < current.Posture.Length; k++)
        {
            X.Feed(current.Posture[k].GetPosition().GetRelativePositionTo(current.RootSeries.Transformations.Last()), "InverseBone" + (k + 1) + Editor.GetActor().Bones[k].GetName() + "Position");
        }

        //Auto-Regressive Trajectory
        for (int k = 0; k < current.TimeSeries.Samples.Length; k++)
        {
            X.FeedXZ(current.RootSeries.GetPosition(k).GetRelativePositionTo(current.Root), "Trajectory" + (k + 1) + "Position");
            X.FeedXZ(current.RootSeries.GetDirection(k).GetRelativeDirectionTo(current.Root), "Trajectory" + (k + 1) + "Direction");
            for (int c = 0; c < Styles.Count; c++)
            {
                X.Feed(Styles[c].Filter(current.StyleSeries.Values[k]), "Trajectory" + (k + 1) + "Style" + "-" + Styles[c].GetID());
            }
        }

        X.Feed(current.ContactSeries.GetContacts(current.TimeSeries.Pivot, "pelvis", "right_wrist", "left_wrist", "right_ankle", "left_ankle"), "Contact-");

        //Inverse Trajectory
        for (int k = 0; k < current.TimeSeries.Samples.Length; k++)
        {
            X.FeedXZ(current.RootSeries.Transformations[k].GetPosition().GetRelativePositionTo(current.GoalSeries.Transformations[current.TimeSeries.Pivot]), "InverseTrajectoryPosition" + "-" + (k + 1));
            X.FeedXZ(current.RootSeries.Transformations[k].GetForward().GetRelativeDirectionTo(current.GoalSeries.Transformations[current.TimeSeries.Pivot]), "InverseTrajectoryDirection" + "-" + (k + 1));
        }

        //Goals
        for (int k = 0; k < current.TimeSeries.Samples.Length; k++)
        {
            X.Feed(current.GoalSeries.Transformations[k].GetPosition().GetRelativePositionTo(current.Root), "GoalPosition" + "-" + (k + 1));
            X.Feed(current.GoalSeries.Transformations[k].GetForward().GetRelativeDirectionTo(current.Root), "GoalDirection" + "-" + (k + 1));
            for (int c = 0; c < Actions.Count; c++)
            {
                X.Feed(Actions[c].Filter(current.GoalSeries.Values[k]), "Action" + (k + 1) + "-" + Actions[c].GetID());
            }
        }
        //Environment
        //X.Feed(current.Environment_cuboid.Occupancies, "EnvironmentCuboid-");
        X.Feed(current.Environment_small.Occupancies, "EnvironmentSmall-");
        X.Feed(current.Environment_big.Occupancies, "EnvironmentBig-");

        //Interaction Geometry
        //for (int k = 0; k < current.Interaction.Points.Length; k++)
        //{
        //    X.Feed(current.Interaction.References[k].GetRelativePositionTo(current.Root), "InteractionPosition" + (k + 1));
        //    X.Feed(current.Interaction.Occupancies[k], "InteractionOccupancy" + (k + 1));
        //}
        if (current.GoalInteraction != null)
        {
            for (int k = 0; k < current.GoalInteraction.Points.Length; k++)
            {
                X.Feed(current.GoalInteraction.References[k].GetRelativePositionTo(current.Root), "InteractionPosition" + (k + 1));
                X.Feed(current.GoalInteraction.Occupancies[k], "InteractionOccupancy" + (k + 1));
            }
        }
        else
        {
            for (int k = 0; k < current.Interaction.Points.Length; k++)
            {
                X.Feed(0f, "InteractionPosition" + (k + 1) + "X");
                X.Feed(0f, "InteractionPosition" + (k + 1) + "Y");
                X.Feed(0f, "InteractionPosition" + (k + 1) + "Z");
                X.Feed(0f, "InteractionOccupancy" + (k + 1));

            }
        }
    }

    private void WriteOutputData(ref Data Y, InputSIGGRAPHAsia current, OutputSIGGRAPHAsia next)
    {
        //Output

        //Auto-Regressive Posture
        for (int k = 0; k < next.Posture.Length; k++)
        {
            Y.Feed(next.Posture[k].GetPosition().GetRelativePositionTo(current.Root), "Bone" + (k + 1) + Editor.GetActor().Bones[k].GetName() + "Position");
            Y.Feed(next.Posture[k].GetForward().GetRelativeDirectionTo(current.Root), "Bone" + (k + 1) + Editor.GetActor().Bones[k].GetName() + "Forward");
            Y.Feed(next.Posture[k].GetUp().GetRelativeDirectionTo(current.Root), "Bone" + (k + 1) + Editor.GetActor().Bones[k].GetName() + "Up");
            Y.Feed(next.Velocities[k].GetRelativeDirectionTo(current.Root), "Bone" + (k + 1) + Editor.GetActor().Bones[k].GetName() + "Velocity");
        }

        //Inverse Posture
        for (int k = 0; k < next.Posture.Length; k++)
        {
            Y.Feed(next.Posture[k].GetPosition().GetRelativePositionTo(current.RootSeries.Transformations.Last()), "InverseBone" + (k + 1) + Editor.GetActor().Bones[k].GetName() + "Position");
        }

        //Auto-Regressive Trajectory
        for (int k = 0; k < next.TimeSeries.Samples.Length; k++)
        {
            Y.FeedXZ(next.RootSeries.GetPosition(k).GetRelativePositionTo(current.Root), "Trajectory" + (k + 1) + "Position");
            Y.FeedXZ(next.RootSeries.GetDirection(k).GetRelativeDirectionTo(current.Root), "Trajectory" + (k + 1) + "Direction");
            for (int c = 0; c < Styles.Count; c++)
            {
                Y.Feed(Styles[c].Filter(next.StyleSeries.Values[k]), "Trajectory" + (k + 1) + "Style" + "-" + Styles[c].GetID());
            }
        }


        //Key Contact
        Y.Feed(next.ContactSeries.GetContacts(next.TimeSeries.Pivot, "pelvis", "right_wrist", "left_wrist", "right_ankle", "left_ankle"), "Contact-");

        //Inverse Trajectory
        //for (int k = next.TimeSeries.Pivot; k < next.TimeSeries.Samples.Length; k++)
        for (int k = 0; k < next.TimeSeries.Samples.Length; k++)
        {
            Y.FeedXZ(next.RootSeries.Transformations[k].GetPosition().GetRelativePositionTo(current.GoalSeries.Transformations[current.TimeSeries.Pivot]), "InverseTrajectoryPosition" + "-" + (k + 1));
            Y.FeedXZ(next.RootSeries.Transformations[k].GetForward().GetRelativeDirectionTo(current.GoalSeries.Transformations[current.TimeSeries.Pivot]), "InverseTrajectoryDirection" + "-" + (k + 1));
        }

        //Goals
        for (int k = 0; k < next.TimeSeries.Samples.Length; k++)
        {
            Y.Feed(next.GoalSeries.Transformations[k].GetPosition().GetRelativePositionTo(current.Root), "GoalPosition" + "-" + (k + 1));
            Y.Feed(next.GoalSeries.Transformations[k].GetForward().GetRelativeDirectionTo(current.Root), "GoalDirection" + "-" + (k + 1));
            for (int c = 0; c < Actions.Count; c++)
            {
                Y.Feed(Actions[c].Filter(next.GoalSeries.Values[k]), "Action" + (k + 1) + "-" + Actions[c].GetID());
            }
        }
    }

    private void WriteInteractionData(ref StreamWriter objname, ref Data objfile, InputSIGGRAPHAsia current)
    {
        Matrix4x4 obj_t = Matrix4x4.identity;
        Vector3 classscale = Vector3.one;
        Vector3 scale = Vector3.one;
        String obj_name_frame = "None";
        if (current.interact_obj != null)
        {
            GameObject model;
            if (current.interact_obj.transform.Find("Model") != null)
            {
                model = current.interact_obj.transform.Find("Model").gameObject;

            }
            else
            {
                model = current.interact_obj.transform.Find("model").gameObject;
            }
            obj_t = model.transform.GetWorldMatrix();
            classscale = current.objclass.transform.localScale;
            scale = current.interact_obj.transform.localScale;
            obj_name_frame = current.interact_obj.name;
        }
        objname.WriteLine(obj_name_frame, "InteractionName");
        //obj_t = current.interact_obj.transform.GetWorldMatrix();
        objfile.Feed(obj_t.GetPosition(), "InteractionPosition");
        objfile.Feed(obj_t.GetForward(), "InteractionForward");
        objfile.Feed(obj_t.GetUp(), "InteractionUp");
        objfile.Feed(classscale, "ClassScale");
        objfile.Feed(scale, "ObjScale");
    }

    private IEnumerator ExportDataSIGGRAPHAsia()
    {
        if (Editor == null)
        {
            Debug.Log("No editor found.");
        }
        else if (!System.IO.Directory.Exists(Application.dataPath + "/../../Export"))
        {
            Debug.Log("No export folder found at " + GetExportPath() + ".");
        }
        else
        {
            Exporting = true;

            Progress = 0f;

            int total = 0;
            int items = 0;
            int sequence = 0;
            DateTime timestamp = Utility.GetTimestamp();

            Data X = new Data(CreateFile("Input"), CreateFile("InputNorm"), CreateFile("InputLabels"));
            Data Y = new Data(CreateFile("Output"), CreateFile("OutputNorm"), CreateFile("OutputLabels"));

            StreamWriter S = CreateFile("Sequences");
            StreamWriter Frame_ind = CreateFile("FramesInd");
            Data World = new Data(CreateFile("World"), CreateFile("WorldNorm"), CreateFile("WorldLabels"));
            StreamWriter ObjName = CreateFile("ObjName");
            Data ObjFile = new Data(CreateFile("ObjFile"), CreateFile("ObjFileNorm"), CreateFile("ObjFileLabels"));

            bool editorSave = Editor.Save;
            bool editorMirror = Editor.Mirror;
            float editorRate = Editor.TargetFramerate;
            int editorSeed = Editor.RandomSeed;
            Editor.Save = false;
            Editor.SetTargetFramerate(Framerate);
            for (int i = 0; i < Files.Count; i++)
            {
                if (!Exporting)
                {
                    break;
                }
                if (Export[i])
                {
                    Index = i;
                    Editor.LoadData(Files[i]);
                    while (!Editor.GetData().GetScene().isLoaded)
                    {
                        Debug.Log("Waiting for scene to be loaded.");
                        yield return new WaitForSeconds(0f);
                    }
                    for (int m = 1; m <= 2; m++)
                    {
                        if (!Exporting)
                        {
                            break;
                        }
                        if (m == 1)
                        {
                            Editor.SetMirror(false);
                        }
                        if (m == 2)
                        {
                            Editor.SetMirror(true);
                        }
                        if (!Editor.Mirror || WriteMirror && Editor.Mirror && Editor.GetData().Symmetric)
                        {
                            Debug.Log("File: " + Editor.GetData().GetName() + " Scene: " + Editor.GetData().GetName() + " " + (Editor.Mirror ? "[Mirror]" : "[Default]"));

                            Sequence seq = Editor.GetData().GetUnrolledSequence();
                            {
                                sequence += 1;
                                Precompute();
                                //Exporting
                                float start = Editor.CeilToTargetTime(Editor.GetData().GetFrame(seq.Start).Timestamp);
                                float end = Editor.FloorToTargetTime(Editor.GetData().GetFrame(seq.End).Timestamp);
                                int sample = 0;
                                while (start + (sample + 1) / Framerate <= end)
                                {
                                    if (!Exporting)
                                    {
                                        break;
                                    }
                                    Editor.SetRandomSeed(sample + 1);
                                    InputSIGGRAPHAsia current = new InputSIGGRAPHAsia(Editor, start + sample / Framerate);
                                    sample += 1;
                                    OutputSIGGRAPHAsia next = new OutputSIGGRAPHAsia(Editor, start + sample / Framerate);

                                    //Write Sequence
                                    S.WriteLine(sequence.ToString());

                                    //Write FrameInd
                                    Frame_ind.WriteLine(sample.ToString());

                                    if (current.Frame.Index + 1 != next.Frame.Index)
                                    {
                                        Debug.Log("Oups! Something went wrong with frame sampling from " + current.Frame.Index + " to " + next.Frame.Index + " at target framerate " + Framerate + ". This should not have happened!");
                                    }

                                    WriteWorldData(ref World, current);
                                    WriteInputData(ref X, current);
                                    WriteOutputData(ref Y, current, next);
                                    WriteInteractionData(ref ObjName, ref ObjFile, current);

                                    //Write Line
                                    X.Store();
                                    Y.Store();
                                    World.Store();
                                    ObjFile.Store();

                                    Progress = (sample / Framerate) / (end - start);
                                    total += 1;
                                    items += 1;
                                    if (items >= BatchSize)
                                    {
                                        Performance = items / (float)Utility.GetElapsedTime(timestamp);
                                        timestamp = Utility.GetTimestamp();
                                        items = 0;
                                        yield return new WaitForSeconds(0f);
                                    }
                                }

                                //Reset Progress
                                Progress = 0f;

                                //Collect Garbage
                                EditorUtility.UnloadUnusedAssetsImmediate();
                                Resources.UnloadUnusedAssets();
                                GC.Collect();
                            }
                        }
                    }
                }
            }
            Editor.Save = editorSave;
            Editor.SetMirror(editorMirror);
            Editor.SetTargetFramerate(editorRate);
            Editor.SetRandomSeed(editorSeed);

            S.Close();
            Frame_ind.Close();
            ObjName.Close();

            X.Finish();
            Y.Finish();
            World.Finish();
            ObjFile.Finish();

            Index = -1;
            Exporting = false;
            yield return new WaitForSeconds(0f);

            Debug.Log("Exported " + total + " samples.");
        }
    }



    public class InputSIGGRAPHAsia
    {
        public Frame Frame;
        public Matrix4x4 Root;
        public Matrix4x4[] Posture;
        public Vector3[] Velocities;
        public TimeSeries TimeSeries;
        public TimeSeries.Root RootSeries;
        public TimeSeries.Style StyleSeries;
        public TimeSeries.Goal GoalSeries;
        public TimeSeries.Contact ContactSeries;
        public TimeSeries.Phase PhaseSeries;
        public CylinderMap Environment_small;
        public CylinderMap Environment_big;
        public CuboidMap Environment_cuboid;
        public CuboidMap Interaction;
        public CuboidMap GoalInteraction;
        public Matrix4x4 Goal;
        public Matrix4x4 Goal_Ground;
        public GameObject interact_obj;
        public GameObject objclass;

        public InputSIGGRAPHAsia(MotionEditor editor, float timestamp)
        {
            editor.LoadFrame(timestamp);
            Frame = editor.GetCurrentFrame();

            Root = editor.GetActor().GetRoot().GetWorldMatrix(true);
            Posture = editor.GetActor().GetBoneTransformations();
            Velocities = editor.GetActor().GetBoneVelocities();
            TimeSeries = ((TimeSeriesModule)editor.GetData().GetModule(Module.ID.TimeSeries)).GetTimeSeries(Frame, editor.Mirror, 6, 6, 1f, 1f, 1, 1f / editor.TargetFramerate);
            RootSeries = (TimeSeries.Root)TimeSeries.GetSeries("Root");
            StyleSeries = (TimeSeries.Style)TimeSeries.GetSeries("Style");
            GoalSeries = (TimeSeries.Goal)TimeSeries.GetSeries("Goal");
            ContactSeries = (TimeSeries.Contact)TimeSeries.GetSeries("Contact");
            Interaction = ((GoalModule)editor.GetData().GetModule(Module.ID.Goal)).Target.GetInteractionGeometry(Frame, editor.Mirror, 1f / editor.TargetFramerate);
            CylinderMapModule module = (CylinderMapModule)editor.GetData().GetModule(Module.ID.CylinderMap);
            Environment_small = module.GetCylinderMap(Frame, editor.Mirror);
            Environment_big = module.GetCylinderMap(Frame, editor.Mirror, 4f, 5, 5);
            CuboidMapModule module_cuboid = (CuboidMapModule)editor.GetData().GetModule(Module.ID.CuboidMap);
            Environment_cuboid = module_cuboid.GetCuboidMap(Frame, editor.Mirror);
            GameObject instance = editor.GetData().GetScene().GetRootGameObjects()[0];
            objclass = instance;
            GoalInteraction = null;
            interact_obj = null;
            if (instance.transform.childCount > 0)
            {
                for (int i = 0; i < instance.transform.GetChild(0).childCount; i++)
                {
                    GameObject obj = instance.transform.GetChild(0).GetChild(i).gameObject;
                    if (obj.activeInHierarchy)
                    {
                        Interaction interaction = obj.GetComponent<Interaction>();
                        GoalModule goalmodule = (GoalModule)editor.GetData().GetModule(Module.ID.Goal);

                        Frame goalframe;
                        if (timestamp > goalmodule.GoalFrame.Timestamp + 1)
                        {
                            goalframe = Frame.GetLastFrame();
                        }
                        else
                        {
                            goalframe = goalmodule.GoalFrame;
                        }
                        Vector3 goal_pos = goalframe.GetBoneTransformation(0, editor.Mirror).GetPosition();
                        RootModule rootmodule = (RootModule)editor.GetData().GetModule(Module.ID.Root);
                        Quaternion goal_rot = rootmodule.GetRootTransformation(goalframe, editor.Mirror).GetRotation();
                        Goal = Matrix4x4.TRS(goal_pos, goal_rot, Vector3.one);
                        Vector3 goal_ground_pos = rootmodule.GetRootPosition(goalframe, editor.Mirror);
                        Goal_Ground = Matrix4x4.TRS(goal_ground_pos, goal_rot, Vector3.one);

                        CuboidMap sensor = new CuboidMap(new Vector3Int(8, 8, 8));
                        sensor.Sense(interaction.GetCenter(), LayerMask.GetMask("Interaction"), interaction.GetExtents());
                        Transformation transformation = interaction.GetComponent<Transformation>();
                        if (transformation != null)
                        {
                            sensor.Retransform(interaction.GetCenter(transformation.GetTransformation(Frame, editor.Mirror)));
                        }
                        GoalInteraction = sensor;
                        interact_obj = obj;
                        break;
                    }
                }
            }

        }
    }

    public class OutputSIGGRAPHAsia
    {
        public Frame Frame;
        public Matrix4x4 Root;
        public Matrix4x4[] Posture;
        public Vector3[] Velocities;
        public TimeSeries TimeSeries;
        public TimeSeries.Root RootSeries;
        public TimeSeries.Style StyleSeries;
        public TimeSeries.Goal GoalSeries;
        public TimeSeries.Contact ContactSeries;
        public TimeSeries.Phase PhaseSeries;

        public OutputSIGGRAPHAsia(MotionEditor editor, float timestamp)
        {
            editor.LoadFrame(timestamp);
            Frame = editor.GetCurrentFrame();

            Root = editor.GetActor().GetRoot().GetWorldMatrix(true);
            Posture = editor.GetActor().GetBoneTransformations();
            Velocities = editor.GetActor().GetBoneVelocities();
            TimeSeries = ((TimeSeriesModule)editor.GetData().GetModule(Module.ID.TimeSeries)).GetTimeSeries(Frame, editor.Mirror, 6, 6, 1f, 1f, 1, 1f / editor.TargetFramerate);
            RootSeries = (TimeSeries.Root)TimeSeries.GetSeries("Root");
            StyleSeries = (TimeSeries.Style)TimeSeries.GetSeries("Style");
            GoalSeries = (TimeSeries.Goal)TimeSeries.GetSeries("Goal");
            ContactSeries = (TimeSeries.Contact)TimeSeries.GetSeries("Contact");
        }
    }

}
#endif