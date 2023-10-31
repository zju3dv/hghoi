#if UNITY_EDITOR
using System.IO;
using System.Threading;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using UnityEngine.SceneManagement;
using UnityEditor.SceneManagement;
using UnityEditorInternal;

[ExecuteInEditMode]
public class ActorEditor : MotionEditor
{

    public string PredictionFolder = "Assets/Demo/Actor_prediction_01";
    public Actor PredCharacter = null;
    public Actor GTDecodeCharacter = null;

    private float[][] PredMotion = null;
    private float[][] GTDecodeMotion = null;
    private string MotionName = string.Empty;

    public string[] TestSequences = { "armchair019", "chair_mo019", "highstool019", "reebokstep019", "sofa019", "table019", "lie_down_19" };

    public void SetPredCharacter(Actor character)
    {
        PredCharacter = character;
    }

    public void SetGTDecodeCharacter(Actor character)
    {
        GTDecodeCharacter = character;
    }

    public void LoadPrediction(string filename, int n)
    {
        string pred_path = PredictionFolder + "/" + filename + "/pred_" + n + ".txt";
        string gt_path = PredictionFolder + "/" + filename + "/gt_" + n + ".txt";
        if (!File.Exists(pred_path))
        {
            Debug.Log(pred_path + " does not exist!");
            PredMotion = null;
            GTDecodeMotion = null;
            return;
        }
        string[] pred = File.ReadAllLines(pred_path);
        string[] gt = File.ReadAllLines(gt_path);
        PredMotion = new float[pred.Length][];
        GTDecodeMotion = new float[pred.Length][];
        for (int i = 0; i < pred.Length; i++)
        {
            string[] pred_per = pred[i].Split(' ');
            string[] gt_per = gt[i].Split(' ');
            if (pred_per.Length != 335)
            {
                Debug.Log("The prediction length is not 335!");
            }
            PredMotion[i] = new float[335];
            GTDecodeMotion[i] = new float[335];
            for (int j = 0; j < 335; j++)
            {
                PredMotion[i][j] = float.Parse(pred_per[j]);
                GTDecodeMotion[i][j] = float.Parse(gt_per[j]);
            }
        }
        Debug.Log("Load prediction data from " + pred_path + "!");
        MotionName = Data.GetName();
    }

    public void LoadFramePrediction()
    {
        Frame frame = GetCurrentFrame();
        int index = frame.Index;
        Sequence seq = GetData().GetUnrolledSequence();
        index = Mathf.Min(index, seq.End - 2);
        index = Mathf.Max(index, seq.Start);
        index = index - seq.Start;
        if (PredMotion != null)
        {
            SetFramePrediction(ref PredCharacter, PredMotion[index]);
            SetFramePrediction(ref GTDecodeCharacter, GTDecodeMotion[index]);
        }
    }

    public void SetFramePrediction(ref Actor character, float[] motion)
    {
        Actor actor = GetActor();
        character.transform.position = actor.transform.position;
        character.transform.rotation = actor.transform.rotation;
        Matrix4x4 root = actor.transform.GetWorldMatrix();
        int C = 12;
        for (int i = 0; i < actor.Bones.Length; i++)
        {
            character.Bones[i].Velocity = actor.Bones[i].Velocity;
            character.Bones[i].Transform.position = actor.Bones[i].Transform.position;
            character.Bones[i].Transform.rotation = actor.Bones[i].Transform.rotation;
            character.Bones[i].ComputeLength();

            Vector3 position = new Vector3(motion[i * C], motion[i * C + 1], motion[i * C + 2]);
            Vector3 forward = new Vector3(motion[i * C + 3], motion[i * C + 4], motion[i * C + 5]);
            forward = forward.normalized.GetRelativeDirectionFrom(root);
            Vector3 upward = new Vector3(motion[i * C + 6], motion[i * C + 7], motion[i * C + 8]);
            upward = upward.normalized.GetRelativeDirectionFrom(root);
            Vector3 velocity = new Vector3(motion[i * C + 9], motion[i * C + 10], motion[i * C + 11]);
            velocity = velocity.GetRelativeDirectionFrom(root);
            character.Bones[i].Velocity = velocity;
            character.Bones[i].Transform.position = position.GetRelativePositionFrom(root);
            character.Bones[i].Transform.rotation = Quaternion.LookRotation(forward, upward);
            //character.Bones[i].ApplyLength();
        }
    }

    [CustomEditor(typeof(ActorEditor))]
    public class ActorEditor_Editor : Editor
    {

        public ActorEditor Target;

        private float RepaintRate = 10f;
        private System.DateTime Timestamp;

        public string[] Names = new string[0];
        public string[] EnumNames = new string[0];
        public string NameFilter = "";
        public string[] TestNames = new string[0];
        public string[] TestEnumNames = new string[0];
        public string[] TrainNames = new string[0];
        public string[] TrainEnumNames = new string[0];

        void Awake()
        {
            Target = (ActorEditor)target;
            Target.Refresh();
            ComputeNames();
            Timestamp = Utility.GetTimestamp();
            EditorApplication.update += EditorUpdate;
        }

        void OnDestroy()
        {
            EditorApplication.update -= EditorUpdate;
            if (Target.Data != null)
            {
                Target.Data.Save();
            }
        }

        public void EditorUpdate()
        {
            if (Utility.GetElapsedTime(Timestamp) >= 1f / RepaintRate)
            {
                Repaint();
                Timestamp = Utility.GetTimestamp();
            }
        }

        public override void OnInspectorGUI()
        {
            Undo.RecordObject(Target, Target.name);
            Inspector();
            if (GUI.changed)
            {
                EditorUtility.SetDirty(Target);
            }
        }

        public void ComputeNames()
        {
            List<string> names = new List<string>();
            List<string> enumNames = new List<string>();
            List<string> testnames = new List<string>();
            List<string> testenumNames = new List<string>();
            List<string> trainnames = new List<string>();
            List<string> trainenumNames = new List<string>();
            for (int i = 0; i < Target.Files.Length; i++)
            {
                if (Target.Files[i].GetName().ToLowerInvariant().Contains(NameFilter.ToLowerInvariant()))
                {
                    names.Add(Target.Files[i].GetName());
                    enumNames.Add("[" + (i + 1) + "]" + " " + Target.Files[i].GetName());
                    if (((IList)Target.TestSequences).Contains(Target.Files[i].GetName()))
                    {
                        testnames.Add(Target.Files[i].GetName());
                        testenumNames.Add("[" + testnames.Count + "]" + " " + Target.Files[i].GetName());
                    }
                    else
                    {
                        trainnames.Add(Target.Files[i].GetName());
                        trainenumNames.Add("[" + trainnames.Count + "]" + " " + Target.Files[i].GetName());
                    }
                }
            }
            Names = names.ToArray();
            EnumNames = enumNames.ToArray();
            TestNames = testnames.ToArray();
            TestEnumNames = testenumNames.ToArray();
            TrainNames = trainnames.ToArray();
            TrainEnumNames = trainenumNames.ToArray();
        }

        public void SetNameFilter(string filter)
        {
            if (NameFilter != filter)
            {
                NameFilter = filter;
                ComputeNames();
            }
        }

        public void Import()
        {
            Target.Import();
            ComputeNames();
        }

        public void LoadPrediction()
        {
            if (Target.PredMotion != null && Target.MotionName == Target.Data.GetName())
            {
                return;
            }
            if (System.Array.FindIndex(TestNames, x => x == Target.Data.GetName()) > -1)
            {
                Target.LoadPrediction("test_decode_motion", System.Array.FindIndex(TestNames, x => x == Target.Data.GetName()) + 1);
            }
            else
            {
                Target.LoadPrediction("train_decode_motion", System.Array.FindIndex(TrainNames, x => x == Target.Data.GetName()));
            }
        }

        public void LoadFramePrediction()
        {
            Target.LoadFramePrediction();
        }

        public void Inspector()
        {
            Target.Refresh();

            Utility.SetGUIColor(UltiDraw.DarkGrey);
            using (new EditorGUILayout.VerticalScope("Box"))
            {
                Utility.ResetGUIColor();

                Utility.SetGUIColor(UltiDraw.LightGrey);
                using (new EditorGUILayout.VerticalScope("Box"))
                {
                    Utility.ResetGUIColor();

                    EditorGUILayout.BeginHorizontal();
                    Target.Folder = EditorGUILayout.TextField("Folder", "Assets/" + Target.Folder.Substring(Mathf.Min(7, Target.Folder.Length)));
                    if (Utility.GUIButton("Import", UltiDraw.DarkGrey, UltiDraw.White))
                    {
                        Import();
                    }
                    EditorGUILayout.EndHorizontal();

                    EditorGUILayout.BeginHorizontal();
                    Target.PredictionFolder = EditorGUILayout.TextField("Prediction Folder", Target.PredictionFolder);
                    //Target.PredictionFolder = EditorGUILayout.TextField("Prediction Folder", "Assets/Demo/Actor_prediction_01");
                    EditorGUILayout.EndHorizontal();


                    Target.SetCharacter((Actor)EditorGUILayout.ObjectField("Character", Target.Character, typeof(Actor), true));

                    Target.SetPredCharacter((Actor)EditorGUILayout.ObjectField("Pred Character", Target.PredCharacter, typeof(Actor), true));
                    Target.SetGTDecodeCharacter((Actor)EditorGUILayout.ObjectField("GT Decode Character", Target.GTDecodeCharacter, typeof(Actor), true));

                    SetNameFilter(EditorGUILayout.TextField("Name Filter", NameFilter));

                    if (Target.Data != null)
                    {
                        Frame frame = Target.GetCurrentFrame();

                        Utility.SetGUIColor(UltiDraw.Grey);
                        using (new EditorGUILayout.VerticalScope("Box"))
                        {
                            Utility.ResetGUIColor();


                            EditorGUILayout.BeginHorizontal();
                            GUILayout.FlexibleSpace();
                            int selectIndex = EditorGUILayout.Popup(System.Array.FindIndex(Names, x => x == Target.Data.GetName()), EnumNames);
                            if (selectIndex != -1)
                            {
                                Target.LoadData(Names[selectIndex]);
                                LoadPrediction();
                            }
                            if (Utility.GUIButton("<", UltiDraw.DarkGrey, UltiDraw.White))
                            {
                                Target.LoadPreviousData();
                                NameFilter = string.Empty;
                                LoadPrediction();
                            }
                            if (Utility.GUIButton(">", UltiDraw.DarkGrey, UltiDraw.White))
                            {
                                Target.LoadNextData();
                                LoadPrediction();
                            }
                            int sliderIndex = EditorGUILayout.IntSlider(System.Array.FindIndex(Target.Files, x => x == Target.Data) + 1, 1, Target.Files.Length);
                            if (Event.current.type == EventType.Used)
                            {
                                Target.LoadData(Target.Files[sliderIndex - 1]);
                                LoadPrediction();
                            }
                            EditorGUILayout.LabelField("/ " + Target.Files.Length, GUILayout.Width(60f));
                            if (Utility.GUIButton("Save", Target.Save ? UltiDraw.Magenta : UltiDraw.LightGrey, UltiDraw.Black))
                            {
                                Target.Save = !Target.Save;
                            }

                            GUILayout.FlexibleSpace();
                            EditorGUILayout.EndHorizontal();

                            if (System.Array.FindIndex(TestNames, x => x == Target.Data.GetName()) > -1)
                            {
                                EditorGUILayout.LabelField("Test: " + TestEnumNames[System.Array.FindIndex(TestNames, x => x == Target.Data.GetName())]);
                            }
                            else
                            {
                                EditorGUILayout.LabelField("Train: " + TrainEnumNames[System.Array.FindIndex(TrainNames, x => x == Target.Data.GetName())]);
                            }

                            Utility.SetGUIColor(UltiDraw.Mustard);
                            using (new EditorGUILayout.VerticalScope("Box"))
                            {
                                Utility.ResetGUIColor();

                                EditorGUILayout.BeginHorizontal();
                                EditorGUILayout.LabelField("Data", GUILayout.Width(50f));
                                EditorGUILayout.ObjectField(Target.Data, typeof(MotionData), true);
                                if (Utility.GUIButton("Export", Target.Data.Export ? UltiDraw.Cyan : UltiDraw.Grey, Target.Data.Export ? UltiDraw.Black : UltiDraw.LightGrey))
                                {
                                    Target.Data.Export = !Target.Data.Export;
                                }
                                if (Utility.GUIButton("Symmetric", Target.Data.Symmetric ? UltiDraw.Cyan : UltiDraw.Grey, Target.Data.Symmetric ? UltiDraw.Black : UltiDraw.LightGrey))
                                {
                                    Target.Data.Symmetric = !Target.Data.Symmetric;
                                }
                                //M: 
                                Target.Data.SAMP = true;
                                if (Utility.GUIButton("SAMP", Target.Data.SAMP ? UltiDraw.Cyan : UltiDraw.Grey, Target.Data.SAMP ? UltiDraw.Black : UltiDraw.LightGrey))
                                {
                                    Target.Data.SAMP = !Target.Data.SAMP;
                                }
                                EditorGUILayout.EndHorizontal();

                                EditorGUILayout.BeginHorizontal();
                                GUILayout.FlexibleSpace();
                                EditorGUILayout.LabelField("Frames: " + Target.Data.GetTotalFrames(), GUILayout.Width(100f));
                                EditorGUILayout.LabelField("Time: " + Target.Data.GetTotalTime().ToString("F3") + "s", GUILayout.Width(100f));
                                EditorGUILayout.LabelField("Framerate: " + Target.Data.Framerate.ToString("F1") + "Hz", GUILayout.Width(130f));
                                EditorGUILayout.LabelField("Target Framerate: ", GUILayout.Width(100f));
                                Target.SetTargetFramerate(EditorGUILayout.FloatField(Target.TargetFramerate, GUILayout.Width(40f)));
                                EditorGUILayout.LabelField("Random Seed: ", GUILayout.Width(80f));
                                Target.SetRandomSeed(EditorGUILayout.IntField(Target.RandomSeed, GUILayout.Width(40f)));
                                GUILayout.FlexibleSpace();
                                EditorGUILayout.EndHorizontal();

                            }

                            if (Utility.GUIButton("Add Sequence", UltiDraw.DarkGrey, UltiDraw.White))
                            {
                                Target.Data.AddSequence();
                            }
                            for (int i = 0; i < Target.Data.Sequences.Length; i++)
                            {
                                Utility.SetGUIColor(UltiDraw.White);
                                using (new EditorGUILayout.VerticalScope("Box"))
                                {
                                    Utility.ResetGUIColor();

                                    EditorGUILayout.BeginHorizontal();
                                    GUILayout.FlexibleSpace();
                                    if (Utility.GUIButton("X", Color.cyan, Color.black, 20f, 15f))
                                    {
                                        Target.Data.Sequences[i].SetStart(frame.Index);
                                    }
                                    EditorGUILayout.LabelField("Start", GUILayout.Width(50f));
                                    Target.Data.Sequences[i].SetStart(Mathf.Clamp(EditorGUILayout.IntField(Target.Data.Sequences[i].Start, GUILayout.Width(100f)), 1, Target.Data.GetTotalFrames()));
                                    EditorGUILayout.LabelField("End", GUILayout.Width(50f));
                                    Target.Data.Sequences[i].SetEnd(Mathf.Clamp(EditorGUILayout.IntField(Target.Data.Sequences[i].End, GUILayout.Width(100f)), 1, Target.Data.GetTotalFrames()));
                                    if (Utility.GUIButton("X", Color.cyan, Color.black, 20f, 15f))
                                    {
                                        Target.Data.Sequences[i].SetEnd(frame.Index);
                                    }

                                    if (Utility.GUIButton("X", UltiDraw.DarkRed, Color.black, 40f, 15f))
                                    {
                                        Target.Data.RemoveSequence(Target.Data.Sequences[i]);
                                        i--;
                                    }
                                    GUILayout.FlexibleSpace();
                                    EditorGUILayout.EndHorizontal();
                                }
                            }

                            EditorGUILayout.BeginVertical(GUILayout.Height(25f));
                            Rect ctrl = EditorGUILayout.GetControlRect();
                            Rect rect = new Rect(ctrl.x, ctrl.y, ctrl.width, 25f);
                            EditorGUI.DrawRect(rect, UltiDraw.Black);

                            //View
                            Vector3 view = Target.GetView();

                            //Sequences
                            UltiDraw.Begin();
                            for (int i = 0; i < Target.Data.Sequences.Length; i++)
                            {
                                float _start = (float)(Mathf.Clamp(Target.Data.Sequences[i].Start, view.x, view.y) - view.x) / view.z;
                                float _end = (float)(Mathf.Clamp(Target.Data.Sequences[i].End, view.x, view.y) - view.x) / view.z;
                                float left = rect.x + _start * rect.width;
                                float right = rect.x + _end * rect.width;
                                Vector3 a = new Vector3(left, rect.y, 0f);
                                Vector3 b = new Vector3(right, rect.y, 0f);
                                Vector3 c = new Vector3(left, rect.y + rect.height, 0f);
                                Vector3 d = new Vector3(right, rect.y + rect.height, 0f);
                                UltiDraw.DrawTriangle(a, c, b, UltiDraw.Yellow.Transparent(0.25f));
                                UltiDraw.DrawTriangle(b, c, d, UltiDraw.Yellow.Transparent(0.25f));
                            }
                            UltiDraw.End();

                            //Current Pivot
                            Target.DrawPivot(rect);

                            EditorGUILayout.EndVertical();

                            Utility.SetGUIColor(UltiDraw.DarkGrey);
                            using (new EditorGUILayout.VerticalScope("Box"))
                            {
                                Utility.ResetGUIColor();
                                EditorGUILayout.BeginHorizontal();
                                //GUILayout.FlexibleSpace();
                                if (Target.Playing)
                                {
                                    if (Utility.GUIButton("||", Color.red, Color.black, 50f, 20f))
                                    {
                                        Target.StopAnimation();
                                    }
                                }
                                else
                                {
                                    if (Utility.GUIButton("|>", Color.green, Color.black, 50f, 20f))
                                    {
                                        Target.PlayAnimation();
                                    }
                                }
                                if (Utility.GUIButton("<<", UltiDraw.Grey, UltiDraw.White, 30f, 20f))
                                {
                                    Target.LoadFrame(Mathf.Max(frame.Timestamp - 1f, 0f));
                                }
                                if (Utility.GUIButton("<", UltiDraw.Grey, UltiDraw.White, 20f, 20f))
                                {
                                    Target.LoadFrame(Mathf.Max(frame.Timestamp - 1f / Target.TargetFramerate, 0f));
                                }
                                if (Utility.GUIButton(">", UltiDraw.Grey, UltiDraw.White, 20f, 20f))
                                {
                                    Target.LoadFrame(Mathf.Min(frame.Timestamp + 1f / Target.TargetFramerate, Target.Data.GetTotalTime()));
                                }
                                if (Utility.GUIButton(">>", UltiDraw.Grey, UltiDraw.White, 30f, 20f))
                                {
                                    Target.LoadFrame(Mathf.Min(frame.Timestamp + 1f, Target.Data.GetTotalTime()));
                                }
                                int index = EditorGUILayout.IntSlider(frame.Index, 1, Target.Data.GetTotalFrames());
                                if (index != frame.Index)
                                {
                                    Target.LoadFrame(index);
                                }
                                EditorGUILayout.LabelField(frame.Timestamp.ToString("F3") + "s", Utility.GetFontColor(Color.white), GUILayout.Width(50f));
                                //GUILayout.FlexibleSpace();
                                EditorGUILayout.EndHorizontal();

                                LoadFramePrediction();

                                EditorGUILayout.BeginHorizontal();
                                EditorGUILayout.LabelField("Timescale", Utility.GetFontColor(Color.white), GUILayout.Width(60f), GUILayout.Height(20f));
                                Target.Timescale = EditorGUILayout.FloatField(Target.Timescale, GUILayout.Width(45f), GUILayout.Height(16f));
                                EditorGUILayout.LabelField("Zoom", Utility.GetFontColor(Color.white), GUILayout.Width(52f));
                                Target.Zoom = EditorGUILayout.Slider(Target.Zoom, 0f, 1f);
                                EditorGUILayout.LabelField((100 - Mathf.RoundToInt(100f * (1f - Target.Zoom))) + "%", Utility.GetFontColor(Color.white), GUILayout.Width(50f));
                                EditorGUILayout.EndHorizontal();
                            }

                            EditorGUILayout.BeginHorizontal();
                            if (Utility.GUIButton("Visualise", Target.Visualise ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black))
                            {
                                Target.Visualise = !Target.Visualise;
                            }
                            if (Utility.GUIButton("Mirror", Target.Mirror ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black))
                            {
                                Target.SetMirror(!Target.Mirror);
                            }
                            if (Utility.GUIButton("Callbacks", Target.Callbacks ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black))
                            {
                                Target.SetCallbacks(!Target.Callbacks);
                            }
                            if (Utility.GUIButton("ObjectCallbacks", Target.ObjectCallbacks ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black))
                            {
                                Target.SetObjectCallbacks(!Target.ObjectCallbacks);
                            }
                            EditorGUILayout.EndHorizontal();

                            EditorGUILayout.BeginHorizontal();
                            if (Utility.GUIButton("Inspect All", UltiDraw.DarkGrey, UltiDraw.White))
                            {
                                Target.Data.InspectAll(true);
                            }
                            if (Utility.GUIButton("Inspect None", UltiDraw.DarkGrey, UltiDraw.White))
                            {
                                Target.Data.InspectAll(false);
                            }
                            if (Utility.GUIButton("Visualise All", UltiDraw.DarkGrey, UltiDraw.White))
                            {
                                Target.Data.VisualiseAll(true);
                            }
                            if (Utility.GUIButton("Visualise None", UltiDraw.DarkGrey, UltiDraw.White))
                            {
                                Target.Data.VisualiseAll(false);
                            }
                            EditorGUILayout.EndHorizontal();

                            for (int i = 0; i < Target.Data.Modules.Length; i++)
                            {
                                Target.Data.Modules[i].Inspector(Target);
                            }

                            Utility.ResetGUIColor();
                            Utility.SetGUIColor(UltiDraw.Cyan);
                            int module = EditorGUILayout.Popup(0, ArrayExtensions.Concat(new string[1] { "Add Module..." }, Module.GetIDNames()));
                            if (module > 0)
                            {
                                Target.Data.AddModule(((Module.ID)(module - 1)));
                            }
                            Utility.ResetGUIColor();

                            Utility.SetGUIColor(UltiDraw.LightGrey);
                            using (new EditorGUILayout.VerticalScope("Box"))
                            {
                                Utility.ResetGUIColor();
                                if (Utility.GUIButton("Camera Focus", Target.CameraFocus ? UltiDraw.Cyan : UltiDraw.Grey, Target.CameraFocus ? UltiDraw.Black : UltiDraw.LightGrey))
                                {
                                    Target.SetCameraFocus(!Target.CameraFocus);
                                }
                                if (Target.CameraFocus)
                                {
                                    Target.FocusHeight = EditorGUILayout.FloatField("Focus Height", Target.FocusHeight);
                                    Target.FocusDistance = EditorGUILayout.FloatField("Focus Distance", Target.FocusDistance);
                                    Target.FocusAngle = EditorGUILayout.Slider("Focus Angle", Target.FocusAngle, 0f, 360f);
                                    Target.FocusSmoothing = EditorGUILayout.Slider("Focus Smoothing", Target.FocusSmoothing, 0f, 1f);
                                }

                                if (Utility.GUIButton("Settings", Target.Settings ? UltiDraw.Cyan : UltiDraw.Grey, Target.Settings ? UltiDraw.Black : UltiDraw.LightGrey))
                                {
                                    Target.Settings = !Target.Settings;
                                }
                                if (Target.Settings)
                                {
                                    Target.SetOffset(EditorGUILayout.Vector3Field("Offset", Target.Data.Offset));
                                    Target.SetScale(EditorGUILayout.FloatField("Scale", Target.Data.Scale));
                                    Target.Data.MirrorAxis = (Axis)EditorGUILayout.EnumPopup("Mirror Axis", Target.Data.MirrorAxis);
                                    for (int i = 0; i < Target.Data.Source.Bones.Length; i++)
                                    {
                                        EditorGUILayout.BeginHorizontal();
                                        EditorGUI.BeginDisabledGroup(true);
                                        EditorGUILayout.TextField(Target.Data.Source.GetBoneNames()[i]);
                                        EditorGUI.EndDisabledGroup();
                                        Target.Data.SetSymmetry(i, EditorGUILayout.Popup(Target.Data.Symmetry[i], Target.Data.Source.GetBoneNames()));
                                        EditorGUILayout.LabelField("Mass", GUILayout.Width(40f));
                                        Target.Data.Source.Bones[i].Mass = EditorGUILayout.FloatField(Target.Data.Source.Bones[i].Mass);
                                        EditorGUILayout.LabelField("Alignment", GUILayout.Width(60f));
                                        Target.Data.Source.Bones[i].Alignment = EditorGUILayout.Vector3Field("", Target.Data.Source.Bones[i].Alignment);
                                        EditorGUILayout.EndHorizontal();
                                    }
                                }
                                if (Utility.GUIButton("Create Actor", UltiDraw.DarkGrey, UltiDraw.White))
                                {
                                    Target.Data.CreateActor();
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
#endif