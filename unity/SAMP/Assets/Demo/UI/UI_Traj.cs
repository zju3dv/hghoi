using System.Collections;
using UnityEngine;
using UnityEngine.UI;

public class UI_Traj : MonoBehaviour
{
    public bool Visualise = true;

    public System_Demo Animation;

    private ColorBlock Active;
    private ColorBlock Inactive;
    private ColorBlock Playing;
    private ColorBlock Stop;

    private Slider slider;
    private int Pivot = 0;

    public void ResetSlider()
    {
        slider.value = 0f;
    }

    private void Start()
    {
        slider = GameObject.Find("Slider").GetComponent<Slider>();
        Transform buttons = transform.Find("Buttons");
        slider.transform.localPosition = new Vector3(-100f, 325f, 0f);
        slider.transform.Find("Text").GetComponent<Text>().text = "-1/0";
    }

    void Awake()
    {
        Transform buttons = transform.Find("Buttons");
        for (int i = 0; i < buttons.childCount; i++)
        {
            Button button = buttons.GetChild(i).GetComponent<Button>();
            button.transform.Find("Text").GetComponent<Text>().text = button.name;
            button.transform.localPosition += new Vector3(0f, i * 50f, 0f);

            Active = button.colors;
            Active.normalColor = UltiDraw.Gold;
            Active.pressedColor = UltiDraw.Gold;
            Active.disabledColor = UltiDraw.Gold;
            Active.highlightedColor = UltiDraw.Gold;
            Inactive = button.colors;
            Inactive.normalColor = UltiDraw.DarkGrey;
            Inactive.pressedColor = UltiDraw.DarkGrey;
            Inactive.disabledColor = UltiDraw.DarkGrey;
            Inactive.highlightedColor = UltiDraw.DarkGrey;

            Playing = button.colors;
            Playing.normalColor = UltiDraw.DarkRed;
            Playing.pressedColor = UltiDraw.DarkRed;
            Playing.disabledColor = UltiDraw.DarkRed;
            Playing.highlightedColor = UltiDraw.DarkRed;

            Stop = button.colors;
            Stop.normalColor = UltiDraw.DarkGreen;
            Stop.pressedColor = UltiDraw.DarkGreen;
            Stop.disabledColor = UltiDraw.DarkGreen;
            Stop.highlightedColor = UltiDraw.DarkGreen;

            button.colors = Inactive;
            switch (button.name)
            {
                case "BiDirectional":
                    button.onClick.AddListener(() =>
                    {
                        Animation.ShowBiDirectional = !Animation.ShowBiDirectional;
                        button.colors = Animation.ShowBiDirectional ? Active : Inactive;
                    });
                    button.colors = Animation.ShowBiDirectional ? Active : Inactive;
                    break;
                case "Root":
                    button.onClick.AddListener(() =>
                    {
                        Animation.ShowRoot = !Animation.ShowRoot;
                        button.colors = Animation.ShowRoot ? Active : Inactive;
                    });
                    button.colors = Animation.ShowRoot ? Active : Inactive;
                    break;
                case "Goal":
                    button.onClick.AddListener(() =>
                    {
                        Animation.ShowGoal = !Animation.ShowGoal;
                        button.colors = Animation.ShowGoal ? Active : Inactive;
                    });
                    button.colors = Animation.ShowGoal ? Active : Inactive;
                    break;
                case "Current":
                    button.onClick.AddListener(() =>
                    {
                        Animation.ShowCurrent = !Animation.ShowCurrent;
                        button.colors = Animation.ShowCurrent ? Active : Inactive;
                    });
                    button.colors = Animation.ShowCurrent ? Active : Inactive;
                    break;
                case "Interaction":
                    button.onClick.AddListener(() =>
                    {
                        Animation.ShowInteraction = !Animation.ShowInteraction;
                        button.colors = Animation.ShowInteraction ? Active : Inactive;
                    });
                    button.colors = Animation.ShowInteraction ? Active : Inactive;
                    break;
                case "Envsmall":
                    button.onClick.AddListener(() =>
                    {
                        Animation.ShowEnvsmall = !Animation.ShowEnvsmall;
                        button.colors = Animation.ShowEnvsmall ? Active : Inactive;
                    });
                    button.colors = Animation.ShowEnvsmall ? Active : Inactive;
                    break;
                case "Envbig":
                    button.onClick.AddListener(() =>
                    {
                        Animation.ShowEnvbig = !Animation.ShowEnvbig;
                        button.colors = Animation.ShowEnvbig ? Active : Inactive;
                    });
                    button.colors = Animation.ShowEnvbig ? Active : Inactive;
                    break;
                case "Skeleton":
                    button.onClick.AddListener(() =>
                    {
                        Animation.Actor.DrawSkeleton = !Animation.Actor.DrawSkeleton;
                        button.colors = Animation.Actor.DrawSkeleton ? Active : Inactive;
                    });
                    button.colors = Animation.Actor.DrawSkeleton ? Active : Inactive;
                    break;
                case "PredSit":
                    button.onClick.AddListener(() =>
                    {
                        Animation.ReceiveData = false;
                        string goalaction = "Sit";
                        Animation.PredictInteraction(goalaction);
                        button.colors = Animation.ReceiveData ? Active : Inactive;
                    });
                    button.colors = Animation.ReceiveData ? Active : Inactive;
                    break;
                case "PredLie":
                    button.onClick.AddListener(() =>
                    {
                        Animation.ReceiveData = false;
                        string goalaction = "Liedown";
                        Animation.PredictInteraction(goalaction);
                        button.colors = Animation.ReceiveData ? Active : Inactive;
                    });
                    button.colors = Animation.ReceiveData ? Active : Inactive;
                    break;
                case "PredIdle":
                    button.onClick.AddListener(() =>
                    {
                        Animation.ReceiveData = false;
                        string goalaction = "Idle";
                        Animation.PredictInteraction(goalaction);
                        button.colors = Animation.ReceiveData ? Active : Inactive;
                    });
                    button.colors = Animation.ReceiveData ? Active : Inactive;
                    break;
                case "SAMP":
                    button.onClick.AddListener(() =>
                    {
                        Animation.ReceiveData = false;
                        Animation.reinit_funcs();
                        button.colors = Animation.ReceiveData ? Active : Inactive;
                    });
                    button.colors = Animation.ReceiveData ? Active : Inactive;
                    break;
                case "Go":
                    button.onClick.AddListener(() =>
                    {
                        Animation.Playing = !Animation.Playing;
                        button.transform.Find("Text").GetComponent<Text>().text = Animation.Playing ? "Stop" : "Go";
                        button.colors = Animation.Playing ? Playing : Stop;
                    });
                    button.colors = Animation.Playing ? Playing : Stop;
                    button.transform.Find("Text").GetComponent<Text>().text = Animation.Playing ? "Stop" : "Go";
                    break;
                case "Next":
                    button.onClick.AddListener(() =>
                    {
                        NextFrame();
                    });
                    break;
                case "Previous":
                    button.onClick.AddListener(() =>
                    {
                        PreviousFrame();
                    });
                    break;
            }
        }
    }

    void NextFrame()
    {
        if (Animation.ReceiveData)
        {
            int num_frames = Animation.TrajData.GetFramesNum;

            Pivot = Pivot + 1;
            slider.value = (float)Pivot / num_frames;
            if (slider.value >= 1f)
            {
                slider.value = 0f;
            }
            if (Pivot >= num_frames)
            {
                Pivot = 0;
            }

            //float interval = 1f / num_frames;
            //slider.value += interval;
            //if (slider.value >= 1f)
            //{
            //    slider.value = 0f;
            //}
        }
    }
    void PreviousFrame()
    {
        if (Animation.ReceiveData)
        {
            int num_frames = Animation.TrajData.GetFramesNum;

            Pivot = Pivot - 1;
            slider.value = (float)Pivot / num_frames;
            if (slider.value <= 0f)
            {
                slider.value = 1f;
            }
            if (Pivot <= -1)
            {
                Pivot = num_frames - 1;
            }

        }
    }

    void Update()
    {
        //Debug.Log(Pivot);
        if (Animation.ReceiveData)
        {
            Pivot = Animation.TrajData.Pivot;
            int num_frames = Animation.TrajData.GetFramesNum;
            if (Animation.Playing)
            {

                //float interval = 1f / num_frames;
                //slider.value += interval;
                Pivot = Pivot + 1;
                slider.value = (float)Pivot / num_frames;
                if (slider.value >= 1f)
                {
                    slider.value = 1f;
                }
            }
            else
            {
                Pivot = (int)Mathf.Round((slider.value * num_frames));
            }
            if (Pivot >= num_frames)
            {
                Pivot = num_frames - 1;
            }
            slider.transform.Find("Text").GetComponent<Text>().text = $"{Pivot + 1}/{num_frames}";
            Animation.TrajData.Pivot = Pivot;
        }
    }

}
