# Unity code

## Description
The runtime Unity code is largly based on the codes of [SAMP](https://samp.is.tue.mpg.de/) by [Mohamed Hassan](https://github.com/mohamedhassanmus) and [Neural State Machine](https://github.com/sebastianstarke/AI4Animation/tree/master/AI4Animation/SIGGRAPH_Asia_2019) by [Sebastian Starke](https://github.com/sebastianstarke).

## Running
Our code uses Unity to visualize and test. We rely on Python to run the network and send data with socket.

### Build socket servers with python
- Build socket servers of each module with Python. Please look at [Python](../../python/README.md). You should run servers on a machine with ip x.x.x.x.
- Please assign the ip (x.x.x.x) in TrajClient (you will see it when you open an .unity, it will be in the left pane. If you want to change ip, please double click it and change the number in the right pane.) and make sure port number is correct in Unity.

### HGHOI_Demo
- Open the Demo Scene (Unity -> Assets -> Demo -> HGHOI_Demo.unity).
- Hit the Play button.
- You can control the character by SAMP first (WSAD, QE, Left-Shift, C, L).
- You can click the button in the left (Sit / Lie) to test HGHOI.

### HGHOI_Test
- Open the Test Scene (Unity -> Assets -> Demo -> HGHOI_Demo.unity).
- Hit the Play button.
- You can see the testing starts.

## Exporting Training Data
### MotionNet Data
You need to export the data from Unity:
- Open the Mocap Scene (Unity -> Assets -> MotionCapture -> MotionNet_scene.unity).
- Click on the Editor game object in the scene hierarchy window.
- Open the MotionNet Exporter (Header -> AI4Animation -> MotionNet Exporter).
- Choose whether you want to export train or test data.
- Click `Reload`
- Click `Export`

## Note
In the demo, there will be many corner cases where the system may fail due to the exponential combinatorial amount of possible actions and interactions of the character with the environment.

## License
By using this code, you agree to adhere with the liscense of [AI4Animation](https://github.com/sebastianstarke/AI4Animation#copyright-information). In addition:

1.	You may use, reproduce, modify, and display the research materials provided under this license (the “Research Materials”) solely for noncommercial purposes. Noncommercial purposes include academic research, teaching, and testing, but do not include commercial licensing or distribution, development of commercial products, or any other activity which results in commercial gain. You may not redistribute the Research Materials.
2.	You agree to (a) comply with all laws and regulations applicable to your use of the Research Materials under this license, including but not limited to any import or export laws; (b) preserve any copyright or other notices from the Research Materials; and (c) for any Research Materials in object code, not attempt to modify, reverse engineer, or decompile such Research Materials except as permitted by applicable law.
3.	THE RESEARCH MATERIALS ARE PROVIDED “AS IS,” WITHOUT WARRANTY OF ANY KIND, AND YOU ASSUME ALL RISKS ASSOCIATED WITH THEIR USE. IN NO EVENT WILL ANYONE BE LIABLE TO YOU FOR ANY ACTUAL, INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF OR IN CONNECTION WITH USE OF THE RESEARCH MATERIALS.
