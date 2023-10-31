using UnityEngine;
using System.Collections;
using System.Collections.Generic;
//public enum RenderingMode
//{
//    Opaque,
//    Cutout,
//    Fade,
//    Transparent,
//}
public class GhostEffect : MonoBehaviour
{
    public Actor actor;
    public int Num = 6;
    public bool Enable = false;
    public Color color = UltiDraw.Blue;
    // Use this for initialization

    public SkinnedMeshRenderer smr;
    List<Ghost> GhostList = new List<Ghost>();

    private void Awake()
    {
        smr = GetComponentInChildren<SkinnedMeshRenderer>();

    }
    void Start()
    {
    }

    // Update is called once per frame
    void LateUpdate()
    {
        if (Enable)
        {
            DrawGhost();
        }
        MoveToUnderground();
    }

    public void MoveToUnderground()
    {
        Vector3 pos = transform.position;
        pos.y = -2f;
        transform.position = pos;
        pos.y = -4f;
        actor.Bones[0].Transform.position = pos;

    }

    public void ClearGhost()
    {
        for (int i = 0; i < GhostList.Count; i++)
        {
            Ghost _ghost = GhostList[i];
            GhostList.Remove(_ghost);
            Destroy(_ghost.mesh);
            Destroy(_ghost);
        }
        Resources.UnloadUnusedAssets();

    }

    public void CreateGhost(ReceiveData.FrameData[] frameposes, Matrix4x4[] root)
    {
        ClearGhost();
        for (int i = 0; i < frameposes.Length; i++)
        {
            Mesh mesh = new Mesh();
            AssignPose(frameposes[i], root[i]);
            smr.BakeMesh(mesh);
            Material material = new Material(smr.material);
            SetMaterialRenderingMode(material);
            GhostList.Add(new Ghost(mesh, material, root[i]));
        }

    }
    public void CreateGhost(ReceiveData.FrameData[] frameposes, Matrix4x4[] root, Color _color)
    {
        ClearGhost();
        for (int i = 0; i < frameposes.Length; i++)
        {
            Mesh mesh = new Mesh();
            AssignPose(frameposes[i], root[i]);
            smr.BakeMesh(mesh);
            Material material = new Material(smr.material);
            SetMaterialRenderingMode(material);
            GhostList.Add(new Ghost(mesh, material, root[i], _color));
        }

    }

    public void CreateGhost(ReceiveData.FrameData[] frameposes, Matrix4x4[] root, Color[] _color, Material[] _material)
    {
        ClearGhost();
        for (int i = 0; i < frameposes.Length; i++)
        {
            Mesh mesh = new Mesh();
            AssignPose(frameposes[i], root[i]);
            smr.BakeMesh(mesh);
            Material material = smr.material;
            if (_material[i] != null)
            {
                material = new Material(_material[i]);
                SetMaterialRenderingMode(material, true);
                GhostList.Add(new Ghost(mesh, material, root[i], _color[i]));
            }
            else
            {
                SetMaterialRenderingMode(material);
                GhostList.Add(new Ghost(mesh, material, root[i], _color[i]));
            }
        }

    }


    public void AssignPose(ReceiveData.FrameData framepose, Matrix4x4 root)
    {
        transform.position = root.GetPosition();
        transform.rotation = root.GetRotation();

        framepose.Reset();
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

        for (int i = 0; i < actor.Bones.Length; i++)
        {

            actor.Bones[i].Transform.position = positions[i];
            actor.Bones[i].Transform.rotation = Quaternion.LookRotation(forwards[i], upwards[i]);
            actor.Bones[i].Velocity = velocities[i];
            actor.Bones[i].ApplyLength();
        }

        framepose.Reset();
    }

    public void DrawGhost()
    {
        for (int i = 0; i < GhostList.Count; i++)
        {
            GhostList[i].Draw(gameObject.layer);
        }
    }
    //private void SetMaterialRenderingMode(Material material, RenderingMode renderingMode)
    private void SetMaterialRenderingMode(Material material, bool notransparent=false)
    {
        if (notransparent)
        {
            material.SetInt("_Cull", (int)UnityEngine.Rendering.CullMode.Back);
            material.SetInt("_ZWrite", 1);
            material.SetInt("_ZTest", (int)UnityEngine.Rendering.CompareFunction.Always);
            material.SetFloat("_Power", 0.25f);


        }
        else
        {
            material.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
            material.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            material.SetInt("_ZWrite", 0);
            material.DisableKeyword("_ALPHATEST_ON");
            material.EnableKeyword("_ALPHABLEND_ON");
            material.DisableKeyword("_ALPHAPREMULTIPLY_ON");
        }

        material.renderQueue = 3000;

        //switch (renderingMode)
        //{
        //    case RenderingMode.Opaque:
        //        material.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.One);
        //        material.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.Zero);
        //        material.SetInt("_ZWrite", 1);
        //        material.DisableKeyword("_ALPHATEST_ON");
        //        material.DisableKeyword("_ALPHABLEND_ON");
        //        material.DisableKeyword("_ALPHAPREMULTIPLY_ON");
        //        material.renderQueue = -1;
        //        break;
        //    case RenderingMode.Cutout:
        //        material.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.One);
        //        material.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.Zero);
        //        material.SetInt("_ZWrite", 1);
        //        material.EnableKeyword("_ALPHATEST_ON");
        //        material.DisableKeyword("_ALPHABLEND_ON");
        //        material.DisableKeyword("_ALPHAPREMULTIPLY_ON");
        //        material.renderQueue = 2450;
        //        break;
        //    case RenderingMode.Fade:
        //        material.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
        //        material.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
        //        material.SetInt("_ZWrite", 0);
        //        material.DisableKeyword("_ALPHATEST_ON");
        //        material.EnableKeyword("_ALPHABLEND_ON");
        //        material.DisableKeyword("_ALPHAPREMULTIPLY_ON");
        //        material.renderQueue = 3000;
        //        break;
        //    case RenderingMode.Transparent:
        //        material.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.One);
        //        material.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
        //        material.SetInt("_ZWrite", 0);
        //        material.DisableKeyword("_ALPHATEST_ON");
        //        material.DisableKeyword("_ALPHABLEND_ON");
        //        material.EnableKeyword("_ALPHAPREMULTIPLY_ON");
        //        material.renderQueue = 3000;
        //        break;
        //}
    }
    void OnRenderObject()
    {
        if (Enable)
        {
            //UltiDraw.Begin();
            //for (int i = 0; i < GhostList.Count; i++)
            //{
            //    for (int j = 0; j < GhostList[i].mesh.vertices.Length; j+=100)
            //    {
            //        //Vector3 v = GhostList[i].mesh.vertices[j];
            //        Vector3 v = smr.sharedMesh.vertices[j];
            //        Vector3 pos = v.GetRelativePositionFrom(GhostList[i].mat);

            //        UltiDraw.DrawSphere(pos, Quaternion.identity, 0.02f, UltiDraw.Red.Transparent(0.25f));

            //    }


            //}
            //UltiDraw.End();
        }
    }
}

public class Ghost : Object
{
    public Mesh mesh;
    public Material material;
    public Matrix4x4 mat;
    public Color color;

    public Ghost(Mesh _mesh, Material _material, Matrix4x4 _mat, Color _color)
    {
        mesh = _mesh;
        material = _material;
        mat = _mat;
        color = _color;
    }

    public Ghost(Mesh _mesh, Material _material, Matrix4x4 _mat)
    {
        mesh = _mesh;
        material = _material;
        mat = _mat;
        color = material.color;
    }



    public void Draw(int layer)
    {
        material.color = color;
        Graphics.DrawMesh(mesh, mat, material, layer);

    }

}