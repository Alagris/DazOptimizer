
bl_info = {
    "name": "Daz Optimizer",
    "blender": (2, 80, 0),
    "category": "Object",
}

if "bpy" in locals():
  import imp
  imp.reload(stages)
  imp.reload(util)
  imp.reload(assets)
  imp.reload(constants)
  print("Reloaded multifiles")
else:
  from .stages import *
  from . import util
  from . import assets
  from . import constants
  print("Imported multifiles")

def install_libraries():
    to_install = ""
    try:
        import PIL
    except ModuleNotFoundError:
        to_install+=" Pillow"

    try:
        import scpipy
    except ModuleNotFoundError:
        to_install+=" scipy==1.10"

    if to_install!="":
        py_exe = sys.executable
        res_path = os.path.realpath(os.path.join(py_exe, "../../lib/site-packages"))
        target = '"--target='+res_path+'"'
        print("=========================================================")
        print("=========================================================")
        print("Run the following command as an admin: ")
        print('&"' + py_exe + '" -m pip install'+to_install)
        print("=========================================================")
        print("=========================================================")


install_libraries()

import io
import os
import re
import shutil
import sys

import base64
import urllib

import numpy as np
import bpy
import bmesh
import json
from collections import namedtuple


class DazOptimize_sidebar(bpy.types.Panel):
    """DazOptim actions"""
    bl_label = "DazOptim"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "DazOptim"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.props = {}

    def draw(self, context):
        col = None
        idx = -1
        for op in operators:
           idx, col = op.draw(self,col,context, idx)

operators = [
    EntryLabel("Character pipeline", 0),
    EntryOp(DazDelCube_operator, "Delete default cube"),
    EntryOp(DazMaleLoad_operator, "Load Daz (male)"),
    EntryOp(DazFemaleLoad_operator, "Load Daz (female)"),
    EntryOp(DazSaveBlend_operator, "Save blend file"),
    EntryOp(DazSaveTextures_operator, "Save textures"),
    EntryOp(RigPhysicsBones, "Rig skirts/dresses for physics"),
    EntryOp(RigPhysicsHair, "Rig anime hair for physics"),
    EntryOp(DazMergeAllRigs_operator, "Merge all rigs"),
    EntryOp(DazRemoveAllSubsurf_operator, "Remove subsurf mods"),
    EntryOp(DazMergeMultiMeshClothes_operator, "Merge clothes sub-meshes"),
    EntryOp(DazMergeAllMaterials_operator, "Merge all materials"),
    EntryOp(DazMergeCumMaterials_operator, "Merge cum materials"),
    EntryOp(DazDecimateCumMeshes_operator, "Decimate cum meshes"),
    EntryOp(DazApplyDecimateCumMeshes_operator, "Apply decimate cum"),
    EntryOp(FixToonEyes, "Fix Nirv Zero eyes"),
    EntryOp(DazGiveErection, "Give erection"),
    EntryFileEnumProp("morphs_file", "morphs"),
    EntryProp('morph_profile', list,[
        ("ALL",'all', ''),
        ("FACS",'only FACS', ''),
        ("BODY",'only body', ''),
        ("SPECIAL",'only special', ''),
        ("GENITALS",'only genitals', ''),
        ("JCM",'only JCM', ''),
        ("FACS_GENITALS", "FACS and gentials", ''),
        ("FACS_GENITALS_SPECIAL", "FACS, genitals and special", ''),
        ("FACS_SPECIAL", "FACS and special", ''),
    ]),
    EntryProp('morph_level', list,[
        ("FULL",'full', "Loads morphs at all profile levels"),
        ("MID",'medium', "Loads morphs at medium profile level"),
        ("MIN",'minimal', "Loads only morphs at minimal profile level"),
    ]),
    EntryOp(SaveMorphs, "Make fav morphs file"),
    EntryOp(LoadMorphs, "Load fav morphs"),
    EntryOp(RebindFavMorphs, "Rebind fav morphs"),
    EntryOp(TransferMorphsToGeografts, "Transfer morphs to geografts"),
    EntryAdditionalBonesEnumProp("additional_bones_file"),
    EntryOp(DazAddAdditionalBones_operator, "Apply additional bones"),
    EntryOp(DazAddBreastBones_operator, "Subdivide breast bones"),
    EntryProp('bake_diffuse', bool, True),
    EntryProp('bake_normal_maps', bool, False),
    EntryProp('bake_roughness_maps', bool, False),
    EntryOp(DazBakeMaterials_operator, "Bake materials"),
    # EntryProp('head_texture', str, ''),
    # EntryProp('arms_texture', str, ''),
    # EntryProp('legs_texture', str, ''),
    # EntryProp('body_texture', str, ''),
    # EntryProp('teeth_texture', str, ''),
    # EntryProp('eyes_texture', str, ''),
    # EntryProp('nails_texture', str, ''),
    # EntryProp('mouth_texture', str, ''),
    # EntryProp('gp_texture', str, ''),
    # EntryProp('penis_texture', str, ''),
    EntryOp(DazSimplifyMaterials_operator, "Simplify materials"),
    EntryOp(DazOptimizeEyes_operator, "Optimize eyes mesh"),
    EntryOp(DazOptimizeEyesForToon_operator, "Optimize eyes for toon"),
    EntryOp(DazOptimizeEyelashes_operator, "Optimize eyelashes"),
    EntryFileEnumProp("eyebrows_file", "eyebrows", extension=".png"),
    EntryOp(DazOptimizeEyebrows_operator, "Optimize eyebrows"),
    EntryOp(DazBakeEyebrows_operator, "Prepare to bake eyebrows (only for artists)"),
    EntryOp(DazRemoveOldEyebrows_operator,"Remove old eyebrows"),
    EntryOp(DazApplyEyebrows_operator, "Apply eyebrows"),
    EntryOp(DazTransferFACSToEyebrow_operator, "Transfer FACS to Eyebrows"),
    EntryOp(DazSimplifyEyesMaterial_operator, "Simplify eyes material"),
    EntryOp(DazSeparateIrisUVs_operator, "Separate iris UVs"),
    EntryOp(DazSimplifyWetKittyMaterials_operator, "Simplify Wet Kitty materials"),
    EntryOp(DazOptimizeGoldenPalaceUVs, "Optimize golden palace UVs"),
    EntryOp(DazSetupGoldenPalaceForBaking, "golden palace prepare baking"),
    EntryOp(DazSelectGoldenPalaceColor_operator, "select golden palace color for baking"),
    EntryOp(DazBakeGoldenPalaceDiffuse, 'Bake'),
    EntryOp(DazSelectGoldenPalaceNormals_operator, "select golden palace normals for baking"),
    EntryOp(DazBakeGoldenPalaceNormal, 'Bake'),
    EntryOp(DazSelectGoldenPalaceRoughness_operator, "select golden palace roughness for baking"),
    EntryOp(DazBakeGoldenPalaceRoughness, 'Bake'),
    EntryOp(DazGoldenPalaceBsdf_operator, "use principled bsdf"),
    EntryOp(DazGoldenPalaceDiffuse_operator, "use diffuse bsdf"),
    EntryOp(DazSaveGoldenPalaceBaked_operator, "Save baked golden palace textures"),
    EntryOp(DazMergeGrografts_operator, "Merge Geografts"),
    EntryOp(DazSimplifyGoldenPalaceMaterials_operator, "Simplify golden palace materials"),
    EntryOp(DazRemoveClitzilla, "Remove GP clitzilla"),
    EntryOp(DazRemoveTentacles, "Remove GP tentacles"),
    EntryOp(DazRemoveRectum, "Simplify GP rectum"),
    EntryOp(DazMergeEyes_operator, "Merge eyes"),
    EntryOp(DazMergeMouth_operator, "Merge mouth"),
    EntryOp(DazRemoveTear_operator, "Remove tear"),
    EntryOp(DazMergeEyebrowsAndEyelashes_operator, "Merge eyebrows+eyelashes"),
    EntryOp(DazMergeToonEyelashesAndBody_operator, "Merge toon eyelashes+body"),
    EntryOp(DazConcatTextures_operator, "Merge textures"),
    EntryOp(DazOptimizeUVs_operator, "Optimize UVs"),
    EntryOp(DazOptimizeUVsHalfGP_operator, "Optimize UVs (half GP)"),
    EntryOp(DazSeparateLipUVs_operator, "Separate Lip UVs"),
    EntryOp(DazMakeSingleMaterial_operator, "Unify skin materials into one"),
    EntryOp(DazMergeEyelashesAndBody_operator, "Merge eyelashes+body"),
    EntryOp(DazFitClothes_operator, "Fit clothes"),
    EntryOp(DazBindFitClothes_operator, "Bind clothes displacement"),
    EntryProp("clothes_displacement", float, 1),
    EntryOp(DazApplyFitClothes_operator, "Apply displacement"),
    EntryOp(DazFitPanties_operator, "Fit panties"),
    EntryOp(DazFitSkinTightClothes_operator, "Fit skin-tight clothes"),
    EntryProp("skin_tight_displacement", float, 0),
    EntryOp(DazApplyFitSkinTightClothes_operator, "Apply skin-tight clothes"),
    EntryOp(DazTransferMissingBonesToClothes_operator, "Transfer new bones to clothes"),
    EntryOp(DazTransferMissingBonesToCum_operator, "Transfer new bones to cum"),
    EntryOp(TransferMorphsToClothes, "Transfer morphs to clothes"),
    EntryOp(TransferMorphsToCum, "Transfer morphs to cum"),
    EntryOp(DazScaleToQuinn, "Scale to Manny height"),
    EntryOp(DazTranslateToQuinn, "Translate to Manny position"),
    EntryOp(DuplicateSkeleton, "Duplicate skeleton"),
    EntryOp(DazConvertToUe5Skeleton_operator, "Convert to UE5 Skeleton"),
    EntryOp(DazReorientBones_operator, "Reorient bones"),
    # EntryOp(DazReweightPelvis_operator, "Reweight pelvis"),
    EntryOp(DazOptimizeHair_operator, "Optimize hair"),
    EntryOp(DazDetachHairFromSkeleton_operator, "Detach hair from skeleton"),
    EntryOp(AddUe5IkBones, "Add UE5 IK bones"),
    EntryOp(DazScaleToUnreal, "Scale to ue5 units"),
    EntryOp(DazExportBodyFbx, "Export body to fbx"),
    EntryOp(DazExportClothesFbx, "Export clothes to fbx"),
    EntryOp(DazExportCumFbx, "Export cum to fbx"),
    EntryOp(DazExportHairFbx, "Export hair to fbx"),
    EntryOp(DazExportGraftsFbx, "Export geografts to fbx"),
    EntryLabel("Animation tools", -1),
    EntryOp(AttachDuplicateSkeleton, "Attach ue5 skeleton"),
    EntryOp(DetachDuplicateSkeleton, "Detach ue5 skeleton"),
    EntryOp(BakeAction, "Bake current daz action to ue5"),
    EntryOp(ExportAction, "Export action to fbx"),
    EntryLabel("Utilities", -1),
    EntryOp(SaveCustomRig, "Save custom rig"),
    EntryOp(DazCompareToUe5Skeleton_operator, "Compare to UE5 Skeleton"),
    EntryOp(HideAllClothes, "Hide all clothes"),
    EntryOp(ShowAllClothes, "Show all clothes"),
    EntryOp(HideAllHair, "Hide all hair"),
    EntryOp(ShowAllHair, "Show all hair"),
    EntryOp(HideAllCum, "Hide all cum"),
    EntryOp(ShowAllCum, "Show all cum"),
    EntryOp(HideAllRigs, "Hide all rigs"),
    EntryOp(UnlockEverything, "Unlock everything"),
    EntryOp(DazAlignPoseQuinn, "Align pose to ue5"),
    EntryOp(RemoveDazBoneConstraints, "Remove daz bone constraints"),
    EntryOp(RemoveDazBoneDrivers, "Remove bone drivers"),
    EntryOp(DazApplyPose, "Apply pose"),
    EntryLabel("Programmer utilities", -1),
    EntryOp(PrintMorphCsv, "Print Morphs CSV"),
    EntryOp(SerializeExtraBones, "Serialize extra bones"),
    EntryOp(SerializeExtraClothes, "Serialize extra clothes"),

]


classes = ([
              DazOptimize_sidebar,
              EasyImportPanel,
          ] + [op.op_class for op in operators if isinstance(op, EntryOp)])


def register():
    for c in classes:
        bpy.utils.register_class(c)
    for o in operators:
        o.on_register()


def unregister():
    for c in classes:
        bpy.utils.unregister_class(c)
    for o in operators:
        o.on_unregister()



if __name__ == '__main__':
    register()
