from typing_extensions import override

from .daz_optim import *

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


def pass_stage(stage):
    s = bpy.context.scene.get('daz_optim_stage', '')
    if stage.stage_id not in s:
        bpy.context.scene['daz_optim_stage'] = s + stage.stage_id

def toggle_stage(toggle_on, toggle_off):
    s:str = bpy.context.scene.get('daz_optim_stage', '')
    for off in toggle_off:
        s = s.replace(off.stage_id, '')
    if toggle_on.stage_id not in s:
        s += toggle_on.stage_id
    bpy.context.scene['daz_optim_stage'] = s


def check_stage(context, required_stage_ids, forbidden_stage_ids):
    stage = context.scene.get('daz_optim_stage', '')
    for i in required_stage_ids:
        if i.stage_id not in stage:
            return False
    for i in forbidden_stage_ids:
        if i.stage_id in stage:
            return False
    return True

def check_stage_any(context, required_any_of_stage_ids, forbidden_stage_ids):
    stage = context.scene.get('daz_optim_stage', '')
    has_one = False
    for i in required_any_of_stage_ids:
        if i.stage_id in stage:
            has_one = True
    if not has_one:
        return False
    for i in forbidden_stage_ids:
        if i.stage_id in stage:
            return False
    return True

def load_daz(context, is_female):
    bpy.ops.daz.easy_import_daz('INVOKE_DEFAULT',
                                # filepath=self.duf_path,
                                # files=[filepath],
                                # directory="",
                                filter_glob="*.duf;*.dsf;*.png;*.jpeg;*.jpg;*.bmp",
                                fitMeshes='DBZFILE',
                                materialMethod='EXTENDED_PRINCIPLED',
                                useMergeMaterials=True,
                                useEliminateEmpties=True,
                                useMergeRigs=False,
                                useApplyTransforms=False,
                                useMergeToes=False,
                                useFavoMorphs=False,
                                useUnits=False,
                                useExpressions=False,
                                useVisemes=False,
                                useHead=False,
                                useFacs=False,
                                useFacsdetails=False,
                                useFacsexpr=False,
                                useAnime=False,
                                usePowerpose=False,
                                useBody=False,
                                useBulges=False,
                                useJcms=False,
                                ignoreFingers=True,
                                useMasculine=False,
                                useFeminine=False,
                                useFlexions=False,
                                useBakedCorrectives=False,
                                useDazFavorites=False,
                                useAdjusters=False,
                                onMorphSuffix='SMART',
                                useTransferFace=True,
                                useTransferHair=False,
                                useTransferGeografts=False,
                                useTransferClothes=False,
                                useTransferHD=False,
                                useMergeGeografts=False,
                                useMakePosable=True,
                                useFinalOptimization=False,
                                ignoreUrl=False,
                                ignoreFinger=False,
                                morphSuffix="",
                                ignoreHdMorphs=False,
                                useMhxOnly=False,
                                duplicateDistance=1,
                                useMergeNonConforming='CONTROLS',
                                useConvertWidgets=True,
                                useHiddenRigs=False,
                                useMergeUvs=True,
                                allowOverlap=False,
                                keepOriginal=False,
                                useFixTiles=True,
                                useSubDDisplacement=True,
                                useGeoNodes=False,
                                # morphStrength=1,
                                skinColor=(0.6, 0.4, 0.25, 1),
                                clothesColor=(0.09, 0.01, 0.015, 1),
                                # useApplyRestPoses=True,
                                favoPath="")
    context.scene['daz_optim_stage'] = DazMaleLoad_operator.stage_id
    assert DazMaleLoad_operator.stage_id == DazFemaleLoad_operator.stage_id
    context.scene['daz_optim_female'] = is_female
    return True



class DazFemaleLoad_operator(bpy.types.Operator):
    """ Load female daz character """
    bl_idname = "dazoptim.load_female"
    bl_label = "Load female Daz character"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 'a'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [], [DazFemaleLoad_operator])

    def execute(self, context):
        if load_daz(context, True):
            pass_stage(self)
        return {'FINISHED'}


class DazMaleLoad_operator(bpy.types.Operator):
    """ Load male daz character """
    bl_idname = "dazoptim.load_male"
    bl_label = "Load male Daz character"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 'a'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [], [DazMaleLoad_operator])

    def execute(self, context):
        if load_daz(context, False):
            pass_stage(self)
        return {'FINISHED'}


class DazSaveBlend_operator(bpy.types.Operator):
    bl_idname = "dazoptim.save_blend"
    bl_label = "Save blend file"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 'c'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazMaleLoad_operator], [])

    def execute(self, context):
        if bpy.types.dazoptim_easy_import_panel.filepath is None and 'duf_filepath' not in bpy.context.scene:
            self.report({"WARNING"}, "Load a DAZ character first!")
            return {'CANCELLED'}
        if bpy.types.dazoptim_easy_import_panel.filepath is not None and 'duf_filepath' not in bpy.context.scene:
            bpy.context.scene['duf_filepath'] = bpy.types.dazoptim_easy_import_panel.filepath

        body_mesh = find_body_mesh()
        is_t = bpy.context.scene['daz_optim_toon'] = is_toon(body_mesh)
        is_gp = bpy.context.scene['daz_optim_gp'] = has_gp()
        if is_t:
            bpy.context.scene['is_nirv_zero'] = 'nirv zero' in body_mesh.name.lower()
        for o in bpy.data.objects:
            if o.name.startswith("Love Loads"):
                bpy.context.scene['has_love_loads'] = True
        save_blend_file(bpy.context.scene['duf_filepath'])
        pass_stage(self)
        if os.path.isdir(DazOptimizer().textures_dir()):
            pass_stage(DazSaveTextures_operator)
        return {'FINISHED'}

class FixToonEyes(bpy.types.Operator):
    bl_idname = "dazoptim.fix_toon_eyes"
    bl_label = "Fix Nirv Zero eyes"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = '?'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazMergeAllRigs_operator], [FixToonEyes]) and bpy.context.scene.get('is_nirv_zero')

    def execute(self, context):
        DazOptimizer().fix_toon_eyes()
        pass_stage(self)
        return {'FINISHED'}

class DazGiveErection(bpy.types.Operator):
    bl_idname = "dazoptim.give_erection"
    bl_label = "Give erection"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = '!'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazMergeAllRigs_operator], [DazGiveErection]) and has_dick() is not None

    def execute(self, context):
        DazOptimizer().give_erection()
        pass_stage(self)
        return {'FINISHED'}

class DazSaveTextures_operator(bpy.types.Operator):
    bl_idname = "dazoptim.save_textures"
    bl_label = "Save Daz textures"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 'd'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazSaveBlend_operator], [])

    def execute(self, context):
        DazOptimizer().save_textures()
        pass_stage(self)
        return {'FINISHED'}


def has_gp():
    return DazOptimizer.get_mesh_by_name('GoldenPalace_G9 Mesh') is not None

def had_gp():
    return bpy.context.scene.get('daz_optim_gp')

class DazMergeCumMaterials_operator(bpy.types.Operator):
    bl_idname = "dazoptim.merge_cum_materials"
    bl_label = "Merge cum materials"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = '['

    @classmethod
    def poll(cls, context):
        return UNLOCK or bpy.context.scene.get('has_love_loads') and check_stage(context, [DazSaveTextures_operator], [DazMergeCumMaterials_operator])

    def execute(self, context):
        DazOptimizer().merge_cum_materials()
        pass_stage(self)
        return {'FINISHED'}

class DazDecimateCumMeshes_operator(bpy.types.Operator):
    bl_idname = "dazoptim.decimate_cum_meshes"
    bl_label = "Decimate cum meshes"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = '-'

    @classmethod
    def poll(cls, context):
        return UNLOCK or bpy.context.scene.get('has_love_loads') and check_stage(context, [DazSaveTextures_operator], [DazDecimateCumMeshes_operator])

    def execute(self, context):
        DazOptimizer().decimate_cum_meshes()
        pass_stage(self)
        return {'FINISHED'}

class DazRemoveClitzilla(bpy.types.Operator):
    bl_idname = "dazoptim.remove_clitzilla"
    bl_label = "Remove clitzilla"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = '$'

    @classmethod
    def poll(cls, context):
        return UNLOCK or (had_gp() and check_stage(context, [DazMergeGrografts_operator], [DazRemoveClitzilla]))

    def execute(self, context):
        DazOptimizer().remove_clitzilla()
        pass_stage(self)
        return {'FINISHED'}

class DazRemoveRectum(bpy.types.Operator):
    bl_idname = "dazoptim.remove_rectum"
    bl_label = "Remove rectum"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = '~'

    @classmethod
    def poll(cls, context):
        return UNLOCK or (had_gp() and check_stage(context, [DazMergeGrografts_operator], [DazRemoveRectum]))

    def execute(self, context):
        DazOptimizer().remove_rectum_bones()
        pass_stage(self)
        return {'FINISHED'}

class DazRemoveTentacles(bpy.types.Operator):
    bl_idname = "dazoptim.remove_tentacles"
    bl_label = "Remove tentacles"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = '='

    @classmethod
    def poll(cls, context):
        return UNLOCK or (had_gp() and check_stage(context, [DazMergeGrografts_operator], [DazRemoveTentacles]))

    def execute(self, context):
        DazOptimizer().remove_tentacles()
        pass_stage(self)
        return {'FINISHED'}

class DazApplyDecimateCumMeshes_operator(bpy.types.Operator):
    bl_idname = "dazoptim.apply_decimate_cum_meshes"
    bl_label = "Apply decimate cum meshes"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = '|'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazDecimateCumMeshes_operator], [DazApplyDecimateCumMeshes_operator])

    def execute(self, context):
        DazOptimizer().apply_decimate_cum_meshes()
        pass_stage(self)
        return {'FINISHED'}

class DazMergeAllMaterials_operator(bpy.types.Operator):
    bl_idname = "dazoptim.merge_all_materials"
    bl_label = "Merge all materials"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = '('

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazSaveTextures_operator], [DazMergeAllMaterials_operator])

    def execute(self, context):
        DazOptimizer().merge_all_materials()
        pass_stage(self)
        return {'FINISHED'}

class DazRemoveAllSubsurf_operator(bpy.types.Operator):
    bl_idname = "dazoptim.remove_subsurf"
    bl_label = "Remove subsurface modifiers"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = '@'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazSaveTextures_operator], [DazRemoveAllSubsurf_operator])

    def execute(self, context):
        DazOptimizer().remove_all_subsurfs()
        pass_stage(self)
        return {'FINISHED'}

class DazMergeAllRigs_operator(bpy.types.Operator):
    bl_idname = "dazoptim.merge_all_rigs"
    bl_label = "Merge all rigs (except hair)"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 'e'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazSaveTextures_operator], [DazMergeAllRigs_operator])

    def execute(self, context):
        DazOptimizer().merge_all_rigs()
        pass_stage(self)
        return {'FINISHED'}

class DazMergeMultiMeshClothes_operator(bpy.types.Operator):
    bl_idname = "dazoptim.merge_multi_mesh_clothes"
    bl_label = "Merge all clothes that are made up of senselessly separated tiny bits"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = '{'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazSaveTextures_operator], [DazMergeMultiMeshClothes_operator])

    def execute(self, context):
        DazOptimizer().merge_multi_mesh_clothes()
        pass_stage(self)
        return {'FINISHED'}

class DazSelectGoldenPalaceColor_operator(bpy.types.Operator):
    bl_idname = "dazoptim.select_gp_color"
    bl_label = "Select golden palace base color for baking"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 'f'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazSetupGoldenPalaceForBaking], [DazMergeGrografts_operator])

    def execute(self, context):
        DazOptimizer().select_gp_color_for_baking()
        toggle_stage(self, [DazSelectGoldenPalaceNormals_operator, DazSelectGoldenPalaceRoughness_operator])
        return {'FINISHED'}


class DazGoldenPalaceBsdf_operator(bpy.types.Operator):
    bl_idname = "dazoptim.gp_baking_principled"
    bl_label = " golden palace use principle bsdf for baking"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 'g'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazSetupGoldenPalaceForBaking], [DazMergeGrografts_operator])

    def execute(self, context):
        DazOptimizer().select_golden_palace_for_bsdf_mode_baking(True)
        pass_stage(self)
        return {'FINISHED'}

class DazGoldenPalaceDiffuse_operator(bpy.types.Operator):
    bl_idname = "dazoptim.gp_baking_diffuse"
    bl_label = " golden palace use diffuse node for baking"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 'h'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazSetupGoldenPalaceForBaking], [DazMergeGrografts_operator])

    def execute(self, context):
        DazOptimizer().select_golden_palace_for_bsdf_mode_baking(False)
        pass_stage(self)
        return {'FINISHED'}

class DazSelectGoldenPalaceNormals_operator(bpy.types.Operator):
    bl_idname = "dazoptim.select_gp_normal"
    bl_label = "Select golden palace normal maps for baking"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 'i'

    @classmethod
    def poll(cls, context):
        return UNLOCK or not bpy.context.scene.get('gp_lacks_Normal') and check_stage(context, [DazSetupGoldenPalaceForBaking], [DazMergeGrografts_operator])

    def execute(self, context):
        DazOptimizer().select_gp_normals_for_baking()
        toggle_stage(self, [DazSelectGoldenPalaceColor_operator, DazSelectGoldenPalaceRoughness_operator])
        return {'FINISHED'}

class DazSelectGoldenPalaceRoughness_operator(bpy.types.Operator):
    bl_idname = "dazoptim.select_gp_roughness"
    bl_label = "Select golden palace roughness maps for baking"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 'j'

    @classmethod
    def poll(cls, context):
        return UNLOCK or not bpy.context.scene.get('gp_lacks_Roughness') and check_stage(context, [DazSetupGoldenPalaceForBaking], [DazMergeGrografts_operator])

    def execute(self, context):
        DazOptimizer().select_gp_roughness_for_baking()
        toggle_stage(self, [DazSelectGoldenPalaceNormals_operator, DazSelectGoldenPalaceColor_operator])
        return {'FINISHED'}

class DazSaveGoldenPalaceBaked_operator(bpy.types.Operator):
    bl_idname = "dazoptim.save_gp_baked"
    bl_label = "Save baked golden palace textures"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 'k'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage_any(context, [DazBakeGoldenPalaceNormal, DazBakeGoldenPalaceDiffuse, DazBakeGoldenPalaceRoughness],[])

    def execute(self, context):
        DazOptimizer().save_gp_textures()
        pass_stage(self)
        return {'FINISHED'}

class DazSimplifyWetKittyMaterials_operator(bpy.types.Operator):
    bl_idname = "dazoptim.simplify_wk"
    bl_label = "Simplify wet kitty materials"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = '}'

    @classmethod
    def poll(cls, context):
        if UNLOCK:
            return True
        return 'Wet Kitty TOON Mesh' in bpy.data.objects and check_stage(context, [DazSimplifyMaterials_operator], [DazSimplifyWetKittyMaterials_operator])

    def execute(self, context):
        DazOptimizer().simplify_wet_kitty()
        pass_stage(self)
        return {'FINISHED'}

class DazSimplifyGoldenPalaceMaterials_operator(bpy.types.Operator):
    bl_idname = "dazoptim.simplify_gp_mats"
    bl_label = "Simplify golden palace materials after baking"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 'l'

    @classmethod
    def poll(cls, context):
        if UNLOCK:
            return True
        if not check_stage(context, [DazOptimizeGoldenPalaceUVs, DazMergeGrografts_operator], [DazSimplifyGoldenPalaceMaterials_operator]):
            return False
        return check_stage(context, [DazSaveGoldenPalaceBaked_operator], []) or os.path.exists(DazOptimizer().get_gp_texture_path("Base Color"))

    def execute(self, context):
        DazOptimizer().simplify_golden_palace_material()
        pass_stage(self)
        return {'FINISHED'}


class DazBakeMaterials_operator(bpy.types.Operator):
    """ Simplify materials """
    bl_idname = "dazoptim.bake_mats"
    bl_label = "Simplify materials"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 'Q'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazSaveTextures_operator], [DazSimplifyMaterials_operator, DazMergeGrografts_operator])

    def execute(self, context):
        DazOptimizer().bake_materials()
        pass_stage(self)
        return {'FINISHED'}

class DazSimplifyMaterials_operator(bpy.types.Operator):
    """ Simplify materials """
    bl_idname = "dazoptim.simpl_mats"
    bl_label = "Simplify materials"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 'm'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazSaveTextures_operator], [DazSimplifyMaterials_operator, DazMergeGrografts_operator])

    def execute(self, context):
        DazOptimizer().simplify_materials()
        pass_stage(self)
        return {'FINISHED'}

class DazOptimizeEyesForToon_operator(bpy.types.Operator):
    """ Optimize eyes for toon """
    bl_idname = "dazoptim.optim_eyes_for_toon"
    bl_label = "Optimize eyes"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 'n'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazSaveTextures_operator], [DazOptimizeEyes_operator]) and bpy.context.scene.get('daz_optim_toon')

    def execute(self, context):
        DazOptimizer().optimize_eyes(optimize_for_toon=True, hard_toon_edges=True)
        pass_stage(self)
        return {'FINISHED'}

class DazOptimizeEyes_operator(bpy.types.Operator):
    """ Optimize eyes """
    bl_idname = "dazoptim.optim_eyes"
    bl_label = "Optimize eyes"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 'n'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazSaveTextures_operator], [DazOptimizeEyes_operator])

    def execute(self, context):
        DazOptimizer().optimize_eyes(False)
        pass_stage(self)
        return {'FINISHED'}


class DazOptimizeEyelashes_operator(bpy.types.Operator):
    """ Optimize eyes """
    bl_idname = "dazoptim.optim_eyelashes"
    bl_label = "Optimize eyelashes"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 'o'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazSaveTextures_operator], [DazOptimizeEyelashes_operator])

    def execute(self, context):
        DazOptimizer().optimize_eyelashes()
        pass_stage(self)
        return {'FINISHED'}

class DazApplyEyebrows_operator(bpy.types.Operator):
    """ Apply eyebrows """
    bl_idname = "dazoptim.apply_eyebrows"
    bl_label = "Apply eyebrows"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 'p'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazRemoveOldEyebrows_operator], [DazApplyEyebrows_operator])

    def execute(self, context):
        DazOptimizer().apply_optimized_eyebrows()
        pass_stage(self)
        return {'FINISHED'}


class DazOptimizeEyebrows_operator(bpy.types.Operator):
    """ Optimize eyes """
    bl_idname = "dazoptim.optim_eyebrows"
    bl_label = "Optimize eyebrows"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 'q'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazSaveTextures_operator], [DazOptimizeEyebrows_operator])

    def execute(self, context):
        DazOptimizer().optimize_eyebrows()
        pass_stage(self)
        return {'FINISHED'}

class DazBakeEyebrows_operator(bpy.types.Operator):
    """ Optimize eyes """
    bl_idname = "dazoptim.bake_eyebrows"
    bl_label = "Bake eyebrows"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazOptimizeEyebrows_operator], [DazRemoveOldEyebrows_operator])

    def execute(self, context):
        d = DazOptimizer()
        s = bpy.context.scene
        s.render.engine = 'CYCLES'
        s.cycles.device = 'GPU'
        s.cycles.bake_type = 'DIFFUSE'
        s.render.bake.margin = 0
        s.render.bake.use_selected_to_active = True
        s.render.bake.use_pass_direct = False
        s.render.bake.use_pass_indirect = False
        s.render.bake.use_pass_color = True
        old_eyebrows = d.get_eyebrows(True)
        new_eyebrows = bpy.data.objects['Eyebrows Mesh']
        select_object(new_eyebrows)
        old_eyebrows.select_set(True)
        return {'FINISHED'}

class DazRemoveOldEyebrows_operator(bpy.types.Operator):
    """ Remove old eyebrows """
    bl_idname = "dazoptim.remove_old_eyebrows"
    bl_label = "Remove old eyebrows"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 'r'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazOptimizeEyebrows_operator], [DazRemoveOldEyebrows_operator]) and not bpy.context.scene.get('daz_optim_toon')

    def execute(self, context):
        DazOptimizer().remove_old_eyebrows()
        pass_stage(self)
        return {'FINISHED'}

class DazTransferFACSToEyebrow_operator(bpy.types.Operator):
    """ Optimize eyes """
    bl_idname = "dazoptim.transfer_facs_to_eyebrows"
    bl_label = "Transfer FACS to eyebrows"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 's'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazMaleLoad_operator], [DazTransferFACSToEyebrow_operator])

    def execute(self, context):
        DazOptimizer().transfer_morphs_to_eyebrows()
        pass_stage(self)
        return {'FINISHED'}

class DazSimplifyEyesMaterial_operator(bpy.types.Operator):
    """ Simplify eyes material """
    bl_idname = "dazoptim.simpl_eyes_mat"
    bl_label = "Simplify eyes material"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 't'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazOptimizeEyes_operator], [DazSimplifyEyesMaterial_operator])

    def execute(self, context):
        DazOptimizer().simplify_eyes_material()
        pass_stage(self)
        return {'FINISHED'}

class DazSeparateIrisUVs_operator(bpy.types.Operator):
    """ Separate iris UVs """
    bl_idname = "dazoptim.sep_iris_uvs"
    bl_label = "Separate iris UVs"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 'u'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazSimplifyEyesMaterial_operator], [DazSeparateIrisUVs_operator])


    def execute(self, context):
            DazOptimizer().separate_iris_uvs()
            pass_stage(self)
            return {'FINISHED'}

class DazMergeGrografts_operator(bpy.types.Operator):
    """ Merge geografts """
    bl_idname = "dazoptim.merge_geografts"
    bl_label = "Merge geografts"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 'v'

    @classmethod
    def poll(cls, context):
        if UNLOCK:
            return True
        required = [DazSimplifyMaterials_operator]
        # if has_gp():
        #     required.append(DazSimplifyGoldenPalaceMaterials_operator)
        return check_stage(context, [DazSimplifyMaterials_operator], [DazMergeGrografts_operator])

    def execute(self, context):
        DazOptimizer().merge_geografts()
        pass_stage(self)
        return {'FINISHED'}


class DazMergeEyes_operator(bpy.types.Operator):
    """ Merge Eyes """
    bl_idname = "dazoptim.merge_eyes"
    bl_label = "Merge eyes"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 'w'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazSimplifyEyesMaterial_operator], [DazMergeEyes_operator, DazMergeEyelashesAndBody_operator])


    def execute(self, context):
            DazOptimizer().merge_eyes()
            pass_stage(self)
            return {'FINISHED'}


class DazMergeMouth_operator(bpy.types.Operator):
    """ Merge Mouth """
    bl_idname = "dazoptim.merge_mouth"
    bl_label = "Merge mouth"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 'x'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazSimplifyMaterials_operator], [DazMergeMouth_operator])

    def execute(self, context):
        DazOptimizer().merge_mouth()
        pass_stage(self)
        return {'FINISHED'}

class DazMergeEyebrowsAndEyelashes_operator(bpy.types.Operator):
    """ Merge eyebrows and eyelashes """
    bl_idname = "dazoptim.merge_eyebrows_and_eyelashes"
    bl_label = "Merge eyebrows and eyelashes"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 'y'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazOptimizeEyelashes_operator], [DazMergeEyebrowsAndEyelashes_operator, DazMergeEyelashesAndBody_operator])

    def execute(self, context):
        DazOptimizer().merge_eyebrows_and_eyelashes()
        pass_stage(self)
        return {'FINISHED'}

class DazRemoveTear_operator(bpy.types.Operator):
    """ Merge Eyes """
    bl_idname = "dazoptim.remove_tear"
    bl_label = "Remove tear"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 'z'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazMaleLoad_operator], [DazRemoveTear_operator])

    def execute(self, context):
        # body = DazOptimizer().get_body_mesh()
        if bpy.context.scene.get('daz_optim_toon'):
            tear = find_object_containing(' Eyelash Base')
        else:
            tear = find_object_containing(' Tear')
        if tear is not None:
            bpy.data.objects.remove(tear)
        pass_stage(self)
        return {'FINISHED'}

class DazMergeEyelashesAndBody_operator(bpy.types.Operator):
    """ Merge eyelashes and body """
    bl_idname = "dazoptim.merge_eyelashes_and_body"
    bl_label = "Merge eyelashes and body"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = '0'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazSimplifyMaterials_operator], [DazMergeEyelashesAndBody_operator]) and not bpy.context.scene.get('daz_optim_toon')

    def execute(self, context):
        DazOptimizer().merge_eyelashes_and_body(join_uvs=True)
        pass_stage(self)
        return {'FINISHED'}

class DazMergeToonEyelashesAndBody_operator(bpy.types.Operator):
    """ Merge eyelashes and body """
    bl_idname = "dazoptim.merge_toon_eyelashes_and_body"
    bl_label = "Merge toon eyelashes and body"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = '0'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazMergeEyebrowsAndEyelashes_operator], [DazMergeToonEyelashesAndBody_operator]) and bpy.context.scene.get('daz_optim_toon')

    def execute(self, context):
        DazOptimizer().merge_eyelashes_and_body(join_uvs=False)
        pass_stage(self)
        return {'FINISHED'}

class DazConcatTextures_operator(bpy.types.Operator):
    """ Concatenate textures into one """
    bl_idname = "dazoptim.concat"
    bl_label = "Concatenate textures"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = '1'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazSimplifyMaterials_operator], [DazConcatTextures_operator])

    def execute(self, context):
        DazOptimizer().concat_textures()
        pass_stage(self)
        return {'FINISHED'}

class DazOptimizeUVs_operator(bpy.types.Operator):
    """ Optimize UVs """
    bl_idname = "dazoptim.optim_uvs"
    bl_label = "Optimize UVs"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = '2'
    @classmethod
    def poll(cls, context):
        return UNLOCK # context.mode == "OBJECT"

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazSimplifyMaterials_operator], [DazOptimizeUVs_operator])

    def execute(self, context):
        DazOptimizer().pack_uvs(use_full_gp=True)
        pass_stage(self)
        return {'FINISHED'}


class DazOptimizeUVsHalfGP_operator(bpy.types.Operator):
    """ Optimize UVs """
    bl_idname = "dazoptim.optim_uvs_half_gp"
    bl_label = "Optimize UVs (half GP)"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = '2'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazSimplifyMaterials_operator, DazOptimizeGoldenPalaceUVs if bpy.context.scene.get('daz_optim_toon') else DazSimplifyGoldenPalaceMaterials_operator] , [DazOptimizeUVsHalfGP_operator])

    def execute(self, context):
        DazOptimizer().pack_uvs(use_full_gp=False)
        pass_stage(self)
        return {'FINISHED'}


class DazMakeSingleMaterial_operator(bpy.types.Operator):
    """ Make a single unified skin material """
    bl_idname = "dazoptim.single_material"
    bl_label = "Make a single unified skin material"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = '3'

    @classmethod
    def poll(cls, context):
        if UNLOCK:
            return True
        if not check_stage(context, [DazOptimizeUVs_operator], [DazMakeSingleMaterial_operator]):
            return False
        return check_stage(context, [DazConcatTextures_operator], []) or os.path.exists(DazOptimizer().get_concat_image_path("Base Color"))

    def execute(self, context):
        DazOptimizer().make_single_material()
        pass_stage(self)
        return {'FINISHED'}


class DazSeparateLipUVs_operator(bpy.types.Operator):
    """ Separate Lip UVs """
    bl_idname = "dazoptim.sep_lip_uvs"
    bl_label = "Separate Lip UVs"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = '4'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazOptimizeUVs_operator], [DazSeparateLipUVs_operator])

    def execute(self, context):
        DazOptimizer().separate_lip_uvs()
        pass_stage(self)
        return {'FINISHED'}


class DazAddBreastBones_operator(bpy.types.Operator):
    """ Add breast bones """
    bl_idname = "dazoptim.breast_bones"
    bl_label = "Subdivide breast bones"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = '5'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazMergeAllRigs_operator], [DazAddBreastBones_operator])

    def execute(self, context):
        DazOptimizer().subdivide_breast_bones()
        pass_stage(self)
        return {'FINISHED'}


class DazAddAdditionalBones_operator(bpy.types.Operator):
    """ Add additional bones """
    bl_idname = "dazoptim.add_additional_bones"
    bl_label = "Add additional bones"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = '^'

    @classmethod
    def poll(cls, context):
        if UNLOCK:
            return True
        if bpy.context.scene.additional_bones_file == '':
            return False
        return  check_stage(context, [DazMergeAllRigs_operator], [DazScaleToUnreal, DazScaleToQuinn, DuplicateSkeleton, DazDetachHairFromSkeleton_operator])

    def execute(self, context):
        bone_names = bpy.context.scene.additional_bones_file
        DazOptimizer().apply_additional_bone(bone_names)
        pass_stage(self)
        already_applied = bpy.context.scene.get('applied_additional_bones', '')
        if len(already_applied)>0:
            already_applied = already_applied+':'
        bpy.context.scene['applied_additional_bones'] = already_applied + bone_names
        return {'FINISHED'}


class DazFitSkinTightClothes_operator(bpy.types.Operator):
    """ fit skin tight clothes """
    bl_idname = "dazoptim.fit_skin_tight_clothes"
    bl_label = "Fit skin tight clothes"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = '8'

    @classmethod
    def poll(cls, context):
        if UNLOCK:
            return True
        required = [DazMergeGrografts_operator]
        if check_stage(context, [DazFitClothes_operator], []):
            required.append(DazApplyFitClothes_operator)
        return check_stage(context, required, [DazFitSkinTightClothes_operator])

    def execute(self, context):
        DazOptimizer().fit_skin_tight_clothes()
        pass_stage(self)
        return {'FINISHED'}


class DazFitClothes_operator(bpy.types.Operator):
    """ fit clothes """
    bl_idname = "dazoptim.fit_clothes"
    bl_label = "Fit clothes"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 'b'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazMergeGrografts_operator], [DazFitClothes_operator, TransferMorphsToClothes])

    def execute(self, context):
        DazOptimizer().fit_clothes()
        pass_stage(self)
        return {'FINISHED'}


class DazBindFitClothes_operator(bpy.types.Operator):
    """ Bind fit clothes """
    bl_idname = "dazoptim.bind_fit_clothes"
    bl_label = "Bind fit clothes"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 'W'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazFitClothes_operator], [DazBindFitClothes_operator])

    def execute(self, context):
        DazOptimizer().bind_clothes_to_extrude()
        pass_stage(self)
        return {'FINISHED'}


class DazApplyFitClothes_operator(bpy.types.Operator):
    """ Apply fit clothes """
    bl_idname = "dazoptim.apply_fit_clothes"
    bl_label = "Apply fit clothes"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 'Y'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazBindFitClothes_operator], [DazApplyFitClothes_operator])

    def execute(self, context):
        DazOptimizer().apply_fit_clothes()
        pass_stage(self)
        return {'FINISHED'}


class DazFitPanties_operator(bpy.types.Operator):
    """ fit panties """
    bl_idname = "dazoptim.fit_panites"
    bl_label = "Fit panties"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 'V'

    @classmethod
    def poll(cls, context):
        if UNLOCK:
            return True
        required = [DazMergeGrografts_operator]
        if check_stage(context, [DazFitClothes_operator], []):
            required.append(DazApplyFitClothes_operator)
        return check_stage(context, required, [DazFitPanties_operator])

    def execute(self, context):
        DazOptimizer().fit_panties()
        pass_stage(self)
        return {'FINISHED'}

BONE_ADDING_OPS = [
            DazAddAdditionalBones_operator, DazAddBreastBones_operator
]
class DazTransferMissingBonesToClothes_operator(bpy.types.Operator):
    """ transfer new bones to clothes """
    bl_idname = "dazoptim.transfer_new_bones_to_clothes"
    bl_label = "transfer new bones to clothes"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = '9'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage_any(context, BONE_ADDING_OPS, [DazTransferMissingBonesToClothes_operator])

    def execute(self, context):
        DazOptimizer().transfer_missing_bones_to_clothes()
        pass_stage(self)
        return {'FINISHED'}

class DazTransferMissingBonesToCum_operator(bpy.types.Operator):
    """ transfer new bones to cum """
    bl_idname = "dazoptim.transfer_new_bones_to_cum"
    bl_label = "transfer new bones to cum"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = '+'

    @classmethod
    def poll(cls, context):
        return UNLOCK or bpy.context.scene.get('has_love_loads') and check_stage_any(context, BONE_ADDING_OPS, [DazTransferMissingBonesToCum_operator])

    def execute(self, context):
        DazOptimizer().transfer_missing_bones_to_cum()
        pass_stage(self)
        return {'FINISHED'}

class DazApplyFitSkinTightClothes_operator(bpy.types.Operator):
    """ fit skin tight clothes """
    bl_idname = "dazoptim.apply_fit_skin_tight_clothes"
    bl_label = "Apply fit skin tight clothes"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 'A'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazFitSkinTightClothes_operator], [DazApplyFitSkinTightClothes_operator])

    def execute(self, context):
        DazOptimizer().apply_fit_skin_tight_clothes()
        pass_stage(self)
        return {'FINISHED'}

class DazOptimizeHair_operator(bpy.types.Operator):
    """ Optimize hair """
    bl_idname = "dazoptim.optim_hair"
    bl_label = "Optimize Hair"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 'B'
    @classmethod
    def poll(cls, context):
        return False

    def execute(self, context):

        return {'FINISHED'}


class DazDetachHairFromSkeleton_operator(bpy.types.Operator):
    """ Detach hair from skeleton """
    bl_idname = "dazoptim.detach_hair_from_skeleton"
    bl_label = "Detach hair from skeleton"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 'C'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazMergeAllRigs_operator], [DazDetachHairFromSkeleton_operator])

    def execute(self, context):
        DazOptimizer().detach_hair_from_skeleton()
        pass_stage(self)
        return {'FINISHED'}


class DazCompareToUe5Skeleton_operator(bpy.types.Operator):
    """ Compare rig to UE5-compatible skeleton """
    bl_idname = "dazoptim.compare_ue5"
    bl_label = "Compare rig to UE5-compatible skeleton"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = None
    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazMaleLoad_operator], [])

    def execute(self, context):
        DazOptimizer().compare_daz_to_ue5_skeleton()
        return {'FINISHED'}

class DuplicateSkeleton(bpy.types.Operator):
    """ Duplicate skeleton """
    bl_idname = "dazoptim.duplicate_skeleton"
    bl_label = "Duplicate skeleton"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = '&'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazMergeAllRigs_operator], [DuplicateSkeleton])

    def execute(self, context):
        DazOptimizer().duplicate_skeleton()
        pass_stage(self)
        return {'FINISHED'}


class AttachDuplicateSkeleton(bpy.types.Operator):
    """ Attach Duplicate skeleton """
    bl_idname = "dazoptim.attach_duplicate_skeleton"
    bl_label = "Attach duplicate skeleton"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DuplicateSkeleton], []) and DazOptimizer.is_duplicate_skeleton_attached()==False

    def execute(self, context):
        DazOptimizer.attach_duplicate_skeleton(True)
        return {'FINISHED'}


class DetachDuplicateSkeleton(bpy.types.Operator):
    """ Detach Duplicate skeleton """
    bl_idname = "dazoptim.detach_duplicate_skeleton"
    bl_label = "Detach duplicate skeleton"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DuplicateSkeleton], []) and DazOptimizer.is_duplicate_skeleton_attached()==True

    def execute(self, context):
        DazOptimizer.attach_duplicate_skeleton(False)
        return {'FINISHED'}


class BakeAction(bpy.types.Operator):
    """ Bake action """
    bl_idname = "dazoptim.bake_action"
    bl_label = "Bake action"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DuplicateSkeleton], []) and DazOptimizer.is_duplicate_skeleton_attached()==True

    def execute(self, context):
        DazOptimizer.bake_animation_to_root()
        return {'FINISHED'}


class ExportAction(bpy.types.Operator):
    """ export action """
    bl_idname = "dazoptim.export_action"
    bl_label = "Export action"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DuplicateSkeleton], [])

    def execute(self, context):
        DazOptimizer().export_animation_to_fbx()
        return {'FINISHED'}


class DazConvertToUe5Skeleton_operator(bpy.types.Operator):
    """ Convert rig to UE5-compatible skeleton """
    bl_idname = "dazoptim.convert_ue5"
    bl_label = "Convert rig to UE5-compatible skeleton"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 'E'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DuplicateSkeleton], [DazConvertToUe5Skeleton_operator, DazDetachHairFromSkeleton_operator])

    def execute(self, context):
        DazOptimizer().convert_daz_to_ue5_skeleton()
        pass_stage(self)
        return {'FINISHED'}



class DazReorientBones_operator(bpy.types.Operator):
    """ Reorient bones """
    bl_idname = "dazoptim.reorient_bones"
    bl_label = "Reorient bones"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 'F'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazConvertToUe5Skeleton_operator], [DazReorientBones_operator, AddUe5IkBones, DazDetachHairFromSkeleton_operator])

    def execute(self, context):
        DazOptimizer().reorient_bones()
        pass_stage(self)
        return {'FINISHED'}

# class DazReweightPelvis_operator(bpy.types.Operator):
#     """ Reweight pelvis """
#     bl_idname = "dazoptim.reweight_pelvis"
#     bl_label = "Reweight pelvis"
#     bl_options = {"REGISTER", "UNDO"}
#     stage_id = ']'
#
#     @classmethod
#     def poll(cls, context):
#         return UNLOCK or check_stage(context, [DazConvertToUe5Skeleton_operator], [DazReweightPelvis_operator])
#
#     def execute(self, context):
#         DazOptimizer().reweight_pelvis()
#         pass_stage(self)
#         return {'FINISHED'}


class DazOptimizeGoldenPalaceUVs(bpy.types.Operator):
    """ Optimize Golden Palace UVs """
    bl_idname = "dazoptim.optim_gp_uvs"
    bl_label = "Optimize Golden Palace UVs"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 'G'

    @classmethod
    def poll(cls, context):
        return UNLOCK or (bpy.context.scene.get('daz_optim_gp') and check_stage(context, [DazMaleLoad_operator], [DazOptimizeGoldenPalaceUVs]))

    def execute(self, context):
        DazOptimizer().unify_golden_palace_uvs()
        pass_stage(self)
        return {'FINISHED'}


class DazSetupGoldenPalaceForBaking(bpy.types.Operator):
    """ Setup Golden Palace for Baking """
    bl_idname = "dazoptim.setup_gp_bake"
    bl_label = "Setup Golden Palace fro baking"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 'H'

    @classmethod
    def poll(cls, context):
        return UNLOCK or not bpy.context.scene.get('daz_optim_toon') and check_stage(context, [DazOptimizeGoldenPalaceUVs], [DazSetupGoldenPalaceForBaking])

    def execute(self, context):
        DazOptimizer().setup_golden_palace_for_baking()
        pass_stage(self)
        return {'FINISHED'}


class DazBakeGoldenPalaceDiffuse(bpy.types.Operator):
    """ Bake Golden Palace """
    bl_idname = "dazoptim.bake_diffuse"
    bl_label = "Bake Golden Palace"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 'I'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazSelectGoldenPalaceColor_operator], [DazMergeGrografts_operator])

    def execute(self, context):
        bpy.ops.object.bake('INVOKE_DEFAULT', type='DIFFUSE')
        pass_stage(self)
        return {'FINISHED'}


class DazBakeGoldenPalaceNormal(bpy.types.Operator):
    """ Bake Golden Palace """
    bl_idname = "dazoptim.bake_normal"
    bl_label = "Bake Golden Palace"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 'J'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazSelectGoldenPalaceNormals_operator], [DazMergeGrografts_operator])

    def execute(self, context):
        bpy.ops.object.bake('INVOKE_DEFAULT', type='NORMAL')
        pass_stage(self)
        return {'FINISHED'}

class DazBakeGoldenPalaceRoughness(bpy.types.Operator):
    """ Bake Golden Palace """
    bl_idname = "dazoptim.bake_rough"
    bl_label = "Bake Golden Palace"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 'K'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazSelectGoldenPalaceRoughness_operator], [DazMergeGrografts_operator])

    def execute(self, context):
        bpy.ops.object.bake('INVOKE_DEFAULT', type='ROUGHNESS')
        pass_stage(self)
        return {'FINISHED'}

class AddUe5IkBones(bpy.types.Operator):
    """ Add UE5 IK bones """
    bl_idname = "dazoptim.add_ue5_ik_bones"
    bl_label = "Add UE5 IK bones"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 'L'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazConvertToUe5Skeleton_operator], [AddUe5IkBones])

    def execute(self, context):
        DazOptimizer().add_ue5_ik_bones()
        pass_stage(self)
        return {'FINISHED'}


class DazScaleToQuinn(bpy.types.Operator):
    """ Scale to quinn """
    bl_idname = "dazoptim.scale_to_quinn"
    bl_label = "Scale to quinn"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 'M'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazMergeAllRigs_operator], [DazScaleToUnreal, DazScaleToQuinn])

    def execute(self, context):
        DazOptimizer().scale_to_quinn()
        pass_stage(self)
        return {'FINISHED'}

class DazTranslateToQuinn(bpy.types.Operator):
    """ Translate to quinn """
    bl_idname = "dazoptim.translate_to_quinn"
    bl_label = "Translate to quinn"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = '6'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazMergeAllRigs_operator], [DazScaleToUnreal, DazTranslateToQuinn])

    def execute(self, context):
        DazOptimizer().translate_to_quinn()
        pass_stage(self)
        return {'FINISHED'}


class DazAlignPoseQuinn(bpy.types.Operator):
    """ Align pose to quinn """
    bl_idname = "dazoptim.align_pose_to_quinn"
    bl_label = "ALign pose to quinn"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = ';'
    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazConvertToUe5Skeleton_operator], [])

    def execute(self, context):
        DazOptimizer().align_pose_to_ue5()
        pass_stage(self)
        return {'FINISHED'}

class DazApplyPose(bpy.types.Operator):
    """ Apply pose """
    bl_idname = "dazoptim.apply_pose"
    bl_label = "Apply pose"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = ':'
    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazAlignPoseQuinn], [DazApplyPose])

    def execute(self, context):
        DazOptimizer().apply_pose()
        pass_stage(self)
        return {'FINISHED'}


class DazScaleToUnreal(bpy.types.Operator):
    """ Scale to unreal """
    bl_idname = "dazoptim.scale_to_ue5"
    bl_label = "Scale to unreal"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 'P'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [], [DazScaleToUnreal])

    def execute(self, context):
        DazOptimizer().scale_to_ue5_units()
        pass_stage(self)
        return {'FINISHED'}


class SaveMorphs(bpy.types.Operator):
    """ Generates favourite morphs """
    bl_idname = "dazoptim.save_fav_morphs"
    bl_label = "Save favourite morphs"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 'Q'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazSaveBlend_operator], [])

    def execute(self, context):
        DazOptimizer().make_fav_morphs_list()
        pass_stage(self)
        return {'FINISHED'}

class LoadMorphs(bpy.types.Operator):
    """ load morphs """
    bl_idname = "dazoptim.load_morphs"
    bl_label = "Load morphs"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 'R'

    @classmethod
    def poll(cls, context):
        if UNLOCK or check_stage(context, [DazSaveBlend_operator], [DazMergeGrografts_operator]):
            return True
        if bpy.data.filepath != '' and os.path.exists(DazOptimizer().get_fav_morphs_path()):
            return True
        return False

    def execute(self, context):
        DazOptimizer().load_fav_morphs()
        pass_stage(self)
        return {'FINISHED'}

class RebindFavMorphs(bpy.types.Operator):
    """ rebind fav morphs """
    bl_idname = "dazoptim.rebind_fav_morphs"
    bl_label = "Rebind fav morphs"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = ')'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [LoadMorphs], [DazMergeGrografts_operator])

    def execute(self, context):
        DazOptimizer().rebind_loaded_fav_morphs()
        pass_stage(self)
        return {'FINISHED'}

class TransferMorphsToGeografts(bpy.types.Operator):
    """ transfer morphs to geografts """
    bl_idname = "dazoptim.transfer_morphs_to_geografts"
    bl_label = "Transfer morphs to geografts"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 'S'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [LoadMorphs], [DazMergeGrografts_operator])

    def execute(self, context):
        DazOptimizer().transfer_morphs_to_geografts()
        pass_stage(self)
        return {'FINISHED'}


class TransferMorphsToClothes(bpy.types.Operator):
    """ transfer morphs to clothes """
    bl_idname = "dazoptim.transfer_morphs_to_clothes"
    bl_label = "Transfer morphs to clothes"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 'T'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [LoadMorphs], [TransferMorphsToClothes])

    def execute(self, context):
        DazOptimizer().transfer_morphs_to_clothes()
        pass_stage(self)
        return {'FINISHED'}


class TransferMorphsToCum(bpy.types.Operator):
    """ transfer morphs to cum """
    bl_idname = "dazoptim.transfer_morphs_to_cum"
    bl_label = "Transfer morphs to cum"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = "'"

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [LoadMorphs], [TransferMorphsToCum])

    def execute(self, context):
        DazOptimizer().transfer_morphs_to_cum()
        pass_stage(self)
        return {'FINISHED'}


class ApplyCustomRig(bpy.types.Operator):
    """ Apply custom rig """
    bl_idname = "dazoptim.apply_custom_rig"
    bl_label = "Apply custom rig"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 'D'

    @classmethod
    def poll(cls, context):
        if bpy.context.scene.custom_rig == '':
            return False
        return UNLOCK or check_stage(context, [DazMaleLoad_operator], [RigPhysicsHair, DazMergeAllRigs_operator])

    def execute(self, context):
        DazOptimizer().apply_custom_rig()
        pass_stage(self)
        return {'FINISHED'}

class RigPhysicsHair(bpy.types.Operator):
    """ rig physics hair """
    bl_idname = "dazoptim.rig_physics_hair"
    bl_label = "Rig Physics Hair"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = "*"

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazMaleLoad_operator], [RigPhysicsHair, DazMergeAllRigs_operator])

    def execute(self, context):
        DazOptimizer().rig_physics_hair()
        pass_stage(self)
        return {'FINISHED'}

class RemoveShapeKeyDrivers(bpy.types.Operator):
    """ Remove shape key drivers """
    bl_idname = "dazoptim.remove_shape_key_drivers"
    bl_label = "Remove shape key drivers"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 'U'
    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [LoadMorphs], [])

    def execute(self, context):
        bpy.ops.daz.unkey_morphs(morphset="Custom", category="Favorites Tara 9", ftype="Custom/Favorites Tara 9")
        pass_stage(self)
        return {'FINISHED'}

class DazExportBodyFbx(bpy.types.Operator):
    """ export fbx """
    bl_idname = "dazoptim.export_body_fbx"
    bl_label = "Export body fbx"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = None

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazSaveTextures_operator], [])

    def execute(self, context):
        DazOptimizer().export_body_to_fbx()
        return {'FINISHED'}

class DazExportClothesFbx(bpy.types.Operator):
    """ export fbx """
    bl_idname = "dazoptim.export_clothes_fbx"
    bl_label = "Export clothes fbx"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = None

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazSaveTextures_operator], [])

    def execute(self, context):
        DazOptimizer().export_clothes_to_fbx()
        return {'FINISHED'}

class DazExportCumFbx(bpy.types.Operator):
    """ export fbx """
    bl_idname = "dazoptim.export_cum_fbx"
    bl_label = "Export cum fbx"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = None

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazSaveTextures_operator], [])

    def execute(self, context):
        DazOptimizer().export_cum_to_fbx()
        return {'FINISHED'}

class DazExportGraftsFbx(bpy.types.Operator):
    """ export fbx """
    bl_idname = "dazoptim.export_grafts_fbx"
    bl_label = "Export grafts fbx"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = None

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazSaveTextures_operator], [])

    def execute(self, context):
        DazOptimizer().export_grafts_to_fbx()
        return {'FINISHED'}

class DazExportHairFbx(bpy.types.Operator):
    """ export fbx """
    bl_idname = "dazoptim.export_hair_fbx"
    bl_label = "Export hair fbx"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = None

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazSaveTextures_operator], [])

    def execute(self, context):
        DazOptimizer().export_hair_to_fbx()
        return {'FINISHED'}

class SaveCustomRig(bpy.types.Operator):
    """ Save custom rig """
    bl_idname = "dazoptim.save_custom_rig"
    bl_label = "Save custom rig"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = None

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazFemaleLoad_operator], [DazMergeGrografts_operator, DazConvertToUe5Skeleton_operator])

    def execute(self, context):
        DazOptimizer.save_custom_rig()
        return {'FINISHED'}

class ApplyAllTransforms(bpy.types.Operator):
    """ Apply all transforms """
    bl_idname = "dazoptim.apply_all_transforms"
    bl_label = "Apply all transforms to all objects"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = None

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazFemaleLoad_operator], [DazMergeGrografts_operator, DazConvertToUe5Skeleton_operator])

    def execute(self, context):
        DazOptimizer().apply_all_transforms()
        return {'FINISHED'}

class SerializeExtraBones(bpy.types.Operator):
    """ Print morph csv """
    bl_idname = "dazoptim.serialize_extra_bones"
    bl_label = "Serialize extra bones"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = None

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazFemaleLoad_operator], [DazMergeGrografts_operator, DazConvertToUe5Skeleton_operator])

    def execute(self, context):
        DazOptimizer().serialize_extra_bones()
        return {'FINISHED'}


class SerializeExtraClothes(bpy.types.Operator):
    """ Serialize extra clothes """
    bl_idname = "dazoptim.serialize_extra_clothes"
    bl_label = "Serialize extra clothes"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = None

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazFemaleLoad_operator], [])

    def execute(self, context):
        DazOptimizer().serialize_extra_clothes()
        return {'FINISHED'}

class PrintMorphCsv(bpy.types.Operator):
    """ Print morph csv """
    bl_idname = "dazoptim.print_morph_csv"
    bl_label = "Print morpgh csv"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = None

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [LoadMorphs], [])

    def execute(self, context):
        DazOptimizer().print_morphs_csv()

        return {'FINISHED'}

class UnlockEverything(bpy.types.Operator):
    """ unlock all stages """
    bl_idname = "dazoptim.unlock_everything"
    bl_label = "Lock/unlock all stages"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = None

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        global UNLOCK
        UNLOCK = not UNLOCK

        return {'FINISHED'}

class HideAllClothes(bpy.types.Operator):
    """ hide all clothes """
    bl_idname = "dazoptim.hide_all_clothes"
    bl_label = "hide all clothes"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = None

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        for c in ClothesStore.get_store().find_all_clothes():
            c.obj.hide_set(True)
        return {'FINISHED'}

class HideAllHair(bpy.types.Operator):
    """ hide all hair """
    bl_idname = "dazoptim.hide_all_hair"
    bl_label = "hide all hair"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = None

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        import assets
        for c in assets.find_all_hair():
            c.hide_set(True)
            rig = get_rig_of(c)
            if rig is not None:
                rig.hide_set(True)
        return {'FINISHED'}

class ShowAllHair(bpy.types.Operator):
    """ show all hair """
    bl_idname = "dazoptim.show_all_hair"
    bl_label = "show all hair"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = None

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        for c in find_all_hair():
            c.hide_set(False)
            rig = get_rig_of(c)
            if rig is not None:
                rig.hide_set(False)
        return {'FINISHED'}

class HideAllRigs(bpy.types.Operator):
    """ hide all rigs """
    bl_idname = "dazoptim.hide_all_rigs"
    bl_label = "hide all rigs"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = None

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        for c in bpy.data.objects:
            if isinstance(c.data, bpy.types.Armature):
                c.hide_set(True)
        return {'FINISHED'}

class HideAllCum(bpy.types.Operator):
    """ hide all cum """
    bl_idname = "dazoptim.hide_all_cum"
    bl_label = "hide all cum"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = None

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        for c in find_cum():
            c.hide_set(True)
        return {'FINISHED'}

class ShowAllCum(bpy.types.Operator):
    """ show all cum """
    bl_idname = "dazoptim.show_all_cum"
    bl_label = "show all cum"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = None

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        for c in find_cum():
            c.hide_set(False)
        return {'FINISHED'}

class RemoveDazBoneConstraints(bpy.types.Operator):
    """ remove daz bone constraints """
    bl_idname = "dazoptim.remove_daz_bone_constraints"
    bl_label = "remove daz bone constraints"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = None

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        DazOptimizer.remove_daz_bone_constraints()
        return {'FINISHED'}


class RemoveDazBoneDrivers(bpy.types.Operator):
    """ remove daz bone drivers """
    bl_idname = "dazoptim.remove_daz_bone_drivers"
    bl_label = "remove daz bone drivers"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = None

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        DazOptimizer.remove_daz_bone_drivers()
        return {'FINISHED'}


class ShowAllClothes(bpy.types.Operator):
    """ show all clothes """
    bl_idname = "dazoptim.show_all_clothes"
    bl_label = "show all clothes"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = None

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        for c in ClothesStore.get_store().find_all_clothes():
            c.obj.hide_set(False)
        return {'FINISHED'}


class EntryLabel:
    def __init__(self, s: str, idx: int):
        self.s = s
        self.idx = idx

    def on_register(self):
        pass

    def on_unregister(self):
        pass

    def draw(self, p, col, context, idx: int):
        p.layout.label(text=self.s)
        col = p.layout.column(align=True)
        return self.idx, col


class EntryOp:
    def __init__(self, op_class: type, op_text: str):
        self.op_text = op_text
        self.op_class = op_class

    def on_register(self):
        pass

    def on_unregister(self):
        pass

    def draw(self, p, col, context, idx: int):
        if idx >= 0:
            text = str(idx) + ". " + self.op_text
        else:
            text = self.op_text
        prop = col.operator(self.op_class.bl_idname, text=text)
        if idx >= 0:
            self.op_class.idx = idx
            return idx+1, col
        else:
            return idx, col


class EntryProp:
    def __init__(self, name: str, prop_type: type, default_value):
        self.name = name
        self.prop_type = prop_type
        self.default_value = default_value

    def on_register(self):
        if self.prop_type is float:
            cons = bpy.props.FloatProperty
        elif self.prop_type is str:
            cons = bpy.props.StringProperty
        elif self.prop_type is bool:
            cons = bpy.props.BoolProperty
        elif self.prop_type is list:
            setattr(bpy.types.Scene, self.name, bpy.props.EnumProperty(
                name=self.name,
                items=self.default_value,
            ))
            return
        else:
            cons = bpy.props.StringProperty
        setattr(bpy.types.Scene, self.name, cons(name=self.name,default=self.default_value))

    def on_unregister(self):
        delattr(bpy.types.Scene, self.name)

    def draw(self, p, col, context, idx: int):
        p.props[self.name] = col.prop(context.scene, self.name)
        return idx, col


class EntryFileEnumProp:
    def __init__(self, name: str, dir_name: str, extension = ".json"):
        self.name = name
        self.extension = extension
        self.dir_name = dir_name

    def items_provider(self):
        return [(i, i, i) for i in collect_resource_files(self.dir_name, self.extension)]

    def on_register(self):
        self.prop = bpy.props.EnumProperty(
            name=self.name,
            items=lambda x, y:self.items_provider(),
        )
        setattr(bpy.types.Scene, self.name, self.prop)

    def on_unregister(self):
        delattr(bpy.types.Scene, self.name)

    def draw(self, p, col, context, idx: int):
        p.props[self.name] = col.prop(context.scene, self.name)
        return idx, col


class EntryAdditionalBonesEnumProp(EntryFileEnumProp):
    def __init__(self, name: str):
        super().__init__(name, "additional_bones")

    @override
    def items_provider(self):
        s: str = bpy.context.scene.get('applied_additional_bones', '')
        applied:{str} = set(s.split(':'))
        return [(i, i, i) for i in collect_resource_files(self.dir_name, self.extension) if i not in applied]


class EntryCustomRigsEnumProp(EntryFileEnumProp):
    def __init__(self, name: str):
        super().__init__(name, "rigs")

    @override
    def items_provider(self):
        s: str = bpy.context.scene.get('applied_custom_rigs', '')
        fps = find_all_fingerprints()
        applied:{str} = set(s.split(':'))
        return [(i, fps[i].name, i) for i in collect_resource_files(self.dir_name, self.extension) if i not in applied and i in fps]