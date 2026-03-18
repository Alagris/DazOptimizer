import base64
import json
import os
import re
import urllib
from . import util
from . import constants
import bpy
import numpy as np


def get_resource_path(file_name, dir_name):
    import os
    n = file_name
    p = bpy.path.abspath('//' + dir_name + '/' + n)
    if os.path.exists(p):
        return p
    p = bpy.path.abspath('//../' + dir_name + '/' + n)
    if os.path.exists(p):
        return p

    from bpy.utils import resource_path
    from pathlib import Path
    USER = Path(resource_path('USER'))
    ADDON = "DazOptim"
    srcPath = USER / "scripts/addons" / ADDON / dir_name / n
    return str(srcPath)


def collect_resource_paths(dir_name, extension):
    from bpy.utils import resource_path
    from pathlib import Path
    USER = Path(resource_path('USER'))
    ADDON = "DazOptim"
    dirs = [bpy.path.abspath('//' + dir_name + '/'),
            bpy.path.abspath('//../' + dir_name + '/'),
            USER / "scripts/addons" / ADDON / dir_name
            ]
    files = set()
    for p in dirs:
        if os.path.exists(p):
            for file in os.listdir(p):
                if file.endswith(extension):
                    files.add(os.path.join(p, file))
    return files


def collect_resource_files(dir_name, extension):
    return [os.path.basename(file)[:-len(extension)] for file in collect_resource_paths(dir_name, extension)]


def get_eyebrows_and_eyelashes_path():
    return get_resource_path(bpy.context.scene.eyebrows_file + '.png', 'eyebrows')


class AdditionalBone:

    def __init__(self, j):
        self.bone_name = j['name']
        self.head = j["head"]
        self.tail = j["tail"]
        self.parent = j["parent"]
        self.connect = j["connect"]
        self.local = j["local"]
        weights = j["weights"]
        weights = base64.decodebytes(weights)
        weights = np.frombuffer(weights, dtype=np.float64)
        self.weights = weights
        self.indices = j["indices"]

    def apply_additional_bone(self, obj):
        rig = util.get_rig_of(obj)
        util.select_object(rig)
        bpy.ops.object.mode_set(mode='EDIT')

        bone = rig.data.edit_bones.get(self.bone_name)
        if bone is None:
            bone = rig.data.edit_bones.new(self.bone_name)

        bone.head = self.head
        bone.tail = self.tail
        parent_bone = rig.data.edit_bones[self.parent]
        bone.parent = parent_bone
        bone.use_connect = self.connect
        bone.use_local_location = self.local
        vg = obj.vertex_groups.get(self.bone_name)
        if vg is None:
            vg = obj.vertex_groups.new(name=self.bone_name)

        is_non_zero = util.rle_decode(self.indices, (constants.NUM_OF_VERTICES_IN_DAZ_BASE_MESH,))
        indices, = np.where(is_non_zero)
        for val, idx in zip(self.weights.tolist(), indices.tolist()):
            vg.add(index=(idx,), weight=val, type='REPLACE')

    @staticmethod
    def serialize_bone_and_weights(obj, bone_names):
        from . import util
        rig = util.get_rig_of(obj)
        util.select_object(rig)
        bpy.ops.object.mode_set(mode='EDIT')
        bones = {}
        for bone_name in bone_names:
            bone = rig.data.edit_bones[bone_name]
            parent_bone = None if bone.parent is None else bone.parent.name
            bone_head = list(bone.head)
            bone_tail = list(bone.tail)
            bones[bone_name] = {
                'name': bone_name,
                'head': bone_head,
                'tail': bone_tail,
                'parent': parent_bone,
                'connect': bone.use_connect,
                'local': bone.use_local_location
            }
        results = {}
        util.select_object(obj)
        for bone_name in bone_names:
            vg = obj.vertex_groups.get(bone_name)
            if vg is not None:
                bone = bones[bone_name]
                weights, indices, is_non_zero = util.get_weights_as_sparse(obj, vg)
                bone['weights'] = weights
                bone['indices'] = indices
                results[bone_name] = AdditionalBone(bone)
        return results

    def to_json(self):
        import base64
        return {
            'head': self.head,
            'tail': self.tail,
            'parent': self.parent,
            'connect': self.connect,
            'local': self.local,
            'weights': base64.b64encode(self.weights).decode("utf-8"),
            'indices': self.indices.tolist()
        }


class AdditionalBones:
    APPLIED_BONES = set()
    def __init__(self, bones):
        if isinstance(bones, str):
            import json
            file_path = AdditionalBones.get_additional_bones_path(bones)
            with open(file_path, 'r') as f:
                bones = json.load(f)
        self.object = bones['object']
        bones = [AdditionalBone(b) for b in bones['bones']]
        self.bones = {b.bone_name: b for b in bones}

    @staticmethod
    def get_additional_bones_path(file_name):
        return get_resource_path(file_name + '.json', 'additional_bones')

    @staticmethod
    def collect_additional_bones_files():
        return collect_resource_files('additional_bones', '.json')

    @staticmethod
    def load_additional_bones():
        import json
        ab = collect_resource_paths('additional_bones', '.json')
        abs = {}
        for f in ab:
            file_name = os.path.basename(f)[:-len(".json")]
            with open(f, 'r') as f:
                abs[file_name] = AdditionalBone(json.load(f))
        return abs

    def apply(self):
        from . import util
        if self.object == 'g9':
            obj = util.find_body_mesh()
        else:
            obj = util.find_by_fingerprint(self.object)
        if obj is not None:
            for b in self.bones.values():
                if self.object == 'g9':
                    AdditionalBones.APPLIED_BONES.add(b.bone_name)
                b.apply_additional_bone(obj)

    @staticmethod
    def is_additional_bone(bone_name):
        return bone_name in AdditionalBones.APPLIED_BONES

class MaskStore:
    SINGLETON = None

    @staticmethod
    def get_store():
        if MaskStore.SINGLETON is None:
            MaskStore.SINGLETON = MaskStore()
        return MaskStore.SINGLETON

    @staticmethod
    def get_masks_path(file_name):
        return get_resource_path(file_name + '.npy', 'masks')

    @staticmethod
    def load_mask(file_name):
        return np.load(MaskStore.get_masks_path(file_name))

    def __init__(self):
        self.cache = {}

    def __getitem__(self, item):
        c = self.cache.get(item)
        if c is None:
            c = MaskStore.load_mask(item)
            self.cache[item] = c
        return c

    def head_rle(self):
        return self['head_rle']

    def body_rle(self):
        return self['body_rle']

    def left_leg_rle(self):
        return self['left_leg_rle']

    def right_leg_rle(self):
        return self['right_leg_rle']

    def butt_rle(self):
        return self['butt_rle']

    def bot_arm_rle(self):
        return self['bot_arm_rle']

    def lip_rle(self):
        return self['lip_rle']

    def top_arm_rle(self):
        return self['top_arm_rle']

    def eyelashes_rle(self):
        return self['eyelashes_rle']

    def eye_socket_rle(self):
        return self['eye_socket_rle']

    def mouth_cavity_rle(self):
        return self['mouth_cavity_rle']

    def toon_mouth_rle(self):
        return self['toon_mouth_rle']


class ClothesMeta:

    def __init__(self, item):
        self.fingerprint = item['fingerprint']
        self.skin_tight = item.get('skin_tight', -1)
        self.skin_tight = constants.CLOTHES_CONSTANTS.get(self.skin_tight, self.skin_tight)
        self.panties = item.get('panties', False)
        self.obj = None
        self.dont_resize = item.get('dont_resize', False)

    def is_skin_tight(self):
        return not self.dont_resize and self.skin_tight >= 0 and not self.panties

    def is_not_skin_tight(self):
        return not self.dont_resize and self.skin_tight < 0

    def is_panties(self):
        return not self.dont_resize and self.panties


class ClothesStore:
    SINGLETON = None

    @staticmethod
    def get_store():
        if ClothesStore.SINGLETON is None:
            ClothesStore.SINGLETON = ClothesStore()
        return ClothesStore.SINGLETON

    def __init__(self):
        import json
        clothes_path = ClothesStore.get_clothes_path('all')
        with open(clothes_path, 'r') as f:
            clothes = json.load(f)
        self.metas = {k: ClothesMeta(v) for k, v in clothes.items()}

    def get(self, name):
        return self.metas.get(name)

    def items(self):
        return self.metas.items()

    def __getitem__(self, item):
        return self.metas[item]

    @staticmethod
    def get_clothes_path(file_name):
        return get_resource_path(file_name + '.json', 'clothes')

    def is_clothes(self, obj) -> ClothesMeta:
        name = obj.name
        if name.endswith(" Mesh"):
            name = name[:-len(" Mesh")]
        return self.get(name)

    def find_all_clothes(self, predicate=None):
        clothes = []
        for obj in bpy.data.objects:
            if isinstance(obj.data, bpy.types.Mesh):
                meta = self.is_clothes(obj)
                if meta is not None and (predicate is None or predicate(meta)):
                    meta.obj = obj
                    clothes.append(meta)
        return clothes

    def find_all_panties(self):
        return self.find_all_clothes(ClothesMeta.is_panties)

    def find_all_skin_tight_clothes(self):
        return self.find_all_clothes(ClothesMeta.is_skin_tight)

    def find_all_non_skin_tight_clothes(self):
        return self.find_all_clothes(ClothesMeta.is_not_skin_tight)


class BoneRelation:
    def __init__(self, j):
        self.start = j['start']
        self.tail = j['tail']
        self.x_axis = j['x_axis']
        self.y_axis = j['y_axis']
        self.z_axis = j['z_axis']
        self.roll = j['roll']
        self.parent_name = j.get('parent_name')


class BoneHierarchy:
    MANNY = None
    QUINN = None
    SINGLETON = None

    @staticmethod
    def get_hierarchy():
        if BoneHierarchy.SINGLETON is None:
            if constants.is_female():
                return BoneHierarchy.get_quinn()
            else:
                return BoneHierarchy.get_manny()
        return BoneHierarchy.SINGLETON

    @staticmethod
    def get_manny():
        if BoneHierarchy.MANNY is None:
            BoneHierarchy.SINGLETON = BoneHierarchy.MANNY = BoneHierarchy.from_json('manny')
        return BoneHierarchy.MANNY

    @staticmethod
    def get_quinn():
        if BoneHierarchy.QUINN is None:
            BoneHierarchy.SINGLETON = BoneHierarchy.QUINN = BoneHierarchy.from_json('quinn')
        return BoneHierarchy.QUINN

    def __init__(self, bones: {str: BoneRelation}):
        self.bones = bones

    @staticmethod
    def from_json(bones: str):
        p = get_resource_path(bones + '.json', 'boneh')
        with open(p, 'r') as f:
            j = json.load(f)
            return BoneHierarchy({k: BoneRelation(v) for k, v in j.items()})

    @staticmethod
    def from_armature(armature='root'):
        if isinstance(armature, str):
            armature = bpy.data.armatures[armature]
        return BoneHierarchy({b.name: BoneRelation({
            'head': list(b.head),
            'tail': list(b.tail),
            'x_axis': list(b.x_axis),
            'y_axis': list(b.y_axis),
            'z_axis': list(b.z_axis),
            'roll': b.roll,
            'parent_name': b.parent.name if b.parent is not None else None
        }) for b in armature.edit_bones})

    def __contains__(self, item):
        return item in self.bones

    def __getitem__(self, item):
        return self.bones[item]


def is_known_bone(bone_name):
    return AdditionalBones.is_additional_bone(bone_name) or constants.is_daz_bone(bone_name)


def is_hair(obj):
    l = obj.name.lower()
    return 'hair' in l or 'ponytail' in l


def find_all_hair():
    return [obj for obj in bpy.data.objects if isinstance(obj.data, bpy.types.Mesh) and is_hair(obj)]


def is_cum(o):
    return o.name.startswith("Love Loads")


def find_cum():
    return [o for o in bpy.data.objects if isinstance(o.data, bpy.types.Mesh) and is_cum(o)]


class Morph:
    def __init__(self, shape_key):
        self.category = None
        self.title = None
        self.figure = ''
        self.profile = 9999
        self.is_male = False
        self.is_female = False
        self.shape_key = shape_key

    def check_gender(self, is_female: bool):
        return self.is_female if is_female else self.is_male

    def check_figure(self, figure: str):
        return figure in self.figure

    def check_category(self, cat: {str}):
        return cat is None or self.category in cat

    def check_profile(self, prof: int):
        return self.profile <= prof

    def check(self, is_female: bool, figure: str, cat: {str}, prof: int):
        return self.check_gender(is_female) and self.check_figure(figure) and self.check_category(
            cat) and self.check_profile(prof)


class MorphsStore:
    SINGLETON = None

    @staticmethod
    def get_store():
        if MorphsStore.SINGLETON is None:
            MorphsStore.SINGLETON = MorphsStore()
        return MorphsStore.SINGLETON

    def __init__(self):
        self.GENERATE_MORPHS_FOR_CLOTHES = True
        self.GENERATE_MORPHS_FOR_HAIR = True
        self.FIGURES = {
            'TOON': "T",
            'ANY': "TG",
            'G9': "G",
        }
        self.PROFILES = {
            'FULL': 10,
            'MID': 5,
            'MIN': 0,
        }
        self.MORPH_CATEGORIES = {
            'BREAST': ("Custom/Breasts", "Custom"),
            'HEAD': ("Custom/Head", "Custom"),
            'ARMS': ("Custom/Arms", "Custom"),
            'LEGS': ("Custom/Legs", "Custom"),
            'BODY': ("Custom/Body", "Custom"),
            'ASS': ("Custom/Ass", "Custom"),
            'GENITALS': ("Custom/Genitals", "Custom"),
            'SPECIAL': ("Custom/Special", "Custom"),
            'FACS': ("Facs", 'Face'),
            'FACSEXPR': ("Facsexpr", 'Face'),
            'FACSDET': ("Facsdetails", 'Face'),
            'JCM': ("JCM", 'JCM'),
        }
        self.CAT_SETS = {
            'BODY': {'BREAST', 'HEAD', 'ARMS', 'LEGS', 'BODY', 'ASS', 'GENITALS', 'SPECIAL'},
            'FACS': {'FACS', 'FACSDET', 'FACSEXPR'},
            'GENITALS': {'GENITALS'},
            'FACS_GENITALS': {'FACS', 'FACSDET', 'FACSEXPR', 'GENITALS'},
            'FACS_GENITALS_SPECIAL': {'FACS', 'FACSDET', 'FACSEXPR', 'GENITALS', 'SPECIAL'},
            'FACS_SPECIAL': {'FACS', 'FACSDET', 'FACSEXPR', 'SPECIAL'},
            'JCM': {'JCM'},
            'SPECIAL': {'SPECIAL'},
            'ALL': None,
        }

        self.profile = 10
        self.file_name = ''
        self.morphs: {str: {str: Morph}} = {}
        self.GENERATE_MORPHS_FOR_CLOTHES = True
        self.GENERATE_MORPHS_FOR_HAIR = True

    def clear(self):
        self.morphs.clear()

    def load_current(self):
        self.load_file(bpy.context.scene.morphs_file)

    @staticmethod
    def get_morphs_path(file_name):
        return get_resource_path(file_name + '.json', 'morphs')

    def load_file(self, file_name=None):
        def process_gender(morphs, m, is_male=False, is_female=False):
            for morph_name, morph_meta in m.items():
                if morph_name not in morphs:
                    morph = morphs[morph_name] = Morph(morph_name)
                else:
                    morph = morphs[morph_name]
                morph.is_male = morph.is_male or is_male
                morph.is_female = morph.is_female or is_female
                profile_meta = morph_meta.get('profile')
                profile_meta = self.PROFILES.get(profile_meta, 9999)
                morph.profile = min(profile_meta, morph.profile)
                fig_meta = self.FIGURES.get(morph_meta.get('figure'), '')
                morph.figure = ''.join(set(fig_meta + morph.figure))
                morph.title = morph_meta.get('name', morph.title)
                cat = morph_meta['category']
                assert cat in self.MORPH_CATEGORIES
                morph.category = cat

        if file_name is None:
            file_name = bpy.context.scene.morphs_file
        if self.file_name != file_name:
            p = MorphsStore.get_morphs_path(file_name)
            with open(p, 'r') as f:
                j = json.load(f)
            self.morphs = {}
            for daz_path, j in j.items():
                assert isinstance(daz_path, str)
                daz_path = daz_path.lower()
                morphs_for_daz_path = self.morphs[daz_path] = {}
                shapes = j['shapes']
                female = shapes.get('female', {})
                process_gender(morphs_for_daz_path, female, False, True)
                male = shapes.get('male', {})
                process_gender(morphs_for_daz_path, male, True, False)
                unisex = shapes.get('unisex', {})
                process_gender(morphs_for_daz_path, unisex, True, True)
            self.file_name = file_name


    def get_figure(self):
        return self.FIGURES.get('TOON' if bpy.context.scene.get('daz_optim_toon') else 'G9')

    def collect_fav_shape_keys(self, categories_to_include: {str}, profiles_to_include: int) -> {str: Morph}:
        if isinstance(profiles_to_include, str):
            profiles_to_include = self.PROFILES[profiles_to_include]
        is_fem = constants.is_female()
        shape_keys = {}
        figure = self.get_figure()
        for obj in bpy.data.objects:
            if isinstance(obj.data, bpy.types.Mesh):
                if not self.GENERATE_MORPHS_FOR_CLOTHES and ClothesStore.get_store().is_clothes(obj):
                    continue
                if not self.GENERATE_MORPHS_FOR_HAIR and is_hair(obj):
                    continue
                if is_cum(obj):
                    continue
                daz_dir = util.obj_daz_dir(obj).lower()
                morphs_for_daz_obj: {str: Morph} = self.morphs.get(daz_dir)
                if morphs_for_daz_obj is not None:
                    for morph in morphs_for_daz_obj.values():
                        if morph.check(is_fem, figure, categories_to_include, profiles_to_include):
                            shape_keys[morph.shape_key] = morph
        return shape_keys

    def make_fav_morphs_list(self, fav_morphs_path, categories_to_include: {str}, profiles_to_include: int,
                             load_all_conflicting_morphs=True):
        from . import util
        from . import constants
        content_dirs = util.get_daz_content_dirs()
        content_dirs = [d[:-1] if d.endswith("/") or d.endswith("\\") else d for d in content_dirs]
        morph_prefixes = constants.get_allowed_morph_prefixes()
        morph_prefixes_regex = re.compile(r"(" + "|".join(morph_prefixes) + ")?(.+)\.dsf")
        shape_keys = self.collect_fav_shape_keys(categories_to_include, profiles_to_include)
        fav_morphs = {
            "filetype": "favo_morphs",
            "root_paths": content_dirs,
        }
        for obj in bpy.data.objects:
            if not isinstance(obj.data, bpy.types.Mesh):
                continue
            if not self.GENERATE_MORPHS_FOR_CLOTHES and ClothesStore.get_store().is_clothes(obj):
                continue
            if not self.GENERATE_MORPHS_FOR_HAIR and is_hair(obj):
                continue
            if is_cum(obj):
                continue
            daz_dir = util.obj_daz_dir(obj)
            collected_shape_keys = {}
            for contentDir in content_dirs:
                morphs_dir_path = contentDir + daz_dir + "/Morphs"
                if os.path.isdir(morphs_dir_path):
                    for root, dirs, files in os.walk(morphs_dir_path):
                        for file in files:
                            m = morph_prefixes_regex.match(file)
                            if m:
                                prefix = m.group(1)
                                shape_key_name = m.group(2)
                                meta: Morph = shape_keys.get(shape_key_name)
                                if meta is not None:
                                    if load_all_conflicting_morphs and prefix is not None:
                                        shape_key_name = prefix + shape_key_name
                                    collected_shape_key = collected_shape_keys.get(shape_key_name)
                                    try:
                                        priority = morph_prefixes.index(prefix)
                                    except ValueError:
                                        priority = -1
                                    if collected_shape_key is None:
                                        collected_shape_key = collected_shape_keys[shape_key_name] = {}
                                    elif priority > collected_shape_key['priority']:
                                        pass
                                    else:
                                        continue
                                    filepath = os.path.join(root, file)
                                    filepath = os.path.relpath(filepath, contentDir)
                                    filepath = filepath.replace("\\", "/")
                                    collected_shape_key['priority'] = priority
                                    collected_shape_key['filepath'] = filepath
                                    collected_shape_key['meta'] = meta
            if len(collected_shape_keys) > 0:
                morphs_dict = {}
                mesh = obj.data
                mesh_url = urllib.parse.quote(obj.daz_importer.DazUrl)
                fav_morphs[mesh_url] = {
                    "finger_print": mesh.daz_importer.DazFingerPrint,
                    "morphs": morphs_dict
                }
                for shape_key_name, collected_shape_key in collected_shape_keys.items():
                    filepath = collected_shape_key['filepath']
                    filepath = urllib.parse.quote(filepath)
                    if not filepath.startswith("/"):
                        filepath = '/' + filepath
                    meta = collected_shape_key['meta']
                    category_key, category = self.MORPH_CATEGORIES[meta.category]
                    if category_key not in morphs_dict:
                        shapes_list = morphs_dict[category_key] = []
                    else:
                        shapes_list = morphs_dict[category_key]
                    shapes_list.append([filepath, shape_key_name, category])
        with open(fav_morphs_path, 'w+') as f:
            json.dump(fav_morphs, f, indent=2)
