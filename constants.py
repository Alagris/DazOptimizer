import bpy

BOT_ARM_TRANS = [-0.072266 - 0.006897, 0.085937 + 0.009853]
TOP_ARM_TRANS = [0.043945, 0.006836]
RIGHT_LEG_TRANS = -0.014389
BODY_TRANS = [0.0, 0.170266]
LIP_TRANS = [1 / 4 + 1 / 16, -1 / 32]
MOUTH_CAVITY_SCALED_TRANS = [1/8 + 1 / 4 + 1 / 16 + 3/8, - 1/16 - 1/64]
MOUTH_CAVITY_TRANS = [1 / 4 + 1 / 16 + 3/8, -1 / 32 - 1/4 -1/64]
EYE_SOCKET_TRANS = [MOUTH_CAVITY_TRANS[0] - 1/8, MOUTH_CAVITY_TRANS[1]]
LEFT_EYE_SOCKET_TRANS = [EYE_SOCKET_TRANS[0] + 1 / 8, EYE_SOCKET_TRANS[1]]
RIGHT_EYE_SOCKET_TRANS = [EYE_SOCKET_TRANS[0] - 1 / 8, EYE_SOCKET_TRANS[1]]
LEFT_LEG_COLOR = (64 * 256 + 24) * 256 + 126
RIGHT_LEG_COLOR = (42 * 256 + 126) * 256 + 24
BUTT_COLOR = (255 * 256 + 24) * 256 + 255
BOT_ARM_COLOR = (62 * 256 + 21) * 256 + 211
LIP_COLOR = (21 * 256 + 109) * 256 + 211
TOP_ARM_COLOR = (255 * 256 + 0) * 256 + 0
BODY_COLOR = (21 * 256 + 211) * 256 + 91
HEAD_COLOR = (204 * 256 + 162) * 256 + 20

MASK_SHAPE = (4096, 4096)
TOON_MOUTH_MASK_SHAPE = (1024, 1024)

EXTRUDED_SK_NAME = 'extruded'
CLOTHES_MIN_DIST_TO_SKIN = 0.004
PANTIE_SCALING = 0.02
CLOTHES_CONSTANTS = {
    'CLOTHES_MIN_DIST_TO_SKIN': CLOTHES_MIN_DIST_TO_SKIN,
    'PANTIE_SCALING': PANTIE_SCALING,
}

TRANSPARENT_TOON_EYEBROWS_MAT_NAME = "ToonEyebrows"
TRANSPARENT_TOON_EYELASHES_MAT_NAME = "ToonEyelashes"

NIRV_ZERO_EYES_DAZ_DIR = "/data/nirvana/nirv zero/nirv zero eyes"

BREAST_GEOGRAFTS = ['BreastacularG9', 'Body Geo', 'STX Gen 9 Nipples Feminine']
DICK_GEOGRAFTS = ['Genesis 9 Anatomical Elements Male']
MALE_ONLY_GEOGRAFTS = DICK_GEOGRAFTS
FEMALE_ONLY_GEOGRAFTS = ['GoldenPalace_G9', 'Wet Kitty TOON'] + BREAST_GEOGRAFTS
GEOGRAFTS = FEMALE_ONLY_GEOGRAFTS + MALE_ONLY_GEOGRAFTS


def is_graft(obj):
    return obj.name in GEOGRAFTS

def has_dick():
    for dick in DICK_GEOGRAFTS:
        d = bpy.data.objects.get(dick+' Mesh')
        if d is not None:
            return d

def is_female():
    return bool(bpy.context.scene['daz_optim_female'])

def is_daz_bone(bone_name):
    return bone_name in DAZ_G9_TO_UE5_BONES or bone_name in OTHER_DAZ_BONES

def get_allowed_morph_prefixes():
    if bpy.context.scene.get('is_nirv_zero'):
        return ["BaseAnime_", "Nirv_Zero_BaseAnim_", "Nirv_zero_", "Nirv_Zero_", "Nirv_"]
    elif bpy.context.scene.get('daz_optim_toon'):
        return ["BaseAnime_"]
    return []

QUINN_HEIGHT = 1.80169
MANNY_HEIGHT = 1.80625
NEW_WK_UV_MAP = 'WK UVs'
NEW_GP_UV_MAP = 'unified_gp_uv'
NEW_TOON_EYELASHES_UV_MAP = 'Toon Eyelashes UVs'
NEW_EYES_UV_MAP = 'optimised_eyes_uvs'


UE5_IK_BONES = {
    'ik_foot_root': '',
    'ik_foot_l': 'foot_l',
    'ik_foot_r': 'foot_r',
    'ik_hand_root': '',
    'ik_hand_gun': 'hand_r',
    'ik_hand_l': 'hand_l',
    'ik_hand_r': 'hand_r',
    'center_of_mass': ''
}

DAZ_G9_TO_UE5_BONES = {
    'r_toes': 'ball_r',
    'l_toes': 'ball_l',
    'r_foot': 'foot_r',
    'l_foot': 'foot_l',
    'r_shin': 'calf_r',
    'l_shin': 'calf_l',
    'l_thigh': 'thigh_l',
    'r_thigh': 'thigh_r',
    'l_thightwist1': 'thigh_twist_01_l',
    'r_thightwist1': 'thigh_twist_01_r',
    'l_thightwist2': 'thigh_twist_02_l',
    'r_thightwist2': 'thigh_twist_02_r',
    'hip': 'pelvis',
    'pelvis': 'spine_01',
    'spine1': 'spine_02',
    'spine2': 'spine_03',
    'spine3': 'spine_04',
    # 'l_pectoral': 'clavicle_pec_l',
    # 'r_pectoral': 'clavicle_pec_r',
    'spine4': 'spine_05',
    'l_shoulder': 'clavicle_l',
    'r_shoulder': 'clavicle_r',
    'l_upperarm': 'upperarm_l',
    'r_upperarm': 'upperarm_r',
    'r_upperarmtwist1': 'upperarm_twist_01_r',
    'l_upperarmtwist1': 'upperarm_twist_01_l',
    'r_upperarmtwist2': 'upperarm_twist_02_r',
    'l_upperarmtwist2': 'upperarm_twist_02_l',
    'l_forearm': 'lowerarm_l',
    'r_forearm': 'lowerarm_r',
    'r_forearmtwist1':'lowerarm_twist_02_r',
    'l_forearmtwist1':'lowerarm_twist_02_l',
    'r_forearmtwist2':'lowerarm_twist_01_r',
    'l_forearmtwist2':'lowerarm_twist_01_l',
    'l_hand': 'hand_l',
    'r_hand': 'hand_r',
    'l_thumb1': 'thumb_01_l',
    'r_thumb1': 'thumb_01_r',
    'l_thumb2': 'thumb_02_l',
    'r_thumb2': 'thumb_02_r',
    'l_thumb3': 'thumb_03_l',
    'r_thumb3': 'thumb_03_r',
    'l_index1': 'index_01_l',
    'r_index1': 'index_01_r',
    'l_index2': 'index_02_l',
    'r_index2': 'index_02_r',
    'l_index3': 'index_03_l',
    'r_index3': 'index_03_r',
    'l_mid1': 'middle_01_l',
    'r_mid1': 'middle_01_r',
    'l_mid2': 'middle_02_l',
    'r_mid2': 'middle_02_r',
    'l_mid3': 'middle_03_l',
    'r_mid3': 'middle_03_r',
    'l_ring1': 'ring_01_l',
    'r_ring1': 'ring_01_r',
    'l_ring2': 'ring_02_l',
    'r_ring2': 'ring_02_r',
    'l_ring3': 'ring_03_l',
    'r_ring3': 'ring_03_r',
    'l_pinky1': 'pinky_01_l',
    'r_pinky1': 'pinky_01_r',
    'l_pinky2': 'pinky_02_l',
    'r_pinky2': 'pinky_02_r',
    'l_pinky3': 'pinky_03_l',
    'r_pinky3': 'pinky_03_r',
    'neck1': 'neck_01',
    'neck2': 'neck_02',
    'head': 'head',
    'l_indexmetacarpal': 'index_metacarpal_l',
    'l_midmetacarpal': 'middle_metacarpal_l',
    'l_ringmetacarpal': 'ring_metacarpal_l',
    'l_pinkymetacarpal': 'pinky_metacarpal_l',
    'r_indexmetacarpal': 'index_metacarpal_r',
    'r_midmetacarpal': 'middle_metacarpal_r',
    'r_ringmetacarpal': 'ring_metacarpal_r',
    'r_pinkymetacarpal': 'pinky_metacarpal_r',
}
OTHER_DAZ_BONES = {
    'l_bigtoe1',
    'l_bigtoe2',
    'l_indextoe1',
    'l_indextoe2',
    'l_midtoe1',
    'l_midtoe2',
    'l_ringtoe1',
    'l_ringtoe2',
    'l_pinkytoe1',
    'l_pinkytoe2',
    'l_metatarsal',
    'r_bigtoe1',
    'r_bigtoe2',
    'r_indextoe1',
    'r_indextoe2',
    'r_midtoe1',
    'r_midtoe2',
    'r_ringtoe1',
    'r_ringtoe2',
    'r_pinkytoe1',
    'r_pinkytoe2',
    'r_metatarsal',
    'lowerjaw',
    'lowerteeth',
    'l_lipcorner',
    'l_liplower',
    'liplowermiddle',
    'r_liplower',
    'r_lipcorner',
    'l_cheeklower',
    'r_cheeklower',
    'chin',
    'l_browinner',
    'l_browouter',
    'r_browinner',
    'r_browouter',
    'centerbrow',
    'l_eyelidupper',
    'l_eyelidlower',
    'r_eyelidupper',
    'r_eyelidlower',
    'l_squint',
    'r_squint',
    'l_cheek',
    'r_cheek',
    'l_nostril',
    'r_nostril',
    'lipuppermiddle',
    'l_lipupper',
    'r_lipupper',
    'l_infraorbital',
    'r_infraorbital',
    'l_ear',
    'r_ear',
    'l_pectoral',
    'r_pectoral',
}

NUM_OF_VERTICES_IN_DAZ_BASE_MESH = 25182



# euelr rotation order 'YZX'
DAZ_TO_UE5_POSE_ROTATIONS = {
    'upperarm_l': [0.03490658849477768, -0.0, -0.10471975803375244],
    'upperarm_twist_01_l': [0,0,0],
    'upperarm_twist_02_l': [0,0,0],
    'lowerarm_l': [0.3490658402442932, 0.0, -0.03490658476948738],
    'lowerarm_twist_01_l': [0,0,0],
    'lowerarm_twist_02_l': [0,0,0],
    'hand_l': [0.10471975058317184, 0.06981316953897476, -0.0],
    'thumb_01_l': [0.01745329238474369, 0.0872664600610733, 0.1745329350233078],
    'thumb_02_l': [0.20943953096866608, 0.0, -0.0],
    'thumb_03_l': [0.27925270795822144, 0.0, -0.0],
    'index_metacarpal_l': [0,0,0],
    'index_01_l': [0.2967059910297394, 0.0, -0.0],
    'index_02_l': [0.0872664600610733, 0.0, -0.0],
    'index_03_l': [0.13962633907794952, 0.0, -0.0],
    'middle_metacarpal_l': [0,0,0],
    'middle_01_l': [0.24434612691402435, 0.0, -0.0],
    'middle_02_l': [0.3141592741012573, 0.0, -0.0],
    'middle_03_l': [0.2967059910297394, 0.0, -0.0],
    'ring_metacarpal_l': [0,0,0],
    'ring_01_l': [0.1745329350233078, 0.0, -0.0],
    'ring_02_l': [0.418878972530365, 0.0, -0.0],
    'ring_03_l': [0.33161255717277527, 0.0, -0.0],
    'pinky_metacarpal_l': [0,0,0],
    'pinky_01_l': [0.24434612691402435, 0.0, -0.0],
    'pinky_02_l': [0.36651912331581116, 0.0, -0.0],
    'pinky_03_l': [0.2617994248867035, 0.0, -0.0]
}


def symmetrize_daz_tu_ue5_pose_rotations():
    symmetric = {bone_name[:-1]+"r": [-y, -z, x] for bone_name, (y, z, x) in DAZ_TO_UE5_POSE_ROTATIONS.items() if bone_name.endswith("_l")}
    DAZ_TO_UE5_POSE_ROTATIONS.update(symmetric)

symmetrize_daz_tu_ue5_pose_rotations()