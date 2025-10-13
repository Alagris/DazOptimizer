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

def rle_encode(x:np.ndarray, dropna=False):
    """
    Run length encoding.
    Based on http://stackoverflow.com/a/32681075, which is based on the rle
    function from R.
    Parameters
    ----------
    x : 1D array_like
        Input array to encode
    dropna: bool, optional
        Drop all runs of NaNs.
    Returns
    -------
    start positions, run lengths, run values
    """
    where = np.flatnonzero
    x = x.reshape(-1)
    n = len(x)
    if n == 0:
        return (np.array([], dtype=int),
                np.array([], dtype=int),
                np.array([], dtype=x.dtype))
    starts = np.r_[0, where(~np.isclose(x[1:], x[:-1], equal_nan=True)) + 1]
    lengths = np.diff(np.r_[starts, n])
    values = x[starts]
    if dropna:
        mask = ~np.isnan(values)
        starts, lengths, values = starts[mask], lengths[mask], values[mask]
    return lengths

def rle_decode(rle: [int], shape)->np.ndarray:
    mat = np.empty(shape, dtype=bool)
    value = False
    offset = 0
    mat1d = mat.reshape(-1)
    for l in rle:
        end = offset+int(l)
        mat1d[offset:end] = value
        value = not value
        offset = end
    return mat

def rle_decode_with_value(rle: [int], value, mat)->np.ndarray:
    mat = mat.reshape(-1)
    is_true = False
    offset = 0
    for l in rle:
        end = offset+l
        if is_true:
            mat[offset:end] = value
        is_true = not is_true
        offset = end
    return mat

def serialize_object(obj, vertices=False, vertex_normals=False, loops=False, polygons=False, polygon_normals=False, uvs=False):
        mesh = obj.data
        # use the vertices numpy array
        if vertices:
            print("vertices=",[tuple(v.co) for v in mesh.vertices])
        if vertex_normals:
            print("vertex_normals=",[tuple(v.normal) for v in mesh.vertices])
        if loops:
            print("loops=", [(v.vertex_index, v.index, v.edge_index) for v in mesh.loops])
        if polygons:
            print("polygons=", [(v.loop_start, v.loop_total) for v in mesh.polygons])
        if polygon_normals:
            print("polygon_normals=", [tuple(v.normal) for v in mesh.polygons])
        if uvs:
            # obj.select_set(True)
            # bpy.context.view_layer.objects.active = obj
            # bpy.ops.object.mode_set(mode='EDIT')
            bm = bmesh.new()
            bm.from_mesh(mesh)
            bm.faces.ensure_lookup_table()
            uv_layer = bm.loops.layers.uv.verify()
            print("uvs=", [[tuple(bm_loop[uv_layer].uv) for bm_loop in bm_face.loops] for bm_face in bm.faces])


class BakedImg:
    def __init__(self, i: bpy.types.Image, p: bool=False):
        self.np = None
        self.image = None
        self.path = None
        if isinstance(i, bpy.types.Image):
            self.image = i
            self.is_alpha = p
            self.path = os.path.basename(i.filepath)
        else:
            self.np = i
            self.path = p

    def to_numpy(self):
        if self.np is None:
            from PIL import Image
            path = bpy.path.abspath(self.image.filepath)
            i = np.array(Image.open(path)) / np.float32(255)
            self.np = (i[:, :, 3] if i.shape[2]>3 else np.array(1)) if self.is_alpha else i[:, :, :3]
        return self.np

    def to_image(self, path):
        img = np_to_pil(self.np)
        img.save(path)
        return bpy.data.images.load(path)


def np_to_pil(x: np.ndarray):
    from PIL import Image
    if x.dtype.kind == 'f':
        x = (x * 255).astype(np.uint8)
    return Image.fromarray(x)


def pil_to_np(img):
    img = np.array(img) / np.float32(255)
    return img


def open_img_to_np(filepath):
    from PIL import Image
    img_path = bpy.path.abspath(filepath)
    img_np = Image.open(img_path)
    img_np = pil_to_np(img_np)
    return img_np

def linearrgb_to_srgb_channel(c):
    if c < 0:
        return 0
    elif c < 0.0031308:
        return 12.92 * c
    else:
        return 1.055 * (c ** (1 / 2.4)) - 0.055

def linearrgb_to_srgb(c):
    r, g, b = c[:3]
    r = linearrgb_to_srgb_channel(r)
    g = linearrgb_to_srgb_channel(g)
    b = linearrgb_to_srgb_channel(b)
    if len(c)==3:
        return np.array((r, g, b))
    else:
        return np.array((r, g, b, c[-1]))

def srgb_to_linearrgb(c):
    mask = c >= 0.04045
    c[mask] = ((c[mask] + 0.055) / 1.055) ** 2.4
    c[~mask] = c[~mask] / 12.92
    return c


class NodesUtils:

    @staticmethod
    def bake_textures(input_socket):

        def tonp(x):
            return x if isinstance(x, np.ndarray) or np.isscalar(x) else x.to_numpy()

        def topa(x):
            return [x.path] if isinstance(x, BakedImg) else []

        def channels(x):
            if x.ndim == 0:
                return 1
            elif x.ndim == 1:
                if len(x)==4 and x[3]==1:
                    return 3
                return len(x)
            elif x.ndim < 3:
                return 1
            else:
                return x.shape[2]

        def size(x, axis):
            if x.ndim <= 1:
                return 1
            else:
                return x.shape[axis]

        def to_size(x, w, h):
            if x.ndim < 3:
                return x
            h2, w2, c2 = x.shape
            if h2==h and w2==w:
                return x
            zeros = np.zeros((h, w, c2))
            zeros[:h2, :w2] = x
            return zeros

        def to_channels(x, c):
            if x.ndim == 0:
                return x.repeat(c)
            elif x.ndim == 1:
                if c == 1:
                    return np.mean(x)
                c2 = len(x)
                if c2 >= c:
                    if c == 1:
                        return np.mean(x)
                    return x[:c]
                o = np.zeros(c)
                if c == 4:
                    o[3] = 1
                o[:c2] = x
                return o
            elif x.ndim < 3:
                return x[:, :, None].repeat(c, axis=2)
            else:
                c2 = x.shape[2]
                if c2 >= c:
                    if c == 1:
                        return np.expand_dims(np.mean(x, axis=2), axis=2)
                    return x[:, :, :c]
                new_shape = list(x.shape)
                new_shape[2] = c
                o = np.zeros(new_shape)
                if c == 4:
                    o[:, :, 3] = 1
                o[:, :, :c2] = x
                return o


        if len(input_socket.links) == 0:
            return np.array(input_socket.default_value)
        elif len(input_socket.links) == 1:
            link = input_socket.links[0]
            node = link.from_node
            src_socket = link.from_socket
            if isinstance(node, bpy.types.ShaderNodeRGB):
                rgb = linearrgb_to_srgb(src_socket.default_value)
                return rgb
            elif isinstance(node, bpy.types.ShaderNodeMath):
                a_i = NodesUtils.bake_textures(node.inputs[0])
                b_i = NodesUtils.bake_textures(node.inputs[1])
                p = os.path.commonprefix(topa(a_i) + topa(b_i))
                a = tonp(a_i)
                b = tonp(b_i)
                if node.operation == "MULTIPLY":
                    c = a * b
                elif node.operation == "ADD":
                    c = a + b
                elif node.operation == "SUBTRACT":
                    c = a - b
                elif node.operation == "DIVIDE":
                    c = a / b
                return BakedImg(c, p)
            elif isinstance(node, bpy.types.ShaderNodeTexImage):
                return BakedImg(node.image, src_socket.name == "Alpha")
            elif isinstance(node, bpy.types.ShaderNodeMix):
                a_i = NodesUtils.bake_textures(node.inputs['A'])
                b_i = NodesUtils.bake_textures(node.inputs['B'])
                alpha_i = NodesUtils.bake_textures(node.inputs['Factor'])
                p = os.path.commonprefix(topa(a_i) + topa(b_i) + topa(alpha_i))
                a = tonp(a_i)
                b = tonp(b_i)
                alpha = tonp(alpha_i)
                t0, t1 = 3000, 1000
                # print(node, "[0] a=",a[t0, t1] if a.ndim>=2 else a, "b=",b[t0, t1] if b.ndim>=2 else b, "alpha=",alpha[t0, t1] if alpha.ndim>=2 else alpha)
                max_channels = max(channels(a), channels(b), channels(alpha))
                a = to_channels(a, max_channels)
                b = to_channels(b, max_channels)
                alpha = to_channels(alpha, max_channels)
                max_height = max(size(a, 0), size(b, 0), size(alpha, 0))
                max_width = max(size(a, 1), size(b, 1), size(alpha, 1))
                a = to_size(a, max_height, max_width)
                b = to_size(b, max_height, max_width)
                alpha = to_size(alpha, max_height, max_width)

                if node.blend_type == 'MIX':
                    img_c = a * (1 - alpha) + b * alpha
                    print(node,
                          "[1] a=", a[t0, t1] if a.ndim >= 2 else a,
                          "b=", b[t0, t1] if b.ndim >= 2 else b,
                          "alpha=", alpha[t0, t1] if alpha.ndim >= 2 else alpha,
                          "c=", img_c[t0, t1] if img_c.ndim >= 2 else img_c)
                    return BakedImg(img_c, p)
                elif node.blend_type == 'MULTIPLY':
                    img_c = a * (b * alpha + 1 - alpha)
                    return BakedImg(img_c, p)
        return None

    @staticmethod
    def add_explicit_uvs(mat, uv_layer):
        node_tree = mat.node_tree
        ns = node_tree.nodes
        ls = node_tree.links
        tex_nodes = NodesUtils.find_all_by_type(node_tree, bpy.types.ShaderNodeTexImage)
        any_free_socket = False
        for tn in tex_nodes:
            if len(tn.inputs['Vector'].links)==0:
                any_free_socket = True
                break
        if any_free_socket:
            uv_node = ns.new('ShaderNodeUVMap')
            uv_node.uv_map = uv_layer
            output_socket = uv_node.outputs['UV']
            for tn in tex_nodes:
                input_socket = tn.inputs['Vector']
                if len(input_socket.links) == 0:
                    ls.new(input_socket, output_socket)

    @staticmethod
    def contains_subgroup(mat, subgroup):
        for n in mat.node_tree.nodes:
            if isinstance(n, bpy.types.ShaderNodeGroup) and n.node_tree.name == subgroup:
                return True

    @staticmethod
    def collect_all_before(node, outputs):
        if node not in outputs:
            outputs.add(node)
            for input_socket in node.inputs:
                for link in input_socket.links:
                    NodesUtils.collect_all_before(link.from_node, outputs)
        return outputs

    @staticmethod
    def delete_all_before(node_tree, node):
        for node in NodesUtils.collect_all_before(node, set()):
            node_tree.nodes.remove(node)

    @staticmethod
    def backwards_search_for(node, t: type, outputs):
        if node not in outputs:
            if isinstance(node, t):
                outputs.add(node)
            for i in node.inputs:
                NodesUtils.from_socket_backwards_search_for(i, t, outputs)
        return outputs

    @staticmethod
    def from_socket_backwards_search_for(input_socket, t: type, outputs):
        for link in input_socket.links:
            NodesUtils.backwards_search_for(link.from_node, t, outputs)
        return outputs

    @staticmethod
    def find_by_type(node_tree, t: type):
        for node in node_tree.nodes:
            if isinstance(node, t):
                return node

    @staticmethod
    def find_all_by_type(node_tree, t: type):
        return [node for node in node_tree.nodes if isinstance(node, t)]

    @staticmethod
    def new_mat(name):
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True
        mat.node_tree.nodes.clear()
        return mat

    @staticmethod
    def remove_all_mats(mesh_object, name=None, excpt=None):
        bpy.context.view_layer.objects.active = mesh_object
        new_mat = None
        mat_names = [m.name for m in mesh_object.material_slots]
        mat_count = len(mesh_object.material_slots)
        if name is not None:
            new_mat = NodesUtils.new_mat(name)
            mesh_object.data.materials.append(new_mat)
            new_slot = mesh_object.material_slots[name]
            mesh_object.active_material_index = new_slot.slot_index
            for _ in range(mat_count):
                bpy.ops.object.material_slot_move(direction='UP')
        if excpt is None:
            excpt = []
        mat_count = len(mesh_object.material_slots)
        for mat_name in mat_names:
            if mat_names in excpt:
                mat = mesh_object.material_slots[mat_name]
                mesh_object.active_material_index = mat.slot_index
                for _ in range(mat_count-mat.slot_index-1):
                    bpy.ops.object.material_slot_move(direction='DOWN')
        for mat_name in mat_names:
            if mat_name not in excpt:
                mat = mesh_object.material_slots[mat_name]
                mesh_object.active_material_index = mat.slot_index
                bpy.ops.object.material_slot_remove()

        return new_mat

    @staticmethod
    def gen_simple_material(node_tree, filepaths, output_socket=None, shift_x=0, uvs=None):
        ns = node_tree.nodes
        ls = node_tree.links
        if output_socket is None:
            output_node = ns.new('ShaderNodeOutputMaterial')
            output_node.location = (shift_x+400, 0)
            output_socket = output_node.inputs['Surface']
        if bpy.context.scene.get('daz_optim_toon'):
            bsdf_node = ns.new('ShaderNodeGroup')
            bsdf_node.node_tree = bpy.data.node_groups['DAZ Toon Diffuse']
            if 'DAZ Toon Light' in bpy.data.node_groups:
                light_node = ns.new('ShaderNodeGroup')
                light_node.node_tree = bpy.data.node_groups['DAZ Toon Light']
                light_node.location = (shift_x+200, 0)
                ls.new(light_node.inputs['Input'], bsdf_node.outputs['Output'])
                ls.new(output_socket, light_node.outputs['Output'])
            else:
                ls.new(output_socket, bsdf_node.outputs['Output'])
            channels = ['Color', 'Normal']
        else:
            bsdf_node = ns.new('ShaderNodeBsdfPrincipled')
            ls.new(output_socket, bsdf_node.outputs['BSDF'])
            channels = ['Base Color', 'Roughness', 'Normal']
        bsdf_node.location = (shift_x, 0)
        bsdf_node.name = 'simple_material_bsdf'
        if isinstance(uvs, str):
            uv_node = ns.new('ShaderNodeUVMap')
            uv_node.location = (-900 + shift_x, 0)
            uv_node.uv_map = uvs
            uvs = uv_node
        if isinstance(uvs, bpy.types.ShaderNodeUVMap):
            uvs = uvs.outputs['UV']
        for idx, channel in enumerate(channels):
            filepath_channel = 'Base Color' if channel == "Color" else channel
            if filepath_channel in filepaths:
                filepath = filepaths[filepath_channel]
                if isinstance(filepath, list):
                    if len(filepath) > 0:
                        filepath = filepath[0]
                    else:
                        continue
                if isinstance(filepath, str):
                    filepath = bpy.data.images.load(filepath)
                    filepath.colorspace_settings.name = 'sRGB' if channel == 'Base Color' else 'Non-Color'
                elif isinstance(filepath, np.ndarray):
                    continue
                tex_node = ns.new('ShaderNodeTexImage')
                tex_node.name = 'simple_material_' + channel
                tex_node.location = (-600 + shift_x, -(idx - 1) * 300)
                tex_node.image = filepath
                if uvs is not None:
                    ls.new(tex_node.inputs['Vector'], uvs)
                if channel == 'Normal':
                    norm_map_node = ns.new('ShaderNodeNormalMap')
                    norm_map_node.location = (-200 + shift_x, -idx * 200)
                    norm_map_node.name = 'simple_material_normal_map'
                    ls.new(bsdf_node.inputs[channel], norm_map_node.outputs['Normal'])
                    ls.new(norm_map_node.inputs['Color'], tex_node.outputs['Color'])
                else:
                    ls.new(bsdf_node.inputs[channel], tex_node.outputs['Color'])


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
HEAD_RLE = np.array( [279, 165, 3211, 189, 529, 170, 3204, 196, 524, 176, 3197, 202, 519, 181, 3191, 207, 516, 185, 3186, 212, 510, 191, 3178, 220, 504, 197, 3171, 227, 498, 203, 3163, 235, 493, 207, 3157, 242, 487, 212, 3151, 248, 482, 217, 3145, 254, 478, 220, 3139, 261, 474, 224, 3133, 267, 470, 227, 3129, 272, 466, 232, 3123, 278, 460, 238, 3118, 283, 454, 243, 3114, 287, 449, 248, 3110, 291, 445, 252, 3106, 295, 440, 257, 3102, 299, 435, 262, 3097, 304, 431, 266, 3092, 309, 427, 270, 3088, 313, 423, 273, 3085, 316, 420, 277, 3081, 320, 416, 280, 3079, 322, 412, 284, 3076, 326, 407, 289, 3073, 329, 403, 292, 3071, 332, 398, 297, 3067, 336, 393, 302, 3064, 339, 389, 306, 3060, 343, 385, 309, 3058, 345, 382, 313, 3055, 348, 378, 316, 3053, 350, 375, 319, 3050, 353, 372, 322, 3048, 356, 368, 325, 3045, 359, 365, 329, 3042, 362, 361, 333, 3039, 364, 358, 337, 3036, 366, 355, 341, 3033, 369, 350, 345, 3030, 372, 347, 349, 3027, 375, 343, 352, 3024, 378, 340, 355, 3021, 382, 337, 357, 3019, 385, 333, 360, 3016, 389, 329, 363, 3014, 392, 325, 366, 3012, 395, 320, 370, 3010, 398, 316, 373, 3008, 400, 313, 376, 3006, 403, 309, 379, 3004, 405, 307, 381, 3002, 407, 304, 385, 2999, 410, 300, 388, 2997, 412, 297, 391, 2995, 415, 294, 393, 2994, 416, 291, 396, 2992, 419, 288, 398, 2990, 422, 284, 402, 2987, 424, 281, 405, 2985, 427, 278, 407, 2983, 429, 275, 410, 2981, 431, 273, 412, 2979, 433, 271, 414, 2977, 435, 268, 417, 2975, 437, 266, 419, 2973, 439, 263, 422, 2971, 441, 261, 424, 2969, 443, 259, 426, 2967, 445, 256, 429, 2965, 447, 254, 431, 2963, 449, 251, 434, 2961, 452, 248, 436, 2959, 454, 246, 438, 2957, 456, 244, 440, 2956, 457, 242, 441, 2955, 459, 240, 443, 2953, 461, 237, 446, 2952, 462, 235, 448, 2950, 464, 233, 450, 2949, 465, 231, 452, 2947, 467, 229, 454, 2945, 469, 227, 455, 2945, 470, 225, 457, 2943, 472, 223, 459, 2941, 474, 221, 461, 2940, 475, 219, 463, 2938, 477, 217, 465, 2936, 479, 215, 467, 2934, 480, 213, 470, 2933, 481, 211, 472, 2931, 483, 209, 474, 2929, 485, 207, 476, 2928, 486, 205, 478, 2926, 488, 203, 480, 2924, 490, 201, 481, 2923, 491, 200, 483, 2922, 492, 198, 485, 2920, 494, 196, 487, 2918, 495, 195, 489, 2917, 496, 193, 491, 2915, 498, 191, 493, 2913, 500, 189, 495, 2911, 501, 188, 497, 2910, 502, 186, 499, 2908, 504, 184, 501, 2906, 506, 183, 502, 2905, 507, 181, 504, 2903, 509, 179, 505, 2902, 511, 177, 507, 2900, 512, 176, 509, 2899, 513, 174, 510, 2898, 515, 172, 512, 2896, 516, 172, 513, 2895, 517, 170, 515, 2893, 519, 168, 516, 2893, 520, 166, 518, 2891, 521, 165, 520, 2890, 522, 163, 521, 2889, 524, 161, 523, 2888, 524, 161, 524, 2886, 526, 159, 526, 2885, 526, 158, 527, 2884, 528, 157, 528, 2883, 528, 156, 530, 2881, 530, 155, 531, 2879, 532, 153, 533, 2878, 532, 152, 535, 2876, 534, 151, 536, 2874, 536, 149, 537, 2874, 537, 147, 539, 2872, 538, 147, 540, 2870, 540, 145, 541, 2869, 542, 143, 543, 2868, 542, 142, 544, 2867, 544, 141, 545, 2865, 545, 140, 547, 2864, 546, 138, 548, 2863, 547, 138, 549, 2862, 548, 136, 551, 2860, 549, 136, 551, 2859, 551, 134, 553, 2858, 551, 134, 553, 2857, 553, 132, 555, 2855, 554, 132, 555, 2855, 555, 130, 557, 2853, 556, 130, 557, 2853, 557, 128, 559, 2851, 558, 128, 559, 2851, 558, 127, 561, 2849, 560, 126, 561, 2849, 560, 125, 563, 2847, 561, 125, 563, 2847, 562, 123, 565, 2845, 563, 123, 566, 2844, 564, 121, 567, 2843, 565, 121, 568, 2842, 566, 119, 569, 2841, 567, 119, 570, 2840, 568, 117, 571, 2840, 568, 117, 572, 2838, 570, 116, 572, 2838, 570, 115, 573, 2837, 572, 114, 574, 2835, 573, 114, 574, 2835, 573, 113, 575, 2834, 575, 112, 576, 2832, 576, 112, 576, 2832, 577, 111, 576, 2831, 578, 110, 578, 2830, 579, 109, 579, 2828, 580, 109, 579, 2828, 581, 107, 581, 2826, 582, 107, 581, 2826, 582, 106, 583, 2825, 583, 105, 583, 2824, 584, 104, 585, 2823, 584, 104, 585, 2823, 585, 103, 586, 2821, 586, 102, 587, 2820, 588, 101, 588, 2819, 588, 101, 588, 2818, 589, 100, 590, 2816, 591, 98, 591, 2816, 591, 98, 592, 2814, 592, 97, 593, 2814, 593, 95, 594, 2813, 594, 94, 596, 2812, 594, 93, 597, 2811, 595, 92, 599, 2810, 596, 90, 600, 2809, 597, 90, 601, 2808, 597, 89, 603, 2806, 598, 88, 604, 2806, 598, 88, 605, 2804, 599, 87, 606, 2804, 599, 86, 608, 2802, 600, 85, 609, 2802, 600, 85, 610, 2801, 601, 83, 611, 2800, 602, 82, 612, 2800, 602, 82, 613, 2799, 602, 81, 614, 2798, 603, 81, 614, 2798, 604, 79, 616, 2796, 605, 79, 616, 2796, 606, 77, 618, 2795, 606, 77, 618, 2794, 607, 77, 618, 2794, 608, 75, 620, 2792, 609, 74, 621, 2792, 609, 73, 622, 2791, 611, 71, 624, 2790, 611, 71, 624, 2789, 613, 69, 625, 2789, 613, 68, 627, 2788, 613, 68, 627, 2787, 615, 66, 628, 2787, 615, 66, 629, 2786, 615, 65, 630, 2785, 617, 64, 630, 2785, 617, 63, 632, 2783, 618, 63, 632, 2783, 618, 63, 633, 2782, 619, 61, 634, 2781, 620, 61, 635, 2780, 620, 61, 635, 2780, 620, 60, 637, 2778, 621, 60, 637, 2778, 622, 59, 638, 2777, 622, 59, 638, 2776, 624, 57, 640, 2775, 624, 57, 640, 2775, 624, 57, 640, 2774, 626, 56, 641, 2773, 626, 56, 641, 2773, 626, 55, 642, 2772, 628, 54, 643, 2771, 628, 54, 643, 2771, 628, 54, 643, 2771, 629, 52, 645, 2769, 630, 52, 645, 2769, 630, 52, 645, 2769, 631, 51, 646, 2768, 631, 50, 647, 2768, 631, 50, 647, 2767, 633, 49, 648, 2766, 633, 49, 648, 2766, 633, 48, 650, 2765, 633, 48, 650, 2764, 635, 47, 650, 2764, 635, 47, 651, 2763, 635, 47, 651, 2763, 635, 46, 652, 2762, 637, 45, 653, 2761, 637, 45, 653, 2761, 637, 45, 653, 2761, 637, 44, 655, 2759, 639, 43, 655, 2759, 639, 43, 655, 2759, 639, 43, 656, 2758, 639, 43, 656, 2758, 639, 43, 656, 2758, 639, 43, 657, 2757, 639, 43, 657, 2757, 639, 43, 657, 2757, 639, 43, 657, 2756, 641, 42, 658, 2755, 641, 42, 658, 2755, 641, 42, 658, 2755, 641, 42, 658, 2755, 641, 41, 659, 2755, 641, 41, 660, 2754, 641, 41, 660, 2754, 641, 41, 661, 2753, 641, 40, 662, 2752, 643, 39, 662, 2752, 643, 39, 663, 2751, 643, 39, 663, 2751, 643, 39, 663, 2750, 644, 39, 664, 2749, 645, 38, 664, 2749, 645, 38, 664, 2749, 645, 38, 664, 2749, 645, 38, 665, 2748, 646, 37, 665, 2748, 646, 37, 665, 2748, 646, 37, 665, 2748, 646, 37, 665, 2748, 646, 37, 665, 2748, 646, 37, 665, 2748, 646, 37, 665, 2748, 646, 37, 666, 2747, 646, 36, 667, 2747, 647, 35, 667, 2747, 647, 35, 668, 2746, 647, 35, 668, 2747, 646, 35, 668, 2747, 646, 35, 669, 2746, 646, 35, 669, 2746, 646, 35, 669, 2746, 646, 35, 669, 2745, 647, 34, 671, 2744, 648, 33, 671, 2744, 648, 33, 671, 2744, 648, 33, 671, 2743, 649, 33, 671, 2743, 649, 33, 671, 2743, 649, 33, 671, 2743, 649, 33, 671, 2743, 649, 33, 671, 2743, 649, 33, 672, 2742, 650, 32, 672, 2742, 650, 32, 672, 2742, 650, 32, 672, 2742, 650, 32, 672, 2741, 651, 32, 673, 2740, 651, 32, 673, 2740, 651, 32, 673, 2740, 651, 32, 673, 2740, 651, 32, 674, 2739, 651, 32, 674, 2739, 651, 32, 674, 2739, 651, 32, 674, 2739, 651, 32, 675, 2738, 651, 32, 675, 2738, 651, 32, 675, 2738, 651, 32, 675, 2738, 651, 32, 676, 2736, 652, 32, 676, 2736, 652, 32, 676, 2736, 652, 32, 676, 2736, 652, 32, 676, 2735, 653, 32, 676, 2735, 653, 32, 676, 2735, 653, 32, 676, 2735, 653, 32, 676, 2734, 654, 32, 677, 2733, 654, 31, 678, 2733, 654, 31, 678, 2733, 654, 31, 678, 2732, 655, 31, 678, 2732, 655, 31, 678, 2732, 655, 31, 678, 2732, 655, 31, 678, 2732, 655, 31, 678, 2732, 655, 31, 679, 2731, 655, 31, 679, 2731, 655, 31, 679, 2731, 656, 30, 679, 2730, 657, 30, 679, 2730, 657, 30, 679, 2730, 657, 30, 679, 2730, 657, 30, 679, 2730, 657, 30, 679, 2730, 657, 30, 680, 2729, 657, 30, 680, 2729, 657, 30, 680, 2729, 657, 30, 680, 2728, 658, 30, 680, 2728, 658, 30, 681, 2727, 658, 30, 681, 2727, 658, 30, 681, 2727, 658, 30, 681, 2727, 658, 30, 682, 2726, 658, 30, 682, 2726, 658, 30, 682, 2726, 658, 30, 682, 2726, 658, 30, 682, 2726, 658, 30, 682, 2726, 658, 30, 682, 2726, 658, 30, 682, 2726, 658, 30, 682, 2726, 658, 30, 682, 2726, 658, 31, 681, 2726, 658, 31, 681, 2726, 659, 30, 681, 2726, 659, 30, 681, 2726, 659, 30, 682, 2725, 659, 30, 682, 2725, 659, 30, 682, 2725, 659, 30, 682, 2725, 659, 30, 683, 2724, 659, 30, 683, 2724, 659, 30, 683, 2724, 659, 30, 683, 2724, 659, 30, 683, 2724, 659, 30, 683, 2724, 659, 30, 683, 2724, 659, 30, 683, 2724, 659, 30, 683, 2724, 659, 30, 683, 2724, 659, 31, 682, 2724, 659, 31, 682, 2724, 659, 31, 682, 2724, 659, 31, 682, 2724, 659, 32, 681, 2724, 659, 32, 681, 2724, 659, 32, 681, 2724, 659, 32, 681, 2724, 659, 32, 681, 2724, 659, 33, 680, 2724, 659, 33, 680, 2724, 659, 33, 680, 2725, 658, 33, 680, 2725, 658, 34, 679, 2725, 658, 34, 679, 2725, 658, 34, 679, 2725, 658, 34, 679, 2725, 658, 35, 678, 2725, 658, 35, 678, 2725, 658, 35, 678, 2725, 658, 35, 678, 2725, 658, 36, 677, 2725, 658, 36, 677, 2725, 658, 36, 677, 2725, 658, 36, 677, 2725, 658, 36, 677, 2725, 658, 37, 676, 2725, 658, 37, 676, 2725, 657, 38, 676, 2725, 657, 38, 676, 2726, 656, 39, 675, 2726, 656, 39, 675, 2726, 656, 39, 675, 2726, 656, 39, 675, 2726, 656, 39, 675, 2726, 656, 40, 674, 2726, 656, 40, 674, 2726, 655, 41, 674, 2726, 655, 41, 674, 2727, 654, 42, 673, 2727, 654, 42, 673, 2727, 653, 43, 673, 2727, 653, 44, 672, 2727, 653, 44, 672, 2727, 652, 45, 672, 2727, 652, 46, 671, 2727, 652, 46, 671, 2727, 651, 48, 670, 2727, 651, 48, 670, 2727, 651, 48, 670, 2727, 651, 49, 669, 2727, 651, 49, 669, 2726, 652, 49, 669, 2726, 652, 50, 668, 2726, 652, 50, 668, 2726, 652, 50, 668, 2726, 652, 51, 667, 2725, 653, 51, 667, 2725, 653, 51, 667, 2725, 653, 52, 666, 2725, 653, 52, 666, 2724, 654, 52, 666, 2724, 654, 53, 665, 2724, 654, 53, 665, 2724, 654, 54, 664, 2724, 654, 54, 664, 2724, 654, 54, 664, 2724, 654, 54, 664, 2724, 654, 54, 664, 2724, 654, 55, 663, 2724, 654, 55, 663, 2724, 654, 55, 663, 2724, 654, 55, 663, 2724, 654, 56, 662, 2724, 654, 56, 662, 2724, 654, 56, 662, 2724, 654, 57, 661, 2724, 654, 57, 661, 2724, 654, 57, 661, 2723, 655, 58, 660, 2723, 655, 58, 660, 2723, 655, 59, 659, 2723, 655, 59, 659, 2723, 655, 60, 658, 2723, 654, 61, 658, 2723, 654, 62, 657, 2723, 654, 62, 657, 2723, 654, 63, 656, 2723, 654, 63, 656, 2723, 654, 64, 655, 2723, 654, 64, 655, 2723, 654, 65, 654, 2723, 654, 65, 654, 2723, 653, 67, 653, 2723, 653, 67, 653, 2723, 653, 68, 652, 2723, 653, 68, 652, 2723, 652, 69, 652, 2723, 652, 69, 652, 2723, 652, 69, 652, 2723, 651, 70, 652, 2723, 651, 70, 652, 2723, 651, 70, 652, 2723, 650, 71, 652, 2723, 650, 71, 652, 2723, 650, 71, 652, 2723, 649, 72, 652, 2723, 649, 72, 651, 2724, 649, 72, 651, 2724, 649, 72, 651, 2724, 649, 72, 651, 2724, 648, 73, 651, 2724, 648, 73, 651, 2724, 648, 73, 651, 2724, 648, 73, 651, 2724, 647, 74, 651, 2724, 647, 74, 651, 2724, 647, 74, 651, 2724, 647, 74, 651, 2724, 646, 75, 651, 2724, 646, 75, 651, 2724, 646, 75, 651, 2724, 646, 75, 651, 2724, 645, 76, 651, 2724, 645, 76, 651, 2723, 646, 76, 650, 2724, 646, 76, 650, 2724, 646, 76, 650, 2724, 645, 78, 649, 2724, 645, 78, 649, 2724, 645, 78, 649, 2724, 645, 78, 649, 2724, 644, 79, 649, 2724, 644, 79, 649, 2724, 644, 79, 648, 2725, 643, 80, 648, 2725, 643, 80, 648, 2725, 643, 81, 647, 2725, 642, 82, 646, 2726, 642, 82, 646, 2726, 641, 83, 646, 2726, 641, 84, 645, 2726, 640, 85, 645, 2726, 640, 85, 644, 2727, 640, 85, 644, 2727, 639, 86, 644, 2727, 639, 87, 643, 2727, 639, 87, 642, 2728, 639, 87, 642, 2728, 638, 88, 642, 2728, 638, 89, 641, 2728, 638, 89, 641, 2728, 637, 90, 640, 2729, 637, 90, 640, 2729, 637, 90, 640, 2729, 636, 92, 639, 2729, 636, 92, 639, 2729, 635, 93, 639, 2729, 635, 93, 639, 2729, 635, 94, 638, 2729, 634, 95, 638, 2729, 634, 95, 637, 2730, 634, 96, 636, 2730, 633, 97, 636, 2730, 633, 97, 636, 2730, 632, 99, 634, 2731, 632, 99, 634, 2731, 631, 101, 633, 2731, 631, 101, 633, 2732, 629, 102, 633, 2732, 629, 103, 631, 2733, 628, 104, 631, 2733, 628, 104, 631, 2733, 627, 106, 630, 2733, 627, 106, 629, 2734, 626, 108, 628, 2734, 626, 108, 628, 2734, 625, 110, 627, 2734, 624, 111, 626, 2735, 624, 112, 625, 2735, 623, 113, 625, 2735, 622, 115, 624, 2735, 621, 116, 623, 2736, 621, 117, 622, 2736, 620, 118, 622, 2736, 620, 119, 621, 2736, 619, 120, 621, 2737, 618, 121, 619, 2738, 617, 122, 619, 2738, 617, 123, 618, 2738, 616, 124, 618, 2738, 616, 125, 616, 2739, 615, 126, 616, 2739, 615, 127, 615, 2739, 614, 128, 614, 2740, 614, 129, 613, 2741, 612, 130, 613, 2741, 612, 131, 611, 2742, 611, 132, 611, 2742, 611, 133, 609, 2743, 610, 134, 609, 2743, 609, 136, 608, 2743, 609, 136, 608, 2743, 608, 138, 607, 2743, 607, 139, 606, 2745, 605, 141, 605, 2745, 605, 141, 605, 2745, 604, 143, 604, 2745, 603, 144, 603, 2263, 54, 430, 602, 145, 602, 2259, 62, 426, 601, 146, 602, 2254, 81, 412, 600, 148, 600, 2246, 94, 408, 600, 149, 599, 2242, 103, 403, 599, 150, 599, 2237, 117, 395, 598, 151, 597, 2234, 125, 391, 597, 153, 596, 2230, 133, 387, 597, 154, 594, 2227, 146, 378, 596, 156, 593, 2224, 153, 374, 596, 157, 592, 2221, 159, 371, 595, 159, 590, 2219, 165, 368, 595, 159, 590, 2214, 173, 365, 594, 161, 589, 2210, 182, 360, 594, 162, 587, 463, 27, 1717, 190, 357, 592, 164, 586, 454, 40, 1709, 198, 353, 591, 165, 586, 445, 52, 1701, 207, 349, 591, 166, 584, 428, 73, 1694, 216, 344, 590, 168, 583, 401, 102, 1687, 225, 341, 588, 169, 583, 383, 122, 1681, 232, 338, 587, 171, 581, 375, 133, 1674, 240, 335, 587, 172, 580, 366, 143, 1669, 247, 332, 586, 173, 580, 357, 154, 1662, 257, 327, 585, 175, 578, 349, 164, 1657, 270, 319, 584, 175, 578, 330, 184, 1643, 287, 315, 583, 177, 576, 327, 189, 1638, 296, 310, 582, 179, 575, 313, 209, 147, 35, 1445, 305, 306, 582, 179, 575, 295, 231, 125, 62, 1432, 312, 303, 581, 181, 573, 278, 263, 74, 117, 1400, 329, 300, 581, 182, 572, 274, 276, 47, 144, 1387, 335, 298, 580, 184, 571, 260, 490, 1364, 351, 296, 580, 184, 570, 252, 508, 1351, 357, 294, 579, 186, 569, 243, 526, 1319, 381, 294, 578, 187, 568, 234, 572, 1278, 387, 292, 577, 188, 567, 226, 590, 1255, 402, 291, 577, 189, 566, 217, 603, 1247, 437, 260, 576, 190, 566, 208, 616, 1238, 460, 243, 575, 191, 564, 205, 624, 1225, 473, 239, 574, 192, 564, 200, 634, 1202, 496, 234, 573, 194, 563, 191, 647, 1194, 509, 226, 572, 195, 561, 188, 654, 1186, 523, 217, 571, 196, 561, 184, 661, 1174, 536, 213, 570, 198, 559, 181, 667, 1154, 568, 200, 568, 200, 558, 176, 674, 1116, 608, 196, 568, 200, 557, 168, 685, 1096, 631, 192, 566, 202, 556, 164, 690, 1077, 658, 183, 566, 202, 555, 160, 697, 1057, 680, 179, 565, 204, 554, 156, 702, 1046, 694, 176, 564, 205, 552, 152, 708, 1041, 702, 172, 563, 206, 552, 148, 713, 1036, 709, 169, 563, 207, 550, 146, 717, 1017, 730, 167, 561, 209, 549, 143, 721, 1012, 737, 164, 561, 209, 548, 141, 725, 1006, 745, 162, 559, 211, 547, 138, 729, 1001, 752, 159, 559, 211, 546, 136, 733, 995, 759, 157, 558, 213, 544, 135, 736, 990, 766, 155, 557, 214, 543, 132, 740, 985, 773, 152, 556, 215, 542, 130, 743, 981, 780, 149, 556, 216, 541, 127, 747, 977, 788, 145, 554, 218, 539, 125, 754, 970, 795, 141, 554, 219, 538, 122, 761, 963, 801, 139, 552, 220, 537, 121, 768, 955, 807, 136, 552, 221, 535, 119, 775, 948, 812, 135, 550, 223, 534, 116, 783, 940, 818, 132, 550, 223, 533, 115, 794, 928, 824, 130, 548, 225, 531, 114, 800, 921, 830, 127, 548, 226, 530, 112, 807, 914, 835, 125, 546, 227, 529, 111, 813, 908, 840, 122, 546, 228, 528, 109, 819, 901, 845, 121, 544, 229, 527, 108, 825, 895, 850, 118, 544, 230, 526, 106, 832, 888, 855, 116, 542, 232, 525, 104, 838, 882, 859, 114, 542, 232, 525, 102, 845, 875, 864, 112, 540, 234, 524, 100, 851, 869, 869, 109, 540, 234, 524, 98, 858, 862, 873, 108, 538, 236, 523, 96, 864, 856, 877, 106, 538, 236, 523, 94, 871, 850, 881, 104, 536, 238, 522, 92, 877, 844, 887, 100, 535, 239, 522, 90, 884, 838, 891, 98, 534, 240, 521, 88, 890, 833, 895, 95, 533, 241, 521, 83, 899, 828, 899, 93, 531, 243, 520, 74, 912, 823, 903, 91, 529, 244, 520, 70, 921, 817, 907, 88, 529, 245, 518, 66, 930, 812, 910, 87, 527, 246, 518, 62, 939, 806, 914, 85, 525, 248, 518, 56, 948, 801, 918, 83, 524, 249, 517, 52, 957, 769, 948, 80, 523, 250, 517, 49, 964, 761, 955, 78, 521, 252, 516, 46, 972, 742, 972, 76, 519, 254, 515, 44, 978, 729, 983, 74, 518, 256, 514, 42, 985, 720, 992, 70, 516, 257, 514, 40, 991, 711, 1001, 67, 514, 259, 513, 38, 997, 703, 1008, 65, 512, 261, 512, 36, 1003, 694, 1016, 62, 511, 262, 512, 34, 1010, 685, 1022, 61, 509, 264, 511, 32, 1016, 676, 1029, 59, 508, 266, 510, 30, 1023, 667, 1035, 57, 506, 268, 510, 28, 1029, 660, 1040, 55, 505, 270, 509, 25, 1037, 652, 1045, 53, 503, 272, 509, 22, 1044, 646, 1049, 51, 501, 275, 508, 20, 1050, 637, 1056, 49, 499, 278, 507, 18, 1056, 629, 1061, 50, 494, 281, 506, 17, 1063, 621, 1066, 48, 4, 1, 486, 285, 505, 14, 1075, 609, 1070, 52, 486, 286, 504, 12, 1081, 603, 1073, 52, 484, 288, 503, 9, 1089, 596, 1077, 51, 483, 289, 501, 7, 1096, 589, 1082, 50, 481, 291, 500, 5, 1103, 580, 1088, 48, 481, 292, 499, 3, 1109, 573, 1093, 47, 479, 293, 499, 1, 1116, 565, 1098, 46, 478, 294, 1619, 559, 1102, 44, 477, 296, 1621, 553, 1107, 43, 476, 297, 1623, 547, 1112, 42, 474, 298, 1625, 542, 1117, 41, 473, 299, 1629, 534, 1122, 40, 471, 301, 1632, 527, 1127, 39, 469, 302, 1637, 520, 1131, 38, 468, 303, 1640, 513, 1136, 36, 467, 305, 1642, 507, 1140, 36, 465, 307, 1644, 501, 1145, 35, 463, 309, 1651, 490, 1149, 35, 461, 311, 1654, 483, 1153, 35, 459, 313, 1656, 478, 1156, 35, 457, 315, 1658, 470, 1162, 35, 456, 316, 1659, 464, 1167, 34, 455, 318, 1661, 458, 1172, 33, 453, 320, 1663, 452, 1176, 33, 451, 322, 1670, 442, 1179, 33, 449, 324, 1673, 435, 1184, 32, 447, 326, 1675, 429, 1188, 32, 445, 328, 1677, 423, 1193, 31, 443, 330, 1678, 417, 1198, 31, 441, 332, 1680, 410, 1203, 31, 439, 334, 1682, 404, 1208, 30, 437, 336, 1684, 398, 1212, 29, 435, 339, 1686, 392, 1217, 28, 433, 341, 1688, 386, 1221, 28, 430, 344, 1689, 381, 1225, 27, 428, 347, 1691, 375, 1230, 26, 425, 349, 1694, 368, 1236, 25, 421, 353, 1696, 361, 1242, 24, 417, 357, 1698, 355, 1247, 22, 413, 361, 1701, 349, 1252, 21, 404, 370, 1702, 344, 1257, 20, 402, 372, 1703, 339, 1262, 18, 401, 374, 1704, 334, 1267, 17, 400, 375, 1705, 330, 1271, 16, 398, 377, 1707, 322, 1280, 13, 396, 379, 1709, 315, 1287, 10, 395, 380, 1712, 309, 1293, 8, 394, 381, 1713, 304, 1299, 6, 392, 383, 1715, 299, 1306, 1, 391, 385, 1717, 293, 1701, 385, 1719, 288, 1703, 387, 1720, 283, 1705, 389, 1721, 278, 1707, 390, 1724, 272, 1709, 392, 1725, 267, 1711, 394, 1727, 261, 1713, 396, 1729, 255, 1715, 398, 1730, 251, 1716, 400, 1731, 246, 1718, 402, 1733, 240, 1720, 404, 1736, 234, 1721, 406, 1738, 228, 1723, 408, 1740, 222, 1726, 409, 1741, 217, 1728, 412, 1741, 212, 1730, 414, 1742, 207, 1732, 415, 1744, 203, 1733, 417, 1746, 197, 1735, 419, 1748, 191, 1737, 421, 1750, 186, 1738, 423, 1751, 181, 1740, 425, 1753, 175, 1742, 427, 1755, 169, 1744, 429, 1756, 164, 1746, 431, 1757, 159, 1748, 433, 1758, 155, 1750, 434, 1759, 150, 1752, 437, 1760, 144, 1754, 439, 1762, 138, 1756, 441, 1764, 132, 1758, 443, 1766, 126, 1760, 445, 1768, 121, 1761, 447, 1770, 115, 1763, 449, 1776, 105, 1765, 451, 1779, 99, 1766, 453, 1781, 93, 1768, 455, 1783, 87, 1770, 457, 1785, 82, 1771, 459, 1787, 77, 1772, 462, 1788, 71, 1774, 464, 1789, 65, 1777, 466, 1793, 57, 1779, 468, 1796, 50, 1780, 471, 1800, 43, 1781, 473, 1803, 37, 1782, 476, 1805, 29, 1785, 478, 1808, 21, 1788, 481, 1809, 14, 1791, 483, 1811, 7, 1793, 486, 3609, 489, 3605, 492, 3602, 495, 3599, 498, 3597, 500, 3594, 503, 3591, 507, 3588, 509, 3585, 513, 3582, 516, 3578, 520, 3575, 523, 3572, 526, 3568, 530, 3565, 533, 3561, 537, 3557, 541, 3554, 544, 3550, 548, 3547, 550, 3544, 554, 3541, 557, 3538, 560, 3534, 564, 3531, 567, 3527, 571, 3523, 575, 3520, 579, 3515, 584, 3511, 589, 3458, 2, 45, 596, 3450, 7, 41, 602, 3443, 12, 37, 609, 3434, 19, 31, 616, 3426, 26, 25, 624, 3418, 33, 17, 632, 3411, 693, 3400, 705, 3387, 746, 3346, 759, 3333, 772, 3305, 809, 3269, 841, 3251, 849, 3248, 857, 3240, 865, 3233, 872, 3225, 873, 3225, 870, 3227, 867, 3230, 865, 3233, 861, 3237, 858, 3240, 855, 3243, 852, 3246, 849, 3248, 847, 3251, 843, 3254, 841, 3256, 839, 3258, 837, 3260, 835, 3262, 833, 3265, 830, 3267, 828, 3269, 823, 3274, 818, 3279, 813, 3284, 808, 3290, 803, 3294, 799, 3298, 796, 3301, 793, 3304, 790, 3307, 788, 3309, 785, 3312, 783, 3314, 781, 3316, 779, 3318, 776, 3321, 774, 3324, 770, 3327, 767, 3330, 764, 3333, 762, 3335, 759, 3338, 757, 3340, 755, 3342, 753, 3344, 751, 3346, 749, 3348, 747, 3349, 746, 3351, 744, 3353, 742, 3354, 741, 3356, 739, 3358, 736, 3361, 734, 3362, 733, 3364, 731, 3366, 729, 3367, 728, 3369, 726, 3370, 725, 3372, 723, 3374, 721, 3375, 720, 3378, 717, 3380, 714, 3383, 712, 3385, 710, 3387, 708, 3389, 706, 3391, 704, 3393, 701, 3396, 699, 3398, 697, 3400, 694, 3402, 693, 3404, 690, 3407, 688, 3408, 687, 3410, 685, 3411, 684, 3413, 682, 3414, 680, 3417, 678, 3418, 677, 3419, 676, 3421, 674, 3422, 672, 3424, 671, 3426, 668, 3428, 666, 3430, 665, 3431, 663, 3434, 661, 3435, 660, 3437, 658, 3439, 656, 3441, 654, 3443, 651, 3446, 649, 3449, 646, 3451, 643, 3454, 641, 3457, 637, 3460, 635, 3463, 632, 3465, 630, 3467, 628, 3470, 624, 3473, 622, 3476, 619, 3478, 617, 3480, 614, 3483, 612, 3485, 609, 3488, 607, 3491, 604, 3493, 601, 3496, 599, 3499, 595, 3502, 593, 3505, 590, 3507, 588, 3509, 586, 3511, 584, 3513, 582, 3515, 580, 3517, 578, 3519, 576, 3521, 574, 3523, 572, 3525, 570, 3526, 569, 3528, 567, 3531, 564, 3533, 562, 3535, 560, 3537, 558, 3539, 556, 3541, 554, 3544, 551, 3546, 549, 3548, 547, 3550, 545, 3552, 542, 3555, 540, 3557, 538, 3559, 536, 3561, 534, 3563, 532, 3565, 530, 3567, 528, 3570, 525, 3572, 523, 3574, 521, 3576, 519, 3578, 517, 3580, 515, 3583, 512, 3585, 510, 3587, 508, 3589, 507, 3590, 505, 3592, 503, 3594, 502, 3595, 500, 3597, 498, 3599, 496, 3601, 494, 3602, 493, 3604, 491, 3606, 489, 3608, 487, 3610, 485, 3612, 484, 3613, 482, 3614, 481, 3616, 479, 3618, 477, 3619, 476, 3621, 474, 3623, 472, 3625, 470, 3626, 469, 3628, 467, 3630, 465, 3632, 463, 3634, 461, 3636, 458, 3639, 456, 3640, 455, 3642, 453, 3644, 451, 3646, 449, 3648, 447, 3650, 445, 3652, 443, 3653, 442, 3655, 440, 3657, 439, 3658, 437, 3660, 435, 3662, 433, 3663, 432, 3665, 430, 3667, 428, 3668, 428, 3669, 426, 3670, 425, 3672, 423, 3673, 422, 3675, 420, 3676, 419, 3678, 418, 3678, 417, 3680, 415, 3681, 415, 3682, 413, 3684, 411, 3685, 410, 3687, 409, 3688, 407, 3689, 406, 3691, 405, 3691, 404, 3693, 403, 3693, 402, 3695, 400, 3696, 400, 3697, 398, 3698, 397, 3700, 396, 3701, 394, 3703, 392, 3704, 391, 3706, 390, 3707, 388, 3708, 387, 3710, 386, 3710, 385, 3712, 384, 3713, 382, 3714, 382, 3715, 380, 3717, 379, 3717, 378, 3719, 377, 3720, 375, 3722, 374, 3722, 373, 3724, 372, 3725, 370, 3726, 370, 3727, 368, 3728, 368, 3729, 366, 3730, 366, 3731, 364, 3732, 364, 3733, 362, 3734, 362, 3735, 360, 3736, 360, 3737, 358, 3738, 358, 3739, 356, 3740, 356, 3741, 354, 3742, 354, 3743, 352, 3744, 352, 3745, 350, 3746, 350, 3747, 348, 3749, 347, 3750, 345, 3752, 344, 3753, 343, 3754, 341, 3755, 341, 3756, 340, 3757, 338, 3758, 338, 3759, 337, 3759, 337, 3760, 335, 3761, 335, 3762, 334, 3762, 334, 3762, 333, 3764, 332, 3764, 332, 3764, 331, 3766, 330, 3766, 330, 3766, 329, 3767, 329, 3768, 328, 3768, 327, 3769, 327, 3769, 327, 3769, 327, 3769, 326, 3770, 326, 3770, 326, 3770, 326, 3770, 326, 3770, 325, 3771, 325, 3771, 324, 3772, 324, 3772, 324, 3772, 323, 3773, 323, 3772, 324, 3772, 323, 3773, 323, 3773, 323, 3772, 324, 3772, 323, 3773, 322, 3773, 322, 3775, 321, 3776, 319, 3778, 317, 3780, 316, 3781, 314, 3782, 313, 3784, 311, 3786, 310, 3786, 309, 3788, 307, 3789, 307, 3790, 305, 3791, 305, 3795, 300, 3796, 299, 3797, 299, 3797, 298, 3799, 296, 3801, 295, 3801, 294, 3803, 293, 3804, 291, 3805, 291, 3806, 289, 3808, 287, 3810, 286, 3810, 285, 3812, 284, 3813, 282, 3814, 282, 3815, 280, 3816, 280, 3817, 278, 3818, 278, 3819, 276, 3820, 275, 3822, 273, 3823, 273, 3824, 271, 3825, 270, 3827, 269, 3827, 268, 3829, 267, 3829, 266, 3830, 266, 3831, 264, 3832, 264, 3832, 263, 3834, 262, 3834, 261, 3835, 261, 3835, 260, 3837, 259, 3838, 257, 3839, 257, 3840, 256, 3841, 254, 3843, 253, 3843, 253, 3844, 251, 3846, 250, 3846, 250, 3847, 248, 3848, 248, 3849, 246, 3850, 246, 3851, 244, 3852, 244, 3852, 243, 3854, 242, 3854, 241, 3855, 241, 3856, 240, 3856, 239, 3857, 239, 3859, 237, 3859, 236, 3862, 234, 3862, 233, 3863, 233, 3863, 232, 3864, 232, 3864, 231, 3866, 230, 3867, 228, 3868, 227, 3870, 226, 3870, 225, 3872, 223, 3873, 223, 3874, 221, 3875, 221, 3876, 219, 3877, 219, 3878, 217, 3879, 217, 3880, 216, 3880, 215, 3882, 214, 3882, 214, 3882, 213, 3884, 212, 3884, 211, 3885, 211, 3886, 209, 3887, 209, 3888, 207, 3889, 207, 3890, 205, 3891, 205, 3892, 203, 3893, 203, 3894, 202, 3894, 201, 3896, 200, 3896, 200, 3897, 198, 3898, 198, 3899, 196, 3900, 196, 3901, 194, 3902, 194, 3903, 192, 3904, 192, 3905, 190, 3906, 190, 3907, 188, 3908, 187, 3910, 186, 3911, 184, 3912, 184, 3913, 182, 3915, 181, 3915, 180, 3917, 179, 3917, 178, 3919, 177, 3919, 176, 3921, 175, 3922, 173, 3923, 173, 3925, 170, 3926, 170, 3926, 170, 3927, 168, 3928, 168, 3929, 166, 3931, 165, 3931, 164, 3933, 163, 3933, 162, 3935, 160, 3936, 160, 3937, 158, 3938, 158, 3939, 156, 3940, 156, 3941, 154, 3942, 154, 3943, 153, 3943, 152, 3945, 151, 3946, 150, 3946, 149, 3948, 148, 3948, 148, 3949, 147, 3949, 146, 3951, 145, 3951, 145, 3952, 144, 3952, 144, 3953, 142, 3954, 142, 3955, 140, 3957, 139, 3957, 139, 3958, 137, 3960, 136, 3960, 136, 3961, 134, 3962, 134, 3963, 133, 3963, 132, 3965, 131, 3965, 131, 3965, 130, 3967, 129, 3967, 129, 3967, 128, 3969, 127, 3969, 126, 3971, 125, 3971, 125, 3971, 124, 3973, 123, 3973, 123, 3973, 122, 3975, 121, 3975, 121, 3976, 120, 3976, 119, 3977, 119, 3978, 118, 3978, 118, 3980, 116, 3980, 115, 3981, 115, 3981, 114, 3983, 113, 3983, 113, 3984, 111, 3985, 111, 3986, 110, 3986, 109, 3988, 108, 3988, 108, 3989, 107, 3989, 106, 3991, 105, 3991, 105, 3992, 104, 3992, 104, 3993, 102, 3994, 102, 3995, 101, 3995, 101, 3995, 100, 3997, 99, 3997, 99, 3997, 99, 3998, 98, 3998, 97, 4000, 96, 4000, 96, 4000, 96, 4001, 94, 4002, 94, 4002, 94, 4003, 93, 4003, 93, 4004, 91, 4005, 91, 4005, 91, 4006, 90, 4006, 89, 4008, 88, 4008, 88, 4009, 87, 4009, 87, 4010, 86, 4010, 86, 4010, 86, 4011, 85, 4011, 84, 4012, 84, 4013, 83, 4013, 83, 4013, 82, 4014, 82, 4015, 81, 4015, 81, 4015, 80, 4016, 80, 4017, 79, 4017, 79, 4018, 78, 4018, 78, 4018, 78, 4019, 77, 4019, 77, 4019, 76, 4021, 75, 4021, 75, 4021, 75, 4022, 74, 4022, 73, 4023, 73, 4024, 72, 4024, 72, 4024, 71, 4026, 70, 4026, 70, 4026, 70, 4026, 70, 4027, 68, 4028, 68, 4028, 68, 4028, 68, 4028, 67, 4030, 65, 4031, 64, 4032, 63, 4033, 62, 4035, 60, 4036, 59, 4037, 59, 4037, 58, 4039, 56, 4040, 55, 4041, 54, 4042, 53, 4044, 51, 4045, 51, 4045, 50, 4046, 49, 4047, 49, 4048, 47, 4049, 46, 4051, 44, 4052, 44, 4052, 43, 4054, 41, 4055, 41, 4055, 40, 4057, 39, 4057, 38, 4058, 38, 4058, 37, 4060, 36, 4060, 35, 4061, 35, 4061, 34, 4062, 34, 4063, 32, 4064, 32, 4064, 31, 4066, 30, 4066, 30, 4066, 29, 4068, 28, 4068, 28, 4068, 27, 4070, 26, 4070, 26, 4070, 26, 4070, 25, 4072, 24, 4072, 24, 4072, 24, 4072, 23, 4073, 23, 4074, 22, 4074, 21, 4075, 21, 4075, 21, 4076, 19, 4077, 19, 4077, 19, 4077, 18, 4079, 17, 4079, 16, 4080, 16, 4080, 16, 4081, 14, 4082, 14, 4082, 14, 4082, 14, 4082, 14, 4082, 14, 4082, 14, 4082, 14, 4083, 13, 4083, 13, 4083, 13, 4084, 12, 4084, 12, 4084, 12, 4085, 11, 4085, 11, 4085, 11, 4085, 11, 4086, 10, 4086, 10, 4086, 10, 4086, 10, 4086, 10, 4087, 9, 4087, 9, 4088, 8, 4088, 8, 4088, 8, 4089, 7, 4089, 7, 4089, 7, 4090, 6, 4090, 6, 4090, 6, 4091, 5, 4091, 5, 4091, 5, 4092, 4, 4092, 4, 4092, 4, 4093, 3, 4093, 3, 4093, 3, 4093, 3, 4094, 2, 4094, 2, 4094, 2, 4094, 2, 4094, 2, 4095, 1, 4095, 1, 4095, 1, 4095, 1, 73727, 1, 4095, 1, 4095, 1, 4095, 1, 4094, 2, 3362812, 4, 4092, 4, 4092, 4, 4092, 4, 4091, 5, 4091, 5, 4091, 5, 4091, 5, 4090, 6, 4090, 6, 4090, 6, 4089, 7, 4089, 7, 4089, 7, 4088, 8, 4088, 8, 4087, 9, 4087, 9, 4087, 9, 4086, 10, 4086, 10, 4086, 10, 4085, 11, 4085, 11, 4085, 11, 4085, 11, 4085, 11, 4085, 11, 4085, 11, 4085, 11, 4084, 12, 4084, 12, 4084, 12, 4084, 12, 4084, 12, 4084, 12, 4084, 12, 4084, 12, 4083, 13, 4083, 13, 4083, 13, 4083, 13, 4082, 14, 4082, 14, 4082, 14, 4082, 14, 4082, 14, 4081, 15, 4081, 16, 4080, 16, 4080, 16, 4079, 18, 4078, 18, 4078, 18, 4078, 19, 4077, 19, 4076, 21, 4075, 21, 4075, 22, 4074, 22, 4073, 24, 4072, 24, 4072, 24, 4072, 24, 4072, 24, 4071, 25, 4071, 25, 4071, 25, 4071, 25, 4070, 27, 4069, 27, 4069, 27, 4069, 27, 4068, 29, 4067, 29, 4067, 29, 4067, 30, 4065, 31, 4065, 31, 4065, 32, 4064, 32, 4064, 32, 4063, 33, 4063, 33, 4063, 33, 4063, 33, 4062, 34, 4062, 35, 4061, 35, 4060, 36, 4060, 36, 4060, 37, 4058, 38, 4058, 38, 4057, 40, 4056, 40, 4055, 41, 4055, 42, 4053, 43, 4053, 44, 4052, 44, 4051, 45, 4051, 46, 4050, 46, 4050, 46, 4049, 48, 4048, 48, 4048, 49, 4046, 50, 4046, 51, 4045, 51, 4044, 53, 4043, 53, 4043, 54, 4042, 54, 4042, 54, 4042, 55, 4041, 55, 4040, 56, 4040, 56, 4040, 56, 4040, 57, 4038, 58, 4038, 58, 4038, 58, 4037, 60, 4036, 60, 4035, 61, 4035, 62, 4034, 62, 4033, 63, 4033, 64, 4032, 64, 4031, 66, 4030, 66, 4030, 66, 4030, 66, 4029, 67, 4029, 68, 4028, 68, 4028, 68, 4028, 68, 4027, 70, 4026, 70, 4026, 70, 4025, 72, 4024, 72, 4024, 72, 4023, 74, 4022, 74, 4022, 75, 4021, 75, 4021, 76, 4019, 77, 4019, 78, 4018, 78, 4018, 79, 4016, 80, 4016, 81, 4015, 81, 4015, 82, 4013, 83, 4013, 84, 4012, 84, 4012, 85, 4011, 85, 4011, 86, 4010, 86, 4010, 87, 4009, 87, 4008, 89, 4007, 90, 4006, 90, 4006, 91, 4004, 93, 4003, 94, 4002, 94, 4001, 96, 4000, 96, 4000, 95, 4000, 96, 4000, 96, 4000, 96, 3999, 96, 4000, 96, 4000, 96, 4000, 96, 3999, 97, 3999, 97, 3999, 97, 3999, 97, 3998, 97, 3999, 97, 3999, 97, 3999, 97, 3999, 97, 3998, 98, 3998, 98, 3998, 98, 3998, 98, 3997, 99, 3997, 99, 3997, 99, 3996, 100, 3996, 100, 3996, 100, 3995, 101, 3995, 101, 3994, 103, 3993, 103, 3993, 103, 3992, 104, 3992, 104, 3992, 104, 3991, 105, 3991, 105, 3990, 106, 3990, 107, 3988, 108, 3988, 108, 3987, 109, 3987, 110, 3985, 111, 3985, 111, 3984, 113, 3983, 113, 3983, 113, 3983, 114, 3981, 115, 3981, 116, 3980, 116, 3980, 117, 3978, 118, 3978, 119, 3977, 119, 3977, 119, 3977, 120, 3975, 121, 3975, 122, 3974, 122, 3974, 123, 3972, 124, 3972, 124, 3972, 125, 3971, 125, 3971, 125, 3970, 127, 3969, 127, 3969, 127, 3969, 128, 3967, 129, 3967, 129, 3967, 130, 3965, 131, 3965, 132, 3964, 132, 3963, 134, 3962, 134, 3961, 135, 3961, 136, 3960, 136, 3960, 136, 3960, 136, 3959, 138, 3958, 138, 3958, 138, 3958, 139, 3957, 139, 3957, 139, 3957, 140, 3956, 140, 3956, 141, 3954, 142, 3954, 142, 3954, 143, 3953, 143, 3952, 144, 3952, 145, 3951, 145, 3950, 146, 3950, 147, 3949, 147, 3948, 148, 3948, 148, 3947, 150, 3946, 150, 3946, 150, 3945, 151, 3945, 152, 3944, 152, 3943, 153, 3943, 154, 3941, 155, 3941, 155, 3940, 157, 3939, 157, 3938, 158, 3938, 159, 3936, 160, 3936, 160, 3935, 162, 3934, 162, 3933, 163, 3933, 164, 3931, 165, 3931, 166, 3929, 167, 3929, 167, 3928, 169, 3927, 169, 3927, 169, 3926, 171, 3925, 171, 3925, 171, 3924, 173, 3923, 173, 3922, 174, 3922, 175, 3920, 176, 3920, 176, 3920, 177, 3919, 177, 3919, 178, 3918, 178, 3918, 178, 3917, 179, 3917, 179, 3917, 180, 3916, 180, 3915, 181, 3915, 181, 3915, 182, 3914, 182, 3913, 183, 3913, 184, 3911, 185, 3911, 185, 3911, 186, 3909, 187, 3909, 188, 3908, 188, 3907, 189, 3907, 190, 3905, 191, 3905, 191, 3904, 193, 3903, 193, 3903, 193, 3902, 195, 3901, 195, 3901, 195, 3901, 196, 3899, 197, 3899, 197, 3899, 198, 3897, 199, 3897, 199, 3897, 200, 3895, 201, 3895, 201, 3895, 201, 3894, 203, 3893, 203, 3893, 203, 3893, 203, 3892, 205, 3891, 205, 3891, 205, 3891, 206, 3889, 207, 3889, 207, 3889, 208, 3887, 209, 3887, 210, 3886, 210, 3885, 211, 3885, 211, 3884, 212, 3884, 213, 3882, 214, 3882, 214, 3881, 215, 3881, 216, 3880, 216, 3879, 217, 3879, 218, 3877, 219, 3877, 219, 3877, 220, 3875, 221, 3875, 221, 3875, 222, 3873, 223, 3873, 223, 3873, 224, 3871, 225, 3871, 225, 3871, 226, 3869, 227, 3869, 228, 3868, 228, 3867, 229, 3867, 230, 3865, 231, 3865, 231, 3865, 232, 3863, 233, 3863, 233, 3863, 234, 3861, 235, 3861, 235, 3860, 237, 3859, 237, 3859, 237, 3859, 238, 3857, 239, 3857, 240, 3856, 240, 3856, 241, 3854, 242, 3854, 242, 3854, 243, 3852, 244, 3852, 244, 3852, 244, 3851, 246, 3850, 246, 3849, 247, 3849, 248, 3847, 249, 3846, 250, 3846, 251, 3844, 252, 3844, 253, 3842, 254, 3842, 254, 3841, 256, 3840, 256, 3840, 256, 3839, 258, 3838, 258, 3838, 259, 3836, 260, 3836, 261, 3835, 261, 3835, 262, 3833, 263, 3833, 264, 3832, 264, 3832, 265, 3831, 265, 3831, 265, 3831, 265, 3831, 265, 3831, 266, 3829, 267, 3829, 267, 3828, 268, 3828, 269, 3826, 270, 3826, 270, 3825, 272, 3824, 272, 3824, 272, 3823, 274, 3822, 274, 3822, 275, 3820, 276, 3820, 277, 3819, 277, 3818, 279, 3817, 279, 3817, 280, 3815, 281, 3815, 281, 3815, 282, 3813, 283, 3813, 283, 3812, 285, 3811, 285, 3811, 285, 3810, 287, 3809, 287, 3809, 288, 3807, 289, 3807, 289, 3807, 290, 3805, 291, 3805, 291, 3805, 292, 3803, 293, 3803, 294, 3802, 294, 3801, 296, 3800, 296, 3799, 298, 3798, 298, 3797, 300, 3796, 300, 3795, 301, 3795, 302, 3793, 303, 3793, 303, 3792, 305, 3791, 305, 3791, 305, 3790, 307, 3789, 307, 3789, 308, 3787, 309, 3787, 309, 3786, 311, 3785, 311, 3785, 311, 3784, 313, 3783, 313, 3783, 314, 3781, 315, 3781, 316, 3780, 316, 3779, 318, 3778, 319, 3777, 319, 3776, 321, 3775, 322, 3774, 322, 3773, 324, 3772, 324, 3771, 326, 3770, 327, 3769, 327, 3769, 328, 3768, 328, 3767, 330, 3766, 330, 3766, 331, 3765, 332, 3763, 333, 3763, 334, 3762, 335, 3760, 337, 3759, 338, 3758, 338, 3757, 340, 3756, 340, 3755, 342, 3754, 342, 3754, 343, 3752, 344, 3752, 345, 3751, 345, 3750, 347, 3749, 348, 3747, 349, 3747, 350, 3745, 351, 3745, 352, 3743, 353, 3743, 354, 3741, 356, 3740, 356, 3739, 360, 3736, 362, 3734, 364, 3731, 367, 3729, 368, 3728, 370, 3725, 372, 3724, 373, 3722, 375, 3721, 376, 3719, 378, 3718, 379, 3716, 381, 3715, 382, 3714, 383, 3712, 385, 3711, 385, 3710, 387, 3709, 388, 3708, 388, 3707, 390, 3706, 390, 3706, 391, 3704, 392, 3704, 393, 3702, 394, 3702, 394, 3702, 395, 3700, 396, 3700, 396, 3700, 397, 3698, 398, 3698, 398, 3697, 399, 3697, 400, 3695, 401, 3695, 401, 3694, 402, 3694, 402, 3693, 403, 3693, 403, 3692, 404, 3692, 404, 3692, 404, 3691, 405, 3691, 405, 3691, 405, 3690, 406, 3690, 406, 3689, 407, 3689, 407, 3689, 406, 3690, 406, 3690, 406, 3689, 407, 3689, 407, 3689, 408, 3688, 408, 3687, 409, 3687, 409, 3687, 409, 3686, 410, 3686, 410, 3686, 410, 3685, 411, 3685, 411, 3684, 412, 3684, 412, 3683, 413, 3683, 413, 3682, 414, 3681, 415, 3681, 415, 3680, 416, 3680, 417, 3679, 417, 3679, 417, 3679, 417, 3679, 417, 3678, 418, 3678, 418, 3678, 418, 3678, 418, 3678, 419, 3677, 419, 3677, 419, 3677, 420, 3676, 420, 3676, 420, 3676, 420, 3676, 421, 3675, 421, 3675, 421, 3675, 421, 3675, 422, 3674, 422, 3674, 422, 3674, 422, 3674, 423, 3673, 423, 3673, 423, 3673, 423, 3673, 423, 3673, 423, 3673, 423, 3673, 423, 3673, 423, 3673, 423, 3673, 423, 3673, 423, 3673, 423, 3673, 423, 3673, 423, 3673, 423, 3673, 423, 3673, 423, 3673, 423, 3674, 422, 3674, 422, 3674, 422, 3674, 422, 3674, 422, 3674, 422, 3674, 422, 3674, 422, 3674, 422, 3674, 422, 3674, 422, 3674, 422, 3674, 422, 3674, 422, 3674, 422, 3674, 422, 3674, 422, 3674, 422, 3674, 422, 3674, 422, 3674, 422, 3674, 422, 3674, 422, 3674, 422, 3674, 422, 3674, 422, 3674, 421, 3675, 421, 3675, 421, 3675, 421, 3675, 421, 3675, 421, 3675, 421, 3675, 421, 3675, 421, 3675, 421, 3675, 421, 3675, 421, 3675, 421, 3675, 421, 3675, 421, 3675, 421, 3675, 421, 3675, 421, 3674, 422, 3674, 422, 3674, 422, 3674, 422, 3674, 422, 3674, 422, 3674, 422, 3674, 422, 3674, 422, 3674, 422, 3674, 422, 3674, 422, 3674, 422, 3674, 423, 3673, 423, 3673, 423, 3673, 423, 3673, 423, 3673, 423, 3673, 423, 3673, 423, 3673, 423, 3673, 424, 3672, 424, 3672, 424, 3672, 424, 3672, 424, 3672, 424, 3671, 425, 3671, 425, 3671, 425, 3671, 425, 3670, 426, 3670, 426, 3670, 426, 3670, 426, 3670, 426, 3670, 426, 3670, 426, 3670, 426, 3670, 427, 3669, 427, 3669, 427, 3669, 427, 3669, 427, 3669, 427, 3669, 427, 3669, 427, 3669, 427, 3669, 427, 3669, 427, 3670, 426, 3670, 426, 3670, 426, 3670, 426, 3671, 425, 3671, 425, 3671, 425, 3671, 425, 3671, 425, 3671, 425, 3671, 425, 3671, 425, 3671, 425, 3671, 425, 3671, 425, 3671, 425, 3671, 425, 3671, 425, 3671, 425, 3671, 425, 3671, 425, 3671, 425, 3672, 424, 3672, 424, 3672, 424, 3672, 424, 3672, 424, 3672, 424, 3672, 424, 3672, 424, 3672, 424, 3673, 423, 3673, 423, 3673, 423, 3673, 423, 3673, 423, 3674, 422, 3674, 422, 3674, 422, 3674, 422, 3675, 421, 3675, 421, 3675, 421, 3675, 421, 3675, 421, 3676, 420, 3676, 420, 3676, 420, 3676, 420, 3677, 419, 3677, 419, 3677, 419, 3677, 419, 3678, 418, 3678, 418, 3678, 418, 3678, 418, 3679, 417, 3679, 417, 3679, 417, 3679, 417, 3679, 417, 3680, 416, 3680, 416, 3680, 416, 3680, 416, 3681, 415, 3681, 415, 3681, 415, 3681, 415, 3681, 415, 3681, 415, 3681, 415, 3681, 415, 3681, 415, 3682, 414, 3682, 414, 3682, 414, 3682, 414, 3682, 414, 3682, 414, 3682, 414, 3682, 414, 3682, 414, 3683, 413, 3683, 413, 3683, 413, 3683, 413, 3683, 413, 3683, 413, 3683, 413, 3683, 413, 3683, 412, 3684, 411, 3685, 411, 3685, 410, 3686, 410, 3686, 409, 3687, 409, 3687, 408, 3688, 407, 3689, 407, 3689, 406, 3690, 406, 3690, 406, 3690, 406, 3690, 406, 3690, 406, 3690, 406, 3690, 406, 3690, 405, 3691, 405, 3691, 405, 3691, 405, 3691, 404, 3692, 404, 3692, 404, 3692, 403, 3693, 403, 3693, 403, 3693, 402, 3694, 402, 3694, 401, 3696, 400, 3696, 399, 3698, 398, 3698, 397, 3700, 395, 3701, 395, 3702, 393, 3703, 392, 3705, 390, 3707, 388, 3708, 387, 3710, 385, 3712, 383, 3713, 382, 3715, 381, 3715, 381, 3716, 379, 3717, 379, 3718, 378, 3719, 376, 3720, 376, 3721, 375, 3721, 374, 3723, 373, 3723, 372, 3725, 371, 3725, 370, 3727, 369, 3727, 368, 3729, 367, 3729, 366, 3731, 365, 3731, 364, 3733, 363, 3733, 363, 3734, 361, 3736, 360, 3737, 359, 3738, 357, 3740, 356, 3740, 355, 3742, 354, 3743, 353, 3743, 352, 3745, 351, 3745, 350, 3747, 349, 3747, 348, 3749, 347, 3749, 346, 3751, 345, 3752, 343, 3754, 342, 3755, 340, 3757, 339, 3758, 337, 3760, 336, 3761, 334, 3763, 333, 3763, 333, 3764, 331, 3766, 330, 3766, 330, 3767, 328, 3768, 328, 3769, 326, 3770, 326, 3771, 324, 3772, 324, 3772, 323, 3774, 322, 3774, 321, 3775, 321, 3776, 319, 3777, 319, 3777, 317, 3779, 317, 3780, 316, 3780, 316, 3780, 312, 3784, 312, 3784, 312, 3784, 312, 3784, 312, 3784, 312, 3784, 312, 3784, 312, 3784, 312, 3784, 312, 3784, 312, 3784, 312, 3784, 312, 3784, 311, 3785, 310, 3786, 310, 3786, 310, 3787, 308, 3788, 308, 3788, 307, 3789, 307, 3790, 305, 3791, 305, 3791, 304, 3792, 304, 3792, 303, 3793, 302, 3794, 302, 3794, 301, 3796, 299, 3797, 299, 3797, 298, 3798, 298, 3798, 297, 3800, 296, 3800, 295, 3801, 295, 3802, 293, 3803, 293, 3803, 292, 3805, 290, 3806, 289, 3807, 289, 3808, 287, 3809, 286, 3810, 286, 3810, 285, 3812, 284, 3812, 283, 3814, 282, 3814, 281, 3816, 280, 3816, 280, 3817, 278, 3818, 278, 3818, 277, 3820, 276, 3820, 275, 3821, 275, 3822, 273, 3823, 272, 3824, 272, 3824, 271, 3826, 270, 3826, 269, 3827, 268, 3829, 267, 3829, 266, 3831, 264, 3832, 264, 3833, 262, 3834, 262, 3835, 260, 3836, 260, 3837, 258, 3838, 258, 3839, 256, 3840, 256, 3841, 254, 3842, 254, 3843, 252, 3844, 252, 3845, 250, 3846, 250, 3847, 248, 3848, 248, 3849, 246, 3851, 245, 3851, 244, 3853, 243, 3853, 242, 3855, 241, 3855, 240, 3857, 239, 3857, 238, 3858, 238, 3859, 236, 3860, 236, 3860, 235, 3862, 234, 3862, 233, 3863, 233, 3864, 231, 3865, 231, 3865, 230, 3867, 229, 3867, 228, 3868, 228, 3869, 227, 3869, 226, 3871, 225, 3871, 225, 3872, 223, 3873, 223, 3874, 221, 3875, 221, 3876, 219, 3877, 218, 3879, 217, 3879, 216, 3881, 215, 3881, 214, 3883, 213, 3883, 212, 3885, 210, 3887, 209, 3887, 208, 3889, 207, 3890, 205, 3891, 204, 3893, 203, 3893, 202, 3895, 200, 3896, 200, 3897, 198, 3898, 197, 3899, 196, 3901, 195, 3901, 194, 3903, 192, 3905, 191, 3905, 190, 3907, 189, 3908, 187, 3909, 187, 3910, 185, 3911, 185, 3912, 183, 3913, 183, 3914, 181, 3915, 181, 3916, 179, 3917, 178, 3919, 177, 3919, 176, 3921, 174, 3922, 174, 3923, 172, 3924, 171, 3925, 170, 3927, 169, 3927, 168, 3928, 167, 3930, 166, 3930, 165, 3931, 165, 3932, 163, 3933, 163, 3933, 162, 3935, 161, 3935, 160, 3936, 159, 3938, 157, 3939, 156, 3941, 154, 3942, 154, 3943, 152, 3944, 151, 3946, 150, 3946, 149, 3948, 148, 3948, 147, 3950, 145, 3951, 145, 3952, 143, 3953, 142, 3955, 141, 3955, 140, 3957, 138, 3958, 137, 3960, 136, 3960, 135, 3962, 133, 3963, 133, 3964, 131, 3965, 131, 3966, 129, 3968, 127, 3969, 127, 3970, 125, 3972, 123, 3973, 123, 3974, 121, 3975, 121, 3976, 119, 3977, 119, 3978, 117, 3980, 116, 3980, 116, 3981, 114, 3983, 113, 3984, 112, 3985, 110, 3986, 110, 3987, 108, 3989, 107, 3989, 106, 3991, 105, 3991, 104, 3993, 102, 3994, 101, 3996, 100, 3996, 99, 3997, 98, 3999, 97, 3999, 96, 4000, 96, 4001, 94, 4002, 94, 4002, 93, 4003, 93, 4004, 91, 4005, 91, 4005, 90, 4006, 90, 4006, 89, 4007, 89, 4007, 88, 4008, 88, 4009, 86, 4011, 85, 4011, 84, 4013, 82, 4015, 80, 4017, 79, 4017, 78, 4019, 76, 4021, 75, 4021, 74, 4023, 73, 4024, 71, 4026, 69, 4027, 69, 4028, 67, 4030, 65, 4032, 64, 4033, 62, 4035, 61, 4036, 59, 4037, 59, 4038, 57, 4040, 55, 4041, 55, 4042, 53, 4043, 53, 4044, 51, 4045, 51, 4046, 49, 4047, 49, 4048, 47, 4049, 47, 4050, 45, 4051, 45, 4052, 43, 4053, 43, 4054, 41, 4055, 41, 4056, 40, 4056, 39, 4058, 38, 4059, 37, 4060, 35, 4061, 35, 4062, 33, 4064, 32, 4064, 32, 4065, 30, 4066, 30, 4067, 29, 4067, 28, 4069, 27, 4069, 27, 4070, 25, 4071, 25, 4072, 24, 4072, 23, 4074, 22, 4074, 22, 4075, 20, 4076, 20, 4077, 19, 4077, 19, 4078, 17, 4079, 17, 4080, 16, 4081, 15, 4081, 15, 4082, 14, 4083, 13, 4083, 13, 4084, 12, 4084, 11, 4086, 10, 4086, 10, 4087, 9, 4088, 8, 4088, 8, 4089, 7, 4089, 7, 4090, 6, 4091, 5, 4091, 5, 4092, 4, 4093, 3, 4093, 3, 4094, 2, 4094, 2, 4095, 1, 585728, 2, 4094, 3, 4093, 4, 4092, 6, 4090, 7, 4089, 9, 4087, 11, 4085, 13, 4082, 16, 4080, 18, 4077, 21, 4075, 22, 4073, 24, 4071, 26, 4070, 26, 4069, 28, 4067, 30, 4066, 31, 4065, 32, 4063, 34, 4062, 35, 4061, 35, 4060, 37, 4059, 37, 4058, 39, 4057, 39, 4056, 41, 4055, 42, 4053, 43, 4052, 45, 4051, 46, 4049, 48, 4047, 50, 4045, 52, 4043, 54, 4041, 56, 4039, 58, 4037, 60, 4035, 62, 4033, 65, 4030, 67, 4027, 70, 4025, 72, 4022, 75, 4019, 78, 4016, 82, 4011, 86, 4007, 91, 4001, 97, 3988, 109, 3986, 111, 3984, 113, 3982, 115, 3980, 117, 3978, 119, 3975, 122, 3973, 125, 3970, 127, 3968, 130, 3965, 132, 3963, 134, 3960, 138, 3957, 140, 3955, 142, 3953, 144, 3951, 146, 3949, 148, 3946, 152, 3943, 154, 3940, 157, 3938, 159, 3936, 161, 3933, 164, 3931, 166, 3929, 168, 3927, 170, 3925, 172, 3923, 174, 3920, 177, 3918, 180, 3915, 182, 3913, 185, 3910, 187, 3908, 189, 3906, 192, 3903, 194, 3901, 196, 3899, 198, 3897, 200, 3895, 202, 3892, 206, 3889, 208, 3886, 212, 3883, 214, 3881, 216, 3878, 220, 3875, 222, 3872, 226, 3869, 228, 3867, 230, 3864, 233, 3862, 236, 3858, 239, 3855, 243, 3851, 246, 3848, 249, 3845, 253, 3841, 256, 3838, 260, 3834, 264, 3830, 268, 3826, 271, 3823, 275, 3819, 279, 3815, 283, 3811, 288, 3806, 292, 3802, 296, 3798, 300, 3794, 305, 3789, 310, 3784, 314, 3780, 319, 3775, 324, 3770, 328, 3766, 333, 3761, 338, 3756, 343, 3751, 348, 3746, 353, 3741, 359, 3735, 365, 3729, 370, 3724, 375, 3719, 380, 3714, 384, 3710, 389, 3704, 395, 3698, 402, 3692, 409, 3685, 413, 3681, 417, 3677, 421, 3673, 425, 3668, 430, 3663, 436, 3658, 441, 3653, 446, 3648, 450, 3644, 454, 3641, 457, 3637, 461, 3634, 464, 3630, 468, 3626, 472, 3622, 476, 3617, 482, 3611, 488, 3604, 494, 3599, 500, 3594, 505, 3589, 510, 3583, 516, 3577, 522, 3572, 526, 3568, 531, 3563, 536, 3558, 541, 3555, 544, 3547, 552, 3541, 559, 3535, 566, 3527, 571, 3523, 576, 3518, 581, 3513, 587, 3507, 592, 3502, 596, 3498, 600, 3494, 605, 3488, 611, 3482, 618, 3476, 625, 3468, 632, 3461, 639, 3455, 644, 3450, 649, 3445, 654, 3440, 658, 3436, 663, 3431, 668, 3426, 672, 3422, 677, 3417, 681, 3412, 686, 3407, 691, 3401, 698, 3393, 706, 3386, 712, 3380, 718, 3374, 724, 3367, 731, 3361, 737, 3350, 749, 3342, 757, 3335, 765, 3326, 775, 3317, 783, 3309, 792, 3301, 797, 3296, 803, 3290, 809, 3283, 817, 3274, 827, 3265, 835, 3259, 842, 3252, 847, 3247, 852, 3241, 858, 3235, 865, 3227, 873, 3218, 881, 3213, 886, 3207, 892, 3201, 899, 3194, 911, 3182, 918, 3175, 924, 3170, 929, 3164, 935, 3158, 450] , dtype=np.uint32)
BODY_RLE = np.array( [593323, 35, 4052, 48, 4044, 57, 1137, 17, 2880, 66, 1129, 25, 773, 26, 2073, 73, 1123, 31, 761, 39, 2064, 81, 1117, 37, 754, 46, 1271, 17, 765, 91, 1113, 41, 747, 54, 1, 17, 1244, 27, 757, 97, 1109, 46, 740, 80, 1236, 34, 749, 104, 1094, 62, 734, 86, 1230, 40, 737, 114, 1089, 68, 729, 92, 1224, 45, 726, 125, 1084, 73, 725, 96, 1220, 50, 719, 130, 1080, 78, 720, 101, 1216, 55, 713, 134, 1077, 81, 716, 106, 1212, 59, 708, 138, 1074, 85, 711, 110, 1210, 63, 703, 141, 1071, 89, 706, 115, 1206, 68, 696, 146, 1069, 92, 701, 119, 1204, 71, 690, 151, 1066, 97, 696, 122, 1202, 74, 685, 155, 1064, 100, 692, 125, 1200, 77, 680, 159, 1062, 104, 687, 128, 1198, 81, 674, 163, 1060, 108, 683, 130, 1196, 84, 669, 167, 1058, 111, 679, 133, 1194, 87, 664, 170, 1057, 114, 675, 136, 1192, 91, 659, 173, 1055, 117, 671, 139, 1190, 94, 655, 176, 1053, 120, 666, 143, 1188, 97, 651, 178, 1052, 123, 662, 146, 1186, 102, 645, 181, 1050, 126, 657, 149, 1186, 106, 639, 183, 1049, 129, 652, 153, 1184, 112, 632, 186, 1048, 131, 648, 156, 1182, 117, 626, 188, 1047, 134, 644, 158, 1182, 121, 621, 190, 1045, 137, 640, 161, 1180, 126, 615, 192, 1045, 139, 637, 163, 1179, 131, 609, 193, 1044, 142, 633, 166, 1177, 136, 604, 195, 1043, 144, 630, 167, 1177, 139, 600, 196, 1042, 149, 625, 169, 1175, 143, 596, 197, 1042, 153, 620, 171, 1174, 145, 593, 199, 1040, 157, 616, 172, 1174, 147, 590, 200, 1039, 161, 612, 174, 1172, 150, 587, 202, 1038, 163, 609, 175, 1172, 152, 584, 203, 1037, 166, 606, 177, 1171, 154, 581, 205, 1036, 168, 603, 178, 1170, 157, 578, 206, 1035, 170, 601, 180, 1169, 159, 575, 208, 1034, 172, 598, 181, 1169, 161, 572, 209, 1033, 174, 597, 182, 1168, 163, 569, 211, 1032, 176, 594, 183, 1167, 167, 565, 213, 1030, 179, 591, 185, 1166, 170, 561, 214, 1030, 181, 589, 186, 1165, 172, 558, 216, 1028, 188, 582, 188, 1164, 174, 556, 217, 1022, 197, 578, 188, 1164, 176, 545, 226, 1022, 200, 574, 190, 1162, 178, 544, 227, 1019, 205, 571, 191, 1160, 181, 542, 228, 1018, 207, 568, 192, 1159, 184, 540, 229, 1017, 209, 566, 193, 1157, 187, 538, 229, 1017, 211, 564, 193, 1156, 191, 535, 230, 1016, 212, 562, 195, 1154, 194, 533, 231, 1015, 214, 560, 195, 1154, 196, 531, 231, 1015, 215, 559, 196, 1152, 199, 529, 232, 1014, 216, 557, 197, 1151, 201, 528, 232, 1014, 217, 556, 197, 1151, 203, 526, 233, 1013, 218, 555, 198, 1149, 206, 524, 233, 1013, 219, 550, 202, 1149, 208, 522, 234, 1012, 220, 545, 206, 1148, 210, 521, 235, 1011, 221, 541, 210, 1147, 212, 517, 237, 1010, 223, 537, 213, 1146, 214, 516, 238, 1009, 224, 534, 215, 1146, 215, 515, 239, 1007, 226, 531, 217, 1146, 217, 513, 240, 1006, 226, 529, 220, 1144, 219, 512, 241, 1004, 228, 527, 221, 1144, 221, 510, 241, 1004, 229, 524, 223, 1144, 222, 509, 242, 1003, 229, 522, 225, 1143, 224, 508, 243, 1001, 231, 519, 227, 1143, 225, 507, 244, 1000, 231, 517, 229, 1143, 226, 506, 245, 999, 232, 514, 231, 1143, 227, 505, 246, 997, 233, 512, 233, 1142, 229, 504, 246, 997, 234, 509, 235, 1142, 230, 503, 247, 996, 234, 508, 237, 1141, 231, 501, 249, 994, 235, 506, 239, 1141, 232, 500, 249, 994, 236, 504, 240, 1141, 233, 499, 250, 992, 237, 503, 241, 1141, 233, 498, 252, 991, 237, 502, 242, 1141, 234, 497, 253, 989, 239, 500, 243, 1140, 236, 496, 254, 988, 239, 499, 244, 1140, 236, 496, 254, 987, 240, 498, 245, 1140, 237, 494, 256, 986, 240, 497, 246, 1140, 237, 494, 257, 984, 242, 495, 247, 1140, 238, 493, 258, 982, 243, 494, 248, 1140, 238, 493, 259, 981, 243, 493, 249, 1140, 239, 492, 260, 979, 244, 493, 249, 1140, 239, 492, 260, 979, 244, 492, 250, 1140, 239, 492, 261, 977, 245, 491, 251, 1140, 240, 490, 263, 976, 245, 491, 251, 1140, 240, 490, 263, 975, 246, 490, 252, 1139, 241, 490, 264, 974, 246, 490, 252, 1139, 242, 488, 265, 973, 248, 488, 253, 1139, 242, 488, 266, 971, 249, 488, 253, 1139, 242, 488, 266, 970, 250, 487, 254, 1139, 242, 487, 268, 968, 251, 487, 254, 1138, 244, 486, 269, 966, 252, 486, 256, 1137, 244, 486, 269, 966, 252, 486, 256, 1136, 245, 486, 270, 964, 253, 485, 257, 1136, 245, 485, 272, 962, 254, 485, 257, 1136, 246, 484, 273, 961, 254, 484, 259, 1134, 247, 484, 274, 959, 255, 484, 259, 1134, 247, 484, 274, 959, 255, 484, 259, 1134, 247, 484, 275, 957, 256, 483, 260, 1133, 249, 483, 276, 956, 256, 483, 260, 1133, 249, 483, 276, 955, 257, 483, 260, 1133, 249, 483, 277, 954, 257, 482, 261, 1133, 249, 483, 278, 952, 258, 482, 261, 1132, 250, 483, 279, 951, 258, 482, 261, 1132, 250, 483, 279, 950, 258, 483, 262, 1131, 250, 483, 280, 948, 259, 482, 263, 1131, 250, 483, 281, 946, 260, 482, 263, 1131, 250, 483, 281, 946, 260, 482, 263, 1130, 251, 483, 282, 944, 260, 483, 263, 1130, 251, 483, 282, 943, 261, 483, 264, 1129, 251, 483, 283, 942, 261, 483, 264, 1129, 251, 483, 283, 941, 261, 484, 264, 1128, 252, 483, 284, 940, 261, 484, 264, 1128, 252, 483, 285, 938, 262, 484, 265, 1126, 253, 483, 285, 937, 263, 484, 265, 1126, 253, 483, 286, 936, 264, 483, 265, 1125, 253, 484, 286, 935, 265, 483, 265, 1125, 253, 484, 287, 933, 266, 483, 265, 1124, 254, 484, 287, 933, 266, 483, 265, 1124, 254, 484, 288, 931, 268, 482, 265, 1124, 253, 486, 287, 931, 268, 482, 265, 1123, 254, 486, 288, 929, 269, 482, 265, 1123, 254, 486, 288, 929, 269, 483, 264, 1123, 253, 487, 289, 927, 270, 483, 264, 1122, 254, 487, 289, 927, 270, 483, 264, 1122, 254, 487, 290, 926, 270, 483, 264, 1122, 253, 488, 290, 925, 271, 484, 263, 1121, 254, 488, 291, 924, 271, 484, 263, 1121, 253, 489, 291, 924, 271, 484, 263, 1121, 253, 490, 291, 922, 272, 485, 262, 1120, 254, 490, 291, 921, 273, 485, 262, 1120, 254, 490, 292, 920, 273, 485, 263, 1118, 255, 490, 292, 919, 274, 486, 262, 1118, 254, 492, 292, 917, 275, 486, 262, 1117, 255, 492, 292, 917, 275, 487, 261, 1117, 255, 492, 293, 915, 276, 487, 261, 1117, 255, 493, 292, 915, 275, 488, 262, 1115, 255, 494, 293, 913, 276, 488, 262, 1115, 255, 494, 293, 913, 276, 488, 262, 1113, 257, 495, 293, 911, 277, 489, 262, 1112, 256, 496, 293, 910, 277, 490, 263, 1111, 256, 496, 294, 909, 277, 490, 263, 1111, 256, 497, 293, 908, 278, 490, 264, 1109, 256, 498, 294, 906, 279, 490, 264, 1108, 257, 498, 294, 905, 280, 490, 265, 1107, 257, 499, 294, 904, 279, 491, 265, 1104, 1, 1, 258, 499, 294, 903, 280, 491, 266, 1103, 260, 499, 295, 901, 281, 491, 266, 1102, 261, 500, 294, 901, 281, 492, 266, 1101, 261, 500, 295, 899, 281, 493, 266, 1100, 262, 501, 294, 899, 281, 493, 267, 1099, 262, 501, 295, 897, 282, 493, 267, 1098, 263, 502, 295, 895, 282, 495, 267, 1097, 263, 502, 295, 895, 282, 495, 267, 1096, 264, 503, 295, 893, 283, 495, 268, 1095, 264, 503, 295, 892, 283, 496, 268, 1094, 265, 503, 296, 890, 284, 496, 269, 1092, 266, 503, 296, 889, 284, 498, 268, 1092, 266, 504, 296, 887, 285, 498, 269, 1090, 266, 505, 296, 887, 284, 499, 269, 1090, 266, 505, 296, 886, 285, 499, 270, 1088, 267, 506, 296, 884, 285, 501, 269, 1088, 267, 506, 296, 884, 285, 501, 270, 1086, 267, 507, 297, 883, 284, 502, 271, 1084, 268, 508, 297, 882, 284, 502, 271, 1084, 268, 508, 298, 881, 284, 503, 271, 1082, 268, 510, 297, 881, 283, 504, 271, 1081, 269, 510, 298, 880, 283, 504, 272, 1080, 269, 511, 298, 878, 284, 504, 273, 1078, 269, 512, 298, 877, 284, 506, 272, 1078, 269, 513, 298, 875, 285, 506, 273, 1076, 270, 514, 297, 875, 285, 506, 274, 1075, 270, 514, 298, 873, 285, 507, 275, 1073, 271, 515, 297, 872, 286, 507, 276, 1071, 272, 516, 297, 870, 286, 509, 276, 1070, 272, 517, 297, 868, 287, 509, 277, 1068, 273, 517, 297, 867, 288, 509, 277, 1067, 273, 519, 297, 866, 287, 510, 278, 1065, 274, 520, 296, 865, 288, 511, 278, 1063, 275, 520, 297, 864, 288, 511, 278, 1063, 275, 521, 297, 862, 288, 512, 279, 1061, 275, 523, 296, 861, 289, 512, 280, 1059, 276, 524, 296, 860, 288, 514, 280, 1058, 276, 524, 297, 858, 289, 514, 281, 1056, 277, 525, 296, 857, 289, 515, 282, 1055, 276, 526, 297, 856, 289, 515, 283, 1053, 277, 527, 296, 855, 290, 515, 284, 1052, 277, 527, 297, 853, 290, 516, 285, 1050, 278, 528, 296, 852, 291, 516, 286, 1048, 279, 528, 297, 850, 291, 517, 286, 1048, 279, 529, 297, 849, 291, 517, 287, 1046, 280, 529, 297, 849, 290, 519, 287, 1044, 281, 530, 297, 847, 291, 519, 287, 1043, 282, 531, 297, 845, 291, 520, 288, 1041, 282, 532, 298, 844, 291, 520, 289, 1040, 282, 533, 298, 842, 291, 522, 289, 1039, 282, 533, 298, 841, 292, 522, 289, 1038, 283, 534, 298, 840, 291, 523, 290, 1037, 282, 535, 299, 838, 292, 523, 291, 1036, 282, 536, 298, 838, 292, 523, 292, 1034, 283, 536, 299, 836, 292, 525, 292, 1033, 282, 538, 298, 835, 293, 525, 293, 1031, 283, 538, 299, 835, 292, 525, 294, 1030, 283, 539, 298, 835, 291, 526, 294, 1029, 283, 540, 299, 834, 291, 527, 294, 1028, 283, 540, 300, 833, 290, 528, 295, 1026, 284, 541, 299, 833, 290, 528, 295, 1025, 285, 541, 300, 832, 289, 530, 295, 1024, 285, 541, 301, 831, 289, 530, 296, 1022, 286, 542, 301, 829, 289, 531, 297, 1020, 287, 542, 301, 828, 289, 533, 297, 1019, 286, 544, 301, 827, 289, 533, 298, 1017, 287, 544, 302, 826, 288, 535, 298, 1016, 287, 545, 301, 825, 288, 536, 299, 1014, 288, 545, 302, 823, 288, 538, 299, 1012, 288, 547, 301, 822, 289, 538, 300, 1011, 288, 548, 301, 821, 288, 540, 300, 1009, 289, 548, 302, 819, 289, 541, 299, 1008, 290, 549, 301, 819, 288, 542, 301, 1005, 291, 550, 301, 818, 288, 543, 301, 1003, 291, 551, 302, 816, 288, 545, 301, 1001, 292, 551, 302, 815, 289, 546, 301, 1000, 292, 552, 302, 814, 288, 547, 302, 998, 293, 552, 302, 813, 289, 548, 302, 997, 292, 553, 303, 812, 288, 550, 302, 996, 292, 554, 302, 811, 289, 550, 303, 994, 293, 554, 303, 809, 289, 552, 303, 993, 293, 555, 303, 808, 289, 553, 303, 992, 293, 555, 303, 808, 288, 555, 303, 990, 293, 557, 303, 806, 289, 555, 303, 990, 293, 557, 304, 804, 289, 557, 303, 988, 294, 558, 304, 803, 289, 558, 304, 986, 294, 558, 304, 803, 288, 559, 305, 984, 294, 560, 304, 802, 288, 560, 305, 983, 294, 560, 305, 800, 288, 562, 305, 981, 295, 560, 305, 800, 288, 562, 306, 979, 296, 561, 305, 798, 288, 564, 306, 978, 296, 561, 305, 797, 289, 564, 307, 976, 296, 562, 306, 796, 288, 566, 307, 974, 297, 563, 305, 795, 289, 567, 307, 972, 298, 563, 306, 794, 288, 568, 308, 970, 299, 563, 306, 793, 288, 570, 308, 968, 299, 565, 306, 792, 288, 571, 308, 966, 300, 565, 307, 790, 288, 572, 309, 964, 301, 566, 306, 789, 288, 574, 309, 962, 302, 566, 307, 788, 288, 574, 310, 960, 303, 567, 307, 786, 288, 575, 311, 958, 303, 568, 307, 786, 288, 576, 311, 956, 304, 569, 307, 784, 288, 577, 312, 954, 305, 570, 307, 782, 289, 578, 312, 952, 306, 570, 307, 782, 288, 579, 313, 950, 306, 571, 308, 780, 289, 580, 313, 948, 307, 572, 308, 779, 288, 581, 314, 946, 308, 572, 309, 777, 289, 581, 316, 942, 310, 572, 309, 776, 289, 583, 316, 940, 310, 574, 309, 775, 289, 583, 318, 936, 312, 574, 310, 773, 289, 584, 320, 933, 313, 574, 310, 773, 289, 584, 322, 930, 314, 575, 310, 771, 289, 586, 322, 928, 314, 576, 311, 769, 290, 586, 324, 924, 316, 577, 311, 767, 290, 587, 325, 922, 317, 577, 311, 766, 291, 587, 326, 919, 319, 578, 311, 764, 291, 588, 327, 916, 321, 578, 312, 763, 291, 589, 327, 913, 322, 580, 311, 762, 291, 590, 328, 911, 323, 580, 312, 760, 291, 591, 330, 907, 325, 581, 312, 759, 291, 591, 331, 904, 327, 581, 313, 758, 290, 593, 332, 900, 328, 583, 312, 757, 291, 593, 334, 896, 330, 583, 313, 755, 292, 593, 335, 893, 332, 584, 313, 753, 292, 595, 336, 889, 333, 585, 313, 752, 293, 595, 337, 886, 335, 586, 313, 751, 293, 595, 338, 883, 337, 586, 313, 750, 293, 597, 339, 879, 338, 588, 313, 748, 294, 597, 340, 876, 340, 588, 314, 747, 293, 599, 341, 872, 341, 590, 313, 746, 294, 599, 343, 868, 343, 590, 314, 744, 294, 600, 345, 864, 345, 591, 314, 743, 294, 600, 346, 861, 347, 591, 315, 741, 294, 602, 347, 858, 348, 592, 315, 739, 295, 602, 348, 855, 349, 594, 315, 737, 295, 603, 350, 852, 350, 594, 315, 736, 296, 603, 352, 848, 352, 595, 315, 735, 295, 604, 353, 845, 354, 595, 316, 733, 296, 604, 355, 841, 355, 597, 316, 732, 295, 605, 357, 837, 357, 597, 317, 730, 296, 605, 359, 833, 359, 598, 317, 728, 296, 606, 361, 829, 361, 598, 318, 727, 296, 606, 364, 825, 362, 599, 317, 727, 295, 607, 367, 820, 363, 600, 318, 725, 296, 607, 370, 816, 364, 601, 319, 722, 296, 608, 372, 813, 365, 601, 320, 720, 297, 608, 374, 809, 367, 602, 320, 719, 296, 609, 376, 805, 368, 603, 321, 717, 297, 609, 377, 802, 370, 604, 321, 716, 296, 610, 379, 798, 372, 605, 321, 714, 297, 612, 378, 791, 378, 605, 322, 712, 297, 613, 381, 787, 378, 607, 322, 711, 297, 615, 382, 781, 381, 607, 323, 710, 296, 616, 384, 777, 383, 607, 325, 707, 297, 616, 386, 772, 386, 608, 325, 705, 298, 617, 387, 767, 389, 608, 327, 703, 297, 618, 389, 764, 390, 608, 328, 701, 298, 618, 391, 762, 390, 609, 328, 699, 299, 618, 394, 755, 394, 609, 329, 698, 298, 619, 397, 752, 394, 610, 329, 696, 299, 620, 398, 748, 395, 611, 330, 695, 298, 621, 400, 746, 395, 612, 330, 693, 299, 621, 402, 743, 396, 612, 331, 692, 298, 622, 404, 737, 400, 613, 331, 690, 299, 622, 406, 732, 402, 614, 332, 688, 299, 623, 408, 727, 405, 615, 332, 687, 299, 623, 410, 722, 408, 615, 333, 685, 299, 624, 412, 717, 411, 616, 333, 684, 299, 624, 414, 712, 414, 616, 334, 682, 299, 626, 416, 707, 415, 618, 334, 680, 300, 626, 419, 702, 417, 618, 335, 678, 300, 627, 421, 698, 419, 619, 335, 677, 300, 627, 423, 694, 421, 620, 335, 675, 300, 629, 424, 690, 422, 621, 336, 673, 301, 629, 426, 684, 426, 622, 336, 671, 301, 630, 429, 677, 430, 623, 336, 669, 302, 630, 433, 670, 432, 624, 337, 668, 301, 632, 436, 663, 435, 624, 338, 666, 302, 632, 441, 655, 438, 625, 338, 664, 302, 633, 450, 643, 440, 626, 339, 663, 302, 633, 454, 636, 443, 626, 340, 661, 302, 634, 458, 630, 444, 628, 340, 659, 303, 634, 473, 608, 451, 628, 341, 656, 305, 634, 473, 608, 451, 628, 342, 654, 306, 634, 475, 604, 453, 629, 342, 653, 305, 635, 477, 600, 455, 629, 344, 650, 306, 635, 490, 584, 457, 630, 345, 648, 307, 635, 499, 572, 460, 631, 345, 645, 308, 636, 503, 565, 463, 631, 346, 644, 308, 636, 506, 559, 466, 631, 347, 642, 309, 636, 509, 553, 469, 632, 347, 640, 309, 637, 511, 548, 472, 632, 349, 637, 310, 637, 513, 543, 475, 633, 349, 635, 311, 637, 515, 539, 477, 633, 350, 633, 311, 638, 516, 503, 17, 16, 479, 633, 351, 631, 312, 639, 517, 497, 25, 10, 480, 634, 353, 629, 312, 639, 519, 490, 33, 3, 484, 634, 354, 611, 1, 15, 313, 639, 521, 484, 524, 634, 356, 624, 313, 640, 523, 478, 528, 634, 357, 621, 315, 640, 525, 472, 531, 635, 358, 619, 316, 640, 527, 467, 534, 635, 359, 617, 317, 640, 529, 462, 537, 635, 360, 615, 317, 641, 533, 455, 540, 635, 362, 612, 318, 641, 536, 449, 543, 635, 363, 610, 319, 641, 539, 443, 546, 635, 364, 608, 319, 643, 540, 439, 548, 635, 366, 605, 320, 643, 543, 433, 551, 635, 367, 602, 322, 643, 546, 427, 554, 636, 368, 599, 322, 644, 548, 423, 556, 636, 369, 597, 323, 645, 552, 416, 558, 636, 370, 595, 324, 645, 556, 410, 560, 636, 371, 592, 325, 646, 561, 404, 561, 637, 371, 590, 326, 647, 564, 398, 563, 637, 372, 587, 328, 647, 578, 383, 563, 638, 374, 584, 329, 647, 582, 367, 575, 638, 375, 581, 330, 649, 595, 353, 575, 639, 375, 577, 333, 649, 599, 347, 577, 639, 376, 575, 334, 649, 604, 342, 576, 640, 377, 573, 335, 649, 613, 333, 576, 640, 379, 569, 336, 650, 622, 324, 576, 640, 380, 566, 338, 650, 631, 313, 577, 641, 381, 563, 340, 650, 635, 304, 582, 641, 382, 562, 340, 650, 640, 295, 586, 641, 384, 559, 341, 650, 649, 282, 589, 642, 385, 557, 341, 651, 653, 274, 593, 642, 387, 554, 342, 651, 657, 265, 597, 643, 389, 548, 346, 651, 667, 246, 606, 643, 391, 540, 352, 651, 676, 233, 610, 643, 392, 538, 352, 652, 685, 219, 614, 644, 394, 528, 360, 652, 689, 206, 623, 644, 395, 523, 364, 652, 702, 184, 632, 644, 398, 515, 369, 652, 706, 171, 640, 645, 401, 511, 370, 652, 1517, 645, 403, 508, 371, 652, 1517, 645, 405, 504, 373, 652, 1517, 645, 407, 500, 375, 652, 1517, 645, 408, 495, 379, 652, 1517, 645, 410, 489, 383, 652, 1517, 645, 411, 481, 390, 652, 1516, 646, 413, 478, 391, 652, 1516, 646, 415, 474, 393, 652, 1516, 646, 416, 458, 6, 8, 394, 652, 1516, 646, 418, 452, 412, 652, 1515, 647, 419, 448, 415, 652, 1515, 646, 421, 444, 417, 653, 1515, 646, 422, 441, 419, 653, 1514, 647, 423, 437, 422, 653, 1514, 647, 424, 433, 425, 653, 1514, 647, 425, 430, 427, 653, 1513, 648, 426, 427, 429, 653, 1513, 648, 427, 424, 431, 653, 1513, 648, 428, 421, 433, 653, 1513, 648, 429, 417, 436, 653, 1513, 647, 431, 413, 439, 653, 1513, 647, 434, 407, 442, 653, 1513, 647, 437, 402, 444, 653, 1512, 648, 440, 395, 448, 653, 1512, 648, 443, 388, 452, 653, 1512, 648, 446, 382, 455, 653, 1512, 648, 448, 377, 458, 653, 1511, 649, 451, 372, 459, 654, 1511, 649, 454, 367, 461, 654, 1511, 649, 456, 363, 463, 654, 1510, 650, 458, 358, 466, 654, 1510, 650, 460, 354, 468, 654, 1510, 650, 463, 348, 471, 654, 1509, 651, 465, 343, 474, 654, 1509, 651, 468, 338, 476, 654, 1509, 651, 471, 332, 479, 654, 1509, 651, 473, 327, 482, 654, 1509, 651, 478, 319, 485, 654, 1509, 651, 482, 312, 488, 654, 1509, 651, 485, 306, 491, 654, 1508, 652, 488, 301, 493, 654, 1508, 652, 490, 297, 495, 654, 1508, 652, 493, 292, 497, 654, 1508, 652, 496, 287, 499, 653, 1509, 652, 498, 283, 501, 653, 1509, 652, 500, 276, 505, 654, 1509, 652, 502, 271, 508, 653, 1510, 652, 504, 266, 511, 653, 1510, 652, 508, 259, 514, 653, 1509, 653, 512, 253, 516, 652, 1510, 653, 516, 245, 520, 652, 1510, 653, 519, 238, 524, 652, 1510, 653, 522, 230, 529, 652, 1510, 653, 524, 224, 533, 651, 1511, 653, 528, 216, 537, 651, 1511, 653, 532, 208, 541, 651, 1511, 653, 535, 202, 544, 651, 1511, 652, 539, 196, 547, 651, 1511, 652, 542, 185, 555, 651, 1511, 652, 545, 178, 559, 651, 1511, 652, 548, 172, 562, 651, 1511, 652, 550, 167, 565, 650, 1512, 651, 554, 162, 567, 650, 1512, 651, 557, 157, 569, 650, 1512, 651, 560, 152, 571, 650, 1512, 651, 563, 147, 573, 650, 1512, 650, 567, 137, 580, 650, 1512, 650, 570, 130, 585, 649, 1513, 649, 575, 122, 588, 649, 1513, 649, 579, 115, 591, 649, 1513, 649, 583, 109, 594, 648, 1513, 649, 592, 97, 597, 647, 1515, 648, 596, 90, 600, 647, 1515, 648, 599, 85, 602, 647, 1515, 648, 602, 79, 606, 645, 1516, 648, 605, 70, 612, 645, 1517, 647, 609, 62, 616, 645, 1517, 647, 618, 50, 619, 645, 1517, 647, 622, 43, 622, 644, 1519, 646, 625, 38, 624, 644, 1519, 646, 628, 33, 626, 644, 1519, 646, 630, 29, 628, 644, 1520, 645, 632, 26, 629, 644, 1520, 645, 634, 22, 631, 644, 1521, 643, 636, 20, 632, 644, 1522, 642, 638, 17, 633, 644, 1522, 642, 639, 15, 634, 644, 1523, 641, 640, 13, 635, 643, 1525, 640, 641, 11, 636, 643, 1525, 640, 642, 9, 637, 643, 1526, 639, 643, 7, 638, 643, 1526, 639, 644, 5, 640, 642, 1527, 637, 646, 3, 641, 642, 1528, 636, 647, 2, 641, 642, 1528, 636, 648, 1, 641, 641, 1530, 634, 1291, 641, 1531, 633, 1291, 641, 1531, 633, 1291, 640, 1533, 631, 1292, 640, 1533, 631, 1292, 640, 1534, 630, 1293, 638, 1535, 630, 1293, 638, 1536, 628, 1294, 637, 1537, 628, 1294, 637, 1538, 627, 1294, 636, 1539, 627, 1294, 636, 1540, 626, 1294, 635, 1541, 626, 1294, 635, 1542, 625, 1294, 634, 1543, 625, 1294, 634, 1544, 623, 1295, 633, 1545, 623, 1295, 633, 1545, 623, 1295, 633, 1546, 622, 1295, 632, 1547, 622, 1295, 632, 1547, 622, 1295, 632, 1548, 621, 1295, 631, 1549, 621, 1295, 631, 1550, 620, 1295, 630, 1551, 619, 1296, 630, 1551, 619, 1296, 630, 1552, 618, 1296, 629, 1553, 618, 1296, 629, 1553, 618, 1297, 627, 1555, 616, 1298, 627, 1555, 616, 1298, 626, 1556, 616, 1298, 626, 1556, 616, 1299, 624, 1558, 614, 1300, 624, 1558, 614, 1300, 624, 1558, 614, 1300, 623, 1559, 614, 1300, 623, 1559, 614, 1300, 623, 1560, 612, 1301, 622, 1561, 612, 1301, 622, 1562, 611, 1301, 621, 1563, 611, 1301, 621, 1563, 610, 1302, 621, 1564, 609, 1302, 620, 1565, 609, 1302, 620, 1565, 609, 1302, 620, 1566, 608, 1302, 619, 1567, 607, 1303, 619, 1567, 607, 1304, 617, 1568, 607, 1305, 616, 1569, 606, 1306, 614, 1570, 605, 1308, 613, 1570, 605, 1309, 611, 1571, 605, 1310, 610, 1571, 604, 1311, 609, 1573, 603, 1312, 608, 1573, 602, 1314, 606, 1575, 601, 1314, 606, 1575, 600, 1316, 605, 1575, 600, 1316, 604, 1577, 599, 1317, 603, 1577, 598, 1318, 603, 1577, 598, 1319, 601, 1579, 597, 1320, 600, 1579, 596, 1322, 599, 1579, 596, 1323, 597, 1580, 596, 1324, 596, 1581, 595, 1325, 595, 1581, 594, 1327, 593, 1582, 594, 1328, 592, 1582, 594, 1329, 591, 1582, 593, 1331, 589, 1584, 592, 1331, 589, 1584, 591, 1333, 587, 1586, 590, 1334, 586, 1586, 589, 1336, 584, 1588, 588, 1337, 583, 1589, 587, 1337, 582, 1591, 585, 1339, 581, 1592, 584, 1340, 579, 1594, 583, 670, 2, 668, 578, 1596, 581, 671, 2, 669, 577, 1597, 580, 671, 2, 669, 575, 1600, 578, 672, 2, 670, 574, 1601, 576, 673, 2, 670, 574, 1602, 575, 673, 2, 671, 573, 1603, 573, 674, 2, 671, 573, 1604, 572, 674, 2, 671, 573, 1605, 570, 675, 2, 672, 571, 1607, 569, 674, 3, 672, 571, 1608, 567, 675, 3, 672, 570, 1610, 566, 675, 3, 673, 569, 1611, 565, 675, 3, 673, 568, 1613, 563, 675, 4, 673, 568, 1614, 562, 675, 4, 673, 567, 1616, 561, 675, 4, 674, 566, 1617, 559, 676, 4, 674, 565, 1619, 558, 676, 4, 674, 565, 1620, 557, 675, 5, 674, 564, 1621, 556, 676, 5, 674, 564, 1622, 555, 676, 6, 674, 562, 1624, 553, 677, 6, 674, 562, 1624, 553, 676, 7, 674, 561, 1626, 551, 677, 7, 674, 561, 1626, 551, 677, 7, 675, 559, 1628, 549, 678, 7, 675, 559, 1628, 549, 677, 8, 675, 558, 1630, 547, 678, 8, 675, 558, 1631, 546, 678, 8, 675, 557, 1632, 546, 678, 9, 675, 556, 1633, 544, 679, 9, 676, 554, 1635, 543, 679, 9, 676, 553, 1637, 542, 679, 9, 677, 552, 1638, 540, 680, 10, 677, 550, 1639, 540, 680, 10, 678, 548, 1641, 538, 681, 10, 679, 547, 1642, 537, 680, 11, 680, 545, 1643, 537, 680, 11, 681, 543, 1645, 535, 681, 12, 680, 542, 1647, 534, 681, 12, 682, 539, 1649, 533, 680, 13, 683, 538, 1649, 532, 681, 13, 684, 536, 1651, 531, 681, 14, 684, 534, 1653, 529, 681, 15, 685, 533, 1653, 529, 681, 15, 686, 531, 1655, 528, 681, 16, 687, 529, 1656, 526, 681, 17, 688, 527, 1658, 525, 681, 17, 689, 526, 1658, 525, 681, 18, 689, 524, 1660, 523, 682, 18, 690, 522, 1662, 522, 682, 19, 690, 520, 1663, 521, 683, 19, 691, 518, 1665, 520, 683, 19, 692, 516, 1666, 519, 684, 20, 692, 514, 1668, 518, 684, 20, 693, 512, 1670, 516, 685, 20, 694, 510, 1671, 515, 686, 21, 693, 509, 1673, 513, 687, 21, 694, 507, 1675, 511, 688, 22, 694, 505, 1676, 510, 689, 22, 694, 504, 1678, 508, 690, 22, 695, 502, 1680, 506, 691, 23, 694, 502, 1681, 504, 691, 24, 695, 500, 1683, 502, 692, 24, 695, 499, 1685, 500, 693, 25, 695, 498, 1686, 498, 694, 25, 695, 497, 1688, 497, 694, 26, 695, 495, 1690, 495, 695, 26, 695, 494, 1692, 493, 696, 26, 696, 492, 1693, 493, 696, 26, 696, 491, 1695, 491, 697, 26, 697, 489, 1697, 490, 696, 28, 696, 488, 1699, 488, 697, 28, 697, 486, 1701, 487, 697, 28, 698, 484, 1703, 485, 698, 28, 698, 484, 1704, 484, 697, 29, 699, 482, 1705, 484, 697, 29, 700, 480, 1707, 482, 698, 29, 701, 478, 1709, 481, 698, 29, 701, 477, 1711, 480, 697, 30, 702, 475, 1713, 478, 698, 31, 702, 473, 1715, 477, 698, 31, 702, 473, 1716, 476, 698, 31, 703, 471, 1717, 476, 698, 31, 704, 469, 1719, 474, 699, 32, 704, 468, 1720, 473, 699, 32, 705, 466, 1722, 472, 699, 32, 705, 465, 1724, 471, 699, 32, 706, 463, 1726, 470, 698, 33, 707, 460, 1729, 468, 699, 33, 707, 459, 1731, 466, 700, 33, 708, 457, 1733, 465, 700, 33, 709, 455, 1735, 463, 700, 34, 710, 453, 1737, 462, 700, 34, 710, 452, 1739, 460, 701, 34, 711, 450, 1741, 458, 702, 34, 712, 448, 1742, 458, 702, 34, 712, 447, 1744, 456, 702, 35, 713, 445, 1746, 454, 703, 36, 713, 443, 1748, 453, 703, 36, 714, 441, 1750, 451, 704, 36, 714, 440, 1752, 449, 704, 37, 715, 438, 1754, 447, 705, 37, 716, 436, 1755, 447, 705, 37, 717, 434, 1757, 445, 705, 38, 718, 432, 1759, 443, 706, 38, 719, 430, 1761, 442, 706, 38, 720, 429, 1762, 440, 706, 40, 719, 428, 1764, 438, 707, 40, 720, 426, 1765, 437, 707, 41, 721, 424, 1767, 436, 707, 41, 722, 422, 1769, 434, 708, 42, 722, 420, 1771, 432, 709, 42, 723, 419, 1772, 431, 709, 42, 724, 417, 1774, 429, 709, 43, 725, 415, 1776, 427, 710, 44, 725, 414, 1776, 426, 711, 44, 726, 412, 1778, 425, 711, 44, 727, 410, 1780, 423, 711, 45, 728, 408, 1781, 422, 712, 46, 728, 406, 1783, 421, 712, 46, 728, 405, 1785, 419, 712, 47, 729, 403, 1787, 417, 713, 48, 729, 401, 1789, 415, 714, 48, 730, 399, 1791, 413, 714, 49, 731, 397, 1793, 411, 715, 49, 732, 395, 1795, 409, 715, 50, 733, 393, 1797, 407, 716, 50, 734, 390, 1800, 405, 716, 51, 735, 388, 1801, 404, 717, 51, 736, 386, 1803, 402, 717, 52, 737, 384, 1805, 400, 718, 52, 738, 382, 1806, 399, 718, 54, 738, 380, 1808, 397, 719, 54, 738, 379, 1810, 395, 720, 54, 739, 377, 1812, 393, 720, 55, 740, 375, 1813, 392, 721, 56, 740, 373, 1815, 391, 721, 56, 741, 371, 1817, 389, 722, 56, 742, 370, 1817, 388, 722, 57, 742, 368, 1820, 387, 722, 57, 743, 366, 1822, 385, 723, 58, 743, 364, 1824, 383, 724, 58, 744, 362, 1825, 382, 725, 58, 745, 360, 1827, 381, 725, 58, 746, 358, 1829, 379, 726, 59, 746, 356, 1831, 377, 727, 59, 747, 354, 1833, 376, 727, 59, 748, 352, 1835, 374, 727, 60, 749, 350, 1837, 372, 728, 60, 750, 348, 1838, 371, 729, 61, 750, 347, 1839, 370, 729, 61, 751, 345, 1841, 368, 729, 62, 751, 343, 1844, 366, 730, 62, 752, 341, 1846, 365, 730, 62, 753, 339, 1848, 363, 731, 62, 753, 338, 1850, 361, 732, 62, 754, 336, 1851, 360, 733, 62, 755, 334, 1853, 358, 734, 64, 754, 332, 1855, 356, 735, 64, 754, 331, 1856, 355, 736, 65, 754, 329, 1860, 351, 734, 68, 755, 327, 1861, 350, 734, 70, 754, 326, 1862, 349, 735, 70, 755, 324, 1864, 347, 736, 70, 756, 322, 1866, 345, 736, 71, 757, 320, 1868, 343, 737, 71, 757, 319, 1870, 342, 737, 72, 757, 317, 1872, 340, 738, 72, 758, 315, 1874, 338, 738, 73, 759, 314, 1874, 338, 738, 73, 760, 312, 1876, 336, 739, 74, 760, 310, 1879, 333, 740, 74, 761, 308, 1881, 331, 741, 74, 762, 306, 1883, 329, 741, 76, 762, 304, 1885, 327, 742, 76, 763, 302, 1887, 325, 743, 76, 764, 299, 1890, 323, 744, 77, 764, 297, 1892, 321, 744, 78, 765, 295, 1894, 319, 745, 78, 765, 294, 1896, 318, 745, 78, 766, 292, 1898, 315, 747, 78, 767, 290, 1900, 313, 747, 79, 768, 288, 1902, 311, 748, 79, 769, 286, 1905, 308, 749, 80, 769, 284, 1907, 306, 750, 80, 769, 283, 1909, 304, 750, 81, 770, 281, 1911, 302, 751, 81, 771, 280, 1912, 300, 752, 82, 770, 278, 1916, 297, 753, 82, 771, 276, 1918, 295, 754, 82, 772, 274, 1920, 292, 755, 84, 772, 272, 1922, 290, 756, 84, 773, 270, 1925, 286, 758, 84, 773, 269, 1927, 284, 759, 85, 773, 266, 1931, 281, 759, 86, 774, 264, 1933, 279, 760, 87, 773, 262, 1936, 277, 761, 87, 774, 259, 1940, 274, 762, 87, 775, 257, 1943, 271, 762, 89, 775, 254, 1947, 268, 763, 89, 775, 253, 1950, 265, 764, 89, 776, 251, 1953, 261, 766, 90, 776, 249, 1956, 258, 766, 91, 777, 247, 1959, 254, 768, 92, 777, 245, 1962, 251, 769, 92, 778, 242, 1966, 248, 770, 92, 779, 240, 1969, 244, 772, 92, 780, 238, 1972, 241, 772, 93, 781, 235, 1976, 237, 774, 94, 781, 232, 1980, 234, 775, 94, 782, 213, 1999, 232, 776, 94, 783, 208, 2005, 229, 777, 94, 784, 202, 2011, 227, 778, 95, 783, 198, 2016, 225, 779, 95, 784, 194, 2020, 223, 780, 95, 785, 190, 2024, 221, 781, 96, 785, 187, 2027, 219, 781, 97, 786, 184, 2031, 216, 782, 97, 787, 181, 2036, 212, 783, 98, 787, 179, 2040, 209, 783, 98, 788, 176, 2044, 206, 783, 99, 789, 174, 2047, 203, 784, 100, 789, 172, 2050, 6, 4, 190, 785, 100, 790, 170, 2053, 4, 4, 189, 786, 100, 791, 168, 2064, 186, 787, 100, 792, 166, 2065, 185, 787, 102, 792, 164, 2066, 184, 788, 102, 794, 161, 2068, 181, 790, 102, 795, 159, 2072, 177, 791, 102, 796, 157, 2076, 173, 791, 104, 796, 155, 2079, 169, 793, 104, 797, 154, 2084, 163, 794, 104, 798, 151, 2095, 152, 796, 105, 799, 148, 2100, 147, 797, 105, 800, 146, 2104, 143, 797, 106, 801, 144, 2108, 138, 799, 107, 802, 141, 2111, 135, 800, 107, 803, 139, 2115, 130, 802, 108, 804, 136, 2119, 126, 803, 108, 805, 134, 2122, 123, 804, 108, 807, 130, 2127, 118, 806, 109, 808, 126, 2132, 114, 807, 109, 810, 122, 2136, 110, 808, 110, 812, 118, 2140, 107, 809, 111, 813, 114, 2144, 104, 810, 111, 815, 110, 2149, 100, 811, 111, 816, 108, 2152, 97, 811, 112, 824, 97, 2158, 93, 812, 112, 842, 76, 2164, 88, 814, 112, 850, 65, 2170, 83, 816, 112, 859, 54, 2175, 78, 818, 113, 867, 41, 2182, 74, 818, 114, 871, 28, 2193, 70, 820, 114, 3095, 66, 821, 114, 3098, 61, 823, 115, 3099, 57, 824, 116, 3101, 53, 826, 116, 3103, 49, 828, 117, 3106, 42, 831, 117, 3111, 34, 834, 117, 3115, 28, 836, 118, 3117, 23, 838, 118, 3120, 18, 840, 118, 3122, 14, 842, 119, 3977, 119, 3977, 118, 3978, 118, 3978, 118, 3978, 118, 3977, 121, 3975, 121, 3975, 123, 3973, 123, 3973, 119, 3977, 119, 3977, 119, 3977, 119, 3977, 119, 3977, 119, 3977, 119, 3978, 118, 3978, 118, 3978, 118, 3979, 117, 3979, 117, 3979, 117, 3979, 117, 3980, 118, 3978, 118, 3978, 122, 3974, 122, 3974, 122, 3974, 122, 3974, 122, 3974, 123, 3973, 123, 3973, 123, 3973, 123, 3973, 123, 3973, 123, 3973, 123, 3973, 123, 3973, 123, 3973, 124, 3972, 124, 3972, 124, 3972, 124, 3972, 125, 3971, 125, 3971, 125, 3971, 125, 3971, 125, 3971, 126, 3969, 127, 3969, 127, 3969, 127, 3969, 128, 3967, 129, 3967, 129, 3967, 129, 3967, 130, 3965, 131, 3965, 131, 3965, 131, 3965, 132, 3963, 133, 3963, 133, 3963, 134, 3961, 135, 3961, 135, 3961, 136, 3959, 137, 3959, 137, 3958, 138, 3958, 138, 3958, 138, 3957, 139, 3957, 139, 3957, 140, 3955, 141, 3955, 141, 3954, 142, 3954, 143, 3952, 144, 3952, 144, 3951, 145, 3951, 146, 3949, 147, 3949, 147, 3948, 148, 3948, 149, 3947, 149, 3946, 150, 3946, 150, 3946, 150, 3946, 151, 3945, 151, 3945, 151, 3945, 151, 3945, 152, 3944, 152, 3943, 153, 3943, 154, 3942, 154, 3942, 154, 3942, 154, 3942, 154, 3942, 154, 3942, 154, 3942, 154, 3941, 155, 3941, 155, 3941, 155, 3941, 155, 3941, 155, 3941, 155, 3941, 155, 3941, 155, 3941, 155, 3940, 156, 3940, 156, 3940, 156, 3940, 156, 3939, 156, 3940, 156, 3940, 156, 3940, 156, 3939, 156, 3940, 156, 3940, 156, 3940, 156, 3940, 156, 3940, 156, 3940, 156, 3940, 156, 3940, 156, 3939, 157, 3939, 157, 3939, 157, 3939, 157, 3938, 158, 3938, 158, 3938, 158, 3938, 158, 3938, 158, 3937, 159, 3937, 159, 3937, 159, 3937, 159, 3937, 159, 3937, 159, 3937, 159, 3937, 159, 3937, 159, 3936, 160, 3936, 160, 3936, 160, 3936, 160, 3935, 161, 3935, 161, 3935, 161, 3935, 161, 3935, 162, 3933, 163, 3933, 163, 3933, 163, 3933, 164, 3931, 165, 3931, 165, 3931, 165, 3931, 165, 3931, 166, 3929, 167, 3929, 167, 3929, 167, 3929, 168, 3928, 168, 3928, 168, 3928, 169, 3927, 169, 3927, 169, 3926, 171, 3925, 171, 3925, 172, 3924, 172, 3923, 173, 3923, 173, 3923, 173, 3922, 175, 3921, 175, 3921, 175, 3920, 176, 3920, 177, 3918, 178, 3918, 178, 3918, 178, 3918, 179, 3916, 180, 3916, 180, 3916, 180, 3916, 181, 3914, 182, 3914, 182, 3914, 182, 3913, 183, 3913, 184, 3912, 184, 3911, 185, 3911, 185, 3911, 185, 3910, 186, 3910, 186, 3910, 186, 3910, 186, 3909, 188, 3908, 188, 3908, 188, 3908, 188, 3908, 189, 3907, 189, 3907, 189, 3907, 190, 3906, 190, 3906, 190, 3905, 192, 3904, 192, 3904, 192, 3904, 193, 3902, 194, 3902, 194, 3902, 194, 3902, 195, 3900, 196, 3900, 196, 3900, 196, 3900, 196, 3899, 197, 3899, 197, 3899, 197, 3899, 197, 3899, 198, 3897, 199, 3897, 199, 3897, 199, 3897, 200, 3895, 201, 3895, 201, 3895, 201, 3894, 202, 3894, 203, 3893, 203, 3892, 204, 3892, 204, 3892, 204, 3891, 205, 3891, 205, 3891, 205, 3890, 206, 3890, 207, 3889, 207, 3888, 208, 3888, 208, 3887, 210, 3886, 210, 3886, 210, 3886, 210, 3886, 211, 3884, 212, 3884, 212, 3884, 212, 3884, 213, 3882, 214, 3882, 214, 3882, 214, 3882, 214, 3882, 215, 3880, 216, 3880, 216, 3880, 216, 3880, 217, 3878, 218, 3878, 218, 3878, 218, 3878, 218, 3877, 220, 3876, 220, 3876, 220, 3876, 220, 3875, 222, 3874, 222, 3874, 222, 3874, 222, 3874, 222, 3873, 224, 3872, 224, 3872, 224, 3872, 224, 3871, 226, 3870, 226, 3870, 226, 3870, 226, 3869, 228, 3868, 228, 3868, 228, 3868, 228, 3867, 230, 3866, 230, 3866, 230, 3865, 232, 3864, 232, 3864, 232, 3863, 234, 3862, 234, 3862, 234, 3861, 236, 3860, 236, 3860, 236, 3860, 236, 3859, 238, 3858, 238, 3858, 238, 3858, 238, 3857, 239, 3857, 239, 3857, 239, 3857, 239, 3857, 239, 3856, 241, 3855, 241, 3855, 241, 3855, 241, 3854, 243, 3853, 243, 3853, 243, 3852, 245, 3851, 245, 3851, 245, 3850, 247, 3849, 247, 3849, 248, 3848, 248, 3848, 248, 3848, 248, 3848, 249, 3846, 250, 3846, 250, 3846, 250, 3846, 251, 3845, 251, 3845, 251, 3845, 251, 3845, 251, 3844, 253, 3843, 253, 3843, 253, 3843, 253, 3842, 255, 3841, 255, 3841, 255, 3840, 256, 3840, 256, 3840, 257, 3838, 258, 3838, 258, 3838, 258, 3837, 259, 3837, 259, 3837, 259, 3837, 259, 3837, 259, 3836, 261, 3835, 261, 3835, 261, 3835, 261, 3834, 263, 3833, 263, 3833, 263, 3832, 264, 3832, 264, 3832, 265, 3830, 266, 3830, 266, 3830, 266, 3830, 267, 3829, 267, 3829, 267, 3829, 268, 3827, 269, 3827, 269, 3827, 270, 3826, 270, 3826, 271, 3825, 271, 3825, 271, 3825, 271, 3825, 272, 3823, 273, 3823, 273, 3823, 273, 3823, 274, 3821, 275, 3821, 275, 3821, 276, 3820, 276, 3820, 276, 3819, 278, 3818, 278, 3818, 278, 3818, 279, 3816, 280, 3816, 280, 3816, 280, 3816, 281, 3815, 281, 3814, 282, 3814, 282, 3814, 283, 3813, 283, 3813, 283, 3813, 283, 3813, 283, 3813, 284, 3811, 285, 3811, 285, 3811, 285, 3811, 286, 3809, 287, 3809, 287, 3809, 287, 3808, 288, 3808, 289, 3807, 289, 3806, 290, 3806, 290, 3806, 291, 3804, 292, 3804, 292, 3804, 292, 3804, 293, 3803, 293, 3802, 294, 3802, 294, 3802, 294, 3802, 294, 3801, 295, 3801, 295, 3801, 295, 3801, 296, 3799, 297, 3799, 297, 3799, 297, 3799, 298, 3798, 298, 3798, 298, 3798, 299, 3797, 299, 3797, 299, 3796, 301, 3795, 301, 3795, 301, 3795, 302, 3793, 303, 3793, 303, 3793, 303, 3793, 304, 3792, 304, 3791, 305, 3791, 305, 3791, 306, 3790, 306, 3790, 306, 3790, 306, 3790, 306, 3790, 307, 3789, 307, 3788, 308, 3788, 308, 3788, 308, 3788, 308, 3787, 309, 3787, 309, 3787, 309, 3787, 310, 3786, 310, 3785, 311, 3785, 311, 3785, 312, 3784, 312, 3783, 313, 3783, 313, 3783, 313, 3783, 313, 3783, 313, 3782, 314, 3782, 314, 3782, 314, 3782, 314, 3782, 316, 3780, 316, 3780, 316, 3780, 316, 3780, 316, 3779, 317, 3779, 317, 3779, 317, 3779, 317, 3778, 318, 3778, 319, 3777, 320, 3776, 320, 3775, 321, 3775, 321, 3775, 322, 3774, 322, 3774, 322, 3774, 322, 3774, 322, 3774, 323, 3773, 323, 3772, 324, 3772, 324, 3772, 325, 3771, 325, 3770, 326, 3770, 326, 3770, 326, 3770, 327, 3769, 327, 3768, 328, 3768, 328, 3768, 328, 3768, 328, 3767, 329, 3767, 329, 3767, 329, 3767, 330, 3765, 331, 3765, 331, 3765, 331, 3765, 332, 3764, 332, 3764, 332, 3764, 332, 3764, 333, 3763, 333, 3763, 333, 3763, 333, 3763, 333, 3763, 333, 3763, 333, 3763, 333, 3763, 333, 3763, 334, 3762, 334, 3762, 334, 3761, 335, 3761, 336, 3760, 336, 3760, 336, 3760, 336, 3760, 336, 3760, 337, 3759, 337, 3759, 337, 3759, 337, 3759, 337, 3759, 337, 3759, 337, 3759, 337, 3759, 337, 3759, 338, 3758, 338, 3758, 338, 3758, 338, 3758, 338, 3758, 338, 3758, 338, 3758, 338, 3758, 338, 3758, 339, 3757, 339, 3757, 339, 3756, 340, 3756, 341, 3755, 341, 3755, 341, 3754, 343, 3753, 343, 3753, 343, 3753, 344, 3751, 345, 3751, 345, 3751, 345, 3751, 345, 3751, 345, 3751, 345, 3751, 346, 3750, 346, 3750, 346, 3749, 347, 3749, 347, 3749, 347, 3749, 347, 3748, 348, 3748, 348, 3748, 347, 3749, 347, 3749, 346, 3749, 347, 3749, 346, 3750, 346, 3750, 345, 3751, 345, 3751, 345, 3751, 344, 3752, 344, 3752, 344, 3751, 344, 3752, 344, 3752, 344, 3752, 344, 3752, 343, 3753, 343, 3753, 343, 3753, 343, 3753, 343, 3752, 344, 3752, 344, 3752, 344, 3752, 344, 3751, 345, 3751, 345, 3751, 345, 3751, 345, 3751, 345, 3750, 346, 3750, 346, 3750, 346, 3750, 347, 3748, 348, 3748, 348, 3748, 348, 3748, 348, 3747, 349, 3747, 349, 3747, 349, 3747, 349, 3746, 351, 3745, 351, 3745, 351, 3745, 351, 3745, 352, 3743, 353, 3743, 353, 3743, 354, 3742, 354, 3741, 355, 3741, 356, 3740, 356, 3739, 358, 3738, 358, 3738, 358, 3737, 360, 3736, 360, 3735, 361, 3735, 362, 3734, 362, 3734, 363, 3733, 363, 3732, 365, 3731, 365, 3731, 366, 3730, 366, 3730, 367, 3729, 367, 3729, 368, 3728, 368, 3728, 368, 3727, 370, 3726, 370, 3726, 370, 3726, 371, 3725, 371, 3725, 372, 3724, 372, 3724, 373, 3723, 373, 3722, 375, 3721, 376, 3720, 376, 3720, 377, 3718, 379, 3717, 379, 3717, 380, 3715, 381, 3715, 382, 3714, 382, 3713, 384, 3712, 384, 3711, 386, 3710, 386, 3710, 387, 3708, 388, 3708, 389, 3707, 389, 3706, 391, 3705, 391, 3705, 392, 3703, 393, 3703, 393, 3703, 394, 3701, 395, 3701, 395, 3701, 396, 3699, 397, 3699, 398, 3697, 399, 3697, 400, 3695, 401, 3695, 402, 3694, 402, 3693, 404, 3692, 404, 3692, 405, 3691, 405, 3691, 406, 3690, 406, 3690, 406, 3690, 407, 3689, 407, 3688, 408, 3688, 408, 3688, 409, 3687, 409, 3686, 410, 3686, 410, 3686, 411, 3685, 411, 3684, 412, 3684, 412, 3684, 413, 3683, 413, 3682, 414, 3682, 414, 3682, 414, 3682, 415, 3681, 415, 3680, 416, 3680, 416, 3680, 417, 3679, 417, 3678, 418, 3678, 418, 3678, 418, 3678, 418, 3678, 418, 3677, 419, 3677, 419, 3677, 419, 3677, 419, 3676, 420, 3676, 420, 3676, 420, 3675, 422, 3674, 422, 3674, 422, 3673, 423, 3673, 423, 3672, 424, 3672, 424, 3672, 424, 3671, 425, 3671, 425, 3671, 425, 3671, 425, 3671, 425, 3672, 424, 3672, 424, 3673, 423, 3673, 423, 3673, 423, 3674, 422, 3674, 422, 3674, 422, 3675, 421, 3675, 421, 3675, 421, 3675, 421, 3676, 420, 3676, 420, 3676, 420, 3676, 420, 3676, 420, 3676, 420, 3676, 420, 3676, 420, 3676, 420, 3676, 420, 3676, 420, 3676, 421, 3675, 421, 3675, 421, 3675, 421, 3675, 421, 3675, 421, 3674, 422, 3674, 422, 3674, 422, 3674, 422, 3673, 423, 3673, 423, 3673, 423, 3673, 423, 3672, 424, 3672, 424, 3672, 424, 3672, 424, 3672, 425, 3671, 425, 3671, 425, 3671, 425, 3671, 425, 3671, 425, 3671, 425, 3671, 425, 3671, 425, 3671, 425, 3671, 425, 3671, 425, 3671, 425, 3671, 425, 3670, 426, 3670, 426, 3670, 426, 3670, 426, 3670, 426, 3670, 426, 3670, 426, 3670, 426, 3670, 426, 3670, 426, 3670, 426, 3670, 426, 3670, 426, 3670, 427, 3669, 427, 3669, 427, 3669, 427, 3669, 427, 3669, 427, 3669, 427, 3669, 427, 3669, 427, 3669, 428, 3668, 428, 3668, 428, 3668, 428, 3668, 428, 3668, 428, 3668, 428, 3668, 428, 3668, 428, 3668, 429, 3667, 429, 3667, 429, 3667, 429, 3667, 429, 3666, 430, 3666, 430, 3666, 430, 3666, 430, 3666, 431, 3665, 431, 3665, 431, 3665, 431, 3665, 432, 3663, 433, 3663, 433, 3663, 433, 3663, 433, 3662, 434, 3662, 434, 3662, 434, 3662, 434, 3662, 434, 3662, 434, 3662, 434, 3662, 434, 3662, 434, 3662, 434, 3662, 434, 3662, 434, 3662, 434, 3662, 434, 3661, 435, 3661, 435, 3661, 435, 3661, 435, 3661, 436, 3660, 436, 3660, 436, 3660, 436, 3660, 436, 3659, 437, 3659, 437, 3659, 437, 3659, 437, 3659, 438, 3658, 438, 3658, 438, 3658, 438, 3658, 438, 3658, 438, 3658, 436, 3660, 436, 3660, 436, 3660, 436, 3660, 436, 3660, 436, 3660, 436, 3660, 436, 3659, 437, 3659, 437, 3659, 437, 3659, 437, 3658, 438, 3658, 438, 3658, 438, 3658, 438, 3658, 438, 3657, 439, 3657, 439, 3657, 439, 3657, 441, 3655, 441, 3655, 441, 3655, 441, 3655, 441, 3655, 441, 3654, 442, 3654, 442, 3654, 442, 3654, 442, 3654, 442, 3654, 442, 3654, 440, 3656, 440, 3656, 440, 3655, 441, 3655, 441, 3655, 441, 3655, 441, 3655, 441, 3655, 441, 3655, 441, 3655, 441, 3655, 441, 3655, 441, 3655, 441, 3655, 441, 3655, 441, 3655, 441, 3655, 441, 3655, 441, 3655, 441, 3655, 441, 3654, 442, 3654, 442, 3654, 442, 3654, 442, 3654, 442, 3654, 442, 3654, 442, 3654, 442, 3654, 442, 3653, 443, 3653, 443, 3653, 443, 3653, 443, 3652, 444, 3652, 444, 3652, 444, 3652, 444, 3652, 444, 3651, 445, 3651, 447, 3649, 447, 3649, 456, 3639, 457, 3639, 457, 3639, 457, 3639, 458, 3638, 458, 3638, 458, 3638, 458, 3638, 458, 3638, 459, 3637, 459, 3637, 459, 3637, 459, 3637, 460, 3636, 460, 3635, 461, 3635, 461, 3635, 461, 3635, 462, 3633, 463, 3633, 463, 3633, 463, 3633, 464, 3631, 465, 3631, 465, 3631, 466, 3630, 466, 3630, 466, 3630, 467, 3629, 467, 3629, 467, 3629, 467, 3628, 468, 3628, 468, 3628, 468, 3628, 469, 3626, 470, 3626, 470, 3626, 470, 3626, 471, 3625, 471, 3624, 472, 3624, 472, 3624, 472, 3624, 473, 3622, 474, 3622, 474, 3622, 474, 3622, 474, 3622, 474, 3621, 475, 3621, 475, 3621, 475, 3621, 475, 3621, 475, 3621, 475, 3621, 475, 3621, 475, 3621, 475, 3621, 475, 3621, 475, 3621, 475, 3621, 476, 3620, 476, 3620, 476, 3620, 476, 3620, 477, 3619, 477, 3619, 477, 3619, 477, 3619, 477, 3619, 478, 3618, 478, 3618, 478, 3618, 478, 3618, 478, 3618, 478, 3617, 479, 3617, 479, 3617, 479, 3617, 479, 3617, 479, 3617, 479, 3617, 479, 3617, 479, 3617, 479, 3616, 480, 3616, 480, 3616, 480, 3616, 480, 3616, 479, 3617, 479, 3617, 479, 3617, 479, 3617, 479, 3617, 479, 3617, 479, 3617, 479, 3617, 479, 3617, 479, 3617, 479, 3617, 479, 3617, 479, 3617, 479, 3617, 479, 3617, 479, 3617, 479, 3617, 479, 3617, 479, 3617, 479, 3617, 479, 3617, 479, 3617, 479, 3617, 479, 3617, 479, 3617, 479, 3617, 479, 3617, 478, 3618, 478, 3618, 478, 3618, 478, 3618, 478, 3618, 477, 3619, 477, 3619, 477, 3619, 477, 3619, 476, 3620, 476, 3620, 476, 3620, 476, 3620, 476, 3620, 476, 3620, 476, 3620, 476, 3620, 476, 3620, 476, 3620, 476, 3620, 476, 3620, 476, 3620, 476, 3620, 476, 3620, 476, 3620, 476, 3620, 475, 3621, 475, 3621, 475, 3621, 475, 3621, 475, 3621, 474, 3622, 474, 3622, 474, 3622, 474, 3622, 473, 3623, 473, 3623, 473, 3623, 473, 3623, 473, 3623, 472, 3624, 472, 3624, 472, 3624, 472, 3624, 471, 3625, 471, 3625, 471, 3625, 471, 3625, 471, 3625, 471, 3625, 471, 3625, 471, 3625, 471, 3625, 470, 3626, 470, 3626, 470, 3626, 470, 3626, 470, 3626, 469, 3627, 469, 3627, 469, 3627, 469, 3627, 468, 3628, 468, 3628, 468, 3628, 468, 3628, 467, 3629, 467, 3629, 467, 3629, 467, 3629, 466, 3630, 466, 3630, 466, 3630, 466, 3630, 466, 3631, 465, 3631, 465, 3631, 465, 3631, 465, 3632, 463, 3633, 463, 3633, 463, 3633, 463, 3633, 463, 3633, 463, 3633, 463, 3633, 463, 3633, 463, 3633, 463, 3633, 463, 3633, 463, 3633, 463, 3633, 463, 3633, 463, 3633, 462, 3634, 462, 3634, 462, 3634, 462, 3634, 461, 3635, 461, 3635, 461, 3635, 461, 3635, 461, 3635, 461, 3635, 461, 3635, 461, 3635, 461, 3635, 461, 3635, 461, 3635, 461, 3635, 461, 3635, 461, 3635, 461, 3635, 461, 3635, 461, 3635, 461, 3635, 460, 3636, 460, 3636, 460, 3636, 460, 3636, 460, 3637, 459, 3637, 459, 3637, 459, 3637, 459, 3638, 458, 3638, 458, 3638, 458, 3638, 458, 3638, 458, 3638, 458, 3638, 458, 3638, 458, 3638, 458, 3639, 457, 3639, 457, 3639, 457, 3639, 457, 3639, 457, 3639, 457, 3639, 457, 3639, 457, 3640, 456, 3640, 456, 3640, 456, 3640, 456, 3640, 456, 3640, 456, 3640, 456, 3640, 456, 3640, 456, 3640, 456, 3641, 455, 3641, 455, 3641, 455, 3641, 455, 3641, 455, 3641, 455, 3641, 455, 3641, 455, 3641, 455, 3641, 455, 3641, 455, 3641, 455, 3641, 455, 3642, 454, 3642, 454, 3642, 454, 3642, 454, 3643, 449, 3647, 449, 3647, 447, 3649, 447, 3649, 447, 3650, 446, 3650, 446, 3650, 446, 3650, 446, 3651, 445, 3651, 445, 3651, 445, 3651, 445, 3651, 445, 3651, 445, 3651, 445, 3651, 445, 3652, 444, 3652, 444, 3652, 444, 3653, 443, 3653, 443, 3653, 443, 3654, 442, 3654, 442, 3654, 442, 3654, 442, 3655, 441, 3655, 441, 3655, 441, 3655, 441, 3656, 440, 3656, 440, 3656, 440, 3656, 440, 3657, 439, 3657, 439, 3657, 439, 3657, 439, 3657, 439, 3658, 438, 3658, 438, 3659, 437, 3659, 437, 3659, 437, 3660, 436, 3660, 436, 3660, 436, 3661, 435, 3661, 435, 3661, 435, 3661, 435, 3662, 436, 3660, 436, 3660, 436, 3660, 436, 3660, 436, 3660, 435, 3661, 435, 3661, 435, 3661, 434, 3662, 434, 3662, 434, 3662, 434, 3662, 433, 3663, 433, 3663, 433, 3663, 433, 3664, 432, 3665, 431, 3666, 430, 3667, 429, 3668, 428, 3668, 427, 3670, 426, 3671, 425, 3671, 425, 3672, 424, 3672, 424, 3673, 423, 3673, 423, 3674, 422, 3674, 421, 3675, 421, 3676, 420, 3676, 420, 3676, 420, 3677, 418, 3678, 418, 3678, 418, 3678, 418, 3679, 416, 3680, 416, 3680, 416, 3680, 416, 3680, 416, 3680, 416, 3680, 416, 3680, 415, 3681, 415, 3681, 415, 3681, 414, 3682, 414, 3682, 414, 3682, 413, 3683, 413, 3683, 413, 3683, 413, 3683, 412, 3685, 411, 3685, 411, 3685, 411, 3685, 411, 3685, 411, 3685, 411, 3685, 411, 3685, 411, 3686, 410, 3686, 410, 3686, 410, 3686, 410, 3686, 410, 3686, 410, 3687, 409, 3687, 409, 3687, 410, 3686, 410, 3687, 409, 3687, 409, 3687, 410, 3686, 410, 3687, 409, 3687, 410, 3686, 410, 3686, 410, 3687, 410, 3686, 410, 3686, 411, 3685, 411, 3685, 412, 3684, 412, 3685, 412, 3684, 413, 3683, 413, 3683, 414, 3683, 414, 3682, 415, 3681, 415, 3682, 414, 3682, 414, 3683, 413, 3683, 413, 3683, 413, 3683, 413, 3683, 413, 3683, 413, 3683, 413, 3683, 413, 3683, 413, 3683, 413, 3683, 415, 3681, 415, 3681, 411, 3684, 412, 3684, 412, 3683, 413, 3685, 411, 3686, 410, 3687, 411, 3686, 410, 3687, 407, 3689, 407, 3690, 406, 3690, 406, 3690, 406, 3690, 406, 3690, 406, 3690, 405, 3692, 403, 3694, 401, 3695, 400, 3697, 398, 3698, 397, 3700, 395, 3701, 395, 3702, 393, 3703, 392, 3704, 392, 3704, 391, 3705, 391, 3705, 390, 3706, 390, 3706, 389, 3707, 389, 3707, 389, 3707, 388, 3708, 388, 3708, 388, 3707, 388, 3708, 388, 3707, 389, 3706, 390, 3706, 389, 3707, 389, 3707, 389, 3707, 389, 3708, 388, 3708, 388, 3709, 387, 3709, 387, 3709, 387, 3709, 387, 3709, 387, 3709, 387, 3709, 387, 3709, 387, 3709, 387, 3709, 387, 3709, 387, 3709, 388, 3708, 388, 3708, 388, 3708, 388, 3708, 389, 3707, 389, 3707, 389, 3707, 390, 3706, 390, 3706, 390, 3706, 391, 3705, 391, 3705, 392, 3704, 392, 3704, 391, 3705, 391, 3705, 391, 3705, 391, 3705, 391, 3705, 391, 3705, 391, 3705, 391, 3705, 391, 3705, 390, 3706, 390, 3706, 390, 3706, 390, 3706, 390, 3706, 390, 3706, 390, 3706, 390, 3706, 390, 3706, 390, 3706, 390, 3706, 390, 3707, 389, 3707, 389, 3707, 388, 3708, 388, 3708, 388, 3708, 388, 3708, 387, 3709, 387, 3709, 387, 3709, 387, 3709, 387, 3709, 387, 3709, 387, 3710, 386, 3710, 386, 3711, 384, 3712, 384, 3712, 384, 3712, 384, 3712, 384, 3713, 383, 3713, 383, 3713, 383, 3713, 383, 3714, 381, 3715, 381, 3716, 380, 3716, 380, 3716, 379, 3717, 379, 3717, 379, 3718, 378, 3719, 376, 3720, 376, 3721, 375, 3721, 375, 3721, 375, 3721, 374, 3722, 374, 3722, 374, 3722, 374, 3722, 373, 3724, 372, 3724, 372, 3724, 372, 3724, 372, 3725, 370, 3726, 370, 3727, 369, 3727, 369, 3727, 368, 3728, 368, 3728, 368, 3729, 367, 3729, 367, 3729, 367, 3729, 367, 3729, 367, 3729, 367, 3729, 366, 3730, 366, 3730, 366, 3730, 366, 3730, 365, 3730, 366, 3730, 366, 3731, 365, 3731, 364, 3732, 364, 3732, 364, 3732, 364, 3732, 364, 3732, 364, 3732, 364, 3733, 363, 3733, 363, 3734, 361, 3735, 361, 3735, 361, 3735, 361, 3736, 360, 3736, 360, 3736, 360, 3736, 360, 3736, 360, 3736, 359, 3737, 359, 3737, 359, 3738, 358, 3738, 358, 3739, 357, 3739, 357, 3739, 357, 3739, 357, 3739, 356, 3740, 356, 3740, 356, 3740, 356, 3740, 356, 3740, 356, 3740, 356, 3740, 356, 3740, 356, 3741, 354, 3742, 354, 3743, 353, 3743, 353, 3744, 352, 3744, 352, 3744, 352, 3744, 352, 3744, 352, 3744, 351, 3745, 351, 3745, 351, 3745, 351, 3745, 351, 3745, 351, 3745, 351, 3746, 350, 3746, 350, 3746, 349, 3747, 349, 3747, 349, 3747, 349, 3747, 349, 3747, 349, 3747, 349, 3747, 349, 3747, 349, 3747, 349, 3747, 349, 3747, 349, 3748, 348, 3748, 348, 3749, 346, 3750, 346, 3750, 346, 3750, 346, 3750, 345, 3751, 345, 3751, 345, 3751, 345, 3751, 345, 3751, 345, 3751, 345, 3751, 345, 3751, 345, 3751, 344, 3753, 343, 3753, 343, 3753, 343, 3753, 343, 3754, 341, 3755, 341, 3756, 340, 3756, 339, 3757, 339, 3757, 339, 3757, 338, 3758, 338, 3759, 337, 3759, 336, 3760, 336, 3760, 335, 3761, 335, 3761, 335, 3761, 334, 3762, 334, 3762, 334, 3762, 333, 3763, 333, 3763, 333, 3763, 333, 3763, 332, 3764, 332, 3764, 332, 3764, 332, 3764, 331, 3766, 330, 3766, 330, 3767, 329, 3767, 328, 3769, 327, 3769, 327, 3769, 327, 3769, 327, 3769, 327, 3769, 327, 3769, 327, 3769, 327, 3769, 327, 3769, 327, 3769, 327, 3770, 326, 3770, 326, 3771, 325, 3771, 325, 3771, 325, 3771, 325, 3771, 324, 3772, 324, 3772, 324, 3772, 324, 3772, 324, 3772, 324, 3772, 324, 3773, 323, 3773, 323, 3773, 323, 3773, 323, 3773, 323, 3774, 322, 3774, 322, 3775, 321, 3775, 321, 3775, 321, 3775, 321, 3775, 321, 3775, 321, 3775, 321, 3775, 321, 3775, 321, 3775, 321, 3775, 321, 3775, 321, 3775, 321, 3775, 322, 3774, 322, 3774, 322, 3775, 321, 3775, 321, 3775, 321, 3775, 321, 3775, 321, 3775, 321, 3775, 322, 3774, 322, 3774, 322, 3774, 322, 3774, 323, 3773, 323, 3773, 323, 3773, 323, 3773, 323, 3773, 322, 3775, 321, 3775, 321, 3776, 320, 3776, 320, 3776, 320, 3776, 320, 3776, 320, 3776, 320, 3776, 320, 3777, 319, 3777, 319, 3777, 319, 3777, 319, 3777, 319, 3777, 319, 3777, 319, 3777, 320, 3776, 319, 3777, 319, 3777, 318, 3778, 318, 3778, 318, 3779, 316, 3781, 315, 3781, 315, 3782, 313, 3783, 313, 3783, 313, 3783, 312, 3784, 312, 3784, 312, 3784, 311, 3785, 311, 3785, 311, 3785, 310, 3786, 310, 3786, 310, 3786, 310, 3786, 309, 3787, 309, 3787, 309, 3788, 308, 3788, 307, 3790, 306, 3790, 306, 3790, 305, 3791, 305, 3791, 305, 3791, 304, 3792, 304, 3792, 304, 3792, 303, 3793, 303, 3794, 302, 3794, 302, 3794, 301, 3795, 301, 3795, 301, 3795, 301, 3795, 300, 3796, 300, 3796, 300, 3796, 300, 3796, 299, 3797, 299, 3798, 298, 3798, 298, 3799, 297, 3799, 297, 3799, 297, 3799, 296, 3801, 295, 3801, 295, 3801, 294, 3803, 293, 3803, 293, 3804, 291, 3805, 291, 3760, 1, 44, 291, 3805, 291, 3805, 290, 3807, 289, 3807, 289, 3808, 288, 3808, 288, 3809, 287, 3809, 287, 3809, 287, 3809, 287, 3809, 286, 3810, 286, 3810, 286, 3810, 286, 3810, 286, 3810, 285, 3812, 284, 3812, 284, 3812, 284, 3812, 283, 3813, 283, 3814, 282, 3814, 282, 3815, 281, 3815, 281, 3815, 281, 3815, 281, 3815, 281, 3816, 279, 3817, 279, 3817, 279, 3817, 279, 3817, 278, 3818, 278, 3819, 277, 3819, 277, 3820, 275, 3821, 275, 3821, 275, 3821, 275, 3822, 274, 3822, 274, 3822, 274, 3822, 274, 3822, 274, 3822, 273, 3823, 273, 3823, 273, 3823, 273, 3824, 272, 3824, 271, 3826, 270, 3826, 270, 3826, 270, 3826, 269, 3828, 268, 3828, 268, 3828, 268, 3828, 267, 3829, 267, 3829, 267, 3829, 267, 3829, 266, 3830, 266, 3831, 265, 3831, 265, 3832, 264, 3832, 264, 3832, 264, 3832, 263, 3833, 263, 3833, 263, 3833, 262, 3834, 262, 3834, 262, 3834, 261, 3835, 261, 3836, 260, 3836, 260, 3837, 258, 3838, 258, 3839, 257, 3839, 257, 3839, 257, 3839, 257, 3839, 257, 3839, 257, 3839, 257, 3839, 257, 3839, 257, 3839, 257, 3839, 257, 3839, 257, 3839, 257, 3839, 257, 3839, 257, 3839, 258, 3839, 257, 3839, 257, 3839, 257, 3839, 258, 3838, 258, 3838, 258, 3838, 259, 3837, 259, 3837, 259, 3837, 260, 3836, 260, 3836, 261, 3835, 261, 3835, 261, 3835, 261, 3835, 261, 3835, 261, 3835, 261, 3835, 261, 3835, 260, 3837, 259, 3837, 258, 3839, 257, 3839, 256, 3840, 256, 3840, 255, 3841, 254, 3842, 254, 3842, 253, 3843, 252, 3844, 252, 3844, 251, 3845, 251, 3845, 250, 3846, 250, 3846, 249, 3847, 248, 3848, 248, 3848, 247, 3849, 247, 3849, 246, 3850, 246, 3851, 244, 3852, 244, 3852, 243, 3853, 243, 3853, 242, 3855, 241, 3856, 239, 3857, 239, 3858, 237, 3859, 237, 3859, 237, 3859, 236, 3860, 236, 3860, 236, 3860, 235, 3861, 235, 3861, 234, 3862, 234, 3862, 234, 3862, 233, 3864, 232, 3864, 232, 3865, 230, 3866, 230, 3866, 230, 3866, 230, 3867, 228, 3868, 228, 3868, 228, 3868, 228, 3868, 228, 3868, 228, 3868, 228, 3868, 228, 3868, 228, 3868, 228, 3868, 228, 3868, 228, 3868, 228, 3868, 228, 3868, 228, 3868, 228, 3868, 228, 3868, 229, 3867, 229, 3867, 229, 3867, 229, 3867, 229, 3867, 229, 3867, 229, 3868, 228, 3868, 228, 3869, 228, 3868, 228, 3868, 228, 3869, 227, 3869, 228, 3868, 228, 3868, 228, 3868, 229, 3867, 229, 3867, 229, 3867, 229, 3867, 228, 3868, 228, 3868, 228, 3868, 227, 3869, 227, 3869, 227, 3870, 226, 3870, 225, 3872, 224, 3872, 224, 3872, 224, 3872, 223, 3873, 223, 3873, 223, 3873, 223, 3873, 222, 3874, 222, 3874, 222, 3874, 222, 3874, 222, 3874, 221, 3875, 221, 3875, 221, 3875, 221, 3875, 220, 3876, 220, 3876, 220, 3876, 220, 3877, 219, 3877, 219, 3877, 219, 3877, 219, 3878, 218, 3878, 218, 3879, 217, 3879, 217, 3879, 217, 3879, 217, 3879, 217, 3879, 217, 3879, 217, 3879, 217, 3879, 217, 3879, 217, 3879, 217, 3879, 217, 3879, 217, 3880, 216, 3880, 216, 3880, 216, 3880, 216, 3880, 216, 3881, 215, 3881, 215, 3882, 214, 3882, 214, 3882, 214, 3882, 214, 3882, 214, 3882, 215, 3881, 215, 3881, 215, 3881, 215, 3881, 215, 3881, 215, 3881, 215, 3882, 214, 3882, 214, 3882, 214, 3882, 213, 3883, 213, 3883, 212, 3884, 212, 3884, 212, 3884, 211, 3885, 211, 3885, 211, 3886, 210, 3886, 209, 3888, 208, 3888, 208, 3888, 208, 3888, 207, 3890, 206, 3890, 206, 3890, 206, 3890, 206, 3890, 206, 3890, 205, 3891, 205, 3891, 205, 3891, 205, 3891, 205, 3891, 205, 3892, 204, 3892, 204, 3893, 203, 3893, 203, 3894, 202, 3894, 202, 3894, 202, 3894, 202, 3894, 202, 3894, 202, 3894, 202, 3895, 201, 3895, 201, 3896, 200, 3896, 200, 3896, 200, 3896, 199, 3897, 199, 3897, 199, 3897, 199, 3897, 198, 3898, 198, 3898, 198, 3898, 198, 3898, 197, 3899, 197, 3899, 197, 3900, 195, 3902, 194, 3902, 194, 3903, 193, 3903, 192, 3904, 192, 3904, 192, 3904, 191, 3905, 191, 3905, 191, 3905, 190, 3906, 190, 3906, 189, 3907, 189, 3907, 189, 3907, 188, 3908, 188, 3908, 188, 3908, 188, 3908, 187, 3909, 187, 3909, 187, 3909, 186, 3910, 186, 3910, 186, 3910, 185, 3911, 185, 3911, 184, 3912, 184, 3912, 184, 3912, 183, 3913, 183, 3913, 183, 3913, 183, 3913, 182, 3914, 182, 3914, 182, 3914, 182, 3914, 181, 3915, 181, 3915, 181, 3915, 181, 3915, 181, 3915, 181, 3915, 180, 3916, 180, 3916, 180, 3916, 180, 3916, 180, 3917, 179, 3917, 178, 3918, 178, 3919, 177, 3919, 177, 3920, 175, 3921, 175, 3921, 175, 3921, 175, 3921, 175, 3921, 175, 3922, 173, 3923, 173, 3923, 173, 3923, 173, 3923, 173, 3923, 173, 3923, 173, 3923, 173, 3923, 173, 3923, 173, 3924, 171, 3925, 171, 3925, 171, 3925, 171, 3926, 170, 3926, 170, 3927, 168, 3928, 168, 3928, 168, 3928, 168, 3928, 168, 3928, 168, 3928, 168, 3929, 167, 3929, 167, 3929, 167, 3929, 166, 3930, 166, 3930, 166, 3930, 166, 3930, 166, 3930, 166, 3930, 165, 3931, 165, 3931, 165, 3931, 165, 3932, 164, 3932, 164, 3932, 164, 3932, 164, 3933, 163, 3933, 163, 3933, 163, 3934, 162, 3934, 162, 3935, 161, 3935, 161, 3935, 161, 3935, 161, 3935, 161, 3935, 161, 3936, 160, 3936, 160, 3936, 160, 3936, 160, 3937, 159, 3937, 159, 3938, 158, 3938, 158, 3938, 158, 3938, 158, 3939, 157, 3939, 157, 3939, 157, 3940, 156, 3940, 156, 3940, 156, 3940, 156, 3940, 156, 3940, 156, 3940, 156, 3940, 155, 3941, 155, 3941, 155, 3941, 155, 3941, 154, 3942, 154, 3942, 154, 3942, 154, 3942, 154, 3942, 154, 3942, 153, 3943, 153, 3943, 153, 3943, 153, 3943, 153, 3942, 154, 3942, 153, 3942, 154, 3942, 154, 3942, 154, 3942, 153, 3944, 152, 3944, 152, 3944, 152, 3945, 151, 3945, 151, 3946, 149, 3947, 149, 3948, 148, 3948, 148, 3948, 148, 3948, 148, 3949, 147, 3949, 147, 3949, 147, 3949, 147, 3949, 147, 3949, 147, 3949, 147, 3949, 147, 3949, 147, 3949, 147, 3949, 147, 3949, 147, 3949, 147, 3949, 147, 3949, 147, 3949, 146, 3950, 146, 3950, 145, 3951, 145, 3951, 145, 3952, 144, 3952, 144, 3952, 144, 3952, 144, 3952, 144, 3952, 144, 3952, 145, 3951, 144, 3952, 144, 3951, 145, 3951, 145, 3950, 146, 3949, 146, 3951, 145, 3951, 145, 3952, 144, 3952, 143, 3953, 143, 3953, 142, 3954, 142, 3954, 142, 3954, 142, 3954, 141, 3955, 141, 3954, 142, 3954, 142, 3953, 143, 3952, 144, 3952, 144, 3953, 143, 3953, 143, 3953, 143, 3953, 143, 3953, 144, 3952, 144, 3952, 145, 3951, 145, 3951, 146, 3949, 148, 3948, 149, 3922, 7, 17, 152, 3916, 12, 15, 147, 3918, 17, 13, 146, 3917, 8, 13, 1, 9, 147, 3947, 148, 3950, 145, 3955, 141, 3956, 139, 3958, 138, 3959, 137, 3959, 137, 3960, 135, 3961, 135, 3962, 134, 3962, 133, 3963, 133, 3963, 133, 3964, 132, 3964, 132, 3965, 131, 3965, 130, 3967, 129, 3967, 129, 3967, 128, 3968, 127, 3970, 126, 3970, 125, 3971, 125, 3971, 125, 3972, 124, 3972, 124, 3972, 124, 3972, 123, 3974, 122, 3974, 122, 3974, 122, 3974, 122, 3975, 121, 3975, 121, 3975, 121, 3976, 120, 3976, 120, 3977, 119, 3977, 119, 3977, 118, 3978, 118, 3979, 117, 3979, 116, 3980, 116, 3980, 115, 3981, 115, 3981, 114, 3982, 114, 3982, 114, 3982, 113, 3983, 113, 3984, 111, 3985, 111, 3986, 109, 3987, 109, 3987, 109, 3987, 108, 3988, 108, 3988, 108, 3989, 106, 3990, 106, 3990, 106, 3990, 105, 3992, 104, 3992, 103, 3993, 103, 3993, 103, 3993, 102, 3994, 102, 3995, 101, 3995, 101, 3995, 100, 3996, 100, 3996, 100, 3996, 99, 3998, 98, 3998, 98, 3998, 97, 3999, 97, 3999, 96, 4001, 95, 4001, 95, 4001, 94, 4002, 94, 4002, 94, 4002, 93, 4003, 93, 4003, 92, 4004, 92, 4004, 92, 4004, 91, 4005, 91, 4005, 90, 4007, 89, 4007, 88, 4008, 88, 4008, 88, 4008, 88, 4008, 87, 260, 6, 3743, 87, 260, 7, 3453, 5, 284, 87, 260, 8, 3436, 19, 287, 85, 261, 10, 3417, 35, 288, 85, 261, 11, 3405, 45, 289, 85, 261, 13, 3402, 45, 290, 84, 262, 15, 3399, 45, 291, 84, 262, 17, 3395, 47, 291, 84, 262, 20, 3391, 47, 292, 83, 263, 23, 3386, 49, 292, 83, 263, 27, 3380, 51, 292, 83, 263, 59, 3346, 53, 292, 82, 264, 61, 3341, 56, 293, 81, 264, 64, 3335, 59, 293, 81, 264, 67, 3328, 63, 293, 80, 265, 71, 3315, 73, 292, 80, 265, 76, 3301, 82, 292, 80, 265, 80, 3288, 92, 291, 79, 266, 89, 3270, 101, 292, 78, 266, 93, 3257, 110, 292, 77, 267, 97, 369, 2, 6, 2, 22, 8, 10, 2, 2827, 114, 293, 77, 267, 106, 360, 2, 6, 2, 22, 8, 10, 2, 2823, 118, 293, 77, 262, 116, 270, 7, 66, 2, 10, 10, 8, 2, 6, 2, 2, 10, 10, 2, 22, 2, 4, 2, 1768, 4, 16, 6, 32, 2, 8, 2, 16, 4, 10, 6, 10, 2, 22, 2, 12, 2, 10, 2, 36, 2, 16, 2, 26, 2, 2, 8, 760, 121, 293, 77, 263, 119, 265, 8, 66, 2, 10, 10, 8, 2, 6, 2, 2, 10, 10, 2, 22, 2, 4, 2, 1768, 4, 16, 6, 32, 2, 8, 2, 16, 4, 10, 6, 10, 2, 22, 2, 12, 2, 10, 2, 36, 2, 16, 2, 26, 2, 2, 8, 530, 4, 222, 2, 2, 121, 293, 76, 264, 128, 255, 182, 1746, 343, 421, 6, 18, 19, 213, 132, 293, 76, 264, 137, 245, 184, 1744, 345, 267, 82, 70, 9, 14, 23, 207, 136, 294, 75, 264, 146, 235, 185, 1743, 346, 267, 137, 13, 51, 200, 140, 294, 74, 265, 155, 224, 188, 1741, 347, 267, 139, 9, 56, 193, 144, 295, 73, 265, 164, 214, 190, 1739, 349, 265, 209, 75, 24, 85, 149, 295, 73, 265, 168, 208, 192, 1738, 350, 265, 218, 62, 65, 44, 153, 295, 72, 266, 172, 202, 195, 1735, 352, 265, 542, 295, 72, 266, 181, 191, 198, 599, 20, 1114, 354, 263, 544, 294, 72, 266, 190, 179, 202, 579, 42, 1109, 356, 263, 544, 294, 71, 267, 263, 103, 206, 360, 264, 1103, 359, 261, 546, 293, 71, 267, 300, 62, 211, 345, 287, 1092, 361, 261, 547, 292, 71, 267, 574, 340, 292, 1088, 365, 259, 548, 293, 69, 268, 575, 330, 301, 1085, 368, 259, 549, 292, 69, 268, 576, 325, 306, 1083, 370, 257, 550, 292, 68, 269, 577, 320, 311, 1081, 372, 255, 552, 291, 68, 269, 578, 310, 320, 1080, 373, 255, 553, 291, 67, 268, 581, 299, 330, 1078, 375, 253, 554, 291, 66, 269, 582, 289, 340, 1076, 377, 251, 556, 290, 66, 269, 584, 278, 350, 1074, 379, 249, 557, 290, 66, 269, 586, 267, 359, 1073, 381, 247, 559, 289, 65, 269, 589, 256, 369, 1070, 384, 245, 561, 288, 65, 269, 592, 32, 2, 210, 379, 1068, 386, 243, 563, 287, 65, 269, 594, 30, 2, 186, 403, 1066, 389, 241, 565, 286, 64, 270, 594, 30, 2, 184, 406, 1063, 392, 239, 567, 286, 63, 270, 594, 214, 409, 1061, 394, 237, 570, 284, 62, 270, 595, 212, 412, 1058, 397, 235, 572, 283, 62, 270, 595, 209, 416, 1056, 399, 232, 575, 282, 62, 270, 595, 206, 420, 1053, 401, 231, 576, 283, 61, 270, 595, 202, 425, 1050, 403, 229, 579, 282, 60, 270, 596, 194, 434, 1047, 405, 227, 581, 282, 60, 270, 598, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 30, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 6, 20, 38, 2, 2, 2, 26, 437, 1045, 410, 2, 2, 217, 584, 281, 60, 270, 598, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 30, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 6, 20, 38, 2, 2, 2, 26, 438, 1042, 412, 2, 2, 2, 1, 69, 1, 141, 587, 282, 58, 270, 1228, 1039, 424, 29, 47, 53, 12, 64, 591, 281, 58, 270, 1230, 1035, 430, 17, 124, 56, 596, 280, 58, 270, 1231, 1031, 592, 17, 617, 280, 57, 270, 1234, 1026, 1230, 279, 57, 270, 1236, 1021, 1234, 278, 57, 269, 1239, 1017, 1237, 278, 55, 270, 1242, 1012, 1239, 278, 55, 270, 1245, 1007, 1242, 277, 55, 270, 1249, 1001, 1245, 276, 55, 270, 1254, 994, 1248, 275, 55, 269, 1259, 988, 1250, 276, 53, 270, 1266, 979, 1252, 276, 53, 270, 1269, 974, 1255, 275, 53, 270, 1272, 968, 1258, 275, 53, 269, 1277, 961, 1261, 275, 53, 269, 1279, 957, 1263, 275, 53, 269, 1281, 953, 1266, 274, 52, 270, 1282, 950, 1268, 274, 52, 269, 1285, 946, 1270, 274, 52, 269, 1286, 943, 1272, 274, 52, 269, 1288, 939, 1275, 273, 52, 269, 1290, 935, 1277, 273, 52, 268, 1292, 932, 1279, 274, 51, 268, 1294, 928, 1281, 274, 51, 268, 1295, 925, 1284, 273, 51, 267, 1298, 921, 1286, 273, 51, 267, 1299, 918, 1287, 275, 50, 267, 1300, 915, 1289, 275, 50, 267, 1302, 912, 1289, 276, 50, 267, 1303, 909, 1291, 276, 50, 267, 1305, 906, 1292, 276, 50, 267, 1306, 904, 1293, 276, 50, 267, 1307, 901, 1295, 276, 51, 266, 1309, 898, 1296, 276, 51, 266, 1310, 895, 1298, 276, 51, 266, 1311, 892, 1300, 276, 51, 265, 1313, 889, 1302, 276, 51, 265, 1314, 886, 1304, 276, 51, 265, 1315, 883, 1306, 276, 51, 265, 1317, 880, 1308, 275, 51, 265, 1318, 878, 1309, 275, 51, 265, 1319, 875, 1312, 274, 52, 264, 1320, 873, 1313, 274, 53, 263, 1321, 870, 1316, 273, 54, 262, 1322, 868, 1317, 273, 55, 260, 1325, 865, 1318, 273, 56, 259, 1326, 862, 1320, 273, 59, 256, 1327, 860, 1322, 272, 59, 256, 1328, 857, 1324, 272, 60, 254, 1330, 854, 1326, 272, 61, 253, 1332, 850, 1328, 272, 62, 252, 1333, 848, 1330, 271, 63, 250, 1336, 845, 1331, 271, 64, 249, 1338, 842, 1333, 270, 65, 248, 1339, 840, 1334, 270, 66, 246, 1341, 838, 1335, 270, 67, 245, 1342, 836, 1337, 269, 68, 243, 1344, 835, 1337, 269, 70, 241, 1345, 833, 1339, 268, 71, 240, 1346, 831, 1340, 268, 72, 238, 1348, 829, 1341, 268, 73, 237, 1350, 826, 1342, 268, 74, 236, 1351, 824, 1344, 267, 75, 234, 1354, 821, 1345, 267, 76, 233, 1355, 819, 1347, 266, 77, 231, 1357, 817, 1348, 266, 78, 230, 1358, 815, 1349, 266, 80, 227, 1361, 812, 1351, 265, 81, 226, 1362, 809, 1353, 265, 82, 224, 1365, 806, 1355, 264, 83, 222, 1367, 803, 1357, 264, 84, 221, 1368, 800, 1360, 263, 85, 219, 1371, 797, 1361, 262, 87, 217, 1373, 794, 1363, 262, 88, 215, 1376, 791, 1364, 262, 89, 214, 1377, 789, 1366, 261, 90, 212, 1379, 787, 1367, 261, 92, 210, 1380, 785, 1368, 261, 94, 208, 1381, 782, 1370, 261, 96, 206, 1382, 780, 1372, 260, 98, 204, 1383, 777, 1374, 260, 101, 201, 1384, 774, 1377, 259, 102, 200, 1385, 772, 1378, 258, 111, 192, 1387, 768, 1380, 258, 112, 191, 1388, 766, 1382, 257, 116, 187, 1389, 764, 1383, 257, 117, 186, 1390, 762, 1385, 256, 118, 185, 1391, 760, 1386, 256, 120, 183, 1392, 757, 1388, 255, 124, 180, 1394, 754, 1389, 255, 125, 179, 1395, 751, 1392, 253, 129, 176, 1397, 748, 1393, 252, 133, 173, 1399, 745, 1395, 251, 136, 170, 1401, 742, 1396, 250, 140, 161, 1408, 740, 1397, 250, 144, 157, 1409, 738, 1398, 249, 146, 156, 1410, 736, 1400, 248, 148, 153, 1412, 734, 1401, 247, 151, 151, 1413, 732, 1403, 245, 155, 148, 1415, 728, 1405, 205, 1, 38, 159, 145, 1416, 726, 1406, 174, 33, 36, 161, 144, 1417, 723, 1409, 163, 44, 33, 165, 141, 1419, 721, 1410, 157, 51, 20, 179, 139, 1420, 719, 1412, 152, 56, 17, 183, 137, 1421, 717, 1413, 148, 62, 11, 188, 136, 1422, 715, 1414, 143, 268, 133, 1424, 713, 1415, 141, 272, 131, 1425, 711, 1417, 136, 278, 129, 1426, 709, 1418, 134, 281, 127, 1428, 707, 1420, 130, 286, 125, 1429, 705, 1421, 128, 290, 123, 1430, 703, 1422, 126, 295, 119, 1432, 701, 1424, 123, 301, 115, 1433, 699, 1425, 121, 304, 113, 1435, 697, 1427, 116, 309, 112, 1436, 695, 1428, 114, 312, 110, 1438, 693, 1429, 113, 315, 108, 1439, 690, 1431, 111, 319, 105, 1441, 688, 1433, 97, 1, 7, 326, 102, 1442, 686, 1435, 95, 338, 100, 1443, 684, 1437, 93, 342, 96, 1445, 682, 1438, 91, 345, 94, 1447, 680, 1439, 90, 348, 91, 1448, 679, 1441, 88, 351, 88, 1450, 677, 1442, 87, 354, 85, 1452, 675, 1444, 84, 358, 82, 1453, 675, 1444, 83, 361, 79, 1455, 673, 1446, 81, 364, 76, 1457, 672, 1446, 80, 367, 73, 1458, 671, 1447, 79, 370, 70, 1460, 669, 1448, 78, 372, 67, 1462, 669, 1449, 76, 374, 65, 1464, 667, 1450, 75, 377, 61, 1467, 665, 1451, 74, 380, 57, 1469, 664, 1452, 73, 383, 53, 1472, 662, 1454, 71, 386, 48, 1476, 660, 1455, 70, 389, 45, 1478, 659, 1456, 68, 392, 42, 1480, 657, 1457, 66, 396, 39, 1482, 656, 1458, 64, 399, 36, 1484, 654, 1459, 63, 403, 32, 1485, 654, 1460, 61, 408, 27, 1487, 652, 1461, 59, 412, 23, 1489, 652, 1462, 57, 415, 19, 1492, 650, 1463, 55, 420, 13, 1495, 649, 1465, 52, 1931, 648, 1465, 51, 1933, 646, 1467, 48, 1935, 646, 1468, 45, 1938, 645, 1468, 44, 1939, 644, 1470, 42, 1941, 643, 1471, 41, 1941, 643, 1472, 39, 1943, 642, 1472, 39, 1943, 642, 1474, 36, 1944, 642, 1475, 34, 1945, 642, 1476, 32, 1946, 642, 1476, 31, 1949, 637, 1482, 27, 1950, 636, 1485, 23, 1954, 633, 1488, 19, 1956, 633, 3464, 631, 3465, 630, 3467, 628, 3468, 627, 3470, 626, 3470, 625, 3472, 623, 3473, 623, 3474, 621, 3475, 620, 3476, 619, 3478, 617, 3479, 617, 3479, 616, 3481, 615, 3481, 614, 3482, 614, 3483, 612, 3484, 612, 3485, 610, 3486, 610, 3486, 609, 3488, 608, 3488, 607, 3489, 607, 3490, 605, 3491, 605, 3492, 603, 3493, 603, 3494, 602, 3494, 601, 3496, 600, 3496, 599, 3497, 599, 3498, 598, 3498, 597, 3500, 596, 3500, 596, 3501, 594, 3502, 594, 3503, 592, 3505, 591, 3505, 591, 3506, 589, 3507, 589, 3508, 588, 3508, 587, 3510, 586, 3510, 586, 3511, 584, 3512, 584, 3513, 583, 3513, 583, 3514, 581, 3515, 581, 3516, 580, 3517, 579, 3517, 579, 3518, 578, 3519, 577, 3520, 574, 3522, 574, 3523, 573, 3523, 572, 3525, 571, 3525, 571, 3526, 570, 3526, 570, 3527, 569, 3527, 569, 3528, 568, 3528, 567, 3530, 566, 3530, 566, 3531, 565, 3531, 564, 3533, 563, 3533, 563, 3534, 562, 3534, 561, 3536, 560, 3536, 560, 3537, 559, 3537, 559, 3538, 558, 3538, 558, 3539, 557, 3540, 556, 3540, 555, 3542, 554, 3543, 553, 3544, 552, 3545, 550, 3547, 549, 3548, 548, 3549, 546, 3551, 545, 3552, 544, 3553, 542, 3556, 540, 3557, 539, 3557, 538, 3558, 538, 3558, 538, 3558, 538, 3558, 537, 3559, 537, 3559, 537, 3559, 537, 3559, 536, 3560, 536, 3560, 536, 3560, 535, 3561, 535, 3561, 535, 3561, 534, 3562, 534, 3562, 534, 3563, 533, 3563, 533, 3563, 533, 3564, 532, 3564, 531, 3566, 530, 3566, 530, 3567, 529, 3567, 528, 3569, 527, 3569, 527, 3569, 527, 3570, 526, 3570, 525, 3571, 525, 3572, 524, 3572, 524, 3573, 522, 3574, 522, 3574, 522, 3575, 521, 3575, 521, 3576, 519, 3577, 519, 3578, 518, 3579, 517, 3580, 515, 3582, 514, 3582, 514, 3583, 512, 3584, 512, 3585, 511, 3586, 509, 3588, 508, 3588, 507, 3589, 507, 3589, 507, 3589, 506, 3589, 507, 3589, 507, 3589, 506, 3590, 506, 3590, 505, 3591, 505, 3591, 505, 3591, 504, 3592, 504, 3592, 504, 3592, 503, 3593, 503, 3593, 503, 3593, 502, 3594, 502, 3594, 502, 3594, 501, 3596, 500, 3596, 500, 3596, 499, 3597, 499, 3598, 498, 3598, 497, 3599, 497, 3600, 496, 3600, 495, 3601, 495, 3602, 494, 3602, 493, 3604, 492, 3604, 491, 3606, 490, 3606, 490, 3606, 489, 3607, 489, 3608, 488, 3608, 487, 3609, 487, 3609, 486, 3610, 486, 3610, 485, 3611, 485, 3610, 485, 3611, 485, 3611, 484, 3611, 485, 3611, 484, 3612, 484, 3612, 483, 3612, 484, 3612, 483, 3613, 483, 3613, 482, 3614, 482, 3614, 481, 3615, 481, 3615, 481, 3615, 481, 3615, 481, 3615, 480, 3616, 480, 3616, 164, 73, 243, 3616, 155, 91, 234, 3616, 150, 101, 228, 3617, 146, 109, 224, 3617, 141, 118, 220, 3618, 136, 126, 215, 3619, 131, 136, 210, 3619, 127, 144, 206, 3619, 124, 156, 196, 3621, 123, 160, 192, 3621, 122, 165, 187, 3622, 122, 170, 182, 3623, 120, 175, 178, 3623, 119, 185, 168, 3624, 119, 194, 159, 3625, 117, 200, 154, 3625, 116, 205, 149, 3627, 114, 210, 145, 3627, 113, 214, 141, 3629, 111, 218, 138, 3629, 110, 222, 134, 3631, 108, 227, 130, 3632, 106, 232, 125, 3633, 105, 235, 122, 3635, 103, 239, 119, 3636, 100, 244, 115, 3638, 98, 248, 111, 3640, 95, 253, 107, 3642, 93, 257, 103, 3644, 91, 260, 100, 3646, 88, 265, 96, 3648, 86, 269, 92, 3650, 83, 272, 90, 3652, 80, 276, 87, 3653, 78, 279, 85, 3657, 72, 284, 81, 3661, 67, 289, 78, 3664, 62, 294, 74, 3668, 57, 300, 70, 3672, 51, 306, 66, 3676, 44, 313, 61, 3682, 17, 339, 57, 4042, 52, 4048, 46, 4054, 40, 4060, 33, 4068, 25, 4075, 17, 603966] , dtype=np.uint32)
LEFT_LEG_RLE = np.array( [29907, 17, 4075, 25, 4066, 33, 4051, 48, 4044, 57, 4036, 69, 4019, 86, 4001, 104, 3974, 131, 3956, 144, 3943, 158, 3929, 176, 3710, 390, 3698, 402, 3685, 415, 3672, 429, 3663, 437, 3645, 456, 3636, 464, 3628, 472, 3614, 486, 3606, 495, 3597, 503, 3589, 511, 3580, 520, 3568, 531, 3561, 538, 3555, 544, 3549, 551, 3537, 563, 3529, 572, 3519, 586, 3506, 594, 3497, 604, 3483, 617, 3475, 625, 3468, 638, 3455, 645, 3449, 651, 3442, 663, 3430, 675, 3418, 682, 3410, 691, 3401, 704, 3387, 713, 3379, 722, 3371, 729, 3364, 735, 3359, 740, 3354, 744, 3349, 752, 3340, 760, 3333, 768, 3325, 775, 3318, 781, 3312, 787, 3306, 792, 3302, 797, 3296, 803, 3290, 808, 3286, 813, 3281, 818, 3276, 822, 3272, 827, 3267, 832, 3262, 837, 3257, 842, 3252, 847, 3247, 851, 3242, 856, 3237, 861, 3233, 865, 3228, 870, 3223, 876, 3217, 882, 3211, 887, 3206, 892, 3202, 896, 3198, 900, 3194, 904, 3190, 909, 3185, 914, 3179, 919, 3174, 924, 3170, 928, 3165, 932, 3161, 937, 3157, 940, 3154, 944, 3150, 948, 3145, 953, 3141, 957, 3136, 962, 3131, 967, 3127, 970, 3124, 974, 3120, 978, 3116, 982, 3112, 987, 3106, 992, 3101, 997, 3096, 1002, 3091, 1007, 3086, 1012, 3082, 1016, 3075, 1023, 3069, 1029, 3064, 1034, 3059, 1039, 3050, 1048, 3044, 1053, 3038, 1060, 3032, 1065, 3026, 1071, 3016, 1082, 3010, 1088, 3003, 1095, 2992, 1106, 2982, 1116, 2971, 1126, 2966, 1132, 2950, 1147, 2945, 1152, 2939, 1159, 2933, 1164, 2927, 1171, 2921, 1176, 2916, 1182, 2910, 1188, 2904, 1193, 2899, 1199, 2894, 1203, 2890, 1207, 2881, 1216, 2876, 1221, 2872, 1226, 2867, 1230, 2864, 1234, 2857, 1241, 2851, 1246, 2847, 1251, 2842, 1255, 2839, 1258, 2833, 1265, 2827, 1270, 2823, 1275, 2818, 1279, 2814, 1283, 2809, 1288, 2799, 1298, 2794, 1303, 2788, 1310, 2782, 1316, 2776, 1322, 2765, 1332, 2760, 1338, 2754, 1343, 2749, 1348, 2743, 1354, 2738, 1359, 2734, 1363, 2634, 1464, 2614, 1483, 2604, 1493, 2599, 1499, 2592, 1505, 2583, 1515, 2577, 1520, 2571, 1526, 2566, 1531, 2560, 1537, 2555, 1542, 2551, 1547, 2546, 1551, 2543, 1555, 2538, 1561, 2532, 1566, 2527, 1571, 2522, 1576, 2517, 1581, 2513, 1585, 2508, 1590, 2503, 1595, 2498, 1600, 2493, 1605, 2488, 1610, 2484, 1614, 2480, 1618, 2476, 1622, 2472, 1626, 2468, 1630, 2463, 1634, 2459, 1639, 2455, 1642, 2452, 1645, 2449, 1649, 2445, 1653, 2441, 1657, 2437, 1661, 2433, 1665, 2429, 1669, 2425, 1673, 2421, 1677, 2417, 1681, 2413, 1684, 2410, 1688, 2406, 1691, 2403, 1695, 2399, 1700, 2394, 1704, 2390, 1708, 2386, 1712, 2383, 1714, 2380, 1718, 2377, 1720, 2375, 1723, 2371, 1727, 2368, 1729, 2365, 1733, 2362, 1735, 2359, 1738, 2356, 1742, 2353, 1745, 2349, 1749, 2345, 1753, 2341, 1757, 2337, 1760, 2334, 1764, 2330, 1768, 2326, 1772, 2323, 1776, 2318, 1780, 2315, 1783, 2311, 1787, 2307, 1790, 2305, 1793, 2301, 1804, 2291, 1809, 2285, 1814, 2281, 1818, 2276, 1822, 2273, 1825, 2270, 1828, 2267, 1830, 2265, 1833, 2262, 1835, 2259, 1839, 2256, 1842, 2253, 1845, 2250, 1848, 2247, 1852, 2243, 1856, 2238, 1860, 2235, 1863, 2232, 1866, 2229, 1869, 2226, 1872, 2223, 1875, 2220, 1878, 2217, 1883, 2212, 1889, 2206, 1894, 2201, 1898, 2198, 1901, 2194, 1905, 2190, 1910, 2186, 1914, 2181, 1920, 2175, 1925, 2170, 1931, 2165, 1935, 2160, 1940, 2155, 1945, 2151, 1950, 2145, 1955, 2140, 1959, 2136, 1963, 2133, 1965, 2130, 1968, 2127, 1971, 2125, 1972, 2123, 1975, 2121, 1976, 2119, 1978, 2118, 1980, 2115, 1982, 2114, 1984, 2112, 1985, 2110, 1987, 2109, 1988, 2108, 1989, 2106, 1991, 2105, 1992, 2104, 1993, 2103, 1994, 2101, 1996, 2100, 1997, 2099, 1997, 2099, 1998, 2098, 1999, 2096, 2000, 2096, 2001, 2095, 2002, 2094, 2003, 2092, 2004, 2092, 2005, 2091, 2006, 2090, 2006, 2090, 2007, 2089, 2007, 2089, 2008, 2088, 2008, 2088, 2009, 2085, 2016, 2080, 2016, 2080, 2018, 2078, 2018, 2078, 2018, 2078, 2018, 2078, 2018, 2078, 2018, 2078, 2018, 2078, 2018, 2078, 2018, 2078, 2018, 2078, 2018, 2078, 2018, 2078, 2016, 2080, 2016, 2080, 2016, 2080, 2016, 2080, 2016, 2080, 2016, 2082, 2016, 2080, 2016, 2080, 2016, 2080, 2016, 2080, 2016, 2080, 2016, 2080, 2016, 2080, 2016, 2080, 2016, 2080, 2016, 2080, 2016, 2080, 2017, 2079, 2018, 2, 4, 2072, 2018, 2, 4, 2072, 2022, 2075, 2021, 2075, 2021, 2, 4, 2069, 2021, 2, 4, 2069, 2019, 4, 4, 2070, 2018, 4, 4, 2070, 2016, 4, 8, 2068, 2016, 4, 8, 2068, 2016, 2, 4, 4, 2, 2068, 2016, 2, 4, 4, 2, 2069, 2021, 2075, 2021, 2075, 2025, 2071, 2025, 2072, 2020, 2076, 2020, 2076, 2020, 2076, 2020, 2076, 2016, 2081, 2015, 2081, 2015, 2081, 2016, 2080, 2016, 2081, 2015, 2081, 2016, 2080, 2016, 2081, 2015, 2081, 2015, 2081, 2016, 2081, 2015, 2081, 2015, 2082, 2014, 2082, 2015, 2081, 2015, 2081, 2015, 2081, 2015, 2081, 2016, 2080, 2016, 2080, 2016, 2080, 2016, 2080, 2016, 2080, 2016, 2080, 2016, 2080, 2016, 2080, 2016, 2081, 2015, 2081, 2015, 2081, 2015, 2081, 2015, 2081, 2015, 2081, 2015, 2081, 2015, 2081, 2016, 2080, 2016, 2081, 2015, 2081, 2015, 2081, 2014, 2082, 2014, 2082, 2014, 2082, 2014, 2082, 2014, 2082, 2014, 2082, 2014, 2083, 2012, 2084, 2012, 2084, 2012, 2084, 2012, 2084, 2012, 2084, 2012, 2084, 2012, 2084, 2012, 2085, 2011, 2085, 2010, 2086, 2010, 2086, 2010, 2086, 2010, 2086, 2010, 2086, 2010, 2086, 2010, 2086, 2010, 2086, 2010, 2086, 2010, 2086, 2010, 2086, 2010, 2086, 2010, 2086, 2010, 2086, 2010, 2086, 2010, 2086, 2010, 2086, 2010, 2087, 2009, 2087, 2009, 2087, 2009, 2087, 2009, 2087, 2009, 2087, 2009, 2087, 2009, 2087, 2009, 2087, 2009, 2088, 2007, 2089, 2007, 2089, 2007, 2089, 2007, 2090, 2005, 2091, 2005, 2091, 2005, 2091, 2004, 2093, 2003, 2093, 2003, 2093, 2002, 2094, 2002, 2094, 2001, 2095, 2001, 2095, 2000, 2096, 2000, 2096, 1999, 2098, 1997, 2099, 1997, 2099, 1996, 2100, 1996, 2101, 1995, 2101, 1995, 2101, 1995, 2101, 1995, 2101, 1995, 2102, 1994, 2102, 1994, 2102, 1994, 2102, 1994, 2103, 1993, 2103, 1993, 2103, 1993, 2104, 1992, 2104, 1992, 2104, 1992, 2105, 1991, 2105, 1991, 2106, 1990, 2106, 1990, 2106, 1990, 2107, 1989, 2107, 1989, 2107, 1989, 2108, 1988, 2108, 1988, 2108, 1988, 2109, 1987, 2109, 1987, 2109, 1987, 2110, 1986, 2110, 1986, 2110, 1986, 2110, 1986, 2110, 1986, 2110, 1986, 2110, 1986, 2111, 1985, 2111, 1985, 2111, 1985, 2111, 1985, 2111, 1985, 2111, 1985, 2111, 1985, 2111, 1985, 2111, 1985, 2111, 1985, 2111, 1985, 2111, 1985, 2111, 1985, 2111, 1985, 2111, 1985, 2111, 1985, 2111, 1985, 2111, 1985, 2111, 1985, 2111, 1985, 2111, 1985, 2111, 1985, 2111, 1985, 2111, 1985, 2112, 1984, 2112, 1984, 2112, 1984, 2112, 1984, 2112, 1984, 2112, 1984, 2112, 1983, 2113, 1983, 2113, 1983, 2114, 1982, 2114, 1982, 2114, 1982, 2114, 1982, 2114, 1982, 2114, 1982, 2114, 1981, 2115, 1981, 2115, 1981, 2116, 1980, 2116, 1979, 2117, 1979, 2117, 1979, 2118, 1978, 2118, 1978, 2118, 1977, 2119, 1977, 2119, 1977, 2120, 1976, 2120, 1975, 2121, 1975, 2121, 1975, 2121, 1975, 2121, 1975, 2121, 1974, 2122, 1974, 2123, 1973, 2123, 1973, 2123, 1972, 2124, 1972, 2124, 1972, 2124, 1971, 2125, 1971, 2125, 1971, 2125, 1970, 2126, 1970, 2127, 1968, 2128, 1968, 2128, 1967, 2129, 1967, 2129, 1966, 2130, 1965, 2131, 1965, 2131, 1964, 2132, 1964, 2133, 1963, 2133, 1962, 2134, 1962, 2134, 1962, 2134, 1961, 2135, 1961, 2135, 1960, 2136, 1960, 2137, 1958, 2138, 1958, 2138, 1957, 2139, 1956, 2140, 1956, 2140, 1955, 2141, 1955, 2141, 1955, 2141, 1954, 2143, 1953, 2143, 1953, 2143, 1952, 2144, 1952, 2144, 1952, 2144, 1951, 2145, 1951, 2145, 1951, 2145, 1950, 2147, 1949, 2147, 1949, 2147, 1949, 2147, 1949, 2147, 1949, 2147, 1949, 2147, 1949, 2147, 1949, 2147, 1949, 2148, 1948, 2148, 1948, 2148, 1947, 2149, 1947, 2149, 1947, 2149, 1947, 2149, 1946, 2150, 1946, 2150, 1946, 2151, 1944, 2152, 1944, 2152, 1944, 2152, 1944, 2152, 1944, 2152, 1944, 2152, 1944, 2152, 1944, 2152, 1944, 2153, 1942, 2154, 1942, 2154, 1942, 2154, 1942, 2154, 1941, 2155, 1941, 2155, 1941, 2155, 1941, 2155, 1941, 2156, 1939, 2157, 1939, 2157, 1939, 2157, 1939, 2158, 1938, 2158, 1938, 2158, 1938, 2158, 1938, 2158, 1938, 2159, 1936, 2160, 1936, 2160, 1936, 2160, 1936, 2160, 1936, 2160, 1936, 2160, 1936, 2160, 1936, 2160, 1936, 2161, 1934, 2162, 1934, 2162, 1934, 2162, 1934, 2162, 1933, 2163, 1933, 2163, 1933, 2163, 1932, 2164, 1932, 2165, 1931, 2165, 1930, 2166, 1930, 2166, 1929, 2167, 1929, 2167, 1928, 2168, 1928, 2168, 1927, 2169, 1926, 2170, 1926, 2170, 1925, 2171, 1924, 2172, 1923, 2173, 1922, 2174, 1921, 2175, 1920, 2176, 1919, 2177, 1918, 2179, 1916, 2180, 1916, 2180, 1915, 2181, 1915, 2182, 1914, 2182, 1913, 2183, 1913, 2184, 1912, 2184, 1911, 2185, 1911, 2185, 1911, 2185, 1910, 2186, 1910, 2186, 1910, 2186, 1909, 2187, 1909, 2187, 1909, 2187, 1908, 2189, 1907, 2189, 1906, 2190, 1906, 2190, 1905, 2192, 1904, 2192, 1903, 2193, 1903, 2193, 1903, 2194, 1902, 2194, 1901, 2195, 1901, 2195, 1901, 2196, 1900, 2196, 1900, 2196, 1899, 2198, 1898, 2198, 1898, 2198, 1898, 2199, 1896, 2200, 1896, 2200, 1896, 2201, 1895, 2201, 1894, 2202, 1894, 2202, 1894, 2202, 1894, 2203, 1893, 2203, 1893, 2203, 1893, 2203, 1893, 2204, 1892, 2204, 1892, 2204, 1892, 2204, 1892, 2204, 1892, 2205, 1891, 2205, 1891, 2205, 1891, 2205, 1891, 2205, 1891, 2205, 1891, 2205, 1891, 2205, 1891, 2205, 1891, 2206, 1890, 2206, 1890, 2206, 1890, 2206, 1890, 2207, 1889, 2207, 1889, 2207, 1889, 2207, 1889, 2208, 1888, 2208, 1886, 2210, 1886, 2210, 1880, 2217, 1879, 2217, 1879, 2217, 1878, 2218, 1878, 2218, 1878, 2219, 1876, 2220, 1876, 2220, 1876, 2220, 1876, 2221, 1875, 2221, 1874, 2222, 1874, 2222, 1874, 2222, 1874, 2223, 1873, 2223, 1873, 2223, 1873, 2223, 1873, 2224, 1872, 2224, 1871, 2225, 1871, 2225, 1871, 2225, 1871, 2226, 1869, 2227, 1869, 2227, 1869, 2227, 1869, 2228, 1867, 2229, 1867, 2229, 1867, 2230, 1866, 2230, 1866, 2230, 1866, 2231, 1865, 2231, 1865, 2232, 1864, 2232, 1864, 2232, 1863, 2233, 1863, 2233, 1863, 2234, 1862, 2234, 1861, 2235, 1861, 2235, 1861, 2236, 1859, 2237, 1859, 2237, 1859, 2237, 1858, 2238, 1858, 2239, 1856, 2240, 1856, 2240, 1856, 2240, 1856, 2241, 1855, 2241, 1854, 2242, 1854, 2242, 1854, 2243, 1853, 2243, 1852, 2244, 1852, 2244, 1852, 2245, 1850, 2246, 1850, 2246, 1850, 2246, 1849, 2247, 1849, 2248, 1847, 2249, 1847, 2249, 1847, 2249, 1847, 2250, 1846, 2250, 1845, 2251, 1845, 2251, 1845, 2251, 1845, 2252, 1843, 2253, 1843, 2253, 1843, 2253, 1842, 2255, 1841, 2255, 1841, 2255, 1840, 2256, 1840, 2257, 1838, 2258, 1838, 2258, 1837, 2259, 1837, 2260, 1835, 2261, 1835, 2261, 1834, 2263, 1833, 2263, 1832, 2264, 1832, 2265, 1830, 2266, 1830, 2266, 1829, 2268, 1828, 2268, 1827, 2269, 1827, 2269, 1826, 2271, 1825, 2271, 1825, 2271, 1825, 2271, 1825, 2272, 1823, 2273, 1823, 2273, 1823, 2274, 1822, 2274, 1821, 2275, 1821, 2276, 1820, 2276, 1819, 2278, 1818, 2278, 1818, 2279, 1816, 2280, 1816, 2281, 1814, 2282, 1814, 2282, 1814, 2283, 1812, 2284, 1812, 2285, 1811, 2285, 1810, 2286, 1810, 2287, 1809, 2287, 1808, 2288, 1808, 2289, 1807, 2289, 1807, 2290, 1806, 2290, 1805, 2291, 1805, 2292, 1804, 2292, 1804, 2292, 1803, 2294, 1802, 2294, 1802, 2294, 1801, 2296, 1800, 2296, 1800, 2296, 1799, 2297, 1799, 2297, 1798, 2299, 1797, 2299, 1796, 2300, 1796, 2300, 1795, 2302, 1793, 2303, 1793, 2303, 1792, 2305, 1790, 2306, 1790, 2306, 1790, 2307, 1788, 2308, 1788, 2309, 1787, 2309, 1786, 2310, 1786, 2311, 1785, 2311, 1784, 2312, 1784, 2313, 1782, 2314, 1782, 2315, 1781, 2315, 1781, 2315, 1781, 2315, 1780, 2316, 1780, 2317, 1779, 2317, 1779, 2317, 1779, 2317, 1779, 2318, 1778, 2318, 1778, 2318, 1778, 2318, 1777, 2320, 1776, 2320, 1776, 2320, 1776, 2320, 1775, 2322, 1774, 2322, 1774, 2322, 1774, 2323, 1773, 2323, 1772, 2324, 1772, 2325, 1771, 2325, 1771, 2325, 1770, 2327, 1769, 2327, 1769, 2327, 1769, 2327, 1768, 2329, 1767, 2329, 1767, 2329, 1767, 2329, 1766, 2331, 1765, 2331, 1765, 2331, 1764, 2332, 1764, 2332, 1764, 2333, 1762, 2334, 1762, 2334, 1761, 2335, 1761, 2336, 1759, 2337, 1759, 2337, 1758, 2338, 1757, 2339, 1757, 2340, 1755, 2341, 1754, 2342, 1753, 2343, 1752, 2345, 1750, 2346, 1749, 2347, 1748, 2349, 1746, 2350, 1745, 2351, 1744, 2353, 1741, 2355, 1740, 2356, 1738, 2358, 1738, 2358, 1737, 2359, 1737, 2359, 1737, 2360, 1735, 2361, 1735, 2361, 1734, 2362, 1734, 2363, 1732, 2364, 1732, 2364, 1731, 2365, 1731, 2365, 1731, 2366, 1730, 2366, 1730, 2366, 1730, 2366, 1730, 2366, 1730, 2366, 1730, 2366, 1730, 2366, 1730, 2366, 1730, 2367, 1729, 2367, 1729, 2367, 1728, 2368, 1728, 2369, 1727, 2369, 1727, 2369, 1726, 2371, 1725, 2371, 1725, 2371, 1725, 2372, 1724, 2372, 1723, 2373, 1723, 2374, 1722, 2374, 1722, 2374, 1721, 2375, 1721, 2376, 1720, 2376, 1720, 2376, 1720, 2376, 1719, 2378, 1718, 2378, 1718, 2378, 1718, 2378, 1717, 2379, 1717, 2380, 1716, 2380, 1715, 2381, 1715, 2381, 1715, 2382, 1713, 2383, 1713, 2383, 1712, 2385, 1711, 2385, 1711, 2385, 1710, 2387, 1709, 2387, 1709, 2388, 1707, 2389, 1707, 2389, 1707, 2390, 1705, 2391, 1705, 2391, 1705, 2392, 1703, 2393, 1703, 2394, 1702, 2394, 1701, 2395, 1701, 2396, 1699, 2397, 1699, 2397, 1699, 2398, 1697, 2399, 1697, 2399, 1697, 2400, 1695, 2401, 1695, 2401, 1694, 2402, 1694, 2403, 1693, 2403, 1692, 2404, 1692, 2404, 1692, 2405, 1690, 2406, 1690, 2406, 1690, 2407, 1688, 2408, 1688, 2408, 1688, 2409, 1687, 2409, 1686, 2411, 1685, 2411, 1685, 2412, 1684, 2412, 1683, 2414, 1682, 2414, 1682, 2414, 1681, 2416, 1680, 2416, 1680, 2417, 1678, 2418, 1678, 2419, 1677, 2419, 1676, 2421, 1675, 2422, 1674, 2422, 1673, 2424, 1672, 2424, 1672, 2424, 1671, 2426, 1670, 2426, 1669, 2427, 1669, 2428, 1667, 2429, 1667, 2430, 1665, 2431, 1665, 2432, 1663, 2433, 1663, 2434, 1662, 2434, 1661, 2436, 1660, 2436, 1660, 2437, 1658, 2438, 1658, 2439, 1657, 2439, 1656, 2441, 1655, 2442, 1653, 2443, 1653, 2444, 1652, 2445, 1650, 2447, 1649, 2447, 1649, 2448, 1647, 2450, 1646, 2450, 1646, 2451, 1645, 2452, 1644, 2453, 1643, 2453, 1643, 2454, 1642, 2455, 1641, 2455, 1641, 2456, 1640, 2457, 1639, 2457, 1639, 2457, 1639, 2458, 1638, 2458, 1638, 2458, 1638, 2459, 1637, 2459, 1637, 2460, 1636, 2460, 1636, 2461, 1635, 2461, 1635, 2462, 1627, 2469, 1627, 2470, 1625, 2471, 1625, 2472, 1624, 2472, 1623, 2474, 1622, 2474, 1622, 2475, 1620, 2477, 1619, 2477, 1619, 2477, 1618, 2479, 1617, 2479, 1616, 2480, 1616, 2480, 1616, 2480, 1615, 2481, 1615, 2481, 1615, 2481, 1614, 2483, 1613, 2483, 1613, 2483, 1612, 2484, 1612, 2485, 1611, 2485, 1610, 2486, 1610, 2486, 1610, 2486, 1609, 2488, 1608, 2488, 1607, 2489, 1607, 2489, 1607, 2489, 1607, 2489, 1606, 2490, 1606, 2490, 1606, 2490, 1606, 2491, 1604, 2492, 1604, 2492, 1604, 2492, 1603, 2494, 1602, 2494, 1602, 2494, 1601, 2495, 1601, 2495, 1601, 2496, 1599, 2497, 1599, 2497, 1599, 2497, 1598, 2498, 1598, 2498, 1598, 2498, 1597, 2499, 1597, 2499, 1596, 2501, 1595, 2501, 1594, 2502, 1594, 2502, 1593, 2504, 1592, 2504, 1591, 2505, 1591, 2505, 1590, 2506, 1590, 2507, 1589, 2507, 1589, 2507, 1589, 2507, 1589, 2508, 1588, 2508, 1588, 2508, 1588, 2508, 1588, 2508, 1588, 2509, 1587, 2509, 1587, 2509, 1587, 2509, 1587, 2509, 1585, 2511, 1585, 2511, 1585, 2511, 1585, 2511, 1585, 2512, 1584, 2512, 1581, 2515, 1580, 2516, 1580, 2516, 1580, 2516, 1580, 2516, 1579, 2517, 1579, 2517, 1579, 2517, 1579, 2517, 1578, 2518, 1578, 2518, 1578, 2518, 1578, 2518, 1577, 2519, 1577, 2519, 1577, 2519, 1576, 2520, 1576, 2520, 1576, 2520, 1575, 2521, 1575, 2521, 1575, 2521, 1574, 2522, 1574, 2522, 1574, 2522, 1573, 2523, 1573, 2523, 1573, 2523, 1572, 2524, 1572, 2524, 1571, 2525, 1571, 2525, 1571, 2525, 1570, 2526, 1570, 2527, 1569, 2527, 1568, 2528, 1568, 2528, 1568, 2529, 1566, 2530, 1566, 2530, 1566, 2530, 1566, 2530, 1565, 2532, 1564, 2532, 1564, 2532, 1564, 2532, 1563, 2534, 1562, 2534, 1562, 2534, 1562, 2534, 1562, 2534, 1561, 2536, 1560, 2536, 1560, 2536, 1560, 2536, 1559, 2537, 1559, 2537, 1559, 2537, 1558, 2538, 1558, 2538, 1558, 2538, 1557, 2539, 1557, 2539, 1557, 2539, 1557, 2539, 1557, 2539, 1557, 2539, 1557, 2539, 1556, 2540, 1556, 2541, 1555, 2541, 1555, 2541, 1554, 2542, 1554, 2542, 1554, 2542, 1553, 2543, 1553, 2543, 1553, 2543, 1552, 2545, 1551, 2545, 1551, 2545, 1551, 2545, 1551, 2545, 1551, 2545, 1551, 2545, 1551, 2545, 1550, 2546, 1550, 2546, 1550, 2546, 1550, 2546, 1549, 2547, 1549, 2547, 1549, 2547, 1549, 2547, 1549, 2547, 1548, 2548, 1548, 2548, 1548, 2548, 1548, 2548, 1547, 2549, 1547, 2549, 1547, 2549, 1547, 2549, 1547, 2549, 1547, 2549, 1547, 2550, 1546, 2550, 1544, 2552, 1544, 2552, 1544, 2552, 1543, 2553, 1543, 2553, 1543, 2553, 1542, 2554, 1542, 2554, 1542, 2554, 1541, 2555, 1541, 2555, 1541, 2555, 1540, 2556, 1540, 2556, 1540, 2556, 1539, 2557, 1539, 2557, 1539, 2557, 1538, 2558, 1538, 2558, 1538, 2558, 1538, 2558, 1537, 2559, 1537, 2559, 1537, 2559, 1537, 2559, 1536, 2560, 1536, 2560, 1536, 2560, 1535, 2561, 1535, 2561, 1535, 2561, 1534, 2562, 1534, 2562, 1533, 2563, 1533, 2563, 1533, 2563, 1533, 2563, 1533, 2563, 1532, 2564, 1532, 2564, 1532, 2564, 1532, 2564, 1531, 2565, 1531, 2565, 1531, 2565, 1531, 2565, 1531, 2565, 1530, 2566, 1530, 2566, 1530, 2566, 1530, 2566, 1529, 2567, 1529, 2567, 1529, 2567, 1528, 2568, 1528, 2568, 1528, 2568, 1527, 2569, 1527, 2569, 1526, 2570, 1526, 2570, 1525, 2571, 1525, 2571, 1524, 2572, 1524, 2572, 1524, 2572, 1524, 2572, 1523, 2573, 1523, 2573, 1523, 2573, 1523, 2573, 1523, 2573, 1522, 2574, 1522, 2574, 1522, 2574, 1522, 2574, 1521, 2575, 1521, 2575, 1521, 2575, 1521, 2575, 1521, 2575, 1520, 2576, 1520, 2576, 1520, 2576, 1520, 2576, 1519, 2577, 1519, 2577, 1519, 2577, 1518, 2578, 1518, 2578, 1518, 2578, 1517, 2579, 1517, 2579, 1516, 2580, 1516, 2580, 1516, 2580, 1516, 2580, 1515, 2581, 1515, 2581, 1515, 2581, 1515, 2581, 1514, 2582, 1514, 2582, 1514, 2582, 1514, 2582, 1514, 2582, 1514, 2582, 1514, 2582, 1514, 2581, 1514, 2582, 1513, 2583, 1513, 2583, 1513, 2583, 1513, 2583, 1513, 2583, 1512, 2584, 1511, 2585, 1511, 2585, 1510, 2586, 1510, 2586, 1509, 2587, 1509, 2587, 1508, 2588, 1508, 2588, 1508, 2588, 1507, 2589, 1507, 2589, 1506, 2590, 1506, 2590, 1506, 2590, 1505, 2591, 1505, 2591, 1505, 2591, 1504, 2592, 1504, 2592, 1503, 2592, 1504, 2592, 1504, 2592, 1504, 2592, 1504, 2592, 1504, 2592, 1504, 2592, 1504, 2592, 1504, 2592, 1503, 2592, 1504, 2592, 1504, 2592, 1504, 2592, 1503, 2593, 1503, 2593, 1503, 2593, 1502, 2594, 1502, 2594, 1502, 2594, 1501, 2595, 1501, 2595, 1500, 2596, 1500, 2596, 1500, 2596, 1500, 2596, 1500, 2596, 1499, 2597, 1499, 2597, 1499, 2597, 1499, 2597, 1498, 2598, 1498, 2598, 1498, 2598, 1497, 2599, 1497, 2599, 1497, 2599, 1496, 2600, 1496, 2600, 1495, 2601, 1495, 2601, 1494, 2602, 1494, 2602, 1493, 2603, 1493, 2603, 1492, 2604, 1492, 2604, 1491, 2605, 1491, 2605, 1490, 2606, 1490, 2606, 1489, 2607, 1489, 2607, 1488, 2608, 1488, 2608, 1487, 2609, 1487, 2609, 1486, 2610, 1486, 2610, 1485, 2611, 1484, 2612, 1484, 2612, 1483, 2613, 1483, 2613, 1482, 2614, 1482, 2614, 1481, 2615, 1480, 2616, 1480, 2616, 1479, 2617, 1478, 2618, 1477, 2619, 1477, 2619, 1476, 2620, 1476, 2620, 1475, 2621, 1475, 2621, 1475, 2621, 1475, 2621, 1475, 2621, 1475, 2621, 1474, 2622, 1474, 2622, 1474, 2622, 1474, 2622, 1473, 2623, 1473, 2623, 1473, 2623, 1472, 2624, 1472, 2624, 1472, 2624, 1471, 2625, 1471, 2625, 1470, 2626, 1470, 2626, 1470, 2626, 1469, 2627, 1469, 2627, 1469, 2627, 1468, 2628, 1468, 2628, 1468, 2627, 1468, 2628, 1468, 2628, 1468, 2628, 1467, 2629, 1467, 2629, 1467, 2629, 1466, 2630, 1466, 2630, 1465, 2630, 1466, 2630, 1465, 2631, 1465, 2631, 1465, 2631, 1464, 2632, 1464, 2632, 1464, 2632, 1463, 2633, 1463, 2633, 1463, 2633, 1462, 2634, 1462, 2634, 1462, 2634, 1461, 2635, 1461, 2635, 1460, 2636, 1460, 2636, 1459, 2637, 1459, 2637, 1458, 2638, 1458, 2638, 1457, 2639, 1457, 2639, 1457, 2639, 1456, 2640, 1456, 2640, 1456, 2640, 1455, 2641, 1455, 2641, 1455, 2641, 1455, 2641, 1455, 2641, 1455, 2641, 1455, 2641, 1455, 2641, 1455, 2641, 1453, 2643, 1453, 2643, 1451, 2645, 1451, 2645, 1453, 2643, 1453, 2643, 1451, 2645, 1451, 2645, 1451, 2645, 1450, 2646, 1450, 2646, 1450, 2646, 1450, 2646, 1449, 2647, 1449, 2647, 1449, 2647, 1449, 2647, 1448, 2648, 1448, 2648, 1448, 2648, 1448, 2648, 1447, 2649, 1447, 2649, 1447, 2649, 1446, 2650, 1446, 2650, 1446, 2650, 1445, 2651, 1445, 2651, 1445, 2651, 1445, 2651, 1445, 2651, 1445, 2651, 1445, 2651, 1445, 2651, 1444, 2652, 1444, 2652, 1444, 2652, 1444, 2652, 1443, 2653, 1443, 2653, 1443, 2653, 1442, 2654, 1442, 2654, 1442, 2654, 1441, 2655, 1441, 2655, 1441, 2655, 1440, 2656, 1440, 2656, 1440, 2656, 1440, 2656, 1439, 2657, 1439, 2657, 1439, 2657, 1439, 2657, 1438, 2658, 1438, 2658, 1438, 2658, 1437, 2659, 1437, 2659, 1437, 2659, 1436, 2660, 1436, 2660, 1436, 2660, 1435, 2661, 1435, 2661, 1435, 2661, 1435, 2661, 1434, 2662, 1434, 2662, 1434, 2662, 1434, 2662, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1432, 2664, 1432, 2664, 1432, 2665, 1431, 2665, 1430, 2666, 1430, 2666, 1430, 2667, 1428, 2668, 1428, 2668, 1428, 2668, 1427, 2669, 1427, 2669, 1426, 2670, 1426, 2670, 1426, 2670, 1426, 2670, 1426, 2670, 1425, 2671, 1425, 2671, 1425, 2671, 1425, 2672, 1423, 2673, 1423, 2673, 1423, 2673, 1423, 2674, 1421, 2675, 1421, 2675, 1421, 2676, 1420, 2676, 1419, 2677, 1419, 2678, 1418, 2678, 1418, 2679, 1417, 2679, 1416, 2681, 1415, 2681, 1415, 2681, 1415, 2681, 1415, 2681, 1415, 2682, 1414, 2682, 1414, 2682, 1414, 2682, 1414, 2683, 1413, 2683, 1413, 2683, 1413, 2683, 1413, 2684, 1412, 2684, 1412, 2684, 1412, 2684, 1412, 2684, 1412, 2684, 1412, 2684, 1410, 2686, 1410, 2686, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1408, 2688, 1408, 2688, 1408, 2688, 1408, 2688, 1407, 2689, 1407, 2689, 1407, 2689, 1407, 2689, 1407, 2689, 1406, 2690, 1406, 2690, 1406, 2690, 1406, 2690, 1405, 2691, 1405, 2691, 1405, 2691, 1404, 2692, 1404, 2691, 1405, 2691, 1404, 2692, 1404, 2691, 1405, 2691, 1405, 2691, 1405, 2691, 1405, 2690, 1406, 2690, 1406, 2690, 1406, 2690, 1406, 2690, 1406, 2690, 1406, 2690, 1406, 2690, 1406, 2690, 1406, 2690, 1406, 2690, 1406, 2690, 1406, 2690, 1406, 2690, 1407, 2688, 1408, 2688, 1408, 2688, 1408, 2688, 1408, 2687, 1410, 2686, 1410, 2686, 1410, 2686, 1410, 2686, 1411, 2685, 1411, 2685, 1411, 2685, 1411, 2685, 1411, 2684, 1412, 2684, 1412, 2684, 1413, 2683, 1413, 2682, 1414, 2682, 1415, 2681, 1415, 2681, 1415, 2680, 1417, 2679, 1417, 2679, 1417, 2679, 1417, 2679, 1418, 2678, 1418, 2678, 1418, 2678, 1418, 2678, 1418, 2678, 1418, 2677, 1419, 2677, 1419, 2677, 1419, 2677, 1420, 2676, 1420, 2676, 1420, 2676, 1420, 2676, 1420, 2676, 1421, 2674, 1422, 2674, 1423, 2673, 1423, 2673, 1423, 2673, 1424, 2672, 1424, 2672, 1424, 2672, 1425, 2671, 1425, 2670, 1427, 2669, 1427, 2669, 1427, 2669, 1428, 2668, 1428, 2668, 1428, 2668, 1429, 2667, 1429, 2667, 1429, 2666, 1430, 2666, 1431, 2665, 1431, 2665, 1431, 2665, 1431, 2665, 1431, 2665, 1432, 2664, 1432, 2663, 1433, 2663, 1433, 2663, 1434, 2662, 1434, 2662, 1434, 2662, 1434, 2662, 1434, 2662, 1434, 2662, 1434, 2661, 1435, 2661, 1435, 2661, 1435, 2661, 1435, 2661, 1435, 2661, 1435, 2661, 1435, 2661, 1435, 2661, 1435, 2660, 1436, 2660, 1436, 2660, 1436, 2660, 1436, 2660, 1436, 2660, 1436, 2660, 1436, 2660, 1436, 2660, 1436, 2660, 1436, 2660, 1436, 2660, 1436, 2660, 1436, 2660, 1436, 2659, 1437, 2659, 1437, 2659, 1437, 2659, 1437, 2658, 1438, 2658, 1438, 2658, 1438, 2658, 1438, 2658, 1438, 2658, 1438, 2658, 1438, 2658, 1438, 2658, 1438, 2658, 1438, 2658, 1438, 2658, 1438, 2658, 1438, 2658, 1438, 2658, 1438, 2658, 1438, 2658, 1438, 2658, 1438, 2658, 1438, 2658, 1438, 2658, 1438, 2658, 1438, 2658, 1438, 2658, 1438, 2658, 1438, 2658, 1438, 2658, 1438, 2658, 1438, 2658, 1438, 2658, 1437, 2659, 1437, 2659, 1437, 2659, 1437, 2659, 1436, 2660, 1436, 2660, 1436, 2660, 1435, 2661, 1435, 2661, 1435, 2661, 1434, 2662, 1434, 2662, 1433, 2663, 1433, 2663, 1432, 2664, 1432, 2664, 1431, 2665, 1430, 2666, 1430, 2666, 1429, 2667, 1428, 2668, 1428, 2668, 1428, 2668, 1428, 2668, 1428, 2668, 1428, 2668, 1428, 2668, 1428, 2668, 1428, 2668, 1428, 2668, 1428, 2668, 1428, 2668, 1428, 2668, 1429, 2667, 1429, 2667, 1429, 2667, 1429, 2667, 1430, 2666, 1430, 2666, 1430, 2666, 1430, 2666, 1430, 2666, 1430, 2667, 1429, 2667, 1429, 2667, 1429, 2667, 1429, 2667, 1429, 2667, 1429, 2667, 1429, 2667, 1429, 2667, 1429, 2668, 1428, 2668, 1428, 2668, 1428, 2668, 1429, 2667, 1429, 2667, 1429, 2667, 1429, 2667, 1429, 2667, 1429, 2668, 1428, 2668, 1428, 2668, 1428, 2668, 1429, 2667, 1429, 2667, 1429, 2667, 1429, 2667, 1429, 2668, 1429, 2667, 1429, 2667, 1429, 2667, 1429, 2667, 1430, 2666, 1430, 2666, 1430, 2666, 1430, 2666, 1430, 2667, 1429, 2667, 1429, 2667, 1429, 2667, 1429, 2667, 1430, 2666, 1430, 2666, 1430, 2666, 1430, 2666, 1430, 2666, 1430, 2666, 1430, 2666, 1430, 2666, 1430, 2666, 1431, 2665, 1431, 2665, 1431, 2665, 1431, 2665, 1431, 2666, 1430, 2666, 1430, 2666, 1430, 2666, 1430, 2666, 1431, 2665, 1431, 2665, 1431, 2665, 1431, 2665, 1431, 2665, 1432, 2664, 1432, 2664, 1432, 2664, 1432, 2664, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2664, 1432, 2664, 1432, 2664, 1432, 2664, 1432, 2664, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2664, 1432, 2664, 1432, 2664, 1432, 2664, 1432, 2664, 1432, 2664, 1432, 2664, 1432, 2664, 1432, 2664, 1432, 2664, 1432, 2664, 1432, 2664, 1432, 2664, 1432, 2664, 1432, 2664, 1432, 2664, 1432, 2664, 1432, 2664, 1432, 2664, 1432, 2664, 1432, 2664, 1432, 2664, 1432, 2664, 1432, 2664, 1432, 2664, 1432, 2664, 1432, 2664, 1432, 2664, 1432, 2665, 1431, 2665, 1431, 2665, 1431, 2665, 1432, 2664, 1432, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2662, 1434, 2662, 1434, 2662, 1434, 2662, 1434, 2662, 1434, 2662, 1434, 2662, 1434, 2662, 1435, 2661, 1435, 2661, 1435, 2661, 1435, 2661, 1435, 2661, 1435, 2661, 1435, 2661, 1435, 2661, 1435, 2661, 1435, 2661, 1435, 2661, 1435, 2661, 1435, 2661, 1435, 2661, 1436, 2660, 1436, 2660, 1436, 2660, 1436, 2660, 1437, 2659, 1437, 2659, 1437, 2659, 1437, 2659, 1437, 2659, 1437, 2659, 1437, 2659, 1437, 2659, 1437, 2659, 1437, 2659, 1437, 2659, 1437, 2659, 1437, 2659, 1437, 2659, 1437, 2659, 1437, 2659, 1437, 2659, 1437, 2659, 1438, 2659, 1437, 2659, 1437, 2659, 1437, 2659, 1437, 2660, 1436, 2660, 1436, 2660, 1436, 2660, 1436, 2660, 1436, 2661, 1435, 2661, 1435, 2661, 1435, 2661, 1435, 2661, 1435, 2661, 1435, 2661, 1435, 2661, 1435, 2661, 1435, 2661, 1435, 2661, 1435, 2661, 1435, 2661, 1435, 2661, 1435, 2661, 1435, 2661, 1435, 2661, 1434, 2662, 1434, 2662, 1434, 2662, 1434, 2662, 1434, 2662, 1434, 2662, 1434, 2662, 1434, 2662, 1434, 2662, 1434, 2662, 1434, 2662, 1434, 2662, 1434, 2662, 1434, 2662, 1434, 2662, 1434, 2662, 1434, 2662, 1434, 2662, 1434, 2662, 1434, 2662, 1434, 2662, 1434, 2662, 1434, 2662, 1434, 2662, 1434, 2662, 1434, 2662, 1434, 2662, 1434, 2662, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1432, 2664, 1432, 2664, 1432, 2664, 1432, 2664, 1431, 2665, 1431, 2665, 1431, 2666, 1430, 2666, 1429, 2667, 1429, 2667, 1429, 2667, 1428, 2668, 1428, 2668, 1428, 2668, 1427, 2669, 1427, 2670, 1426, 2670, 1425, 2671, 1425, 2671, 1425, 2671, 1424, 2672, 1424, 2672, 1424, 2672, 1423, 2673, 1423, 2673, 1422, 2674, 1422, 2674, 1422, 2674, 1422, 2674, 1422, 2674, 1421, 2675, 1421, 2675, 1421, 2675, 1421, 2675, 1420, 2676, 1420, 2676, 1420, 2676, 1419, 2677, 1419, 2677, 1419, 2677, 1418, 2678, 1418, 2678, 1417, 2679, 1417, 2679, 1417, 2679, 1416, 2680, 1416, 2680, 1416, 2680, 1415, 2681, 1415, 2681, 1414, 2682, 1414, 2682, 1414, 2681, 1414, 2682, 1414, 2682, 1414, 2682, 1413, 2683, 1413, 2682, 1414, 2682, 1413, 2683, 1413, 2683, 1413, 2682, 1413, 2683, 1413, 2683, 1413, 2683, 1412, 2684, 1412, 2684, 1411, 2685, 1411, 2685, 1411, 2685, 1411, 2684, 1412, 2684, 1411, 2685, 1411, 2685, 1411, 2685, 1411, 2685, 1410, 2686, 1410, 2686, 1410, 2686, 1410, 2685, 1411, 2685, 1410, 2686, 1410, 2686, 1410, 2686, 1410, 2686, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2686, 1410, 2686, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2686, 1409, 2687, 1409, 2687, 1409, 2687, 1408, 2687, 1409, 2687, 1409, 2687, 1408, 2688, 1408, 2688, 1407, 2689, 1407, 2689, 1407, 2689, 1407, 2688, 1407, 2689, 1407, 2689, 1407, 2689, 1407, 2689, 1406, 2690, 1406, 2690, 1406, 2690, 1405, 2691, 1405, 2691, 1405, 2691, 1404, 2692, 1404, 2692, 1403, 2693, 1403, 2693, 1402, 2694, 1402, 2694, 1401, 2695, 1401, 2695, 1401, 2695, 1400, 2696, 1400, 2696, 1399, 2697, 1399, 2697, 1398, 2698, 1398, 2698, 1397, 2699, 1397, 2699, 1397, 2699, 1397, 2699, 1396, 2700, 1396, 2700, 1396, 2700, 1396, 2700, 1396, 2700, 1395, 2701, 1395, 2701, 1395, 2701, 1395, 2701, 1394, 2702, 1394, 2702, 1394, 2702, 1394, 2702, 1394, 2702, 1394, 2702, 1394, 2702, 1394, 2702, 1394, 2702, 1394, 2702, 1394, 2702, 1394, 2702, 1394, 2702, 1394, 2702, 1392, 2704, 1392, 2704, 1394, 2702, 1394, 2702, 1394, 2702, 1394, 2702, 1392, 2704, 1392, 2704, 1392, 2704, 1392, 2704, 1390, 2707, 1389, 2707, 1389, 2707, 1389, 2707, 1389, 2708, 1388, 2708, 1387, 2709, 1387, 2709, 1387, 2709, 1387, 2710, 1386, 2710, 1386, 2710, 1386, 2710, 1386, 2711, 1385, 2711, 1384, 2712, 1384, 2713, 1383, 2713, 1383, 2713, 1382, 2715, 1381, 2715, 1381, 2715, 1381, 2715, 1380, 2716, 1380, 2716, 1380, 2716, 1380, 2716, 1380, 2717, 1379, 2717, 1379, 2717, 1379, 2717, 1379, 2718, 1377, 2719, 1377, 2719, 1377, 2720, 1376, 2720, 1375, 2721, 1375, 2722, 1374, 2722, 1374, 2723, 1373, 2723, 1372, 2724, 1372, 2725, 1371, 2725, 1371, 2725, 1370, 2727, 1369, 2727, 1369, 2728, 1368, 2728, 1368, 2728, 1367, 2730, 1366, 2730, 1366, 2730, 1366, 2731, 1364, 2732, 1364, 2732, 1364, 2733, 1363, 2733, 1363, 2733, 1362, 2734, 1362, 2735, 1361, 2735, 1361, 2735, 1360, 2736, 1360, 2737, 1359, 2737, 1359, 2737, 1358, 2739, 1357, 2739, 1357, 2739, 1357, 2740, 1355, 2741, 1355, 2741, 1355, 2742, 1354, 2742, 1354, 2742, 1353, 2744, 1352, 2744, 1352, 2744, 1352, 2745, 1351, 2745, 1351, 2746, 1350, 2746, 1350, 2746, 1350, 2747, 1348, 2748, 1348, 2748, 1348, 2749, 1347, 2749, 1346, 2750, 1346, 2751, 1345, 2751, 1344, 2752, 1344, 2753, 1343, 2753, 1342, 2754, 1342, 2755, 1340, 2756, 1340, 2757, 1339, 2757, 1338, 2759, 1337, 2759, 1337, 2760, 1335, 2761, 1335, 2762, 1333, 2763, 1333, 2764, 1332, 2764, 1331, 2765, 1331, 2766, 1330, 2766, 1329, 2767, 1329, 2768, 1328, 2768, 1327, 2770, 1326, 2770, 1326, 2770, 1325, 2772, 1324, 2772, 1324, 2772, 1323, 2774, 1322, 2774, 1321, 2776, 1320, 2776, 1319, 2778, 1318, 2778, 1317, 2780, 1315, 2781, 1315, 2782, 1313, 2783, 1312, 2784, 1311, 2786, 1309, 2787, 1309, 2787, 1309, 2788, 1308, 2788, 1307, 2789, 1307, 2790, 1306, 2790, 1306, 2791, 1304, 2792, 1304, 2792, 1304, 2793, 1303, 2793, 1302, 2794, 1302, 2795, 1301, 2795, 1301, 2796, 1299, 2797, 1299, 2797, 1299, 2798, 1298, 2798, 1297, 2799, 1297, 2800, 1296, 2800, 1296, 2800, 1296, 2801, 1294, 2802, 1294, 2802, 1294, 2803, 1293, 2803, 1292, 2804, 1292, 2805, 1291, 2805, 1290, 2807, 1289, 2807, 1289, 2808, 1287, 2809, 1287, 2810, 1286, 2810, 1285, 2812, 1284, 2812, 1284, 2812, 1284, 2813, 1282, 2814, 1282, 2814, 1282, 2815, 1281, 2815, 1281, 2815, 1281, 2816, 1280, 2816, 1280, 2817, 1279, 2817, 1278, 2818, 1278, 2819, 1277, 2819, 1277, 2819, 1276, 2821, 1275, 2821, 1275, 2822, 1274, 2822, 1274, 2822, 1273, 2824, 1272, 2824, 1272, 2824, 1272, 2825, 1270, 2826, 1270, 2827, 1269, 2827, 1269, 2827, 1268, 2829, 1267, 2829, 1267, 2829, 1267, 2830, 1265, 2831, 1265, 2832, 1264, 2832, 1264, 2833, 1263, 2833, 1262, 2835, 1261, 2835, 1261, 2836, 1260, 2836, 1259, 2838, 1258, 2838, 1258, 2838, 1258, 2838, 1257, 2839, 1257, 2840, 1256, 2840, 1256, 2840, 1256, 2840, 1256, 2840, 1256, 2841, 1255, 2841, 1255, 2842, 1253, 2843, 1253, 2843, 1253, 2844, 1252, 2844, 1251, 2846, 1250, 2846, 1250, 2847, 1248, 2848, 1248, 2849, 1247, 2850, 1245, 2851, 1245, 2852, 1244, 2853, 1242, 2854, 1242, 2855, 1241, 2856, 1239, 2857, 1239, 2858, 1238, 2859, 1236, 2860, 1236, 2861, 1235, 2861, 1234, 2863, 1233, 2863, 1233, 2864, 1232, 2864, 1231, 2866, 1230, 2866, 1230, 2867, 1229, 2867, 1228, 2868, 1228, 2869, 1227, 2869, 1226, 2870, 1226, 2871, 1225, 2871, 1224, 2873, 1223, 2873, 1223, 2874, 1221, 2875, 1221, 2876, 1220, 2876, 1220, 2877, 1219, 2877, 1218, 2879, 1217, 2879, 1217, 2880, 1216, 2880, 1215, 2881, 1215, 2881, 1215, 2881, 1215, 2881, 1214, 2882, 1214, 2882, 1214, 2882, 1214, 2882, 1213, 2883, 1213, 2883, 1213, 2883, 1212, 2884, 1212, 2884, 1212, 2885, 1210, 2886, 1210, 2886, 1210, 2886, 1209, 2887, 1209, 2888, 1208, 2888, 1207, 2889, 1207, 2889, 1207, 2890, 1205, 2891, 1205, 2891, 1204, 2893, 1203, 2893, 1203, 2893, 1202, 2895, 1201, 2895, 1201, 2896, 1199, 2897, 1199, 2898, 1198, 2898, 1197, 2900, 1196, 2900, 1196, 2901, 1194, 2902, 1194, 2903, 1193, 2903, 1192, 2904, 1192, 2905, 1190, 2906, 1190, 2906, 1190, 2907, 1189, 2907, 1189, 2908, 1187, 2909, 1187, 2910, 1186, 2910, 1186, 2911, 1184, 2912, 1184, 2913, 1183, 2913, 1182, 2915, 1181, 2915, 1181, 2916, 1179, 2917, 1179, 2918, 1178, 2918, 1178, 2919, 1177, 2919, 1177, 2920, 1176, 2920, 1176, 2921, 1174, 2922, 1174, 2923, 1173, 2923, 1173, 2924, 1171, 2925, 1171, 2926, 1170, 2926, 1170, 2926, 1170, 2927, 1168, 2928, 1168, 2928, 1168, 2929, 1167, 2929, 1167, 2930, 1166, 2930, 1166, 2931, 1165, 2931, 1165, 2932, 1163, 2934, 1162, 2934, 1162, 2935, 1161, 2936, 1159, 2937, 1159, 2938, 1158, 2938, 1157, 2940, 1156, 2941, 1155, 2941, 1154, 2943, 1153, 2943, 1153, 2944, 1152, 2944, 1152, 2945, 1151, 2945, 1151, 2946, 1149, 2947, 1149, 2948, 1148, 2948, 1148, 2949, 1147, 2949, 1147, 2950, 1146, 2951, 1145, 2951, 1145, 2952, 1143, 2954, 1142, 2954, 1142, 2955, 1141, 2955, 1140, 2957, 1139, 2958, 1138, 2958, 1138, 2959, 1137, 2959, 1136, 2961, 1135, 2961, 1135, 2962, 1134, 2962, 1133, 2964, 1132, 2964, 1132, 2965, 1130, 2966, 1130, 2966, 1130, 2967, 1128, 2968, 1128, 2968, 1128, 2969, 1126, 2970, 1126, 2971, 1125, 2971, 1124, 2973, 1123, 2973, 1123, 2974, 1121, 2975, 1121, 2976, 1119, 2977, 1119, 2978, 1118, 2978, 1118, 2979, 1116, 2980, 1116, 2981, 1115, 2981, 1115, 2982, 1113, 2983, 1113, 2984, 1112, 2984, 1112, 2984, 1112, 2984, 1111, 2985, 1111, 2986, 1110, 2986, 1110, 2986, 1109, 2987, 1109, 2988, 1108, 2988, 1107, 2989, 1107, 2990, 1106, 2990, 1105, 2991, 1105, 2992, 1103, 2993, 1103, 2994, 1102, 2994, 1102, 2994, 1102, 2995, 1100, 2996, 1100, 2996, 1100, 2997, 1099, 2997, 1098, 2998, 1098, 2999, 1097, 2999, 1097, 2999, 1097, 3000, 1095, 3001, 1095, 3001, 1095, 3002, 1094, 3002, 1093, 3004, 1092, 3004, 1092, 3005, 1091, 3005, 1091, 3006, 1089, 3007, 1089, 3008, 1088, 3008, 1088, 3008, 1087, 3010, 1086, 3010, 1086, 3010, 1085, 3012, 1084, 3012, 1084, 3012, 1083, 3014, 1082, 3014, 1082, 3015, 1080, 3016, 1080, 3016, 1080, 3017, 1079, 3017, 1078, 3018, 1078, 3019, 1077, 3019, 1077, 3019, 1077, 3020, 1076, 3020, 1076, 3020, 1076, 3021, 1075, 3021, 1075, 3021, 1075, 3022, 1074, 3022, 1074, 3023, 1073, 3023, 1073, 3023, 1073, 3024, 1072, 3024, 1072, 3024, 1072, 3025, 1071, 3025, 1071, 3026, 1070, 3026, 1070, 3027, 1069, 3027, 1069, 3028, 1068, 3028, 1068, 3029, 1067, 3029, 1067, 3030, 1066, 3030, 1066, 3030, 1066, 3031, 1063, 3033, 1063, 3033, 1062, 3035, 1061, 3035, 1060, 3037, 1059, 3037, 1059, 3038, 1058, 3038, 1057, 3040, 1056, 3041, 1055, 3041, 1055, 3042, 1054, 3042, 1054, 3043, 1053, 3043, 1053, 3044, 1052, 3045, 1051, 3045, 1051, 3046, 1050, 3046, 1050, 3047, 1049, 3047, 1049, 3048, 1048, 3049, 1047, 3049, 1047, 3050, 1046, 3051, 1045, 3052, 1044, 3053, 1043, 3053, 1043, 3054, 1041, 3056, 1040, 3056, 1040, 3057, 1039, 3058, 1038, 3059, 1037, 3059, 1037, 3060, 1036, 3061, 1034, 3062, 1034, 3063, 1033, 3064, 1032, 3065, 1030, 3066, 1030, 3067, 1029, 3067, 1029, 3067, 1029, 3067, 1028, 3068, 1028, 3068, 1028, 3068, 1028, 3068, 1027, 3069, 1027, 3069, 1027, 3070, 1026, 3070, 1026, 3070, 1025, 3071, 1025, 3072, 1024, 3072, 1024, 3072, 1024, 3073, 1023, 3073, 1023, 3073, 1023, 3074, 1022, 3074, 1021, 3076, 1020, 3076, 1020, 3076, 1020, 3077, 1018, 3078, 1018, 3078, 1018, 3079, 1016, 3080, 1016, 3081, 1015, 3081, 1014, 3083, 1013, 3083, 1012, 3084, 1012, 3084, 1011, 3085, 1011, 3085, 1011, 3085, 1011, 3087, 1009, 3087, 1010, 3087, 1009, 3087, 1009, 3087, 1009, 3088, 1009, 3085, 1011, 3085, 1011, 3085, 1011, 3085, 1012, 3084, 1012, 3084, 1012, 3084, 1013, 3083, 1013, 3085, 1011, 3085, 1012, 3089, 1007, 3089, 1007, 3090, 1007, 3089, 1007, 3090, 1006, 3090, 1006, 3091, 1006, 3090, 1006, 3091, 1005, 3091, 1005, 3092, 1004, 3092, 1004, 3093, 1003, 3093, 1003, 3094, 1002, 3094, 1003, 3094, 1002, 3094, 1002, 3095, 1001, 240, 44, 2811, 1001, 236, 52, 2808, 1000, 232, 61, 2803, 1000, 228, 74, 2794, 1000, 225, 81, 2790, 1001, 221, 89, 2786, 1000, 219, 95, 2782, 1000, 217, 100, 2779, 1001, 214, 105, 2777, 1000, 213, 108, 2775, 1000, 211, 114, 2771, 1001, 209, 119, 2768, 1000, 207, 124, 2765, 1000, 205, 129, 2762, 1000, 203, 133, 2761, 1000, 200, 137, 2759, 1000, 198, 141, 2757, 1000, 196, 146, 2754, 1000, 195, 149, 2753, 999, 193, 153, 2751, 999, 191, 157, 2749, 999, 189, 161, 2747, 999, 187, 165, 2746, 998, 185, 169, 2744, 998, 183, 173, 2742, 998, 181, 176, 2742, 997, 180, 179, 2740, 997, 178, 183, 2738, 997, 177, 186, 2737, 996, 175, 189, 2736, 996, 173, 193, 2734, 996, 172, 195, 2734, 995, 170, 198, 2733, 995, 169, 201, 2731, 995, 167, 205, 2730, 994, 165, 209, 2728, 994, 164, 212, 2726, 994, 162, 216, 2725, 993, 161, 218, 2724, 993, 160, 221, 2723, 992, 158, 224, 2722, 992, 157, 226, 2722, 991, 155, 230, 2720, 991, 154, 232, 2720, 990, 153, 235, 2719, 989, 152, 237, 2718, 989, 151, 239, 2718, 988, 150, 241, 2717, 988, 149, 243, 2716, 988, 148, 245, 2716, 987, 147, 247, 2715, 988, 145, 249, 2714, 988, 144, 251, 2714, 987, 143, 253, 2713, 987, 142, 255, 2713, 987, 140, 257, 2712, 987, 139, 259, 2712, 986, 138, 261, 2711, 986, 137, 263, 2711, 986, 135, 265, 2711, 985, 134, 267, 2710, 985, 133, 268, 2711, 984, 132, 270, 2711, 983, 131, 272, 2710, 983, 130, 274, 2710, 982, 130, 275, 2709, 982, 129, 277, 2709, 981, 128, 279, 2708, 982, 127, 280, 2708, 981, 126, 283, 2706, 981, 126, 284, 2706, 980, 125, 286, 2705, 980, 125, 287, 2704, 980, 124, 289, 2704, 979, 124, 293, 2700, 979, 123, 294, 2700, 980, 121, 295, 2701, 979, 120, 296, 2701, 980, 118, 298, 2701, 979, 117, 300, 2700, 980, 116, 300, 2701, 980, 114, 302, 2700, 981, 112, 304, 2700, 980, 112, 304, 2700, 981, 110, 306, 2700, 981, 108, 308, 2699, 981, 107, 310, 2699, 981, 105, 312, 2698, 981, 104, 314, 2698, 981, 102, 316, 2697, 981, 101, 318, 2697, 981, 99, 320, 2696, 981, 98, 322, 2696, 980, 98, 323, 2695, 981, 96, 324, 2696, 980, 95, 326, 2695, 981, 94, 327, 2695, 981, 92, 328, 2695, 981, 92, 329, 2695, 981, 90, 330, 2695, 982, 89, 331, 2695, 981, 88, 332, 2695, 982, 87, 333, 2695, 981, 86, 335, 2694, 982, 85, 335, 2695, 982, 83, 337, 2694, 983, 82, 337, 2695, 983, 80, 339, 2694, 984, 78, 344, 2690, 985, 77, 344, 2690, 986, 75, 345, 2690, 987, 73, 346, 2692, 985, 73, 350, 2688, 986, 71, 351, 2688, 987, 69, 352, 2688, 988, 67, 353, 2686, 991, 66, 353, 2686, 992, 64, 354, 2686, 993, 62, 355, 2686, 993, 62, 355, 2686, 994, 60, 356, 2686, 995, 59, 356, 2688, 994, 57, 358, 2687, 995, 56, 358, 2687, 996, 54, 360, 2686, 997, 51, 362, 2684, 999, 50, 363, 2684, 1001, 47, 365, 2683, 1002, 44, 367, 2683, 1003, 42, 368, 2683, 1004, 39, 371, 2682, 1005, 37, 372, 2682, 1006, 34, 374, 2682, 1008, 29, 377, 2684, 1007, 26, 380, 2683, 1008, 22, 383, 2683, 1009, 18, 386, 2683, 1010, 14, 389, 2681, 1415, 2681, 1415, 2681, 1415, 2681, 1415, 2681, 1415, 2681, 1415, 2681, 1415, 2681, 1415, 2681, 1415, 2681, 1415, 2681, 1415, 2681, 1415, 2681, 1415, 2681, 1415, 2681, 1415, 2681, 1415, 2681, 1415, 2681, 1415, 2681, 1415, 2681, 1415, 2681, 1415, 2681, 1415, 2681, 1415, 2681, 1415, 2681, 1415, 2681, 1415, 2681, 1415, 2681, 1415, 2683, 1413, 2683, 1413, 2683, 1413, 2683, 1413, 2683, 2, 1, 1410, 2683, 2, 1, 1410, 2686, 1410, 2686, 1410, 2686, 1410, 2686, 1410, 2686, 1410, 2686, 1410, 2686, 1410, 2686, 1411, 2685, 1411, 2685, 1411, 2685, 1411, 2685, 1411, 2685, 1412, 2684, 1412, 2684, 1412, 2684, 1412, 2684, 1413, 2683, 1413, 2683, 1413, 2683, 1413, 2683, 1413, 2683, 1414, 2682, 1414, 2682, 1414, 2682, 1414, 2682, 1415, 2681, 1415, 2681, 1415, 2681, 1415, 2681, 1415, 2681, 1415, 2681, 1415, 2681, 1415, 2681, 1415, 2681, 1415, 2681, 1415, 2681, 1415, 2681, 1415, 2681, 1415, 2681, 1416, 2680, 1416, 2680, 1416, 2680, 1416, 2680, 1417, 2679, 1417, 2679, 1417, 2679, 1417, 2679, 1417, 2679, 1417, 2678, 1418, 2678, 1418, 2678, 1418, 2678, 1418, 2678, 1418, 2678, 1418, 2678, 1418, 2678, 1418, 2678, 1418, 2678, 1418, 2678, 1418, 2678, 1418, 2678, 1419, 2677, 1419, 2677, 1419, 2677, 1419, 2677, 1419, 2677, 1419, 2677, 1419, 2677, 1419, 2677, 1419, 2677, 1419, 2677, 1419, 2677, 1419, 2677, 1419, 2677, 1419, 2677, 1419, 2677, 1419, 2677, 1419, 2677, 1419, 2677, 1420, 2676, 1420, 2676, 1420, 2676, 1420, 2676, 1420, 2676, 1420, 2675, 1421, 2675, 1421, 2675, 1421, 2675, 1422, 2674, 1422, 2674, 1422, 2674, 1422, 2674, 1422, 2674, 1422, 2674, 1422, 2674, 1422, 2674, 1422, 2674, 1422, 2674, 1422, 2674, 1422, 2674, 1422, 2674, 1422, 2674, 1422, 2674, 1422, 2673, 1423, 2673, 1423, 2673, 1423, 2673, 1423, 2673, 1423, 2673, 1423, 2673, 1423, 2673, 1423, 2672, 1424, 2672, 1424, 2672, 1424, 2672, 1424, 2672, 1424, 2672, 1424, 2672, 1424, 2672, 1424, 2672, 1424, 2672, 1424, 2672, 1424, 2672, 1424, 2672, 1424, 2672, 1424, 2672, 1424, 2672, 1424, 2672, 1424, 2672, 1424, 2672, 1424, 2672, 1424, 2672, 1424, 2672, 1424, 2672, 1424, 2672, 1424, 2672, 1424, 2672, 1424, 2672, 1424, 2672, 1424, 2672, 1424, 2672, 1424, 2672, 1424, 2672, 1424, 2672, 1424, 2672, 1424, 2672, 1424, 2672, 1424, 2672, 1424, 2671, 1425, 2671, 1425, 2671, 1425, 2671, 1425, 2671, 1425, 2671, 1425, 2671, 1425, 2671, 1425, 2671, 1425, 2671, 1425, 2671, 1425, 2671, 1425, 2671, 1425, 2671, 1425, 2671, 1425, 2671, 1425, 2671, 1425, 2671, 1425, 2671, 1425, 2671, 1425, 2671, 1425, 2671, 1425, 2671, 1425, 2671, 1425, 2671, 1425, 2671, 1425, 2671, 1425, 2670, 1426, 2670, 1426, 2670, 1426, 2670, 1426, 2670, 1426, 2670, 1426, 2670, 1426, 2670, 1426, 2670, 1426, 2670, 1426, 2670, 1426, 2670, 1425, 2671, 1425, 2671, 1425, 2671, 1425, 2671, 1424, 2672, 1424, 2672, 1424, 2672, 1424, 2672, 1424, 2672, 1424, 2672, 1424, 2672, 1424, 2672, 1424, 2672, 1424, 2672, 1424, 2672, 1424, 2671, 1425, 2671, 1425, 2671, 1425, 2671, 1425, 2671, 1425, 2671, 1425, 2671, 1425, 2671, 1425, 2671, 1425, 2670, 1426, 2670, 1426, 2670, 1426, 2670, 1426, 2670, 1426, 2670, 1426, 2670, 1426, 2670, 1426, 2670, 1426, 2669, 1427, 2669, 1427, 2669, 1426, 2670, 1426, 2669, 1427, 2669, 1427, 2669, 1427, 2669, 1427, 2668, 1428, 2668, 1428, 2668, 1428, 2668, 1427, 2669, 1427, 2668, 1428, 2668, 1428, 2668, 1428, 2668, 1428, 2667, 1429, 2667, 1429, 2667, 1429, 2666, 1430, 2665, 1431, 2664, 1432, 2663, 1433, 2662, 1434, 2661, 1435, 2660, 1436, 2659, 1437, 2658, 1438, 2657, 1439, 2656, 1440, 2655, 1441, 2654, 1442, 2653, 1443, 2652, 1444, 2652, 1444, 2651, 1445, 2650, 1446, 2650, 1446, 2649, 1447, 2648, 1448, 2647, 1449, 2647, 1449, 2646, 1450, 2645, 1451, 2645, 1451, 2644, 1452, 2644, 1452, 2643, 1454, 2642, 1454, 2641, 1455, 2641, 1455, 2640, 1456, 2640, 1456, 2639, 1457, 2639, 1457, 2638, 1458, 2638, 1458, 2637, 1459, 2637, 1459, 2637, 1459, 2636, 1460, 2636, 1460, 2636, 1460, 2635, 1461, 2635, 1461, 2635, 1462, 2633, 1463, 2633, 1463, 2633, 1463, 2632, 1464, 2632, 1464, 2632, 1464, 2631, 1465, 2631, 1465, 2631, 1465, 2631, 1465, 2630, 1466, 2630, 1466, 2630, 1466, 2630, 1466, 2630, 1466, 2630, 1466, 2630, 1466, 2630, 1466, 2630, 1467, 2629, 1467, 2629, 1467, 2629, 1467, 2629, 1467, 2629, 1467, 2629, 1467, 2629, 1467, 2629, 1467, 2629, 1468, 2627, 1469, 2627, 1469, 2627, 1469, 2627, 1469, 2627, 1469, 2627, 1469, 2627, 1469, 2627, 1470, 2626, 1470, 2626, 1470, 2626, 1470, 2626, 1470, 2626, 1470, 2626, 1470, 2626, 1470, 2626, 1470, 2626, 1470, 2626, 1470, 2626, 1470, 2626, 1470, 2626, 1470, 2626, 1470, 2626, 1470, 2626, 1470, 2626, 1470, 2626, 1471, 2626, 1470, 2626, 1470, 2626, 1470, 2626, 1470, 2627, 1469, 2627, 1469, 2627, 1469, 2628, 1468, 2628, 1469, 2627, 1469, 2628, 1468, 2628, 1468, 2628, 1468, 2628, 1468, 2627, 1469, 2627, 1470, 2626, 1470, 2625, 1471, 2625, 1472, 2624, 1472, 2623, 1473, 2623, 1474, 2621, 1475, 2621, 1475, 2621, 1475, 2620, 1477, 2619, 1477, 2619, 1477, 2618, 1478, 2618, 1478, 2618, 1479, 2616, 1480, 2616, 1480, 2616, 1481, 2614, 1482, 2614, 1482, 2614, 1483, 2612, 1484, 2612, 1484, 2612, 1485, 2611, 1485, 2, 4, 2604, 1486, 2, 4, 2604, 1494, 2602, 1494, 2602, 1494, 2602, 1494, 2602, 1494, 2602, 1494, 2602, 1494, 2602, 1494, 2601, 1495, 2601, 1495, 2601, 1495, 2601, 1495, 2601, 1495, 2601, 1495, 2601, 1493, 2603, 1493, 2603, 1493, 2603, 1493, 2603, 1494, 2602, 1494, 2602, 1494, 2602, 1495, 2601, 1495, 2601, 1495, 2601, 1496, 2601, 1495, 2601, 1495, 2601, 1496, 2600, 1496, 2601, 1495, 2601, 1495, 2601, 1496, 2600, 1496, 2601, 1495, 2601, 1495, 2601, 1495, 2601, 1495, 2602, 1494, 2602, 1495, 2601, 1495, 2602, 1494, 2602, 1495, 2601, 1495, 2602, 1494, 2602, 1495, 2602, 1494, 2602, 1494, 2603, 1493, 2603, 1494, 2603, 1493, 2604, 1492, 2604, 1492, 2605, 1491, 2606, 1490, 2607, 1489, 2608, 1488, 2609, 1487, 2610, 1486, 2611, 1485, 2612, 1484, 2613, 1483, 2614, 1482, 2615, 1482, 2614, 1482, 2618, 1478, 2620, 1476, 2621, 1476, 2622, 1474, 2623, 1473, 2625, 1471, 2627, 1469, 2629, 1467, 2632, 1464, 2635, 1461, 2639, 1458, 2642, 1454, 2645, 1451, 2648, 1448, 2651, 1445, 2653, 278, 82, 1083, 2656, 238, 128, 1074, 2659, 213, 155, 1069, 2663, 207, 161, 1065, 2672, 198, 175, 1052, 2680, 189, 179, 1048, 2689, 180, 186, 1041, 2712, 157, 189, 1038, 2716, 153, 190, 1037, 2724, 145, 196, 1032, 2726, 142, 198, 1030, 2729, 137, 192, 1038, 2733, 133, 192, 1038, 2742, 126, 188, 1041, 2750, 117, 188, 1041, 2767, 100, 188, 1041, 2767, 100, 188, 1041, 2769, 6, 12, 78, 192, 10, 7, 1022, 2769, 6, 12, 78, 192, 10, 9, 1020, 3077, 1019, 3078, 1018, 3079, 1017, 3080, 1017, 3080, 1016, 3082, 1014, 3083, 1013, 3085, 1011, 3086, 1011, 3086, 1010, 3087, 1009, 3089, 1007, 3089, 1008, 3089, 1007, 3090, 1006, 3091, 1005, 3092, 1004, 3093, 1003, 3094, 1002, 3094, 1002, 3095, 1001, 3096, 1001, 3096, 1000, 3097, 999, 3098, 998, 3098, 998, 3099, 998, 3099, 997, 3100, 996, 3101, 995, 3102, 995, 3102, 994, 3102, 994, 3103, 993, 3104, 993, 3104, 992, 3105, 991, 3106, 990, 3107, 990, 3106, 990, 3107, 989, 3108, 988, 3109, 987, 3109, 987, 3110, 986, 3111, 985, 3111, 985, 3111, 985, 3111, 985, 3111, 985, 3111, 985, 3111, 985, 3111, 985, 3111, 985, 3111, 985, 3111, 985, 3113, 984, 3112, 984, 3112, 984, 3112, 984, 3112, 2, 2, 980, 3112, 2, 2, 981, 3115, 981, 3115, 981, 3120, 976, 3120, 977, 3120, 976, 3121, 975, 3121, 975, 3121, 975, 3121, 975, 3121, 975, 3121, 975, 3121, 975, 3121, 976, 3121, 975, 3121, 975, 3120, 976, 3120, 976, 3119, 977, 3119, 977, 3119, 977, 3118, 978, 3118, 979, 3117, 979, 3116, 980, 3116, 980, 3116, 980, 3116, 980, 3115, 981, 3115, 981, 3115, 981, 3115, 981, 3114, 982, 3114, 982, 3114, 982, 3114, 982, 3113, 983, 3113, 983, 3113, 983, 3113, 983, 3113, 983, 3112, 984, 3112, 984, 3111, 985, 3111, 985, 3111, 985, 3110, 986, 3110, 986, 3110, 986, 3109, 988, 3108, 988, 3108, 988, 3108, 988, 3107, 989, 3107, 989, 3107, 989, 3107, 989, 3107, 989, 3107, 989, 3107, 989, 3107, 989, 3107, 989, 3106, 990, 3106, 990, 3106, 990, 3105, 994, 3102, 994, 3102, 996, 3100, 996, 3100, 996, 3100, 996, 3100, 994, 3102, 994, 3102, 994, 3102, 994, 3102, 994, 3102, 994, 3102, 994, 3102, 994, 3102, 994, 3101, 995, 3101, 995, 3101, 995, 3101, 995, 3101, 995, 3101, 995, 3101, 995, 3101, 997, 3099, 997, 3099, 997, 3099, 997, 3099, 997, 3099, 997, 3099, 997, 3099, 997, 3099, 997, 3099, 997, 3099, 995, 3101, 995, 3101, 995, 3101, 995, 3101, 995, 3101, 995, 3101, 995, 3101, 995, 3101, 995, 3101, 995, 3101, 995, 3101, 995, 3101, 995, 3101, 995, 3101, 995, 3101, 995, 3101, 995, 3101, 995, 3101, 995, 3101, 995, 3101, 995, 3101, 995, 3101, 997, 3099, 997, 3099, 997, 3099, 997, 3099, 997, 3099, 997, 3099, 997, 3099, 997, 3099, 995, 3101, 995, 3101, 988, 3108, 988, 3108, 988, 3108, 988, 3108, 987, 3109, 987, 3109, 987, 3109, 987, 3109, 987, 3109, 986, 3110, 986, 3110, 986, 3110, 986, 3110, 985, 3111, 985, 3111, 985, 3111, 985, 3111, 984, 3112, 984, 3112, 984, 3112, 984, 3112, 983, 3113, 983, 3113, 983, 3113, 983, 3113, 983, 3113, 982, 3114, 982, 3114, 982, 3114, 982, 3114, 981, 3115, 981, 3115, 981, 3115, 980, 3116, 980, 3116, 980, 3116, 979, 3117, 979, 3117, 979, 3117, 978, 3118, 978, 3118, 978, 3118, 977, 3119, 977, 3119, 977, 3119, 976, 3120, 976, 3120, 975, 3121, 975, 3121, 975, 3121, 974, 3122, 974, 3122, 974, 3122, 973, 3123, 973, 3123, 972, 3124, 972, 3124, 971, 3125, 971, 3125, 970, 3126, 970, 3126, 969, 3127, 969, 3127, 968, 3128, 968, 3128, 968, 3128, 967, 3129, 967, 3129, 967, 3129, 966, 3130, 966, 3130, 965, 3131, 965, 3131, 964, 3132, 964, 3132, 963, 3133, 963, 3133, 962, 3134, 962, 3134, 961, 3135, 961, 3135, 960, 3136, 960, 3136, 959, 3137, 958, 3138, 958, 3139, 956, 3140, 955, 3141, 954, 3142, 954, 3142, 953, 3143, 952, 3144, 952, 3144, 951, 3145, 950, 3147, 949, 3146, 949, 3147, 949, 3147, 948, 3148, 947, 3149, 947, 3149, 946, 3150, 945, 3151, 944, 3152, 943, 3153, 942, 3154, 942, 3154, 941, 3155, 940, 3156, 939, 3157, 939, 3157, 938, 3160, 935, 3161, 935, 3161, 934, 3162, 933, 3161, 934, 3162, 933, 3163, 932, 3164, 931, 3167, 928, 3168, 927, 3173, 922, 3174, 921, 3171, 924, 3172, 923, 3173, 922, 3174, 921, 3173, 922, 3174, 921, 3175, 920, 3176, 919, 3177, 918, 3178, 917, 3181, 914, 3182, 913, 3183, 912, 3184, 910, 3186, 909, 3187, 908, 3188, 907, 3189, 906, 3190, 905, 3191, 904, 3192, 903, 3193, 902, 3194, 901, 3195, 900, 3196, 899, 3197, 897, 3199, 896, 3200, 894, 3202, 893, 3203, 892, 3202, 893, 3203, 892, 3204, 891, 3205, 890, 3206, 889, 3207, 888, 3210, 4, 6, 875, 3211, 4, 6, 873, 3228, 867, 3230, 864, 3232, 863, 3233, 862, 3235, 859, 3237, 858, 3239, 856, 3240, 855, 3242, 853, 3243, 852, 3245, 849, 3248, 847, 3249, 845, 3252, 842, 3255, 839, 3258, 837, 3259, 835, 3262, 833, 3264, 831, 3265, 829, 3268, 827, 3270, 824, 3273, 822, 3275, 820, 3277, 817, 3280, 815, 3282, 812, 3285, 810, 3287, 808, 3289, 806, 3292, 802, 3295, 800, 3297, 797, 3300, 795, 3302, 793, 3304, 790, 3308, 787, 3310, 784, 3314, 780, 3317, 777, 3321, 773, 3324, 770, 3328, 766, 3332, 762, 3336, 759, 3340, 754, 3345, 749, 3348, 746, 3352, 741, 3356, 738, 3360, 734, 3364, 730, 3368, 727, 3371, 723, 3375, 719, 3379, 715, 3383, 710, 3388, 705, 3393, 702, 3396, 698, 3400, 695, 3403, 691, 3407, 687, 3411, 683, 3415, 678, 3420, 673, 3425, 668, 3430, 665, 3433, 661, 3438, 656, 3443, 651, 3447, 646, 3453, 640, 3459, 633, 3467, 625, 3475, 618, 3481, 612, 3487, 606, 3493, 599, 3501, 591, 3510, 583, 3517, 576, 3529, 564, 3541, 551, 3554, 528, 3570, 522, 3576, 511, 3588, 504, 3595, 497, 3603, 484, 3613, 465, 3631, 4, 2, 442, 3648, 4, 2, 442, 3648, 4, 2, 442, 2464] , dtype=np.uint32)
RIGHT_LEG_RLE = np.array( [68291, 81, 231, 17, 3758, 172, 145, 30, 3740, 199, 124, 51, 3713, 212, 117, 72, 3687, 427, 3660, 445, 3647, 453, 3638, 463, 3629, 476, 3617, 488, 3605, 500, 3589, 511, 3581, 520, 3571, 529, 3563, 538, 3555, 545, 3548, 552, 3541, 559, 3533, 566, 3526, 573, 3520, 579, 3514, 585, 3508, 591, 3501, 597, 3495, 603, 3490, 608, 3485, 613, 3481, 617, 3477, 622, 3472, 627, 3467, 631, 3463, 638, 3456, 644, 3450, 649, 3445, 654, 3439, 659, 3434, 665, 3428, 671, 3423, 675, 3419, 679, 3415, 683, 3412, 686, 3408, 690, 3405, 694, 3401, 698, 3396, 702, 3392, 707, 3387, 712, 3382, 717, 3377, 723, 3362, 738, 3354, 745, 3348, 751, 3342, 757, 3337, 762, 3332, 767, 3327, 771, 3324, 777, 3317, 783, 3312, 788, 3307, 793, 3301, 809, 3286, 814, 3280, 821, 3274, 826, 3269, 832, 3262, 843, 3252, 848, 3246, 854, 3241, 864, 3231, 869, 3226, 873, 3221, 878, 3217, 882, 3211, 888, 3204, 895, 3197, 901, 3190, 909, 3183, 916, 3177, 921, 3172, 926, 3168, 930, 3161, 937, 3155, 943, 3150, 948, 3145, 953, 3141, 957, 3136, 962, 3131, 967, 3126, 972, 3121, 976, 3117, 981, 3113, 984, 3109, 990, 3103, 997, 3097, 1002, 3092, 1007, 3087, 1011, 3083, 1015, 3079, 1019, 3074, 1025, 3068, 1030, 3064, 1034, 3060, 1038, 3056, 1041, 3054, 1046, 3048, 1052, 3043, 1058, 3036, 1064, 3030, 1070, 3025, 1083, 3011, 1085, 3010, 1088, 3007, 1093, 3001, 1099, 2996, 1104, 2990, 1110, 2985, 1116, 2979, 1121, 2974, 1126, 2968, 1132, 2963, 1138, 2956, 1144, 2951, 1150, 2945, 1155, 2939, 1161, 2934, 1166, 2928, 1173, 2921, 1179, 2915, 1186, 2909, 1191, 2903, 1197, 2898, 1202, 2893, 1208, 2887, 1218, 2877, 1228, 2867, 1302, 2792, 1322, 2773, 1350, 2745, 1360, 2735, 1380, 2715, 1385, 2710, 1390, 2704, 1401, 2694, 1406, 2689, 1412, 2683, 1417, 2678, 1423, 2672, 1428, 2667, 1434, 2661, 1439, 2656, 1443, 2652, 1447, 2648, 1451, 2644, 1456, 2638, 1462, 2633, 1466, 2629, 1470, 2625, 1474, 2621, 1478, 2617, 1482, 2612, 1486, 2609, 1489, 7, 26, 2571, 1494, 1, 39, 2559, 1541, 2553, 1547, 2546, 1554, 2539, 1560, 2534, 1565, 2529, 1570, 2524, 1575, 2520, 1579, 2515, 1583, 2512, 1587, 2502, 2, 2, 1593, 2499, 1599, 2493, 1605, 2491, 1608, 2487, 1613, 2481, 1618, 2476, 1623, 2470, 1628, 2465, 1633, 2460, 1638, 2455, 1642, 2451, 1647, 2447, 1650, 2444, 1653, 2441, 1657, 2438, 1660, 2434, 1664, 2430, 1669, 2425, 1674, 2420, 1678, 2416, 1682, 2412, 1686, 2408, 1689, 2405, 1693, 2401, 1696, 2398, 1700, 2394, 1704, 2390, 1707, 2387, 1711, 2383, 1714, 2380, 1718, 2376, 1723, 2370, 1728, 2367, 1731, 2363, 1735, 2360, 1737, 2357, 1741, 2354, 1743, 2351, 1746, 2348, 1750, 2345, 1752, 2342, 1756, 2338, 1759, 2335, 1763, 2332, 1766, 2328, 1769, 2326, 1772, 2322, 1776, 2318, 1780, 2315, 1782, 2312, 1786, 2309, 1788, 2307, 1791, 2303, 1795, 2300, 1797, 2297, 1801, 2294, 1803, 2292, 1805, 2289, 1808, 2287, 1810, 2284, 1814, 2281, 1816, 2279, 1819, 2275, 1823, 2272, 1825, 2269, 1829, 2266, 1831, 2264, 1833, 2262, 1835, 2260, 1837, 2258, 1840, 2254, 1843, 2251, 1847, 2247, 1851, 2244, 1853, 2241, 1857, 2238, 1859, 2236, 1861, 2234, 1863, 2232, 1865, 2230, 1867, 2227, 1870, 2225, 1872, 2223, 1874, 2221, 1876, 2219, 1878, 2217, 1880, 2214, 1883, 2211, 1886, 2206, 1891, 2202, 1894, 2199, 1897, 2197, 1899, 2195, 1901, 2193, 1903, 2191, 1906, 2188, 1908, 2185, 1911, 2182, 1914, 2180, 1917, 2177, 1919, 2175, 1921, 2173, 1924, 2170, 1926, 2168, 1928, 2166, 1931, 2162, 1934, 2160, 1937, 2156, 1940, 2153, 1944, 2150, 1946, 2148, 1949, 2145, 1952, 2142, 1954, 2140, 1957, 2137, 1960, 2134, 1963, 2131, 1966, 2128, 1969, 2125, 1972, 2122, 1975, 2119, 1977, 2117, 1979, 2115, 1981, 2112, 1984, 2110, 1986, 2108, 1988, 2106, 1990, 2103, 1993, 2101, 1995, 2099, 1997, 2097, 1999, 2096, 2000, 2094, 2002, 2093, 2003, 2092, 2004, 2090, 2006, 2089, 2007, 2087, 2009, 2085, 2011, 2083, 2013, 2081, 2015, 2079, 2017, 2076, 2020, 2073, 2023, 2071, 2025, 2069, 2027, 2067, 2029, 2066, 2030, 2064, 2032, 2063, 2033, 2062, 2034, 2061, 2035, 2060, 2036, 2059, 2037, 2058, 2038, 2057, 2039, 2056, 2040, 2055, 2041, 2054, 2042, 2054, 2042, 2053, 2043, 2052, 2044, 2052, 2044, 2051, 2045, 2051, 2045, 2050, 2046, 2050, 2046, 2049, 2047, 2049, 2047, 2049, 2047, 2048, 2048, 2048, 2048, 2048, 2048, 2047, 2049, 2047, 2049, 2046, 2050, 2046, 2050, 2046, 2050, 2045, 2051, 2045, 2051, 2045, 2051, 2044, 2052, 2044, 2052, 2044, 2052, 2044, 2052, 2043, 2053, 2043, 2053, 2043, 2053, 2043, 2053, 2043, 2052, 2044, 2052, 2044, 2052, 2044, 2052, 2044, 2051, 2045, 2051, 2045, 2051, 2045, 2051, 2045, 2051, 2045, 2050, 2046, 2050, 2046, 2050, 2046, 2050, 2046, 2050, 2046, 2050, 2046, 2050, 2046, 2050, 2046, 2050, 2046, 2049, 2047, 2049, 2047, 2049, 2047, 2049, 2047, 2049, 2047, 2049, 2047, 2049, 2047, 2049, 2047, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2047, 2049, 2047, 2049, 2047, 2049, 2047, 2049, 2046, 2050, 2046, 2050, 2046, 2050, 2046, 2050, 2046, 2050, 2045, 2051, 2045, 2051, 2045, 2051, 2045, 2051, 2044, 2052, 2044, 2052, 2044, 2052, 2043, 2053, 2043, 2053, 2043, 2053, 2042, 2054, 2042, 2054, 2042, 2054, 2042, 2054, 2042, 2054, 2042, 2054, 2042, 2054, 2042, 2054, 2041, 2055, 2041, 2055, 2041, 2055, 2041, 2055, 2040, 2056, 2040, 2056, 2040, 2056, 2040, 2056, 2040, 2056, 2039, 2057, 2039, 2057, 2039, 2057, 2039, 2057, 2038, 2058, 2038, 2058, 2038, 2058, 2038, 2058, 2037, 2059, 2037, 2059, 2037, 2059, 2037, 2059, 2036, 2060, 2036, 2060, 2036, 2060, 2036, 2060, 2036, 2060, 2035, 2061, 2035, 2061, 2035, 2061, 2035, 2061, 2034, 2062, 2034, 2062, 2034, 2062, 2034, 2062, 2034, 2062, 2033, 2063, 2033, 2063, 2033, 2063, 2033, 2063, 2032, 2064, 2032, 2064, 2032, 2064, 2032, 2064, 2032, 2064, 2031, 2065, 2031, 2065, 2031, 2065, 2031, 2065, 2031, 2065, 2031, 2065, 2031, 2065, 2031, 2065, 2031, 2065, 2030, 2066, 2030, 2066, 2030, 2066, 2030, 2066, 2029, 2067, 2029, 2067, 2029, 2067, 2029, 2068, 2027, 2069, 2027, 2069, 2027, 2069, 2027, 2070, 2026, 2070, 2026, 2070, 2026, 2070, 2026, 2070, 2026, 2071, 2024, 2072, 2024, 2072, 2024, 2072, 2024, 2072, 2023, 2073, 2023, 2073, 2023, 2073, 2023, 2074, 2022, 2074, 2021, 2075, 2021, 2075, 2021, 2076, 2020, 2076, 2019, 2077, 2019, 2078, 2018, 2078, 2018, 2078, 2018, 2079, 2016, 2080, 2016, 2080, 2016, 2081, 2015, 2081, 2014, 2082, 2014, 2083, 2013, 2083, 2013, 2083, 2013, 2084, 2011, 2085, 2011, 2085, 2011, 2086, 2010, 2086, 2009, 2087, 2009, 2087, 2009, 2088, 2008, 2088, 2007, 2089, 2007, 2089, 2007, 2090, 2006, 2090, 2005, 2091, 2005, 2092, 2004, 2092, 2003, 2093, 2003, 2094, 2002, 2094, 2001, 2095, 2001, 2096, 2000, 2096, 1999, 2097, 1999, 2098, 1998, 2098, 1997, 2099, 1997, 2100, 1996, 2100, 1995, 2102, 1994, 2102, 1993, 2103, 1993, 2104, 1992, 2104, 1991, 2105, 1991, 2106, 1990, 2106, 1989, 2108, 1988, 2108, 1988, 2108, 1987, 2110, 1986, 2110, 1986, 2110, 1985, 2112, 1984, 2112, 1984, 2113, 1982, 2114, 1982, 2115, 1980, 2116, 1980, 2117, 1978, 2119, 1977, 2119, 1976, 2121, 1974, 2123, 1973, 2124, 1971, 2125, 1971, 2126, 1969, 2128, 1968, 2128, 1968, 2129, 1966, 2131, 1965, 2131, 1965, 2132, 1964, 2132, 1963, 2134, 1962, 2135, 1961, 2135, 1961, 2136, 1960, 2136, 1959, 2138, 1958, 2138, 1958, 2139, 1957, 2139, 1956, 2141, 1955, 2141, 1955, 2142, 1953, 2143, 1953, 2144, 1952, 2144, 1951, 2145, 1951, 2145, 1950, 2146, 1950, 2146, 1950, 2146, 1950, 2146, 1949, 2147, 1949, 2147, 1949, 2147, 1949, 2147, 1948, 2148, 1948, 2148, 1948, 2148, 1948, 2148, 1948, 2148, 1947, 2149, 1947, 2150, 1946, 2150, 1946, 2150, 1945, 2151, 1945, 2152, 1944, 2152, 1944, 2152, 1943, 2153, 1943, 2154, 1942, 2154, 1942, 2154, 1941, 2155, 1941, 2155, 1941, 2155, 1940, 2156, 1940, 2156, 1940, 2156, 1940, 2157, 1939, 2157, 1939, 2157, 1939, 2157, 1938, 2159, 1937, 2159, 1937, 2159, 1937, 2160, 1935, 2161, 1935, 2161, 1935, 2162, 1934, 2162, 1933, 2164, 1932, 2164, 1932, 2164, 1932, 2164, 1932, 2164, 1932, 2165, 1931, 2165, 1931, 2165, 1931, 2165, 1930, 2166, 1930, 2166, 1930, 2166, 1930, 2166, 1930, 2166, 1930, 2167, 1929, 2167, 1929, 2167, 1929, 2167, 1928, 2169, 1927, 2169, 1927, 2169, 1927, 2169, 1927, 2169, 1927, 2170, 1926, 2170, 1926, 2170, 1926, 2170, 1925, 2172, 1924, 2172, 1924, 2172, 1924, 2172, 1923, 2174, 1922, 2174, 1922, 2174, 1921, 2175, 1921, 2176, 1920, 2176, 1919, 2177, 1919, 2178, 1918, 2178, 1917, 2179, 1917, 2180, 1916, 2180, 1916, 2180, 1915, 2182, 1914, 2182, 1914, 2182, 1914, 2183, 1912, 2184, 1912, 2184, 1912, 2185, 1910, 2186, 1910, 2187, 1909, 2187, 1908, 2188, 1908, 2189, 1907, 2189, 1906, 2190, 1906, 2191, 1905, 2191, 1905, 2192, 1904, 2192, 1903, 2193, 1903, 2194, 1902, 2194, 1902, 2194, 1901, 2196, 1900, 2196, 1900, 2197, 1898, 2198, 1898, 2199, 1897, 2199, 1896, 2201, 1895, 2202, 1893, 2203, 1893, 2204, 1892, 2204, 1891, 2206, 1890, 2206, 1890, 2207, 1888, 2209, 1887, 2209, 1886, 2211, 1885, 2212, 1884, 2212, 1883, 2214, 1882, 2214, 1882, 2215, 1880, 2217, 1879, 2217, 1878, 2219, 1877, 2220, 1875, 2221, 1875, 2222, 1874, 2222, 1873, 2224, 1872, 2224, 1872, 2225, 1871, 2225, 1871, 2226, 1870, 2226, 1870, 2227, 1869, 2227, 1869, 2228, 1868, 2228, 1868, 2229, 1867, 2229, 1865, 2232, 1864, 2232, 1862, 2235, 1861, 2235, 1861, 2236, 1859, 2238, 1858, 2238, 1858, 2239, 1857, 2239, 1856, 2241, 1855, 2241, 1855, 2242, 1853, 2243, 1853, 2244, 1852, 2244, 1851, 2246, 1850, 2246, 1849, 2248, 1848, 2248, 1848, 2249, 1847, 2250, 1846, 2250, 1845, 2252, 1844, 2253, 1843, 2254, 1842, 2254, 1841, 2256, 1840, 2257, 1839, 2257, 1838, 2259, 1837, 2260, 1836, 2260, 1835, 2262, 1834, 2262, 1833, 2264, 1832, 2264, 1831, 2266, 1830, 2266, 1829, 2268, 1827, 2269, 1827, 2269, 1826, 2271, 1825, 2271, 1824, 2272, 1824, 2273, 1822, 2274, 1822, 2275, 1821, 2275, 1821, 2276, 1819, 2277, 1819, 2278, 1818, 2278, 1818, 2279, 1816, 2280, 1816, 2281, 1815, 2281, 1815, 2281, 1815, 2282, 1813, 2283, 1813, 2283, 1813, 2284, 1812, 2284, 1811, 2285, 1811, 2286, 1810, 2286, 1810, 2286, 1810, 2287, 1808, 2288, 1808, 2288, 1808, 2289, 1807, 2289, 1806, 2291, 1805, 2291, 1805, 2292, 1804, 2292, 1803, 2294, 1802, 2294, 1802, 2295, 1801, 2295, 1800, 2297, 1799, 2297, 1799, 2298, 1798, 2298, 1798, 2298, 1797, 2299, 1797, 2299, 1797, 2299, 1797, 2299, 1796, 2300, 1796, 2298, 1798, 2298, 1798, 2298, 1797, 2299, 1797, 2299, 1797, 2299, 1797, 2301, 4, 2, 1788, 2302, 4, 2, 1788, 2308, 1788, 2309, 1787, 2309, 1787, 2309, 1787, 2310, 1786, 2310, 1786, 2311, 1785, 2311, 1785, 2311, 1785, 2312, 1784, 2312, 1784, 2312, 1784, 2313, 1783, 2313, 1783, 2314, 1782, 2314, 1782, 2315, 1781, 2315, 1781, 2316, 1780, 2316, 1780, 2317, 1779, 2317, 1779, 2318, 1778, 2318, 1778, 2319, 1775, 2321, 1775, 2322, 1774, 2322, 1774, 2323, 1773, 2323, 1768, 1, 4, 2324, 1767, 2329, 1767, 2330, 1766, 2330, 1765, 2332, 1764, 2332, 1764, 2333, 1763, 2333, 1763, 2333, 1762, 2334, 1762, 2334, 1762, 2334, 1762, 2334, 1761, 2335, 1761, 2335, 1761, 2335, 1761, 2335, 1760, 2336, 1760, 2336, 1760, 2336, 1760, 2336, 1759, 2337, 1759, 2337, 1759, 2337, 1758, 2338, 1758, 2338, 1758, 2338, 1757, 2339, 1757, 2339, 1757, 2339, 1757, 2339, 1757, 2339, 1757, 2339, 1757, 2339, 1757, 2339, 1756, 2340, 1756, 2340, 1756, 2340, 1756, 2340, 1755, 2341, 1755, 2341, 1755, 2341, 1754, 2342, 1754, 2342, 1754, 2342, 1753, 2343, 1753, 2343, 1752, 2344, 1752, 2344, 1752, 2344, 1751, 2345, 1751, 2345, 1751, 2345, 1750, 2346, 1750, 2346, 1750, 2346, 1750, 2346, 1750, 2346, 1750, 2346, 1750, 2347, 1748, 2348, 1748, 2348, 1748, 2348, 1748, 2349, 1746, 2350, 1746, 2350, 1746, 2351, 1745, 2351, 1745, 2351, 1744, 2353, 1743, 2353, 1743, 2354, 1742, 2354, 1741, 2355, 1741, 2355, 1741, 2355, 1741, 2356, 1740, 2356, 1739, 2357, 1739, 2357, 1739, 2358, 1738, 2358, 1737, 2359, 1737, 2359, 1737, 2359, 1736, 2361, 1735, 2361, 1735, 2361, 1734, 2362, 1734, 2363, 1732, 2364, 1732, 2364, 1732, 2364, 1731, 2365, 1731, 2366, 1730, 2366, 1729, 2367, 1729, 2367, 1729, 2368, 1727, 2369, 1727, 2369, 1727, 2369, 1727, 2370, 1725, 2371, 1725, 2371, 1725, 2371, 1725, 2372, 1724, 2372, 1724, 2372, 1724, 2372, 1724, 2372, 1724, 2373, 1723, 2373, 1723, 2373, 1723, 2373, 1723, 2374, 1722, 2374, 1722, 2374, 1722, 2374, 1722, 2375, 1721, 2375, 1721, 2375, 1716, 1, 4, 2375, 1716, 1, 4, 2376, 1715, 1, 4, 2376, 1715, 1, 4, 2376, 1715, 1, 4, 2377, 1714, 1, 4, 2377, 1714, 1, 4, 2377, 1714, 2383, 1712, 2384, 1712, 2384, 1712, 2385, 1711, 2385, 1711, 2385, 1711, 2385, 1711, 2385, 1711, 2386, 1710, 2386, 1709, 2387, 1709, 2387, 1709, 2388, 1708, 2388, 1707, 2389, 1707, 2389, 1707, 2390, 1706, 2390, 1705, 2391, 1705, 2391, 1705, 2392, 1704, 2392, 1703, 2393, 1703, 2393, 1703, 2393, 1703, 2394, 1702, 2394, 1702, 2394, 1701, 2395, 1701, 2396, 1700, 2396, 1700, 2396, 1700, 2397, 1699, 2397, 1699, 2397, 1699, 2398, 1697, 2399, 1697, 2400, 1696, 2400, 1696, 2400, 1696, 2401, 1695, 2401, 1695, 2401, 1695, 2402, 1694, 2402, 1693, 2403, 1693, 2404, 1692, 2404, 1692, 2404, 1691, 2406, 1690, 2406, 1690, 2406, 1690, 2407, 1689, 2407, 1688, 2409, 1687, 2409, 1687, 2409, 1687, 2410, 1686, 2410, 1686, 2410, 1686, 2411, 1685, 2411, 1685, 2412, 1683, 2413, 1683, 2414, 1682, 2414, 1682, 2415, 1681, 2415, 1681, 2416, 1680, 2416, 1680, 2417, 1679, 2417, 1678, 2418, 1678, 2419, 1677, 2419, 1677, 2419, 1676, 2421, 1675, 2421, 1675, 2422, 1674, 2422, 1674, 2422, 1673, 2424, 1672, 2424, 1672, 2424, 1672, 2425, 1671, 2425, 1671, 2426, 1670, 2426, 1670, 2427, 1669, 2427, 1668, 2429, 1667, 2430, 1666, 2430, 1666, 2431, 1664, 2433, 1663, 2433, 1663, 2433, 1663, 2434, 1661, 2435, 1661, 2435, 1661, 2436, 1660, 2436, 1659, 2438, 1658, 2438, 1658, 2439, 1657, 2439, 1657, 2440, 1655, 2441, 1655, 2441, 1655, 2442, 1654, 2442, 1654, 2443, 1653, 2443, 1653, 2443, 1653, 2444, 1652, 2444, 1651, 2445, 1651, 2446, 1650, 2446, 1650, 2447, 1648, 2448, 1648, 2449, 1647, 2449, 1647, 2450, 1646, 2450, 1646, 2451, 1645, 2451, 1645, 2451, 1645, 2451, 1645, 2451, 1645, 2451, 1645, 2451, 1645, 2452, 1644, 2452, 1643, 2453, 1643, 2453, 1643, 2454, 1642, 2454, 1641, 2455, 1641, 2456, 1640, 2456, 1640, 2456, 1640, 2457, 1638, 2458, 1638, 2459, 1637, 2459, 1637, 2460, 1636, 2460, 1636, 2461, 1635, 2461, 1635, 2461, 1635, 2461, 1634, 2463, 1633, 2463, 1633, 2463, 1633, 2463, 1632, 2464, 1632, 2465, 1631, 2465, 1631, 2465, 1630, 2466, 1630, 2467, 1629, 2467, 1629, 2467, 1628, 2469, 1627, 2469, 1627, 2469, 1627, 2470, 1626, 2470, 1625, 2472, 1624, 2472, 1624, 2472, 1624, 2472, 1623, 2473, 1623, 2474, 1622, 2474, 1621, 2475, 1621, 2475, 1621, 2476, 1619, 2477, 1619, 2477, 1618, 2479, 1617, 2479, 1617, 2479, 1617, 2480, 1616, 2480, 1615, 2481, 1615, 2482, 1614, 2482, 1614, 2482, 1613, 2484, 1612, 2484, 1612, 2484, 1611, 2486, 1610, 2486, 1610, 2486, 1609, 2488, 1608, 2488, 1607, 2489, 1607, 2490, 1605, 2491, 1605, 2491, 1604, 2493, 1602, 2494, 1602, 2495, 1600, 2496, 1600, 2496, 1599, 2498, 1598, 2498, 1597, 2499, 1597, 2500, 1595, 2501, 1595, 2501, 1594, 2503, 1593, 2503, 1592, 2504, 1592, 2504, 1591, 2506, 1589, 2507, 1589, 2507, 1588, 2508, 1587, 2510, 1586, 2510, 1586, 2510, 1585, 2512, 1584, 2512, 1584, 2512, 1583, 2514, 1582, 2514, 1581, 2515, 1581, 2516, 1579, 2517, 1579, 2517, 1578, 2519, 1577, 2519, 1577, 2519, 1577, 2520, 1576, 2520, 1576, 2521, 1575, 2521, 1575, 2521, 1575, 2522, 1574, 2522, 1574, 2522, 1574, 2523, 1573, 2523, 1573, 2524, 1572, 2524, 1572, 2525, 1571, 2525, 1571, 2526, 1570, 2526, 1569, 2528, 1568, 2528, 1568, 2529, 1567, 2529, 1566, 2530, 1566, 2531, 1565, 2531, 1564, 2532, 1564, 2533, 1563, 2533, 1562, 2535, 1561, 2535, 1560, 2537, 1559, 2537, 1558, 2539, 1557, 2539, 1556, 2541, 1554, 2542, 1554, 2543, 1552, 2544, 1551, 2546, 1549, 2547, 1548, 2549, 1546, 2550, 1545, 2552, 1543, 2553, 1542, 2555, 1540, 2556, 1539, 2558, 1536, 2560, 1536, 2561, 1534, 2562, 1534, 2563, 1533, 2563, 1533, 2564, 1531, 2565, 1531, 2566, 1530, 2566, 1530, 2567, 1528, 2569, 1527, 2569, 1527, 2570, 1525, 2571, 1525, 2572, 1524, 2572, 1523, 2574, 1522, 2575, 1521, 2575, 1520, 2577, 1519, 2578, 1518, 2579, 1517, 2580, 1515, 2581, 1515, 2582, 1514, 2582, 1514, 2582, 1514, 2582, 1514, 2582, 1514, 2582, 1514, 2582, 1514, 2582, 1513, 2585, 1511, 2585, 1511, 2583, 1513, 2583, 1512, 2584, 1512, 2584, 1512, 2586, 1510, 2586, 1510, 2591, 1505, 2592, 1504, 2592, 1504, 2593, 1503, 2593, 1503, 2594, 1502, 2594, 1502, 2595, 1501, 2596, 1500, 2596, 1499, 2598, 1498, 2598, 1498, 2597, 1499, 2597, 1499, 2597, 1499, 2597, 1499, 2597, 1499, 2597, 1499, 2597, 1499, 2597, 1499, 2596, 1500, 2596, 1500, 2596, 1500, 2596, 1500, 2596, 1500, 2596, 1500, 2596, 1500, 2596, 1500, 2596, 1500, 2596, 1500, 2596, 1500, 2596, 1500, 2596, 1500, 2596, 1500, 2596, 1500, 2596, 1500, 2596, 1500, 2597, 1499, 2597, 1499, 2597, 1499, 2597, 1499, 2598, 1498, 2598, 1498, 2598, 1498, 2599, 1497, 2599, 1497, 2599, 1497, 2600, 1496, 2600, 1496, 2600, 1496, 2600, 1496, 2600, 1496, 2600, 1496, 2600, 1496, 2601, 1495, 2601, 1495, 2601, 1495, 2601, 1495, 2601, 1495, 2601, 1495, 2601, 1495, 2601, 1495, 2601, 1495, 2602, 1494, 2602, 1494, 2602, 1494, 2602, 1494, 2602, 1494, 2602, 1494, 2602, 1494, 2602, 1494, 2602, 1494, 2603, 1493, 2603, 1493, 2603, 1493, 2603, 1493, 2604, 1492, 2604, 1492, 2604, 1492, 2605, 1491, 2605, 1491, 2605, 1491, 2606, 1490, 2606, 1490, 2607, 1489, 2607, 1489, 2607, 1489, 2608, 1488, 2608, 1488, 2608, 1488, 2609, 1487, 2609, 1487, 2609, 1487, 2609, 1487, 2609, 1487, 2609, 1487, 2609, 1487, 2610, 1486, 2610, 1486, 2610, 1486, 2610, 1486, 2611, 1486, 2610, 1486, 2610, 1486, 2611, 1485, 2611, 1485, 2611, 1485, 2612, 1484, 2612, 1484, 2612, 1484, 2613, 1484, 2612, 1484, 2612, 1484, 2613, 1483, 2613, 1483, 2613, 1483, 2614, 1482, 2614, 1482, 2615, 1481, 2615, 1482, 2614, 1482, 2614, 1482, 2614, 1482, 2615, 1481, 2615, 1482, 2614, 1482, 2614, 1482, 2615, 1481, 2615, 1482, 2614, 1482, 2615, 1481, 2615, 1481, 2615, 1481, 2616, 1481, 2615, 1481, 2616, 1480, 2616, 1480, 2616, 1481, 2616, 1480, 2616, 1480, 2616, 1480, 2617, 1479, 2617, 1479, 2617, 1479, 2618, 1478, 2618, 1478, 2618, 1479, 2618, 1478, 2618, 1478, 2618, 1478, 2619, 1478, 2618, 1478, 2619, 1477, 2619, 1477, 2619, 1478, 2618, 1478, 2619, 1477, 2619, 1477, 2619, 1477, 2619, 1477, 2620, 1476, 2620, 1476, 2620, 1476, 2621, 1475, 2621, 1475, 2621, 1475, 2622, 1474, 2622, 1474, 2622, 1474, 2622, 1474, 2622, 1474, 2622, 1474, 2622, 1475, 2622, 1474, 2622, 1474, 2622, 1474, 2622, 1474, 2623, 1473, 2623, 1473, 2623, 1473, 2623, 1473, 2623, 1473, 2624, 1472, 2624, 1472, 2624, 1472, 2624, 1472, 2625, 1471, 2625, 1471, 2625, 1471, 2626, 1470, 2626, 1470, 2626, 1470, 2627, 1469, 2627, 1469, 2627, 1469, 2628, 1468, 2628, 1468, 2628, 1468, 2628, 1468, 2629, 1467, 2629, 1467, 2629, 1467, 2629, 1467, 2630, 1466, 2630, 1466, 2630, 1466, 2631, 1465, 2631, 1464, 2632, 1464, 2633, 1463, 2633, 1463, 2633, 1463, 2634, 1462, 2634, 1462, 2634, 1462, 2635, 1461, 2635, 1460, 2636, 1460, 2637, 1459, 2637, 1459, 2637, 1459, 2638, 1458, 2638, 1458, 2638, 1458, 2638, 1458, 2639, 1456, 2640, 1456, 2640, 1456, 2640, 1456, 2641, 1454, 2642, 1454, 2642, 1454, 2643, 1453, 2643, 1453, 2643, 1453, 2644, 1452, 2644, 1452, 2644, 1452, 2645, 1451, 2645, 1451, 2645, 1451, 2646, 1450, 2646, 1450, 2646, 1450, 2647, 1449, 2647, 1449, 2648, 1448, 2648, 1448, 2648, 1448, 2648, 1448, 2648, 1448, 2649, 1447, 2649, 1446, 2650, 1446, 2650, 1446, 2651, 1445, 2651, 1444, 2652, 1444, 2652, 1444, 2653, 1443, 2653, 1443, 2653, 1442, 2654, 1443, 2654, 1442, 2654, 1442, 2654, 1442, 2655, 1442, 2654, 1442, 2654, 1442, 2655, 1441, 2655, 1442, 2655, 1441, 2655, 1441, 2656, 1440, 2656, 1440, 2657, 1439, 2657, 1439, 2657, 1439, 2658, 1438, 2658, 1439, 2658, 1438, 2658, 1438, 2658, 1438, 2658, 1438, 2658, 1438, 2659, 1437, 2659, 1437, 2659, 1437, 2659, 1438, 2659, 1437, 2659, 1437, 2659, 1437, 2660, 1436, 2660, 1436, 2660, 1436, 2661, 1435, 2661, 1435, 2662, 1434, 2662, 1434, 2662, 1434, 2662, 1434, 2662, 1434, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2664, 1432, 2664, 1432, 2664, 1432, 2665, 1431, 2665, 1431, 2665, 1431, 2666, 1430, 2666, 1430, 2666, 1430, 2667, 1429, 2667, 1429, 2667, 1429, 2667, 1429, 2668, 1428, 2668, 1428, 2668, 1428, 2668, 1428, 2669, 1427, 2669, 1427, 2669, 1427, 2669, 1427, 2670, 1426, 2670, 1426, 2670, 1426, 2670, 1426, 2671, 1425, 2671, 1425, 2671, 1425, 2671, 1425, 2671, 1425, 2672, 1424, 2672, 1424, 2672, 1424, 2672, 1424, 2673, 1423, 2673, 1423, 2673, 1423, 2673, 1423, 2673, 1423, 2674, 1422, 2674, 1422, 2674, 1422, 2674, 1422, 2675, 1421, 2675, 1421, 2675, 1421, 2676, 1420, 2676, 1420, 2676, 1420, 2677, 1419, 2677, 1419, 2678, 1418, 2678, 1418, 2678, 1418, 2678, 1418, 2678, 1418, 2679, 1417, 2679, 1417, 2679, 1417, 2679, 1417, 2680, 1416, 2680, 1416, 2680, 1416, 2680, 1416, 2680, 1416, 2681, 1415, 2681, 1415, 2681, 1415, 2681, 1415, 2681, 1415, 2681, 1415, 2681, 1415, 2681, 1415, 2682, 1414, 2682, 1414, 2682, 1414, 2682, 1414, 2683, 1413, 2683, 1414, 2682, 1414, 2682, 1414, 2682, 1414, 2683, 1414, 2682, 1414, 2682, 1414, 2682, 1414, 2683, 1413, 2683, 1413, 2683, 1413, 2683, 1413, 2683, 1413, 2684, 1412, 2684, 1412, 2684, 1412, 2684, 1412, 2685, 1411, 2685, 1411, 2685, 1411, 2685, 1411, 2685, 1411, 2686, 1411, 2685, 1411, 2685, 1411, 2685, 1411, 2686, 1410, 2686, 1410, 2686, 1410, 2686, 1410, 2686, 1410, 2687, 1410, 2686, 1410, 2686, 1410, 2686, 1410, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2688, 1408, 2688, 1409, 2687, 1409, 2687, 1409, 2688, 1408, 2688, 1408, 2688, 1408, 2689, 1407, 2689, 1407, 2689, 1407, 2690, 1407, 2689, 1407, 2689, 1407, 2689, 1407, 2689, 1407, 2689, 1408, 2688, 1408, 2688, 1408, 2689, 1407, 2689, 1408, 2688, 1408, 2688, 1408, 2689, 1407, 2689, 1407, 2689, 1407, 2689, 1407, 2689, 1407, 2690, 1407, 2689, 1407, 2689, 1407, 2689, 1407, 2690, 1406, 2690, 1407, 2689, 1407, 2689, 1407, 2689, 1407, 2690, 1407, 2689, 1407, 2689, 1407, 2689, 1407, 2690, 1406, 2690, 1406, 2690, 1406, 2690, 1406, 2691, 1405, 2691, 1406, 2690, 1406, 2690, 1406, 2691, 1405, 2691, 1405, 2691, 1405, 2691, 1405, 2691, 1405, 2692, 1404, 2692, 1404, 2692, 1404, 2692, 1404, 2693, 1403, 2693, 1403, 2693, 1403, 2693, 1403, 2694, 1403, 2693, 1403, 2693, 1403, 2693, 1403, 2693, 1404, 2692, 1404, 2691, 1405, 2691, 1406, 2690, 1406, 2690, 1406, 2690, 1407, 2689, 1407, 2689, 1407, 2689, 1407, 2689, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2686, 1410, 2686, 1410, 2686, 1410, 2686, 1410, 2685, 1413, 2683, 1413, 2683, 1413, 2683, 1413, 2683, 1413, 2683, 1413, 2683, 1413, 2683, 1413, 2683, 1413, 2682, 1414, 2682, 1414, 2682, 1414, 2682, 1412, 2684, 1412, 2684, 1412, 2684, 1412, 2684, 1412, 2683, 1413, 2682, 1410, 2685, 1411, 2684, 1412, 2683, 1413, 2682, 1418, 2677, 1419, 2676, 1420, 2675, 1421, 2675, 1421, 2674, 1422, 2673, 1425, 2671, 1425, 2670, 1426, 2670, 1426, 2669, 1427, 2669, 1427, 2668, 1428, 2668, 1428, 2667, 1429, 2667, 1429, 2666, 1428, 2668, 1428, 2667, 1427, 2669, 1427, 2668, 1430, 2666, 1430, 2666, 1428, 2667, 1429, 2667, 1431, 2665, 1431, 2664, 1434, 2662, 1434, 2662, 1434, 2662, 1434, 2661, 1433, 2663, 1433, 2663, 1431, 2665, 1431, 2665, 1431, 2665, 1431, 2665, 1431, 2665, 1431, 2665, 1435, 2660, 1436, 2660, 1436, 2660, 1436, 2660, 1436, 2660, 1436, 2660, 1436, 2660, 1436, 2660, 1436, 2660, 1436, 2660, 1436, 2660, 1436, 2660, 1436, 2660, 1436, 2660, 1436, 2660, 1436, 2660, 1436, 2660, 1436, 2660, 1436, 2660, 1436, 2660, 1434, 2662, 1434, 2662, 1434, 2662, 1434, 2662, 1434, 2662, 1434, 2662, 1432, 2664, 1432, 2664, 1432, 2664, 1432, 2664, 1434, 2662, 1434, 2662, 1436, 2660, 1436, 2660, 1436, 2660, 1436, 2660, 1436, 2660, 1436, 2660, 1436, 2660, 1436, 2660, 1436, 2660, 1436, 2660, 1436, 2660, 1436, 2660, 1436, 2660, 1436, 2660, 1434, 2662, 1434, 2662, 1436, 2660, 1436, 2660, 1436, 2660, 1436, 2660, 1436, 2660, 1436, 2660, 1436, 2660, 1436, 2660, 1436, 2660, 1436, 2660, 1436, 2660, 1436, 2660, 1436, 2660, 1436, 2660, 1436, 2661, 1435, 2661, 1433, 2663, 1433, 2663, 1431, 2665, 1431, 2665, 1433, 2663, 1433, 2663, 1435, 2661, 1435, 2661, 1435, 2661, 1435, 2661, 1435, 2661, 1435, 2661, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1435, 2661, 1435, 2661, 1435, 2661, 1435, 2661, 1435, 2661, 1435, 2661, 1435, 2661, 1435, 2661, 1435, 2661, 1435, 2661, 1435, 2661, 1435, 2661, 1435, 2661, 1435, 2661, 1435, 2661, 1435, 2661, 1435, 2661, 1435, 2661, 1435, 2661, 1435, 2661, 1435, 2662, 1434, 2662, 1432, 2664, 1432, 2664, 1432, 2665, 1431, 2665, 1431, 2665, 1431, 2665, 1431, 2665, 1431, 2665, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1431, 2665, 1431, 2665, 1431, 2665, 1431, 2665, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2664, 1432, 2664, 1432, 2664, 1432, 2664, 1432, 2665, 1431, 2665, 1431, 2665, 1431, 2665, 1431, 2665, 1431, 2666, 1430, 2666, 1430, 2666, 1430, 2666, 1430, 2667, 1429, 2667, 1429, 2667, 1429, 2667, 1429, 2667, 1429, 2668, 1428, 2668, 1428, 2668, 1426, 2670, 1426, 2670, 1428, 2668, 1428, 2668, 1428, 2668, 1428, 2668, 1428, 2669, 1427, 2669, 1427, 2669, 1427, 2669, 1427, 2669, 1427, 2669, 1427, 2669, 1427, 2669, 1427, 2669, 1427, 2670, 1426, 2670, 1426, 2670, 1426, 2670, 1426, 2671, 1425, 2671, 1425, 2671, 1425, 2671, 1425, 2671, 1425, 2671, 1425, 2671, 1425, 2671, 1425, 2671, 1425, 2671, 1425, 2671, 1425, 2671, 1425, 2671, 1425, 2671, 1425, 2671, 1425, 2671, 1425, 2671, 1425, 2671, 1425, 2671, 1425, 2671, 1425, 2671, 1425, 2671, 1425, 2671, 1425, 2672, 1424, 2672, 1424, 2672, 1424, 2672, 1424, 2673, 1423, 2673, 1423, 2673, 1423, 2673, 1423, 2673, 1423, 2674, 1422, 2674, 1422, 2674, 1422, 2674, 1422, 2675, 1421, 2675, 1421, 2675, 1421, 2675, 1421, 2675, 1421, 2676, 1420, 2676, 1420, 2676, 1420, 2676, 1420, 2677, 1419, 2677, 1419, 2677, 1419, 2677, 1419, 2678, 1418, 2678, 1418, 2678, 1418, 2678, 1418, 2678, 1418, 2678, 1418, 2678, 1418, 2678, 1418, 2678, 1418, 2679, 1417, 2679, 1417, 2679, 1417, 2679, 1417, 2679, 1417, 2679, 1417, 2679, 1417, 2679, 1417, 2679, 1417, 2679, 1417, 2679, 1417, 2679, 1417, 2679, 1417, 2679, 1417, 2679, 1417, 2679, 1417, 2679, 1417, 2679, 1417, 2679, 1417, 2680, 1416, 2680, 1416, 2680, 1416, 2680, 1416, 2681, 1415, 2681, 1413, 2683, 1413, 2683, 1413, 2684, 1412, 2684, 1414, 2682, 1414, 2682, 1414, 2682, 1414, 2682, 1414, 2682, 1414, 2682, 1412, 2684, 1412, 2684, 1414, 2682, 1414, 2682, 1410, 2686, 1410, 2686, 1412, 2684, 1412, 2684, 1412, 2684, 1412, 2684, 1414, 2682, 1414, 2683, 1413, 2683, 1413, 2683, 1413, 2683, 1413, 2683, 1413, 2683, 1413, 2683, 1413, 2683, 1413, 2684, 1412, 2684, 1412, 2684, 1412, 2684, 1412, 2684, 1412, 2684, 1412, 2684, 1410, 2686, 1410, 2686, 1408, 2689, 1407, 2689, 1407, 2689, 1407, 2689, 1409, 2688, 1408, 2688, 1408, 2688, 1408, 2688, 1410, 2686, 1410, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2688, 1408, 2688, 1408, 2688, 1408, 2688, 1408, 2689, 1407, 2689, 1407, 2689, 1407, 2689, 1405, 2692, 1404, 2692, 1406, 2690, 1406, 2690, 1406, 2691, 1405, 2691, 1405, 2691, 1405, 2691, 1405, 2691, 1405, 2691, 1403, 2694, 1402, 2694, 1404, 2692, 1404, 2692, 1404, 2693, 1403, 2693, 1403, 2693, 1403, 2693, 1403, 2694, 1402, 2694, 1402, 2694, 1402, 2694, 1402, 2694, 1402, 2694, 1402, 2694, 1402, 2694, 1400, 2696, 1400, 2697, 1399, 2697, 1399, 2697, 1401, 2695, 1401, 2695, 1401, 2695, 1401, 2695, 1401, 2695, 1401, 2695, 1401, 2696, 1400, 2696, 1400, 2696, 1400, 2696, 1398, 2699, 1397, 2699, 1399, 2697, 1399, 2697, 1399, 2698, 1398, 2698, 1398, 2698, 1398, 2698, 1398, 2699, 1397, 2699, 1397, 2699, 1397, 2700, 1394, 2702, 1394, 2702, 1394, 2703, 1393, 2703, 1393, 2703, 1393, 2704, 1392, 2704, 1392, 2704, 1392, 2704, 1392, 2705, 1391, 2705, 1391, 2705, 1391, 2705, 1391, 2706, 1389, 2707, 1390, 2706, 1390, 2707, 1389, 2707, 1390, 2706, 1390, 2707, 1389, 2707, 1389, 2707, 1390, 2707, 1389, 2707, 1389, 2707, 1389, 2707, 1389, 2707, 1389, 2707, 1389, 2707, 1389, 2707, 1389, 2707, 1389, 2707, 1389, 2707, 1389, 2707, 1389, 2707, 1389, 2707, 1389, 2708, 1388, 2708, 1388, 2708, 1387, 2709, 1387, 2710, 1386, 2710, 1386, 2710, 1386, 2711, 1385, 2711, 1385, 2711, 1385, 2711, 1385, 2711, 1384, 2712, 1384, 2712, 1386, 2710, 1386, 2710, 1386, 2710, 1386, 2710, 1386, 2711, 1385, 2711, 1385, 2711, 1385, 2711, 1383, 2714, 1382, 2714, 1380, 2716, 1379, 2717, 1379, 2718, 1378, 2718, 1378, 2718, 1377, 2719, 1377, 2720, 1376, 2720, 1376, 2720, 1376, 2721, 1375, 2721, 1375, 2721, 1375, 2722, 1374, 2722, 1373, 2723, 1373, 2723, 1373, 2723, 1373, 2723, 1372, 2724, 1372, 2724, 1372, 2725, 1371, 2725, 1371, 2725, 1370, 2726, 1370, 2726, 1370, 2726, 1370, 2726, 1369, 2727, 1369, 2727, 1369, 2728, 1368, 2728, 1368, 2728, 1367, 2729, 1367, 2730, 1366, 2730, 1366, 2730, 1365, 2731, 1365, 2731, 1365, 2732, 1363, 2733, 1363, 2733, 1363, 2733, 1362, 2735, 1361, 2735, 1360, 2736, 1360, 2736, 1360, 2737, 1358, 2738, 1358, 2738, 1358, 2738, 1357, 2740, 1356, 2740, 1355, 2741, 1355, 2742, 1354, 2742, 1354, 2742, 1354, 2743, 1352, 2744, 1352, 2745, 1351, 2745, 1351, 2746, 1349, 2747, 1349, 2747, 1349, 2748, 1347, 2749, 1347, 2749, 1347, 2749, 1346, 2751, 1345, 2751, 1345, 2751, 1345, 2751, 1345, 2751, 1345, 2752, 1344, 2752, 1343, 2753, 1343, 2753, 1343, 2753, 1343, 2753, 1342, 2754, 1342, 2754, 1342, 2754, 1341, 2756, 1340, 2756, 1340, 2756, 1339, 2757, 1339, 2758, 1337, 2759, 1337, 2759, 1337, 2759, 1337, 2760, 1336, 2760, 1335, 2761, 1335, 2761, 1335, 2762, 1334, 2762, 1333, 2763, 1333, 2763, 1333, 2763, 1332, 2765, 1331, 2765, 1331, 2765, 1330, 2766, 1330, 2767, 1328, 2768, 1328, 2768, 1327, 2769, 1327, 2769, 1326, 2771, 1325, 2771, 1324, 2772, 1324, 2772, 1323, 2774, 1322, 2774, 1322, 2774, 1321, 2776, 1320, 2776, 1320, 2776, 1319, 2778, 1318, 2778, 1318, 2778, 1317, 2779, 1317, 2779, 1317, 2779, 1316, 2780, 1316, 2780, 1316, 2781, 1314, 2782, 1314, 2782, 1313, 2783, 1313, 2783, 1313, 2783, 1313, 2783, 1312, 2784, 1312, 2784, 1312, 2785, 1311, 2785, 1310, 2786, 1310, 2786, 1310, 2786, 1310, 2786, 1310, 2786, 1309, 2787, 1309, 2787, 1309, 2788, 1308, 2788, 1307, 2789, 1307, 2789, 1307, 2790, 1306, 2790, 1306, 2790, 1305, 2792, 1304, 2792, 1304, 2792, 1304, 2793, 1302, 2794, 1302, 2795, 1301, 2795, 1300, 2796, 1300, 2796, 1300, 2796, 1299, 2798, 1298, 2798, 1298, 2798, 1297, 2799, 1297, 2800, 1296, 2800, 1295, 2801, 1295, 2802, 1294, 2802, 1293, 2803, 1293, 2804, 1291, 2805, 1291, 2806, 1289, 2807, 1289, 2807, 1289, 2807, 1288, 2809, 1287, 2809, 1286, 2810, 1286, 2810, 1286, 2811, 1284, 2812, 1284, 2812, 1284, 2813, 1282, 2814, 1282, 2814, 1281, 2816, 1280, 2816, 1280, 2817, 1278, 2818, 1278, 2818, 1278, 2818, 1277, 2819, 1277, 2820, 1275, 2821, 1275, 2821, 1274, 2822, 1274, 2823, 1272, 2824, 1272, 2824, 1271, 2826, 1270, 2826, 1270, 2826, 1269, 2828, 1268, 2828, 1268, 2829, 1266, 2830, 1266, 2830, 1266, 2831, 1264, 2832, 1264, 2832, 1263, 2834, 1262, 2834, 1261, 2835, 1261, 2835, 1260, 2836, 1259, 2837, 1259, 2837, 1258, 2838, 1258, 2839, 1256, 2840, 1256, 2840, 1255, 2841, 1255, 2842, 1253, 2843, 1253, 2843, 1252, 2845, 1251, 2845, 1250, 2846, 1250, 2847, 1248, 2848, 1248, 2849, 1246, 2850, 1246, 2850, 1245, 2851, 1245, 2851, 1244, 2853, 1243, 2853, 1242, 2854, 1242, 2854, 1241, 2856, 1240, 2856, 1239, 2857, 1239, 2858, 1237, 2859, 1237, 2859, 1236, 2861, 1234, 2862, 1234, 2863, 1232, 2864, 1232, 2864, 1231, 2866, 1230, 2866, 1229, 2867, 1228, 2869, 1227, 2869, 1226, 2871, 1224, 2872, 1223, 2873, 1223, 2874, 1222, 2874, 1222, 2874, 1222, 2875, 1220, 2876, 1219, 2878, 1217, 2879, 1217, 2880, 1215, 2881, 1214, 2882, 1214, 2883, 1212, 2884, 1211, 2885, 1210, 2886, 1210, 2886, 1209, 2887, 1208, 2888, 1208, 2888, 1207, 2889, 1206, 2891, 1204, 2892, 1204, 2892, 1203, 2893, 1202, 2895, 1201, 2895, 1200, 2896, 1200, 2896, 1199, 2897, 1198, 2899, 1196, 2900, 1195, 2901, 1193, 2903, 1193, 2904, 1192, 2904, 1191, 2905, 1191, 2906, 1190, 2906, 1189, 2907, 1189, 2908, 1187, 2909, 1187, 2910, 1185, 2911, 1184, 2913, 1183, 2913, 1182, 2915, 1180, 2916, 1180, 2917, 1178, 2918, 1178, 2919, 1176, 2920, 1175, 2922, 1174, 2922, 1173, 2924, 1171, 2926, 1170, 2926, 1169, 2928, 1168, 2928, 1167, 2929, 1167, 2929, 1166, 2930, 1165, 2931, 1165, 2931, 1164, 2932, 1164, 2932, 1163, 2934, 1162, 2934, 1161, 2935, 1161, 2935, 1160, 2937, 1159, 2937, 1158, 2938, 1158, 2938, 1157, 2940, 1156, 2940, 1155, 2941, 1155, 2941, 1154, 2943, 1153, 2943, 1152, 2944, 1152, 2945, 1150, 2946, 1150, 2946, 1149, 2948, 1148, 2948, 1148, 2948, 1147, 2950, 1146, 2950, 1146, 2950, 1145, 2951, 1145, 2952, 1143, 2953, 1143, 2953, 1142, 2954, 1142, 2955, 1140, 2956, 1140, 2956, 1139, 2957, 1139, 2957, 1138, 2959, 1137, 2959, 1137, 2959, 1136, 2960, 1136, 2961, 1135, 2961, 1134, 2962, 1134, 2963, 1132, 2964, 1132, 2964, 1132, 2965, 1130, 2966, 1130, 2967, 1129, 2967, 1128, 2969, 1127, 2969, 1127, 2969, 1127, 2970, 1125, 2971, 1125, 2971, 1125, 2971, 1125, 2972, 1124, 2972, 1123, 2973, 1123, 2973, 1123, 2974, 1122, 2974, 1121, 2975, 1121, 2975, 1121, 2976, 1120, 2976, 1120, 2976, 1119, 2978, 1117, 2979, 1116, 2980, 1115, 2982, 1113, 2983, 1112, 2984, 1111, 2985, 1111, 2985, 1110, 2986, 1109, 2987, 1109, 2988, 1107, 2989, 1107, 2989, 1106, 2990, 1106, 2991, 1104, 2992, 1104, 2992, 1104, 2993, 1102, 2994, 1102, 2994, 1102, 2995, 1100, 2996, 1100, 2996, 1099, 2998, 1098, 2998, 1098, 2998, 1097, 2999, 1097, 2999, 1097, 3000, 1095, 3001, 1095, 3001, 1095, 3001, 1095, 3002, 1093, 3003, 1093, 3003, 1093, 3004, 1092, 3004, 1092, 3004, 1091, 3006, 1090, 3006, 1089, 3008, 1088, 3008, 1088, 3008, 1087, 3009, 1087, 3009, 1087, 3010, 1085, 3011, 1085, 3011, 1085, 3011, 1085, 3012, 1083, 3013, 1083, 3013, 1083, 3014, 1082, 3014, 1082, 3014, 1081, 3016, 1080, 3016, 1079, 3017, 1078, 3019, 1077, 3019, 1076, 3020, 1076, 3020, 1075, 3022, 1074, 3022, 1073, 3023, 1073, 3023, 1072, 3025, 1071, 3025, 1070, 3026, 1070, 3027, 1068, 3028, 1068, 3028, 1067, 3030, 1066, 3030, 1065, 3032, 1064, 3032, 1063, 3033, 1063, 3033, 1062, 3034, 1062, 3035, 1060, 3036, 1060, 3036, 1059, 3037, 1059, 3038, 1057, 3039, 1056, 3040, 1055, 3041, 1055, 3042, 1053, 3043, 1052, 3044, 1052, 3044, 1051, 3046, 1050, 3046, 1049, 3047, 1049, 3047, 1048, 3048, 1048, 3048, 1047, 3050, 1046, 3050, 1045, 3051, 1045, 3051, 1044, 3053, 1043, 3053, 1042, 3054, 1042, 3054, 1042, 3055, 1040, 3056, 1040, 3056, 1039, 3057, 1039, 3058, 1037, 3059, 1037, 3059, 1036, 3061, 1035, 3061, 1035, 3061, 1034, 3063, 1033, 3063, 1032, 3064, 1031, 3066, 1030, 3066, 1029, 3067, 1028, 3068, 1028, 3069, 1026, 3070, 1026, 3070, 1025, 3071, 1025, 3072, 1023, 3073, 1023, 3073, 1023, 3074, 1021, 3075, 1021, 3075, 1021, 3076, 1019, 3077, 1019, 3077, 1018, 3079, 1017, 3079, 1017, 3079, 1016, 3080, 1016, 3080, 1016, 3081, 1014, 3082, 1014, 3082, 1013, 3083, 1013, 3084, 1012, 3084, 1011, 3085, 1011, 3085, 1011, 3085, 1010, 3087, 1009, 3087, 1008, 3088, 1008, 3088, 1008, 3088, 1007, 3089, 1007, 3089, 1006, 3090, 1006, 3091, 1004, 3092, 1004, 3092, 1003, 3093, 1003, 3093, 1003, 3093, 1002, 3094, 1002, 3094, 1002, 3094, 1001, 3096, 1000, 3096, 999, 3097, 999, 3097, 999, 3098, 997, 3099, 997, 3099, 997, 3099, 996, 3100, 996, 3100, 996, 3100, 996, 3100, 995, 3101, 995, 3101, 995, 3101, 995, 3101, 995, 3101, 995, 3101, 995, 3101, 995, 3101, 994, 3102, 994, 3102, 994, 3102, 994, 3102, 994, 3102, 993, 3103, 993, 3103, 993, 3104, 992, 3104, 991, 3105, 991, 3105, 991, 3105, 991, 3105, 991, 3105, 990, 3106, 990, 3106, 990, 3107, 989, 2812, 35, 260, 988, 2809, 43, 256, 988, 2804, 51, 253, 988, 2800, 58, 250, 988, 2797, 63, 248, 987, 2795, 68, 246, 987, 2793, 72, 244, 987, 2791, 75, 243, 986, 2774, 95, 241, 986, 2770, 100, 240, 985, 2766, 106, 239, 985, 2762, 111, 238, 984, 2758, 117, 237, 984, 2754, 122, 236, 983, 2751, 127, 235, 983, 2747, 132, 234, 982, 2745, 137, 232, 982, 2742, 142, 231, 981, 2740, 146, 229, 980, 2739, 149, 228, 980, 2737, 153, 226, 979, 2736, 157, 224, 979, 2734, 162, 221, 978, 2733, 167, 218, 978, 2731, 171, 216, 978, 2730, 174, 214, 977, 2729, 178, 212, 977, 2728, 181, 210, 977, 2726, 185, 208, 976, 2725, 189, 206, 976, 2724, 192, 204, 976, 2722, 195, 203, 976, 2721, 198, 201, 975, 2721, 201, 199, 975, 2720, 204, 197, 975, 2719, 207, 195, 975, 2717, 211, 193, 974, 2717, 214, 191, 974, 2716, 217, 189, 973, 2716, 220, 187, 973, 2715, 223, 185, 972, 2715, 226, 183, 972, 2713, 230, 181, 972, 2712, 233, 179, 971, 2712, 235, 178, 971, 2711, 237, 177, 971, 2710, 239, 176, 970, 2710, 242, 174, 970, 2709, 244, 173, 969, 2709, 247, 171, 969, 2708, 249, 170, 969, 2707, 252, 168, 968, 2707, 255, 166, 968, 2706, 257, 165, 968, 2705, 260, 163, 967, 2705, 262, 162, 967, 2704, 264, 161, 966, 2704, 266, 160, 966, 2703, 268, 159, 966, 2703, 269, 158, 965, 2703, 271, 157, 965, 2702, 273, 156, 965, 2702, 274, 155, 964, 2702, 276, 154, 964, 2701, 278, 153, 964, 2700, 280, 152, 963, 2701, 281, 151, 963, 2700, 283, 150, 963, 2699, 285, 149, 962, 2700, 286, 148, 962, 2699, 288, 147, 962, 2699, 289, 146, 961, 2699, 291, 145, 961, 2698, 293, 144, 960, 2699, 294, 143, 960, 2698, 296, 142, 960, 2697, 298, 141, 959, 2698, 299, 140, 959, 2697, 300, 140, 959, 2696, 302, 139, 958, 2696, 304, 138, 958, 2696, 304, 138, 958, 2695, 306, 137, 958, 2694, 307, 137, 957, 2695, 308, 136, 957, 2694, 309, 136, 957, 2694, 310, 135, 957, 2693, 311, 135, 957, 2693, 312, 134, 956, 2693, 313, 134, 956, 2693, 314, 133, 956, 2692, 316, 132, 956, 2692, 317, 131, 955, 2692, 318, 131, 955, 2692, 319, 130, 955, 2691, 321, 129, 955, 2691, 322, 128, 954, 2691, 324, 127, 954, 2691, 325, 126, 954, 2691, 325, 126, 954, 2690, 327, 125, 953, 2691, 328, 124, 953, 2691, 329, 123, 953, 2690, 331, 122, 953, 2690, 332, 121, 953, 2689, 334, 120, 952, 2690, 334, 120, 952, 2690, 335, 119, 952, 2689, 337, 118, 952, 2689, 338, 117, 951, 2689, 342, 114, 951, 2689, 345, 111, 951, 2688, 348, 109, 951, 2688, 351, 106, 951, 2687, 355, 103, 951, 2687, 357, 101, 951, 2687, 359, 99, 951, 2686, 362, 97, 951, 2686, 365, 93, 951, 2687, 367, 88, 954, 2686, 373, 80, 957, 2686, 377, 73, 960, 2686, 381, 64, 965, 2686, 390, 51, 968, 2686, 400, 38, 972, 2686, 1410, 2686, 1410, 2686, 1409, 2687, 1409, 2686, 1410, 2686, 1410, 2685, 1411, 2685, 1411, 2685, 1411, 2684, 1412, 2684, 1412, 2684, 1412, 2683, 1413, 2683, 1413, 2682, 1414, 2682, 1414, 2681, 1414, 2682, 1414, 2682, 1414, 2682, 1413, 2682, 1414, 2682, 1414, 2682, 1414, 2682, 1413, 2682, 1414, 2682, 1414, 2682, 1414, 2682, 1413, 2683, 1413, 2683, 1413, 2683, 1413, 2683, 1413, 2683, 1412, 2683, 1413, 2683, 1415, 2681, 1415, 2681, 1415, 2681, 1415, 2681, 1415, 2681, 1415, 2681, 1415, 2681, 1415, 2680, 1416, 2680, 1416, 2680, 1416, 2680, 1416, 2680, 1416, 2680, 1416, 2680, 1414, 2682, 1409, 1, 4, 2682, 1409, 2687, 1409, 2687, 1409, 2687, 1408, 2688, 1408, 2688, 1408, 2687, 1409, 2687, 1409, 2687, 1408, 2688, 1408, 2687, 1409, 2687, 1409, 2687, 1408, 2688, 1408, 2688, 1408, 2688, 1408, 2688, 1408, 2688, 1407, 2689, 1407, 2689, 1407, 2689, 1407, 2689, 1406, 2690, 1406, 2690, 1406, 2690, 1406, 2690, 1405, 2691, 1405, 2691, 1405, 2691, 1405, 2691, 1405, 2691, 1405, 2691, 1405, 2691, 1405, 2691, 1405, 2691, 1404, 2692, 1404, 2692, 1404, 2691, 1405, 2691, 1405, 2691, 1405, 2691, 1405, 2691, 1405, 2691, 1405, 2691, 1405, 2691, 1405, 2691, 1405, 2690, 1406, 2690, 1406, 2690, 1406, 2690, 1406, 2690, 1406, 2690, 1406, 2690, 1406, 2690, 1406, 2690, 1406, 2690, 1406, 2690, 1406, 2690, 1406, 2690, 1406, 2690, 1406, 2690, 1406, 2690, 1406, 2690, 1406, 2690, 1406, 2690, 1406, 2690, 1406, 2690, 1406, 2690, 1406, 2690, 1406, 2690, 1406, 2690, 1406, 2690, 1407, 2689, 1407, 2689, 1407, 2689, 1407, 2689, 1407, 2689, 1407, 2689, 1407, 2689, 1407, 2689, 1407, 2689, 1407, 2689, 1407, 2689, 1407, 2689, 1407, 2689, 1407, 2689, 1407, 2689, 1407, 2689, 1407, 2689, 1407, 2689, 1408, 2688, 1408, 2688, 1408, 2688, 1408, 2688, 1408, 2688, 1408, 2688, 1408, 2688, 1408, 2688, 1408, 2688, 1408, 2688, 1408, 2688, 1408, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1409, 2687, 1410, 2686, 1410, 2686, 1410, 2686, 1410, 2686, 1410, 2686, 1410, 2686, 1410, 2686, 1410, 2686, 1410, 2686, 1411, 2685, 1411, 2685, 1411, 2685, 1411, 2685, 1411, 2685, 1412, 2684, 1412, 2684, 1412, 2684, 1412, 2684, 1413, 2683, 1413, 2683, 1413, 2683, 1413, 2683, 1414, 2682, 1414, 2682, 1415, 2681, 1415, 2681, 1415, 2681, 1416, 2680, 1416, 2680, 1416, 2680, 1417, 2679, 1418, 2678, 1418, 2678, 1419, 2677, 1420, 2676, 1420, 2676, 1421, 2676, 1420, 2676, 1421, 2675, 1421, 2675, 1422, 2674, 1422, 2674, 1422, 2674, 1423, 2673, 1423, 2673, 1423, 2674, 1423, 2673, 1423, 2673, 1423, 2673, 1424, 2672, 1424, 2672, 1424, 2672, 1425, 2671, 1425, 2671, 1425, 2671, 1426, 2670, 1426, 2670, 1426, 2670, 1426, 2670, 1427, 2669, 1427, 2669, 1427, 2669, 1427, 2669, 1428, 2668, 1428, 2669, 1427, 2669, 1427, 2669, 1428, 2668, 1428, 2668, 1428, 2668, 1428, 2668, 1428, 2668, 1428, 2668, 1428, 2668, 1428, 2668, 1428, 2668, 1429, 2667, 1429, 2667, 1429, 2667, 1429, 2667, 1429, 2667, 1429, 2667, 1429, 2668, 1428, 2668, 1428, 2668, 1429, 2667, 1429, 2667, 1429, 2667, 1429, 2667, 1429, 2667, 1429, 2667, 1429, 2668, 1428, 2668, 1428, 2668, 1429, 2667, 1429, 2667, 1429, 2667, 1429, 2667, 1429, 2667, 1429, 2667, 1429, 2667, 1429, 2667, 1429, 2667, 1430, 2666, 1430, 2666, 1430, 2666, 1430, 2666, 1430, 2666, 1430, 2666, 1430, 2666, 1430, 2666, 1430, 2666, 1430, 2666, 1430, 2666, 1430, 2666, 1430, 2666, 1430, 2666, 1430, 2666, 1430, 2666, 1430, 2666, 1430, 2666, 1431, 2665, 1431, 2665, 1431, 2665, 1431, 2665, 1431, 2665, 1431, 2665, 1431, 2666, 1430, 2666, 1430, 2666, 1431, 2665, 1431, 2665, 1431, 2665, 1431, 2665, 1431, 2665, 1432, 2664, 1432, 2664, 1432, 2664, 1432, 2664, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1433, 2663, 1434, 2662, 1434, 2662, 1435, 2661, 1436, 2660, 1436, 2660, 1437, 2659, 1437, 2659, 1438, 2658, 1438, 2658, 1439, 2657, 1439, 2657, 1439, 2657, 1439, 2656, 1440, 2656, 1440, 2656, 1440, 2656, 1440, 2655, 1441, 2655, 1441, 2655, 1441, 2655, 1441, 2654, 1442, 2654, 1442, 2654, 1442, 2653, 1443, 2652, 1444, 2651, 1445, 2650, 1446, 2649, 1447, 2648, 1448, 2647, 1449, 2646, 1450, 2645, 1451, 2644, 1452, 2644, 1452, 2643, 1453, 2642, 1454, 2642, 1454, 2641, 1455, 2641, 1455, 2640, 1456, 2639, 1457, 2639, 1457, 2638, 1458, 2637, 1459, 2637, 1459, 2636, 1460, 2636, 1460, 2635, 1461, 2635, 1461, 2634, 1462, 2634, 1462, 2634, 1462, 2633, 1463, 2633, 1463, 2633, 1463, 2632, 1464, 2632, 1464, 2632, 1464, 2632, 1464, 2631, 1465, 2631, 1465, 2630, 1466, 2630, 1466, 2629, 1467, 2629, 1467, 2628, 1468, 2628, 1467, 2629, 1467, 2628, 1468, 2628, 1468, 2628, 1468, 2627, 1469, 2627, 1469, 2627, 1469, 2626, 1470, 2626, 1470, 2626, 1470, 2625, 1471, 2625, 1471, 2625, 1471, 2624, 1472, 2624, 1472, 2623, 1473, 2623, 1473, 2623, 1472, 2623, 1473, 2623, 1473, 2623, 1473, 2622, 1474, 2622, 1474, 2621, 1475, 2621, 1475, 2621, 1475, 2620, 1476, 2620, 1476, 2620, 1476, 2619, 1477, 2619, 1477, 2619, 1477, 2619, 1477, 2618, 1478, 2618, 1478, 2618, 1478, 2618, 1478, 2618, 1478, 2617, 1479, 2617, 1479, 2617, 1479, 2617, 1479, 2616, 1480, 2616, 1480, 2616, 1479, 2617, 1479, 2617, 1479, 2616, 1480, 2616, 1480, 2616, 1480, 2616, 1480, 2615, 1481, 2615, 1481, 2615, 1481, 2615, 1481, 2615, 1481, 2615, 1481, 2615, 1481, 2614, 1482, 2614, 1482, 2614, 1482, 2613, 1483, 2613, 1482, 2614, 1482, 2613, 1483, 2613, 1483, 2613, 1483, 2613, 1483, 2612, 1484, 2612, 1484, 2612, 1483, 2613, 1483, 2613, 1483, 2613, 1482, 2614, 1482, 2614, 1482, 2614, 1481, 2614, 1482, 2614, 1482, 2614, 1482, 2614, 1481, 2615, 1481, 2615, 1481, 2615, 1481, 2615, 1480, 2615, 1480, 2616, 1479, 2617, 1477, 2619, 1475, 2621, 1473, 2623, 1470, 2626, 1467, 2629, 1463, 2633, 1454, 2641, 1446, 2650, 1437, 2659, 1428, 2668, 1420, 2676, 1415, 2681, 1411, 2685, 1402, 2694, 1397, 2699, 1393, 2703, 1100, 105, 184, 2707, 1091, 116, 178, 2711, 1082, 127, 167, 2720, 1068, 143, 160, 2725, 1064, 150, 153, 2729, 1055, 162, 109, 1, 10, 2759, 1051, 170, 102, 6, 8, 2759, 1051, 174, 98, 6, 8, 2759, 1051, 178, 96, 4, 8, 2759, 1051, 190, 84, 4, 8, 2759, 1035, 4, 10, 196, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 46, 2777, 1035, 4, 10, 196, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 46, 2777, 1028, 3068, 1026, 3070, 1025, 3071, 1023, 3073, 1022, 3074, 1020, 3076, 1018, 3077, 1017, 3079, 1016, 3080, 1016, 3080, 1015, 3081, 1014, 3082, 1013, 3083, 1012, 3084, 1011, 3085, 1011, 3085, 1010, 3086, 1009, 3087, 1009, 3087, 1008, 3088, 1008, 3088, 1007, 3089, 1006, 3090, 1006, 3090, 1005, 3091, 1005, 3091, 1005, 3091, 1004, 3092, 1004, 3092, 1004, 3092, 1003, 3093, 1003, 3093, 1002, 3094, 1002, 3094, 1001, 3095, 1001, 3095, 1000, 3096, 1000, 3096, 999, 3097, 999, 3097, 999, 3097, 998, 3098, 998, 3098, 998, 3098, 998, 3098, 998, 3098, 997, 3099, 997, 3099, 997, 3099, 997, 3099, 996, 3101, 995, 3101, 995, 3101, 994, 3102, 994, 3102, 994, 3102, 993, 3103, 993, 3103, 992, 3104, 992, 3105, 991, 3105, 991, 3105, 991, 3105, 990, 3106, 990, 3106, 990, 3106, 990, 3106, 990, 3106, 990, 3107, 989, 3107, 989, 3107, 989, 3107, 988, 3108, 988, 3108, 988, 3108, 988, 3108, 988, 3108, 988, 3108, 988, 3108, 988, 3108, 988, 3108, 988, 3108, 988, 3108, 988, 3108, 988, 3108, 988, 3107, 989, 3107, 989, 3106, 990, 3106, 990, 3105, 991, 3104, 992, 3103, 993, 3103, 993, 3102, 994, 3101, 995, 3101, 995, 3100, 996, 3100, 996, 3099, 997, 3099, 997, 3098, 998, 3098, 998, 3098, 998, 3097, 999, 3097, 999, 3097, 999, 3096, 1000, 3096, 1001, 3095, 1001, 3095, 1001, 3094, 1002, 3094, 1002, 3094, 1002, 3094, 1002, 3094, 1002, 3093, 1003, 3093, 1004, 3092, 1004, 3092, 1004, 3091, 1005, 3091, 1005, 3091, 1005, 3091, 1005, 3091, 1005, 3091, 1006, 3090, 1006, 3090, 1006, 3090, 1007, 3088, 1008, 3088, 1008, 3088, 1009, 3087, 1009, 3087, 1009, 3087, 1009, 3087, 1010, 3086, 1010, 3086, 1010, 3086, 1010, 3086, 1010, 3086, 1010, 3086, 1010, 3086, 1010, 3086, 1010, 3086, 1010, 3086, 1010, 3086, 1010, 3086, 1010, 3086, 1011, 3085, 1011, 3085, 1011, 3085, 1011, 3085, 1012, 3084, 1012, 3084, 1012, 3085, 1011, 3085, 1011, 3085, 1011, 3085, 1011, 3086, 1010, 3086, 1010, 3086, 1010, 3087, 1009, 3087, 1009, 3087, 1009, 3088, 1008, 3088, 1008, 3089, 1007, 3089, 1007, 3089, 1007, 3090, 1006, 3090, 1006, 3090, 1006, 3091, 1005, 3091, 1005, 3091, 1005, 3091, 1005, 3091, 1005, 3091, 1005, 3091, 1005, 3091, 1005, 3091, 1005, 3094, 1002, 3095, 1001, 3095, 1001, 3095, 1001, 3095, 1001, 3095, 1001, 3095, 1001, 3095, 1001, 3097, 999, 3098, 998, 3098, 998, 3099, 997, 3099, 997, 3100, 996, 3100, 996, 3101, 995, 3101, 995, 3101, 995, 3102, 994, 3102, 994, 3102, 994, 3103, 993, 3103, 993, 3103, 993, 3104, 992, 3104, 992, 3105, 991, 3105, 991, 3105, 991, 3106, 990, 3106, 990, 3106, 990, 3107, 989, 3107, 989, 3107, 989, 3108, 988, 3108, 988, 3108, 988, 3109, 987, 3109, 987, 3109, 987, 3110, 986, 3110, 986, 3111, 985, 3111, 985, 3112, 984, 3112, 984, 3113, 983, 3113, 983, 3114, 982, 3114, 982, 3115, 981, 3115, 981, 3115, 981, 3116, 980, 3116, 980, 3116, 980, 3117, 979, 3117, 979, 3118, 978, 3118, 978, 3119, 977, 3119, 977, 3120, 976, 3120, 976, 3121, 975, 3121, 975, 3122, 974, 3122, 974, 3123, 973, 3123, 973, 3124, 972, 3124, 972, 3125, 971, 3125, 971, 3126, 970, 3126, 970, 3127, 969, 3127, 969, 3128, 967, 3130, 966, 3130, 966, 3131, 965, 3131, 965, 3132, 964, 3132, 964, 3133, 963, 3134, 962, 3134, 962, 3135, 961, 3136, 960, 3136, 960, 3137, 959, 3137, 959, 3138, 958, 3139, 957, 3139, 957, 3140, 955, 3142, 954, 3143, 953, 3143, 953, 3144, 952, 3145, 951, 3145, 951, 3146, 950, 3147, 949, 3148, 947, 3150, 946, 3151, 945, 3151, 945, 3152, 943, 3154, 942, 3155, 941, 3156, 940, 3157, 938, 3159, 937, 3160, 936, 3161, 935, 3162, 933, 3164, 932, 3165, 931, 3166, 930, 3167, 929, 3168, 927, 3170, 926, 3171, 925, 3172, 924, 3173, 922, 3175, 921, 3176, 920, 3177, 919, 3178, 918, 3179, 916, 3181, 915, 3182, 914, 3183, 913, 3184, 911, 3186, 910, 3187, 909, 3189, 907, 3190, 905, 3192, 904, 3193, 903, 3194, 902, 3195, 900, 3198, 898, 3199, 897, 3200, 895, 3202, 894, 3203, 893, 3204, 892, 3205, 891, 3206, 890, 3207, 889, 3208, 888, 3209, 887, 3210, 886, 3211, 885, 3212, 884, 3213, 883, 3214, 882, 3215, 881, 3216, 880, 3218, 878, 3219, 873, 3224, 872, 3225, 871, 3226, 870, 3227, 863, 3234, 861, 3236, 859, 3238, 858, 3239, 856, 3241, 854, 3243, 853, 3244, 851, 3246, 849, 3248, 847, 3250, 845, 3252, 843, 3254, 841, 3256, 839, 3258, 837, 3260, 835, 3262, 833, 3264, 830, 3267, 828, 3269, 825, 3273, 821, 3276, 818, 3280, 815, 3282, 812, 3285, 809, 3288, 806, 3291, 803, 3294, 800, 3297, 797, 3300, 794, 3303, 791, 3307, 787, 3310, 784, 3313, 781, 3316, 777, 3320, 773, 3324, 771, 3327, 767, 3330, 765, 3333, 761, 3337, 758, 3340, 755, 3343, 751, 3347, 748, 3349, 745, 3352, 743, 3354, 741, 3357, 737, 3360, 735, 3363, 731, 3367, 728, 3370, 725, 3372, 722, 3376, 719, 3379, 715, 3382, 712, 3386, 708, 3389, 705, 3393, 701, 3396, 699, 3398, 697, 3401, 693, 3404, 691, 3407, 687, 3410, 684, 3413, 681, 3417, 677, 3420, 674, 3424, 670, 3428, 666, 3432, 662, 3436, 658, 3440, 654, 3444, 650, 3447, 646, 3452, 641, 3456, 638, 3460, 633, 3465, 628, 3470, 624, 3475, 618, 3480, 613, 3485, 608, 3491, 602, 3496, 597, 3501, 591, 3507, 585, 3514, 579, 3520, 573, 3525, 568, 3531, 563, 3536, 557, 3541, 552, 3547, 547, 3552, 541, 3559, 534, 1180] , dtype=np.uint32)
BUTT_RLE = np.array( [9838378, 21, 4073, 25, 435, 27, 3608, 28, 429, 35, 3564, 17, 21, 31, 425, 41, 3557, 26, 10, 38, 421, 47, 3551, 78, 418, 51, 3546, 82, 415, 55, 3542, 84, 413, 59, 3529, 96, 411, 69, 3508, 108, 409, 82, 3486, 119, 408, 95, 3462, 131, 407, 107, 3440, 142, 406, 119, 3418, 153, 405, 132, 3394, 166, 403, 144, 3372, 178, 401, 157, 3348, 190, 400, 169, 3326, 202, 398, 181, 3304, 213, 397, 194, 3280, 226, 395, 206, 3258, 237, 395, 217, 3236, 249, 393, 230, 3212, 261, 392, 242, 3190, 272, 392, 254, 3166, 284, 391, 266, 3144, 295, 391, 277, 3122, 306, 390, 290, 3098, 318, 390, 301, 3076, 329, 389, 314, 3052, 341, 389, 325, 3030, 352, 389, 336, 3008, 363, 388, 349, 2984, 375, 388, 360, 2962, 386, 388, 371, 2940, 396, 388, 384, 2916, 408, 388, 395, 2894, 418, 389, 407, 2870, 429, 390, 418, 2848, 440, 389, 430, 2826, 451, 389, 442, 2802, 463, 389, 453, 2780, 474, 389, 465, 2756, 487, 388, 476, 2734, 498, 388, 487, 2712, 510, 387, 499, 2688, 522, 387, 510, 2666, 533, 387, 521, 2644, 544, 387, 533, 2620, 556, 387, 544, 2598, 567, 387, 556, 2574, 579, 387, 567, 2552, 590, 387, 578, 2530, 601, 387, 590, 2506, 613, 387, 601, 2484, 624, 387, 613, 2460, 636, 388, 623, 2438, 647, 388, 634, 2416, 658, 388, 646, 2392, 670, 388, 657, 2370, 681, 389, 667, 2348, 692, 389, 679, 2324, 704, 389, 690, 2302, 715, 390, 696, 2288, 722, 390, 698, 2284, 724, 390, 700, 2280, 726, 391, 701, 2276, 728, 391, 703, 2272, 730, 392, 704, 2268, 732, 392, 707, 2262, 735, 391, 710, 2258, 737, 391, 712, 2254, 739, 391, 714, 2250, 741, 391, 716, 2246, 743, 391, 717, 2244, 744, 391, 719, 2240, 746, 391, 720, 2238, 747, 391, 721, 2236, 748, 391, 722, 2234, 749, 390, 724, 2232, 750, 390, 725, 2230, 751, 390, 726, 2228, 752, 390, 727, 2226, 753, 390, 728, 2224, 754, 390, 729, 2222, 755, 390, 730, 2220, 756, 390, 731, 2218, 757, 390, 732, 2216, 758, 390, 733, 2214, 759, 390, 734, 2212, 760, 390, 735, 2210, 762, 389, 736, 2208, 763, 389, 737, 2206, 764, 388, 739, 2204, 765, 388, 740, 2202, 766, 388, 741, 2200, 767, 388, 742, 2198, 768, 387, 744, 2196, 769, 387, 745, 2194, 770, 387, 746, 2192, 771, 387, 747, 2190, 772, 387, 748, 2188, 773, 387, 749, 2186, 774, 387, 750, 2184, 776, 386, 751, 2182, 777, 386, 752, 2180, 779, 384, 754, 2178, 780, 384, 755, 2176, 781, 384, 756, 2174, 782, 384, 757, 2172, 783, 384, 758, 2170, 785, 383, 759, 2168, 786, 383, 760, 2166, 787, 383, 761, 2164, 788, 383, 762, 2162, 790, 381, 764, 2160, 791, 381, 765, 2158, 793, 380, 766, 2156, 794, 380, 767, 2154, 795, 379, 769, 2152, 796, 379, 770, 2150, 797, 379, 771, 2148, 799, 378, 772, 2146, 801, 376, 774, 2144, 802, 376, 775, 2142, 804, 375, 776, 2140, 805, 375, 777, 2138, 806, 375, 778, 2136, 807, 374, 780, 2134, 808, 374, 781, 2132, 809, 374, 782, 2130, 810, 374, 783, 2128, 811, 373, 785, 2126, 813, 372, 786, 2124, 814, 372, 787, 2122, 815, 372, 788, 2120, 816, 372, 789, 2118, 818, 370, 791, 2116, 819, 370, 792, 2114, 821, 369, 793, 2112, 822, 369, 794, 2110, 823, 368, 796, 2108, 824, 368, 797, 2106, 825, 368, 798, 2104, 827, 367, 798, 2104, 827, 367, 799, 2102, 828, 367, 799, 2102, 828, 367, 799, 2102, 828, 367, 799, 2102, 828, 367, 799, 2102, 828, 366, 801, 2100, 829, 366, 801, 2100, 829, 366, 801, 2100, 829, 366, 801, 2100, 829, 365, 803, 2098, 829, 366, 803, 2098, 829, 366, 803, 2098, 830, 365, 803, 2098, 830, 364, 804, 2098, 830, 364, 805, 2096, 831, 364, 805, 2096, 831, 364, 805, 2096, 831, 364, 805, 2096, 831, 364, 805, 2096, 831, 364, 806, 2094, 833, 363, 806, 2094, 833, 363, 806, 2094, 834, 361, 807, 2094, 834, 361, 808, 2092, 835, 361, 808, 2092, 835, 361, 808, 2092, 836, 360, 808, 2092, 836, 360, 808, 2092, 836, 360, 809, 2090, 837, 360, 809, 2090, 837, 360, 809, 2090, 837, 359, 810, 2090, 837, 359, 811, 2088, 838, 359, 811, 2088, 839, 358, 811, 2088, 839, 358, 811, 2088, 840, 357, 811, 2088, 840, 357, 812, 2086, 841, 357, 812, 2086, 841, 357, 812, 2086, 841, 356, 813, 2086, 841, 356, 813, 2086, 841, 356, 814, 2084, 842, 356, 814, 2084, 842, 356, 814, 2084, 842, 356, 814, 2084, 842, 356, 815, 2082, 843, 356, 815, 2082, 843, 356, 815, 2082, 844, 354, 816, 2082, 844, 354, 816, 2082, 845, 353, 817, 2080, 846, 353, 817, 2080, 847, 352, 817, 2080, 847, 352, 817, 2080, 847, 352, 818, 2078, 848, 352, 818, 2078, 848, 352, 818, 2078, 848, 351, 819, 2078, 848, 351, 819, 2078, 848, 351, 820, 2076, 849, 351, 820, 2076, 849, 351, 820, 2076, 849, 351, 820, 2076, 849, 351, 820, 2076, 850, 350, 821, 2074, 851, 350, 821, 2074, 851, 349, 822, 2074, 851, 349, 822, 2074, 851, 349, 823, 2072, 852, 349, 823, 2072, 852, 349, 823, 2072, 852, 349, 823, 2072, 852, 349, 823, 2072, 852, 349, 824, 2070, 853, 349, 824, 2070, 853, 349, 824, 2070, 853, 349, 824, 2070, 853, 349, 825, 2068, 855, 348, 825, 2068, 855, 348, 825, 2068, 856, 346, 826, 2068, 856, 346, 826, 2068, 856, 346, 827, 2066, 857, 346, 827, 2066, 857, 345, 828, 2066, 857, 345, 828, 2066, 857, 345, 828, 2066, 857, 345, 829, 2064, 858, 345, 829, 2064, 858, 345, 829, 2064, 858, 345, 829, 2064, 858, 345, 830, 2062, 859, 345, 830, 2062, 859, 344, 831, 2062, 860, 343, 831, 2062, 860, 343, 831, 2062, 860, 343, 832, 2060, 861, 343, 832, 2060, 862, 341, 834, 2058, 863, 341, 835, 2056, 865, 340, 837, 2052, 867, 339, 840, 2048, 869, 339, 841, 2046, 870, 339, 843, 2042, 872, 338, 845, 2040, 873, 338, 847, 2036, 876, 337, 849, 2032, 878, 336, 851, 2030, 879, 336, 853, 2026, 881, 335, 855, 2024, 882, 335, 857, 2020, 884, 335, 859, 2016, 886, 334, 861, 2014, 887, 334, 863, 2010, 889, 334, 864, 2008, 890, 333, 867, 2004, 892, 333, 869, 2000, 894, 333, 870, 1998, 895, 333, 872, 1994, 897, 332, 874, 1992, 898, 332, 876, 1988, 900, 332, 878, 1984, 902, 332, 879, 1982, 903, 331, 881, 1980, 905, 330, 881, 1980, 905, 330, 881, 1980, 906, 329, 881, 1980, 906, 328, 882, 1980, 907, 327, 881, 1982, 906, 327, 881, 1982, 906, 327, 881, 1982, 906, 327, 881, 1982, 906, 327, 881, 1982, 906, 327, 881, 1982, 906, 327, 881, 1982, 906, 327, 881, 1982, 906, 327, 881, 1982, 906, 327, 880, 1984, 905, 327, 880, 1984, 906, 326, 880, 1984, 906, 326, 880, 1984, 907, 325, 880, 1984, 907, 325, 880, 1984, 907, 325, 880, 1984, 907, 325, 880, 1984, 907, 324, 881, 1984, 907, 324, 881, 1984, 907, 324, 880, 1986, 906, 324, 880, 1986, 906, 324, 880, 1986, 906, 324, 880, 1986, 906, 324, 880, 1986, 907, 323, 880, 1986, 907, 323, 880, 1986, 907, 323, 880, 1986, 907, 323, 880, 1986, 907, 323, 879, 1988, 907, 322, 879, 1988, 907, 322, 879, 1988, 908, 321, 879, 1988, 908, 321, 879, 1988, 908, 321, 879, 1988, 908, 321, 879, 1988, 908, 321, 879, 1988, 908, 321, 879, 1988, 908, 321, 879, 1988, 908, 321, 878, 1990, 907, 321, 878, 1990, 907, 321, 878, 1990, 907, 321, 878, 1990, 907, 321, 878, 1990, 907, 321, 878, 1990, 907, 322, 877, 1990, 907, 322, 877, 1990, 907, 322, 877, 1990, 908, 321, 877, 1990, 908, 321, 876, 1992, 907, 321, 876, 1992, 907, 321, 876, 1992, 907, 321, 876, 1992, 907, 321, 876, 1992, 907, 322, 875, 1992, 907, 322, 875, 1992, 907, 322, 875, 1992, 907, 322, 875, 1992, 907, 323, 873, 1994, 906, 323, 873, 1994, 906, 323, 873, 1994, 906, 323, 873, 1994, 906, 323, 873, 1994, 906, 322, 874, 1994, 907, 321, 874, 1994, 907, 321, 874, 1994, 908, 320, 874, 1994, 908, 320, 874, 1994, 908, 320, 873, 1996, 907, 320, 873, 1996, 907, 320, 873, 1996, 907, 320, 873, 1996, 907, 320, 873, 1996, 908, 319, 873, 1996, 908, 319, 873, 1996, 908, 319, 873, 1996, 908, 319, 873, 1996, 908, 319, 872, 1998, 907, 319, 872, 1998, 907, 319, 872, 1998, 907, 320, 871, 1998, 907, 319, 872, 1998, 907, 319, 872, 1998, 907, 318, 873, 1998, 907, 318, 873, 1998, 907, 318, 873, 1998, 908, 316, 874, 1998, 909, 315, 873, 2000, 908, 315, 873, 2000, 909, 313, 874, 2000, 909, 313, 874, 2000, 909, 313, 874, 2000, 909, 312, 875, 2000, 909, 312, 875, 2000, 909, 312, 875, 2000, 909, 311, 876, 2000, 909, 311, 875, 2002, 908, 311, 875, 2002, 908, 310, 876, 2002, 908, 310, 876, 2002, 908, 310, 876, 2002, 908, 310, 876, 2002, 908, 309, 877, 2002, 908, 309, 877, 2002, 908, 309, 877, 2002, 909, 308, 877, 2002, 909, 307, 877, 2004, 909, 306, 877, 2004, 909, 306, 877, 2004, 909, 305, 878, 2004, 909, 305, 878, 2004, 909, 305, 878, 2004, 909, 304, 879, 2004, 909, 304, 879, 2004, 909, 304, 879, 2004, 909, 303, 879, 2006, 908, 303, 879, 2006, 909, 302, 879, 2006, 909, 302, 879, 2006, 909, 301, 880, 2006, 909, 301, 880, 2006, 909, 301, 880, 2006, 909, 301, 880, 2006, 909, 300, 881, 2006, 909, 300, 881, 2006, 909, 300, 880, 2008, 908, 300, 880, 2008, 908, 299, 881, 2008, 908, 299, 881, 2008, 909, 298, 881, 2008, 909, 298, 881, 2008, 910, 297, 881, 2008, 910, 297, 881, 2008, 910, 297, 881, 2008, 910, 296, 882, 2008, 911, 295, 881, 2010, 910, 295, 881, 2010, 910, 294, 882, 2010, 911, 293, 882, 2010, 911, 293, 882, 2010, 912, 291, 883, 2010, 912, 291, 883, 2010, 867, 1, 44, 291, 883, 2010, 912, 291, 883, 2010, 912, 290, 883, 2012, 912, 289, 883, 2012, 912, 289, 883, 2012, 913, 288, 883, 2012, 913, 288, 883, 2012, 914, 287, 883, 2012, 914, 287, 883, 2012, 914, 287, 883, 2012, 914, 287, 883, 2012, 914, 286, 884, 2012, 914, 286, 883, 2014, 913, 286, 883, 2014, 913, 286, 883, 2014, 913, 286, 883, 2014, 913, 285, 884, 2014, 914, 284, 884, 2014, 914, 284, 884, 2014, 914, 284, 884, 2014, 914, 283, 885, 2014, 914, 283, 884, 2016, 914, 282, 884, 2016, 914, 282, 884, 2016, 915, 281, 884, 2016, 915, 281, 884, 2016, 915, 281, 884, 2016, 915, 281, 884, 2016, 915, 281, 884, 2016, 916, 279, 885, 2016, 916, 279, 885, 2016, 916, 279, 884, 2018, 915, 279, 884, 2018, 915, 278, 885, 2018, 915, 278, 885, 2018, 916, 277, 885, 2018, 916, 277, 885, 2018, 917, 275, 886, 2018, 917, 275, 886, 2018, 917, 275, 886, 2018, 917, 275, 885, 2020, 917, 274, 885, 2020, 917, 274, 885, 2020, 917, 274, 885, 2020, 917, 274, 885, 2020, 917, 274, 885, 2020, 917, 273, 886, 2020, 917, 273, 886, 2020, 917, 273, 886, 2020, 917, 273, 886, 2020, 918, 272, 885, 2022, 917, 271, 886, 2022, 918, 270, 886, 2022, 918, 270, 886, 2022, 918, 270, 886, 2022, 918, 269, 887, 2022, 919, 268, 887, 2022, 919, 268, 887, 2022, 919, 268, 887, 2022, 919, 267, 888, 2022, 919, 267, 887, 2024, 918, 267, 887, 2024, 918, 267, 887, 2024, 918, 266, 888, 2024, 918, 266, 888, 2024, 919, 265, 888, 2024, 919, 265, 888, 2024, 920, 264, 888, 2024, 920, 264, 888, 2024, 920, 264, 887, 2026, 919, 263, 888, 2026, 919, 263, 888, 2026, 919, 263, 888, 2026, 919, 262, 889, 2026, 919, 262, 889, 2026, 919, 262, 889, 2026, 919, 261, 890, 2026, 919, 261, 890, 2026, 920, 260, 890, 2026, 920, 260, 889, 2028, 920, 258, 890, 2028, 920, 258, 890, 2028, 921, 257, 890, 2028, 921, 257, 890, 2028, 921, 257, 890, 2028, 921, 257, 890, 2028, 921, 257, 890, 2028, 921, 257, 890, 2028, 921, 257, 889, 2030, 920, 257, 889, 2030, 920, 257, 889, 2030, 920, 257, 889, 2030, 920, 257, 889, 2030, 920, 257, 889, 2030, 920, 257, 889, 2030, 920, 257, 889, 2030, 920, 257, 889, 2030, 920, 258, 887, 2032, 920, 257, 887, 2032, 920, 257, 887, 2032, 920, 257, 887, 2032, 920, 258, 886, 2032, 920, 258, 886, 2032, 920, 258, 886, 2032, 920, 259, 884, 2034, 919, 259, 884, 2034, 919, 259, 884, 2034, 919, 260, 883, 2034, 919, 260, 883, 2034, 919, 261, 882, 2034, 919, 261, 882, 2034, 919, 261, 881, 2036, 918, 261, 881, 2036, 918, 261, 881, 2036, 918, 261, 881, 2036, 918, 261, 881, 2036, 918, 261, 881, 2036, 918, 260, 882, 2036, 919, 259, 881, 2038, 918, 258, 882, 2038, 919, 257, 882, 2038, 919, 256, 883, 2038, 919, 256, 883, 2038, 919, 255, 884, 2038, 919, 254, 885, 2038, 919, 254, 884, 2040, 918, 253, 885, 2040, 918, 252, 886, 2040, 918, 252, 886, 2040, 918, 251, 887, 2040, 918, 251, 887, 2040, 918, 250, 888, 2040, 918, 250, 887, 2042, 917, 249, 888, 2042, 917, 248, 889, 2042, 917, 248, 889, 2042, 917, 247, 890, 2042, 917, 247, 890, 2042, 917, 246, 891, 2042, 917, 246, 890, 2044, 917, 244, 891, 2044, 917, 244, 891, 2044, 917, 243, 892, 2044, 917, 243, 892, 2044, 917, 242, 893, 2044, 918, 241, 893, 2044, 919, 239, 894, 2044, 919, 239, 893, 2046, 919, 237, 894, 2046, 919, 237, 894, 2046, 919, 237, 894, 2046, 919, 236, 895, 2046, 919, 236, 895, 2046, 919, 236, 895, 2046, 919, 235, 895, 2048, 918, 235, 895, 2048, 918, 234, 896, 2048, 918, 234, 896, 2048, 918, 234, 896, 2048, 918, 233, 897, 2048, 919, 232, 897, 2048, 919, 232, 896, 2050, 919, 230, 897, 2050, 919, 230, 897, 2050, 919, 230, 897, 2050, 919, 230, 897, 2050, 920, 228, 898, 2050, 920, 228, 898, 2050, 920, 228, 897, 2052, 919, 228, 897, 2052, 919, 228, 897, 2052, 919, 228, 897, 2052, 919, 228, 897, 2052, 919, 228, 897, 2052, 919, 228, 897, 2052, 919, 228, 896, 2054, 918, 228, 896, 2054, 918, 228, 896, 2054, 918, 228, 896, 2054, 918, 228, 896, 2054, 918, 228, 896, 2054, 918, 228, 896, 2054, 918, 228, 895, 2056, 917, 229, 894, 2056, 917, 229, 894, 2056, 917, 229, 894, 2056, 917, 229, 894, 2056, 917, 229, 894, 2056, 917, 229, 894, 2056, 917, 229, 893, 2058, 917, 228, 893, 2058, 917, 228, 893, 2058, 918, 228, 892, 2058, 918, 228, 892, 2058, 918, 228, 892, 2058, 919, 227, 892, 2058, 919, 228, 891, 2058, 919, 228, 890, 2060, 918, 228, 890, 2060, 918, 229, 889, 2060, 918, 229, 889, 2060, 918, 229, 889, 2060, 918, 229, 889, 2060, 918, 228, 890, 2060, 918, 228, 889, 2062, 917, 228, 889, 2062, 917, 227, 890, 2062, 917, 227, 890, 2062, 917, 227, 890, 2062, 918, 226, 890, 2062, 918, 225, 891, 2062, 919, 224, 890, 2064, 918, 224, 890, 2064, 918, 224, 890, 2064, 918, 223, 891, 2064, 918, 223, 891, 2064, 918, 223, 891, 2064, 918, 223, 891, 2064, 918, 222, 891, 2066, 917, 222, 891, 2066, 917, 222, 891, 2066, 917, 222, 891, 2066, 917, 222, 891, 2066, 917, 221, 892, 2066, 917, 221, 892, 2066, 917, 221, 891, 2068, 916, 221, 891, 2068, 916, 220, 892, 2068, 916, 220, 892, 2068, 916, 220, 892, 2068, 916, 220, 892, 2068, 917, 219, 892, 2068, 917, 219, 891, 2070, 916, 219, 891, 2070, 916, 219, 891, 2070, 917, 218, 891, 2070, 917, 218, 891, 2070, 918, 217, 891, 2070, 918, 217, 891, 2070, 918, 217, 890, 2072, 917, 217, 890, 2072, 917, 217, 890, 2072, 917, 217, 890, 2072, 917, 217, 890, 2072, 917, 217, 890, 2072, 917, 217, 890, 2072, 917, 217, 890, 2072, 917, 217, 889, 2074, 916, 217, 889, 2074, 916, 217, 889, 2074, 917, 216, 889, 2074, 917, 216, 889, 2074, 917, 216, 889, 2074, 917, 216, 889, 2074, 917, 216, 888, 2076, 917, 215, 888, 2076, 917, 215, 888, 2076, 918, 214, 888, 2076, 918, 214, 888, 2076, 918, 214, 888, 2076, 918, 214, 888, 2076, 918, 214, 887, 2078, 917, 215, 886, 2078, 917, 215, 886, 2078, 917, 215, 886, 2078, 917, 215, 886, 2078, 917, 215, 886, 2078, 917, 215, 886, 2078, 917, 215, 885, 2080, 917, 214, 885, 2080, 917, 214, 885, 2080, 917, 214, 885, 2080, 917, 213, 886, 2080, 917, 213, 886, 2080, 917, 212, 887, 2080, 917, 212, 886, 2082, 916, 212, 886, 2082, 916, 211, 887, 2082, 916, 211, 887, 2082, 916, 211, 887, 2082, 917, 210, 887, 2082, 917, 209, 888, 2082, 918, 208, 887, 2084, 917, 208, 887, 2084, 917, 208, 887, 2084, 917, 207, 888, 2084, 918, 206, 888, 2084, 918, 206, 888, 2084, 918, 206, 888, 2084, 918, 206, 887, 2086, 917, 206, 887, 2086, 917, 205, 888, 2086, 917, 205, 888, 2086, 917, 205, 888, 2086, 917, 205, 888, 2086, 917, 205, 888, 2086, 917, 205, 888, 2086, 918, 204, 887, 2088, 917, 204, 887, 2088, 918, 203, 887, 2088, 918, 203, 887, 2088, 919, 202, 887, 2088, 919, 202, 887, 2088, 919, 202, 887, 2088, 919, 202, 886, 2090, 918, 202, 886, 2090, 918, 202, 886, 2090, 918, 202, 886, 2090, 919, 201, 886, 2090, 919, 201, 886, 2090, 920, 200, 886, 2090, 920, 200, 885, 2092, 919, 200, 885, 2092, 919, 199, 886, 2092, 919, 199, 886, 2092, 919, 199, 886, 2092, 919, 199, 886, 2092, 919, 198, 887, 2092, 919, 198, 886, 2094, 918, 198, 886, 2094, 918, 198, 886, 2094, 918, 197, 887, 2094, 918, 197, 887, 2094, 918, 197, 887, 2094, 919, 195, 888, 2094, 920, 194, 887, 2096, 919, 194, 887, 2096, 920, 193, 887, 2096, 920, 192, 888, 2096, 920, 192, 888, 2096, 920, 192, 888, 2096, 920, 191, 889, 2096, 920, 191, 888, 2098, 919, 191, 888, 2098, 919, 190, 889, 2098, 919, 190, 889, 2098, 919, 189, 890, 2098, 919, 189, 890, 2098, 919, 189, 890, 2098, 919, 188, 890, 2100, 918, 188, 890, 2100, 918, 188, 890, 2100, 918, 188, 890, 2100, 918, 187, 891, 2100, 918, 187, 891, 2100, 918, 187, 891, 2100, 918, 186, 892, 2100, 918, 186, 891, 2102, 917, 186, 891, 2102, 917, 185, 892, 2102, 917, 185, 892, 2102, 917, 184, 893, 2102, 917, 184, 893, 2102, 917, 184, 893, 2102, 917, 183, 893, 2104, 916, 183, 893, 2104, 916, 183, 893, 2104, 916, 183, 893, 2104, 916, 182, 894, 2104, 916, 182, 894, 2104, 916, 182, 894, 2104, 916, 182, 893, 2106, 915, 181, 894, 2106, 915, 181, 894, 2106, 915, 181, 894, 2106, 915, 181, 894, 2106, 915, 181, 894, 2106, 915, 181, 894, 2106, 915, 180, 894, 2108, 914, 180, 894, 2108, 914, 180, 894, 2108, 914, 180, 894, 2108, 914, 180, 894, 2108, 915, 179, 894, 2108, 915, 178, 895, 2108, 915, 178, 894, 2110, 915, 177, 894, 2110, 915, 177, 894, 2110, 916, 175, 895, 2110, 916, 175, 895, 2110, 916, 175, 895, 2110, 916, 175, 895, 2110, 916, 175, 894, 2112, 915, 175, 894, 2112, 916, 173, 895, 2112, 916, 173, 895, 2112, 916, 173, 895, 2112, 916, 173, 895, 2112, 916, 173, 895, 2112, 916, 173, 894, 2114, 915, 173, 894, 2114, 915, 173, 894, 2114, 915, 173, 894, 2114, 915, 173, 894, 2114, 916, 171, 894, 2116, 915, 171, 893, 2118, 914, 171, 893, 2118, 914, 171, 892, 2120, 914, 170, 891, 2122, 913, 170, 891, 2122, 914, 168, 891, 2124, 913, 168, 890, 2126, 912, 168, 889, 2128, 911, 168, 889, 2128, 911, 168, 888, 2130, 910, 168, 887, 2132, 909, 168, 887, 2132, 910, 167, 886, 2134, 909, 167, 885, 2136, 908, 167, 885, 2136, 908, 166, 885, 2138, 907, 166, 884, 2140, 906, 166, 884, 2140, 906, 166, 883, 2142, 905, 166, 882, 2144, 904, 166, 882, 2144, 904, 165, 882, 2146, 903, 165, 881, 2148, 902, 165, 881, 2148, 902, 165, 880, 2150, 902, 164, 879, 2152, 901, 164, 878, 2154, 900, 164, 878, 2154, 900, 164, 877, 2156, 900, 163, 876, 2158, 899, 163, 876, 2158, 899, 163, 875, 2160, 899, 162, 874, 2162, 898, 162, 874, 2162, 899, 161, 873, 2164, 898, 161, 872, 2166, 897, 161, 872, 2166, 897, 161, 871, 2168, 896, 161, 870, 2170, 895, 161, 870, 2170, 896, 160, 869, 2172, 895, 160, 868, 2174, 894, 160, 867, 2176, 893, 160, 867, 2176, 894, 159, 866, 2178, 893, 159, 865, 2180, 893, 158, 865, 2180, 893, 158, 864, 2182, 892, 158, 863, 2184, 891, 158, 863, 2184, 892, 157, 862, 2186, 891, 157, 861, 2188, 890, 157, 861, 2188, 891, 156, 860, 2190, 890, 156, 859, 2192, 889, 156, 859, 2192, 889, 156, 858, 2194, 888, 156, 857, 2196, 887, 156, 857, 2196, 887, 156, 856, 2198, 886, 155, 856, 2200, 885, 155, 855, 2202, 884, 155, 855, 2202, 884, 155, 854, 2204, 883, 154, 854, 2206, 882, 154, 854, 2206, 882, 154, 853, 2208, 881, 154, 852, 2210, 880, 154, 852, 2210, 880, 154, 851, 2212, 879, 153, 852, 2212, 879, 153, 851, 2214, 878, 153, 851, 2214, 878, 153, 851, 2214, 878, 153, 850, 2216, 876, 154, 850, 2216, 876, 153, 850, 2218, 874, 154, 850, 2218, 874, 154, 849, 2220, 873, 154, 849, 2220, 873, 153, 850, 2220, 874, 152, 849, 2222, 873, 152, 849, 2222, 873, 152, 848, 2224, 873, 151, 848, 2224, 873, 151, 847, 2226, 873, 149, 848, 2226, 873, 149, 847, 2228, 873, 148, 847, 2228, 873, 148, 847, 2228, 873, 148, 846, 2230, 872, 148, 846, 2230, 873, 147, 845, 2232, 872, 147, 845, 2232, 872, 147, 844, 2234, 871, 147, 844, 2234, 871, 147, 844, 2234, 871, 147, 843, 2236, 870, 147, 843, 2236, 870, 147, 842, 2238, 869, 147, 842, 2238, 869, 147, 841, 2240, 868, 147, 841, 2240, 868, 147, 841, 2240, 868, 147, 840, 2242, 867, 147, 840, 2242, 867, 147, 839, 2244, 866, 146, 840, 2244, 866, 146, 839, 2246, 865, 145, 840, 2246, 865, 145, 839, 2248, 864, 145, 839, 2248, 865, 144, 839, 2248, 865, 144, 838, 2250, 864, 144, 838, 2250, 864, 144, 837, 2252, 863, 144, 837, 2252, 863, 144, 836, 2254, 862, 145, 835, 2254, 862, 144, 836, 2254, 862, 144, 835, 2256, 860, 145, 835, 2256, 860, 145, 834, 2258, 858, 146, 834, 2258, 857, 146, 834, 2260, 857, 145, 834, 2260, 857, 145, 833, 2262, 857, 144, 833, 2262, 857, 143, 834, 2262, 857, 143, 833, 2264, 856, 142, 834, 2264, 856, 142, 833, 2266, 855, 142, 833, 2266, 855, 142, 832, 2268, 854, 141, 833, 2268, 854, 141, 833, 2268, 853, 142, 832, 2270, 852, 142, 832, 2270, 851, 143, 831, 2272, 849, 144, 831, 2272, 849, 144, 830, 2274, 849, 143, 830, 2274, 849, 143, 830, 2274, 849, 143, 829, 2276, 848, 143, 829, 2276, 848, 144, 827, 2278, 847, 144, 827, 2278, 847, 145, 825, 2280, 846, 145, 825, 2280, 846, 146, 823, 2282, 844, 148, 822, 2282, 844, 149, 821, 2282, 819, 7, 17, 152, 818, 2284, 814, 12, 15, 156, 815, 2284, 810, 17, 13, 159, 812, 2286, 806, 22, 9, 162, 811, 2286, 803, 197, 809, 2288, 799, 201, 808, 2288, 798, 204, 806, 2288, 794, 220, 793, 2290, 790, 224, 792, 2290, 788, 227, 790, 2292, 786, 229, 788, 2294, 784, 232, 785, 2296, 782, 236, 781, 2298, 779, 240, 778, 2300, 777, 243, 775, 2302, 769, 252, 772, 2304, 766, 256, 768, 2308, 763, 259, 765, 2310, 761, 262, 762, 2312, 759, 265, 759, 2314, 757, 268, 756, 2316, 756, 270, 753, 2318, 753, 274, 750, 2320, 750, 278, 746, 2324, 746, 282, 743, 2326, 743, 286, 740, 2328, 740, 291, 736, 2330, 736, 296, 733, 2332, 733, 300, 730, 2334, 730, 304, 726, 2338, 726, 308, 723, 2340, 723, 312, 720, 2342, 720, 316, 717, 2344, 717, 320, 714, 2346, 714, 324, 711, 2348, 711, 328, 707, 2352, 707, 332, 704, 2354, 704, 336, 701, 2356, 701, 340, 698, 2358, 698, 344, 695, 2360, 695, 349, 691, 2362, 691, 354, 688, 2364, 688, 358, 684, 2368, 684, 362, 681, 2370, 681, 366, 678, 2372, 678, 370, 675, 2374, 675, 374, 672, 2376, 672, 378, 669, 2378, 669, 382, 665, 2382, 665, 386, 660, 2388, 660, 390, 655, 2394, 655, 394, 650, 2400, 650, 398, 646, 2404, 646, 402, 641, 2410, 641, 407, 635, 2416, 635, 412, 631, 2420, 631, 416, 626, 2426, 626, 420, 621, 2432, 621, 424, 616, 2438, 616, 428, 612, 2442, 612, 432, 607, 2448, 607, 436, 602, 2454, 602, 440, 598, 2458, 598, 444, 593, 2464, 593, 448, 588, 2470, 588, 452, 583, 2476, 583, 456, 579, 2480, 579, 460, 574, 2486, 574, 465, 568, 2492, 568, 470, 563, 2498, 563, 474, 559, 2502, 559, 478, 554, 2508, 554, 482, 549, 2514, 549, 486, 545, 2518, 545, 490, 540, 2524, 540, 494, 535, 2530, 535, 498, 530, 2536, 530, 502, 526, 2540, 526, 506, 521, 2546, 521, 510, 516, 2552, 516, 514, 512, 2556, 512, 518, 507, 2562, 507, 523, 501, 2568, 501, 528, 496, 2574, 496, 532, 492, 2578, 492, 536, 488, 2582, 488, 540, 484, 2586, 484, 544, 481, 2588, 481, 548, 477, 2592, 477, 552, 473, 2596, 473, 556, 469, 2600, 469, 560, 465, 2604, 465, 564, 462, 2606, 462, 568, 458, 2610, 458, 572, 454, 2614, 454, 576, 450, 2618, 450, 581, 445, 2622, 445, 586, 442, 2624, 442, 590, 438, 2628, 438, 594, 434, 2632, 434, 598, 430, 2636, 430, 602, 426, 2640, 426, 606, 423, 2642, 411, 7, 5, 616, 413, 2646, 407, 631, 410, 2650, 393, 644, 407, 2654, 375, 662, 403, 2658, 356, 680, 400, 2662, 343, 693, 397, 2664, 341, 696, 393, 2668, 338, 699, 389, 2672, 334, 704, 384, 2676, 331, 708, 379, 2680, 327, 714, 374, 2682, 324, 748, 340, 2686, 320, 752, 336, 2690, 315, 758, 331, 2694, 310, 764, 326, 2698, 304, 772, 321, 2700, 294, 786, 314, 2704, 283, 799, 308, 2708, 272, 817, 297, 2712, 261, 830, 291, 2716, 250, 843, 286, 2718, 244, 857, 276, 2721, 238, 866, 270, 2724, 231, 875, 265, 2732, 220, 888, 255, 2735, 213, 902, 245, 2738, 207, 915, 235, 2742, 200, 928, 224, 2747, 193, 941, 214, 2752, 75, 24, 85, 950, 208, 2763, 62, 65, 44, 958, 202, 3903, 191, 3914, 179, 3990, 103, 4030, 62, 2440550] , dtype=np.uint32)
BOT_ARM_RLE = np.array( [8381426, 12, 4, 28, 4045, 19, 4, 28, 4041, 57, 4036, 60, 4033, 63, 4031, 65, 4028, 71, 4022, 77, 4017, 82, 4012, 86, 4008, 90, 4005, 93, 4001, 96, 3999, 99, 3996, 101, 3994, 103, 3992, 105, 3990, 107, 3988, 109, 3986, 112, 3983, 114, 3981, 116, 3979, 119, 3977, 120, 3975, 123, 3972, 125, 3971, 126, 3969, 128, 3967, 130, 3965, 132, 3964, 133, 3738, 26, 198, 135, 3733, 40, 2, 4, 2, 10, 2, 40, 16, 2, 2, 12, 95, 137, 3729, 61, 2, 40, 16, 2, 2, 12, 95, 138, 3725, 140, 92, 140, 3722, 142, 91, 141, 3720, 144, 90, 143, 3717, 146, 90, 144, 3715, 147, 89, 145, 3713, 153, 84, 147, 3711, 157, 81, 148, 3709, 161, 77, 151, 3706, 164, 75, 152, 3704, 168, 71, 154, 3702, 172, 68, 155, 3700, 175, 65, 157, 3698, 178, 63, 158, 3696, 181, 34, 17, 9, 160, 3694, 183, 29, 25, 5, 161, 3692, 186, 24, 31, 1, 163, 3690, 188, 20, 199, 3688, 191, 16, 202, 3686, 194, 12, 204, 3685, 196, 9, 207, 3683, 199, 6, 209, 3682, 200, 3, 211, 3681, 202, 1, 213, 3679, 417, 3679, 418, 3677, 419, 3677, 420, 3675, 421, 3675, 421, 3674, 423, 3673, 423, 3673, 423, 3672, 425, 3671, 425, 3671, 425, 3670, 426, 3670, 427, 3669, 427, 3669, 427, 3668, 428, 3668, 428, 3668, 428, 3668, 428, 3668, 428, 3667, 429, 3667, 429, 3667, 429, 3667, 429, 3666, 430, 3666, 430, 3666, 430, 3664, 432, 3662, 434, 3660, 437, 3658, 438, 3656, 441, 3654, 442, 3653, 444, 3651, 445, 3650, 447, 3648, 448, 3647, 450, 3645, 451, 3644, 453, 3642, 454, 3641, 456, 3640, 456, 3639, 458, 3637, 459, 3637, 460, 3635, 461, 3634, 463, 3632, 464, 3632, 465, 3630, 466, 3629, 468, 3628, 468, 3627, 469, 3627, 470, 3625, 471, 2343, 54, 1228, 471, 2334, 81, 1209, 473, 2324, 94, 1205, 473, 2320, 102, 1201, 473, 2316, 115, 1191, 474, 2312, 123, 1187, 475, 2308, 131, 1182, 475, 2305, 138, 1177, 476, 2302, 144, 1174, 476, 2299, 150, 1170, 477, 2296, 156, 1166, 478, 2294, 162, 1161, 479, 2291, 169, 1156, 480, 2288, 175, 1152, 481, 2286, 180, 1148, 483, 2282, 190, 1141, 483, 2279, 197, 1135, 485, 2277, 203, 1130, 486, 2275, 209, 1125, 487, 2273, 216, 1118, 489, 2270, 223, 1113, 490, 2268, 228, 1108, 492, 2266, 233, 1104, 493, 2264, 242, 1096, 494, 2263, 252, 1086, 495, 2261, 264, 1075, 496, 2260, 269, 1070, 497, 2258, 274, 1066, 498, 2255, 280, 1062, 499, 2253, 289, 1054, 500, 2251, 295, 1048, 502, 2249, 302, 1042, 503, 2247, 308, 1036, 505, 2245, 315, 1030, 506, 2240, 324, 1024, 508, 2236, 331, 1019, 510, 2233, 337, 1014, 512, 2230, 342, 928, 35, 47, 514, 2228, 347, 921, 43, 42, 515, 2223, 355, 915, 51, 35, 517, 2214, 366, 910, 58, 30, 518, 2205, 378, 905, 64, 25, 519, 2201, 385, 900, 70, 20, 520, 2198, 390, 896, 75, 16, 521, 2195, 395, 892, 80, 12, 522, 2179, 413, 888, 84, 9, 523, 2175, 418, 885, 89, 5, 524, 2161, 434, 881, 94, 1, 525, 2152, 444, 879, 621, 2148, 450, 875, 623, 2143, 457, 872, 625, 2124, 476, 870, 625, 2116, 487, 867, 626, 2107, 497, 865, 627, 2098, 507, 863, 628, 2038, 35, 21, 512, 861, 629, 2034, 43, 3, 527, 859, 630, 2031, 577, 857, 631, 2028, 582, 853, 632, 2026, 586, 851, 633, 2023, 590, 848, 635, 2020, 594, 846, 636, 2013, 602, 844, 637, 2009, 608, 840, 639, 2006, 612, 837, 642, 2002, 620, 830, 644, 1994, 637, 819, 646, 1986, 649, 813, 648, 1982, 657, 808, 649, 1979, 664, 802, 651, 1968, 678, 798, 652, 1960, 689, 794, 653, 1956, 695, 790, 655, 1942, 712, 786, 657, 1937, 719, 781, 659, 1914, 744, 778, 660, 1887, 773, 775, 661, 1878, 784, 771, 663, 1869, 794, 769, 664, 1865, 800, 765, 666, 1860, 806, 763, 668, 1855, 811, 761, 669, 1851, 816, 759, 670, 1842, 826, 757, 672, 1837, 831, 755, 673, 1834, 836, 752, 674, 1831, 840, 750, 675, 1828, 844, 748, 676, 1825, 848, 746, 677, 1822, 852, 744, 679, 1819, 855, 743, 679, 1817, 858, 741, 680, 1815, 861, 739, 682, 1812, 864, 737, 683, 1810, 867, 735, 684, 1808, 870, 733, 685, 1806, 872, 732, 686, 1805, 874, 731, 687, 1802, 877, 729, 688, 1801, 878, 728, 689, 1800, 880, 727, 690, 1797, 882, 726, 691, 1795, 885, 725, 691, 1793, 888, 723, 692, 1791, 892, 721, 692, 1789, 896, 718, 693, 1787, 900, 716, 694, 1784, 904, 714, 694, 1781, 909, 711, 695, 1778, 914, 709, 696, 1775, 918, 704, 699, 1773, 922, 702, 699, 1771, 926, 698, 702, 1768, 930, 696, 702, 1766, 933, 695, 703, 1760, 940, 693, 703, 1756, 945, 692, 704, 1750, 951, 691, 704, 1746, 957, 691, 703, 1741, 962, 689, 704, 1737, 968, 687, 705, 1731, 974, 686, 705, 1727, 981, 683, 705, 1722, 989, 679, 707, 1717, 995, 677, 707, 1713, 1001, 675, 708, 1703, 1012, 673, 708, 1694, 1023, 671, 708, 1685, 1034, 669, 709, 1680, 1040, 667, 709, 1676, 1046, 665, 709, 1666, 1058, 663, 710, 1661, 1064, 661, 710, 1657, 1070, 659, 711, 1646, 1082, 657, 712, 1637, 1092, 655, 712, 1628, 1103, 653, 713, 1618, 1114, 651, 713, 1614, 1120, 649, 714, 1608, 1126, 648, 714, 1599, 1137, 646, 715, 1594, 1142, 644, 716, 1589, 1148, 643, 716, 1585, 1158, 637, 717, 1579, 1163, 637, 717, 1570, 1174, 635, 717, 1566, 1178, 634, 719, 1562, 1181, 634, 719, 1559, 1184, 633, 720, 1557, 1187, 632, 720, 1552, 1194, 630, 721, 1547, 1199, 628, 722, 1542, 1205, 627, 722, 1538, 1210, 626, 723, 1534, 1214, 624, 724, 1531, 1218, 623, 725, 1523, 1227, 621, 725, 1519, 1233, 619, 726, 1513, 1240, 616, 727, 1509, 1245, 615, 727, 1505, 1251, 613, 728, 1500, 1256, 612, 728, 1497, 1260, 611, 728, 1494, 1264, 610, 729, 1485, 1273, 609, 729, 1481, 1278, 608, 730, 1476, 1284, 606, 730, 1467, 1294, 605, 730, 1463, 1299, 604, 731, 1459, 1303, 603, 731, 1456, 1307, 602, 731, 1449, 1315, 601, 732, 1440, 1324, 600, 732, 1436, 1329, 599, 733, 1421, 1344, 598, 733, 1403, 1363, 598, 732, 1399, 1368, 597, 733, 1395, 1371, 597, 733, 1384, 1383, 596, 733, 1375, 1393, 596, 733, 1366, 1401, 596, 733, 1357, 1411, 595, 734, 1352, 1416, 595, 733, 1349, 1420, 594, 733, 1346, 1423, 594, 734, 1337, 1432, 594, 733, 1333, 1437, 593, 733, 1328, 1442, 594, 733, 1323, 1447, 593, 733, 1319, 1451, 593, 734, 1314, 1456, 592, 734, 1309, 1461, 592, 734, 1305, 1466, 591, 735, 1300, 1470, 590, 736, 1296, 1474, 590, 736, 1291, 1480, 588, 738, 1281, 1489, 588, 738, 1272, 1498, 588, 738, 1268, 1503, 586, 739, 1263, 1508, 586, 740, 1253, 1517, 586, 740, 1244, 1526, 585, 741, 1240, 1531, 584, 742, 1226, 1544, 583, 743, 1222, 1548, 583, 744, 1218, 1551, 583, 744, 1215, 1554, 582, 746, 1206, 1562, 582, 746, 1202, 1567, 581, 746, 1190, 1579, 580, 748, 1185, 1583, 580, 748, 1180, 1588, 580, 748, 1171, 1597, 580, 749, 1166, 1601, 579, 750, 1152, 1615, 579, 750, 1148, 1619, 579, 751, 1143, 1623, 579, 752, 1124, 1641, 579, 752, 1115, 1650, 579, 753, 1096, 1668, 579, 753, 1092, 1672, 579, 754, 1088, 1675, 579, 754, 1085, 1678, 578, 756, 1067, 1696, 577, 756, 1063, 1701, 576, 756, 1059, 1706, 575, 757, 1054, 1711, 574, 757, 1051, 1715, 573, 757, 1048, 1719, 572, 758, 1045, 1721, 571, 760, 1042, 1724, 570, 760, 1039, 1728, 569, 761, 1034, 1732, 568, 763, 1030, 1736, 567, 763, 1027, 1740, 566, 764, 1023, 1744, 564, 766, 1019, 1747, 564, 767, 1015, 1751, 563, 767, 1007, 1760, 562, 768, 1002, 1764, 561, 770, 997, 1769, 560, 770, 993, 1774, 559, 771, 989, 1778, 558, 771, 978, 1789, 558, 772, 973, 1794, 556, 773, 968, 1800, 555, 774, 958, 1809, 555, 774, 954, 1814, 554, 775, 940, 1828, 552, 777, 935, 1833, 551, 778, 931, 1836, 551, 779, 919, 1848, 550, 780, 900, 1867, 549, 780, 896, 1871, 549, 781, 881, 1886, 548, 782, 876, 1890, 548, 782, 872, 1895, 547, 783, 867, 1900, 545, 785, 861, 1905, 545, 786, 856, 1910, 544, 786, 853, 1914, 543, 787, 849, 1918, 541, 789, 846, 1921, 540, 790, 843, 1924, 539, 791, 840, 1926, 539, 792, 838, 1928, 537, 794, 835, 1931, 536, 794, 834, 1932, 536, 795, 832, 1934, 535, 796, 830, 1935, 535, 797, 828, 1937, 533, 799, 826, 1939, 532, 800, 824, 1940, 532, 801, 822, 1942, 530, 803, 820, 1944, 529, 804, 817, 1946, 529, 805, 814, 1949, 527, 807, 811, 1952, 526, 808, 807, 1956, 525, 809, 804, 1958, 524, 810, 802, 1961, 523, 811, 799, 1964, 522, 812, 796, 1966, 522, 813, 793, 1969, 520, 815, 789, 1972, 520, 816, 785, 1977, 518, 817, 782, 1979, 518, 817, 779, 1982, 517, 819, 775, 1986, 516, 820, 772, 1989, 515, 820, 770, 1991, 514, 822, 767, 1994, 513, 822, 764, 1998, 512, 823, 761, 2001, 510, 825, 756, 2005, 510, 825, 752, 2010, 509, 826, 748, 2014, 507, 828, 744, 2017, 507, 829, 741, 2020, 506, 830, 735, 2026, 505, 831, 730, 2031, 503, 832, 727, 2034, 503, 833, 723, 2038, 502, 834, 719, 2042, 501, 835, 715, 2045, 501, 836, 711, 2049, 499, 838, 708, 2052, 498, 839, 704, 2056, 497, 840, 700, 2059, 497, 841, 697, 2062, 495, 843, 693, 2066, 494, 844, 689, 2069, 493, 846, 686, 2072, 492, 847, 683, 2074, 491, 848, 681, 2077, 490, 849, 678, 2079, 489, 851, 675, 2082, 488, 852, 672, 2084, 487, 854, 669, 2087, 486, 855, 666, 2089, 486, 856, 663, 2092, 484, 857, 661, 2094, 484, 858, 658, 2097, 483, 859, 653, 2102, 481, 861, 649, 2105, 481, 862, 645, 2109, 480, 863, 642, 2112, 479, 864, 639, 2114, 478, 866, 636, 2117, 477, 867, 632, 2120, 477, 868, 629, 2123, 476, 869, 625, 2126, 476, 870, 621, 2130, 475, 871, 618, 2132, 475, 872, 614, 2135, 475, 873, 610, 2139, 473, 875, 607, 2141, 472, 877, 603, 2144, 470, 880, 600, 2147, 468, 882, 596, 2151, 465, 884, 593, 2154, 464, 886, 588, 2159, 462, 888, 582, 1, 1, 2163, 460, 889, 578, 2169, 459, 891, 571, 2176, 457, 893, 567, 2180, 455, 895, 564, 2183, 453, 896, 562, 2185, 452, 898, 558, 2189, 450, 900, 555, 2192, 448, 901, 553, 2194, 447, 903, 550, 2197, 445, 904, 546, 2201, 444, 906, 542, 2205, 442, 907, 539, 2208, 441, 909, 536, 2211, 439, 910, 531, 2216, 439, 911, 526, 2221, 437, 912, 521, 2226, 436, 914, 516, 2231, 435, 915, 511, 2235, 434, 917, 506, 2240, 433, 917, 503, 2244, 431, 919, 499, 2247, 431, 921, 494, 2251, 429, 923, 489, 2256, 428, 924, 484, 2260, 428, 925, 1, 6, 468, 4, 1, 2264, 426, 933, 466, 2271, 426, 935, 460, 2276, 425, 935, 457, 2279, 424, 936, 453, 2284, 422, 937, 449, 2288, 422, 937, 445, 2292, 421, 938, 444, 2294, 419, 940, 443, 2294, 419, 941, 441, 2295, 418, 943, 438, 2298, 417, 945, 433, 2301, 416, 947, 429, 2305, 414, 950, 424, 2308, 412, 953, 419, 2312, 411, 956, 413, 2317, 408, 960, 408, 2320, 407, 962, 404, 2323, 406, 965, 400, 2326, 404, 967, 396, 2329, 403, 969, 392, 2332, 402, 971, 389, 2334, 401, 973, 383, 2340, 399, 975, 378, 2344, 398, 977, 374, 2347, 397, 979, 370, 2350, 396, 981, 366, 2354, 395, 982, 361, 2358, 394, 984, 351, 2367, 393, 985, 347, 2372, 392, 987, 342, 2375, 391, 989, 338, 2378, 389, 993, 324, 3, 6, 2382, 388, 995, 321, 4, 2, 2386, 386, 998, 318, 2394, 386, 1000, 314, 2397, 385, 1001, 311, 2399, 385, 1004, 305, 2402, 385, 1007, 299, 2405, 385, 1009, 293, 2410, 385, 1010, 287, 2414, 384, 1013, 282, 2417, 384, 1014, 279, 2419, 383, 1017, 275, 2421, 383, 1018, 272, 2423, 383, 1019, 269, 2425, 382, 1021, 265, 2428, 382, 1022, 261, 2431, 382, 1023, 258, 2433, 381, 1026, 253, 2436, 381, 1028, 248, 2439, 381, 1030, 242, 2443, 381, 1031, 236, 2448, 380, 1034, 230, 2452, 380, 1035, 224, 2457, 380, 1036, 221, 2459, 380, 1038, 216, 2462, 380, 1040, 211, 2466, 379, 1042, 205, 2470, 379, 1044, 198, 2475, 379, 1046, 193, 2478, 379, 1047, 189, 2481, 378, 1050, 184, 2484, 378, 1051, 179, 2488, 378, 1052, 172, 2494, 378, 1053, 167, 2498, 378, 1054, 157, 2507, 378, 1055, 155, 2508, 378, 1056, 153, 2509, 378, 1057, 151, 2510, 378, 1058, 149, 2511, 378, 1059, 147, 2512, 378, 1060, 145, 2513, 378, 1060, 144, 2514, 378, 1061, 141, 2516, 378, 1062, 139, 2517, 378, 1062, 137, 2519, 378, 1063, 134, 2521, 378, 1063, 133, 2522, 378, 1064, 131, 2523, 378, 1064, 129, 2525, 378, 1065, 127, 2526, 378, 1065, 126, 2527, 378, 1065, 125, 2528, 378, 1066, 122, 2530, 378, 1066, 121, 2531, 378, 1066, 119, 2533, 378, 1067, 116, 2535, 378, 1067, 114, 2537, 378, 1067, 111, 2540, 378, 1067, 108, 2543, 378, 1068, 103, 2547, 378, 1068, 103, 2547, 378, 1068, 102, 2548, 378, 1068, 102, 2548, 378, 1068, 101, 2549, 378, 1079, 6, 12, 2, 2, 4, 30, 2, 12, 2, 2567, 379, 1078, 6, 12, 2, 2, 4, 30, 2, 12, 2, 2567, 379, 3717, 379, 3717, 379, 3717, 380, 3716, 380, 3716, 380, 3716, 380, 3716, 380, 3716, 381, 3715, 381, 3715, 381, 3715, 381, 3715, 382, 3714, 382, 3714, 382, 3714, 383, 3713, 383, 3713, 383, 3713, 384, 3712, 384, 3712, 385, 3711, 385, 3711, 385, 3711, 386, 3710, 386, 3710, 386, 3710, 387, 3709, 387, 3709, 388, 3708, 388, 3708, 389, 3707, 389, 3707, 389, 3707, 390, 3706, 390, 3706, 390, 3706, 390, 3706, 391, 3705, 391, 3705, 391, 3705, 392, 3704, 392, 3704, 392, 3704, 393, 3703, 393, 3703, 394, 3703, 393, 3703, 394, 3703, 393, 3703, 394, 3703, 394, 3702, 394, 3703, 394, 3702, 395, 3701, 396, 3701, 396, 3700, 397, 3699, 398, 3699, 398, 3698, 399, 3697, 400, 3696, 401, 3696, 401, 3695, 402, 3694, 403, 3693, 404, 3692, 405, 3691, 406, 3690, 408, 3688, 409, 3687, 411, 3685, 413, 3683, 415, 3681, 416, 3680, 418, 3678, 420, 3676, 421, 3675, 423, 3673, 424, 3672, 426, 3670, 428, 3668, 430, 3666, 433, 3663, 436, 3660, 438, 3657, 441, 3655, 443, 3653, 444, 3652, 445, 3650, 448, 3648, 449, 3647, 451, 3646, 451, 3645, 452, 3644, 454, 3642, 455, 3641, 457, 3639, 459, 3637, 461, 3635, 463, 3633, 464, 3633, 464, 3632, 465, 3631, 466, 3630, 468, 3628, 469, 3627, 470, 3626, 471, 3625, 472, 3624, 473, 3623, 475, 3621, 476, 3620, 478, 3618, 479, 3617, 480, 3617, 481, 3615, 482, 3614, 484, 3612, 486, 3611, 487, 3609, 488, 3608, 490, 3606, 492, 3604, 493, 3603, 495, 3601, 496, 3600, 498, 3598, 500, 3596, 502, 3594, 504, 3592, 506, 3590, 508, 3588, 510, 3586, 512, 3584, 514, 3582, 516, 3580, 518, 3578, 520, 3576, 522, 3574, 524, 3572, 526, 3570, 528, 3568, 530, 3566, 532, 3564, 534, 3562, 536, 3560, 538, 3558, 540, 3556, 542, 3554, 544, 3552, 546, 3550, 548, 3548, 550, 3546, 552, 3544, 554, 3542, 556, 3540, 558, 3538, 561, 3535, 564, 3532, 566, 3530, 568, 3528, 570, 3526, 572, 3524, 574, 3522, 576, 3520, 578, 3518, 580, 3516, 582, 3514, 584, 3512, 586, 3510, 588, 3508, 590, 3506, 593, 3503, 596, 3500, 598, 3498, 599, 3497, 601, 3495, 603, 3493, 605, 3491, 608, 3488, 611, 3485, 613, 3484, 615, 3481, 617, 3479, 619, 3477, 619, 3477, 619, 3477, 619, 3477, 618, 3478, 618, 3478, 618, 3478, 618, 3478, 617, 3479, 617, 3479, 617, 3479, 617, 3479, 617, 3479, 617, 3479, 617, 3480, 616, 3480, 616, 3480, 616, 3481, 615, 3481, 615, 3481, 615, 3482, 614, 3482, 614, 3482, 614, 3482, 614, 3483, 613, 3483, 613, 3483, 613, 3483, 613, 3483, 613, 3483, 613, 3483, 613, 3483, 613, 3483, 613, 3483, 613, 3483, 613, 3483, 613, 3483, 613, 3483, 613, 3483, 613, 3483, 613, 3483, 613, 3482, 614, 3482, 614, 3482, 615, 3481, 615, 3480, 616, 3480, 616, 3480, 616, 3479, 617, 3479, 617, 3479, 617, 3478, 619, 3477, 619, 3476, 620, 3476, 620, 3475, 622, 3474, 622, 3473, 623, 3472, 624, 3472, 624, 3472, 625, 3470, 626, 3470, 626, 3470, 626, 3470, 626, 3469, 627, 3469, 627, 3469, 627, 3468, 628, 3468, 628, 3468, 628, 3467, 629, 3467, 629, 3466, 630, 3466, 630, 3465, 631, 3465, 631, 3464, 632, 3463, 633, 3463, 633, 3462, 634, 3461, 635, 3460, 636, 3459, 637, 3458, 638, 3457, 639, 3456, 640, 3455, 642, 3453, 643, 3453, 643, 3453, 643, 3453, 643, 3453, 643, 3453, 643, 3452, 644, 3452, 644, 3452, 645, 3451, 645, 3451, 645, 3451, 645, 3451, 645, 3451, 645, 3451, 645, 3450, 646, 3450, 646, 3450, 647, 3449, 647, 3448, 648, 3448, 648, 3448, 649, 3446, 650, 3446, 650, 3446, 650, 3445, 651, 3445, 652, 3443, 653, 3443, 653, 3443, 653, 3442, 654, 3442, 654, 3442, 654, 3441, 655, 3441, 655, 3441, 656, 3439, 657, 3439, 657, 3439, 657, 3439, 658, 3437, 659, 3437, 659, 3437, 659, 3437, 659, 3437, 659, 3437, 659, 3437, 659, 3437, 659, 3437, 659, 3436, 660, 3436, 660, 3436, 660, 3436, 660, 3435, 661, 3435, 661, 3435, 659, 3437, 657, 3438, 656, 3440, 655, 3441, 653, 3443, 652, 3443, 652, 3444, 651, 3445, 650, 3446, 649, 3447, 647, 3448, 647, 3449, 646, 3450, 645, 3451, 644, 3451, 644, 3452, 643, 3453, 642, 3454, 641, 3455, 640, 3456, 639, 3457, 639, 3457, 638, 3458, 637, 3459, 636, 3460, 635, 3461, 634, 3462, 633, 3463, 633, 3463, 632, 3464, 631, 3465, 631, 3465, 630, 3466, 629, 3467, 628, 3468, 628, 3469, 626, 3470, 625, 3471, 625, 3472, 623, 3473, 623, 3473, 622, 3475, 621, 3475, 620, 3476, 620, 3476, 619, 3478, 618, 3478, 617, 3479, 617, 3479, 616, 3480, 616, 3480, 615, 3481, 615, 3481, 615, 3481, 614, 3482, 614, 3482, 614, 3482, 613, 3483, 613, 3483, 612, 3484, 612, 3484, 611, 3485, 611, 3485, 610, 3486, 610, 3486, 609, 3487, 609, 3487, 608, 3488, 608, 3488, 608, 3488, 607, 3489, 607, 3489, 607, 3489, 606, 3490, 606, 3490, 606, 3490, 606, 3490, 605, 3491, 605, 3491, 605, 3491, 605, 3491, 605, 3491, 604, 3492, 604, 3492, 604, 3492, 604, 3492, 603, 3493, 603, 3493, 600, 3496, 600, 3496, 600, 3495, 601, 3495, 601, 3495, 601, 3495, 601, 3494, 602, 3494, 602, 3494, 602, 3493, 602, 3494, 602, 3494, 602, 3493, 603, 3493, 602, 3494, 602, 3494, 602, 3494, 602, 3494, 602, 3494, 601, 3494, 602, 3494, 602, 3494, 602, 3494, 601, 3494, 602, 3494, 602, 3494, 602, 3493, 603, 3493, 603, 3493, 603, 3492, 603, 3493, 603, 3492, 604, 3492, 603, 3492, 604, 3492, 604, 3491, 604, 3491, 605, 3491, 605, 3490, 606, 3490, 605, 3490, 606, 3490, 606, 3489, 607, 3488, 608, 3488, 607, 3488, 608, 3487, 609, 3487, 609, 3486, 609, 3487, 609, 3486, 610, 3486, 610, 3485, 611, 3485, 611, 3484, 612, 3484, 612, 3483, 613, 3483, 613, 3482, 614, 3481, 615, 3481, 615, 3480, 616, 3479, 617, 3478, 618, 3478, 618, 3477, 619, 3476, 619, 3477, 619, 3476, 620, 3475, 621, 3474, 622, 3473, 623, 3472, 624, 3471, 625, 3470, 626, 3469, 627, 3468, 628, 3467, 629, 3466, 630, 3465, 631, 3465, 631, 3464, 632, 3463, 633, 3462, 634, 3461, 635, 3460, 636, 3459, 637, 3459, 637, 3458, 638, 3457, 639, 3456, 640, 3455, 641, 3454, 642, 3453, 643, 3452, 644, 3451, 645, 3450, 646, 3449, 647, 3447, 649, 3446, 650, 3446, 650, 3445, 651, 3444, 652, 3443, 653, 3442, 654, 3441, 655, 3441, 655, 3440, 656, 3439, 657, 3438, 658, 3437, 659, 3436, 660, 3435, 661, 3434, 662, 3433, 663, 3432, 664, 3431, 665, 3429, 667, 3428, 668, 3427, 670, 3425, 671, 3424, 672, 3423, 673, 3421, 675, 3420, 676, 3418, 678, 3417, 679, 3416, 680, 3414, 683, 3412, 684, 3411, 685, 3410, 686, 3409, 687, 3408, 688, 3406, 690, 3405, 691, 3403, 693, 3402, 695, 3400, 696, 3398, 698, 3397, 699, 3395, 702, 3393, 703, 3392, 704, 3391, 706, 3389, 707, 3388, 708, 3387, 710, 3385, 711, 3384, 713, 3381, 715, 3380, 716, 3379, 717, 3378, 718, 3377, 720, 3375, 721, 3373, 723, 3372, 724, 3370, 727, 3367, 729, 3365, 731, 3363, 733, 3361, 735, 3360, 736, 3359, 737, 3357, 739, 3356, 740, 3354, 742, 3352, 744, 3351, 745, 3349, 747, 3348, 748, 3346, 750, 3344, 754, 3340, 756, 3338, 758, 3337, 759, 3336, 760, 3335, 761, 3333, 763, 3332, 764, 3330, 766, 3328, 768, 3326, 770, 3325, 771, 3323, 773, 3321, 775, 3319, 777, 3317, 779, 3315, 781, 3313, 783, 3311, 787, 3307, 789, 3305, 794, 3300, 796, 3298, 798, 3296, 800, 3294, 803, 3291, 805, 3289, 807, 3287, 809, 3285, 812, 3282, 814, 3280, 816, 3277, 820, 3273, 823, 3270, 826, 3267, 830, 3263, 833, 3261, 836, 3257, 839, 3254, 843, 3249, 847, 3245, 852, 3240, 857, 3234, 862, 3230, 867, 3224, 873, 3219, 878, 3200, 896, 3196, 901, 3191, 905, 3186, 911, 3181, 915, 3172, 925, 3163, 933, 3157, 940, 3152, 944, 3147, 950, 3142, 955, 3136, 960, 3132, 965, 3127, 970, 3122, 975, 3102, 995, 3097, 1000, 3092, 1005, 3077, 1020, 3072, 1025, 3053, 1044, 3038, 1059, 3033, 1065, 3022, 1075, 3017, 1080, 3013, 1084, 3009, 1088, 3005, 1092, 3000, 1098, 2996, 1101, 2993, 1105, 2989, 1108, 2985, 1112, 2981, 1117, 2975, 1122, 81, 2, 2887, 1128, 79, 2, 2886, 1131, 71, 8, 18, 2, 6, 2, 2856, 1135, 69, 8, 18, 2, 6, 2, 2855, 1139, 60, 72, 2823, 1144, 55, 74, 2821, 1148, 52, 75, 2819, 1153, 47, 77, 2817, 1158, 42, 81, 2815, 1161, 37, 83, 2815, 1164, 31, 86, 2814, 1168, 25, 89, 2814, 1172, 17, 95, 2812, 1284, 2811, 1283, 2813, 1283, 2812, 1284, 2812, 1284, 2811, 1285, 2811, 1284, 2811, 1283, 2812, 1282, 2814, 1280, 2815, 1279, 2816, 1278, 2818, 1276, 2819, 1275, 2821, 1273, 2823, 1271, 2824, 1270, 2826, 1268, 2827, 1268, 2828, 1264, 2831, 1265, 2831, 1261, 2834, 1262, 2833, 1262, 2834, 1261, 2834, 1261, 2834, 1261, 2834, 1261, 2834, 1261, 2834, 1260, 2835, 1260, 2836, 1259, 2836, 1258, 2837, 1258, 2837, 1257, 2838, 1257, 2838, 1257, 2838, 1257, 2839, 1256, 2839, 1256, 2839, 1256, 2839, 1256, 2840, 1255, 2841, 1254, 2841, 1254, 2842, 1254, 2842, 1253, 2843, 1251, 2844, 1251, 2845, 1250, 2846, 1248, 2847, 1248, 2848, 1246, 2850, 1245, 2850, 1245, 2851, 1244, 2851, 1244, 2852, 1243, 2853, 1242, 2854, 1241, 2855, 1240, 2855, 1240, 2856, 1239, 2857, 1239, 2857, 1238, 2857, 1238, 2858, 1237, 2859, 1236, 2859, 1236, 2860, 1235, 2861, 1234, 2861, 1234, 2862, 1233, 2862, 1233, 2863, 1232, 2864, 1231, 2864, 1232, 2864, 1231, 2865, 1230, 2866, 1230, 2866, 1229, 2867, 1228, 2868, 1227, 2869, 1225, 2871, 1225, 2871, 1223, 2873, 1223, 2872, 1223, 2873, 1222, 2874, 1222, 2874, 1221, 2875, 1220, 2876, 1219, 2877, 1218, 2878, 1217, 2879, 1216, 2880, 1216, 2880, 1215, 2881, 1214, 2882, 1213, 2883, 1212, 2884, 1211, 2885, 1195, 2901, 1191, 2904, 1189, 2907, 1186, 2910, 1184, 2912, 1181, 2914, 1179, 2917, 1177, 2919, 1174, 2922, 1171, 2925, 1169, 2926, 1168, 2928, 1166, 2930, 1165, 2931, 1163, 2932, 1163, 2933, 1160, 2936, 1157, 2939, 1155, 2941, 1153, 2943, 1151, 2945, 1150, 2946, 1147, 2949, 1144, 2952, 1141, 2955, 1139, 2957, 1137, 2959, 1134, 2962, 1130, 2966, 1125, 2971, 1121, 2975, 1118, 2978, 1115, 2981, 1113, 2983, 1110, 2986, 1107, 2989, 1105, 2991, 1102, 2994, 1099, 2997, 1097, 2999, 1095, 3001, 1093, 3003, 1091, 3005, 1089, 3007, 1086, 3010, 1074, 2, 2, 2, 2, 3014, 1074, 2, 2, 2, 1, 3015, 1070, 3026, 1070, 3026, 1070, 3026, 1069, 3027, 1066, 3030, 1064, 3032, 1061, 3035, 1058, 3038, 1056, 3040, 1054, 3042, 1052, 3044, 1051, 3045, 1049, 3047, 1046, 3050, 1043, 3053, 1041, 3055, 1039, 3057, 1037, 3059, 1036, 3060, 1034, 3062, 1033, 3063, 1032, 3064, 1031, 3065, 1030, 3066, 1028, 3068, 1026, 3070, 1024, 3072, 1022, 3075, 1019, 3077, 1017, 3079, 1015, 993, 4, 2084, 1013, 993, 8, 2082, 1005, 999, 12, 2080, 1005, 997, 17, 2077, 1005, 995, 22, 57, 8, 2009, 1005, 992, 29, 44, 18, 2009, 1003, 990, 96, 2007, 1001, 990, 99, 2006, 999, 989, 104, 2004, 997, 988, 109, 2002, 996, 987, 113, 2000, 994, 987, 118, 1997, 993, 986, 123, 1994, 991, 986, 129, 1990, 989, 987, 135, 1986, 987, 986, 141, 1982, 985, 987, 150, 1974, 984, 986, 189, 1937, 983, 985, 196, 1932, 981, 985, 202, 1929, 979, 985, 221, 1911, 977, 983, 236, 1901, 975, 980, 243, 1898, 974, 979, 249, 1894, 973, 978, 268, 1878, 971, 977, 279, 1869, 970, 975, 291, 1860, 969, 974, 302, 1852, 967, 973, 309, 1847, 966, 973, 314, 1843, 964, 973, 334, 1826, 962, 972, 345, 1818, 959, 973, 355, 1810, 957, 972, 362, 1806, 955, 972, 367, 1804, 952, 972, 393, 1780, 950, 971, 398, 1779, 947, 971, 402, 1778, 944, 971, 407, 1776, 941, 970, 428, 1758, 939, 969, 443, 1747, 936, 969, 448, 1744, 934, 968, 468, 1727, 933, 967, 474, 1723, 931, 967, 479, 1720, 929, 966, 508, 1694, 928, 965, 569, 1635, 926, 965, 574, 1632, 924, 964, 603, 1606, 922, 964, 668, 1543, 921, 963, 696, 1517, 919, 962, 730, 1485, 918, 962, 735, 1482, 916, 962, 745, 1475, 913, 962, 764, 1458, 911, 962, 792, 1432, 909, 962, 802, 1424, 908, 961, 857, 1371, 906, 961, 863, 1367, 904, 961, 868, 1364, 903, 960, 872, 1362, 901, 960, 874, 1362, 900, 958, 878, 1361, 898, 958, 881, 1360, 896, 958, 884, 1359, 895, 957, 888, 1357, 892, 958, 892, 1355, 891, 957, 897, 1352, 890, 956, 903, 1348, 889, 954, 909, 1345, 888, 953, 919, 1336, 888, 952, 937, 1320, 886, 952, 941, 1318, 885, 951, 945, 1315, 884, 951, 950, 1312, 882, 951, 960, 1303, 882, 949, 971, 1295, 880, 949, 981, 1286, 880, 948, 991, 1278, 878, 947, 1002, 1269, 878, 946, 1007, 1265, 877, 946, 1012, 1262, 876, 945, 1022, 1253, 876, 944, 1028, 1248, 875, 944, 1033, 1245, 874, 943, 1052, 300, 28, 899, 874, 941, 1063, 154, 165, 899, 873, 941, 1384, 898, 873, 940, 1386, 898, 871, 940, 1388, 898, 870, 939, 1390, 898, 868, 939, 1392, 898, 867, 938, 1394, 897, 866, 938, 1396, 897, 865, 937, 1398, 897, 863, 937, 1400, 896, 863, 937, 1401, 896, 861, 937, 1404, 894, 861, 935, 1407, 894, 860, 934, 1410, 892, 859, 934, 1413, 891, 858, 932, 1417, 889, 858, 931, 1421, 886, 857, 931, 1425, 884, 856, 930, 1430, 880, 856, 929, 1439, 872, 856, 928, 1454, 859, 854, 929, 1458, 855, 854, 928, 1468, 846, 854, 927, 1478, 837, 854, 926, 1484, 833, 852, 926, 1489, 829, 852, 924, 1499, 821, 852, 923, 1510, 811, 852, 922, 1520, 44, 6, 752, 851, 922, 1575, 748, 851, 921, 1583, 741, 851, 920, 1587, 738, 851, 919, 1591, 735, 851, 918, 1596, 731, 851, 917, 1602, 726, 851, 916, 1607, 722, 851, 916, 1616, 713, 851, 914, 1627, 704, 851, 913, 1637, 695, 851, 912, 1642, 691, 851, 911, 1647, 687, 851, 910, 1653, 681, 852, 909, 1658, 677, 852, 908, 1664, 672, 852, 907, 1669, 668, 852, 906, 1674, 663, 854, 904, 1694, 644, 854, 903, 1699, 640, 854, 902, 1709, 630, 855, 901, 1718, 622, 856, 899, 1742, 599, 856, 898, 1747, 594, 857, 897, 1757, 585, 858, 895, 1763, 579, 859, 894, 1768, 575, 859, 894, 1777, 565, 861, 892, 1783, 560, 861, 891, 1788, 555, 863, 889, 1798, 545, 864, 888, 1808, 536, 865, 885, 1819, 526, 866, 884, 1825, 520, 868, 882, 1830, 515, 870, 880, 1835, 510, 871, 879, 1840, 505, 873, 877, 1847, 498, 875, 875, 1850, 495, 877, 873, 1852, 493, 879, 871, 1855, 490, 881, 869, 1857, 488, 883, 868, 1858, 485, 886, 866, 1861, 482, 888, 864, 1863, 479, 891, 862, 1866, 477, 892, 861, 1868, 475, 894, 857, 1872, 473, 895, 855, 1876, 468, 899, 852, 1880, 458, 1, 6, 901, 849, 1885, 453, 911, 846, 1895, 443, 915, 842, 1905, 433, 919, 839, 1912, 329, 8, 88, 924, 834, 1916, 322, 14, 84, 930, 53, 12, 763, 1921, 314, 21, 81, 935, 45, 16, 762, 1926, 306, 27, 77, 999, 760, 1945, 279, 39, 72, 1001, 759, 1955, 260, 52, 67, 1003, 758, 1974, 239, 59, 60, 1007, 756, 1984, 226, 81, 39, 1010, 755, 2021, 182, 98, 26, 1014, 754, 3343, 752, 3344, 751, 3346, 750, 3346, 749, 3348, 747, 3349, 746, 3351, 744, 3353, 742, 3354, 741, 3356, 739, 3358, 737, 3360, 735, 3361, 734, 3363, 732, 3365, 730, 3366, 729, 3368, 727, 3370, 725, 3372, 723, 3373, 722, 3375, 720, 3377, 719, 3377, 717, 3380, 715, 3382, 713, 3384, 711, 3386, 709, 3388, 707, 3390, 705, 3391, 704, 3393, 702, 3394, 701, 3396, 699, 3397, 697, 3400, 695, 3402, 693, 3403, 692, 3405, 690, 3407, 688, 3409, 686, 3411, 684, 3413, 682, 3415, 680, 3417, 678, 3419, 676, 3421, 673, 3424, 670, 3428, 667, 3430, 664, 3432, 662, 3435, 659, 3437, 657, 3440, 655, 3441, 653, 3443, 652, 3445, 650, 3446, 647, 3450, 644, 3452, 642, 3455, 640, 3456, 638, 3459, 635, 3462, 632, 3464, 630, 3467, 628, 3469, 625, 3471, 623, 3474, 620, 3476, 618, 3479, 616, 3481, 613, 3483, 612, 3485, 610, 3485, 598, 3497, 599, 3497, 601, 3494, 602, 3494, 602, 3493, 603, 3488, 2, 3, 601, 3490, 2, 2, 602, 3488, 604, 3492, 604, 3492, 602, 3494, 602, 3494, 602, 3494, 601, 3495, 601, 3495, 601, 3494, 602, 3493, 603, 3492, 604, 3491, 605, 3491, 605, 3490, 606, 3489, 606, 3490, 606, 3489, 607, 3489, 607, 3488, 607, 3489, 607, 3488, 608, 3488, 607, 3488, 608, 3488, 608, 3487, 608, 3487, 609, 3486, 610, 3486, 610, 3485, 611, 3484, 612, 3484, 612, 3483, 612, 3484, 612, 3483, 613, 3483, 613, 3482, 613, 3483, 613, 3483, 613, 3482, 614, 3482, 614, 3482, 613, 3482, 614, 3482, 614, 3482, 614, 3482, 613, 3482, 614, 3482, 614, 3482, 613, 3483, 613, 3483, 613, 3482, 614, 3481, 615, 3480, 616, 3479, 617, 3478, 618, 3477, 619, 3477, 619, 3476, 619, 3476, 620, 3476, 620, 3475, 621, 3475, 620, 3475, 621, 3475, 621, 3474, 622, 3474, 622, 3474, 621, 3474, 622, 3474, 622, 3474, 622, 3473, 623, 3473, 623, 3473, 623, 3473, 623, 3472, 624, 3472, 623, 3473, 623, 3473, 623, 3473, 623, 3473, 622, 3474, 622, 3474, 622, 3474, 622, 3474, 622, 3474, 622, 3474, 622, 3474, 622, 3474, 622, 3473, 623, 3473, 623, 3473, 623, 3472, 624, 3472, 624, 3472, 623, 3473, 623, 3472, 624, 3472, 624, 3472, 624, 3472, 624, 3472, 624, 3471, 625, 3471, 625, 3471, 624, 3472, 624, 3471, 625, 3471, 625, 3471, 624, 3472, 624, 3472, 624, 3472, 623, 3473, 623, 3473, 623, 3473, 623, 3473, 623, 3473, 623, 3473, 623, 3473, 623, 3473, 623, 3473, 623, 3473, 622, 3474, 622, 3475, 621, 3475, 621, 3475, 621, 3475, 621, 3476, 620, 3476, 620, 3476, 620, 3477, 618, 3478, 618, 3478, 618, 3479, 617, 3479, 617, 3480, 616, 3480, 616, 3481, 615, 3481, 615, 3482, 613, 3483, 613, 3484, 612, 3484, 612, 3485, 611, 3485, 611, 3486, 610, 3486, 610, 3487, 609, 3488, 607, 3489, 607, 3490, 606, 3491, 605, 3492, 604, 3493, 603, 3494, 602, 3495, 601, 3496, 600, 3497, 598, 3499, 597, 3500, 596, 3502, 594, 3503, 592, 3506, 49, 2, 14, 4, 36, 3, 482, 3508, 47, 2, 14, 4, 36, 4, 481, 3510, 37, 69, 480, 3513, 31, 74, 478, 3516, 25, 78, 476, 3521, 17, 84, 474, 3624, 472, 3626, 470, 3629, 466, 3633, 463, 3637, 459, 3642, 454, 3646, 450, 3650, 445, 3655, 441, 3664, 432, 3669, 427, 3673, 423, 3678, 418, 3682, 414, 3687, 409, 3691, 404, 3693, 403, 3693, 403, 3693, 403, 3693, 402, 3694, 402, 3694, 402, 3694, 402, 3694, 402, 3694, 401, 3695, 401, 3695, 400, 3696, 399, 3697, 398, 3698, 397, 3699, 396, 3701, 394, 3702, 393, 3703, 393, 3703, 392, 3704, 392, 3704, 392, 3704, 391, 3705, 391, 3705, 390, 3707, 389, 3707, 388, 3708, 387, 3709, 386, 3711, 384, 3712, 384, 3712, 383, 3714, 382, 3714, 382, 3714, 381, 3716, 380, 3716, 380, 3717, 378, 3718, 377, 3719, 376, 3721, 374, 3722, 373, 3723, 373, 3724, 371, 3725, 370, 3727, 369, 3727, 368, 3729, 366, 3730, 366, 3731, 364, 3732, 364, 3733, 362, 3734, 361, 3735, 361, 3736, 359, 3737, 358, 3738, 358, 3739, 356, 3740, 355, 3741, 354, 3743, 352, 3744, 352, 3745, 350, 3746, 349, 3748, 348, 3748, 347, 3750, 345, 3752, 343, 3753, 342, 3755, 340, 3756, 338, 3758, 337, 3759, 336, 3762, 333, 3763, 331, 3769, 326, 3771, 323, 3774, 321, 3776, 318, 3780, 314, 3783, 120, 2, 189, 3787, 118, 4, 185, 3790, 117, 6, 181, 3793, 114, 11, 177, 3796, 112, 14, 171, 3800, 111, 18, 165, 3804, 109, 23, 156, 3810, 107, 27, 77, 13, 10, 10, 40, 3814, 105, 121, 5, 3868, 104, 122, 1, 3872, 101, 3996, 100, 3996, 100, 3996, 100, 3996, 100, 3996, 100, 3996, 100, 3996, 100, 4000, 96, 4004, 92, 4006, 85, 4014, 77, 4031, 60, 4047, 38, 4062, 28, 4070, 21, 269571] , dtype=np.uint32)
LIP_RLE = np.array( [9680875, 42, 4038, 74, 4011, 96, 3991, 114, 3975, 128, 3961, 142, 3948, 154, 3937, 164, 3927, 174, 3917, 184, 3907, 194, 3898, 202, 3890, 210, 3882, 218, 3874, 226, 3866, 234, 3859, 240, 3852, 248, 3845, 254, 3839, 260, 3833, 266, 3827, 272, 3821, 278, 3815, 284, 3810, 288, 3805, 294, 3799, 300, 3794, 304, 3789, 310, 3784, 314, 3780, 318, 3775, 324, 3770, 328, 3766, 332, 3762, 336, 3758, 340, 3753, 346, 3748, 350, 3744, 354, 3741, 356, 3738, 360, 3734, 364, 3730, 368, 3726, 372, 3722, 376, 3719, 378, 3716, 382, 3712, 386, 3709, 388, 3706, 392, 3702, 396, 3699, 398, 3696, 402, 3693, 404, 3690, 408, 3687, 410, 3684, 414, 3681, 416, 3679, 418, 3676, 422, 3673, 424, 3671, 426, 3668, 430, 3665, 432, 3663, 434, 3660, 438, 3657, 440, 3655, 442, 3653, 444, 3651, 446, 3649, 448, 3646, 452, 3643, 454, 3641, 456, 3639, 459, 3636, 463, 3632, 466, 3629, 469, 3626, 471, 3624, 473, 3622, 475, 3620, 477, 3618, 479, 3616, 481, 3615, 482, 3613, 483, 3612, 485, 3610, 486, 3609, 488, 3607, 490, 3606, 491, 3604, 493, 3602, 495, 3599, 498, 3596, 500, 3594, 503, 3592, 504, 3591, 506, 3589, 507, 3588, 508, 3587, 510, 3585, 511, 3584, 513, 3583, 514, 3581, 516, 3580, 516, 3579, 518, 3578, 518, 3577, 520, 3575, 521, 3575, 521, 3574, 523, 3573, 523, 3572, 524, 3572, 525, 3571, 525, 3570, 526, 3570, 526, 3570, 526, 3570, 526, 3570, 526, 3570, 527, 3569, 527, 3569, 527, 3569, 528, 3568, 528, 3568, 528, 3568, 529, 3567, 529, 3568, 528, 3567, 530, 3566, 530, 3566, 530, 3566, 530, 3565, 532, 3564, 532, 3564, 532, 3564, 532, 3564, 532, 3563, 534, 3562, 534, 3562, 534, 3562, 534, 3562, 534, 3562, 534, 3562, 534, 3561, 536, 3560, 536, 3560, 536, 3560, 536, 3560, 536, 3560, 536, 3560, 536, 3560, 536, 3560, 536, 3560, 536, 3560, 536, 3560, 536, 3560, 536, 3560, 536, 3560, 536, 3560, 536, 3560, 536, 3560, 536, 3560, 536, 3561, 534, 3562, 534, 3562, 534, 3562, 534, 3562, 534, 3562, 534, 3562, 534, 3563, 532, 3564, 532, 3564, 532, 3564, 532, 3564, 532, 3565, 530, 3566, 530, 3566, 530, 3566, 530, 3567, 528, 3568, 528, 3568, 528, 3569, 526, 3570, 526, 3570, 526, 3571, 524, 3572, 524, 3572, 524, 3573, 522, 3574, 522, 3575, 520, 3576, 520, 3577, 518, 3578, 518, 3578, 518, 3579, 516, 3580, 516, 3581, 514, 3582, 514, 3583, 512, 3585, 510, 3586, 510, 3587, 508, 3589, 507, 3591, 504, 3593, 503, 3594, 501, 3596, 499, 3598, 498, 3599, 496, 3601, 494, 3603, 493, 3604, 491, 3606, 489, 3608, 488, 3609, 486, 3611, 484, 3613, 482, 3615, 480, 3617, 479, 3618, 477, 3620, 475, 3622, 473, 3625, 470, 3627, 468, 3630, 466, 3631, 463, 3634, 460, 3638, 456, 3641, 454, 3643, 452, 3645, 450, 3647, 448, 3649, 446, 3652, 443, 3654, 439, 3658, 436, 3662, 432, 3665, 430, 3668, 427, 3670, 424, 3673, 420, 3678, 416, 3682, 412, 3686, 409, 3688, 407, 3690, 405, 3693, 402, 3695, 400, 3698, 397, 3700, 393, 3704, 389, 3709, 385, 3713, 381, 3717, 377, 3720, 374, 3723, 371, 3727, 367, 3731, 363, 3735, 359, 3738, 356, 3741, 352, 3745, 349, 3749, 344, 3754, 339, 3758, 336, 3761, 332, 3766, 327, 3771, 323, 3775, 318, 3779, 315, 3783, 311, 3788, 306, 3792, 302, 3797, 297, 3802, 292, 3806, 288, 3818, 276, 3823, 272, 3826, 267, 3831, 259, 3839, 254, 3844, 250, 3850, 244, 3855, 240, 3858, 237, 3861, 233, 3865, 225, 3875, 218, 3881, 213, 3885, 209, 3889, 206, 3891, 199, 3898, 195, 3902, 192, 3905, 181, 3916, 177, 3920, 174, 3923, 171, 3929, 166, 3935, 160, 3941, 154, 3947, 148, 3954, 141, 3962, 128, 3975, 114, 3991, 96, 4011, 74, 4038, 42, 5826539] , dtype=np.uint32)
TOP_ARM_RLE = np.array( [252870, 26, 4062, 46, 4, 20, 4, 4, 4, 22, 2, 2, 2, 2, 128, 16, 2, 4, 4, 10, 10, 2, 2, 6, 2, 6, 10, 22, 3760, 46, 4, 20, 4, 4, 4, 22, 2, 2, 2, 2, 128, 16, 2, 4, 4, 10, 10, 2, 2, 6, 2, 6, 10, 22, 3758, 132, 104, 72, 2, 4, 2, 28, 3750, 134, 104, 72, 2, 4, 2, 28, 3748, 136, 104, 108, 3746, 138, 97, 115, 3745, 139, 93, 119, 3743, 141, 89, 123, 3742, 142, 85, 130, 3738, 143, 82, 137, 3733, 144, 79, 145, 3727, 145, 77, 148, 3725, 146, 74, 153, 3722, 147, 71, 158, 3719, 148, 69, 165, 3713, 149, 67, 171, 3707, 151, 65, 176, 3703, 152, 64, 180, 3698, 154, 62, 184, 3695, 155, 61, 187, 3692, 158, 58, 190, 3689, 159, 57, 192, 3687, 160, 56, 195, 3684, 161, 55, 197, 3682, 162, 53, 200, 3680, 163, 52, 202, 3678, 164, 51, 204, 3676, 165, 50, 206, 3674, 166, 49, 208, 3673, 166, 48, 210, 3671, 167, 47, 212, 3669, 168, 46, 214, 3668, 168, 45, 216, 3666, 169, 44, 217, 3666, 167, 45, 219, 3664, 168, 45, 220, 3663, 168, 44, 221, 3662, 169, 43, 223, 3661, 167, 41, 227, 3661, 167, 32, 237, 3659, 168, 28, 241, 3659, 168, 25, 245, 3658, 170, 20, 248, 3657, 171, 18, 250, 3657, 171, 16, 253, 3656, 171, 14, 255, 3656, 171, 13, 256, 3655, 172, 11, 259, 3654, 174, 8, 260, 3654, 174, 7, 261, 3654, 174, 6, 262, 3654, 174, 5, 264, 3652, 175, 4, 265, 3652, 175, 3, 266, 3652, 173, 4, 267, 3651, 174, 3, 268, 3651, 445, 3651, 445, 3650, 446, 3650, 446, 3650, 446, 3649, 447, 3649, 447, 3648, 448, 3648, 448, 3648, 448, 3647, 449, 3647, 449, 3647, 448, 3647, 449, 3647, 449, 3647, 449, 3647, 448, 3647, 449, 3647, 449, 3647, 448, 3648, 448, 3648, 448, 3648, 447, 3649, 447, 3649, 446, 3650, 446, 3650, 445, 3650, 446, 3650, 445, 3651, 444, 3651, 445, 3651, 444, 3652, 443, 3653, 442, 3653, 442, 3654, 441, 3655, 441, 3655, 441, 3655, 441, 3655, 441, 3655, 441, 3655, 441, 3655, 441, 3654, 442, 3654, 442, 3654, 442, 3653, 443, 3653, 443, 3652, 444, 3652, 444, 3651, 445, 3651, 445, 3651, 445, 3650, 446, 3650, 444, 3652, 444, 3651, 447, 3649, 447, 3649, 447, 3649, 447, 3648, 448, 3648, 448, 1340, 17, 2291, 448, 95, 44, 1197, 30, 2282, 448, 91, 52, 1190, 42, 2273, 448, 88, 58, 1178, 60, 2264, 448, 85, 64, 1171, 68, 2260, 448, 83, 68, 986, 45, 135, 74, 2257, 448, 81, 73, 979, 53, 128, 80, 2254, 448, 79, 78, 968, 65, 122, 90, 2245, 449, 78, 81, 957, 78, 115, 97, 2241, 449, 76, 85, 946, 92, 107, 104, 2237, 449, 75, 88, 940, 113, 2, 2, 8, 20, 4, 20, 6, 14, 11, 116, 2228, 448, 73, 92, 934, 118, 2, 2, 8, 20, 4, 20, 6, 145, 2223, 451, 68, 97, 928, 156, 2, 174, 2220, 453, 64, 100, 924, 159, 2, 177, 2217, 455, 60, 103, 920, 348, 2210, 457, 56, 106, 912, 364, 2200, 460, 53, 108, 907, 372, 2196, 462, 49, 111, 902, 379, 2193, 463, 47, 113, 897, 386, 2190, 465, 44, 115, 893, 391, 2188, 466, 41, 118, 889, 399, 2182, 468, 38, 121, 886, 410, 2173, 469, 35, 124, 883, 416, 2168, 471, 32, 126, 881, 421, 2165, 472, 29, 129, 876, 428, 2162, 473, 27, 131, 872, 438, 2154, 475, 24, 133, 869, 450, 2145, 476, 22, 135, 866, 456, 2141, 477, 19, 137, 863, 464, 2135, 479, 16, 140, 859, 471, 2131, 479, 15, 141, 856, 478, 2127, 480, 12, 144, 852, 490, 2117, 482, 10, 145, 849, 502, 2108, 482, 9, 147, 846, 508, 2104, 483, 7, 148, 843, 514, 2100, 484, 5, 151, 839, 528, 2089, 485, 2, 154, 836, 539, 2080, 642, 833, 550, 2070, 643, 831, 570, 2052, 644, 828, 576, 2048, 645, 825, 582, 2044, 645, 822, 589, 2039, 647, 818, 606, 2025, 649, 4, 22, 2, 4, 782, 617, 2016, 649, 4, 22, 2, 4, 780, 628, 2007, 675, 2, 6, 776, 648, 1988, 676, 2, 6, 774, 659, 1979, 676, 2, 6, 772, 665, 1975, 676, 2, 6, 768, 674, 1969, 677, 2, 6, 764, 682, 1965, 677, 2, 6, 761, 699, 1951, 677, 4, 2, 760, 706, 1946, 678, 4, 2, 758, 713, 1941, 684, 756, 724, 1932, 684, 754, 735, 1922, 687, 750, 741, 1918, 687, 747, 747, 1915, 687, 744, 753, 1912, 687, 741, 764, 1903, 688, 739, 770, 1899, 688, 737, 776, 1895, 688, 735, 787, 1886, 688, 733, 793, 1881, 689, 731, 799, 1877, 689, 729, 805, 1873, 689, 727, 812, 1868, 689, 725, 818, 1863, 690, 723, 824, 1859, 690, 721, 830, 1855, 690, 719, 837, 1850, 690, 718, 842, 1846, 688, 718, 847, 1842, 689, 717, 851, 1839, 689, 716, 859, 1832, 689, 714, 865, 1828, 689, 2, 4, 707, 871, 1822, 690, 2, 4, 705, 877, 1818, 722, 678, 882, 1814, 722, 677, 887, 1810, 724, 673, 892, 1807, 724, 672, 896, 1804, 724, 670, 901, 1801, 724, 669, 906, 1796, 725, 668, 911, 1792, 725, 667, 917, 1787, 725, 666, 922, 1782, 726, 665, 926, 1779, 726, 663, 935, 1772, 726, 662, 945, 1762, 727, 661, 955, 1753, 727, 660, 965, 1744, 727, 659, 974, 1736, 727, 658, 979, 1731, 728, 657, 983, 1728, 728, 656, 987, 1725, 728, 655, 991, 1722, 728, 654, 995, 1719, 728, 653, 999, 1715, 729, 653, 1001, 1713, 729, 652, 1004, 1711, 729, 651, 1007, 1709, 729, 651, 1008, 1707, 730, 650, 1011, 1705, 730, 650, 1012, 1704, 730, 649, 1014, 1703, 730, 649, 1015, 2, 17, 1683, 730, 648, 1039, 1678, 731, 648, 1044, 1673, 731, 648, 1048, 1668, 732, 647, 1054, 1663, 732, 647, 1058, 1659, 732, 646, 1064, 1653, 733, 645, 1073, 1645, 733, 644, 1092, 1627, 733, 643, 1097, 1622, 734, 642, 1103, 1617, 734, 641, 1108, 1613, 732, 642, 1114, 1608, 732, 641, 1119, 1603, 733, 640, 1133, 1590, 733, 639, 1143, 1581, 735, 636, 1148, 1577, 735, 635, 1154, 1571, 736, 633, 1160, 1567, 736, 632, 1164, 1564, 736, 631, 1168, 1561, 736, 630, 1176, 1553, 737, 629, 1181, 1549, 737, 628, 1187, 1544, 737, 626, 1193, 1540, 737, 625, 1197, 1537, 735, 626, 1201, 1533, 736, 625, 1205, 1530, 732, 628, 1211, 1525, 732, 627, 1221, 1516, 736, 622, 1230, 1507, 737, 621, 1240, 1498, 739, 618, 1245, 1494, 739, 617, 1249, 1491, 739, 616, 1253, 1488, 739, 616, 1260, 1480, 740, 615, 1265, 1476, 740, 614, 1271, 1471, 740, 614, 1275, 1466, 741, 613, 1280, 1462, 741, 613, 1284, 1458, 741, 612, 1290, 1452, 742, 612, 1294, 1448, 742, 611, 1299, 1444, 742, 611, 1303, 1439, 743, 611, 1308, 1434, 743, 610, 1317, 1426, 743, 610, 1321, 1422, 743, 610, 1326, 1416, 744, 609, 1331, 1412, 744, 609, 1336, 1407, 744, 609, 1344, 1399, 744, 609, 1348, 1394, 745, 608, 1354, 1389, 745, 608, 1371, 1371, 746, 608, 1375, 1367, 746, 608, 1378, 1364, 746, 608, 1398, 1343, 747, 608, 1407, 1334, 747, 608, 1416, 1325, 747, 608, 1434, 1306, 748, 608, 1443, 1297, 748, 608, 1447, 1293, 748, 608, 1460, 1280, 748, 608, 1464, 1275, 749, 604, 1473, 1270, 747, 606, 1482, 1261, 747, 606, 1491, 1252, 733, 620, 1500, 1243, 733, 620, 1509, 1233, 734, 620, 1513, 1229, 734, 620, 1516, 1226, 734, 620, 1519, 1222, 735, 622, 1520, 1219, 735, 622, 1533, 1205, 736, 620, 1539, 1201, 736, 620, 1544, 1195, 737, 619, 1549, 1191, 737, 618, 1555, 1185, 738, 617, 1560, 1181, 738, 616, 1565, 1176, 739, 615, 1570, 1172, 739, 614, 1576, 1167, 738, 614, 1581, 1162, 739, 614, 1585, 1158, 739, 613, 1590, 1154, 739, 612, 1596, 1148, 739, 613, 1600, 1144, 739, 612, 1606, 1138, 740, 612, 1614, 1130, 739, 612, 1624, 1120, 740, 612, 1670, 1074, 740, 611, 1675, 1069, 740, 611, 1681, 1064, 740, 611, 1690, 1054, 740, 611, 1695, 1050, 740, 611, 1699, 1045, 740, 611, 1704, 1041, 740, 611, 1707, 1037, 740, 601, 1721, 1034, 739, 602, 1723, 1031, 740, 604, 1723, 1029, 739, 605, 1725, 1026, 739, 600, 2, 6, 1724, 1025, 738, 601, 2, 6, 1726, 1022, 738, 602, 6, 2, 1727, 1021, 737, 603, 6, 2, 1728, 1019, 737, 604, 1738, 1017, 736, 605, 1739, 1015, 736, 604, 1749, 1007, 735, 605, 1758, 997, 735, 606, 1767, 988, 733, 608, 1771, 983, 733, 609, 1775, 978, 733, 610, 1779, 973, 733, 611, 1782, 969, 733, 612, 1785, 965, 733, 613, 1793, 956, 734, 613, 1797, 952, 733, 614, 1802, 946, 733, 615, 1806, 941, 734, 615, 1810, 936, 734, 616, 1814, 931, 734, 617, 1819, 925, 734, 618, 1823, 920, 734, 621, 1826, 915, 733, 622, 1830, 910, 733, 622, 1836, 904, 733, 622, 1841, 899, 733, 623, 1844, 895, 733, 623, 1848, 891, 733, 623, 1860, 879, 732, 624, 1861, 879, 731, 624, 1865, 875, 730, 625, 1869, 871, 730, 625, 1872, 869, 730, 625, 1877, 863, 730, 625, 1882, 858, 730, 625, 1886, 854, 731, 624, 1890, 851, 730, 624, 1893, 848, 730, 624, 1896, 844, 732, 623, 1900, 840, 732, 624, 1904, 835, 733, 623, 1908, 830, 734, 623, 1912, 826, 734, 624, 1914, 822, 736, 623, 1917, 819, 736, 624, 1919, 816, 737, 623, 1924, 811, 738, 623, 1929, 805, 738, 623, 1934, 800, 739, 622, 1938, 796, 740, 622, 1941, 792, 740, 622, 1944, 789, 741, 622, 1949, 783, 741, 622, 1954, 778, 742, 622, 1959, 773, 741, 622, 1969, 763, 742, 622, 1973, 758, 742, 622, 1977, 754, 742, 615, 1998, 740, 743, 615, 1998, 739, 743, 614, 2006, 732, 743, 615, 2006, 731, 743, 616, 2006, 730, 744, 616, 2010, 725, 744, 619, 2011, 721, 744, 620, 2014, 717, 745, 620, 2016, 714, 745, 621, 2020, 709, 745, 622, 2029, 698, 746, 623, 2033, 693, 746, 622, 2040, 687, 746, 623, 2044, 682, 746, 624, 2049, 676, 747, 624, 2053, 671, 747, 625, 2058, 665, 748, 625, 2062, 660, 748, 626, 2067, 654, 749, 626, 2075, 645, 749, 627, 2079, 640, 750, 626, 2083, 637, 749, 627, 2086, 633, 750, 627, 2094, 624, 750, 627, 2104, 615, 750, 627, 2108, 610, 750, 628, 2112, 605, 751, 627, 2117, 600, 751, 628, 2122, 595, 750, 628, 2127, 590, 751, 628, 2132, 584, 751, 628, 2141, 575, 751, 629, 2145, 570, 752, 628, 2151, 564, 753, 628, 2155, 559, 754, 627, 2159, 554, 756, 627, 2162, 550, 757, 626, 2166, 546, 758, 626, 2169, 542, 759, 625, 2173, 538, 760, 625, 2175, 535, 761, 624, 2179, 531, 762, 624, 2182, 527, 763, 623, 2185, 524, 764, 623, 2187, 521, 765, 622, 2191, 517, 766, 622, 2195, 513, 766, 621, 2199, 509, 767, 621, 2202, 505, 767, 621, 2205, 503, 767, 621, 2208, 499, 768, 620, 2212, 496, 768, 620, 2214, 493, 768, 620, 2217, 491, 768, 620, 2220, 487, 769, 620, 2224, 483, 768, 620, 2228, 480, 768, 620, 2231, 476, 769, 620, 2233, 473, 769, 620, 2236, 469, 771, 620, 2238, 465, 772, 620, 2241, 462, 773, 620, 2243, 458, 774, 620, 2248, 453, 775, 620, 2252, 448, 784, 4, 20, 2, 12, 2, 14, 4, 70, 483, 2256, 444, 785, 4, 20, 2, 12, 2, 14, 4, 70, 474, 2, 4, 2262, 440, 920, 2, 18, 448, 2, 4, 2264, 437, 921, 2, 18, 448, 4, 2, 2266, 434, 942, 448, 4, 2, 2268, 431, 943, 448, 4, 2, 2270, 428, 936, 456, 4, 2, 2272, 425, 937, 454, 2283, 421, 936, 456, 2286, 418, 936, 456, 2289, 413, 948, 446, 2292, 409, 949, 446, 2295, 405, 950, 446, 2297, 402, 951, 446, 2301, 397, 952, 446, 2305, 391, 954, 446, 2308, 387, 957, 444, 2311, 383, 958, 444, 2313, 380, 959, 444, 2316, 376, 960, 444, 2319, 372, 961, 444, 2321, 368, 963, 444, 2325, 363, 964, 444, 2329, 358, 965, 444, 2332, 354, 966, 444, 2335, 350, 967, 444, 2337, 347, 968, 443, 2340, 343, 970, 443, 2343, 339, 971, 443, 2347, 334, 972, 443, 2350, 329, 974, 442, 2354, 325, 975, 442, 2356, 321, 977, 10, 2, 2, 2, 426, 2359, 317, 978, 10, 2, 2, 2, 426, 2362, 313, 1009, 411, 2365, 310, 1010, 411, 2368, 306, 1013, 409, 2371, 301, 1015, 409, 2373, 298, 1016, 408, 2379, 292, 1017, 408, 2383, 287, 1016, 410, 2388, 281, 1017, 410, 2392, 276, 1018, 410, 2396, 270, 1020, 409, 2401, 265, 1021, 409, 2404, 261, 1022, 408, 2408, 256, 1024, 408, 2411, 251, 1026, 408, 2414, 246, 1030, 405, 2418, 241, 1032, 405, 2420, 237, 1034, 405, 2425, 231, 1035, 404, 2430, 225, 1037, 404, 2433, 221, 1038, 404, 2436, 217, 1039, 404, 2439, 212, 1041, 403, 2444, 207, 1040, 405, 2448, 201, 1042, 405, 2453, 195, 1043, 405, 2457, 189, 1045, 405, 2460, 184, 1047, 404, 2464, 180, 1048, 404, 2467, 175, 1050, 404, 2471, 169, 1052, 404, 2475, 163, 1054, 403, 2481, 157, 1055, 403, 2485, 151, 1059, 401, 2488, 147, 1060, 401, 2491, 141, 1063, 401, 2493, 136, 1066, 401, 2496, 131, 1068, 401, 2499, 126, 1070, 400, 2503, 121, 1072, 400, 2506, 117, 1073, 400, 2509, 112, 1075, 399, 2512, 109, 1076, 399, 2514, 106, 1077, 399, 2516, 102, 1079, 398, 2518, 99, 1081, 398, 2520, 94, 1084, 398, 2521, 90, 1087, 398, 2522, 86, 1090, 397, 2524, 83, 1092, 397, 2525, 80, 1094, 397, 2526, 77, 1096, 397, 2527, 75, 1097, 397, 2528, 72, 1099, 396, 2530, 70, 1100, 396, 2531, 68, 1101, 396, 2532, 66, 1102, 396, 2532, 65, 1103, 395, 2534, 63, 1104, 395, 2535, 61, 1105, 395, 2535, 60, 1106, 395, 2536, 58, 1107, 395, 2536, 57, 1108, 394, 2572, 4, 8, 4, 1114, 394, 2572, 4, 8, 4, 1114, 394, 3698, 398, 3698, 397, 3695, 401, 3695, 401, 3701, 395, 3701, 394, 3702, 394, 3702, 394, 3702, 394, 3702, 393, 3703, 393, 3703, 393, 3703, 393, 3703, 393, 3705, 391, 3705, 391, 3705, 391, 3705, 391, 3705, 390, 3706, 390, 3706, 390, 3706, 390, 3706, 390, 3706, 390, 3706, 390, 3706, 390, 3704, 392, 3704, 391, 3707, 389, 3707, 389, 3707, 389, 3707, 389, 3705, 391, 3705, 391, 3696, 399, 3697, 399, 3696, 400, 3696, 399, 3697, 399, 3696, 400, 3696, 399, 3696, 400, 3696, 400, 3695, 401, 3695, 400, 3696, 400, 3695, 401, 3695, 401, 3695, 401, 3695, 401, 3694, 402, 3694, 402, 3694, 402, 3693, 403, 3693, 403, 3693, 403, 3692, 404, 3692, 404, 3691, 405, 3691, 405, 3690, 406, 3690, 403, 3692, 404, 3691, 403, 3693, 403, 3692, 404, 3692, 404, 3691, 405, 3691, 405, 3690, 406, 3689, 407, 3689, 409, 3686, 410, 3685, 411, 3685, 411, 3684, 410, 3686, 410, 3685, 411, 3684, 412, 3684, 412, 3683, 413, 3682, 414, 3681, 415, 3680, 416, 3679, 417, 3678, 420, 3675, 421, 3674, 422, 3673, 423, 3672, 424, 3670, 426, 3669, 427, 3667, 429, 3665, 431, 3663, 433, 3661, 435, 3659, 437, 3657, 439, 3655, 441, 3653, 443, 3651, 445, 3649, 447, 3647, 449, 3644, 452, 3641, 455, 3637, 459, 3632, 464, 3628, 468, 3619, 477, 3615, 481, 3611, 485, 3600, 496, 3597, 499, 3593, 503, 3584, 512, 3575, 521, 3557, 539, 3554, 542, 3552, 544, 3552, 544, 3552, 544, 3552, 544, 3554, 542, 3554, 542, 3554, 542, 3554, 542, 3554, 542, 3554, 542, 3552, 544, 3552, 544, 3544, 552, 3543, 553, 3543, 553, 3542, 554, 3542, 554, 3541, 555, 3541, 555, 3540, 556, 3539, 557, 3539, 557, 3538, 558, 3537, 559, 3536, 560, 3535, 561, 3534, 562, 3533, 563, 3532, 564, 3531, 565, 3530, 566, 3529, 567, 3527, 569, 3526, 570, 3524, 572, 3522, 574, 3520, 576, 3517, 579, 3514, 582, 3510, 586, 3494, 602, 3491, 605, 3488, 608, 3485, 611, 3482, 614, 3479, 617, 3475, 621, 3470, 626, 3467, 629, 3463, 633, 3458, 638, 3456, 640, 3453, 644, 3450, 645, 3448, 648, 3445, 651, 3441, 655, 3437, 659, 3433, 663, 3430, 664, 3429, 667, 3426, 670, 3426, 670, 3426, 670, 3426, 670, 3426, 670, 3426, 670, 3426, 672, 3424, 672, 3424, 672, 3424, 672, 3424, 672, 3412, 2, 2, 2, 6, 672, 3412, 2, 2, 2, 6, 674, 3389, 1, 14, 692, 3387, 3, 14, 690, 3386, 710, 3384, 714, 3381, 715, 3379, 718, 3377, 719, 3375, 718, 3376, 720, 3375, 719, 3375, 721, 3374, 722, 3373, 723, 3372, 724, 3372, 724, 3371, 725, 3371, 725, 3371, 725, 3370, 726, 3370, 728, 3367, 729, 3367, 733, 3362, 734, 3362, 730, 3365, 731, 3364, 730, 3366, 730, 3365, 731, 3364, 732, 3363, 733, 3362, 734, 3361, 735, 3360, 736, 3359, 739, 3356, 740, 3355, 741, 3354, 742, 3352, 744, 3351, 745, 3349, 747, 3347, 749, 3345, 751, 3342, 754, 3339, 757, 3335, 761, 3330, 766, 3330, 766, 3330, 766, 3329, 767, 3329, 767, 3329, 767, 3328, 768, 3328, 768, 3328, 768, 3327, 769, 3327, 769, 3326, 770, 3326, 770, 3326, 770, 3326, 770, 3326, 770, 3325, 771, 3325, 771, 3325, 771, 3325, 771, 3324, 770, 3326, 770, 3326, 770, 3325, 771, 3325, 771, 3325, 771, 3324, 772, 3324, 772, 3323, 773, 3323, 773, 3322, 774, 3322, 774, 3321, 775, 3321, 775, 3321, 775, 3321, 775, 3321, 775, 3321, 775, 3321, 775, 3321, 775, 3321, 775, 3321, 775, 3321, 775, 3321, 775, 3323, 773, 3324, 772, 3326, 770, 3328, 768, 3330, 768, 3329, 767, 3331, 777, 3320, 776, 3322, 774, 3324, 773, 3324, 772, 3326, 770, 3327, 769, 3329, 767, 3332, 765, 3333, 763, 3335, 761, 3337, 759, 3339, 757, 3341, 755, 3343, 753, 3345, 751, 3347, 749, 3349, 748, 3350, 746, 3352, 744, 3356, 740, 3359, 737, 3362, 734, 3364, 732, 3367, 729, 3370, 726, 3372, 725, 3373, 723, 3375, 721, 3377, 719, 3379, 718, 3381, 715, 3384, 712, 3386, 710, 3388, 708, 3390, 706, 3392, 704, 3394, 702, 3397, 699, 3400, 696, 3402, 694, 3404, 692, 3406, 690, 3408, 688, 3410, 687, 3411, 685, 3413, 683, 3415, 681, 3417, 680, 3419, 677, 3422, 674, 3424, 672, 3426, 670, 3428, 668, 3430, 666, 3432, 664, 3434, 662, 3436, 660, 3437, 659, 3439, 657, 3440, 656, 3441, 656, 3441, 655, 3442, 654, 3443, 653, 3444, 653, 3444, 652, 3445, 651, 3446, 650, 3447, 649, 3449, 648, 3449, 647, 3451, 645, 3452, 644, 3453, 643, 3454, 642, 3455, 641, 3456, 640, 3458, 638, 3459, 637, 3460, 636, 3461, 635, 3462, 634, 3463, 633, 3464, 632, 3465, 631, 3466, 630, 3467, 629, 3468, 629, 3468, 628, 3469, 627, 3470, 626, 3471, 626, 3471, 625, 3472, 624, 3472, 625, 3472, 624, 3473, 623, 3473, 624, 3473, 623, 3474, 623, 3474, 622, 3474, 623, 3474, 622, 3476, 621, 3477, 620, 3478, 618, 3479, 617, 3481, 615, 3482, 614, 3483, 613, 3485, 611, 3486, 610, 3488, 608, 3489, 607, 3490, 606, 3491, 605, 3492, 604, 3493, 603, 3494, 602, 3495, 601, 3496, 600, 3497, 599, 3498, 598, 3498, 598, 3499, 597, 3500, 596, 3500, 596, 3501, 597, 3499, 597, 3500, 598, 3498, 598, 3499, 597, 3499, 598, 3498, 598, 3499, 598, 3498, 598, 3498, 599, 3498, 598, 3498, 599, 3497, 600, 3496, 600, 3497, 600, 3496, 600, 3496, 601, 3495, 601, 3495, 602, 3494, 603, 3493, 603, 3493, 604, 3492, 605, 3491, 605, 3491, 605, 3491, 606, 3490, 606, 3490, 606, 3490, 607, 3489, 607, 3489, 608, 3488, 608, 3488, 609, 3487, 609, 3487, 610, 3486, 611, 3485, 611, 3485, 612, 3484, 613, 3483, 613, 3482, 614, 3482, 615, 3481, 615, 3481, 615, 3481, 616, 3480, 616, 3480, 617, 3479, 617, 3479, 618, 3477, 619, 3477, 620, 3476, 621, 3475, 621, 3474, 623, 3473, 624, 3472, 624, 3471, 626, 3470, 626, 3470, 626, 3469, 627, 3469, 628, 3467, 630, 3466, 630, 3465, 631, 3465, 632, 3463, 634, 3461, 635, 3461, 636, 3459, 637, 3458, 639, 3456, 640, 3455, 641, 3454, 643, 3453, 643, 3453, 643, 3451, 646, 3450, 646, 3450, 647, 3450, 646, 3450, 647, 3449, 647, 3450, 647, 3449, 648, 3448, 648, 3449, 648, 3448, 649, 3447, 649, 3447, 650, 3447, 649, 3447, 650, 3446, 651, 3445, 651, 3445, 652, 3444, 653, 3443, 654, 3442, 655, 3441, 656, 3440, 657, 3439, 657, 3439, 658, 3438, 659, 3437, 660, 3436, 660, 3436, 661, 3435, 662, 3434, 662, 3434, 663, 3433, 664, 3432, 665, 3431, 666, 3430, 667, 3429, 668, 3428, 668, 3428, 668, 3427, 671, 3425, 672, 3424, 671, 3425, 671, 3425, 669, 3427, 669, 3427, 667, 3429, 667, 3429, 667, 3429, 667, 3429, 667, 3429, 667, 3429, 667, 12, 2, 3415, 667, 12, 3, 3414, 669, 8, 6, 3413, 669, 8, 7, 3412, 685, 3411, 686, 3409, 688, 3408, 688, 3408, 689, 3407, 690, 3406, 691, 3405, 692, 3404, 693, 3403, 694, 3402, 695, 3401, 696, 3400, 697, 3399, 698, 3398, 699, 3395, 702, 3394, 703, 3393, 704, 3392, 705, 3391, 706, 3390, 708, 3388, 709, 3387, 710, 3386, 711, 3385, 712, 3386, 711, 3385, 712, 3384, 713, 3383, 714, 3382, 715, 3381, 716, 3378, 719, 3377, 721, 3377, 720, 3376, 720, 3376, 721, 3375, 722, 3374, 723, 3373, 724, 3372, 725, 3371, 726, 3370, 727, 3369, 728, 3368, 729, 3367, 730, 3366, 731, 3365, 732, 3364, 733, 3363, 734, 3362, 735, 3361, 736, 3360, 737, 3359, 738, 3358, 739, 3357, 740, 3356, 741, 3355, 742, 3354, 744, 3352, 745, 3351, 746, 3350, 747, 3349, 748, 3348, 749, 3345, 753, 3343, 754, 3340, 758, 3338, 760, 3338, 760, 3336, 761, 3335, 762, 3334, 763, 3333, 764, 3332, 765, 3331, 767, 3329, 768, 3328, 770, 3326, 772, 3324, 774, 3322, 775, 3321, 777, 3319, 779, 3319, 779, 3317, 781, 3315, 783, 3313, 785, 3311, 786, 3310, 788, 73, 1, 3234, 789, 71, 2, 3234, 791, 3305, 793, 3303, 795, 85, 4, 3212, 798, 82, 4, 3212, 801, 79, 2, 70, 2, 4, 2, 28, 2, 3074, 2, 18, 4, 8, 805, 75, 2, 70, 2, 4, 2, 28, 2, 3074, 2, 18, 4, 8, 810, 37, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 22, 20, 4, 10, 4, 4, 10, 12, 2, 4, 2, 10, 14, 4, 2, 24, 2, 3048, 8, 8, 10, 4, 816, 32, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 22, 20, 4, 10, 4, 4, 10, 12, 2, 4, 2, 10, 14, 4, 2, 24, 2, 3048, 8, 8, 10, 4, 825, 17, 190, 3034, 1063, 3033, 1064, 3032, 1065, 3031, 1066, 3028, 1069, 3027, 1070, 3028, 1070, 3026, 1070, 3026, 1070, 3026, 1075, 3021, 1077, 3019, 1078, 3018, 1078, 3018, 1079, 3017, 1079, 3017, 1080, 3014, 1083, 3013, 1083, 3013, 1084, 3012, 1085, 3011, 1086, 3010, 1086, 3010, 1086, 3010, 1088, 3010, 1086, 3010, 1089, 3007, 1090, 3006, 1091, 3005, 1093, 3003, 1094, 2960, 4, 8, 8, 20, 1098, 2958, 4, 8, 8, 20, 1100, 2948, 1150, 2946, 1152, 2944, 1152, 2944, 1153, 2944, 1152, 2944, 1153, 2943, 1154, 2943, 1153, 2943, 1154, 2942, 1155, 2942, 1155, 2941, 1156, 2941, 1156, 2940, 1157, 2939, 1158, 2939, 1158, 2938, 1159, 2937, 1160, 2937, 1161, 2935, 1162, 2934, 1164, 2932, 1166, 2931, 1167, 2929, 1170, 2926, 1173, 2923, 1177, 2919, 1181, 2915, 1181, 2915, 1182, 2914, 1182, 2914, 1182, 2915, 1181, 2915, 1182, 2914, 1182, 2914, 1182, 2914, 1182, 2914, 1183, 2913, 1183, 2913, 1183, 2913, 1183, 2914, 1183, 2913, 1183, 2913, 1183, 2913, 1184, 2912, 1184, 2912, 1184, 2912, 1185, 2911, 1185, 2911, 1186, 2910, 1186, 2910, 1187, 2909, 1187, 2909, 1188, 2908, 1189, 2908, 1188, 2908, 1189, 2907, 1190, 2906, 1190, 2907, 1189, 2907, 1189, 2907, 1189, 2907, 1192, 2905, 1191, 2905, 1192, 2904, 1193, 2903, 1194, 2903, 1194, 2902, 1195, 2901, 1196, 2900, 1196, 2900, 1197, 2900, 1196, 2900, 1196, 2900, 1197, 2899, 1197, 2900, 1196, 2900, 1197, 2899, 1197, 2900, 1197, 2899, 1197, 2900, 1197, 2899, 1197, 2900, 1197, 2899, 1198, 2899, 1197, 2900, 1197, 2900, 1197, 2900, 1197, 2900, 1196, 2900, 1197, 2900, 1196, 2901, 1196, 2900, 1196, 2901, 1196, 2900, 1196, 2901, 1196, 2900, 1196, 2901, 1196, 2900, 1197, 2899, 1197, 2900, 1197, 2899, 1197, 2899, 1198, 2899, 1197, 2899, 1198, 2898, 1198, 2898, 1199, 2898, 1198, 2898, 1199, 2897, 1199, 2897, 1200, 2896, 1200, 2896, 1200, 2896, 1201, 2895, 1201, 2895, 1199, 2897, 1199, 2897, 1199, 2897, 1199, 2897, 1199, 2897, 1199, 2897, 1199, 2897, 1199, 2897, 1199, 2900, 1196, 2900, 1198, 2904, 1192, 2904, 1192, 2904, 1192, 2904, 1192, 2904, 1192, 2904, 1192, 2904, 1192, 2904, 1192, 2912, 1184, 2912, 1182, 2914, 1182, 2914, 1182, 2914, 1182, 2914, 1182, 2916, 1180, 2916, 1180, 2916, 1180, 2916, 1180, 2916, 1180, 2916, 1180, 2916, 1180, 2916, 1182, 2914, 1182, 2914, 1182, 2914, 1182, 2914, 1182, 2914, 1182, 2914, 1182, 2914, 1181, 2915, 1181, 2913, 8, 12, 4, 18, 1141, 2913, 8, 12, 4, 18, 1140, 2966, 2, 2, 2, 6, 1118, 2966, 2, 2, 2, 6, 1118, 2978, 1117, 2979, 1117, 2963, 1133, 2963, 1133, 2961, 1134, 2962, 1134, 2966, 2, 2, 2, 8, 1116, 2966, 2, 2, 2, 8, 1116, 2980, 1116, 2980, 1116, 2980, 1116, 2980, 1116, 2084, 2, 6, 2, 6, 4, 878, 1113, 2085, 2, 6, 2, 6, 4, 878, 1113, 1943, 2, 140, 2, 4, 4, 8, 2, 878, 1113, 1943, 2, 140, 2, 4, 4, 8, 2, 878, 1113, 1943, 4, 138, 2, 2, 4, 10, 4, 876, 1113, 1943, 4, 138, 2, 2, 4, 10, 4, 876, 1113, 1939, 8, 138, 26, 872, 1113, 1939, 8, 138, 26, 872, 1113, 1935, 16, 134, 32, 864, 1115, 1935, 16, 134, 32, 864, 1115, 1945, 4, 136, 4, 16, 4, 872, 1115, 1945, 4, 136, 4, 16, 4, 872, 1115, 1945, 2, 140, 2, 16, 4, 872, 1115, 1945, 2, 140, 2, 16, 4, 872, 1115, 1945, 2, 158, 4, 872, 1115, 1945, 2, 158, 4, 872, 1115, 1889, 2, 54, 4, 46, 2, 108, 4, 876, 8, 20, 26, 20, 1037, 1889, 2, 54, 4, 46, 2, 108, 4, 876, 8, 20, 26, 20, 1037, 1761, 2, 16, 2, 6, 4, 14, 2, 8, 2, 16, 6, 12, 4, 6, 10, 14, 6, 10, 8, 10, 2, 10, 4, 8, 8, 8, 2, 14, 2, 18, 2, 10, 4, 6, 6, 20, 8, 8, 10, 12, 16, 2, 18, 4, 8, 2, 8, 886, 8, 40, 1023, 1761, 2, 16, 2, 6, 4, 14, 2, 8, 2, 16, 6, 12, 4, 6, 10, 14, 6, 10, 8, 10, 2, 10, 4, 8, 8, 8, 2, 14, 2, 18, 2, 10, 4, 6, 6, 20, 8, 8, 10, 12, 16, 2, 18, 4, 8, 2, 8, 886, 8, 40, 1023, 1745, 402, 878, 8, 40, 1023, 1745, 402, 878, 8, 40, 1023, 1747, 400, 874, 18, 38, 1019, 1747, 400, 874, 18, 38, 1019, 1747, 400, 876, 16, 38, 1019, 1747, 400, 876, 16, 38, 1019, 1735, 412, 882, 4, 44, 1019, 1735, 412, 882, 4, 44, 1019, 1735, 412, 884, 2, 44, 1019, 1735, 412, 884, 2, 44, 1019, 1733, 416, 928, 1019, 1733, 416, 928, 1019, 1733, 416, 928, 1019, 1733, 416, 928, 1019, 1733, 414, 930, 1019, 1733, 414, 930, 1019, 1733, 414, 932, 1017, 1733, 414, 932, 1017, 1733, 414, 932, 1017, 1733, 414, 932, 1017, 1729, 418, 932, 1017, 1729, 418, 932, 1017, 1731, 416, 932, 1017, 1731, 416, 932, 1017, 1733, 414, 932, 1017, 1733, 414, 932, 1017, 1635, 2, 20, 8, 68, 414, 932, 1017, 1635, 2, 20, 8, 68, 414, 932, 1017, 1481, 2, 26, 2, 124, 2, 20, 8, 68, 414, 932, 8, 8, 6, 8, 987, 1481, 2, 26, 2, 124, 2, 20, 8, 68, 414, 932, 8, 8, 6, 8, 987, 1479, 6, 4, 4, 6, 4, 6, 10, 42, 6, 20, 4, 14, 4, 10, 4, 10, 4, 14, 20, 24, 10, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 418, 4, 2, 8, 4, 6, 2, 908, 2, 12, 2, 12, 985, 1479, 6, 4, 4, 6, 4, 6, 10, 42, 6, 20, 4, 14, 4, 10, 4, 10, 4, 14, 20, 24, 10, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 418, 4, 2, 8, 4, 6, 2, 908, 2, 12, 2, 12, 985, 1457, 722, 904, 2, 26, 985, 1456, 723, 904, 2, 26, 985, 1455, 724, 932, 985, 1454, 725, 932, 985, 1453, 728, 928, 987, 1452, 729, 928, 987, 1451, 732, 922, 991, 1450, 733, 922, 991, 1449, 732, 928, 987, 1448, 733, 928, 987, 1447, 732, 932, 985, 1445, 734, 932, 985, 1444, 735, 932, 985, 1442, 737, 932, 985, 1440, 739, 932, 984, 1439, 741, 932, 984, 1437, 743, 932, 984, 1434, 746, 932, 984, 1431, 749, 932, 984, 1427, 753, 932, 983, 1425, 756, 932, 983, 1423, 758, 932, 983, 1420, 761, 932, 983, 1418, 763, 932, 982, 1418, 764, 930, 984, 1418, 764, 930, 984, 1418, 764, 932, 982, 1418, 764, 932, 982, 1418, 764, 932, 982, 1418, 764, 932, 982, 1368, 2, 2, 2, 42, 768, 8, 4, 12, 4, 932, 952, 1360, 2, 2, 2, 2, 2, 2, 2, 2, 2, 38, 768, 8, 4, 12, 4, 932, 952, 1350, 32, 17, 815, 930, 952, 1332, 882, 930, 952, 1314, 902, 912, 968, 1306, 910, 912, 968, 1291, 923, 912, 12, 2, 956, 1287, 927, 912, 12, 2, 956, 1261, 953, 930, 952, 1243, 971, 930, 951, 1189, 1026, 932, 947, 1164, 1053, 932, 945, 1157, 1062, 932, 943, 1148, 1073, 932, 941, 1147, 1076, 932, 939, 1145, 1080, 932, 937, 1143, 1084, 932, 936, 1140, 1088, 932, 934, 1137, 1093, 932, 933, 1134, 1097, 932, 932, 1127, 1105, 932, 931, 1125, 1108, 932, 929, 1124, 1113, 2, 2, 924, 929, 1122, 1117, 2, 2, 924, 927, 1120, 1125, 924, 925, 1118, 1129, 924, 924, 1110, 1138, 924, 922, 1103, 1147, 924, 921, 1099, 1148, 928, 920, 1098, 1150, 928, 919, 1099, 1150, 930, 916, 1100, 1150, 930, 910, 1087, 7, 2, 2, 6, 1164, 12, 2, 904, 910, 1083, 11, 2, 2, 6, 1164, 12, 2, 904, 910, 790, 7, 117, 3, 162, 1209, 896, 912, 726, 3, 59, 13, 111, 33, 117, 1226, 896, 912, 726, 5, 55, 25, 100, 39, 109, 1229, 896, 911, 726, 9, 49, 31, 95, 45, 102, 1232, 896, 910, 727, 12, 43, 38, 45, 3, 41, 64, 81, 1236, 898, 907, 727, 17, 35, 92, 37, 75, 45, 1263, 898, 906, 728, 147, 31, 1386, 898, 905, 728, 151, 25, 1389, 898, 904, 728, 156, 17, 1391, 900, 903, 729, 1564, 900, 903, 728, 1565, 900, 902, 728, 1566, 900, 901, 728, 1567, 898, 903, 727, 1568, 898, 896, 4, 2, 727, 1569, 898, 896, 732, 1570, 898, 896, 731, 1573, 898, 894, 730, 1574, 898, 894, 729, 1575, 898, 893, 729, 1576, 898, 892, 728, 1580, 896, 891, 728, 1581, 896, 890, 727, 1587, 890, 891, 726, 1589, 890, 890, 725, 1585, 896, 889, 723, 1588, 896, 888, 721, 1591, 896, 887, 718, 1595, 896, 887, 603, 2, 2, 2, 104, 1600, 4, 2, 892, 884, 604, 2, 2, 2, 104, 1600, 4, 2, 892, 883, 567, 4, 10, 12, 10, 8, 24, 2, 38, 6, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1610, 12, 2, 876, 883, 567, 4, 10, 12, 10, 8, 24, 2, 35, 9, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1610, 12, 2, 876, 882, 566, 79, 25, 1676, 866, 884, 566, 83, 17, 1680, 866, 883, 566, 1781, 866, 883, 566, 1781, 866, 882, 567, 1781, 866, 882, 566, 1782, 866, 882, 566, 1782, 866, 881, 567, 1782, 866, 881, 566, 1783, 866, 881, 566, 1783, 866, 880, 566, 1784, 866, 880, 566, 1784, 866, 880, 565, 1785, 866, 880, 565, 1785, 866, 879, 565, 1786, 866, 879, 564, 1787, 866, 879, 564, 1789, 864, 879, 563, 1790, 864, 879, 562, 1789, 866, 879, 561, 1790, 866, 879, 560, 1791, 868, 877, 559, 1792, 868, 877, 559, 1792, 868, 877, 559, 1792, 868, 877, 561, 1790, 868, 877, 561, 1790, 868, 877, 561, 1792, 866, 877, 561, 1792, 866, 877, 561, 1790, 868, 877, 561, 1790, 868, 877, 547, 2, 2, 2, 2, 2, 2, 1794, 4, 6, 16, 4, 834, 880, 546, 2, 2, 2, 2, 2, 2, 1794, 4, 6, 16, 4, 834, 880, 440, 5, 96, 1841, 829, 885, 437, 10, 91, 1844, 829, 885, 434, 14, 86, 1850, 827, 886, 429, 20, 79, 1855, 826, 887, 347, 10, 45, 49, 75, 1855, 828, 887, 346, 107, 70, 1858, 828, 888, 344, 111, 64, 1861, 827, 889, 343, 115, 57, 1865, 827, 889, 341, 121, 44, 1874, 827, 890, 339, 2040, 826, 891, 337, 2042, 826, 892, 334, 2044, 825, 893, 332, 2048, 823, 894, 328, 2051, 822, 895, 299, 1, 25, 2052, 824, 896, 298, 5, 17, 2056, 823, 898, 297, 2078, 822, 899, 297, 2078, 822, 900, 296, 2078, 821, 902, 295, 2078, 820, 904, 292, 2080, 819, 906, 291, 2080, 819, 907, 290, 2080, 818, 909, 289, 2080, 817, 911, 290, 2080, 815, 912, 289, 2080, 814, 914, 288, 2078, 815, 915, 288, 2078, 814, 916, 288, 2078, 813, 917, 288, 2078, 812, 920, 2, 2, 45, 6, 89, 14, 8, 6, 76, 6, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2082, 8, 2, 799, 921, 2, 4, 42, 46, 45, 3, 2, 14, 8, 6, 76, 6, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2082, 8, 2, 798, 930, 37, 64, 26, 44, 58, 2140, 796, 934, 31, 141, 51, 2143, 795, 938, 25, 148, 44, 2147, 793, 943, 17, 156, 36, 2151, 791, 3305, 790, 3306, 788, 3309, 785, 3311, 783, 3313, 781, 3316, 778, 3318, 776, 3320, 774, 3323, 770, 3326, 767, 3330, 762, 3334, 757, 3340, 754, 3342, 751, 3346, 747, 3350, 742, 3354, 737, 3360, 734, 3363, 730, 3367, 726, 3370, 722, 3375, 716, 3381, 712, 3384, 709, 3388, 705, 3392, 700, 3397, 690, 3407, 688, 3409, 688, 3409, 687, 3409, 687, 3410, 687, 3410, 686, 3411, 685, 3412, 684, 3413, 684, 3413, 683, 3413, 683, 3414, 682, 3415, 681, 3416, 680, 3417, 679, 3418, 678, 3419, 677, 3420, 676, 3421, 675, 3422, 674, 3423, 673, 3424, 672, 3425, 671, 3426, 670, 3427, 669, 3428, 667, 3430, 666, 3431, 665, 3432, 664, 3432, 663, 3434, 662, 3435, 661, 3436, 659, 3438, 658, 3439, 657, 3440, 655, 3442, 654, 3443, 652, 3445, 651, 3446, 649, 3449, 647, 3450, 645, 3452, 643, 3454, 642, 3455, 640, 3457, 638, 3460, 635, 3462, 633, 3465, 630, 3468, 627, 3471, 624, 3475, 621, 3478, 618, 3479, 617, 3481, 615, 3482, 614, 3484, 612, 3486, 610, 3488, 608, 3490, 606, 3492, 604, 3494, 603, 3494, 603, 3495, 602, 3495, 602, 3496, 600, 3498, 599, 3499, 598, 3499, 598, 3500, 597, 3501, 596, 3502, 595, 3504, 592, 3507, 590, 3507, 590, 3506, 590, 3506, 591, 3505, 592, 3504, 593, 3503, 593, 3503, 594, 3504, 593, 3503, 593, 3501, 596, 3500, 596, 3500, 597, 3499, 597, 3501, 596, 3500, 596, 3500, 596, 3500, 597, 3499, 597, 3499, 597, 3497, 602, 2, 6, 3486, 602, 2, 6, 3486, 602, 2, 8, 3484, 602, 2, 8, 3484, 612, 3484, 612, 3484, 612, 3484, 612, 3484, 612, 3484, 612, 3484, 612, 3484, 612, 3484, 612, 3484, 612, 3484, 612, 3484, 612, 3484, 612, 3484, 612, 3486, 610, 3486, 610, 3484, 612, 3484, 612, 3484, 612, 3484, 612, 3484, 611, 3485, 611, 3487, 609, 3487, 609, 3489, 608, 3488, 608, 3484, 612, 3484, 612, 3484, 612, 3484, 613, 3483, 613, 3483, 613, 3483, 613, 3483, 614, 3482, 614, 3482, 614, 3484, 612, 3484, 612, 3483, 613, 3482, 614, 3481, 615, 3481, 615, 3480, 616, 3480, 616, 3480, 616, 3480, 616, 3480, 616, 3480, 616, 3480, 616, 3480, 616, 3480, 616, 3480, 616, 3480, 616, 3481, 615, 3481, 615, 3481, 615, 3482, 614, 3482, 614, 3482, 614, 3482, 613, 3483, 613, 3483, 613, 3485, 611, 3485, 610, 3488, 608, 3489, 607, 3490, 605, 3492, 604, 3493, 603, 3494, 601, 3496, 600, 3497, 598, 3499, 597, 3500, 595, 3502, 594, 3503, 415, 1, 176, 3505, 414, 2, 174, 3507, 412, 5, 172, 3508, 410, 6, 171, 3509, 409, 8, 169, 3510, 408, 10, 167, 3512, 406, 12, 165, 3513, 405, 14, 163, 3514, 404, 16, 161, 3515, 403, 18, 159, 3516, 402, 20, 157, 3518, 400, 22, 155, 3519, 398, 25, 153, 3520, 397, 27, 151, 3521, 396, 29, 150, 3521, 395, 31, 148, 3522, 394, 34, 146, 3522, 393, 35, 146, 3522, 391, 37, 145, 3524, 389, 40, 143, 3524, 387, 42, 143, 3524, 385, 48, 138, 3525, 383, 52, 136, 3526, 379, 56, 134, 3527, 376, 61, 132, 3528, 371, 67, 129, 3529, 366, 74, 127, 3529, 366, 76, 124, 3530, 366, 77, 122, 3532, 366, 78, 120, 3532, 366, 79, 118, 3533, 367, 80, 115, 3535, 366, 82, 112, 3536, 366, 84, 109, 3538, 365, 85, 107, 3539, 365, 87, 104, 3540, 365, 89, 101, 3541, 365, 89, 100, 3543, 364, 89, 99, 3544, 364, 89, 98, 3546, 363, 89, 99, 3545, 363, 89, 99, 3546, 362, 90, 98, 3546, 362, 90, 98, 3546, 362, 91, 97, 3546, 362, 92, 96, 3547, 361, 93, 95, 3547, 361, 94, 93, 3548, 361, 96, 91, 3549, 360, 99, 87, 3550, 360, 103, 83, 3550, 360, 109, 76, 3551, 360, 109, 75, 3553, 359, 124, 59, 3554, 359, 126, 56, 3555, 359, 129, 51, 3557, 359, 133, 39, 3565, 359, 137, 32, 3568, 359, 3737, 359, 3737, 359, 3737, 359, 3737, 359, 3737, 359, 3737, 359, 3736, 360, 3736, 360, 3736, 360, 3736, 360, 3736, 360, 3736, 360, 3736, 360, 3735, 361, 3735, 361, 3735, 361, 3735, 185, 1, 175, 3734, 183, 4, 175, 3734, 180, 7, 175, 3734, 176, 11, 175, 3734, 168, 20, 174, 3734, 168, 20, 174, 3734, 168, 20, 174, 3734, 168, 20, 174, 3734, 168, 21, 173, 3734, 168, 21, 173, 3734, 168, 22, 171, 3735, 168, 23, 170, 3736, 167, 23, 170, 3736, 167, 24, 169, 3737, 166, 24, 169, 3737, 166, 24, 169, 3738, 165, 25, 168, 3738, 165, 25, 168, 3738, 164, 27, 167, 3739, 163, 27, 167, 3739, 163, 28, 166, 3740, 162, 29, 165, 3738, 164, 29, 165, 3738, 164, 30, 164, 3738, 163, 31, 163, 3739, 163, 31, 163, 3739, 163, 31, 163, 3739, 163, 31, 163, 3739, 163, 31, 162, 3740, 162, 32, 162, 3742, 160, 32, 162, 3742, 160, 32, 162, 3744, 158, 38, 155, 3745, 158, 39, 154, 3743, 160, 39, 153, 3744, 160, 40, 152, 3744, 159, 42, 151, 3744, 159, 42, 151, 3744, 159, 42, 151, 3744, 159, 43, 150, 3756, 147, 43, 148, 3758, 146, 45, 146, 3762, 143, 45, 146, 3762, 142, 47, 144, 3764, 140, 48, 144, 3764, 139, 50, 142, 3765, 138, 52, 140, 3767, 135, 55, 138, 3768, 134, 57, 136, 3770, 132, 58, 136, 3770, 130, 61, 134, 3772, 126, 65, 132, 3774, 122, 69, 131, 3775, 119, 72, 129, 3777, 112, 79, 127, 3779, 108, 83, 126, 3779, 107, 85, 124, 3781, 105, 87, 122, 3783, 103, 89, 120, 3784, 104, 90, 117, 3786, 103, 83, 123, 3788, 102, 83, 123, 3789, 101, 83, 122, 3791, 100, 83, 122, 3792, 99, 83, 121, 3794, 98, 83, 120, 3796, 98, 82, 119, 3797, 98, 82, 118, 3798, 98, 80, 119, 3799, 98, 80, 119, 3799, 98, 80, 118, 3800, 98, 81, 116, 3805, 92, 90, 10, 2, 97, 3805, 2, 1, 81, 2, 6, 90, 10, 2, 96, 3811, 77, 117, 90, 3814, 69, 125, 87, 3817, 37, 156, 85, 3823, 26, 163, 82, 4016, 79, 4017, 78, 4019, 75, 4022, 72, 4024, 70, 4027, 68, 4029, 65, 4032, 63, 4034, 60, 4037, 57, 4041, 54, 4043, 51, 4046, 48, 4050, 44, 4055, 39, 4062, 28, 4072, 22, 4081, 6, 1, 1, 1, 1, 1, 4, 8422112] , dtype=np.uint32)
EYELASHES_RLE = np.array([4774846, 16, 47, 16, 3994, 118, 3975, 125, 3969, 132, 3961, 140, 3953, 149, 3945, 158, 3935, 169, 1997, 16, 47, 16, 47, 16, 1785, 183, 47, 15, 1909, 193, 1746, 264, 1889, 199, 1741, 271, 1880, 207, 1735, 279, 1870, 215, 1728, 288, 1859, 224, 1722, 297, 1846, 234, 1715, 308, 1831, 245, 1709, 319, 1817, 255, 1701, 330, 1807, 261, 1693, 338, 1800, 269, 1685, 346, 1793, 275, 1677, 354, 1786, 283, 1668, 363, 1777, 291, 1659, 372, 1770, 297, 1652, 379, 1763, 305, 1646, 385, 1755, 312, 1641, 390, 1747, 320, 1636, 395, 1738, 330, 1630, 401, 1727, 340, 1624, 407, 1719, 349, 1618, 413, 1713, 355, 1611, 420, 1706, 361, 1606, 425, 1701, 367, 1601, 430, 1694, 374, 1596, 435, 1686, 382, 1591, 440, 1679, 389, 1586, 445, 1671, 397, 1580, 451, 1663, 406, 1574, 457, 1653, 415, 1569, 462, 1643, 426, 1562, 469, 1631, 437, 1557, 474, 1617, 451, 1551, 481, 1597, 469, 1547, 485, 1591, 475, 1543, 489, 1584, 482, 1539, 494, 1576, 490, 1535, 497, 1568, 498, 1531, 501, 1559, 507, 1527, 506, 1548, 518, 1522, 510, 1535, 531, 1519, 514, 1524, 542, 1514, 519, 1520, 546, 1509, 523, 1518, 548, 1505, 528, 1514, 552, 1500, 533, 1510, 556, 1495, 538, 1506, 560, 1490, 543, 1502, 564, 1485, 548, 1498, 568, 1480, 552, 1494, 574, 1474, 556, 1491, 578, 1469, 560, 1487, 584, 1463, 564, 1483, 589, 1458, 568, 1477, 596, 1453, 572, 1465, 608, 1448, 577, 1461, 612, 1444, 581, 1455, 618, 1440, 585, 1444, 630, 1434, 590, 1440, 634, 1430, 594, 1434, 640, 1425, 600, 1425, 649, 1419, 605, 1419, 655, 1415, 609, 1413, 662, 1409, 615, 1408, 667, 1403, 620, 1402, 673, 1398, 626, 1393, 682, 1392, 632, 1386, 688, 1387, 637, 1382, 692, 1381, 644, 1375, 698, 1376, 650, 1368, 703, 1371, 657, 1363, 707, 1366, 663, 1356, 713, 1361, 669, 1348, 720, 1357, 673, 1344, 724, 1353, 677, 1338, 730, 1349, 681, 1331, 737, 1344, 686, 1325, 743, 1340, 690, 1321, 747, 1336, 694, 1315, 753, 1331, 699, 1309, 759, 1327, 703, 1305, 763, 1322, 708, 1300, 768, 1317, 713, 1296, 772, 1313, 717, 1293, 776, 1307, 723, 1288, 780, 1302, 728, 1284, 784, 1297, 733, 1278, 791, 1291, 739, 1270, 798, 1286, 744, 1264, 805, 1279, 751, 1259, 810, 1273, 757, 1252, 816, 1267, 763, 1244, 825, 1261, 769, 1237, 832, 1254, 776, 1230, 839, 1246, 784, 1225, 844, 1240, 790, 1218, 851, 1235, 794, 1211, 860, 1229, 798, 1207, 865, 1223, 803, 1202, 872, 1217, 807, 1198, 877, 1211, 812, 1192, 885, 1204, 817, 1186, 892, 1199, 821, 1182, 896, 1194, 826, 1176, 903, 1188, 831, 1170, 909, 1183, 836, 1166, 913, 1178, 841, 1160, 920, 1172, 846, 1156, 924, 1166, 853, 1149, 931, 1160, 858, 1143, 938, 1153, 864, 1139, 941, 1149, 869, 1133, 946, 1144, 874, 1127, 952, 1138, 880, 1122, 958, 1132, 885, 1119, 961, 1126, 891, 1114, 966, 1120, 897, 1109, 972, 1112, 904, 1106, 975, 1104, 912, 1101, 980, 1095, 921, 1094, 988, 1090, 925, 1089, 993, 1085, 930, 1084, 999, 1080, 934, 1081, 1002, 1075, 939, 1077, 1007, 1070, 944, 1073, 1010, 1065, 949, 1071, 1013, 1058, 955, 1068, 1016, 1053, 960, 1065, 1020, 1046, 966, 1060, 1025, 1040, 972, 1058, 1028, 1032, 980, 1054, 1031, 1024, 988, 1051, 1035, 1014, 997, 1047, 1040, 1001, 1009, 1044, 1043, 983, 1028, 1040, 1047, 980, 1030, 1037, 1051, 977, 1032, 1035, 1054, 974, 1034, 1032, 1057, 972, 1037, 1028, 1061, 969, 1039, 1026, 1064, 966, 1041, 1023, 1068, 963, 1044, 1019, 1072, 960, 1046, 1016, 1076, 957, 1048, 1013, 1080, 954, 1051, 1010, 1083, 951, 1053, 1007, 1087, 948, 1056, 1003, 1091, 945, 1058, 1001, 1094, 942, 1061, 997, 1098, 939, 1063, 995, 1101, 936, 1066, 992, 1104, 933, 1068, 990, 1108, 928, 1072, 986, 1112, 925, 1074, 984, 1115, 922, 1077, 980, 1120, 918, 1079, 977, 1124, 915, 1082, 974, 1128, 911, 1085, 971, 1132, 906, 1088, 969, 1135, 903, 1091, 966, 1139, 899, 1094, 963, 1143, 895, 1097, 960, 1147, 891, 1099, 957, 1152, 887, 1102, 954, 1156, 883, 1105, 950, 1162, 878, 1108, 947, 1166, 874, 1111, 944, 1171, 869, 1114, 941, 1175, 865, 1117, 937, 1181, 860, 1120, 934, 1187, 854, 1123, 930, 1193, 849, 1126, 926, 1200, 842, 1130, 922, 1207, 836, 1133, 918, 1215, 829, 1136, 916, 1221, 822, 1139, 913, 1227, 816, 1143, 910, 1232, 810, 1146, 907, 1239, 802, 1150, 904, 1247, 794, 1153, 902, 1255, 785, 1156, 899, 1265, 775, 1158, 897, 1267, 772, 1162, 894, 1269, 770, 1164, 892, 1271, 768, 1167, 889, 1273, 766, 1170, 885, 1276, 763, 1173, 883, 1278, 761, 1176, 879, 1282, 758, 1179, 875, 1285, 755, 1183, 871, 1288, 753, 1185, 869, 1290, 751, 1188, 866, 1292, 748, 1192, 863, 1294, 746, 1195, 860, 1297, 742, 1199, 856, 1300, 740, 1202, 853, 1302, 737, 1206, 850, 1304, 735, 1209, 846, 1308, 731, 1213, 843, 1310, 729, 1216, 840, 1312, 726, 1220, 836, 1315, 724, 1223, 833, 1318, 720, 1227, 829, 1321, 718, 1230, 825, 1324, 715, 1235, 821, 1327, 711, 1239, 818, 1329, 709, 1242, 814, 1332, 706, 1247, 810, 1335, 702, 1251, 806, 1338, 699, 1256, 801, 1342, 696, 1260, 797, 1344, 693, 1264, 793, 1348, 689, 1269, 788, 1351, 686, 1274, 784, 1354, 682, 1279, 779, 1357, 679, 1284, 775, 1360, 675, 1289, 771, 1362, 672, 1294, 766, 1366, 668, 1297, 764, 1368, 665, 1300, 761, 1372, 661, 1304, 757, 1376, 657, 1307, 755, 1378, 654, 1311, 752, 1381, 650, 1315, 748, 1385, 645, 1321, 744, 1388, 641, 1325, 740, 1391, 315, 3, 320, 1331, 734, 1395, 311, 7, 315, 1343, 724, 1398, 307, 11, 311, 1347, 720, 1402, 303, 15, 306, 1351, 717, 1406, 299, 19, 301, 1356, 713, 1410, 295, 23, 297, 1360, 709, 1414, 291, 28, 291, 1366, 705, 1417, 287, 32, 286, 1371, 701, 1421, 283, 36, 281, 1376, 697, 1425, 279, 41, 275, 1380, 694, 1429, 274, 46, 270, 1385, 690, 1433, 270, 51, 263, 1391, 687, 1436, 266, 56, 257, 1398, 681, 1440, 261, 61, 251, 1408, 673, 1445, 256, 66, 245, 1413, 668, 1450, 251, 72, 238, 1419, 664, 1454, 246, 78, 230, 1428, 658, 1459, 241, 83, 223, 1436, 652, 1463, 236, 89, 215, 1443, 646, 1470, 230, 96, 206, 1452, 22, 12, 604, 1477, 224, 102, 197, 1463, 13, 18, 600, 1481, 219, 109, 186, 1505, 593, 1487, 213, 115, 175, 1530, 574, 1492, 206, 123, 160, 1545, 566, 1499, 200, 131, 139, 1567, 556, 1506, 193, 139, 131, 1576, 549, 1511, 187, 147, 121, 1587, 539, 1519, 179, 156, 111, 1596, 528, 1529, 171, 167, 99, 1613, 513, 1537, 163, 178, 85, 1687, 441, 1545, 157, 189, 69, 1699, 431, 1555, 149, 204, 47, 1731, 405, 1565, 139, 2043, 338, 1580, 131, 2051, 329, 1590, 121, 2067, 313, 1600, 111, 2078, 300, 1613, 99, 2088, 292, 1624, 85, 2111, 270, 1638, 69, 2124, 261, 1653, 47, 2141, 239, 3873, 207, 3893, 201, 3897, 198, 3899, 96, 9, 90, 3901, 91, 55, 47, 3906, 70, 77, 39, 3914, 62, 4055, 35, 3981249, 6, 4090, 12, 4084, 17, 4079, 22, 4074, 26, 4070, 31, 4065, 35, 4061, 38, 6, 13, 4039, 62, 4034, 66, 4030, 68, 4028, 71, 4025, 73, 4023, 77, 4019, 81, 4015, 83, 4013, 86, 4010, 92, 4004, 98, 3998, 120, 3976, 132, 3964, 136, 3960, 138, 3958, 141, 3955, 143, 3953, 146, 3950, 148, 3948, 150, 3946, 151, 3945, 155, 3941, 157, 3939, 160, 3936, 162, 3934, 166, 3930, 168, 3928, 170, 3926, 172, 3924, 175, 3921, 177, 3919, 179, 3917, 180, 3916, 182, 3914, 183, 3913, 184, 3912, 186, 3910, 188, 3908, 189, 3906, 192, 3900, 197, 3848, 249, 3798, 300, 3792, 306, 3784, 313, 3754, 344, 3748, 350, 3744, 353, 3740, 358, 3734, 363, 3731, 366, 3727, 371, 3723, 374, 3720, 377, 3717, 381, 3711, 386, 3708, 389, 3704, 393, 3701, 396, 3698, 399, 3696, 402, 3692, 406, 3689, 408, 3687, 411, 3683, 415, 3680, 417, 3678, 420, 3674, 423, 3671, 426, 3669, 429, 3665, 432, 3662, 435, 3659, 439, 3654, 443, 3651, 446, 3648, 449, 3646, 452, 3642, 455, 3640, 457, 3638, 460, 3634, 463, 3631, 466, 3628, 470, 3624, 473, 3622, 475, 3619, 479, 3615, 483, 3611, 487, 3607, 491, 3603, 494, 3600, 498, 3597, 500, 3594, 503, 3590, 508, 3584, 514, 3580, 517, 3577, 521, 3574, 523, 3571, 526, 3569, 529, 3566, 532, 3562, 535, 3558, 540, 3554, 544, 3550, 548, 3547, 551, 3541, 556, 3538, 560, 3534, 565, 3530, 568, 3526, 573, 3521, 577, 3518, 580, 3514, 583, 3511, 587, 3507, 590, 3504, 593, 3502, 596, 3498, 600, 3495, 602, 3493, 605, 3487, 610, 3484, 613, 3481, 617, 3478, 619, 3475, 622, 3472, 626, 3469, 628, 3466, 631, 3464, 634, 1870, 47, 1544, 636, 1858, 69, 1531, 639, 1849, 85, 1522, 641, 1841, 99, 1514, 644, 1833, 111, 1506, 647, 1827, 121, 1500, 649, 1821, 131, 1494, 652, 1815, 139, 1488, 655, 1809, 149, 1481, 658, 1804, 157, 1474, 663, 1799, 163, 1469, 666, 1794, 171, 1462, 670, 1790, 177, 1457, 674, 1784, 185, 1451, 678, 1779, 191, 1446, 681, 1775, 197, 1439, 687, 1770, 203, 1434, 690, 1766, 209, 1429, 693, 1762, 215, 1425, 696, 1758, 219, 1421, 699, 1754, 225, 1415, 703, 1750, 231, 1410, 707, 1746, 235, 1406, 711, 1741, 241, 1402, 713, 1553, 47, 138, 245, 1398, 717, 1540, 69, 125, 249, 1395, 719, 1531, 85, 114, 255, 1391, 721, 1523, 99, 105, 259, 1385, 727, 1515, 111, 97, 263, 1381, 730, 1509, 121, 90, 267, 1377, 733, 1503, 131, 83, 271, 1374, 736, 1497, 139, 77, 275, 1370, 739, 1491, 149, 70, 279, 1367, 741, 1486, 157, 64, 283, 1364, 743, 1482, 163, 59, 287, 1360, 747, 1476, 171, 53, 291, 1357, 749, 1472, 177, 48, 295, 1354, 752, 1466, 185, 42, 299, 1351, 754, 1462, 191, 37, 303, 1348, 756, 1458, 197, 32, 307, 1345, 759, 1453, 203, 28, 309, 1343, 761, 1449, 209, 23, 313, 1339, 764, 1445, 215, 18, 317, 1336, 767, 1441, 219, 14, 321, 1333, 770, 1436, 225, 10, 328, 1326, 772, 1432, 231, 5, 335, 1320, 775, 1428, 235, 1, 342, 1314, 777, 1424, 585, 1309, 779, 1364, 47, 10, 592, 1302, 782, 1352, 664, 1297, 784, 1343, 675, 1293, 786, 1335, 686, 1287, 789, 1328, 695, 1283, 791, 1322, 704, 1278, 794, 1315, 712, 1274, 796, 1310, 719, 1270, 799, 1303, 727, 1266, 801, 1298, 734, 1259, 806, 1294, 740, 1254, 810, 1288, 746, 1250, 814, 1283, 752, 1246, 817, 1277, 759, 1241, 821, 1272, 764, 1237, 824, 1268, 770, 1232, 828, 1263, 775, 1228, 831, 1259, 780, 1225, 833, 1255, 786, 1220, 836, 1252, 790, 1216, 840, 1247, 795, 1213, 842, 1243, 800, 1209, 846, 1239, 804, 1206, 848, 1235, 809, 1203, 850, 1232, 813, 1197, 855, 1229, 817, 1193, 859, 1224, 822, 1189, 862, 1221, 826, 1186, 865, 1217, 830, 1182, 869, 1213, 834, 1177, 873, 1210, 838, 1173, 877, 1206, 842, 1169, 881, 1202, 845, 1167, 883, 1199, 849, 1163, 887, 1195, 853, 1160, 890, 1191, 857, 1157, 892, 1188, 860, 1154, 896, 1184, 864, 1150, 899, 1181, 868, 1147, 902, 1177, 871, 1142, 910, 1172, 874, 1138, 914, 1168, 878, 1134, 918, 1164, 882, 1131, 920, 1161, 887, 1126, 924, 1158, 890, 1122, 927, 1155, 894, 1119, 930, 1151, 899, 1114, 934, 1148, 902, 1109, 939, 1144, 906, 1103, 945, 1141, 909, 1099, 948, 1138, 913, 1095, 952, 1135, 916, 1092, 955, 1131, 920, 1088, 958, 1129, 923, 1084, 962, 1125, 927, 1080, 966, 1122, 930, 1075, 971, 1118, 934, 1071, 976, 1114, 937, 1066, 981, 1110, 941, 1062, 985, 1107, 944, 1058, 988, 1105, 946, 1056, 991, 1101, 950, 1052, 995, 1098, 953, 1047, 999, 1096, 956, 1043, 1003, 1092, 959, 1040, 1006, 1090, 962, 1037, 1009, 1087, 965, 1032, 1014, 1084, 967, 1029, 1018, 1080, 971, 1024, 1023, 1077, 973, 1021, 1026, 1075, 976, 1017, 1030, 1072, 979, 1014, 1032, 1069, 983, 1008, 1037, 1067, 986, 1004, 1041, 1064, 990, 999, 1045, 1061, 993, 996, 1047, 1059, 996, 992, 1051, 1056, 999, 987, 1055, 1053, 1003, 981, 1060, 1051, 1006, 977, 1064, 1048, 1009, 971, 1070, 1045, 1012, 967, 1074, 1042, 1015, 963, 1078, 1039, 1018, 960, 1080, 1037, 1021, 956, 1084, 1034, 1024, 952, 1088, 1031, 1027, 948, 1091, 1029, 1030, 944, 1095, 1026, 1032, 941, 1098, 1024, 1035, 937, 1101, 1022, 1038, 934, 1104, 1019, 1041, 930, 1107, 1017, 1043, 927, 1111, 1014, 1046, 923, 1115, 1011, 1049, 919, 1118, 1009, 1051, 916, 1122, 1006, 1054, 910, 1128, 1003, 1056, 907, 1131, 1001, 1059, 902, 1136, 998, 1061, 899, 1140, 991, 1068, 895, 1143, 986, 1073, 893, 1146, 981, 1078, 889, 1149, 976, 1083, 887, 1152, 971, 1088, 884, 1155, 965, 1093, 880, 1159, 961, 1098, 876, 1163, 956, 1102, 873, 1167, 951, 1117, 860, 1169, 947, 1131, 847, 1173, 942, 1142, 837, 1177, 938, 1151, 828, 1181, 933, 1160, 820, 1185, 928, 1168, 814, 1187, 925, 1175, 807, 1191, 920, 1182, 801, 1194, 917, 1189, 795, 1196, 914, 1195, 789, 1200, 909, 1201, 785, 1202, 906, 1207, 780, 1204, 903, 1212, 775, 1208, 899, 1220, 767, 1212, 895, 1230, 758, 1215, 891, 1239, 749, 1219, 887, 1247, 742, 1221, 884, 1254, 730, 1230, 880, 1261, 721, 1236, 876, 1267, 715, 1240, 872, 1274, 708, 1244, 868, 1280, 703, 1247, 864, 1285, 698, 1253, 858, 1291, 693, 1256, 855, 1295, 689, 1259, 851, 1301, 684, 1261, 848, 1306, 680, 1264, 844, 1311, 676, 1267, 841, 1315, 673, 1269, 837, 1320, 667, 1276, 831, 1325, 659, 1286, 825, 1328, 653, 1294, 819, 1333, 645, 1301, 816, 1337, 638, 1309, 810, 1341, 633, 1316, 805, 1345, 626, 1322, 772, 1378, 621, 1329, 757, 1391, 615, 1337, 745, 1402, 609, 1342, 736, 1411, 604, 1349, 726, 1419, 599, 1356, 717, 1426, 594, 1361, 710, 1433, 589, 1368, 702, 1439, 585, 1372, 695, 1446, 580, 1378, 688, 1452, 575, 1385, 681, 1457, 571, 1391, 673, 1463, 566, 1400, 664, 1468, 562, 1407, 655, 1474, 558, 1414, 647, 1479, 553, 1421, 640, 1484, 549, 1428, 632, 1488, 546, 1434, 625, 1493, 542, 1439, 619, 1498, 538, 1445, 613, 1502, 534, 1450, 607, 1506, 531, 1456, 600, 1511, 527, 1461, 595, 1515, 523, 1466, 589, 1519, 520, 1471, 584, 1523, 516, 1476, 579, 1526, 513, 1481, 573, 1531, 509, 1485, 569, 1534, 506, 1490, 564, 1538, 503, 1494, 559, 1541, 500, 1498, 555, 1545, 496, 1503, 550, 1548, 493, 1507, 546, 1552, 490, 1510, 542, 1555, 487, 1515, 537, 1559, 483, 1519, 533, 1562, 481, 1522, 529, 1565, 478, 1527, 524, 1569, 475, 1533, 517, 1572, 472, 1539, 511, 1575, 470, 1542, 508, 1578, 466, 1553, 497, 1581, 464, 1560, 489, 1584, 461, 1566, 483, 1587, 459, 1569, 480, 1590, 455, 1575, 474, 1593, 453, 1580, 468, 1596, 450, 1584, 465, 1598, 448, 1587, 18, 47, 396, 1602, 445, 1589, 6, 69, 384, 1604, 442, 1674, 374, 1607, 440, 1682, 366, 1609, 438, 1689, 358, 1654, 393, 1696, 352, 1666, 381, 1702, 345, 1676, 372, 1707, 340, 1684, 364, 1713, 333, 1692, 356, 1719, 328, 1698, 350, 1723, 323, 1705, 344, 1728, 318, 1710, 339, 1732, 314, 1716, 332, 1738, 308, 1722, 327, 1742, 304, 1726, 323, 1746, 300, 1731, 318, 1750, 239, 47, 9, 1736, 314, 1754, 225, 1807, 309, 1758, 214, 1818, 304, 1762, 205, 1828, 300, 1766, 196, 1837, 296, 1770, 188, 1845, 292, 1773, 181, 1853, 288, 1777, 174, 1859, 285, 1780, 167, 1867, 168, 47, 66, 1783, 161, 1874, 154, 69, 54, 1787, 155, 1879, 144, 85, 45, 1790, 149, 1886, 134, 99, 37, 1793, 1, 47, 96, 1891, 126, 111, 30, 1853, 81, 1897, 119, 121, 24, 1862, 70, 1903, 111, 131, 18, 1870, 60, 1908, 105, 139, 13, 1877, 51, 1913, 98, 149, 7, 1883, 43, 1918, 92, 157, 2, 1889, 35, 1923, 87, 2055, 29, 1927, 81, 2064, 21, 1932, 76, 2071, 14, 1937, 70, 2078, 9, 1941, 65, 2085, 2, 1946, 60, 4038, 55, 4043, 50, 4048, 45, 4053, 41, 4056, 37, 4061, 32, 4066, 28, 4070, 23, 4074, 20, 4078, 16, 4082, 11, 4086, 8, 4090, 4, 4093, 1, 5108741], dtype=np.int64)
EYE_SOCKET_RLE = np.array( [734096, 106, 2049, 106, 1829, 118, 2037, 118, 1814, 131, 2029, 131, 1799, 142, 2019, 142, 1741, 200, 2007, 200, 1683, 214, 1991, 214, 1673, 233, 1961, 233, 1666, 250, 1933, 250, 1660, 259, 1921, 259, 1654, 266, 1913, 266, 1649, 271, 1907, 271, 1645, 276, 1901, 276, 1641, 281, 1895, 281, 1637, 285, 1891, 285, 1633, 289, 1887, 289, 1628, 295, 1881, 295, 1623, 300, 1875, 300, 1619, 306, 1867, 306, 1615, 312, 1859, 312, 1611, 317, 1853, 317, 1607, 322, 1847, 322, 1602, 329, 1839, 329, 1597, 335, 1831, 335, 1593, 340, 1825, 340, 1589, 345, 1819, 345, 1585, 350, 1813, 350, 1581, 354, 1809, 354, 1575, 361, 1803, 361, 1567, 368, 1797, 368, 1560, 374, 1791, 374, 1554, 380, 1785, 380, 1547, 388, 1777, 388, 1539, 395, 1771, 395, 1532, 401, 1765, 401, 1526, 407, 1759, 407, 1519, 413, 1755, 413, 1492, 439, 1749, 439, 1463, 448, 1743, 448, 1453, 455, 1737, 455, 1446, 462, 1729, 462, 1440, 469, 1721, 469, 1434, 476, 1713, 476, 1429, 484, 1701, 484, 1425, 490, 1693, 490, 1421, 495, 1687, 495, 1417, 500, 1681, 500, 1413, 505, 1675, 505, 1410, 508, 1671, 508, 1407, 517, 1657, 517, 1404, 524, 1645, 524, 1401, 530, 1637, 530, 1398, 535, 1629, 535, 1396, 542, 1617, 542, 1394, 547, 1609, 547, 1392, 551, 1603, 551, 1389, 556, 1597, 556, 1386, 560, 1591, 560, 1384, 563, 1587, 563, 1382, 567, 1581, 567, 1380, 571, 1575, 571, 1379, 574, 1569, 574, 1378, 579, 1561, 579, 1376, 584, 1553, 584, 1374, 588, 1547, 588, 1372, 592, 1541, 592, 1370, 596, 1535, 596, 1369, 598, 1531, 598, 1368, 602, 1525, 602, 1366, 606, 1519, 606, 1365, 609, 1513, 609, 1364, 613, 1507, 613, 1362, 616, 1503, 616, 1361, 618, 1499, 618, 1360, 622, 1493, 622, 1359, 625, 1487, 625, 1358, 629, 1481, 629, 1357, 631, 1477, 631, 1356, 634, 1473, 634, 1355, 636, 1469, 636, 1354, 639, 1465, 639, 1353, 642, 1459, 642, 1352, 647, 1451, 647, 1351, 651, 1443, 651, 1351, 654, 1437, 654, 1350, 661, 1425, 661, 1349, 665, 1417, 665, 1349, 668, 1411, 668, 1348, 672, 1405, 672, 1347, 675, 1399, 675, 1347, 677, 1395, 677, 1346, 680, 1391, 680, 1345, 682, 1387, 682, 1345, 684, 1383, 684, 1345, 687, 1377, 687, 1344, 691, 1371, 691, 1343, 693, 1367, 693, 1343, 695, 1363, 695, 1343, 697, 1359, 697, 1343, 699, 1355, 699, 1343, 701, 1351, 701, 1342, 705, 1345, 705, 1341, 707, 1341, 707, 1341, 710, 1335, 710, 1341, 714, 1327, 714, 1341, 720, 1315, 720, 1341, 724, 1307, 724, 1341, 727, 1301, 727, 1341, 730, 1295, 730, 1341, 733, 1289, 733, 1341, 735, 1285, 735, 1341, 737, 1281, 737, 1341, 739, 1277, 739, 1341, 741, 1273, 741, 1341, 743, 1269, 743, 1341, 744, 1267, 744, 1341, 746, 1263, 746, 1341, 747, 1261, 747, 1341, 749, 1257, 749, 1341, 751, 1253, 751, 1342, 752, 1249, 752, 1343, 755, 1243, 755, 1343, 758, 1237, 758, 1343, 762, 1229, 762, 1343, 765, 1223, 765, 1343, 768, 1217, 768, 1344, 770, 1211, 770, 1345, 772, 1207, 772, 1345, 775, 1201, 775, 1345, 778, 1195, 778, 1346, 780, 1189, 780, 1347, 783, 1183, 783, 1347, 785, 1179, 785, 1348, 786, 1175, 786, 1349, 788, 1171, 788, 1349, 790, 1167, 790, 1350, 791, 1163, 791, 1351, 794, 1157, 794, 1351, 796, 1153, 796, 1352, 797, 1149, 797, 1353, 799, 1145, 799, 1354, 800, 1141, 800, 1355, 802, 1137, 802, 1356, 802, 1135, 802, 1357, 804, 1131, 804, 1358, 804, 1129, 804, 1359, 806, 1125, 806, 1360, 806, 1123, 806, 1361, 807, 1121, 807, 1362, 807, 1119, 807, 1364, 807, 1117, 807, 1365, 809, 1113, 809, 1366, 809, 1111, 809, 1368, 809, 1109, 809, 1369, 810, 1107, 810, 1370, 810, 1105, 810, 1372, 809, 1105, 809, 1373, 810, 1103, 810, 1373, 811, 1101, 811, 1373, 812, 1099, 812, 1373, 813, 1097, 813, 1373, 814, 1095, 814, 1373, 814, 1095, 814, 1373, 815, 1093, 815, 1373, 816, 1091, 816, 1373, 816, 1091, 816, 1373, 817, 1089, 817, 1373, 818, 1087, 818, 1374, 817, 1087, 817, 1375, 818, 1085, 818, 1375, 818, 1085, 818, 1375, 819, 1083, 819, 1375, 819, 1083, 819, 1375, 820, 1081, 820, 1376, 819, 1081, 819, 1377, 820, 1079, 820, 1377, 820, 1079, 820, 1377, 821, 1077, 821, 1378, 820, 1077, 820, 1379, 820, 1077, 820, 1379, 821, 1075, 821, 1380, 820, 1075, 820, 1381, 820, 1075, 820, 1381, 821, 1073, 821, 1382, 820, 1073, 820, 1383, 820, 1073, 820, 1383, 821, 1071, 821, 1384, 820, 1071, 820, 1385, 820, 1071, 820, 1386, 819, 1071, 819, 1387, 820, 1069, 820, 1388, 819, 1069, 819, 1389, 819, 1069, 819, 1389, 819, 1069, 819, 1390, 818, 1069, 818, 1391, 818, 1069, 818, 1391, 819, 1067, 819, 1392, 818, 1067, 818, 1393, 818, 1067, 818, 1393, 818, 1067, 818, 1394, 817, 1067, 817, 1395, 817, 1067, 817, 1396, 816, 1067, 816, 1397, 816, 1067, 816, 1398, 815, 1067, 815, 1399, 815, 1067, 815, 1400, 814, 1067, 814, 1401, 814, 1067, 814, 1402, 813, 1067, 813, 1403, 813, 1067, 813, 1404, 813, 1065, 813, 1406, 812, 1065, 812, 1407, 812, 1065, 812, 1408, 811, 1065, 811, 1409, 811, 1065, 811, 1409, 811, 1065, 811, 1409, 811, 1065, 811, 1409, 811, 1065, 811, 1409, 811, 1065, 811, 1409, 811, 1065, 811, 1409, 812, 1063, 812, 1409, 812, 1063, 812, 1410, 812, 1061, 812, 1411, 812, 1061, 812, 1411, 813, 1059, 813, 1411, 813, 1059, 813, 1411, 813, 1059, 813, 1411, 814, 1057, 814, 1412, 813, 1057, 813, 1413, 813, 1057, 813, 1413, 814, 1055, 814, 1413, 814, 1055, 814, 1414, 813, 1055, 813, 1415, 814, 1053, 814, 1415, 814, 1053, 814, 1415, 814, 1053, 814, 1416, 813, 1053, 813, 1417, 814, 1051, 814, 1417, 814, 1051, 814, 1417, 814, 1051, 814, 1417, 814, 1051, 814, 1417, 814, 1051, 814, 1418, 813, 1051, 813, 1419, 814, 1049, 814, 1419, 814, 1049, 814, 1419, 814, 1049, 814, 1420, 813, 1049, 813, 1421, 813, 1049, 813, 1421, 813, 1049, 813, 1422, 812, 1049, 812, 1423, 812, 1049, 812, 1423, 812, 1049, 812, 1424, 811, 1049, 811, 1425, 811, 1049, 811, 1425, 811, 1049, 811, 1426, 810, 1049, 810, 1427, 810, 1049, 810, 1428, 809, 1049, 809, 1429, 809, 1049, 809, 1430, 808, 1049, 808, 1431, 808, 1049, 808, 1432, 807, 1049, 807, 1433, 806, 1051, 806, 1434, 805, 1051, 805, 1435, 805, 1051, 805, 1436, 804, 1051, 804, 1437, 804, 1051, 804, 1438, 803, 1051, 803, 1439, 802, 1053, 802, 1440, 801, 1053, 801, 1441, 801, 1053, 801, 1442, 800, 1053, 800, 1443, 799, 1055, 799, 1444, 798, 1055, 798, 1445, 798, 1055, 798, 1446, 796, 1057, 796, 1447, 796, 1057, 796, 1448, 795, 1057, 795, 1450, 793, 1059, 793, 1451, 793, 1059, 793, 1452, 792, 1059, 792, 1454, 790, 1061, 790, 1455, 790, 1061, 790, 1456, 788, 1063, 788, 1458, 787, 1063, 787, 1460, 785, 1065, 785, 1462, 784, 1065, 784, 1464, 782, 1067, 782, 1465, 782, 1067, 782, 1466, 780, 1069, 780, 1468, 779, 1069, 779, 1470, 777, 1071, 777, 1472, 775, 1073, 775, 1475, 773, 1073, 773, 1478, 771, 1075, 771, 1480, 769, 1077, 769, 1482, 768, 1077, 768, 1484, 766, 1079, 766, 1485, 765, 1081, 765, 1486, 763, 1083, 763, 1488, 761, 1085, 761, 1490, 760, 1085, 760, 1492, 758, 1087, 758, 1495, 755, 1089, 755, 1498, 754, 1089, 754, 1500, 752, 1091, 752, 1502, 750, 1093, 750, 1504, 749, 1093, 749, 1507, 746, 1095, 746, 1510, 744, 1097, 744, 1513, 741, 1099, 741, 1516, 739, 1101, 739, 1519, 736, 1103, 736, 1523, 734, 1103, 734, 1527, 731, 1105, 731, 1531, 728, 1107, 728, 1535, 725, 1109, 725, 1540, 721, 1111, 721, 1545, 717, 1115, 717, 1549, 714, 1117, 714, 1553, 712, 1117, 712, 1556, 710, 1119, 710, 1559, 708, 1119, 708, 1562, 706, 1121, 706, 1565, 704, 1121, 704, 1568, 702, 1123, 702, 1571, 699, 1125, 699, 1575, 697, 1125, 697, 1579, 694, 1127, 694, 1583, 691, 1129, 691, 1587, 689, 1129, 689, 1592, 685, 1131, 685, 1598, 681, 1133, 681, 1604, 677, 1135, 677, 1609, 674, 1137, 674, 1613, 671, 1139, 671, 1618, 668, 1139, 668, 1624, 664, 1141, 664, 1630, 660, 1143, 660, 1637, 655, 1145, 655, 1645, 650, 1147, 650, 1652, 647, 1147, 647, 1658, 643, 1149, 643, 1665, 639, 1149, 639, 1675, 632, 1151, 632, 1685, 628, 1151, 628, 1693, 623, 1153, 623, 1703, 617, 1153, 617, 1714, 611, 1155, 611, 1723, 606, 1157, 606, 1733, 600, 1157, 600, 1743, 595, 1159, 595, 1751, 590, 1161, 590, 1759, 586, 1161, 586, 1766, 582, 1163, 582, 1772, 578, 1165, 578, 1779, 573, 1167, 573, 1789, 566, 1169, 566, 1803, 557, 1171, 557, 1817, 551, 1171, 551, 1831, 542, 1173, 542, 1845, 535, 1175, 535, 1860, 525, 1177, 525, 1875, 518, 1179, 518, 1889, 508, 1183, 508, 1903, 501, 1185, 501, 1932, 477, 1187, 477, 1961, 470, 1189, 470, 1975, 461, 1191, 461, 1989, 453, 1195, 453, 2011, 436, 1197, 436, 2030, 432, 1199, 432, 2037, 426, 1203, 426, 2047, 419, 1205, 419, 2120, 351, 1207, 351, 2193, 344, 1209, 344, 2222, 320, 1211, 320, 2251, 312, 1215, 312, 2421, 147, 1217, 147, 2589, 141, 1221, 141, 2599, 134, 1223, 134, 2609, 128, 1227, 128, 2617, 122, 1231, 122, 2625, 116, 1235, 116, 2632, 111, 1239, 111, 2638, 106, 1243, 106, 2645, 99, 1249, 99, 2653, 92, 1255, 92, 2660, 86, 1261, 86, 2666, 79, 1269, 79, 2673, 69, 1281, 69, 2683, 49, 1309, 49, 14443846] , dtype=np.uint32)
MOUTH_CAVITY_RLE = np.array( [99947, 57, 4034, 67, 22, 29, 3973, 76, 13, 43, 3960, 85, 4, 52, 3951, 149, 3945, 156, 3938, 162, 537, 174, 3221, 169, 527, 184, 3213, 177, 517, 194, 3206, 183, 508, 203, 3192, 5, 3, 189, 500, 212, 3182, 207, 490, 221, 3176, 218, 477, 230, 3166, 228, 467, 239, 3157, 235, 460, 249, 3150, 248, 445, 257, 3144, 255, 435, 264, 3105, 297, 426, 271, 3092, 312, 416, 278, 3085, 321, 408, 284, 3079, 329, 395, 297, 3070, 339, 385, 304, 3066, 350, 374, 309, 3061, 357, 362, 318, 3057, 364, 352, 325, 3053, 368, 345, 335, 3046, 372, 339, 341, 3041, 377, 333, 347, 3037, 381, 329, 351, 3033, 385, 324, 356, 3029, 389, 318, 362, 3025, 396, 311, 366, 3019, 402, 306, 374, 3012, 406, 300, 380, 3008, 410, 296, 384, 3004, 414, 291, 389, 3001, 420, 282, 395, 2997, 424, 278, 402, 2990, 428, 274, 406, 2986, 432, 270, 410, 2982, 436, 265, 418, 2974, 441, 261, 422, 2970, 445, 257, 426, 2966, 449, 251, 432, 2962, 453, 247, 436, 2958, 460, 239, 441, 2952, 467, 234, 445, 2948, 474, 227, 449, 2944, 480, 221, 454, 2939, 487, 214, 458, 2935, 494, 205, 464, 2931, 498, 201, 468, 2928, 501, 196, 473, 2924, 505, 192, 480, 2918, 509, 187, 484, 2915, 514, 181, 488, 2913, 516, 177, 492, 2910, 519, 173, 496, 2907, 525, 166, 500, 2904, 528, 162, 504, 2900, 532, 158, 508, 2896, 536, 154, 511, 2893, 540, 149, 516, 2890, 543, 143, 521, 2887, 548, 138, 524, 2885, 551, 133, 529, 2882, 554, 129, 532, 2881, 556, 125, 535, 2879, 559, 121, 537, 2878, 565, 114, 540, 2877, 567, 110, 543, 2875, 570, 106, 545, 2875, 575, 99, 548, 2873, 578, 95, 550, 2873, 580, 91, 553, 2871, 583, 87, 555, 2871, 585, 83, 558, 2870, 587, 78, 561, 2870, 589, 74, 563, 2870, 591, 70, 566, 2868, 595, 65, 568, 2868, 597, 61, 570, 2868, 599, 57, 572, 2868, 601, 53, 574, 2868, 602, 50, 577, 2867, 604, 46, 579, 2867, 605, 43, 581, 2867, 606, 40, 583, 2867, 607, 37, 585, 2867, 608, 33, 587, 2869, 608, 30, 589, 2869, 610, 25, 592, 2869, 611, 19, 597, 2869, 612, 16, 599, 2869, 613, 12, 601, 2871, 613, 9, 603, 2871, 614, 6, 604, 2872, 614, 4, 606, 2872, 615, 2, 607, 2872, 1223, 2873, 1223, 2873, 1222, 2874, 1222, 2874, 1221, 2876, 1220, 2876, 1219, 2877, 1219, 2877, 1218, 2878, 1218, 2879, 1217, 2879, 1217, 2880, 1215, 2881, 1215, 2882, 1214, 2882, 1214, 2882, 1214, 2882, 1213, 2884, 1212, 2884, 1212, 2884, 1212, 2884, 1211, 2885, 1211, 2886, 1210, 2886, 1210, 2886, 1210, 2886, 1209, 2888, 1208, 2888, 1208, 2889, 1207, 2889, 1206, 2891, 1205, 2891, 1205, 2892, 1204, 2892, 1204, 2892, 1203, 2894, 1202, 2894, 1201, 2896, 1200, 2896, 1200, 2897, 1198, 2898, 1198, 2899, 1197, 2900, 1196, 2900, 1195, 2902, 1194, 2903, 1192, 2905, 1191, 2906, 1189, 2908, 1188, 2908, 1187, 2909, 1187, 2909, 1187, 2909, 1186, 2910, 1186, 2910, 1185, 2911, 1185, 2911, 1184, 2912, 1184, 2912, 1184, 2912, 1184, 2912, 1183, 2913, 1183, 2914, 1181, 2915, 1181, 2915, 1181, 2915, 1180, 2916, 1180, 2916, 1180, 2917, 1179, 2917, 1178, 2918, 1178, 2918, 1177, 2920, 1176, 2920, 1175, 2921, 1175, 2922, 1173, 2923, 1173, 2923, 1173, 2924, 1171, 2925, 1171, 2925, 1170, 2927, 1169, 2927, 1168, 2929, 1167, 2929, 1167, 2930, 1166, 2930, 1165, 2932, 1164, 2932, 1163, 2934, 1162, 2934, 1162, 2935, 1160, 2937, 1159, 2937, 1158, 2939, 1157, 2940, 1155, 2941, 1155, 2942, 1154, 2943, 1153, 2944, 1151, 2946, 1150, 2947, 1148, 2948, 1148, 2949, 1147, 2950, 1145, 2952, 1144, 2953, 1142, 2956, 1140, 2957, 1138, 2958, 1138, 2958, 1137, 2959, 1137, 2959, 1136, 2960, 1136, 2960, 1135, 2961, 1135, 2961, 1134, 2962, 1134, 2962, 1134, 2962, 1134, 2962, 1134, 2962, 1133, 2963, 1133, 2963, 1132, 2964, 1132, 2964, 1131, 2965, 1131, 2965, 1130, 2966, 1130, 2966, 1129, 2967, 1129, 2967, 1128, 2968, 1128, 2968, 1127, 2969, 1127, 2969, 1127, 2969, 1127, 2970, 1126, 2970, 1125, 2971, 1125, 2971, 1124, 2972, 1124, 2973, 1122, 2974, 1122, 2975, 1121, 2975, 1121, 2976, 1120, 2976, 1119, 2978, 1118, 2979, 1116, 2980, 1116, 2981, 1114, 2983, 1113, 2984, 1111, 2985, 1111, 2986, 1109, 2988, 1108, 2989, 1106, 2991, 1105, 2992, 1104, 2993, 1102, 2995, 1101, 2996, 1100, 2996, 1100, 2997, 1098, 2998, 1098, 2999, 1096, 3001, 1095, 3001, 1094, 3003, 1093, 3004, 1091, 3006, 1090, 3006, 1090, 3007, 1088, 3009, 1087, 3009, 1086, 3010, 1086, 3010, 1085, 3011, 1085, 3011, 1085, 3012, 1084, 3012, 1083, 3013, 1083, 3013, 1082, 3014, 1082, 3015, 1080, 3016, 1080, 3016, 1079, 3017, 1079, 3018, 1078, 3018, 1077, 3019, 1077, 3019, 1077, 3019, 1077, 3020, 1075, 3021, 1075, 3022, 1073, 3023, 1073, 3024, 1071, 3025, 1071, 3025, 1070, 3026, 1069, 3027, 1069, 3028, 1067, 3029, 1066, 3031, 1065, 3031, 1064, 3033, 1063, 3033, 1062, 3034, 1061, 3035, 1061, 3035, 1060, 3037, 1059, 3037, 1059, 3037, 1059, 3037, 1058, 3039, 1057, 3039, 1056, 3041, 1055, 3041, 1054, 3043, 1053, 3043, 1052, 3045, 1051, 3045, 1051, 3045, 1050, 3047, 1049, 3047, 1048, 3049, 1047, 3049, 1046, 3051, 1045, 3051, 1044, 3053, 1043, 3054, 1041, 3055, 1041, 3056, 1039, 3057, 1039, 3057, 1038, 3059, 1037, 3059, 1036, 3061, 1035, 3061, 1034, 3063, 1033, 3063, 1032, 3065, 1031, 3065, 1030, 3067, 1029, 3067, 1028, 3069, 1027, 3069, 1027, 3069, 1026, 3071, 1025, 3071, 1024, 3072, 1024, 3072, 1023, 3074, 1022, 3074, 1022, 3074, 1022, 3074, 1021, 3075, 1021, 3076, 407, 3, 609, 3077, 406, 4, 609, 3077, 405, 6, 607, 3078, 408, 3, 607, 3078, 1018, 3078, 1018, 3078, 1018, 3078, 1017, 3079, 1017, 3079, 1016, 3080, 1016, 3080, 1015, 3081, 1015, 3081, 1014, 3082, 1013, 3083, 1013, 3083, 1013, 3082, 1014, 3082, 1014, 3082, 1014, 3082, 1015, 3081, 1015, 3080, 1016, 3080, 1016, 3079, 1018, 3078, 1018, 3077, 1020, 3076, 1020, 3076, 1021, 3075, 1021, 3074, 1023, 3073, 1023, 3073, 1024, 3072, 1024, 3072, 1024, 3071, 1025, 3071, 1025, 3071, 1026, 3070, 1026, 3069, 1027, 3069, 1027, 3069, 1027, 3068, 1029, 3067, 1029, 3066, 1030, 3066, 1030, 3065, 1032, 3064, 1032, 3063, 1034, 3062, 1034, 3061, 1036, 3060, 1036, 3059, 1037, 3059, 1037, 3058, 1038, 3058, 1039, 3056, 1040, 3056, 1041, 3055, 1041, 3055, 1042, 3053, 1043, 3053, 1044, 3052, 1044, 3052, 1045, 3051, 1045, 3050, 1047, 3049, 1047, 3049, 1048, 3048, 1048, 3047, 1049, 3047, 1049, 3047, 1049, 3047, 1050, 3046, 1050, 3045, 1051, 3045, 1051, 3044, 1052, 3044, 1052, 3043, 1053, 3043, 1053, 3043, 1053, 3043, 1054, 3041, 1055, 3041, 1055, 3041, 1055, 3041, 1055, 3041, 1056, 3039, 1057, 3039, 1058, 3037, 1059, 3037, 1060, 3035, 1061, 3035, 1062, 3034, 1064, 3032, 1066, 3029, 1068, 3028, 1070, 3026, 1071, 3025, 1072, 3024, 1072, 3023, 1074, 3022, 1075, 3021, 1075, 3021, 1076, 3019, 1077, 3019, 1078, 3018, 1078, 3018, 1079, 3017, 1079, 3016, 1080, 3016, 1080, 3015, 1081, 3015, 1082, 3013, 1083, 3013, 1083, 3013, 1083, 3013, 1084, 3012, 1084, 3011, 1086, 3009, 1087, 3008, 1089, 3007, 1089, 3006, 1091, 3004, 1092, 3003, 1094, 3002, 1094, 3001, 1096, 2999, 1098, 2997, 1099, 2997, 1100, 2995, 1102, 2993, 1103, 2993, 1104, 2991, 1105, 2991, 1106, 2989, 1107, 2989, 1108, 2987, 1109, 2987, 1110, 2986, 1110, 2986, 1111, 2984, 1112, 2984, 1113, 2983, 1113, 2982, 1115, 2981, 1115, 2980, 1117, 2979, 1117, 2979, 1118, 2978, 1118, 2978, 1118, 2977, 1120, 2976, 1121, 2975, 1121, 2975, 1122, 2974, 1123, 2974, 1122, 2974, 1123, 2973, 1123, 2973, 1124, 2971, 1125, 2970, 1127, 2969, 1127, 2968, 1129, 2967, 1130, 2965, 1131, 2965, 1132, 2963, 1134, 2962, 1134, 2961, 1136, 2960, 1136, 2959, 1138, 2958, 1139, 2956, 1140, 2956, 1141, 2954, 1143, 2953, 1143, 2953, 1144, 2951, 1145, 2951, 1146, 2949, 1147, 2948, 1149, 2946, 1150, 2946, 1151, 2944, 1153, 2942, 1154, 2942, 1155, 2940, 1157, 2939, 1157, 2938, 1159, 2937, 1159, 2936, 1161, 2935, 1161, 2935, 1162, 2933, 1163, 2933, 1164, 2931, 1166, 2930, 1166, 2929, 1168, 2928, 1169, 2926, 1171, 2925, 1171, 2924, 1173, 2923, 1174, 2921, 1175, 2921, 1176, 2919, 1177, 2919, 1178, 2917, 1179, 2917, 1180, 2915, 1181, 2915, 1182, 2913, 1183, 2913, 1184, 2912, 1184, 2912, 1185, 2911, 1186, 2909, 1187, 2909, 1188, 2908, 1188, 2908, 1189, 2906, 1190, 2906, 1191, 2904, 1192, 2904, 1193, 2902, 1194, 2902, 1195, 2901, 1195, 2901, 1195, 2901, 1195, 2900, 1197, 2899, 1197, 2899, 1197, 2899, 1198, 2897, 1199, 2897, 1200, 2896, 1200, 2896, 1200, 2896, 1200, 2895, 1202, 2894, 1202, 2894, 1203, 2893, 1203, 2893, 1204, 2891, 1205, 2891, 1206, 2890, 1206, 2890, 1207, 2889, 1207, 2889, 1207, 2889, 1207, 2889, 1207, 2889, 1208, 2888, 1208, 2888, 1208, 2888, 1208, 2888, 1208, 2888, 1207, 2890, 1206, 2890, 1206, 2890, 1206, 2890, 1206, 2890, 1205, 2892, 1204, 2892, 1203, 2894, 1202, 2894, 1202, 2895, 1200, 2896, 1200, 2897, 1198, 2898, 1198, 2899, 1196, 2900, 1196, 2901, 1194, 2903, 1192, 2904, 1192, 2905, 1190, 2907, 1188, 2909, 1185, 2912, 1183, 2914, 1180, 2918, 1177, 2920, 1175, 2923, 594, 34, 545, 2925, 570, 58, 542, 2928, 564, 65, 539, 2933, 554, 72, 536, 2939, 534, 91, 531, 2944, 512, 112, 528, 2946, 508, 116, 525, 2948, 505, 122, 520, 2951, 501, 128, 514, 2954, 498, 135, 508, 2957, 494, 164, 479, 2959, 492, 169, 474, 2962, 488, 174, 470, 2965, 486, 179, 464, 2968, 485, 181, 460, 2971, 483, 184, 456, 2974, 481, 188, 451, 2978, 477, 192, 447, 2981, 475, 195, 443, 2985, 471, 199, 436, 2991, 468, 203, 432, 2994, 465, 207, 428, 2998, 461, 214, 418, 3004, 458, 220, 410, 3010, 451, 230, 403, 3014, 445, 236, 399, 3018, 438, 243, 395, 3024, 424, 255, 391, 3028, 385, 294, 386, 3033, 369, 310, 382, 3040, 350, 326, 376, 3048, 333, 342, 369, 3054, 326, 349, 362, 3061, 320, 357, 356, 3068, 310, 365, 351, 3088, 282, 377, 344, 3096, 270, 390, 336, 3102, 260, 401, 328, 3112, 250, 408, 321, 3126, 227, 426, 313, 3135, 218, 435, 303, 3143, 210, 444, 295, 3152, 203, 448, 291, 3161, 194, 452, 287, 3165, 187, 462, 277, 3174, 155, 499, 263, 3184, 145, 506, 256, 3198, 127, 517, 245, 3212, 113, 531, 235, 3231, 89, 550, 217, 3257, 54, 573, 196, 3274, 44, 587, 188, 3279, 37, 596, 182, 3283, 31, 605, 173, 3289, 24, 612, 166, 3299, 14, 619, 160, 3941, 150, 3950, 142, 3959, 132, 3973, 118, 13805176] , dtype=np.uint32)
TOON_MOUTH_RLE = np.array( [552224, 19, 999, 31, 989, 39, 982, 45, 976, 51, 970, 57, 953, 73, 945, 81, 939, 87, 934, 95, 926, 104, 916, 112, 908, 119, 902, 125, 896, 131, 890, 151, 871, 159, 862, 166, 855, 172, 846, 181, 839, 188, 833, 193, 828, 198, 822, 204, 817, 209, 812, 214, 807, 218, 804, 222, 800, 225, 797, 229, 793, 232, 790, 235, 786, 239, 781, 244, 777, 249, 772, 253, 768, 257, 765, 260, 762, 263, 759, 266, 756, 270, 752, 274, 749, 276, 746, 280, 743, 282, 740, 286, 737, 288, 735, 290, 733, 292, 730, 295, 728, 298, 724, 301, 722, 303, 719, 306, 717, 309, 714, 312, 711, 315, 708, 317, 705, 321, 702, 323, 700, 326, 697, 328, 694, 331, 692, 333, 690, 335, 688, 338, 685, 340, 682, 343, 234, 19, 13, 19, 395, 352, 220, 31, 1, 31, 388, 359, 206, 75, 383, 369, 191, 84, 379, 384, 173, 91, 376, 390, 164, 97, 372, 395, 157, 102, 369, 400, 149, 108, 366, 407, 110, 19, 10, 144, 333, 412, 62, 69, 1, 153, 326, 418, 51, 233, 322, 424, 22, 259, 318, 429, 12, 268, 314, 434, 4, 275, 311, 715, 308, 718, 305, 721, 303, 723, 300, 726, 298, 727, 296, 730, 294, 731, 292, 734, 290, 735, 288, 737, 287, 738, 285, 740, 284, 742, 282, 743, 280, 745, 279, 746, 278, 747, 276, 748, 276, 749, 275, 750, 273, 752, 272, 753, 271, 754, 270, 754, 269, 756, 268, 757, 267, 757, 267, 758, 266, 759, 265, 759, 264, 761, 263, 761, 263, 762, 262, 762, 262, 763, 261, 763, 261, 764, 260, 764, 260, 765, 259, 765, 259, 765, 259, 766, 258, 766, 258, 766, 258, 767, 257, 767, 257, 767, 257, 768, 256, 768, 256, 768, 256, 770, 254, 772, 252, 773, 251, 775, 249, 776, 248, 778, 246, 779, 245, 780, 244, 781, 243, 782, 242, 784, 240, 785, 239, 786, 238, 787, 237, 789, 235, 790, 234, 791, 233, 792, 232, 793, 231, 795, 229, 796, 228, 797, 227, 798, 226, 799, 225, 799, 225, 800, 224, 801, 223, 802, 222, 803, 221, 804, 220, 804, 220, 805, 219, 806, 218, 806, 218, 807, 217, 808, 216, 808, 216, 809, 215, 810, 214, 811, 213, 812, 212, 812, 212, 813, 211, 814, 210, 814, 210, 815, 209, 816, 208, 816, 208, 817, 207, 817, 207, 818, 206, 818, 206, 819, 205, 819, 205, 820, 204, 820, 204, 821, 203, 821, 203, 821, 203, 822, 202, 822, 202, 822, 202, 823, 201, 823, 201, 823, 201, 824, 200, 824, 200, 824, 200, 824, 200, 825, 199, 825, 199, 825, 199, 825, 199, 825, 199, 825, 199, 826, 198, 826, 198, 826, 198, 826, 198, 826, 198, 826, 198, 826, 198, 826, 198, 826, 198, 826, 198, 826, 198, 826, 198, 826, 198, 826, 198, 826, 198, 826, 198, 826, 198, 826, 198, 826, 198, 826, 198, 826, 198, 826, 198, 826, 198, 826, 198, 826, 198, 826, 198, 826, 198, 826, 198, 826, 198, 826, 198, 826, 198, 826, 198, 826, 198, 826, 198, 826, 198, 826, 198, 826, 198, 826, 198, 826, 198, 826, 198, 826, 199, 825, 199, 825, 199, 825, 199, 825, 199, 825, 199, 825, 200, 824, 200, 824, 200, 824, 200, 824, 201, 823, 201, 823, 201, 823, 201, 823, 201, 823, 201, 823, 201, 823, 201, 823, 201, 823, 201, 823, 201, 823, 201, 822, 202, 822, 202, 822, 202, 822, 202, 822, 202, 822, 202, 821, 203, 821, 203, 821, 203, 821, 203, 821, 203, 821, 203, 821, 203, 821, 203, 821, 203, 820, 204, 820, 204, 820, 204, 820, 204, 820, 204, 820, 204, 821, 203, 821, 203, 821, 203, 821, 203, 822, 202, 822, 202, 822, 202, 821, 203, 821, 203, 821, 203, 821, 203, 821, 203, 821, 203, 820, 204, 820, 204, 820, 204, 820, 204, 820, 204, 820, 203, 821, 203, 821, 203, 821, 203, 821, 203, 821, 203, 821, 202, 822, 202, 822, 202, 821, 203, 821, 203, 821, 203, 821, 203, 821, 203, 821, 203, 821, 203, 821, 203, 821, 203, 820, 204, 820, 204, 820, 204, 820, 204, 820, 204, 820, 204, 819, 205, 819, 205, 819, 205, 819, 205, 819, 205, 819, 205, 819, 205, 819, 205, 819, 205, 819, 205, 819, 205, 819, 205, 819, 205, 819, 205, 819, 205, 819, 205, 819, 205, 819, 205, 819, 205, 820, 204, 820, 204, 820, 204, 820, 204, 820, 204, 820, 204, 821, 203, 821, 203, 821, 203, 821, 203, 822, 202, 822, 202, 821, 203, 821, 203, 821, 203, 820, 204, 820, 204, 820, 204, 820, 204, 819, 205, 819, 205, 819, 205, 819, 205, 819, 205, 819, 205, 818, 206, 818, 206, 818, 206, 818, 206, 818, 206, 818, 206, 818, 206, 818, 206, 818, 206, 818, 206, 818, 206, 818, 206, 818, 206, 818, 206, 818, 206, 818, 206, 818, 206, 818, 206, 818, 206, 818, 206, 818, 206, 818, 206, 818, 206, 818, 206, 818, 206, 818, 206, 818, 206, 818, 206, 818, 206, 818, 206, 818, 206, 818, 206, 818, 206, 818, 206, 818, 206, 818, 206, 818, 206, 818, 206, 818, 206, 818, 206, 818, 205, 819, 205, 819, 205, 819, 205, 819, 205, 819, 205, 819, 205, 819, 205, 819, 205, 819, 205, 819, 205, 819, 205, 819, 205, 819, 205, 819, 205, 819, 205, 819, 205, 819, 205, 819, 205, 819, 206, 818, 206, 818, 206, 819, 205, 819, 205, 819, 205, 819, 206, 818, 206, 818, 206, 818, 206, 818, 207, 817, 207, 817, 207, 817, 208, 816, 208, 816, 208, 816, 209, 815, 209, 816, 208, 816, 209, 815, 209, 815, 210, 814, 210, 814, 210, 815, 209, 815, 209, 815, 209, 815, 209, 815, 209, 815, 209, 815, 209, 815, 209, 816, 208, 816, 208, 816, 208, 816, 208, 816, 209, 815, 209, 816, 208, 816, 208, 816, 208, 816, 208, 817, 208, 816, 208, 816, 208, 817, 207, 817, 208, 816, 208, 817, 207, 817, 105] , dtype=np.uint32)

TRANSPARENT_TOON_EYEBROWS_MAT_NAME = "ToonEyebrows"
TRANSPARENT_TOON_EYELASHES_MAT_NAME = "ToonEyelashes"

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


def camel_case_to_spaces(text:str)->str:
    import re
    max_length = 0
    longest_part = text
    for part in text.split('_'):
        if len(part) > max_length:
            longest_part = part
            max_length = len(part)
    text = longest_part
    return re.sub(r'((?<=[a-z])[A-Z0-9]|(?<!\A)[A-Z](?=[a-z]))', r' \1', text)

FIGURE_TOON = "T"
FIGURE_ANY = "TG"
FIGURE_G9 = "G"
MorphMeta = namedtuple('MorphMeta', ['category', 'title', 'figure', 'usage_profiles'])

CAT_BREAST = "Custom/Breasts"
CAT_HEAD = "Custom/Head"
CAT_ARMS = "Custom/Arms"
CAT_LEGS = "Custom/Legs"
CAT_BODY = "Custom/Body"
CAT_ASS = "Custom/Ass"
CAT_GENITALS = "Custom/Genitals"
CAT_SPECIAL = "Custom/Special"
CAT_FACS = "Facs"
CAT_FACSEXPR = "Facsexpr"
CAT_FACSDET = "Facsdetails"
CAT_JCM = "JCM"
CATS_BODY = {CAT_BREAST,CAT_HEAD,CAT_ARMS,CAT_LEGS,CAT_BODY,CAT_ASS,CAT_GENITALS,CAT_SPECIAL}
CATS_FACS = {CAT_FACS, CAT_FACSDET, CAT_FACSEXPR}
NIRV_ZERO_EYES_DAZ_DIR = "/data/nirvana/nirv zero/nirv zero eyes"
PROFILE_FULL = "F"
PROFILE_MIN = "M"
PROFILE_MID = "D"
MATCH_PROFILE_MIN = PROFILE_MIN
MATCH_PROFILE_MID = MATCH_PROFILE_MIN + PROFILE_MID
MATCH_PROFILE_FULL = MATCH_PROFILE_MID + PROFILE_FULL
USED_PROFILES = MATCH_PROFILE_MID
GENERATE_MORPHS_FOR_CLOTHES = True
GENERATE_MORPHS_FOR_HAIR = True
MORPHS = {
    "/data/rudy studio/wtkt toon/wtkt toon":{
        "shapes": { "female":{
                        'OPEN 02': MorphMeta(CAT_SPECIAL,"open 02",FIGURE_TOON, PROFILE_FULL),
            'OPEN 03': MorphMeta(CAT_SPECIAL,"open 03",FIGURE_TOON, PROFILE_FULL),
            'OPEN 04': MorphMeta(CAT_SPECIAL,"open 04",FIGURE_TOON, PROFILE_FULL),
            'OPEN EXTREME MORE': MorphMeta(CAT_SPECIAL,"open extreme more",FIGURE_TOON, PROFILE_FULL),
            'OPEN EXTREME': MorphMeta(CAT_SPECIAL,"open extreme",FIGURE_TOON, PROFILE_FULL),
            'FULL OPEN 01':MorphMeta(CAT_SPECIAL,"full open 01",FIGURE_TOON, PROFILE_FULL),
            'FULL OPEN 02':MorphMeta(CAT_SPECIAL,"full open 02",FIGURE_TOON, PROFILE_MIN),
            'FULL OPEN 03':MorphMeta(CAT_SPECIAL,"full open 03",FIGURE_TOON, PROFILE_FULL),
            'URETHRA OPEN': MorphMeta(CAT_SPECIAL,"urethra open",FIGURE_TOON, PROFILE_FULL),
            'OPEN 01': MorphMeta(CAT_SPECIAL,"open 01",FIGURE_TOON, PROFILE_FULL),
            'CLOSED 02':MorphMeta(CAT_SPECIAL,"closed 02",FIGURE_TOON, PROFILE_FULL),
            'CLOSED 03':MorphMeta(CAT_SPECIAL,"closed 03",FIGURE_TOON, PROFILE_FULL),
            'CLOSED 04':MorphMeta(CAT_SPECIAL,"closed 04",FIGURE_TOON, PROFILE_FULL),
            'CLOSED 01':MorphMeta(CAT_SPECIAL,"closed 01",FIGURE_TOON, PROFILE_FULL),
            'FBM CAMEL TOE':MorphMeta(CAT_GENITALS,"fbm camel toe",FIGURE_TOON, PROFILE_MIN),
            'FULL IN OUT':MorphMeta(CAT_GENITALS,"full in out",FIGURE_TOON, PROFILE_MIN),
            'PERINEUM': MorphMeta(CAT_GENITALS,"perineum",FIGURE_TOON, PROFILE_FULL),
            'REN_02 corr': MorphMeta(CAT_GENITALS,"ren_02 corr",FIGURE_TOON, PROFILE_FULL),
            'URETHRA IN OUT': MorphMeta(CAT_GENITALS,"urethra in out",FIGURE_TOON, PROFILE_FULL),
            'CAMEL TOE':MorphMeta(CAT_GENITALS,"camel to",FIGURE_TOON, PROFILE_FULL),
            'MONS PUBIS BIG':MorphMeta(CAT_GENITALS,"mons pubis big",FIGURE_TOON, PROFILE_MIN),
            'PERPUCE FRONT BACK': MorphMeta(CAT_GENITALS,"perpuce front back",FIGURE_TOON, PROFILE_FULL),
            'PERPUCE MOVE 2': MorphMeta(CAT_GENITALS,"perpuce move 2",FIGURE_TOON, PROFILE_FULL),
            'PERPUCE MOVE': MorphMeta(CAT_GENITALS,"perpuce move",FIGURE_TOON, PROFILE_FULL),
            'PERPUCE SHAPE': MorphMeta(CAT_GENITALS,"perpuce shape",FIGURE_TOON, PROFILE_FULL),
            'PERPUCE SIZE': MorphMeta(CAT_GENITALS,"perpuce size",FIGURE_TOON, PROFILE_FULL),
            'PERPUCE THICKNESS': MorphMeta(CAT_GENITALS,"perpuce thickness",FIGURE_TOON, PROFILE_FULL),
            'LABIA MAJORA OPEN BACK':MorphMeta(CAT_SPECIAL,"labia majora open back",FIGURE_TOON, PROFILE_FULL),
            'LABIA MAJORA BIG 02':MorphMeta(CAT_GENITALS,"labia majora big 02",FIGURE_TOON, PROFILE_FULL),
            'LABIA MAJORA BIG':MorphMeta(CAT_GENITALS,"labia majora big",FIGURE_TOON, PROFILE_MID),
            'LABIA MAJORA DEPTH':MorphMeta(CAT_GENITALS,"labia majora depth",FIGURE_TOON, PROFILE_FULL),
            'LABIA MAJORA SHAPE 01':MorphMeta(CAT_GENITALS,"labia majora shape 01",FIGURE_TOON, PROFILE_MID),
            'LABIA MAJORA SHAPE 02':MorphMeta(CAT_GENITALS,"labia majora shape 02",FIGURE_TOON, PROFILE_MID),
            'VAGINA OPEN 01': MorphMeta(CAT_SPECIAL,"vagina open 01",FIGURE_TOON, PROFILE_FULL),
            'VAGINA DIAMETER': MorphMeta(CAT_GENITALS,"vagina diameter",FIGURE_TOON, PROFILE_FULL),
            'VAGINA IN OUT': MorphMeta(CAT_GENITALS,"vagina in out",FIGURE_TOON, PROFILE_FULL),
            'VAGINA SHAPE 02': MorphMeta(CAT_GENITALS,"vagina shape 02",FIGURE_TOON, PROFILE_FULL),
            'VAGINA SHAPE': MorphMeta(CAT_GENITALS,"vagina shape",FIGURE_TOON, PROFILE_FULL),
            'ANUS OPEN MORE':MorphMeta(CAT_SPECIAL,"anus open more",FIGURE_TOON, PROFILE_FULL),
            'ANUS OPEN verse 2':MorphMeta(CAT_SPECIAL,"anus open verse 2",FIGURE_TOON, PROFILE_FULL),
            'ANUS OPEN verse':MorphMeta(CAT_SPECIAL,"anus open verse",FIGURE_TOON, PROFILE_FULL),
            'ANUS OPEN':MorphMeta(CAT_SPECIAL,"anus open",FIGURE_TOON, PROFILE_FULL),
            'ANUS 01':MorphMeta(CAT_GENITALS,"anus 01",FIGURE_TOON, PROFILE_FULL),
            'ANUS IN OUT 2':MorphMeta(CAT_GENITALS,"anus in out 2",FIGURE_TOON, PROFILE_FULL),
            'ANUS IN OUT ROTATED':MorphMeta(CAT_GENITALS,"anus in out rotate",FIGURE_TOON, PROFILE_FULL),
            'ANUS IN OUT':MorphMeta(CAT_GENITALS,"anus in out",FIGURE_TOON, PROFILE_FULL),
            'ANUS SHAPE':MorphMeta(CAT_GENITALS,"anus shape",FIGURE_TOON, PROFILE_FULL),
            'ANUS SIZE 2':MorphMeta(CAT_GENITALS,"anus size 2",FIGURE_TOON, PROFILE_FULL),
            'ANUS SIZE':MorphMeta(CAT_GENITALS,"anus size",FIGURE_TOON, PROFILE_FULL),
            'ANUS SIZE3':MorphMeta(CAT_GENITALS,"anus size",FIGURE_TOON, PROFILE_FULL),
            'ANUS STRETCH 2':MorphMeta(CAT_GENITALS,"anus stretch 2",FIGURE_TOON, PROFILE_FULL),
            'ANUS STRETCH':MorphMeta(CAT_GENITALS,"anus stretch",FIGURE_TOON, PROFILE_FULL),
            'LABIA MINORA OPEN 01':MorphMeta(CAT_SPECIAL,"labia minora open 01",FIGURE_TOON, PROFILE_FULL),
            'LABIA MINORA OPEN 02':MorphMeta(CAT_SPECIAL,"labia minora open 02",FIGURE_TOON, PROFILE_FULL),
            'LABIA MINORA OPEN LEFT 2':MorphMeta(CAT_SPECIAL,"labia minora open left 2",FIGURE_TOON, PROFILE_FULL),
            'LABIA MINORA OPEN LEFT':MorphMeta(CAT_SPECIAL,"labia minora open left",FIGURE_TOON, PROFILE_FULL),
            'LABIA MINORA OPEN RIGHT 2':MorphMeta(CAT_SPECIAL,"labia minora open right 2",FIGURE_TOON, PROFILE_FULL),
            'LABIA MINORA OPEN RIGHT':MorphMeta(CAT_SPECIAL,"labia minora open right",FIGURE_TOON, PROFILE_FULL),
            'LABIA MINORA OPEN':MorphMeta(CAT_SPECIAL,"labia minora ope",FIGURE_TOON, PROFILE_FULL),
            'LABIA MINORA FRONT CLOSED':MorphMeta(CAT_SPECIAL,"labia minora front close",FIGURE_TOON, PROFILE_FULL),
            'LABIA MINORA FRONT SMALL':MorphMeta(CAT_GENITALS,"labia minora front small",FIGURE_TOON, PROFILE_FULL),
            'LABIA MINORA IN OUT':MorphMeta(CAT_GENITALS,"labia minora in out",FIGURE_TOON, PROFILE_FULL),
            'LABIA MINORA LEFT LONG':MorphMeta(CAT_GENITALS,"labia minora left long",FIGURE_TOON, PROFILE_FULL),
            'LABIA MINORA RIGHT LONG':MorphMeta(CAT_GENITALS,"labia minora right long",FIGURE_TOON, PROFILE_FULL),
            'LABIA MINORA SHAPE 01':MorphMeta(CAT_GENITALS,"labia minora shape 01",FIGURE_TOON, PROFILE_MIN),
            'LABIA MINORA SHAPE 02':MorphMeta(CAT_GENITALS,"labia minora shape 02",FIGURE_TOON, PROFILE_MIN),
            'LABIA MINORA SHAPE 03':MorphMeta(CAT_GENITALS,"labia minora shape 03",FIGURE_TOON, PROFILE_MIN),
            'LABIA MINORA BACK  L SMALL':MorphMeta(CAT_GENITALS,"labia minora back l small",FIGURE_TOON, PROFILE_FULL),
            'LABIA MINORA SHAPE 04':MorphMeta(CAT_GENITALS,"labia minora shape 04",FIGURE_TOON, PROFILE_MIN),
            'LABIA MINORA BACK 01':MorphMeta(CAT_GENITALS,"labia minora back 01",FIGURE_TOON, PROFILE_FULL),
            'LABIA MINORA SHAPE 05':MorphMeta(CAT_GENITALS,"labia minora shape 05",FIGURE_TOON, PROFILE_MIN),
            'LABIA MINORA BACK IN OUT':MorphMeta(CAT_GENITALS,"labia minora back in out",FIGURE_TOON, PROFILE_FULL),
            'LABIA MINORA SHAPE 06':MorphMeta(CAT_GENITALS,"labia minora shape 06",FIGURE_TOON, PROFILE_MIN),
            'LABIA MINORA BACK R SMALL':MorphMeta(CAT_GENITALS,"labia minora back r small",FIGURE_TOON, PROFILE_FULL),
            'LABIA MINORA SMALL':MorphMeta(CAT_GENITALS,"labia minora small",FIGURE_TOON, PROFILE_FULL),
            'LABIA MINORA FRONT  L BIG':MorphMeta(CAT_GENITALS,"labia minora front l big",FIGURE_TOON, PROFILE_FULL),
            'LABIA MINORA THICKNES 01':MorphMeta(CAT_GENITALS,"labia minora thicknes 01",FIGURE_TOON, PROFILE_FULL),
            'LABIA MINORA FRONT  R BIG':MorphMeta(CAT_GENITALS,"labia minora front r big",FIGURE_TOON, PROFILE_FULL),
            'LABIA MINORA THIKNESS':MorphMeta(CAT_GENITALS,"labia minora thiknes",FIGURE_TOON, PROFILE_FULL),
            'LABIA MINORA FRONT BIG':MorphMeta(CAT_GENITALS,"labia minora front big",FIGURE_TOON, PROFILE_FULL),
            'CLITORIS FRONT BACK':MorphMeta(CAT_GENITALS,"clitoris front back",FIGURE_TOON, PROFILE_MID),
            'CLITORIS IN OUT':MorphMeta(CAT_GENITALS,"clitoris in out",FIGURE_TOON, PROFILE_MID),
            'CLITORIS MOVE':MorphMeta(CAT_GENITALS,"clitoris move",FIGURE_TOON, PROFILE_MID),
            'CLITORIS SIZE - BACK':MorphMeta(CAT_GENITALS,"clitoris size - back",FIGURE_TOON, PROFILE_FULL),
            'CLITORIS SIZE':MorphMeta(CAT_GENITALS,"clitoris size",FIGURE_TOON, PROFILE_FULL),
            'CLITORIS SMALL':MorphMeta(CAT_GENITALS,"clitoris size",FIGURE_TOON, PROFILE_MIN),
        }}
    },
    "/data/vyusur/body geo/body geo":{
        "shapes": { "female": {
            "Navel":MorphMeta(CAT_BREAST,"Navel", FIGURE_ANY, PROFILE_FULL),
            "Nipples":MorphMeta(CAT_BREAST,"Nipples", FIGURE_ANY, PROFILE_FULL),
            "Nipples2":MorphMeta(CAT_BREAST,"Nipples 2", FIGURE_ANY, PROFILE_MID),
            "Nipples3":MorphMeta(CAT_BREAST,"Nipples 3", FIGURE_ANY, PROFILE_FULL),
        }},
    },
    "/data/daz 3d/genesis 9/base": {
        "shapes": {
            "female": {
                "body_bs_AnimeBodyHeavyFeminine":MorphMeta(CAT_BODY, "Heavy", FIGURE_TOON, PROFILE_MIN),
                "body_bs_AnimeBodyMuscularFeminine":MorphMeta(CAT_BODY, "Muscular", FIGURE_TOON, PROFILE_MIN),
                "body_bs_BreastsLargeExtra":MorphMeta(CAT_BREAST, "Breasts Large Extra", FIGURE_TOON, PROFILE_FULL),
                "body_bs_BreastsPoint":MorphMeta(CAT_BREAST, "Breasts Point", FIGURE_TOON, PROFILE_FULL),
                "body_bs_BreastsTipDown":MorphMeta(CAT_BREAST, "Breasts Tip Down", FIGURE_TOON, PROFILE_FULL),
                "body_bs_BreastsTipIn":MorphMeta(CAT_BREAST, "Breasts Tip In", FIGURE_TOON, PROFILE_MIN),
                "body_bs_BreastsTipOut":MorphMeta(CAT_BREAST, "Breasts Tip Out", FIGURE_TOON, PROFILE_FULL),
                "body_bs_BreastsTipUp":MorphMeta(CAT_BREAST, "Breasts Tip Up", FIGURE_TOON, PROFILE_MIN),
                "body_bs_BreastsWidth":MorphMeta(CAT_BREAST, "Breasts Width", FIGURE_TOON, PROFILE_FULL),
                "body_bs_BodyPearFigure":MorphMeta(CAT_BODY,"Pear Figure", FIGURE_ANY, PROFILE_MIN),
                "body_bs_BodyVoluptuous":MorphMeta(CAT_BODY,"Voluptuous", FIGURE_ANY, PROFILE_MIN),
                "body_bs_BreastsCleavage":MorphMeta(CAT_BREAST,"Breasts Cleavage", FIGURE_ANY, PROFILE_MIN),
                "body_bs_BreastsDiameter":MorphMeta(CAT_BREAST,"Breasts Diameter", FIGURE_ANY, PROFILE_FULL),
                "body_bs_BreastsDownwardSlope":MorphMeta(CAT_BREAST,"Breasts Downward Slope", FIGURE_ANY, PROFILE_FULL),
                "body_bs_BreastsFullnessLower":MorphMeta(CAT_BREAST,"Breasts Fullness Lower", FIGURE_ANY, PROFILE_FULL),
                "body_bs_BreastsFullnessUpper":MorphMeta(CAT_BREAST,"Breasts Fullness Upper", FIGURE_ANY, PROFILE_FULL),
                "body_bs_BreastsGone":MorphMeta(CAT_BREAST,"Breasts Gone", FIGURE_ANY, PROFILE_FULL),
                "body_bs_BreastsHeavy":MorphMeta(CAT_BREAST,"Breasts Heavy", FIGURE_ANY, PROFILE_FULL),
                "body_bs_BreastsLarge":MorphMeta(CAT_BREAST,"Breasts Large", FIGURE_ANY, PROFILE_FULL),
                "body_bs_BreastsLargeHigh":MorphMeta(CAT_BREAST,"Breasts Large High", FIGURE_ANY, PROFILE_FULL),
                "body_bs_BreastsNatural":MorphMeta(CAT_BREAST,"Breasts Natural", FIGURE_ANY, PROFILE_MIN),
                "body_bs_BreastsPerkSide":MorphMeta(CAT_BREAST,"Breasts Perk Side", FIGURE_ANY, PROFILE_MIN),
                "body_bs_BreastsShape01":MorphMeta(CAT_BREAST,"Breasts Shape 01", FIGURE_ANY, PROFILE_MIN),
                "body_bs_BreastsShape02":MorphMeta(CAT_BREAST,"Breasts Shape 02", FIGURE_ANY, PROFILE_MIN),
                "body_bs_BreastsShape03":MorphMeta(CAT_BREAST,"Breasts Shape 03", FIGURE_ANY, PROFILE_MIN),
                "body_bs_BreastsShape04":MorphMeta(CAT_BREAST,"Breasts Shape 04", FIGURE_ANY, PROFILE_MIN),
                "body_bs_BreastsShape05":MorphMeta(CAT_BREAST,"Breasts Shape 05", FIGURE_ANY, PROFILE_MIN),
                "body_bs_BreastsShape06":MorphMeta(CAT_BREAST,"Breasts Shape 06", FIGURE_ANY, PROFILE_FULL),
                "body_bs_BreastsSidesDepth":MorphMeta(CAT_BREAST,"Breasts Sides Depth", FIGURE_ANY, PROFILE_FULL),
                "body_bs_BreastsSmall":MorphMeta(CAT_BREAST,"Breasts Small", FIGURE_ANY, PROFILE_FULL),
                "body_bs_NipplesAreolaeDepthFeminine":MorphMeta(CAT_BREAST,"Nipples Areolae Depth Feminine", FIGURE_ANY, PROFILE_MIN),
                "body_bs_NipplesAreolaeDiameterFeminine":MorphMeta(CAT_BREAST,"Nipples Areolae Diameter Feminine", FIGURE_ANY, PROFILE_MIN),
                "body_bs_NipplesDepthFeminine_HD3":MorphMeta(CAT_BREAST,"Nipples Depth Feminine", FIGURE_ANY, PROFILE_FULL),
                "body_bs_NipplesDiameterFeminine":MorphMeta(CAT_BREAST,"Nipples Diameter Feminine", FIGURE_ANY, PROFILE_FULL),
                "body_bs_Pregnant":MorphMeta(CAT_SPECIAL, "Pregnant", FIGURE_ANY, PROFILE_MIN),
                "Hip_Back_Dimples": MorphMeta(CAT_BODY, "", FIGURE_TOON, PROFILE_FULL),
                "Breast_Size_Reduction": MorphMeta(CAT_BREAST, "", FIGURE_TOON, PROFILE_MIN),
                "Eye_Style_01": MorphMeta(CAT_SPECIAL, "", FIGURE_TOON, PROFILE_FULL),
                "Heart_Pupil": MorphMeta(CAT_SPECIAL, "", FIGURE_TOON, PROFILE_FULL),
                "Iris_Smoothing": MorphMeta(CAT_SPECIAL, "", FIGURE_TOON, PROFILE_FULL),
                "Left_Eye_Effect_Zoom_and_Out": MorphMeta(CAT_SPECIAL, "", FIGURE_TOON, PROFILE_FULL),
                "Navel_Sag_HD": MorphMeta(CAT_SPECIAL, "", FIGURE_TOON, PROFILE_FULL),
                "O_Shaped_Cat_Mouth": MorphMeta(CAT_SPECIAL, "", FIGURE_TOON, PROFILE_FULL),
                "Oblique_Eyebrow": MorphMeta(CAT_FACS, "", FIGURE_TOON, PROFILE_FULL),
                "Pseudo_Anime_Canine_Teeth_Left": MorphMeta(CAT_SPECIAL, "", FIGURE_TOON, PROFILE_FULL),
                "Pseudo_Anime_Canine_Teeth_Right": MorphMeta(CAT_SPECIAL, "", FIGURE_TOON, PROFILE_FULL),
                "Pupi_Constricting_Left": MorphMeta(CAT_SPECIAL, "", FIGURE_TOON, PROFILE_FULL),
                "Pupi_Constrictingl_Right": MorphMeta(CAT_SPECIAL, "", FIGURE_TOON, PROFILE_FULL),
                "Pupils_Shrink_and_Dilate": MorphMeta(CAT_SPECIAL, "", FIGURE_TOON, PROFILE_FULL),
                "Right_Eye_Effect_Zoom_and_Out": MorphMeta(CAT_SPECIAL, "", FIGURE_TOON, PROFILE_FULL),
                "Slanted_Eyebrow": MorphMeta(CAT_FACS, "", FIGURE_TOON, PROFILE_FULL),
            },
            "male": {
                "body_bs_AnimeBodyHeavyMasculine":MorphMeta(CAT_BODY, "Heavy", FIGURE_TOON, PROFILE_MIN),
                "body_bs_AnimeBodyMuscularMasculine":MorphMeta(CAT_BODY, "Muscular", FIGURE_TOON, PROFILE_MIN),
                "body_bs_BodyPortly":MorphMeta(CAT_BODY,"Body Portly", FIGURE_ANY, PROFILE_MIN),
                "body_bs_BodyStocky":MorphMeta(CAT_BODY,"Body Stocky", FIGURE_ANY, PROFILE_MIN),
                "body_bs_NipplesAreolaeDepthMasculine":MorphMeta(CAT_BODY,"Nipples Areolae Depth Masculine", FIGURE_ANY, PROFILE_FULL),
                "body_bs_NipplesAreolaeDiameterMasculine":MorphMeta(CAT_BODY,"Nipples Areolae Diameter Masculine", FIGURE_ANY, PROFILE_FULL),
                "body_bs_NipplesDepthMasculine_HD3":MorphMeta(CAT_BODY,"Nipples Depth Masculine", FIGURE_ANY, PROFILE_FULL),
                "body_bs_NipplesDiameterMasculine":MorphMeta(CAT_BODY,"Nipples Diameter Masculine", FIGURE_ANY, PROFILE_FULL),
                "body_bs_PectoralsCleavage":MorphMeta(CAT_BODY,"Pectorals Cleavage", FIGURE_ANY, PROFILE_FULL),
                "body_bs_PectoralsDiameter":MorphMeta(CAT_BODY,"Pectorals Diameter", FIGURE_ANY, PROFILE_MIN),
                "body_bs_PectoralsHeight":MorphMeta(CAT_BODY,"Pectorals Height", FIGURE_ANY, PROFILE_MIN),
                "body_bs_PectoralsHeightOuter":MorphMeta(CAT_BODY,"Pectorals Height Outer", FIGURE_ANY, PROFILE_FULL),
                "body_bs_PectoralsSag":MorphMeta(CAT_BODY,"Pectorals Sag", FIGURE_ANY, PROFILE_FULL),
                "body_bs_PectoralsSize":MorphMeta(CAT_BODY,"Pectorals Size", FIGURE_ANY, PROFILE_MIN),
                "body_bs_PectoralsUnderCurve":MorphMeta(CAT_BODY,"Pectorals Under Curve", FIGURE_ANY, PROFILE_FULL),
                "body_bs_PectoralsWidth":MorphMeta(CAT_BODY,"Pectorals Width", FIGURE_ANY, PROFILE_MIN),
            },

            "unisex": {
                "facs_cbs_JawOpen": MorphMeta(CAT_FACS, "", FIGURE_TOON, PROFILE_MIN),
                "facs_cbs_JawOpenWide": MorphMeta(CAT_FACS, "", FIGURE_TOON, PROFILE_FULL),
                "facs_cbs_EyeBlinkCrescentLeftFix":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_cbs_EyeBlinkCrescentRightFix":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_cbs_EyeBlinkLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_cbs_EyeBlinkRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_MouthUp":MorphMeta(CAT_FACS, "", FIGURE_ANY, PROFILE_FULL),
                "facs_bs_EyeLidCloseUpperRight":MorphMeta(CAT_FACS, "", FIGURE_ANY, PROFILE_FULL),
                "facs_bs_MouthCornerMoveSide_SideRight":MorphMeta(CAT_FACS, "", FIGURE_ANY, PROFILE_FULL),
                "facs_bs_MouthLipsCat":MorphMeta(CAT_FACS, "", FIGURE_ANY, PROFILE_FULL),
                "facs_bs_MouthCornerMoveSide_SideLeft":MorphMeta(CAT_FACS, "", FIGURE_ANY, PROFILE_FULL),
                "facs_bs_MouthCornerMoveBackLeft":MorphMeta(CAT_FACS, "", FIGURE_ANY, PROFILE_FULL),
                "facs_bs_MouthCornerMoveBackRight":MorphMeta(CAT_FACS, "", FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_EyeBlinkCrescentRight":MorphMeta(CAT_FACS, "", FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_EyeBlinkCrescentLeft":MorphMeta(CAT_FACS, "", FIGURE_ANY, PROFILE_FULL),
                "facs_bs_MouthLipsSide":MorphMeta(CAT_FACS, "", FIGURE_ANY, PROFILE_FULL),
                "pupils_Dilate_Right":MorphMeta(CAT_HEAD, "Pupils dilate right", FIGURE_ANY, PROFILE_FULL),
                "pupils_Dilate_Left":MorphMeta(CAT_HEAD, "Pupils dilate left", FIGURE_ANY, PROFILE_FULL),
                "facs_bs_BrowDownLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_BrowDownRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_BrowInnerUpLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_BrowInnerUpRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_BrowOuterUpLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_BrowOuterUpRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_BrowSqueezeLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_BrowSqueezeRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_CheekHollowLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_CheekHollowRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_CheekInflateCounterLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_CheekInflateCounterRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_CheekInflateLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_CheekInflateRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_CheekPuffLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_CheekPuffRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_CheekSquintLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_CheekSquintRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_EarPullBackLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_EarPullBackRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_EyeBlinkLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_EyeBlinkRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_EyeCornerTightnessInnerLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_EyeCornerTightnessInnerRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_EyeCornerTightnessOuterLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_EyeCornerTightnessOuterRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_EyeLidCloseUpperLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_EyeLookDownLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_EyeLookDownRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_EyeLookInLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_EyeLookInRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_EyeLookOutLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_EyeLookOutRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_EyeLookUpLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_EyeLookUpRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_EyePupilsDilate":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_EyeSquintLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_EyeSquintRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_EyelidCloseLowerLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_EyelidCloseLowerRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_EyelidCloseUpperRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_EyelidOpenLowerLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_EyelidOpenLowerRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_EyelidOpenUpperLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_EyelidOpenUpperRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_JawChinCompression":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_JawClenchLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_JawClenchRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_JawForward":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_JawLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_JawOpen":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_JawOpenWide":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_JawRecess":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_JawRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_MouthBackLowerLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_MouthBackLowerRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_MouthBackUpperLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_MouthBackUpperRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_MouthCloseLowerLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_MouthCloseLowerRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_MouthCloseUpperLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_MouthCloseUpperRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_MouthCompressLowerLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_MouthCompressLowerRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_MouthCompressUpperLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_MouthCompressUpperRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_MouthCornerMoveFwd-BackLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_MouthCornerMoveFwd-BackRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_MouthCornerMoveSide-SideLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_MouthCornerMoveSide-SideRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_MouthCornerMoveUp-DownLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_MouthCornerMoveUp-DownRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_MouthCornerTightnessLowerLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_MouthCornerTightnessLowerRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_MouthCornerTightnessUpperLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_MouthCornerTightnessUpperRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_MouthDimpleLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_MouthDimpleRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_MouthForward-BackMiddleLower":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_MouthForward-BackMiddleUpper":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_MouthForwardLowerLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_MouthForwardLowerRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_MouthForwardUpperLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_MouthForwardUpperRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_MouthFrownLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_MouthFrownRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_MouthFunnelLowerLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_MouthFunnelLowerRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_MouthFunnelUpperLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_MouthFunnelUpperRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_MouthLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_MouthLipsSide-SideLower":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_MouthLipsSide-SideUpper":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_MouthLipsVolumeLowerLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_MouthLipsVolumeLowerRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_MouthLipsVolumeUpperLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_MouthLipsVolumeUpperRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_MouthLowerDownLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_MouthLowerDownRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_MouthPressLowerLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_MouthPressLowerRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_MouthPressUpperLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_MouthPressUpperRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_MouthPurseLowerLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_MouthPurseLowerRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_MouthPurseUpperLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_MouthPurseUpperRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_MouthRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_MouthRollLowerLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_MouthRollLowerRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_MouthRollUpperLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_MouthRollUpperRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_MouthShrugLowerLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_MouthShrugLowerRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_MouthShrugUpperLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_MouthShrugUpperRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_MouthSmileLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_MouthSmileRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_MouthSmileWidenLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_MouthSmileWidenRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_MouthStickyLowerCenter":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_MouthStickyLowerInnerLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_MouthStickyLowerInnerRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_MouthStickyLowerOuterLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_MouthStickyLowerOuterRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_MouthStickyUpperCenter":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_MouthStickyUpperInnerLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_MouthStickyUpperInnerRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_MouthStickyUpperOuterLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_MouthStickyUpperOuterRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_MouthStretchLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_MouthStretchRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_MouthUp-Down":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_MouthUp-DownMiddleLower":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_MouthUp-DownMiddleUpper":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_MouthUpperUpLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_MouthUpperUpRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_NasalCompressLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_NasalCompressRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_NasalCreaseFlexLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_NasalCreaseFlexRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_NasalFlareLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_NasalFlareRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_NasalStretchLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_NasalStretchRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_NeckFlexLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_NeckFlexRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_NeckLarynxUp":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_NeckMastoidFlexLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_NeckMastoidFlexRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_NeckUnderChinFlex":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_NoseSneerLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_NoseSneerRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_NoseSneerUpperLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_NoseSneerUpperRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_TongueIn":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_TongueNarrow":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_TongueOut":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_bs_TongueRoll":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_TongueTipNarrow":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_bs_TongueTipWide":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_cbs_BIUL_BDL":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_cbs_BIUR_BDR":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_cbs_EBL_BIUL":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_cbs_EBL_BOUL":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_cbs_EBL_CSL":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_cbs_EBL_ELIL":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_cbs_EBL_ELOL":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_cbs_EBL_ESL":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_cbs_EBL_NSL":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_cbs_EBR_BIUR":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_cbs_EBR_BOUR":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_cbs_EBR_CSR":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_cbs_EBR_ELIR":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_cbs_EBR_ELOR":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_cbs_EBR_ESR":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_cbs_EBR_NSR":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_cbs_EyeFullCompressionLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_cbs_EyeFullCompressionRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_cbs_JO_MF":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_cbs_JO_ML":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_cbs_JO_MR":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_cbs_MP_CPL":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_cbs_MP_CPR":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_cbs_MP_MC":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_cbs_MP_MF":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_ctrl_BrowDown":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_BrowInnerUp":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_ctrl_BrowSqueeze":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_BrowUp":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_CheekHollow":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_CheekPuff":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_ctrl_CheekSquint":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_EarPullBack":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_EyeBlink":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_EyeCrossed":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_EyeFullCompressionLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_EyeFullCompressionRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_EyeLookAuto":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_ctrl_EyeLookDownLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_ctrl_EyeLookDownRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_ctrl_EyeLookInLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_ctrl_EyeLookInRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_ctrl_EyeLookOutLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_EyeLookOutRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_EyeLookSide-Side":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_EyeLookSide-SideLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_ctrl_EyeLookSide-SideRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_ctrl_EyeLookUp-Down":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_EyeLookUp-DownLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_ctrl_EyeLookUp-DownRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_ctrl_EyeLookUpLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_EyeLookUpRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_EyeRestingFocalPoint":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_EyeSquint":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_EyeWide":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_ctrl_EyeWideLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_ctrl_EyeWideRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_ctrl_EyelidOpen-CloseLower":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_EyelidOpen-CloseLowerLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_EyelidOpen-CloseUpper":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_EyelidOpen-CloseUpperLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_EyelidOpen-CloseUpperRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_EyelidOpenCloseLowerRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_JawClench":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_JawForward-Recess":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_ctrl_JawLeft-Right":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_ctrl_MouthClose":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_ctrl_MouthCompress":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_MouthDimple":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_MouthForward-Back":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_MouthForward-BackLowerLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_MouthForward-BackLowerRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_MouthForward-BackUpperLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_MouthForward-BackUpperRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_MouthFrown":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_MouthFunnel":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_ctrl_MouthLeft-Right":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_ctrl_MouthLowerDown":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_MouthPress":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_MouthPressLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_ctrl_MouthPressRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_ctrl_MouthPucker":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_ctrl_MouthPurse":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_MouthRollLower":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_ctrl_MouthRollUpper":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_ctrl_MouthShrugLower":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_ctrl_MouthShrugUpper":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_ctrl_MouthSmile":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_MouthSmileWiden":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_MouthStickyControlLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_MouthStickyControlRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_MouthStickyPower":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_MIN),
                "facs_ctrl_MouthStretch":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_MouthUpperUp":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_NasalCompress":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_NasalCreaseFlex":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_NasalFlare":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_NasalStretch":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_NeckFlex":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_NeckSwallow":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_NoseSneer":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_NoseSneerUpper":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_TongueLeft":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_TongueRight":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_TongueTipBend":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_TongueTwist":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_TongueUp":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_vAA":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_vEE":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_vEH":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_vER":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_vF":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_vIH":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_vIY":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_vK":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_vL":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_vM":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_vOW":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_vS":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_vSH":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_vT":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_vTH":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_vUW":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "facs_ctrl_vW":MorphMeta(CAT_FACS,"",FIGURE_ANY, PROFILE_FULL),
                "head_ctrl_ToonIrisOval":MorphMeta(CAT_HEAD, "Iris Oval", FIGURE_TOON, PROFILE_FULL),
                "body_bs_FingersTaper":MorphMeta(CAT_ARMS, "Fingers Taper", FIGURE_TOON, PROFILE_FULL),
                "head_bs_EarElfLongOutward":MorphMeta(CAT_HEAD, "Ear Elf Long Outward", FIGURE_TOON, PROFILE_MIN),
                "head_bs_EyeInnerHeightSize":MorphMeta(CAT_HEAD, "Eye Inner Height", FIGURE_TOON, PROFILE_FULL),
                "head_bs_EyeLowerCurveBalance":MorphMeta(CAT_HEAD, "Eye Lower Curve Balance", FIGURE_TOON, PROFILE_FULL),
                "head_bs_EyeOuterArcDown":MorphMeta(CAT_HEAD, "Eye Outer Arc Down", FIGURE_TOON, PROFILE_FULL),
                "head_bs_EyeSwoopUp":MorphMeta(CAT_HEAD, "Eye Swoop Up", FIGURE_TOON, PROFILE_FULL),
                "head_bs_EyeUpperCurveBalance":MorphMeta(CAT_HEAD, "Eye Upper Curve Balance", FIGURE_TOON, PROFILE_FULL),
                "head_bs_EyeUpperPeak":MorphMeta(CAT_HEAD, "Eye Upper Peak", FIGURE_TOON, PROFILE_FULL),
                "body_bs_AbdominalsCenterDefine": MorphMeta(CAT_BODY, "Abdominals Center Define", FIGURE_ANY, PROFILE_FULL),
                "body_bs_AbdominalsOuterDefine": MorphMeta(CAT_BODY, "Abdominals Outer Define", FIGURE_ANY, PROFILE_FULL),
                "body_bs_AbdominalsWidth": MorphMeta(CAT_BODY, "Abdominals Width", FIGURE_ANY, PROFILE_FULL),
                "body_bs_BodyEmaciated": MorphMeta(CAT_BODY, "Body Emaciated", FIGURE_ANY, PROFILE_MIN),
                "body_bs_BodyFitnessDetails": MorphMeta(CAT_BODY, "Body Fitness Details", FIGURE_ANY, PROFILE_MIN),
                "body_bs_BodyFitnessMass": MorphMeta(CAT_BODY, "Body Fitness Mass", FIGURE_ANY, PROFILE_MIN),
                "body_bs_BodyHeavy": MorphMeta(CAT_BODY, "Body Heavy", FIGURE_ANY, PROFILE_MIN),
                "body_bs_BodyLithe": MorphMeta(CAT_BODY, "Body Lithe", FIGURE_ANY, PROFILE_MIN),
                "body_bs_BodyMuscularDetails": MorphMeta(CAT_BODY, "Body Muscular Details", FIGURE_ANY, PROFILE_MIN),
                "body_bs_BodyMuscularMass": MorphMeta(CAT_BODY, "Body Muscular Mass", FIGURE_ANY, PROFILE_MIN),
                "body_bs_BodyOlder": MorphMeta(CAT_BODY, "Body Older", FIGURE_ANY, PROFILE_MIN),
                "body_bs_BodyThin": MorphMeta(CAT_BODY, "Body Thin", FIGURE_ANY, PROFILE_MIN),
                "body_bs_BodyTone": MorphMeta(CAT_BODY, "Body Tone", FIGURE_ANY, PROFILE_MIN),
                "body_bs_CalvesSize": MorphMeta(CAT_LEGS, "Calves Size", FIGURE_ANY, PROFILE_FULL),
                "body_bs_CollarboneDetail": MorphMeta(CAT_BODY, "Collarbone Detail", FIGURE_ANY, PROFILE_FULL),
                "body_bs_FingersWidth": MorphMeta(CAT_ARMS, "Fingers Width", FIGURE_ANY, PROFILE_FULL),
                "body_bs_FootArchDepth": MorphMeta(CAT_LEGS, "Foot Arch Depth", FIGURE_ANY, PROFILE_FULL),
                "body_bs_GluteCrease": MorphMeta(CAT_ASS, "Glute Crease", FIGURE_ANY, PROFILE_FULL),
                "body_bs_GluteDepthLower": MorphMeta(CAT_ASS, "Glute Depth Lower", FIGURE_ANY, PROFILE_FULL),
                "body_bs_GluteDepthUpper": MorphMeta(CAT_ASS, "Glute Depth Upper", FIGURE_ANY, PROFILE_FULL),
                "body_bs_GluteSize": MorphMeta(CAT_ASS, "Glute Size", FIGURE_ANY, PROFILE_MIN),
                "body_bs_GluteWidth": MorphMeta(CAT_ASS, "Glute Width", FIGURE_ANY, PROFILE_MIN),
                "body_bs_HipBackDimples": MorphMeta(CAT_ASS, "Hip Back Dimples", FIGURE_ANY, PROFILE_FULL),
                "body_bs_HipBoneCrest": MorphMeta(CAT_ASS, "Hip Bone Crest", FIGURE_ANY, PROFILE_FULL),
                "body_bs_HipBoneSize": MorphMeta(CAT_ASS, "Hip Bone Size", FIGURE_ANY, PROFILE_MIN),
                "body_bs_HipGenitalBulge": MorphMeta(CAT_ASS, "Hip Genital Bulge", FIGURE_ANY, PROFILE_FULL),
                "body_bs_HipPelvicTilt": MorphMeta(CAT_ASS, "Hip Pelvic Tilt", FIGURE_ANY, PROFILE_FULL),
                "body_bs_HipSize": MorphMeta(CAT_ASS, "Hip Size", FIGURE_ANY, PROFILE_FULL),
                "body_bs_HipVDefine": MorphMeta(CAT_ASS, "Hip V Define", FIGURE_ANY, PROFILE_FULL),
                "body_bs_KneeBonesSize": MorphMeta(CAT_LEGS, "Knee Bones Size", FIGURE_ANY, PROFILE_FULL),
                "body_bs_LatsSize": MorphMeta(CAT_BODY, "Lats Size", FIGURE_ANY, PROFILE_FULL),
                "body_bs_LoveHandles": MorphMeta(CAT_BODY, "Love Handles", FIGURE_ANY, PROFILE_MID),
                "body_bs_MassAnkles": MorphMeta(CAT_LEGS, "Mass Ankles", FIGURE_ANY, PROFILE_MID),
                "body_bs_MassBody": MorphMeta(CAT_BODY, "Mass Body", FIGURE_ANY, PROFILE_MID),
                "body_bs_MassFeet": MorphMeta(CAT_LEGS, "Mass Feet", FIGURE_ANY, PROFILE_MID),
                "body_bs_MassForearms": MorphMeta(CAT_ARMS, "Mass Forearms", FIGURE_ANY, PROFILE_MID),
                "body_bs_MassHands": MorphMeta(CAT_ARMS, "Mass Hands", FIGURE_ANY, PROFILE_MID),
                "body_bs_MassKnees": MorphMeta(CAT_LEGS, "Mass Knees", FIGURE_ANY, PROFILE_MIN),
                "body_bs_MassLowerTorso": MorphMeta(CAT_BODY, "Mass Lower Torso", FIGURE_ANY, PROFILE_MID),
                "body_bs_MassNeck": MorphMeta(CAT_HEAD, "Mass Neck", FIGURE_ANY, PROFILE_MID),
                "body_bs_MassShins": MorphMeta(CAT_LEGS, "Mass Shins", FIGURE_ANY, PROFILE_MID),
                "body_bs_MassShoulders": MorphMeta(CAT_ARMS, "Mass Shoulders", FIGURE_ANY, PROFILE_MID),
                "body_bs_MassThighs": MorphMeta(CAT_LEGS, "Mass Thighs", FIGURE_ANY, PROFILE_MIN),
                "body_bs_MassUpperarms": MorphMeta(CAT_ARMS, "Mass Upperarms", FIGURE_ANY, PROFILE_MID),
                "body_bs_MassUpperTorso": MorphMeta(CAT_BODY, "Mass Upper Torso", FIGURE_ANY, PROFILE_MID),
                "body_bs_MassWrist": MorphMeta(CAT_ARMS, "Mass Wrist", FIGURE_ANY, PROFILE_FULL),
                "body_bs_NailsLengthRound": MorphMeta(CAT_ARMS, "Nails Length Round", FIGURE_ANY, PROFILE_MID),
                "body_bs_NailsLengthSharp": MorphMeta(CAT_ARMS, "Nails Length Sharp", FIGURE_ANY, PROFILE_MID),
                "body_bs_NailsLengthSquare": MorphMeta(CAT_ARMS, "Nails Length Square", FIGURE_ANY, PROFILE_FULL),
                "body_bs_NavelDepth_HD3": MorphMeta(CAT_BODY, "Navel Depth", FIGURE_ANY, PROFILE_FULL),
                "body_bs_NavelHollow_HD3": MorphMeta(CAT_BODY, "Navel Hollow", FIGURE_ANY, PROFILE_FULL),
                "body_bs_NavelHorizontal_HD3": MorphMeta(CAT_BODY, "Navel Horizontal", FIGURE_ANY, PROFILE_FULL),
                "body_bs_NavelOut_HD3": MorphMeta(CAT_BODY, "Navel Out", FIGURE_ANY, PROFILE_FULL),
                "body_bs_NavelSize_HD3": MorphMeta(CAT_BODY, "Navel Size", FIGURE_ANY, PROFILE_FULL),
                "body_bs_NavelVertical_HD3": MorphMeta(CAT_BODY, "Navel Vertical", FIGURE_ANY, PROFILE_FULL),
                "body_bs_RibcageArched": MorphMeta(CAT_BODY, "Ribcage Arched", FIGURE_ANY, PROFILE_FULL),
                "body_bs_RibcagePointed": MorphMeta(CAT_BODY, "Ribcage Pointed", FIGURE_ANY, PROFILE_FULL),
                "body_bs_RibcageSize": MorphMeta(CAT_BODY, "Ribcage Size", FIGURE_ANY, PROFILE_FULL),
                "body_bs_ScapulaDepth": MorphMeta(CAT_BODY, "Scapula Depth", FIGURE_ANY, PROFILE_FULL),
                "body_bs_ScapulaSize": MorphMeta(CAT_BODY, "Scapula Size", FIGURE_ANY, PROFILE_FULL),
                "body_bs_SternumDepth": MorphMeta(CAT_BODY, "Sternum Depth", FIGURE_ANY, PROFILE_FULL),
                "body_bs_SternumHeight": MorphMeta(CAT_BODY, "Sternum Height", FIGURE_ANY, PROFILE_FULL),
                "body_bs_SternumWidth": MorphMeta(CAT_BODY, "Sternum Width", FIGURE_ANY, PROFILE_FULL),
                "body_bs_StomachDepth": MorphMeta(CAT_BODY, "Stomach Depth", FIGURE_ANY, PROFILE_FULL),
                "body_bs_StomachDepthLower": MorphMeta(CAT_BODY, "Stomach Depth Lower", FIGURE_ANY, PROFILE_FULL),
                "body_bs_StomachSoften": MorphMeta(CAT_BODY, "Stomach Soften", FIGURE_ANY, PROFILE_FULL),
                "body_bs_TaperForearmA": MorphMeta(CAT_ARMS, "Taper Forearm A", FIGURE_ANY, PROFILE_FULL),
                "body_bs_TaperForearmB": MorphMeta(CAT_ARMS, "Taper Forearm B", FIGURE_ANY, PROFILE_FULL),
                "body_bs_TaperNeckA": MorphMeta(CAT_BODY, "Taper Neck A", FIGURE_ANY, PROFILE_FULL),
                "body_bs_TaperNeckB": MorphMeta(CAT_BODY, "Taper Neck B", FIGURE_ANY, PROFILE_FULL),
                "body_bs_TaperShinA": MorphMeta(CAT_LEGS, "Taper Shin A", FIGURE_ANY, PROFILE_FULL),
                "body_bs_TaperShinB": MorphMeta(CAT_LEGS, "Taper Shin B", FIGURE_ANY, PROFILE_FULL),
                "body_bs_TaperThighA": MorphMeta(CAT_LEGS, "Taper Thigh A", FIGURE_ANY, PROFILE_MID),
                "body_bs_TaperThighB": MorphMeta(CAT_LEGS, "Taper Thigh B", FIGURE_ANY, PROFILE_MID),
                "body_bs_TaperUpperArmA": MorphMeta(CAT_ARMS, "Taper Upper Arm A", FIGURE_ANY, PROFILE_MID),
                "body_bs_TaperUpperArmB": MorphMeta(CAT_ARMS, "Taper Upper Arm B", FIGURE_ANY, PROFILE_MID),
                "body_bs_ThighDepth": MorphMeta(CAT_LEGS, "Thigh Depth", FIGURE_ANY, PROFILE_FULL),
                "body_bs_ThighTone": MorphMeta(CAT_LEGS, "Thigh Tone", FIGURE_ANY, PROFILE_FULL),
                "body_bs_TrapsSize": MorphMeta(CAT_BODY, "Traps Size", FIGURE_ANY, PROFILE_FULL),
                "body_bs_UpperArmTaperWidth": MorphMeta(CAT_ARMS, "Upper Arm Taper Width", FIGURE_ANY, PROFILE_FULL),
                "body_bs_WaistDepth": MorphMeta(CAT_BODY, "Waist Depth", FIGURE_ANY, PROFILE_MID),
                "body_bs_WaistWidth": MorphMeta(CAT_BODY, "Waist Width", FIGURE_ANY, PROFILE_MID),
                "body_bs_WaistWidthUpper": MorphMeta(CAT_BODY, "Waist Width Upper", FIGURE_ANY, PROFILE_FULL),
                "body_ctrl_BodyFitness": MorphMeta(CAT_BODY, "Body Fitness", FIGURE_ANY, PROFILE_MIN),
                "body_ctrl_BodyMuscular": MorphMeta(CAT_BODY, "Body Muscular", FIGURE_ANY, PROFILE_MIN),
            }
        }
    },
    "/data/daz 3d/genesis 9/anatomical elements male": {
        "shapes": {
            "male": {
                "body_ctrl_PenileLength":MorphMeta(CAT_GENITALS,"Penile Length", FIGURE_ANY, PROFILE_MIN),
                "body_ctrl_PenileWidth":MorphMeta(CAT_GENITALS,"Penile Width", FIGURE_ANY, PROFILE_MIN),
            }
        }
    },
    "/data/meipex/goldenpalace_genitalia_g9/g9goldenpalace_graft": {
        "shapes": {
            "female": {
                "GP_MonsVeneris_Front-Back1":MorphMeta(CAT_GENITALS,"Mons Veneris Front-Back 1", FIGURE_ANY, PROFILE_MIN),
                "GP_MonsVeneris_Front-Back2":MorphMeta(CAT_GENITALS,"Mons Veneris Front-Back 2", FIGURE_ANY, PROFILE_MIN),
                "GP_MonsVeneris_Up-Down":MorphMeta(CAT_GENITALS,"Mons Veneris Up-Down", FIGURE_ANY, PROFILE_FULL),
                "GP_PR-Aicie":MorphMeta(CAT_GENITALS,"Aicie", FIGURE_ANY, PROFILE_FULL),
                "GP_PR-Angelica":MorphMeta(CAT_GENITALS,"Angelica", FIGURE_ANY, PROFILE_MIN),
                "GP_PR-Astra":MorphMeta(CAT_GENITALS,"Astra", FIGURE_ANY, PROFILE_MIN),
                "GP_PR-Augusta":MorphMeta(CAT_GENITALS,"Augusta", FIGURE_ANY, PROFILE_MIN),
                "GP_PR-Bammi":MorphMeta(CAT_GENITALS,"Bammi", FIGURE_ANY, PROFILE_FULL),
                "GP_PR-Carmen":MorphMeta(CAT_GENITALS,"Carmen", FIGURE_ANY, PROFILE_MIN),
                "GP_PR-Chiktad":MorphMeta(CAT_GENITALS,"Chiktad", FIGURE_ANY, PROFILE_FULL),
                "GP_PR-Crystal":MorphMeta(CAT_GENITALS,"Crystal", FIGURE_ANY, PROFILE_MIN),
                "GP_PR-Felicitas":MorphMeta(CAT_GENITALS,"Felicitas", FIGURE_ANY, PROFILE_MIN),
                "GP_PR-Gaia":MorphMeta(CAT_GENITALS,"Gaia", FIGURE_ANY, PROFILE_MIN),
                "GP_PR-Ilzolla":MorphMeta(CAT_GENITALS,"Ilzolla", FIGURE_ANY, PROFILE_FULL),
                "GP_PR-Krognea":MorphMeta(CAT_GENITALS,"Krognea", FIGURE_ANY, PROFILE_FULL),
                "GP_PR-Laetitia":MorphMeta(CAT_GENITALS,"Laetitia", FIGURE_ANY, PROFILE_MIN),
                "GP_PR-Livia":MorphMeta(CAT_GENITALS,"Livia", FIGURE_ANY, PROFILE_MIN),
                "GP_PR-Raad":MorphMeta(CAT_GENITALS,"Raad", FIGURE_ANY, PROFILE_FULL),
                "GP_PR-Remphea":MorphMeta(CAT_GENITALS,"Remphea", FIGURE_ANY, PROFILE_FULL),
                "GP_PR-Tophae":MorphMeta(CAT_GENITALS,"Tophae", FIGURE_ANY, PROFILE_FULL),
                "GP_PR-Valentina":MorphMeta(CAT_GENITALS,"Valentina", FIGURE_ANY, PROFILE_MIN),
                "GP_PR-Yelqi":MorphMeta(CAT_GENITALS,"Yelqi", FIGURE_ANY, PROFILE_FULL),
                "GP_PR-Zetill":MorphMeta(CAT_GENITALS,"Zetill", FIGURE_ANY, PROFILE_FULL),
                "GP_PR_Albina":MorphMeta(CAT_GENITALS,"Albina", FIGURE_ANY, PROFILE_MIN),
                "GP_PR_Amazon":MorphMeta(CAT_GENITALS,"Amazon", FIGURE_ANY, PROFILE_FULL),
                "GP_PR_Aurelia":MorphMeta(CAT_GENITALS,"Aurelia", FIGURE_ANY, PROFILE_MIN),
                "GP_PR_Majora01":MorphMeta(CAT_GENITALS,"Majora 01", FIGURE_ANY, PROFILE_FULL),
                "GP_PR_Majora02":MorphMeta(CAT_GENITALS,"Majora 02", FIGURE_ANY, PROFILE_FULL),
                "GP_PR_Majora03":MorphMeta(CAT_GENITALS,"Majora 03", FIGURE_ANY, PROFILE_FULL),
                "GP_PR_Minora01":MorphMeta(CAT_GENITALS,"Minora 01", FIGURE_ANY, PROFILE_FULL),
                "GP_PR_Minora02":MorphMeta(CAT_GENITALS,"Minora 02", FIGURE_ANY, PROFILE_FULL),
                "GP_PR_Minora03":MorphMeta(CAT_GENITALS,"Minora 03", FIGURE_ANY, PROFILE_FULL),
                "GP_PR_Minora04":MorphMeta(CAT_GENITALS,"Minora 04", FIGURE_ANY, PROFILE_FULL),
                "GP_PR_Minora05":MorphMeta(CAT_GENITALS,"Minora 05", FIGURE_ANY, PROFILE_FULL),
                "GP_PR_Minora06":MorphMeta(CAT_GENITALS,"Minora 06", FIGURE_ANY, PROFILE_FULL),
                "GP_PR_Minora07":MorphMeta(CAT_GENITALS,"Minora 07", FIGURE_ANY, PROFILE_FULL),
                "GP_PR_Minora08":MorphMeta(CAT_GENITALS,"Minora 08", FIGURE_ANY, PROFILE_FULL),
                "GP_PR_Minora09":MorphMeta(CAT_GENITALS,"Minora 09", FIGURE_ANY, PROFILE_FULL),
                "GP_PR_Minora10":MorphMeta(CAT_GENITALS,"Minora 10", FIGURE_ANY, PROFILE_FULL),
                "GP_Anus_Bump1":MorphMeta(CAT_GENITALS,"Anus Bump 1", FIGURE_ANY, PROFILE_FULL),
                "GP_Anus_Bump2":MorphMeta(CAT_GENITALS,"Anus Bump 2", FIGURE_ANY, PROFILE_FULL),
                "GP_Anus_Bump3":MorphMeta(CAT_GENITALS,"Anus Bump 3", FIGURE_ANY, PROFILE_FULL),
                "GP_Anus_Bump4":MorphMeta(CAT_GENITALS,"Anus Bump 4", FIGURE_ANY, PROFILE_FULL),
                "GP_Anus_Bump5":MorphMeta(CAT_GENITALS,"Anus Bump 5", FIGURE_ANY, PROFILE_FULL),
                "GP_Anus_Contraction":MorphMeta(CAT_GENITALS,"Anus Contraction", FIGURE_ANY, PROFILE_FULL),
                "GP_Anus_ContractionMore":MorphMeta(CAT_GENITALS,"Anus Contraction More", FIGURE_ANY, PROFILE_FULL),
                "GP_Anus_Contraction_Shape1":MorphMeta(CAT_GENITALS,"Anus Contraction", FIGURE_ANY, PROFILE_FULL),
                "GP_Anus_Contraction_Shape2":MorphMeta(CAT_GENITALS,"Anus Contraction", FIGURE_ANY, PROFILE_FULL),
                "GP_Anus_Contraction_Wrinkles1":MorphMeta(CAT_GENITALS,"Anus Contraction", FIGURE_ANY, PROFILE_FULL),
                "GP_Anus_Contraction_Wrinkles2":MorphMeta(CAT_GENITALS,"Anus Contraction", FIGURE_ANY, PROFILE_FULL),
                "GP_Anus_Contraction_Wrinkles3":MorphMeta(CAT_GENITALS,"Anus Contraction", FIGURE_ANY, PROFILE_FULL),
                "GP_Anus_Open1":MorphMeta(CAT_SPECIAL,"Anus Open 1", FIGURE_ANY, PROFILE_FULL),
                "GP_Anus_Open2":MorphMeta(CAT_SPECIAL,"Anus Open 2", FIGURE_ANY, PROFILE_MIN),
                "GP_Anus_Open3":MorphMeta(CAT_SPECIAL,"Anus Open 3", FIGURE_ANY, PROFILE_FULL),
                "GP_PR_Vagina_Open":MorphMeta(CAT_SPECIAL,"Vagina Open", FIGURE_ANY, PROFILE_FULL),
                "GP_Vagina_Open1":MorphMeta(CAT_SPECIAL,"Vagina Open 1", FIGURE_ANY, PROFILE_FULL),
                "GP_Vagina_Open2":MorphMeta(CAT_SPECIAL,"Vagina Open 2", FIGURE_ANY, PROFILE_MIN),
                "GP_Vagina_Open3":MorphMeta(CAT_SPECIAL,"Vagina Open 3", FIGURE_ANY, PROFILE_FULL),

            }
        }
    }
}


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

EXTRUDED_SK_NAME = 'extruded'
CLOTHES_MIN_DIST_TO_SKIN = 0.004


class ClothesMeta:

    def __init__(self, fingerprint, skin_tight, panties, transfer_jiggle_bones=False):
        self.fingerprint = fingerprint
        self.skin_tight = skin_tight
        self.panties = panties
        self.obj = None
        self.transfer_jiggle_bones = transfer_jiggle_bones

PANTIE_SCALING = 0.02



CLOTHES = {
    "Romance Bra": ClothesMeta('13332-26195-12864', CLOTHES_MIN_DIST_TO_SKIN, False),
    "Romance Choker": ClothesMeta('823-1590-768', CLOTHES_MIN_DIST_TO_SKIN, False),
    "Romance Panties": ClothesMeta('4110-7999-3890', PANTIE_SCALING, True),
    "Romance Thigh Straps": ClothesMeta('7852-15680-7840', CLOTHES_MIN_DIST_TO_SKIN, False),
    'SU Bunny Suit Clothes G9': ClothesMeta('7097-14070-6972', CLOTHES_MIN_DIST_TO_SKIN, False),
    'SU Bunny Suit Pantyhose G9': ClothesMeta('8926-17819-8894', CLOTHES_MIN_DIST_TO_SKIN, False),
    'SU Bunny Suit Top G9': ClothesMeta('6677-13295-6618', CLOTHES_MIN_DIST_TO_SKIN, False),
    'M3DAO ArmBands': ClothesMeta('6696-12834-6138', CLOTHES_MIN_DIST_TO_SKIN, False),
    'M3DAO BodySuit': ClothesMeta('9047-18026-8976', CLOTHES_MIN_DIST_TO_SKIN, False),
    'M3DAO Covers': ClothesMeta('390-724-336', CLOTHES_MIN_DIST_TO_SKIN, False),
    'M3DAO Harness Bodysuit': ClothesMeta('39533-76438-36896', CLOTHES_MIN_DIST_TO_SKIN, False),
    'M3DAO Legbands': ClothesMeta('11054-21144-10090', CLOTHES_MIN_DIST_TO_SKIN, False),
    'M3DAO Upper ArmBands': ClothesMeta('1144-2184-1040', CLOTHES_MIN_DIST_TO_SKIN, False),
    'M3DEORing': ClothesMeta('801-1606-805', CLOTHES_MIN_DIST_TO_SKIN, False),
    'Midnight Lingerie Corset': ClothesMeta('18102-35732-17622', CLOTHES_MIN_DIST_TO_SKIN, False),
    'Midnight Lingerie Panties': ClothesMeta('3601-6736-3136', PANTIE_SCALING, True),
    'Midnight Lingerie Stockings': ClothesMeta('4358-8688-4332', CLOTHES_MIN_DIST_TO_SKIN, False),
    'Refined Lingerie Corset for Genesis 9': ClothesMeta('29842-58911-29056', CLOTHES_MIN_DIST_TO_SKIN, False),
    'Refined Lingerie Gloves for Genesis 9': ClothesMeta('24066-48052-23988', CLOTHES_MIN_DIST_TO_SKIN, False),
    'Refined Lingerie Panties for Genesis 9': ClothesMeta('6236-12222-5987', CLOTHES_MIN_DIST_TO_SKIN, True),
    'Refined Lingerie Stockings for Genesis 9': ClothesMeta('14072-28090-14020', CLOTHES_MIN_DIST_TO_SKIN, False),
    'Scarlett Lingerie': ClothesMeta('11517-22868-11385', CLOTHES_MIN_DIST_TO_SKIN, False),
    'SU Sexy Line Bikini G9': ClothesMeta('7880-14804-6913', CLOTHES_MIN_DIST_TO_SKIN, False),
    'XF Court Stylish Bottom': ClothesMeta('949-1838-888', CLOTHES_MIN_DIST_TO_SKIN, False),
    'XF Court Stylish Corset': ClothesMeta('32855-64044-31240', CLOTHES_MIN_DIST_TO_SKIN, False),
    'XF Court Stylish Stockings': ClothesMeta('4904-9740-4836', CLOTHES_MIN_DIST_TO_SKIN, False),
    'LVA Arm Guard': ClothesMeta('28366-56444-28222', -1, False),
    'LVA Belt': ClothesMeta('1948-3872-1928', -1, False),
    'LVA Boots': ClothesMeta('6420-12662-6244', -1, False),
    'LVA Pant': ClothesMeta('4526-8995-4468', -1, False),
    'LVA Shirt': ClothesMeta('10438-20704-10264', -1, False),
    'LVA Vest': ClothesMeta('3751-7337-3585', -1, False),
    'LVA Vest Straps': ClothesMeta('5580-11130-5568', -1, False),
    'BattleMage Medallion G9': ClothesMeta('16291-32492-16202', -1, False),
    'BattleMage Shirt G9': ClothesMeta('59984-117646-57745', -1, False),
    'BattleMage Skirt G9': ClothesMeta('62529-124028-61617', -1, False),
    'Dark Side Corset': ClothesMeta('2416-4708-2292', CLOTHES_MIN_DIST_TO_SKIN, False),
    'Dark Side Jacket': ClothesMeta('11447-22693-11243', -1, False),
    'Dark Side Pants': ClothesMeta('5387-10716-5328', CLOTHES_MIN_DIST_TO_SKIN, False),
    'Dark Side Top': ClothesMeta('1117-2163-1044', CLOTHES_MIN_DIST_TO_SKIN, False),
    'RMO Dress': ClothesMeta('12119-23967-11846', CLOTHES_MIN_DIST_TO_SKIN, False),
    'VALBattleMage Bracers G9': ClothesMeta('16106-30990-14938', -1, False),
    'VALBattleMage Pantie G9': ClothesMeta('582-1067-484', 0.01, True),
    'White Witch Dress': ClothesMeta('22767-45315-22576', -1, False),
    'White Witch Hat': ClothesMeta('6987-13875-6903', -1, False),
    'White Witch Sleeves': ClothesMeta('19852-39574-19726', -1, False),
    'DCKA Guard Vest': ClothesMeta('31098-60151-29196', -1, False),
    'Urban Kunoichi Glove': ClothesMeta('7894-15692-7800', -1, False),
    'Urban Kunoichi Heels': ClothesMeta('6524-12840-6320', -1, False),
    'Urban Kunoichi Jacket': ClothesMeta('4682-9316-4632', -1, False),
    'Urban Kunoichi Mask': ClothesMeta('1207-2328-1120', -1, False),
    'Urban Kunoichi Pants': ClothesMeta('3763-7429-3675', -1, False),
    'Urban Kunoichi Top': ClothesMeta('840-1626-784', -1, False),
    'PPSS Boots': ClothesMeta('17589-35086-17501', -1, False),
    'PPSS Jacket': ClothesMeta('49380-98350-48997', -1, False),
    'PPSS Skirt': ClothesMeta('11467-22703-11232', -1, False),
    'PPSS Stockings': ClothesMeta('10912-21736-10824', CLOTHES_MIN_DIST_TO_SKIN, False),
    'PPSS Tshirt': ClothesMeta('12234-24008-11773', -1, False),
    'Nirv Zero\'s Bra': ClothesMeta('1954-3838-1884', -1, False),
    'Nirv Zero\'s Cat Slippers': ClothesMeta('1940-3848-1932', -1, False),
    'Nirv Zero\'s Underwear': ClothesMeta('3003-5896-2892', -1, False),
    'dForce Nirv Zero\'s Dresses': ClothesMeta('10859-21641-10780', -1, False),
    'SS14 TOON': ClothesMeta('11797-23057-11281', -1, False),
    'SS35 BRA TOON': ClothesMeta('23787-47090-23306', -1, False),
    'SS35 THONG TOON': ClothesMeta('16038-31713-15680', CLOTHES_MIN_DIST_TO_SKIN, True),
    'SXS25 PANTS TOON': ClothesMeta('1702-3278-1574', CLOTHES_MIN_DIST_TO_SKIN, True),
    'SXS25 TOP TOON': ClothesMeta('2639-5079-2436', CLOTHES_MIN_DIST_TO_SKIN, False),
    'SXS29G9TOON': ClothesMeta('7299-13864-6532', CLOTHES_MIN_DIST_TO_SKIN, False),
    'the Demon Servant Crown': ClothesMeta('2894-5628-2754', -1, False),
    'the Demon Servant Gauntlet': ClothesMeta('5112-9594-4498', -1, False),
    'the Demon Servant Greaves': ClothesMeta('10436-20112-9702', -1, False),
    'the Demon Servant Hands': ClothesMeta('4326-8256-3952', -1, False),
    'the Demon Servant Pants': ClothesMeta('2516-4977-2460', -1, False),
    'the Demon Servant Shawl': ClothesMeta('2688-5312-2624', -1, False),
    'the Demon Servant Shirt': ClothesMeta('4912-9788-4876', -1, False),
    'the Demon Servant Skirt': ClothesMeta('2106-4088-1984', -1, False),
    'the Demon Servant Top Armor': ClothesMeta('3765-7357-3590', -1, False),
    'the Demon Servant Waist Armors': ClothesMeta('1554-2914-1366', -1, False),
    'Viking Warrior Pants': ClothesMeta('2687-5323-2635', -1, False),
    'Viking Warrior Tunic': ClothesMeta('6975-13839-6856', -1, False),
    'Warrior Gloves': ClothesMeta('3964-7828-3856', -1, False),
    'Warrior Gloves Shield': ClothesMeta('2052-3768-1728', -1, False),
    'Warrior Pants': ClothesMeta('2343-4640-2296', -1, False),
    'Warrior Skirt': ClothesMeta('2216-4296-2080', -1, False),
    'Warrior Tops': ClothesMeta('3913-7680-3766', -1, False),
    'SAO Dress': ClothesMeta('37250-74468-37212', -1, False),
    'SAO Gloves': ClothesMeta('11836-23030-11220', -1, False),
    'SAO Shin Guard': ClothesMeta('7526-15022-7532', -1, False),
    'SAO Shoes': ClothesMeta('6860-13628-6798', -1, False),
    'SAO Socks': ClothesMeta('5168-10268-5100', CLOTHES_MIN_DIST_TO_SKIN, False),
    'SAO Thigh Guard': ClothesMeta('1988-3968-1984', -1, False),
    'SAO Arm Guard': ClothesMeta('7486-14958-7484', -1, False),
    'SAO Belt': ClothesMeta('8172-16188-8104', -1, False),
    'SAO Breast Guard': ClothesMeta('1242-2493-1253', -1, False),
    'SAO Shoulder Guard': ClothesMeta('2737-5459-2734', -1, False),
    'SAO Necklace': ClothesMeta('690-1380-690', -1, False),
    'G9 Base Bra': ClothesMeta('9748-19504-9754', CLOTHES_MIN_DIST_TO_SKIN, False),
    'G9 Base Panty': ClothesMeta('8624-17254-8630', -1, True),
    'G9 Base Shirt': ClothesMeta('8038-16050-8010', -1, False),
    'G9 Base Shorts': ClothesMeta('8256-16387-8130', -1, False),
    'VALBattleMage_Bracers': ClothesMeta('16106-30990-14938', -1, False),
    'VALBattleMage_Pantie': ClothesMeta('582-1067-484', PANTIE_SCALING, True),
    'BattleMage_Medallion': ClothesMeta('16291-32492-16202', -1, False),
    'BattleMage_Shirt': ClothesMeta('59984-117646-57745', -1, False),
    'BattleMage_Skirt': ClothesMeta('62529-124028-61617', -1, False),
    'KuJ Bracelet 02 L': ClothesMeta('3252-6456-3208', -1, False),
    'KuJ Bracelet 02 R': ClothesMeta('3252-6456-3208', -1, False),
    'KuJ Cloth Shoes': ClothesMeta('4622-9194-4574', -1, False),
    'Soaring Dragon Kung Fu Dress': ClothesMeta('15905-31513-15606', -1, False),
    'LLA Choker L01': ClothesMeta('1595-3150-1555', -1, False),
    'LLA Choker L02': ClothesMeta('7194-14321-7127', -1, False),
    'LLA Choker L03': ClothesMeta('1228-2409-1181', -1, False),
    'BW KK Panty G9': ClothesMeta('5031-9894-4862', PANTIE_SCALING, True),
    'DGA Collar 01': ClothesMeta('4619-10165-5564', -1, False),
    'DGA Collar 02': ClothesMeta('4298-9067-4858', -1, False),
    'DGA Collar 03': ClothesMeta('5312-11216-5904', -1, False),
    'DGA Handcuffs': ClothesMeta('8620-18253-9587', -1, False),
    'G9 Base Bikini': ClothesMeta('4046-7915-3870',PANTIE_SCALING, True),
    'AWl Bra': ClothesMeta('4604-8988-4376', -1, False),
    'SU Latex Outfit G9': ClothesMeta('46879-93730-46852', -1, False),
}
HairMeta = namedtuple('HairMeta', ['fingerprint', 'is_cards'])
# {o.name: o.data.daz_importer.DazFingerPrint for o in bpy.data.objects if isinstance(o.data, bpy.types.Mesh)}
HAIR = {
    "HS BBH Hair G9": HairMeta('43931-65146-21843', True)
}

QUINN_HEIGHT = 1.80169
MANNY_HEIGHT = 1.80625
NEW_WK_UV_MAP = 'WK UVs'
NEW_GP_UV_MAP = 'unified_gp_uv'
NEW_TOON_EYELASHES_UV_MAP = 'Toon Eyelashes UVs'
NEW_EYES_UV_MAP = 'optimised_eyes_uvs'
# {b.name:(list(b.head),list(b.tail),list(b.x_axis),list(b.y_axis), list(b.z_axis), b.roll, b.parent.name if b.parent is not None else None) for b in bpy.data.armatures['root'].edit_bones}
UE5_QUINN_BONE_HIERARCHY = {
    'pelvis': ([1.643580311187394e-21, -2.379462480545044, 98.69322204589844], [-5.365959345684158e-17, 6.169952392578125, 98.15494537353516], [-4.371138828673793e-08, 0.06283622235059738, 0.9980238080024719], [0.0, 0.9980236887931824, -0.06283620744943619], [-1.0, -7.450580596923828e-09, -4.330649971961975e-08], -1.5707963705062866, None), 'spine_01': ([-1.7832282883057326e-16, -2.2241389751434326, 101.16018676757812], [-8.601528861618135e-06, 2.622631788253784, 102.3366928100586], [-7.580956662422977e-08, -0.2358900010585785, 0.9717797636985779], [-1.7285346984863281e-06, 0.9717797040939331, 0.2358900010585785], [-1.0, -1.6540288925170898e-06, -4.805624485015869e-07], -1.5707966089248657, 'pelvis'), 'spine_02': ([-2.585645456423813e-12, -3.4006454944610596, 106.0069580078125], [6.939096692804014e-06, 4.171287536621094, 106.91241455078125], [-9.792394450869324e-08, -0.11873472481966019, 0.9929260015487671], [9.08970832824707e-07, 0.9929259419441223, 0.11873471736907959], [-1.0, 9.164214134216309e-07, 9.313225746154785e-09], -1.5707963705062866, 'spine_01'), 'spine_03': ([-6.531394891075293e-13, -4.3061017990112305, 113.57888793945312], [-2.2327729311655276e-05, 4.528489112854004, 113.03821563720703], [-1.2082978173566516e-07, 0.061085186898708344, 0.9981325268745422], [-2.5257468223571777e-06, 0.9981324076652527, -0.061085183173418045], [-0.9999999403953552, -2.522021532058716e-06, 3.329478204250336e-08], -1.5707963705062866, 'spine_02'), 'spine_04': ([-4.685855869990263e-12, -3.7654271125793457, 122.41348266601562], [-0.00010570820450084284, 13.352170944213867, 118.78105163574219], [3.7649272144335555e-08, 0.20758217573165894, 0.978217601776123], [-6.034970283508301e-06, 0.978217601776123, -0.20758220553398132], [-1.0, -5.9157609939575195e-06, 1.2889504432678223e-06], -1.5707956552505493, 'spine_03'), 'spine_05': ([-3.344880969424757e-11, -0.13299313187599182, 139.5310821533203], [-2.7126177883474156e-05, 12.1780366897583, 136.86053466796875], [8.304853338358953e-08, 0.2119927704334259, 0.977271318435669], [-2.1457672119140625e-06, 0.9772712588310242, -0.2119927704334259], [-1.0, -2.1010637283325195e-06, 5.364418029785156e-07], -1.570796012878418, 'spine_04'), 'neck_01': ([-3.247187123633921e-07, 2.392911911010742, 151.17532348632812], [-1.3728736121265683e-05, 8.094247817993164, 152.48060607910156], [-2.314143898729526e-08, -0.2231692671775818, 0.9747796654701233], [-2.294778823852539e-06, 0.9747797250747681, 0.2231692522764206], [-1.0, -2.2351741790771484e-06, -5.327165126800537e-07], -1.5707966089248657, 'spine_05'), 'neck_02': ([-4.690110131377878e-07, 1.0876317024230957, 156.8766632080078], [3.389727862668224e-05, 6.7141008377075195, 158.10252380371094], [2.89502395389718e-08, -0.21287983655929565, 0.977078378200531], [5.9604644775390625e-06, 0.9770783185958862, 0.21287982165813446], [-1.0, 5.841255187988281e-06, 1.300126314163208e-06], -1.5707956552505493, 'neck_01'), 'head': ([-6.118225428508595e-07, -0.138239786028862, 162.5031280517578], [-6.118301598689868e-07, 5.620223522186279, 162.5031280517578], [-4.371138828673793e-08, -1.0842021724855044e-19, 0.9999999403953552], [-1.3227514786620653e-12, 0.9999997615814209, 0.0], [-0.9999999403953552, -1.322751803922717e-12, -4.371138828673793e-08], -1.5707963705062866, 'neck_02'), 'clavicle_l': ([0.9313598871231079, 2.0850887298583984, 145.01641845703125], [0.5747506618499756, 14.892984390258789, 145.026123046875], [0.9893950819969177, 0.027655694633722305, -0.14259208738803864], [-0.02783212997019291, 0.9996123909950256, 0.0007574104238301516], [0.142557755112648, 0.003219262696802616, 0.9897812008857727], 0.14309033751487732, 'spine_05'), 'upperarm_l': ([16.055343627929688, 2.5078365802764893, 142.8367462158203], [15.660313606262207, 16.025672912597656, 142.07244873046875], [0.6031758785247803, -0.027443476021289825, -0.7971358299255371], [-0.029163861647248268, 0.9979805946350098, -0.056425753980875015], [0.7970747947692871, 0.057282209396362305, 0.601157546043396], 0.9238215684890747, 'clavicle_l'), 'lowerarm_l': ([32.395591735839844, 1.7643836736679077, 121.2420425415039], [36.79384994506836, 11.990445137023926, 114.43598175048828], [0.5010383725166321, -0.6184275150299072, -0.6053990125656128], [0.33709362149238586, 0.7837513089179993, -0.5216337442398071], [0.7970749139785767, 0.05728235840797424, 0.6011574268341064], 0.9047160148620605, 'upperarm_l'), 'lowerarm_twist_02_l': ([36.75381851196289, -3.6149399280548096, 115.97604370117188], [41.170310974121094, 6.622430801391602, 109.1988525390625], [0.5040994882583618, -0.6173170804977417, -0.6039894223213196], [0.3384912312030792, 0.7846179604530334, -0.5194211006164551], [0.794548511505127, 0.05739486217498779, 0.6044819951057434], 0.9005398750305176, 'lowerarm_l'), 'lowerarm_twist_01_l': ([41.112037658691406, -8.994263648986816, 110.71004486083984], [45.52853012084961, 1.2431068420410156, 103.93285369873047], [0.504099428653717, -0.6173170804977417, -0.6039894819259644], [0.33849114179611206, 0.7846180200576782, -0.5194211006164551], [0.7945486307144165, 0.05739480257034302, 0.6044818758964539], 0.9005399346351624, 'lowerarm_l'), 'lowerarm_correctiveRoot_l': ([32.39558792114258, 1.7643842697143555, 121.24205017089844], [33.47399139404297, 4.257497787475586, 119.58784484863281], [0.5045016407966614, -0.6181725859642029, -0.6027776598930359], [0.3390776515007019, 0.7838985919952393, -0.5201244354248047], [0.7940431833267212, 0.05801522731781006, 0.6050865650177002], 0.8994996547698975, 'lowerarm_l'), 'lowerarm_in_l': ([30.901811599731445, 0.9199522137641907, 118.4891357421875], [29.823406219482422, -1.5731613636016846, 120.14334106445312], [0.5045006275177002, -0.6181730628013611, -0.6027782559394836], [-0.33907824754714966, -0.783898651599884, 0.5201244354248047], [-0.794044017791748, -0.05801361799240112, -0.6050862073898315], -2.0549449920654297, 'lowerarm_correctiveRoot_l'), 'lowerarm_out_l': ([34.61832809448242, 2.2575509548187256, 121.5589370727539], [35.69673156738281, 4.750663757324219, 119.90473175048828], [0.5045014023780823, -0.6181726455688477, -0.6027777194976807], [0.3390776216983795, 0.7838984727859497, -0.5201244950294495], [0.7940431833267212, 0.05801516771316528, 0.6050864458084106], 0.8994998335838318, 'lowerarm_correctiveRoot_l'), 'lowerarm_fwd_l': ([32.57822799682617, -1.2115038633346558, 122.08163452148438], [35.10360336303711, -1.026995301246643, 124.00605773925781], [0.5045021772384644, -0.6181716918945312, -0.6027780175209045], [0.7940423488616943, 0.058014191687107086, 0.6050877571105957], [-0.33907848596572876, -0.7838993668556213, 0.5201227068901062], 0.2518954277038574, 'lowerarm_correctiveRoot_l'), 'lowerarm_bck_l': ([33.35948944091797, 3.5823652744293213, 117.71720123291016], [30.83411407470703, 3.3978567123413086, 115.79277801513672], [0.5045017004013062, -0.6181730628013611, -0.6027772426605225], [-0.7940425872802734, -0.05801421403884888, -0.6050878763198853], [0.33907923102378845, 0.7838983535766602, -0.5201240181922913], 1.5873812437057495, 'lowerarm_correctiveRoot_l'), 'hand_l': ([45.47026824951172, -14.373586654663086, 105.44406127929688], [43.32125473022461, -13.542137145996094, 103.7276611328125], [0.3689539432525635, -0.5668855905532837, -0.736555278301239], [-0.7479367256164551, 0.289375364780426, -0.5973712205886841], [0.5517820715904236, 0.7712991833686829, -0.31722840666770935], 1.53066885471344, 'lowerarm_l'), 'middle_metacarpal_l': ([46.46968460083008, -16.44110107421875, 103.30458068847656], [42.60802459716797, -15.22430419921875, 99.4931869506836], [0.4783998131752014, -0.5711278915405273, -0.6670432090759277], [-0.6944742798805237, 0.21882668137550354, -0.6854344606399536], [0.5374374985694885, 0.7911560535430908, -0.29194700717926025], 1.4172159433364868, 'hand_l'), 'middle_01_l': ([49.12984848022461, -19.616886138916016, 99.59544372558594], [44.8939094543457, -17.752235412597656, 97.92720794677734], [0.19370383024215698, -0.37209847569465637, -0.9077564477920532], [-0.861021876335144, 0.3790198564529419, -0.33909541368484497], [0.4702344536781311, 0.8472820520401001, -0.2469673454761505], 1.6094300746917725, 'middle_metacarpal_l'), 'middle_02_l': ([50.08279800415039, -21.44748306274414, 95.12958526611328], [47.540138244628906, -20.05021095275879, 95.06310272216797], [-0.10341450572013855, -0.23418131470680237, -0.9666770100593567], [-0.8761590719223022, 0.48147717118263245, -0.02290879562497139], [0.4707978367805481, 0.84459388256073, -0.25497183203697205], 1.815131664276123, 'middle_01_l'), 'middle_03_l': ([49.78268814086914, -22.127092361450195, 92.32423400878906], [47.25783920288086, -20.69822120666504, 92.39747619628906], [-0.1450585275888443, -0.2067243605852127, -0.9675863981246948], [-0.8700225353240967, 0.49236613512039185, 0.025238055735826492], [0.4711892604827881, 0.8454826474189758, -0.2512766718864441], 1.8395966291427612, 'middle_02_l'), 'pinky_metacarpal_l': ([47.45344543457031, -14.507956504821777, 102.49078369140625], [43.200714111328125, -14.737249374389648, 100.46082305908203], [0.43112051486968994, -0.19320553541183472, -0.8813663125038147], [-0.9013941287994385, -0.04860010743141174, -0.43026334047317505], [0.040294766426086426, 0.9799538850784302, -0.19510680437088013], 1.3201090097427368, 'hand_l'), 'pinky_01_l': ([49.48744583129883, -15.419486999511719, 98.33253479003906], [46.68084716796875, -15.430356979370117, 97.62963104248047], [0.2378927767276764, -0.2180425077676773, -0.9465011358261108], [-0.9700334668159485, -0.0037569552659988403, -0.24294181168079376], [0.0494157075881958, 0.9759319424629211, -0.21240229904651642], 1.5452070236206055, 'pinky_metacarpal_l'), 'pinky_02_l': ([50.17572784423828, -16.05034637451172, 95.5940170288086], [48.39780807495117, -15.924253463745117, 95.7746810913086], [-0.11321155726909637, -0.20264267921447754, -0.9726864695549011], [-0.9924094080924988, 0.07038325071334839, 0.10084408521652222], [0.04802554100751877, 0.9767200350761414, -0.20907270908355713], 1.8766344785690308, 'pinky_01_l'), 'pinky_03_l': ([49.9729118347168, -16.413387298583984, 93.85143280029297], [48.20936584472656, -16.267051696777344, 94.13075256347656], [-0.1692902147769928, -0.19689355790615082, -0.9656984210014343], [-0.9843875765800476, 0.08168255537748337, 0.15591253340244293], [0.048182517290115356, 0.9770163297653198, -0.20764772593975067], 1.9267386198043823, 'pinky_02_l'), 'ring_metacarpal_l': ([47.01427459716797, -15.500272750854492, 102.8641586303711], [43.04698181152344, -14.99853801727295, 99.87474822998047], [0.5249080657958984, -0.38184434175491333, -0.7607012987136841], [-0.7946079969406128, 0.10049229860305786, -0.5987482070922852], [0.30507323145866394, 0.918747067451477, -0.2506673038005829], 1.3189440965652466, 'hand_l'), 'ring_01_l': ([49.63501739501953, -17.40673065185547, 99.06614685058594], [45.71913528442383, -16.546310424804688, 97.65200805664062], [0.2508905529975891, -0.3447958827018738, -0.9045273661613464], [-0.9210846424102783, 0.2023860216140747, -0.332630455493927], [0.29775330424308777, 0.9166001081466675, -0.26680946350097656], 1.584036111831665, 'ring_metacarpal_l'), 'ring_02_l': ([50.70164489746094, -18.872594833374023, 95.22065734863281], [47.67238998413086, -17.78849220275879, 95.55603790283203], [-0.18676640093326569, -0.2261125147342682, -0.9560291171073914], [-0.9364487528800964, 0.3351341187953949, 0.10367788374423981], [0.29695504903793335, 0.9146357178688049, -0.27433449029922485], 1.9234178066253662, 'ring_01_l'), 'ring_03_l': ([50.09749221801758, -19.604034423828125, 92.12806701660156], [47.12635040283203, -18.46601676940918, 92.71232604980469], [-0.2598011791706085, -0.19246572256088257, -0.9462876319885254], [-0.9184846878051758, 0.351801335811615, 0.1806151121854782], [0.29814308881759644, 0.9160747528076172, -0.26817524433135986], 1.9720498323440552, 'ring_02_l'), 'thumb_01_l': ([43.83118438720703, -17.227506637573242, 103.68328857421875], [45.632896423339844, -15.049038887023926, 100.01427459716797], [-0.22926218807697296, -0.7833815813064575, -0.5777127146720886], [0.38898766040802, 0.47032874822616577, -0.7921359539031982], [0.8922597765922546, -0.4063299298286438, 0.1968972384929657], 1.5928101539611816, 'hand_l'), 'thumb_02_l': ([42.769290924072266, -20.855968475341797, 101.00743103027344], [43.92807388305664, -18.908594131469727, 99.52008056640625], [0.01768944412469864, -0.613513708114624, -0.7894860506057739], [0.4275033473968506, 0.7184340357780457, -0.5487200021743774], [0.9038407802581787, -0.32780134677886963, 0.2749878168106079], 1.3996456861495972, 'thumb_01_l'), 'thumb_03_l': ([42.817230224609375, -22.518943786621094, 98.86746215820312], [43.957862854003906, -20.349897384643555, 97.7092514038086], [0.07690688222646713, -0.500801146030426, -0.8621388673782349], [0.4208071529865265, 0.8002140522003174, -0.4272921681404114], [0.9038841724395752, -0.3299325108528137, 0.2722826600074768], 1.375587821006775, 'thumb_02_l'), 'index_metacarpal_l': ([45.52758026123047, -17.880849838256836, 103.4845199584961], [42.78710174560547, -15.444792747497559, 99.5517578125], [0.36226651072502136, -0.6580795049667358, -0.6600711941719055], [-0.5096747875213623, 0.4530584514141083, -0.7314159870147705], [0.78038090467453, 0.6013892292976379, -0.1712786853313446], 1.4389764070510864, 'hand_l'), 'index_01_l': ([47.47545623779297, -21.419279098510742, 99.93536376953125], [44.138954162597656, -19.061845779418945, 97.89932250976562], [0.23416607081890106, -0.4241243600845337, -0.8748055696487427], [-0.7309582233428955, 0.5164645910263062, -0.44605427980422974], [0.6409885883331299, 0.7438970804214478, -0.189078688621521], 1.5410600900650024, 'index_metacarpal_l'), 'index_02_l': ([48.5443229675293, -23.35521697998047, 95.9422607421875], [46.64827346801758, -21.881587982177734, 95.2973403930664], [0.07811330258846283, -0.3136444389820099, -0.9463220834732056], [-0.7625482678413391, 0.5926603078842163, -0.2593724727630615], [0.6421982049942017, 0.7418767213821411, -0.19287438690662384], 1.6429150104522705, 'index_01_l'), 'index_03_l': ([48.73854446411133, -24.13507843017578, 93.58926391601562], [46.84253692626953, -22.661617279052734, 92.94384002685547], [0.077602818608284, -0.31454959511756897, -0.9460635781288147], [-0.762531578540802, 0.5925928354263306, -0.2595749795436859], [0.6422797441482544, 0.7415471076965332, -0.19386713206768036], 1.6438645124435425, 'index_02_l'), 'wrist_inner_l': ([43.99201583862305, -14.006704330444336, 104.66062927246094], [42.128379821777344, -15.801892280578613, 105.90953063964844], [0.20173388719558716, -0.691785991191864, -0.6933510899543762], [-0.6486150622367859, -0.6247925758361816, 0.43466439843177795], [-0.7338951230049133, 0.3620314300060272, -0.5747443437576294], -3.0333235263824463, 'hand_l'), 'wrist_outer_l': ([46.52814483642578, -15.05665397644043, 106.61029815673828], [48.391780853271484, -13.261466026306152, 105.36139678955078], [0.2017352283000946, -0.6917865872383118, -0.6933500170707703], [0.6486150026321411, 0.6247925162315369, -0.43466436862945557], [0.7338948249816895, -0.3620300590991974, 0.5747457146644592], 1.07253897190094, 'hand_l'), 'weapon_l': ([45.023475646972656, -14.176285743713379, 103.61087799072266], [45.023475646972656, -11.303030014038086, 103.61087799072266], [1.0, 0.0, 8.940694584680386e-08], [0.0, 1.0, 0.0], [-8.940694584680386e-08, 0.0, 1.0], -8.940694584680386e-08, 'hand_l'), 'upperarm_twist_01_l': ([21.502090454101562, 2.2600176334381104, 135.6385040283203], [21.50179672241211, 2.2699975967407227, 135.637939453125], [0.6064955592155457, -0.02710215002298355, -0.7946248054504395], [-0.029372481629252434, 0.9979729652404785, -0.056456200778484344], [0.7945442199707031, 0.05758053809404373, 0.6044701337814331], 0.9196518659591675, 'upperarm_l'), 'upperarm_twistCor_01_l': ([21.502079010009766, 2.2600197792053223, 135.63851928710938], [21.501785278320312, 2.2699997425079346, 135.63795471191406], [0.6098030805587769, -0.026861833408474922, -0.7920976281166077], [-0.029372485354542732, 0.9979729652404785, -0.056456200778484344], [0.7920085787773132, 0.057693030685186386, 0.607778012752533], 0.9154823422431946, 'upperarm_twist_01_l'), 'upperarm_twist_02_l': ([26.94883918762207, 2.012200355529785, 128.4402618408203], [26.87037467956543, 4.697267055511475, 128.2884521484375], [0.603175938129425, -0.02744254283607006, -0.7971360087394714], [-0.029163556173443794, 0.9979810118675232, -0.056424371898174286], [0.7970749139785767, 0.05728113278746605, 0.6011576652526855], 0.9238215088844299, 'upperarm_l'), 'upperarm_tricep_l': ([26.766712188720703, 6.785869598388672, 127.99034881591797], [24.542009353637695, 6.405128002166748, 126.52593231201172], [0.5503312945365906, -0.004282146692276001, -0.8349354267120361], [-0.8268738985061646, -0.1415133774280548, -0.5442918539047241], [-0.11582383513450623, 0.9899270534515381, -0.08142006397247314], 0.9929671883583069, 'upperarm_twist_02_l'), 'upperarm_bicep_l': ([27.03663444519043, -1.2349298000335693, 128.03732299804688], [29.261337280273438, -0.8541884422302246, 129.50173950195312], [0.5503314137458801, -0.004283040761947632, -0.8349352478981018], [0.8268738389015198, 0.14151328802108765, 0.5442918539047241], [0.11582326889038086, -0.9899270534515381, 0.08142074942588806], 0.9843149185180664, 'upperarm_twist_02_l'), 'upperarm_twistCor_02_l': ([26.94883918762207, 2.0122013092041016, 128.4402618408203], [26.86959457397461, 4.69730281829834, 128.2894744873047], [0.6097986698150635, -0.026485955342650414, -0.7921136617660522], [-0.029453493654727936, 0.9979937076568604, -0.05604437738656998], [0.7920088171958923, 0.05750630050897598, 0.6077951788902283], 0.9154823422431946, 'upperarm_twist_02_l'), 'upperarm_correctiveRoot_l': ([16.055341720581055, 2.5078353881835938, 142.8367462158203], [15.856494903564453, 9.312311172485352, 142.4520263671875], [0.603175938129425, -0.027442876249551773, -0.7971357703208923], [-0.02916393242776394, 0.9979805946350098, -0.0564250648021698], [0.7970746755599976, 0.05728185176849365, 0.6011576056480408], 0.9238215088844299, 'upperarm_l'), 'upperarm_bck_l': ([17.2547664642334, 8.838794708251953, 141.57009887695312], [12.042211532592773, 10.646023750305176, 137.5636444091797], [0.6031763553619385, -0.027444183826446533, -0.7971355319023132], [-0.7645007967948914, 0.2650577425956726, -0.587607741355896], [0.2274133563041687, 0.9638418555259705, 0.13889560103416443], 0.9439528584480286, 'upperarm_correctiveRoot_l'), 'upperarm_fwd_l': ([17.711565017700195, -4.123559474945068, 140.28204345703125], [23.10483169555664, -4.820291042327881, 144.39491271972656], [0.6110705137252808, 0.18039757013320923, -0.7707459926605225], [0.7910051345825195, -0.10218638181686401, 0.6032152771949768], [0.03005889058113098, -0.9782711863517761, -0.20513847470283508], 1.101639986038208, 'upperarm_correctiveRoot_l'), 'upperarm_in_l': ([16.032108306884766, 0.4910411834716797, 135.38485717773438], [16.230955123901367, -6.313435077667236, 135.7695770263672], [0.996228039264679, 0.024404138326644897, -0.08328187465667725], [0.02916393056511879, -0.9979807734489441, 0.0564250648021698], [-0.08173367381095886, -0.05864088982343674, -0.994930624961853], 0.871405839920044, 'upperarm_correctiveRoot_l'), 'upperarm_out_l': ([20.484752655029297, 3.1250243186950684, 146.35325622558594], [20.285905838012695, 9.929500579833984, 145.96853637695312], [0.603175938129425, -0.027442872524261475, -0.7971358299255371], [-0.029163936153054237, 0.9979807734489441, -0.0564250648021698], [0.7970747351646423, 0.05728185921907425, 0.6011576056480408], 0.9238215088844299, 'upperarm_correctiveRoot_l'), 'clavicle_out_l': ([12.393397331237793, 2.452388286590576, 148.91917419433594], [12.982848167419434, 15.250591278076172, 148.75192260742188], [0.9949822425842285, -0.046980611979961395, -0.08833568543195724], [0.04600461944937706, 0.9988558888435364, -0.013053413480520248], [0.08884788304567337, 0.00892406515777111, 0.9960052967071533], 0.08875899016857147, 'clavicle_l'), 'clavicle_scap_l': ([9.198046684265137, 8.43429183959961, 141.3830108642578], [8.608595848083496, -4.3639116287231445, 141.55026245117188], [-0.9949837923049927, 0.04698058217763901, 0.08833283185958862], [-0.04600461944937706, -0.9988560080528259, 0.0130534116178751], [0.08884570002555847, 0.008924187161028385, 0.9960054755210876], 0.4641929566860199, 'clavicle_l'), 'clavicle_r': ([-0.9313597083091736, 2.085087776184082, 145.01596069335938], [-1.2918318510055542, -10.861546516418457, 145.0061492919922], [0.9893938899040222, -0.02765568532049656, 0.14260214567184448], [-0.02783212810754776, -0.9996124505996704, -0.0007575405761599541], [0.14256781339645386, -0.003219415433704853, -0.9897798299789429], -3.052915096282959, 'spine_05'), 'upperarm_r': ([-16.055299758911133, 2.5078377723693848, 142.83670043945312], [-16.4503231048584, -11.009780883789062, 143.60098266601562], [0.6031743288040161, 0.027443379163742065, 0.7971382737159729], [-0.029163848608732224, -0.9979808926582336, 0.0564255490899086], [0.7970770597457886, -0.05728205665946007, -0.6011601090431213], -0.03025894984602928, 'clavicle_r'), 'lowerarm_r': ([-32.39527893066406, 1.7643965482711792, 121.24234008789062], [-27.99693489074707, -8.461864471435547, 128.0485382080078], [0.5010374784469604, 0.6184276342391968, 0.605399489402771], [0.3370935320854187, -0.7837510704994202, 0.5216338634490967], [0.7970751523971558, -0.05728229880332947, -0.6011566519737244], 2.052152395248413, 'upperarm_r'), 'lowerarm_twist_02_r': ([-36.753570556640625, -3.6149983406066895, 115.97628021240234], [-32.33699417114258, -13.852567672729492, 122.75360107421875], [0.5040997266769409, 0.6173168420791626, 0.6039894223213196], [0.3384910821914673, -0.7846180200576782, 0.5194209218025208], [0.7945483922958374, -0.057395026087760925, -0.6044819951057434], 2.055636167526245, 'lowerarm_r'), 'lowerarm_twist_01_r': ([-41.11184310913086, -8.99438190460205, 110.71022033691406], [-36.69526672363281, -19.231952667236328, 117.487548828125], [0.5040988326072693, 0.6173170804977417, 0.603989839553833], [0.3384908139705658, -0.7846177816390991, 0.5194213390350342], [0.7945488095283508, -0.057394951581954956, -0.6044813394546509], 2.055636167526245, 'lowerarm_r'), 'lowerarm_correctiveRoot_r': ([-32.395565032958984, 1.7643935680389404, 121.24256134033203], [-31.31407928466797, -0.7358431816101074, 122.9014892578125], [0.5045015215873718, 0.6181720495223999, 0.6027780771255493], [0.3390780985355377, -0.7838989496231079, 0.5201234817504883], [0.7940428256988525, -0.05801433324813843, -0.6050868034362793], 2.0549449920654297, 'lowerarm_r'), 'lowerarm_out_r': ([-34.999427795410156, 2.5642249584198, 121.5035629272461], [-33.91794204711914, 0.06398820877075195, 123.16249084472656], [0.5045015215873718, 0.6181720495223999, 0.6027780771255493], [0.3390780985355377, -0.7838989496231079, 0.5201234817504883], [0.7940428256988525, -0.05801433324813843, -0.6050868034362793], 2.0549449920654297, 'lowerarm_correctiveRoot_r'), 'lowerarm_in_r': ([-31.367900848388672, 0.7733377814292908, 118.63737487792969], [-32.44938659667969, 3.2735748291015625, 116.97844696044922], [0.504501461982727, 0.6181721091270447, 0.6027781963348389], [-0.33907803893089294, 0.7838990092277527, -0.5201234817504883], [-0.7940430641174316, 0.058014124631881714, 0.6050867438316345], -0.8994998335838318, 'lowerarm_correctiveRoot_r'), 'lowerarm_fwd_r': ([-32.80950164794922, -1.024467945098877, 121.97138214111328], [-30.27691078186035, -1.2095037698745728, 120.04146575927734], [0.5045015215873718, 0.6181719303131104, 0.6027782559394836], [0.7940431237220764, -0.05801427364349365, -0.605086624622345], [-0.3390778601169586, 0.7838990688323975, -0.5201234817504883], -1.5873812437057495, 'lowerarm_correctiveRoot_r'), 'lowerarm_bck_r': ([-33.73392868041992, 3.518070697784424, 117.74079132080078], [-36.26652145385742, 3.70310640335083, 119.67070770263672], [0.504500687122345, 0.6181731224060059, 0.6027778387069702], [-0.7940433025360107, 0.05801421403884888, 0.6050863265991211], [0.33907845616340637, -0.7838981747627258, 0.5201244354248047], -0.25189530849456787, 'lowerarm_correctiveRoot_r'), 'hand_r': ([-45.47012710571289, -14.373771667480469, 105.44416046142578], [-47.61912536621094, -15.205216407775879, 107.1605453491211], [0.3689533472061157, 0.5668851137161255, 0.7365559935569763], [-0.7479372024536133, -0.2893759608268738, 0.597370445728302], [0.5517820119857788, -0.7712994813919067, 0.3172282576560974], -0.26303935050964355, 'lowerarm_r'), 'middle_metacarpal_r': ([-46.46969985961914, -16.441049575805664, 103.30459594726562], [-50.33140563964844, -17.657861709594727, 107.11603546142578], [0.478399395942688, 0.5711284279823303, 0.6670430302619934], [-0.6944742202758789, -0.2188267707824707, 0.6854345202445984], [0.5374380350112915, -0.7911556363105774, 0.29194730520248413], -0.1666814237833023, 'hand_r'), 'middle_01_r': ([-49.12989044189453, -19.616914749145508, 99.59545135498047], [-53.365779876708984, -21.481544494628906, 101.26366424560547], [0.1937030553817749, 0.3720989227294922, 0.9077562093734741], [-0.8610218167304993, -0.37902000546455383, 0.33909469842910767], [0.4702345132827759, -0.8472816944122314, 0.24696800112724304], -0.781812310218811, 'middle_metacarpal_r'), 'middle_02_r': ([-50.08280944824219, -21.447446823120117, 95.12962341308594], [-52.62548828125, -22.844728469848633, 95.19610595703125], [-0.10341409593820572, 0.23418118059635162, 0.966676652431488], [-0.8761590719223022, -0.48147687315940857, 0.022908655926585197], [0.47079750895500183, -0.8445940017700195, 0.2549716830253601], -1.2741796970367432, 'middle_01_r'), 'middle_03_r': ([-49.78269577026367, -22.127086639404297, 92.32426452636719], [-52.30756378173828, -23.55596923828125, 92.25102233886719], [-0.1450587511062622, 0.20672547817230225, 0.9675857424736023], [-0.8700221180915833, -0.49236616492271423, -0.025237906724214554], [0.4711892306804657, -0.8454821109771729, 0.25127777457237244], -1.3599953651428223, 'middle_02_r'), 'pinky_metacarpal_r': ([-47.45343017578125, -14.507929801940918, 102.49083709716797], [-51.706199645996094, -14.278635025024414, 104.52081298828125], [0.4311199188232422, 0.19320544600486755, 0.8813668489456177], [-0.9013944864273071, 0.04860010743141174, 0.430262953042984], [0.04029463231563568, -0.9799540042877197, 0.1951066255569458], -0.9307873249053955, 'hand_r'), 'pinky_01_r': ([-49.487403869628906, -15.419437408447266, 98.33252716064453], [-52.29402542114258, -15.408567428588867, 99.03543853759766], [0.23789328336715698, 0.2180425375699997, 0.9465010166168213], [-0.9700333476066589, 0.0037569254636764526, 0.2429424524307251], [0.049415841698646545, -0.9759320616722107, 0.21240234375], -1.1055866479873657, 'pinky_metacarpal_r'), 'pinky_02_r': ([-50.175743103027344, -16.050365447998047, 95.59400939941406], [-51.95362854003906, -16.176456451416016, 95.41334533691406], [-0.11321362107992172, 0.20264321565628052, 0.9726861119270325], [-0.9924092292785645, -0.07038353383541107, -0.10084603726863861], [0.04802533984184265, -0.9767200350761414, 0.20907337963581085], -1.4674952030181885, 'pinky_01_r'), 'pinky_03_r': ([-49.97294235229492, -16.413362503051758, 93.85144805908203], [-51.73645782470703, -16.559694290161133, 93.57212829589844], [-0.16929250955581665, 0.19689235091209412, 0.9656981825828552], [-0.9843870997428894, -0.08168178796768188, -0.15591515600681305], [0.048181548714637756, -0.9770165681838989, 0.2076464742422104], -1.5290188789367676, 'pinky_02_r'), 'ring_metacarpal_r': ([-47.01424789428711, -15.500242233276367, 102.8641357421875], [-50.98158264160156, -16.001981735229492, 105.85357666015625], [0.5249075889587402, 0.3818451762199402, 0.7607011795043945], [-0.7946081161499023, -0.10049223899841309, 0.5987480878829956], [0.30507367849349976, -0.9187467098236084, 0.25066810846328735], -0.5311552882194519, 'hand_r'), 'ring_01_r': ([-49.63505935668945, -17.406747817993164, 99.06613159179688], [-53.55094909667969, -18.267169952392578, 100.48027801513672], [0.2508915066719055, 0.34479594230651855, 0.9045271277427673], [-0.9210842847824097, -0.20238599181175232, 0.3326314091682434], [0.29775360226631165, -0.9166001081466675, 0.2668091058731079], -0.8644459843635559, 'ring_metacarpal_r'), 'ring_02_r': ([-50.70161056518555, -18.872573852539062, 95.22059631347656], [-53.73076248168945, -19.956640243530273, 94.8852310180664], [-0.18676531314849854, 0.2261132299900055, 0.9560290575027466], [-0.9364486336708069, -0.33513426780700684, -0.1036766842007637], [0.2969554364681244, -0.914635419845581, 0.2743348777294159], -1.4387024641036987, 'ring_01_r'), 'ring_03_r': ([-50.09751892089844, -19.604036331176758, 92.12811279296875], [-53.06856155395508, -20.742013931274414, 91.54386901855469], [-0.25980255007743835, 0.19246649742126465, 0.9462869763374329], [-0.9184845089912415, -0.35180068016052246, -0.1806163340806961], [0.29814186692237854, -0.9160746932029724, 0.26817622780799866], -1.557877779006958, 'ring_02_r'), 'thumb_01_r': ([-43.83119201660156, -17.227561950683594, 103.6832504272461], [-42.02947998046875, -19.406028747558594, 107.35226440429688], [-0.22926200926303864, 0.7833819389343262, 0.5777125358581543], [0.38898777961730957, -0.4703287184238434, 0.7921363711357117], [0.8922597765922546, 0.4063299298286438, -0.1968975067138672], 2.5057528018951416, 'hand_r'), 'thumb_02_r': ([-42.769248962402344, -20.856016159057617, 101.00740814208984], [-41.610469818115234, -22.80338478088379, 102.4947509765625], [0.01769164204597473, 0.6135136485099792, 0.7894860506057739], [0.4275035560131073, -0.7184345722198486, 0.5487191081047058], [0.9038405418395996, 0.3278004229068756, -0.2749895453453064], 2.7233684062957764, 'thumb_01_r'), 'thumb_03_r': ([-42.817203521728516, -22.518949508666992, 98.86741638183594], [-41.67657470703125, -24.687990188598633, 100.02561950683594], [0.07690916955471039, 0.500800609588623, 0.862139105796814], [0.4208073019981384, -0.8002148270606995, 0.4272908568382263], [0.9038840532302856, 0.32993197441101074, -0.2722843587398529], 2.9310925006866455, 'thumb_02_r'), 'index_metacarpal_r': ([-45.52760696411133, -17.88080406188965, 103.48453521728516], [-48.26808547973633, -20.316858291625977, 107.41728973388672], [0.36226603388786316, 0.6580796241760254, 0.6600714921951294], [-0.5096755027770996, -0.45305854082107544, 0.7314155697822571], [0.7803806662559509, -0.6013892889022827, 0.17127956449985504], 0.22178088128566742, 'hand_r'), 'index_01_r': ([-47.47541809082031, -21.419231414794922, 99.93534851074219], [-50.811920166015625, -23.776662826538086, 101.97138214111328], [0.23416456580162048, 0.4241241216659546, 0.8748061656951904], [-0.7309589385986328, -0.5164647102355957, 0.44605305790901184], [0.6409883499145508, -0.7438971400260925, 0.18907934427261353], -0.5047159194946289, 'index_metacarpal_r'), 'index_02_r': ([-48.54435348510742, -23.355239868164062, 95.94229888916016], [-50.440406799316406, -24.828872680664062, 96.58721923828125], [0.07811369001865387, 0.3136433959007263, 0.9463223218917847], [-0.7625481486320496, -0.5926605463027954, 0.259371817111969], [0.6421982049942017, -0.7418768405914307, 0.19287347793579102], -0.8429534435272217, 'index_01_r'), 'index_03_r': ([-48.738563537597656, -24.135107040405273, 93.58929443359375], [-50.63457489013672, -25.608570098876953, 94.2347183227539], [0.0776008665561676, 0.31455153226852417, 0.9460630416870117], [-0.7625318765640259, -0.592592716217041, 0.2595745623111725], [0.6422797441482544, -0.7415465116500854, 0.19386988878250122], -0.8415106534957886, 'index_02_r'), 'wrist_inner_r': ([-43.99203872680664, -14.006736755371094, 104.66065216064453], [-45.85566329956055, -12.21156120300293, 103.41175842285156], [0.20173506438732147, 0.6917867064476013, 0.6933497786521912], [-0.6486150622367859, 0.6247919797897339, -0.43466436862945557], [-0.7338945865631104, -0.36203011870384216, 0.574745774269104], -1.07253897190094, 'hand_r'), 'wrist_outer_r': ([-46.52817916870117, -15.056665420532227, 106.61029052734375], [-44.664554595947266, -16.85184097290039, 107.85918426513672], [0.20173373818397522, 0.6917856931686401, 0.6933510303497314], [0.6486148834228516, -0.6247918605804443, 0.4346643388271332], [0.7338950634002686, 0.36203116178512573, -0.5747443437576294], 3.0333235263824463, 'hand_r'), 'weapon_r': ([-45.02346420288086, -14.176295280456543, 103.61087799072266], [-45.02346420288086, -11.303059577941895, 103.61087799072266], [1.0, 0.0, -1.2665984172599565e-07], [0.0, 0.9999999403953552, 0.0], [1.2665984172599565e-07, 0.0, 1.0], 1.2665985593685036e-07, 'hand_r'), 'upperarm_twist_01_r': ([-21.50205421447754, 2.2600209712982178, 135.63865661621094], [-21.502347946166992, 2.2500410079956055, 135.63922119140625], [0.6076196432113647, 0.027020558714866638, 0.7937694787979126], [-0.029372481629252434, -0.9979729652404785, 0.05645619332790375], [0.7936857342720032, -0.057618867605924606, -0.6055985689163208], -0.04122878238558769, 'upperarm_r'), 'upperarm_twistCor_01_r': ([-21.50206756591797, 2.2600224018096924, 135.638671875], [-21.502361297607422, 2.25004243850708, 135.6392364501953], [0.6163171529769897, 0.02638484537601471, 0.7870570421218872], [-0.029372483491897583, -0.9979729056358337, 0.056456197053194046], [0.7869510054588318, -0.057912733405828476, -0.6142971515655518], -0.0522337332367897, 'upperarm_twist_01_r'), 'upperarm_twist_02_r': ([-26.94881820678711, 2.012204885482788, 128.44061279296875], [-27.02741813659668, -0.6774518489837646, 128.59268188476562], [0.6031560897827148, 0.027443591505289078, 0.7971519827842712], [-0.02916402742266655, -0.9979807734489441, 0.05642430856823921], [0.7970907688140869, -0.05728082358837128, -0.6011416912078857], -0.03025902807712555, 'upperarm_r'), 'upperarm_tricep_r': ([-27.034244537353516, 6.785796165466309, 127.98381042480469], [-29.26275062561035, 7.16718864440918, 129.4507293701172], [0.5503310561180115, 0.00428318977355957, 0.8349354863166809], [-0.8268740177154541, 0.1415134072303772, 0.544291615486145], [-0.11582325398921967, -0.9899270534515381, 0.08142083883285522], -0.984315037727356, 'upperarm_twist_02_r'), 'upperarm_bicep_r': ([-27.285133361816406, -1.2347993850708008, 128.0377960205078], [-25.05662727355957, -1.6161917448043823, 126.57087707519531], [0.5503309965133667, 0.004281878471374512, 0.8349356651306152], [0.8268740177154541, -0.1415134072303772, -0.544291615486145], [0.11582403630018234, 0.9899271726608276, -0.08141973614692688], -0.9929671883583069, 'upperarm_twist_02_r'), 'upperarm_twistCor_02_r': ([-26.948816299438477, 2.012205123901367, 128.44061279296875], [-27.028196334838867, -0.6774861812591553, 128.59165954589844], [0.6097874045372009, 0.026487205177545547, 0.7921233773231506], [-0.02945348620414734, -0.9979937672615051, 0.056044988334178925], [0.792018473148346, -0.057506315410137177, -0.6077879071235657], -0.05223371833562851, 'upperarm_twist_02_r'), 'upperarm_correctiveRoot_r': ([-16.05529022216797, 2.5078377723693848, 142.83670043945312], [-16.251861572265625, -4.218789577484131, 143.21702575683594], [0.6031803488731384, 0.02744337171316147, 0.79713374376297], [-0.029163863509893417, -0.9979807734489441, 0.05642610043287277], [0.7970725893974304, -0.05728260800242424, -0.6011662483215332], -0.03025902807712555, 'upperarm_r'), 'upperarm_bck_r': ([-17.499773025512695, 8.819766998291016, 141.54013061523438], [-22.652690887451172, 7.033214092254639, 145.50074768066406], [0.6031766533851624, 0.027442723512649536, 0.7971353530883789], [-0.7645007967948914, -0.2650578022003174, 0.5876078009605408], [0.22741252183914185, -0.9638419151306152, -0.13889676332473755], -0.8870223164558411, 'upperarm_correctiveRoot_r'), 'upperarm_in_r': ([-16.153444290161133, 0.7541298270225525, 135.94166564941406], [-15.956872940063477, 7.480756759643555, 135.56134033203125], [0.9962280988693237, -0.024404525756835938, 0.08327021449804306], [0.029163863509893417, 0.9979807734489441, -0.05642610415816307], [-0.08172500133514404, 0.05864175409078598, 0.9949281811714172], -0.08267513662576675, 'upperarm_correctiveRoot_r'), 'upperarm_fwd_r': ([-17.959993362426758, -4.123559474945068, 140.28199768066406], [-12.628429412841797, -3.4347991943359375, 136.2161865234375], [0.6110702157020569, -0.1803973764181137, 0.7707464098930359], [0.7910054922103882, 0.10218638181686401, -0.6032149195671082], [0.030058681964874268, 0.9782713055610657, 0.20513838529586792], -0.7369288802146912, 'upperarm_correctiveRoot_r'), 'upperarm_out_r': ([-20.733247756958008, 3.12514328956604, 146.35324096679688], [-20.929819107055664, -3.6014835834503174, 146.7335662841797], [0.6031804084777832, 0.027443375438451767, 0.7971336245536804], [-0.029163863509893417, -0.9979807734489441, 0.05642610788345337], [0.7970724701881409, -0.05728261172771454, -0.6011660695075989], -0.03025902435183525, 'upperarm_correctiveRoot_r'), 'clavicle_out_r': ([-12.652523040771484, 2.224787473678589, 148.89109802246094], [-12.056687355041504, -10.71204948425293, 149.06016540527344], [0.9949826002120972, 0.04698082059621811, 0.08834543824195862], [0.046004604548215866, -0.9988560080528259, 0.013053730130195618], [0.08885824680328369, -0.008923877030611038, -0.9960044622421265], 2.6773993968963623, 'clavicle_r'), 'clavicle_scap_r': ([-9.444722175598145, 8.430452346801758, 141.380615234375], [-10.040557861328125, 21.367290496826172, 141.2115478515625], [-0.994982123374939, -0.046980615705251694, -0.088335782289505], [-0.046004582196474075, 0.9988558292388916, -0.01305372640490532], [0.08884799480438232, -0.008924376219511032, -0.9960052371025085], 3.0528335571289062, 'clavicle_r'), 'clavicle_pec_r': ([-10.180403709411621, -11.844744682312012, 133.43756103515625], [-11.6460542678833, -24.35272979736328, 133.74649047851562], [0.9902726411819458, -0.11407146602869034, 0.07954677939414978], [-0.11634589731693268, -0.9929055571556091, 0.024523360654711723], [0.07618993520736694, -0.033540815114974976, -0.9965285062789917], -2.6478891372680664, 'spine_05'), 'spine_04_latissimus_l': ([12.8051118850708, 1.272215723991394, 130.64862060546875], [9.779763221740723, 13.4971284866333, 130.9520721435547], [0.9489966630935669, 0.23992939293384552, -0.2045462727546692], [-0.24015747010707855, 0.9704350829124451, 0.024088513106107712], [0.20427840948104858, 0.026263393461704254, 0.9785604476928711], 0.20899757742881775, 'spine_05'), 'spine_04_latissimus_r': ([-12.816333770751953, 1.270466685295105, 130.6429443359375], [-12.168295860290527, -11.301322937011719, 131.11563110351562], [0.9957676529884338, 0.05412319675087929, 0.07431498169898987], [0.05144238844513893, -0.9979708194732666, 0.03752271085977554], [0.07619869709014893, -0.03354056179523468, -0.996529757976532], 1.956608772277832, 'spine_05'), 'clavicle_pec_l': ([9.829132080078125, -11.606084823608398, 133.3883056640625], [8.363271713256836, 0.9018764495849609, 133.07931518554688], [0.9902834892272949, 0.11409129947423935, -0.07950976490974426], [-0.11636257916688919, 0.9929039478302002, -0.024528205394744873], [0.07614708691835403, 0.033541835844516754, 0.9965322017669678], 0.07818517833948135, 'spine_05'), 'thigh_r': ([-11.154600143432617, -2.6504101753234863, 95.47184753417969], [-14.52326488494873, -25.274572372436523, 95.14061737060547], [-0.041702598333358765, 0.020835785195231438, -0.9988996982574463], [-0.1472577303647995, -0.988991916179657, -0.014479388482868671], [-0.9882182478904724, 0.14649389684200287, 0.04431489109992981], 1.3315120935440063, 'pelvis'), 'calf_r': ([-13.062662124633789, -1.6972013711929321, 49.76964569091797], [-16.149436950683594, -22.308422088623047, 49.07025909423828], [-0.038887202739715576, 0.03970403969287872, -0.998451828956604], [-0.14802679419517517, -0.9884145259857178, -0.03353921324014664], [-0.9882183074951172, 0.14649377763271332, 0.04431524872779846], 1.0833139419555664, 'thigh_r'), 'foot_r': ([-14.684453964233398, -0.04133917763829231, 8.128620147705078], [-15.8768892288208, -8.003486633300781, 7.874436378479004], [0.004651784896850586, 0.031212609261274338, -0.999492883682251], [-0.14803746342658997, -0.9884781241416931, -0.03155619278550148], [-0.9889703392982483, 0.14811065793037415, 2.6345252990722656e-05], 1.153082013130188, 'calf_r'), 'ball_r': ([-16.62828254699707, -13.316017150878906, 1.165023684501648], [-16.590831756591797, -13.064607620239258, -6.885920524597168], [0.14803732931613922, 0.9884782433509827, 0.03155617043375969], [0.004649411886930466, 0.03121178410947323, -0.9995020627975464], [-0.988970935344696, 0.14811033010482788, 2.458132803440094e-05], -1.426717758178711, 'foot_r'), 'ankle_bck_r': ([-14.80817985534668, 4.074426174163818, 8.961807250976562], [-6.866764068603516, 2.828071117401123, 8.449187278747559], [-0.04547547176480293, 0.1182255744934082, -0.9919449090957642], [0.9859046339988708, -0.15473151206970215, -0.06364034116268158], [-0.1610090732574463, -0.980857253074646, -0.10952268540859222], 1.755211353302002, 'foot_r'), 'ankle_fwd_r': ([-15.015572547912598, -4.599365711212158, 9.74659252166748], [-22.956989288330078, -3.353010654449463, 10.259212493896484], [-0.045475490391254425, 0.11822542548179626, -0.9919449687004089], [-0.9859045743942261, 0.15473148226737976, 0.06364034116268158], [0.1610090434551239, 0.980857253074646, 0.1095224991440773], 1.5153030157089233, 'foot_r'), 'calf_twist_02_r': ([-13.652667999267578, -1.137923002243042, 35.89151382446289], [-13.65414810180664, -1.1478078365325928, 35.891197204589844], [-0.0426328182220459, 0.03836340084671974, -0.998340904712677], [-0.14800956845283508, -0.9884788393974304, -0.03166184574365616], [-0.9880656003952026, 0.14641621708869934, 0.04782599210739136], 1.1038062572479248, 'calf_r'), 'calf_twistCor_02_r': ([-13.652669906616211, -1.1379228830337524, 35.89151382446289], [-13.654150009155273, -1.1478077173233032, 35.891197204589844], [-0.04263293743133545, 0.03836340829730034, -0.9983408451080322], [-0.14800956845283508, -0.9884788393974304, -0.03166183829307556], [-0.9880656003952026, 0.14641623198986053, 0.04782611131668091], 1.1038061380386353, 'calf_twist_02_r'), 'calf_twist_01_r': ([-14.24267292022705, -0.5786444544792175, 22.013389587402344], [-17.32964324951172, -21.191198348999023, 21.3553524017334], [-0.04406324028968811, 0.03846868872642517, -0.9982856512069702], [-0.1480361521244049, -0.9884783625602722, -0.03155627101659775], [-0.9879997372627258, 0.14639224112033844, 0.04925131797790527], 1.1038062572479248, 'calf_r'), 'calf_correctiveRoot_r': ([-13.062662124633789, -1.6972014904022217, 49.7696418762207], [-13.794204711914062, -6.578701972961426, 49.60402297973633], [-0.042447566986083984, 0.040229879319667816, -0.998297393321991], [-0.1481218785047531, -0.9884005188941956, -0.03353431075811386], [-0.9880584478378296, 0.14644484221935272, 0.04790958762168884], 1.0800517797470093, 'calf_r'), 'calf_kneeBack_r': ([-11.942852020263672, 3.421912431716919, 50.19159698486328], [-7.097161769866943, 2.5180978775024414, 49.884925842285156], [-0.04612335190176964, 0.09027397632598877, -0.9948481917381287], [0.9811491370201111, -0.18300321698188782, -0.06209437921643257], [-0.18766599893569946, -0.978958785533905, -0.08013156056404114], 1.7259536981582642, 'calf_correctiveRoot_r'), 'calf_knee_r': ([-13.61893081665039, -6.282660961151123, 49.6568603515625], [-18.46462059020996, -5.378846645355225, 49.963531494140625], [-0.046123769134283066, 0.09027352929115295, -0.9948484301567078], [-0.9811495542526245, 0.1830032467842102, 0.06209440529346466], [0.18766596913337708, 0.9789589047431946, 0.08013102412223816], 1.5420457124710083, 'calf_correctiveRoot_r'), 'thigh_twist_01_r': ([-11.790620803833008, -2.332674026489258, 80.2377700805664], [-11.792093276977539, -2.342564105987549, 80.23763275146484], [-0.04432171583175659, 0.020466573536396027, -0.9988163709640503], [-0.1472468078136444, -0.9890044331550598, -0.01373286359012127], [-0.9881065487861633, 0.14646250009536743, 0.04684598743915558], 1.338954210281372, 'thigh_r'), 'thigh_twistCor_01_r': ([-11.790620803833008, -2.332674264907837, 80.23777770996094], [-11.792093276977539, -2.342564344406128, 80.23764038085938], [-0.04432158172130585, 0.020466558635234833, -0.9988163709640503], [-0.14724679291248322, -0.9890044331550598, -0.01373286172747612], [-0.9881065487861633, 0.14646250009536743, 0.04684586822986603], 1.3389543294906616, 'thigh_twist_01_r'), 'thigh_twist_02_r': ([-12.426642417907715, -2.0149378776550293, 65.00370788574219], [-12.428114891052246, -2.0248279571533203, 65.00357055664062], [-0.04432171583175659, 0.020466573536396027, -0.9988163709640503], [-0.1472468078136444, -0.9890044331550598, -0.01373286359012127], [-0.9881065487861633, 0.14646250009536743, 0.04684598743915558], 1.338954210281372, 'thigh_r'), 'thigh_twistCor_02_r': ([-12.426642417907715, -2.0149378776550293, 65.00370025634766], [-12.428114891052246, -2.0248279571533203, 65.0035629272461], [-0.04432158172130585, 0.020466558635234833, -0.9988163709640503], [-0.14724679291248322, -0.9890044331550598, -0.01373286172747612], [-0.9881065487861633, 0.14646250009536743, 0.04684586822986603], 1.3389543294906616, 'thigh_twist_02_r'), 'thigh_correctiveRoot_r': ([-11.1545991897583, -2.6504111289978027, 95.47183990478516], [-12.717266082763672, -13.145381927490234, 95.31819152832031], [-0.041708558797836304, 0.020835040137171745, -0.9989088773727417], [-0.1472577005624771, -0.9889920949935913, -0.014479032717645168], [-0.9882183074951172, 0.14649371802806854, 0.044318437576293945], 1.3315123319625854, 'thigh_r'), 'thigh_fwd_r': ([-11.110912322998047, -10.515107154846191, 101.63066864013672], [-21.597673416137695, -8.96054744720459, 102.10091400146484], [-0.041704606264829636, 0.020834339782595634, -0.9989127516746521], [-0.9882186055183411, 0.14649373292922974, 0.044313497841358185], [0.1472577154636383, 0.9889921545982361, 0.014479417353868484], 1.5945450067520142, 'thigh_correctiveRoot_r'), 'thigh_bck_r': ([-7.042505741119385, 7.976897716522217, 99.3572998046875], [3.4442543983459473, 6.422338008880615, 98.88705444335938], [-0.041704632341861725, 0.020834174007177353, -0.9989127516746521], [0.9882186055183411, -0.14649373292922974, -0.04431351274251938], [-0.1472577154636383, -0.9889922142028809, -0.014479253441095352], 1.6366710662841797, 'thigh_correctiveRoot_r'), 'thigh_out_r': ([-15.516352653503418, -0.7498688101768494, 101.59100341796875], [-13.953685760498047, 9.745101928710938, 101.7446517944336], [-0.041704535484313965, 0.020833924412727356, -0.9989127516746521], [0.1472577154636383, 0.9889921545982361, 0.014479035511612892], [0.9882185459136963, -0.14649376273155212, -0.044313423335552216], 1.6140568256378174, 'thigh_correctiveRoot_r'), 'thigh_in_r': ([-2.5934596061706543, -4.55668306350708, 84.67036437988281], [-4.156126976013184, -15.051652908325195, 84.51671600341797], [-0.0417088121175766, 0.020834919065237045, -0.9989100694656372], [-0.1472577601671219, -0.9889921545982361, -0.014479032717645168], [-0.9882183074951172, 0.14649376273155212, 0.044318392872810364], 1.3315123319625854, 'thigh_correctiveRoot_r'), 'thigh_bck_lwr_r': ([-7.867544651031494, 7.824114799499512, 89.27513885498047], [2.6192164421081543, 6.269554615020752, 88.80489349365234], [-0.04170459881424904, 0.020834174007177353, -0.9989127516746521], [0.9882186055183411, -0.14649377763271332, -0.04431350529193878], [-0.1472577452659607, -0.9889922142028809, -0.014479253441095352], 1.6366710662841797, 'thigh_correctiveRoot_r'), 'thigh_fwd_lwr_r': ([-11.396660804748535, -10.007637977600098, 95.84226989746094], [-21.8834228515625, -8.453078269958496, 96.31251525878906], [-0.04170459508895874, 0.020834332332015038, -0.9989126920700073], [-0.9882186055183411, 0.14649371802806854, 0.044313497841358185], [0.14725768566131592, 0.9889921545982361, 0.014479413628578186], 1.5945450067520142, 'thigh_correctiveRoot_r'), 'thigh_l': ([11.154585838317871, -2.6504464149475098, 95.4719009399414], [7.785913467407227, 19.973770141601562, 95.80313110351562], [-0.04170451685786247, -0.02083425410091877, 0.9989127516746521], [-0.1472577154636383, 0.9889921545982361, 0.014479349367320538], [-0.9882184863090515, -0.14649376273155212, -0.044313445687294006], -1.6140568256378174, 'pelvis'), 'calf_l': ([13.062654495239258, -1.697235107421875, 49.76959228515625], [9.975883483886719, 18.913965225219727, 50.46897888183594], [-0.038886602967977524, -0.03970364108681679, 0.9984545111656189], [-0.1480267345905304, 0.9884144067764282, 0.033539220690727234], [-0.9882185459136963, -0.14649374783039093, -0.04431328549981117], -1.6126508712768555, 'thigh_l'), 'foot_l': ([14.684441566467285, -0.04137573391199112, 8.128623962402344], [13.555163383483887, 7.499056339263916, 8.369344711303711], [0.0046496083959937096, -0.03121177665889263, 0.9995020627975464], [-0.14803749322891235, 0.9884780049324036, 0.0315561518073082], [-0.9889708161354065, -0.148110494017601, -2.4475622922182083e-05], -1.5684703588485718, 'calf_l'), 'ball_l': ([16.628278732299805, -13.316049575805664, 1.1650261878967285], [16.663747787475586, -13.554142951965332, 8.789551734924316], [0.14803726971149445, -0.9884783625602722, -0.03155617415904999], [0.004649650305509567, -0.03121175244450569, 0.9995020031929016], [-0.988970935344696, -0.1481102705001831, -2.442672848701477e-05], -1.4174141883850098, 'foot_l'), 'ankle_bck_l': ([14.534130096435547, 3.3269777297973633, 9.013772010803223], [22.148130416870117, 3.4453370571136475, 9.465774536132812], [-0.057188764214515686, -0.11035729944705963, 0.9922453165054321], [0.9981223940849304, 0.015515770763158798, 0.05925318971276283], [-0.021934475749731064, 0.9937707781791687, 0.10926276445388794], -1.5194954872131348, 'foot_l'), 'ankle_fwd_l': ([15.519561767578125, -4.047924995422363, 9.392012596130371], [7.905561923980713, -4.166284561157227, 8.940010070800781], [-0.057188816368579865, -0.11035724729299545, 0.9922452569007874], [-0.9981223940849304, -0.015515804290771484, -0.05925320088863373], [0.02193450927734375, -0.9937708973884583, -0.10926270484924316], -1.7406872510910034, 'foot_l'), 'calf_twist_02_l': ([13.652709007263184, -1.1379598379135132, 35.89148712158203], [13.651228904724121, -1.1280750036239624, 35.89180374145508], [-0.04405714198946953, -0.038572799414396286, 0.9982839822769165], [-0.14800956845283508, 0.9884787797927856, 0.031661827117204666], [-0.9880040287971497, -0.14636069536209106, -0.049258701503276825], -1.6177418231964111, 'calf_l'), 'calf_twistCor_02_l': ([13.652709007263184, -1.1379594802856445, 35.891483306884766], [13.651228904724121, -1.1280746459960938, 35.89179992675781], [-0.04405714198946953, -0.038572799414396286, 0.9982839822769165], [-0.14800956845283508, 0.9884787797927856, 0.031661827117204666], [-0.9880040287971497, -0.14636069536209106, -0.049258701503276825], -1.6177418231964111, 'calf_twist_02_l'), 'calf_twist_01_l': ([14.242714881896973, -0.5786824226379395, 22.013378143310547], [11.155747413635254, 20.033851623535156, 22.67141342163086], [-0.04406438767910004, -0.038468506187200546, 0.9982878565788269], [-0.1480361968278885, 0.9884786009788513, 0.031556230038404465], [-0.9879998564720154, -0.14639221131801605, -0.04925142601132393], -1.6177418231964111, 'calf_l'), 'calf_correctiveRoot_l': ([13.062698364257812, -1.697237253189087, 49.769596099853516], [12.331402778625488, 3.1826164722442627, 49.93516159057617], [-0.04244058579206467, -0.04023046791553497, 0.9982887506484985], [-0.14812181890010834, 0.988400399684906, 0.033534787595272064], [-0.9880579710006714, -0.14644509553909302, -0.04790729284286499], -1.616249442100525, 'calf_l'), 'calf_kneeBack_l': ([11.968843460083008, 3.4207024574279785, 50.18914794921875], [16.81289291381836, 4.324208736419678, 50.49580001831055], [-0.04614325612783432, -0.09026077389717102, 0.994848370552063], [0.9811480045318604, 0.1830025613307953, 0.06211147829890251], [-0.187666118144989, 0.9789600968360901, 0.08011487126350403], -1.5420761108398438, 'calf_correctiveRoot_l'), 'calf_knee_l': ([13.622564315795898, -6.285680770874023, 49.65370559692383], [8.778514862060547, -7.189187049865723, 49.34705352783203], [-0.04614312946796417, -0.09026098251342773, 0.994848370552063], [-0.9811480641365051, -0.1830025315284729, -0.06211148202419281], [0.1876661479473114, -0.9789600372314453, -0.08011513203382492], -1.7259578704833984, 'calf_correctiveRoot_l'), 'thigh_twist_01_l': ([11.79062557220459, -2.332709789276123, 80.23778533935547], [11.789153099060059, -2.322819709777832, 80.23792266845703], [-0.04671255871653557, -0.020822107791900635, 0.9986913204193115], [-0.14724679291248322, 0.9890044927597046, 0.01373287197202444], [-0.9879961609840393, -0.14641261100769043, -0.04926491901278496], -1.6190690994262695, 'thigh_l'), 'thigh_twistCor_01_l': ([11.790624618530273, -2.3327105045318604, 80.23778533935547], [11.789152145385742, -2.3228204250335693, 80.23792266845703], [-0.04671255871653557, -0.020822107791900635, 0.9986913204193115], [-0.14724679291248322, 0.9890044927597046, 0.01373287197202444], [-0.9879961609840393, -0.14641261100769043, -0.04926491901278496], -1.6190690994262695, 'thigh_twist_01_l'), 'thigh_twist_02_l': ([12.426664352416992, -2.0149731636047363, 65.00369262695312], [12.425191879272461, -2.0050830841064453, 65.00382995605469], [-0.04671255871653557, -0.020822107791900635, 0.9986913204193115], [-0.14724679291248322, 0.9890044927597046, 0.01373287197202444], [-0.9879961609840393, -0.14641261100769043, -0.04926491901278496], -1.6190690994262695, 'thigh_l'), 'thigh_twistCor_02_l': ([12.426665306091309, -2.0149736404418945, 65.0036849975586], [12.425192832946777, -2.0050835609436035, 65.00382232666016], [-0.04671255871653557, -0.020822107791900635, 0.9986913204193115], [-0.14724679291248322, 0.9890044927597046, 0.01373287197202444], [-0.9879961609840393, -0.14641261100769043, -0.04926491901278496], -1.6190690994262695, 'thigh_twist_02_l'), 'thigh_correctiveRoot_l': ([11.154585838317871, -2.650446653366089, 95.47189331054688], [9.58743953704834, 7.874608993530273, 95.62598419189453], [-0.04170452803373337, -0.020834119990468025, 0.9989127516746521], [-0.14725768566131592, 0.9889920949935913, 0.01447922084480524], [-0.9882185459136963, -0.14649373292922974, -0.04431343451142311], -1.6140568256378174, 'thigh_l'), 'thigh_bck_l': ([7.071425914764404, 7.974049091339111, 99.35671997070312], [17.58824920654297, 9.533065795898438, 99.82831573486328], [-0.04170478880405426, -0.02083456702530384, 0.9989126920700073], [0.9882186651229858, 0.14649377763271332, 0.04431374743580818], [-0.14725777506828308, 0.9889922142028809, 0.014479607343673706], -1.5945450067520142, 'thigh_correctiveRoot_l'), 'thigh_fwd_l': ([11.104652404785156, -10.49928092956543, 101.63545989990234], [0.58782958984375, -12.058297157287598, 101.16386413574219], [-0.041704922914505005, -0.020833972841501236, 0.9989128112792969], [-0.9882186651229858, -0.14649373292922974, -0.044313736259937286], [0.14725768566131592, -0.9889922142028809, -0.014478996396064758], -1.6366710662841797, 'thigh_correctiveRoot_l'), 'thigh_out_l': ([15.532029151916504, -0.7629575133323669, 101.58912658691406], [17.09917640686035, -11.288013458251953, 101.4350357055664], [-0.04170660674571991, -0.020834671333432198, 0.9989110231399536], [0.14725777506828308, -0.9889921545982361, -0.014479231089353561], [0.988218367099762, 0.14649377763271332, 0.04431596398353577], -1.3315120935440063, 'thigh_correctiveRoot_l'), 'thigh_bck_lwr_l': ([8.27383804321289, 7.799194812774658, 89.5567398071289], [18.790660858154297, 9.358211517333984, 90.02833557128906], [-0.04170478880405426, -0.02083456702530384, 0.9989126920700073], [0.9882186651229858, 0.14649377763271332, 0.04431374743580818], [-0.14725777506828308, 0.9889922142028809, 0.014479607343673706], -1.5945450067520142, 'thigh_correctiveRoot_l'), 'thigh_in_l': ([2.6235599517822266, -4.617661952972412, 84.73601531982422], [1.0564134120941162, 5.907393932342529, 84.89010620117188], [-0.04170452058315277, -0.020834119990468025, 0.9989127516746521], [-0.1472577154636383, 0.9889920949935913, 0.01447922084480524], [-0.9882185459136963, -0.14649376273155212, -0.04431343451142311], -1.6140568256378174, 'thigh_correctiveRoot_l'), 'thigh_fwd_lwr_l': ([11.562108039855957, -10.524767875671387, 95.75232696533203], [1.0452852249145508, -12.083784103393555, 95.28073120117188], [-0.041704922914505005, -0.020833972841501236, 0.9989128112792969], [-0.9882186651229858, -0.14649373292922974, -0.044313736259937286], [0.14725768566131592, -0.9889922142028809, -0.014478996396064758], -1.6366710662841797, 'thigh_correctiveRoot_l'), 'ik_foot_root': ([0.0, 0.0, 0.0], [0.0, 16.815488815307617, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], 0.0, None), 'ik_foot_l': ([14.711810111999512, -0.04145237058401108, 8.143789291381836], [13.1902437210083, 16.70410919189453, 8.321894645690918], [0.005696810316294432, -0.010117617435753345, 0.9999327063560486], [-0.09048602730035782, 0.9958416223526001, 0.010591746307909489], [-0.9958815574645996, -0.09054026007652283, 0.004757619462907314], -1.5655581951141357, 'ik_foot_root'), 'ik_foot_r': ([-14.711830139160156, -0.041416238993406296, 8.143787384033203], [-16.233386993408203, -16.78697967529297, 7.965682029724121], [0.005694568157196045, 0.010116450488567352, -0.9999494552612305], [-0.09048542380332947, -0.9958414435386658, -0.010591740719974041], [-0.9958816766738892, 0.09053952246904373, -0.00475938618183136], 1.3429853916168213, 'ik_foot_root'), 'ik_hand_root': ([0.0, 0.0, 0.0], [0.0, 115.94215393066406, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], 0.0, None), 'ik_hand_gun': ([-45.55488967895508, -14.400569915771484, 105.64070129394531], [-81.57011413574219, -21.950908660888672, 132.49493408203125], [0.5101569890975952, 0.35417795181274414, 0.7837716341018677], [-0.7905885577201843, -0.16574130952358246, 0.5894909501075745], [0.3386879861354828, -0.920373797416687, 0.1954549252986908], -0.5627425312995911, 'ik_hand_root'), 'ik_hand_l': ([45.55500793457031, -14.400376319885254, 105.6406021118164], [9.539794921875, -6.850041389465332, 78.7863540649414], [0.5101572275161743, -0.35417842864990234, -0.7837711572647095], [-0.79058837890625, 0.1657411754131317, -0.5894913077354431], [0.33868828415870667, 0.9203735589981079, -0.19545534253120422], 1.2974462509155273, 'ik_hand_gun'), 'ik_hand_r': ([-45.554893493652344, -14.400568962097168, 105.64070129394531], [-81.57011413574219, -21.950910568237305, 132.49493408203125], [0.5101569890975952, 0.35417795181274414, 0.7837716937065125], [-0.7905885577201843, -0.165741428732872, 0.5894910097122192], [0.33868807554244995, -0.920373797416687, 0.19545471668243408], -0.5627425312995911, 'ik_hand_gun'), 'interaction': ([-1.7763568394002505e-15, 1.8735013540549517e-16, 3.552713678800501e-15], [-1.7486012637846216e-15, 1.0, 3.553979993056646e-15], [1.0, -2.7755575615628914e-17, 5.0550926292136644e-17], [2.7755575615628914e-17, 1.0, 1.266314256145179e-18], [-5.0550926292136644e-17, -1.266314256145179e-18, 1.0], -5.0550926292136644e-17, None), 'center_of_mass': ([-1.7763568394002505e-15, 1.8735013540549517e-16, 3.552713678800501e-15], [-1.7486012637846216e-15, 1.0, 3.553979993056646e-15], [1.0, -2.7755575615628914e-17, 5.0550926292136644e-17], [2.7755575615628914e-17, 1.0, 1.266314256145179e-18], [-5.0550926292136644e-17, -1.266314256145179e-18, 1.0], -5.0550926292136644e-17, None)
}
UE5_MANNY_BONE_HIERARCHY = {
    'pelvis': ([-3.4314998759166215e-17, -2.2808661460876465, 95.89678192138672], [-3.5306464058817175e-17, 5.7597222328186035, 95.38624572753906], [-4.371138828673793e-08, 0.06336726993322372, 0.9979901909828186], [0.0, 0.9979899525642395, -0.06336726248264313], [-0.9999998807907104, -7.450580596923828e-09, -4.377216100692749e-08], -1.5707963705062866, None), 'spine_01': ([4.3202198143678383e-16, -2.0478618144989014, 99.56644439697266], [4.394248376375496e-16, 4.626304626464844, 100.842529296875], [-4.371138828673793e-08, -0.18779587745666504, 0.9822081327438354], [0.0, 0.9822081923484802, 0.18779589235782623], [-1.0, 0.0, -4.842877388000488e-08], -1.5707963705062866, 'pelvis'), 'spine_02': ([2.4642779827050603e-16, -3.323951005935669, 106.24060821533203], [2.500792514270009e-16, 3.85463547706604, 107.16781616210938], [-4.371138828673793e-08, -0.12809890508651733, 0.9917614459991455], [0.0, 0.9917615056037903, 0.12809891998767853], [-1.0, 1.4901161193847656e-08, -4.284083843231201e-08], -1.5707963705062866, 'spine_01'), 'spine_03': ([1.2350714458857041e-16, -4.251158237457275, 113.41919708251953], [1.224850140492509e-16, 4.256053447723389, 112.8860092163086], [-4.371138828673793e-08, 0.0625520721077919, 0.9980417490005493], [0.0, 0.9980418086051941, -0.0625520795583725], [-1.0, -3.725290298461914e-09, -4.377216100692749e-08], -1.5707963705062866, 'spine_02'), 'spine_04': ([-1.2879212725362482e-16, -3.717967987060547, 121.92640686035156], [-0.00015252000594045967, 15.457849502563477, 118.73353576660156], [2.2830988655186957e-08, 0.1642438918352127, 0.9864197373390198], [-7.852911949157715e-06, 0.9864194393157959, -0.16424386203289032], [-0.9999998211860657, -7.763504981994629e-06, 1.3150274753570557e-06], -1.5707956552505493, 'spine_03'), 'spine_05': ([-4.0858163206924925e-12, -0.5250967741012573, 141.1022186279297], [1.9961485242525612e-10, 12.443700790405273, 138.7840118408203], [-4.371001693925791e-08, 0.17596350610256195, 0.9843966960906982], [0.0, 0.9843966960906982, -0.17596349120140076], [-1.0, 0.0, -4.0978193283081055e-08], -1.5707963705062866, 'spine_04'), 'neck_01': ([-1.5987330816841228e-12, 1.5667136907577515, 152.8044891357422], [7.227859977199103e-11, 6.529595851898193, 154.0229034423828], [-4.37131362218679e-08, -0.23842522501945496, 0.9711608290672302], [0.0, 0.9711607694625854, 0.23842519521713257], [-1.0, 1.4901161193847656e-08, -3.725290298461914e-08], -1.5707963705062866, 'spine_05'), 'neck_02': ([-3.5366425066296614e-11, 0.34828680753707886, 157.76736450195312], [8.400277692999225e-06, 5.156027317047119, 158.77877807617188], [1.6804932556624408e-08, -0.2058658003807068, 0.9785802960395813], [1.6987323760986328e-06, 0.9785803556442261, 0.20586583018302917], [-1.0000001192092896, 1.6689300537109375e-06, 3.6135315895080566e-07], -1.5707961320877075, 'neck_01'), 'head': ([-7.321034750651378e-11, -0.66312575340271, 162.57510375976562], [-7.594790768505888e-11, 4.249850273132324, 162.57510375976562], [-4.371138828673793e-08, -5.421010862427522e-20, 1.0], [-5.572101482477498e-13, 0.9999999403953552, 0.0], [-1.0, -5.572101482477498e-13, -4.371138828673793e-08], -1.5707963705062866, 'neck_02'), 'clavicle_l': ([1.4279042482376099, 1.7398159503936768, 146.30105590820312], [0.7565951347351074, 16.354270935058594, 146.35374450683594], [0.9872232675552368, 0.045897822827100754, -0.15258972346782684], [-0.04588591679930687, 0.998940110206604, 0.003601417411118746], [0.15259331464767456, 0.0034463165793567896, 0.9882830381393433], 0.1532718390226364, 'spine_05'), 'upperarm_l': ([19.009883880615234, 2.5572352409362793, 143.58352661132812], [18.557323455810547, 16.416963577270508, 142.86732482910156], [0.5760335922241211, -0.023414012044668198, -0.817090630531311], [-0.03259207308292389, 0.9981368780136108, -0.05157875269651413], [0.8167759776115417, 0.05634177103638649, 0.5741972327232361], 0.9573954939842224, 'clavicle_l'), 'lowerarm_l': ([35.006996154785156, 1.907007098197937, 120.89195251464844], [39.598907470703125, 12.27879810333252, 113.3424072265625], [0.4682975709438324, -0.6460624933242798, -0.6027441620826721], [0.33700767159461975, 0.7612022161483765, -0.5540730953216553], [0.8167760968208313, 0.056341707706451416, 0.5741972327232361], 0.9373546242713928, 'upperarm_l'), 'lowerarm_twist_02_l': ([39.260860443115234, -3.9616291522979736, 115.41680908203125], [43.87297821044922, 6.729214668273926, 108.33940887451172], [0.504099428653717, -0.6173170804977417, -0.6039894223213196], [0.3384905755519867, 0.7846178412437439, -0.5194215774536133], [0.794548749923706, 0.057395339012145996, 0.6044815182685852], 0.9005401730537415, 'lowerarm_l'), 'lowerarm_twist_01_l': ([43.51472473144531, -9.83026123046875, 109.94166564941406], [48.1268424987793, 0.8605823516845703, 102.86426544189453], [0.5040993690490723, -0.6173170804977417, -0.6039894223213196], [0.3384905457496643, 0.7846179008483887, -0.5194215774536133], [0.7945488095283508, 0.05739527940750122, 0.6044815182685852], 0.900540292263031, 'lowerarm_l'), 'lowerarm_correctiveRoot_l': ([35.006988525390625, 1.9070026874542236, 120.89195251464844], [36.439971923828125, 4.979351997375488, 118.7169189453125], [0.49292558431625366, -0.6440500617027283, -0.5849989652633667], [0.35576915740966797, 0.7627772688865662, -0.5399991869926453], [0.7940104603767395, 0.05805480480194092, 0.6051256060600281], 0.898339569568634, 'lowerarm_l'), 'lowerarm_in_l': ([32.448978424072266, 1.7618275880813599, 117.67044830322266], [31.015995025634766, -1.3105214834213257, 119.8454818725586], [0.49292421340942383, -0.644049882888794, -0.5850000381469727], [-0.3557690680027008, -0.7627770900726318, 0.5399990677833557], [-0.7940109968185425, -0.05805410444736481, -0.6051244735717773], -2.063460111618042, 'lowerarm_correctiveRoot_l'), 'lowerarm_out_l': ([37.660160064697266, 2.127636194229126, 121.88644409179688], [39.093143463134766, 5.199985504150391, 119.71141052246094], [0.49292558431625366, -0.6440500617027283, -0.5849989652633667], [0.35576915740966797, 0.7627772688865662, -0.5399991869926453], [0.7940104603767395, 0.05805480480194092, 0.6051256060600281], 0.898339569568634, 'lowerarm_correctiveRoot_l'), 'lowerarm_fwd_l': ([34.19145965576172, -2.0002317428588867, 121.6115951538086], [37.38960647583008, -1.7663984298706055, 124.0489501953125], [0.49292632937431335, -0.644049346446991, -0.5849993228912354], [0.794009804725647, 0.05805421620607376, 0.6051266193389893], [-0.35576969385147095, -0.7627780437469482, 0.5399978756904602], 0.21838371455669403, 'lowerarm_correctiveRoot_l'), 'lowerarm_bck_l': ([36.1588020324707, 5.164400577545166, 117.14408874511719], [32.960655212402344, 4.930567264556885, 114.70673370361328], [0.4929260015487671, -0.6440501809120178, -0.5849986672401428], [-0.7940099239349365, -0.05805423855781555, -0.6051267385482788], [0.3557702898979187, 0.7627771496772766, -0.5399986505508423], 1.620790958404541, 'lowerarm_correctiveRoot_l'), 'hand_l': ([47.76859664916992, -15.698895454406738, 104.46652221679688], [45.770538330078125, -14.899195671081543, 102.24182891845703], [0.4577661156654358, -0.6224098205566406, -0.6348671913146973], [-0.6455091834068298, 0.25835758447647095, -0.718727707862854], [0.611365795135498, 0.7388216257095337, -0.2835038900375366], 1.431865930557251, 'lowerarm_l'), 'wrist_inner_l': ([46.398956298828125, -14.804336547851562, 103.71019744873047], [44.299476623535156, -16.51398468017578, 105.21029663085938], [0.24255630373954773, -0.7908658385276794, -0.561869740486145], [-0.6782752871513367, -0.5523331165313721, 0.4846344292163849], [-0.6936202049255371, 0.26355114579200745, -0.6703968048095703], -2.842864751815796, 'hand_l'), 'wrist_outer_l': ([49.14667892456055, -16.381174087524414, 106.09811401367188], [51.246158599853516, -14.671525001525879, 104.59801483154297], [0.24255481362342834, -0.7908651232719421, -0.5618714094161987], [0.6782752275466919, 0.5523333549499512, -0.4846344590187073], [0.6936208009719849, -0.26355305314064026, 0.6703953146934509], 0.942070484161377, 'hand_l'), 'index_metacarpal_l': ([48.13905715942383, -19.700077056884766, 103.23075103759766], [44.60148620605469, -18.00130271911621, 98.84896087646484], [0.38282468914985657, -0.7142105102539062, -0.5859596729278564], [-0.6014023423194885, 0.288798987865448, -0.7449233531951904], [0.701256513595581, 0.6375724673271179, -0.31896862387657166], 1.5212289094924927, 'hand_l'), 'index_01_l': ([50.53190612792969, -23.731523513793945, 99.6780014038086], [47.65989303588867, -21.493892669677734, 97.8365478515625], [0.11282521486282349, -0.5410324931144714, -0.8333992958068848], [-0.7039246559143066, 0.5484387874603271, -0.4513365626335144], [0.701256275177002, 0.6375725269317627, -0.31896835565567017], 1.7043224573135376, 'index_metacarpal_l'), 'index_02_l': ([50.99222183227539, -25.938932418823242, 96.27772521972656], [49.15165328979492, -24.20271110534668, 95.70167541503906], [-0.07187905162572861, -0.38190513849258423, -0.921402096748352], [-0.7092767357826233, 0.6690656542778015, -0.22198501229286194], [0.7012559175491333, 0.6375730037689209, -0.3189682364463806], 1.8071624040603638, 'index_01_l'), 'index_03_l': ([50.80569839477539, -26.92997932434082, 93.8866958618164], [49.04929733276367, -25.02024269104004, 93.84251403808594], [-0.2238839715719223, -0.22783096134662628, -0.9476122856140137], [-0.676842212677002, 0.7359311580657959, -0.017025763168931007], [0.7012564539909363, 0.637572169303894, -0.31896883249282837], 1.88884699344635, 'index_02_l'), 'middle_metacarpal_l': ([49.68858337402344, -18.12984848022461, 102.91677856445312], [45.60165023803711, -17.049423217773438, 98.52157592773438], [0.48489779233932495, -0.630664587020874, -0.6059176325798035], [-0.6701858639717102, 0.17717084288597107, -0.7207366824150085], [0.5618941187858582, 0.755561113357544, -0.336752712726593], 1.4446133375167847, 'hand_l'), 'middle_01_l': ([52.645591735839844, -21.97576904296875, 99.22176361083984], [48.3818473815918, -19.488704681396484, 97.68757629394531], [0.062225744128227234, -0.444548636674881, -0.8935907483100891], [-0.8248658180236816, 0.48114845156669617, -0.2968045175075531], [0.5618938207626343, 0.7555611729621887, -0.33675307035446167], 1.7572219371795654, 'middle_metacarpal_l'), 'middle_02_l': ([52.96723175048828, -24.273645401000977, 94.60277557373047], [51.00453186035156, -22.7706356048584, 94.70014190673828], [-0.23431962728500366, -0.24504151940345764, -0.9407705068588257], [-0.793328583240509, 0.6075206398963928, 0.03935575485229492], [0.5618935823440552, 0.7555617690086365, -0.33675241470336914], 1.9339780807495117, 'middle_01_l'), 'middle_03_l': ([52.38752365112305, -24.879878997802734, 92.27529907226562], [50.55530548095703, -23.294431686401367, 92.77534484863281], [-0.36851924657821655, -0.13582521677017212, -0.9196439981460571], [-0.740587055683136, 0.6408416628837585, 0.20211973786354065], [0.5618933439254761, 0.7555616497993469, -0.3367529809474945], 2.0150935649871826, 'middle_02_l'), 'thumb_01_l': ([46.22671127319336, -18.49582290649414, 102.95843505859375], [47.79423522949219, -16.872589111328125, 99.20677185058594], [-0.23677723109722137, -0.8517224788665771, -0.46744561195373535], [0.35804468393325806, 0.37076956033706665, -0.8569329977035522], [0.9031836986541748, -0.3702685534954071, 0.21716484427452087], 1.5851043462753296, 'hand_l'), 'thumb_02_l': ([45.190101623535156, -22.22467041015625, 100.91194915771484], [46.66328048706055, -20.206193923950195, 99.1011734008789], [-0.045745253562927246, -0.6483667492866516, -0.7599526643753052], [0.4773745536804199, 0.6540747880935669, -0.5867707133293152], [0.8775084018707275, -0.3896239697933197, 0.2795928716659546], 1.4289443492889404, 'thumb_01_l'), 'thumb_03_l': ([45.04893493652344, -24.22552490234375, 98.56673431396484], [46.52424621582031, -21.890268325805664, 97.19071197509766], [0.037845149636268616, -0.5249375104904175, -0.850299060344696], [0.4780656397342682, 0.7567256689071655, -0.44589167833328247], [0.8775084018707275, -0.3896239101886749, 0.27959302067756653], 1.3890994787216187, 'thumb_02_l'), 'pinky_metacarpal_l': ([50.945125579833984, -16.074230194091797, 101.9043197631836], [46.668941497802734, -16.44251823425293, 99.41109466552734], [0.5016597509384155, -0.2771416902542114, -0.8194695711135864], [-0.8615046739578247, -0.0741974413394928, -0.5022994875907898], [0.07840554416179657, 0.9579600691795349, -0.27598077058792114], 1.3245493173599243, 'hand_l'), 'pinky_01_l': ([53.53984451293945, -17.628040313720703, 97.96849822998047], [49.98955535888672, -16.967432022094727, 96.73526763916016], [0.26354044675827026, -0.2970004081726074, -0.9177893400192261], [-0.9303706884384155, 0.17311564087867737, -0.3231741189956665], [0.25486648082733154, 0.9390536546707153, -0.23069749772548676], 1.54279625415802, 'pinky_metacarpal_l'), 'pinky_02_l': ([54.54551696777344, -18.76139259338379, 94.46621704101562], [52.581844329833984, -18.212364196777344, 94.5316390991211], [-0.09220176190137863, -0.21388913691043854, -0.9724969267845154], [-0.9625705480575562, 0.26912763714790344, 0.03206917271018028], [0.2548665404319763, 0.9390536546707153, -0.2306974083185196], 1.8280504941940308, 'pinky_01_l'), 'pinky_03_l': ([54.35741424560547, -19.197731018066406, 92.48229217529297], [52.417091369628906, -18.613323211669922, 92.71752166748047], [-0.17436721920967102, -0.19003373384475708, -0.9661692976951599], [-0.9511250853538513, 0.28647035360336304, 0.11530692875385284], [0.25486665964126587, 0.9390537142753601, -0.23069685697555542], 1.8911104202270508, 'pinky_02_l'), 'ring_metacarpal_l': ([50.33090591430664, -17.132648468017578, 102.40469360351562], [46.20484161376953, -16.915407180786133, 98.55731964111328], [0.5906050205230713, -0.46483317017555237, -0.6596332788467407], [-0.7308336496353149, 0.03847908973693848, -0.6814702749252319], [0.34215202927589417, 0.8845618963241577, -0.31698980927467346], 1.3041715621948242, 'hand_l'), 'ring_01_l': ([53.68851852416992, -19.776769638061523, 98.71566009521484], [49.29463195800781, -17.99098777770996, 97.20719146728516], [0.15623807907104492, -0.38420724868774414, -0.909930944442749], [-0.8828345537185669, 0.3588053584098816, -0.3030865788459778], [0.4429363012313843, 0.8506723642349243, -0.2831324338912964], 1.664319396018982, 'ring_metacarpal_l'), 'ring_02_l': ([54.46611404418945, -21.688983917236328, 94.1869125366211], [52.46003723144531, -20.637603759765625, 94.20745086669922], [-0.13914057612419128, -0.24675150215625763, -0.9590378403663635], [-0.8856902122497559, 0.46418821811676025, 0.009067757055163383], [0.442936509847641, 0.8506721258163452, -0.28313273191452026], 1.8633527755737305, 'ring_01_l'), 'ring_03_l': ([54.15095138549805, -22.24786949157715, 92.01470184326172], [52.2207145690918, -21.120872497558594, 92.38107299804688], [-0.27847933769226074, -0.16964119672775269, -0.9453418254852295], [-0.8522067666053772, 0.4975733757019043, 0.1617542803287506], [0.4429367184638977, 0.8506718873977661, -0.28313323855400085], 1.9552093744277954, 'ring_02_l'), 'weapon_l': ([47.76859664916992, -15.698893547058105, 102.63404846191406], [45.32147216796875, -15.185871124267578, 100.80938720703125], [0.5101569890975952, -0.35417693853378296, -0.783771812915802], [-0.7905886769294739, 0.16574138402938843, -0.5894904732704163], [0.3386874198913574, 0.9203740358352661, -0.1954542100429535], 1.29744553565979, 'hand_l'), 'upperarm_twist_01_l': ([24.342262268066406, 2.3404924869537354, 136.0196533203125], [24.341968536376953, 2.3504724502563477, 136.0190887451172], [0.6064954400062561, -0.027102157473564148, -0.7946248650550842], [-0.029372481629252434, 0.9979729652404785, -0.056456200778484344], [0.7945442795753479, 0.057580530643463135, 0.6044700145721436], 0.919651985168457, 'upperarm_l'), 'upperarm_twistCor_01_l': ([24.342248916625977, 2.3404929637908936, 136.0196533203125], [24.341989517211914, 2.350471019744873, 136.01904296875], [0.636078953742981, -0.030626431107521057, -0.7710159420967102], [-0.025939758867025375, 0.9977985620498657, -0.06103472784161568], [0.7711877822875977, 0.0588228739798069, 0.6338842511177063], 0.8819054961204529, 'upperarm_twist_01_l'), 'upperarm_twist_02_l': ([29.67462921142578, 2.1237497329711914, 128.455810546875], [29.56857681274414, 5.371639251708984, 128.28797912597656], [0.5760335922241211, -0.023413285613059998, -0.8170906901359558], [-0.03259187936782837, 0.9981370568275452, -0.05157773196697235], [0.8167760372161865, 0.0563410222530365, 0.5741972923278809], 0.9573954343795776, 'upperarm_l'), 'upperarm_tricep_l': ([29.71939468383789, 6.968902587890625, 127.56551361083984], [27.09709358215332, 6.505381107330322, 125.69551849365234], [0.5815087556838989, -0.007867008447647095, -0.8135021924972534], [-0.8058822154998779, -0.14244887232780457, -0.5746845006942749], [-0.11136143654584885, 0.9897708892822266, -0.08917522430419922], 0.9592723846435547, 'upperarm_twist_02_l'), 'upperarm_bicep_l': ([28.747356414794922, -2.4242286682128906, 127.10053253173828], [31.369657516479492, -1.960707187652588, 128.9705352783203], [0.5815102458000183, -0.007865309715270996, -0.8135011196136475], [0.805881142616272, 0.14244863390922546, 0.5746860504150391], [0.11136199533939362, -0.9897709488868713, 0.089173823595047], 0.9433759450912476, 'upperarm_twist_02_l'), 'upperarm_twistCor_02_l': ([29.674612045288086, 2.1237497329711914, 128.45582580566406], [29.602296829223633, 5.36998176574707, 128.24380493164062], [0.6647401452064514, -0.0339350663125515, -0.7463035583496094], [-0.022223811596632004, 0.9976274371147156, -0.06515796482563019], [0.7467440366744995, 0.05989881977438927, 0.6624088287353516], 0.8441556096076965, 'upperarm_twist_02_l'), 'upperarm_correctiveRoot_l': ([19.009889602661133, 2.557234764099121, 143.58351135253906], [18.75910758972168, 10.237445831298828, 143.1866455078125], [0.5760336518287659, -0.02341296523809433, -0.8170906901359558], [-0.03259218484163284, 0.9981370568275452, -0.051577553153038025], [0.8167760372161865, 0.05634116753935814, 0.5741972923278809], 0.9573954343795776, 'upperarm_l'), 'upperarm_bck_l': ([19.921419143676758, 10.174810409545898, 142.2115478515625], [13.886870384216309, 12.221551895141602, 137.89865112304688], [0.5760338306427002, -0.023414045572280884, -0.8170902729034424], [-0.7842631936073303, 0.26599904894828796, -0.5605134963989258], [0.230469211935997, 0.9636887907981873, 0.1348617672920227], 0.9745635390281677, 'upperarm_correctiveRoot_l'), 'upperarm_fwd_l': ([20.29393768310547, -6.072179317474365, 140.7523651123047], [26.534412384033203, -6.8658223152160645, 145.1832275390625], [0.5844900608062744, 0.1843377649784088, -0.7901841402053833], [0.8110263347625732, -0.10314363241195679, 0.5758450031280518], [0.024647481739521027, -0.9774358868598938, -0.20978927612304688], 1.139780879020691, 'upperarm_correctiveRoot_l'), 'upperarm_in_l': ([18.830799102783203, 1.5954713821411133, 135.6397247314453], [19.081581115722656, -6.0847392082214355, 136.03659057617188], [0.9928188323974609, 0.02638886868953705, -0.11668902635574341], [0.032592181116342545, -0.9981369972229004, 0.051577553153038025], [-0.1151077151298523, -0.05501013249158859, -0.9918311238288879], 1.01083242893219, 'upperarm_correctiveRoot_l'), 'upperarm_out_l': ([23.875741958618164, 2.93072509765625, 146.73345947265625], [23.62495994567871, 10.61093521118164, 146.3365936279297], [0.5760336518287659, -0.02341296896338463, -0.8170906901359558], [-0.03259218856692314, 0.9981370568275452, -0.05157756060361862], [0.8167760372161865, 0.056341174989938736, 0.5741972923278809], 0.9573954343795776, 'upperarm_correctiveRoot_l'), 'clavicle_out_l': ([14.06929874420166, 1.860264539718628, 150.70509338378906], [14.476118087768555, 16.483684539794922, 150.5447235107422], [0.9947131872177124, -0.02875377982854843, -0.09858453273773193], [0.027807287871837616, 0.9995532631874084, -0.010961746796965599], [0.09885567426681519, 0.008162426762282848, 0.995068371295929], 0.09890332072973251, 'clavicle_l'), 'clavicle_scap_l': ([10.634288787841797, 9.761409759521484, 142.8348846435547], [10.227469444274902, -4.862010955810547, 142.99525451660156], [-0.9947164058685303, 0.028753522783517838, 0.098553866147995], [-0.02780728042125702, -0.9995531439781189, 0.01096174493432045], [0.0988251268863678, 0.00816330686211586, 0.9950713515281677], 0.6521273255348206, 'clavicle_l'), 'clavicle_r': ([-1.427872657775879, 1.7398524284362793, 146.30099487304688], [-2.1051013469696045, -13.003473281860352, 146.24783325195312], [0.987223207950592, -0.04589799791574478, 0.15259771049022675], [-0.04588591679930687, -0.998940110206604, -0.0036019880790263414], [0.15260131657123566, -0.003446128685027361, -0.9882819056510925], 3.1381962299346924, 'spine_05'), 'upperarm_r': ([-19.009883880615234, 2.5572714805603027, 143.5830078125], [-19.46243667602539, -11.302225112915039, 144.2991943359375], [0.5760335922241211, 0.0234137624502182, 0.8170918226242065], [-0.03259206563234329, -0.9981369972229004, 0.05157851055264473], [0.8167765736579895, -0.05634164810180664, -0.5742007493972778], -0.1696859747171402, 'clavicle_r'), 'lowerarm_r': ([-35.00673294067383, 1.9070528745651245, 120.89181518554688], [-30.41480827331543, -8.464766502380371, 128.44137573242188], [0.4682980179786682, 0.6460621356964111, 0.6027440428733826], [0.3370077610015869, -0.7612022161483765, 0.5540727972984314], [0.8167755007743835, -0.05634203553199768, -0.57419753074646], 2.0302627086639404, 'upperarm_r'), 'lowerarm_twist_02_r': ([-39.260589599609375, -3.9615631103515625, 115.41668701171875], [-34.648460388183594, -14.652436256408691, 122.49411010742188], [0.5040990114212036, 0.6173171401023865, 0.6039896607398987], [0.33849042654037476, -0.7846176624298096, 0.5194216966629028], [0.7945488691329956, -0.05739554762840271, -0.6044811010360718], 2.0556342601776123, 'lowerarm_r'), 'lowerarm_twist_01_r': ([-43.514434814453125, -9.830169677734375, 109.94156646728516], [-38.902305603027344, -20.521041870117188, 117.01898956298828], [0.5040991306304932, 0.617317259311676, 0.6039895415306091], [0.33849048614501953, -0.7846177220344543, 0.5194218158721924], [0.7945491075515747, -0.05739523470401764, -0.6044812202453613], 2.0556342601776123, 'lowerarm_r'), 'lowerarm_correctiveRoot_r': ([-35.006980895996094, 1.9070382118225098, 120.89198303222656], [-33.560943603515625, -1.1933014392852783, 123.08683013916016], [0.49292439222335815, 0.6440495252609253, 0.5850003361701965], [0.35576891899108887, -0.7627774477005005, 0.5399987697601318], [0.7940109968185425, -0.058054059743881226, -0.6051245331764221], 2.063460111618042, 'lowerarm_r'), 'lowerarm_out_r': ([-37.897396087646484, 2.0926992893218994, 121.91439056396484], [-36.45135498046875, -1.0076406002044678, 124.10923767089844], [0.4929261803627014, 0.644050121307373, 0.584998607635498], [0.3557697534561157, -0.7627773284912109, 0.5399987697601318], [0.7940098643302917, -0.058054566383361816, -0.6051265001296997], 2.063460111618042, 'lowerarm_correctiveRoot_r'), 'lowerarm_in_r': ([-32.68295669555664, 1.7555944919586182, 117.63418579101562], [-34.12899398803711, 4.855934143066406, 115.43933868408203], [0.4929257333278656, 0.6440496444702148, 0.5849992632865906], [-0.35576894879341125, 0.7627774477005005, -0.5399988889694214], [-0.7940104603767395, 0.05805477499961853, 0.6051256656646729], -0.898339569568634, 'lowerarm_correctiveRoot_r'), 'lowerarm_fwd_r': ([-34.44465637207031, -2.0101914405822754, 121.63212585449219], [-31.217370986938477, -2.246155023574829, 119.17256927490234], [0.4929257035255432, 0.644049346446991, 0.5849996209144592], [0.7940104007720947, -0.058054208755493164, -0.6051256656646729], [-0.35576915740966797, 0.7627779245376587, -0.5399982929229736], -1.620790958404541, 'lowerarm_correctiveRoot_r'), 'lowerarm_bck_r': ([-36.38434982299805, 5.167483806610107, 117.1427993774414], [-39.61163330078125, 5.40344762802124, 119.60235595703125], [0.4929257929325104, 0.644049882888794, 0.584999144077301], [-0.7940102219581604, 0.05805429816246033, 0.6051259636878967], [0.355769544839859, -0.7627775073051453, 0.5399986505508423], -0.21838371455669403, 'lowerarm_correctiveRoot_r'), 'hand_r': ([-47.768314361572266, -15.698809623718262, 104.46641540527344], [-49.76641082763672, -16.498523712158203, 106.69115447998047], [0.4577668607234955, 0.6224098801612854, 0.6348664164543152], [-0.6455087065696716, -0.25835707783699036, 0.7187281847000122], [0.6113657355308533, -0.7388216853141785, 0.2835036814212799], -0.031692638993263245, 'lowerarm_r'), 'wrist_inner_r': ([-46.39899444580078, -14.80429458618164, 103.70999145507812], [-48.498512268066406, -13.094613075256348, 102.2098617553711], [0.24255499243736267, 0.790865421295166, 0.5618711113929749], [-0.6782748103141785, 0.552333414554596, -0.4846351146697998], [-0.6936212778091431, -0.26355230808258057, 0.6703951954841614], -0.942070484161377, 'hand_r'), 'wrist_outer_r': ([-49.146697998046875, -16.38119888305664, 106.09798431396484], [-47.04718017578125, -18.090879440307617, 107.59811401367188], [0.24255505204200745, 0.7908656597137451, 0.5618706941604614], [0.6782748699188232, -0.5523332357406616, 0.48463520407676697], [0.6936210989952087, 0.2635520398616791, -0.6703954935073853], 2.842864751815796, 'hand_r'), 'pinky_metacarpal_r': ([-50.94509506225586, -16.074193954467773, 101.90398406982422], [-55.2210578918457, -15.705924034118652, 104.39707946777344], [0.5016597509384155, 0.2771407663822174, 0.819469690322876], [-0.8615046143531799, 0.07419764995574951, 0.5022993087768555], [0.07840487360954285, -0.9579602479934692, 0.2759798765182495], -0.7613198161125183, 'hand_r'), 'pinky_01_r': ([-53.539794921875, -17.62799835205078, 97.96847534179688], [-57.090110778808594, -18.288612365722656, 99.20171356201172], [0.26353996992111206, 0.2970008850097656, 0.917789101600647], [-0.930370569229126, -0.17311584949493408, 0.3231736421585083], [0.25486668944358826, -0.9390534162521362, 0.2306981086730957], -0.9301532506942749, 'pinky_metacarpal_r'), 'pinky_02_r': ([-54.545501708984375, -18.761402130126953, 94.46619415283203], [-56.50917053222656, -19.3104305267334, 94.40077209472656], [-0.09220190346240997, 0.2138892114162445, 0.9724968671798706], [-0.9625702500343323, -0.26912805438041687, -0.03206923231482506], [0.25486695766448975, -0.9390535354614258, 0.23069751262664795], -1.3801498413085938, 'pinky_01_r'), 'pinky_03_r': ([-54.35740280151367, -19.197704315185547, 92.48226165771484], [-56.297725677490234, -19.78211212158203, 92.24703216552734], [-0.17436763644218445, 0.19003519415855408, 0.9661689400672913], [-0.9511250257492065, -0.28647035360336304, -0.11530689895153046], [0.25486642122268677, -0.939053475856781, 0.23069843649864197], -1.491767406463623, 'pinky_02_r'), 'ring_metacarpal_r': ([-50.33089828491211, -17.132600784301758, 102.40498352050781], [-54.45711135864258, -17.349849700927734, 106.25250244140625], [0.5906054973602295, 0.46483340859413147, 0.6596325039863586], [-0.7308329939842224, -0.0384790301322937, 0.6814708113670349], [0.3421524167060852, -0.8845615983009338, 0.3169897198677063], -0.3364998698234558, 'hand_r'), 'ring_01_r': ([-53.68850326538086, -19.77680015563965, 98.71568298339844], [-58.08243179321289, -21.562599182128906, 100.22416687011719], [0.15623784065246582, 0.38420772552490234, 0.9099308848381042], [-0.8828346729278564, -0.358805388212204, 0.3030868172645569], [0.44293642044067383, -0.8506721258163452, 0.2831329107284546], -0.8158660531044006, 'ring_metacarpal_r'), 'ring_02_r': ([-54.46609878540039, -21.689008712768555, 94.18688201904297], [-56.47216033935547, -22.740379333496094, 94.16634368896484], [-0.13914072513580322, 0.24675193428993225, 0.9590378403663635], [-0.8856905698776245, -0.4641876518726349, -0.009067794308066368], [0.44293591380119324, -0.8506723046302795, 0.28313326835632324], -1.2987148761749268, 'ring_01_r'), 'ring_03_r': ([-54.1509895324707, -22.247905731201172, 92.01468658447266], [-56.08121109008789, -23.374893188476562, 91.6483154296875], [-0.27848032116889954, 0.16964095830917358, 0.9453414082527161], [-0.8522065281867981, -0.49757298827171326, -0.1617555022239685], [0.44293615221977234, -0.8506720662117004, 0.28313344717025757], -1.561535120010376, 'ring_02_r'), 'middle_metacarpal_r': ([-49.6885986328125, -18.129798889160156, 102.9169921875], [-53.77564239501953, -19.21025276184082, 107.31230926513672], [0.4848979711532593, 0.6306639909744263, 0.6059181690216064], [-0.6701862215995789, -0.17717090249061584, 0.7207363247871399], [0.5618935823440552, -0.7555615305900574, 0.33675259351730347], -0.053529489785432816, 'hand_r'), 'middle_01_r': ([-52.645606994628906, -21.97580337524414, 99.22179412841797], [-56.909339904785156, -24.462860107421875, 100.75597381591797], [0.06222556531429291, 0.44454827904701233, 0.8935906887054443], [-0.8248658180236816, -0.4811483919620514, 0.2968038320541382], [0.5618934631347656, -0.7555614709854126, 0.33675310015678406], -0.6935766339302063, 'middle_metacarpal_r'), 'middle_02_r': ([-52.96720886230469, -24.273611068725586, 94.602783203125], [-54.92995071411133, -25.776653289794922, 94.50541687011719], [-0.23431964218616486, 0.24504214525222778, 0.9407707452774048], [-0.793328583240509, -0.6075207591056824, -0.03935485705733299], [0.5618937611579895, -0.7555614709854126, 0.33675283193588257], -1.3067479133605957, 'middle_01_r'), 'middle_03_r': ([-52.38749313354492, -24.87990379333496, 92.2752685546875], [-54.219749450683594, -26.46538543701172, 91.77520751953125], [-0.3685222864151001, 0.1358262598514557, 0.9196425676345825], [-0.7405866980552673, -0.6408419013023376, -0.2021215856075287], [0.5618921518325806, -0.755561351776123, 0.3367556035518646], -1.6593598127365112, 'middle_02_r'), 'index_metacarpal_r': ([-48.13909149169922, -19.700098037719727, 103.23098754882812], [-51.676727294921875, -21.398902893066406, 107.61285400390625], [0.3828241229057312, 0.7142106890678406, 0.5859597325325012], [-0.6014026403427124, -0.2887989580631256, 0.744922935962677], [0.7012564539909363, -0.637572169303894, 0.3189689517021179], 0.16283421218395233, 'hand_r'), 'index_01_r': ([-50.53190231323242, -23.731504440307617, 99.67799377441406], [-53.403873443603516, -25.969100952148438, 101.51942443847656], [0.11282672733068466, 0.5410323143005371, 0.8333990573883057], [-0.7039245367050171, -0.548438310623169, 0.45133745670318604], [0.7012561559677124, -0.6375730037689209, 0.3189675211906433], -0.29698172211647034, 'index_metacarpal_r'), 'index_02_r': ([-50.9921989440918, -25.938905715942383, 96.27778625488281], [-52.83282470703125, -27.675182342529297, 96.8538589477539], [-0.07187743484973907, 0.38190585374832153, 0.921401858329773], [-0.7092762589454651, -0.6690657734870911, 0.22198684513568878], [0.701256513595581, -0.6375726461410522, 0.3189675211906433], -0.7277937531471252, 'index_01_r'), 'index_03_r': ([-50.80569839477539, -26.930004119873047, 93.88668823242188], [-52.562156677246094, -28.839801788330078, 93.93087005615234], [-0.22388391196727753, 0.22783057391643524, 0.9476121664047241], [-0.676842451095581, -0.7359309792518616, 0.01702522113919258], [0.7012561559677124, -0.6375726461410522, 0.3189685344696045], -1.2024484872817993, 'index_02_r'), 'thumb_01_r': ([-46.2266960144043, -18.495803833007812, 102.9579849243164], [-44.659236907958984, -20.118967056274414, 106.70948791503906], [-0.23677730560302734, 0.8517224788665771, 0.4674455523490906], [0.3580450415611267, -0.3707691431045532, 0.8569327592849731], [0.9031833410263062, 0.3702687621116638, -0.21716532111167908], 2.3766536712646484, 'hand_r'), 'thumb_02_r': ([-45.19009780883789, -22.22470474243164, 100.9119873046875], [-43.716941833496094, -24.24315071105957, 102.72273254394531], [-0.04574437439441681, 0.6483662128448486, 0.759952962398529], [0.47737452387809753, -0.6540750861167908, 0.5867699384689331], [0.877508282661438, 0.3896232545375824, -0.2795933485031128], 2.7948570251464844, 'thumb_01_r'), 'thumb_03_r': ([-45.04890441894531, -24.22550392150879, 98.56678771972656], [-43.57361602783203, -26.56072425842285, 99.94278717041016], [0.03784579038619995, 0.5249371528625488, 0.850299060344696], [0.47806572914123535, -0.7567256689071655, 0.44589126110076904], [0.877508282661438, 0.3896235525608063, -0.2795935869216919], 3.0295119285583496, 'thumb_02_r'), 'weapon_r': ([-47.76831817626953, -15.69880485534668, 102.6339340209961], [-50.21548843383789, -16.211837768554688, 104.45863342285156], [0.5101574063301086, 0.3541777729988098, 0.783771276473999], [-0.7905880808830261, -0.16574153304100037, 0.5894913673400879], [0.3386881351470947, -0.9203736782073975, 0.1954544484615326], -0.5627421140670776, 'hand_r'), 'upperarm_twist_01_r': ([-24.342252731323242, 2.3405263423919678, 136.0193328857422], [-24.342546463012695, 2.3305463790893555, 136.0198974609375], [0.6076135039329529, 0.027021007612347603, 0.7937741875648499], [-0.029372479766607285, -0.9979729652404785, 0.056456197053194046], [0.7936905026435852, -0.05761865898966789, -0.6055923700332642], -0.041221003979444504, 'upperarm_r'), 'upperarm_twistCor_01_r': ([-24.342262268066406, 2.340528964996338, 136.01934814453125], [-24.34252166748047, 2.3305509090423584, 136.01995849609375], [0.6366257667541504, 0.03058464080095291, 0.7705671191215515], [-0.025939758867025375, -0.9977985620498657, 0.06103472784161568], [0.7707378268241882, -0.058844614773988724, -0.6344357132911682], 0.0774538516998291, 'upperarm_twist_01_r'), 'upperarm_twist_02_r': ([-29.67461395263672, 2.1237823963165283, 128.4556427001953], [-29.780237197875977, -1.1109371185302734, 128.622802734375], [0.5760608911514282, 0.02341349795460701, 0.8170725703239441], [-0.03259214758872986, -0.9981369972229004, 0.05158054828643799], [0.816757321357727, -0.05634366720914841, -0.5742279291152954], -0.1696859747171402, 'upperarm_r'), 'upperarm_tricep_r': ([-29.949310302734375, 6.968440055847168, 127.58340454101562], [-32.560977935791016, 7.430081844329834, 129.44581604003906], [0.5815087556838989, 0.007867425680160522, 0.8135020732879639], [-0.8058822751045227, 0.1424487829208374, 0.5746843218803406], [-0.11136114597320557, -0.9897708892822266, 0.08917546272277832], -0.9433759450912476, 'upperarm_twist_02_r'), 'upperarm_bicep_r': ([-28.97818374633789, -2.42470383644104, 127.11780548095703], [-26.36651611328125, -2.886345863342285, 125.2553939819336], [0.5815086364746094, 0.007866829633712769, 0.8135021924972534], [0.8058822751045227, -0.14244887232780457, -0.5746843814849854], [0.1113615557551384, 0.9897708892822266, -0.08917498588562012], -0.9592723846435547, 'upperarm_twist_02_r'), 'upperarm_twistCor_02_r': ([-29.67460060119629, 2.1237845420837402, 128.45567321777344], [-29.746623992919922, -1.1092839241027832, 128.66683959960938], [0.6647424697875977, 0.033935729414224625, 0.7463022470474243], [-0.022224245592951775, -0.9976273775100708, 0.06515957415103912], [0.7467437982559204, -0.05990035459399223, -0.6624165177345276], 0.18674930930137634, 'upperarm_twist_02_r'), 'upperarm_correctiveRoot_r': ([-19.009883880615234, 2.557269811630249, 143.58299255371094], [-19.262758255004883, -5.187027931213379, 143.98318481445312], [0.5760454535484314, 0.023413674905896187, 0.817083477973938], [-0.03259214386343956, -0.9981369376182556, 0.05157945305109024], [0.8167682886123657, -0.05634258687496185, -0.5742124915122986], -0.1696859747171402, 'upperarm_r'), 'upperarm_bck_r': ([-20.153121948242188, 10.175762176513672, 142.21299743652344], [-26.238027572631836, 8.111940383911133, 146.5618896484375], [0.576034665107727, 0.023413002490997314, 0.8170897960662842], [-0.7842629551887512, -0.2659989595413208, 0.560513973236084], [0.23046837747097015, -0.963688850402832, -0.13486260175704956], -0.9259811043739319, 'upperarm_correctiveRoot_r'), 'upperarm_in_r': ([-19.079315185546875, 1.6020208597183228, 135.6416778564453], [-18.826440811157227, 9.346319198608398, 135.24148559570312], [0.9928202033042908, -0.026389600709080696, 0.11666855961084366], [0.03259213641285896, 0.9981368780136108, -0.05157945305109024], [-0.1150900349020958, 0.055011603981256485, 0.991830587387085], -0.11624898761510849, 'upperarm_correctiveRoot_r'), 'upperarm_fwd_r': ([-20.52543067932129, -6.072142124176025, 140.7523193359375], [-14.232881546020508, -5.271876811981201, 136.2844696044922], [0.5844912528991699, -0.18433761596679688, 0.7901833057403564], [0.8110256195068359, 0.10314351320266724, -0.5758461952209473], [0.024647846817970276, 0.9774359464645386, 0.20978906750679016], -0.7669719457626343, 'upperarm_correctiveRoot_r'), 'upperarm_out_r': ([-24.107315063476562, 2.930767297744751, 146.73350524902344], [-24.36018943786621, -4.813530921936035, 147.13369750976562], [0.5760454535484314, 0.023413674905896187, 0.817083477973938], [-0.03259214386343956, -0.9981369376182556, 0.05157945305109024], [0.8167682886123657, -0.05634258687496185, -0.5742124915122986], -0.1696859747171402, 'upperarm_correctiveRoot_r'), 'clavicle_out_r': ([-14.299903869628906, 1.8724745512008667, 150.70455932617188], [-13.889496803283691, -12.879898071289062, 150.86634826660156], [0.9947144985198975, 0.02875376120209694, 0.09857386350631714], [0.027807312086224556, -0.9995535612106323, 0.010962080210447311], [0.0988452136516571, -0.008163057267665863, -0.9950695037841797], 2.4894652366638184, 'clavicle_r'), 'clavicle_scap_r': ([-10.865946769714355, 9.761637687683105, 142.8354949951172], [-11.27635383605957, 24.514009475708008, 142.6737060546875], [-0.9947132468223572, -0.02875383198261261, -0.09858453273773193], [-0.02780730463564396, 0.9995533227920532, -0.010962081141769886], [0.09885568171739578, -0.008162754587829113, -0.995068371295929], 3.042689323425293, 'clavicle_r'), 'clavicle_pec_r': ([-9.635502815246582, -9.942605018615723, 133.40135192871094], [-11.168313026428223, -23.02654457092285, 133.2438507080078], [0.9902886152267456, -0.11692190170288086, 0.07531909644603729], [-0.11634799838066101, -0.9931367039680481, -0.011955132707953453], [0.0762009397149086, 0.0030755065381526947, -0.997087836265564], 3.0128989219665527, 'spine_05'), 'spine_04_latissimus_l': ([15.44347858428955, 2.2761011123657227, 130.48175048828125], [12.279584884643555, 15.040692329406738, 131.2683563232422], [0.9489972591400146, 0.24727660417556763, -0.19559749960899353], [-0.2401554137468338, 0.9688964486122131, 0.05970732867717743], [0.20427794754505157, -0.009688295423984528, 0.9788649678230286], 0.20451901853084564, 'spine_05'), 'spine_04_latissimus_r': ([-15.454672813415527, 2.27587890625, 130.4764404296875], [-14.77698040008545, -10.881035804748535, 130.48764038085938], [0.9957670569419861, 0.05135543644428253, 0.07625008374452591], [0.05144026130437851, -0.9986759424209595, 0.0008501328411512077], [0.0761926919221878, 0.003075788728892803, -0.9970884323120117], -3.098296880722046, 'spine_05'), 'clavicle_pec_l': ([9.310201644897461, -9.716368675231934, 133.35574340820312], [7.777227878570557, 3.367551803588867, 133.51316833496094], [0.990283727645874, 0.11693192273378372, -0.07526613771915436], [-0.11636040359735489, 0.9931352734565735, 0.011949341744184494], [0.07614671438932419, -0.0030752411112189293, 0.9970918297767639], 0.07604043185710907, 'spine_05'), 'thigh_r': ([-9.969091415405273, -2.5500237941741943, 93.54337310791016], [-13.133283615112305, -23.987470626831055, 93.74345397949219], [-0.0545215830206871, -0.001271165907382965, -0.9985072016716003], [-0.14601290225982666, -0.9892395734786987, 0.009232791140675545], [-0.9877790808677673, 0.14629895985126495, 0.0537489652633667], 1.6426386833190918, 'pelvis'), 'calf_r': ([-12.332228660583496, -2.6051414012908936, 50.2666130065918], [-15.503074645996094, -23.40973472595215, 48.6219596862793], [-0.041574329137802124, 0.08503449708223343, -0.9955047965049744], [-0.15021288394927979, -0.9855788350105286, -0.07791239023208618], [-0.9877773523330688, 0.14629916846752167, 0.05375397205352783], 0.5658085346221924, 'thigh_r'), 'foot_r': ([-14.087557792663574, 0.9848349690437317, 8.23822021484375], [-15.395964622497559, -7.751678943634033, 7.959315776824951], [0.00465160608291626, 0.031212609261274338, -0.9994927048683167], [-0.14803744852542877, -0.9884780645370483, -0.03155616670846939], [-0.9889703989028931, 0.14811065793037415, 2.6553869247436523e-05], 1.1530821323394775, 'calf_r'), 'ball_r': ([-15.777702331542969, -13.938241004943848, 0.7514234781265259], [-15.736608505249023, -13.66238021850586, -8.082521438598633], [0.14803741872310638, 0.9884781837463379, 0.03155620023608208], [0.004649490118026733, 0.03121180273592472, -0.9995018839836121], [-0.9889708161354065, 0.14811038970947266, 2.47173011302948e-05], -1.426717758178711, 'foot_r'), 'ankle_fwd_r': ([-14.461820602416992, -4.179941654205322, 10.244355201721191], [-23.175588607788086, -2.812370538711548, 10.806830406188965], [-0.04547552019357681, 0.11822539567947388, -0.9919449687004089], [-0.9859046339988708, 0.15473154187202454, 0.0636403039097786], [0.1610090732574463, 0.980857253074646, 0.10952247679233551], 1.5153030157089233, 'foot_r'), 'ankle_bck_r': ([-14.410563468933105, 5.101284027099609, 8.892108917236328], [-5.696795463562012, 3.733712911605835, 8.329633712768555], [-0.04547540843486786, 0.11822549253702164, -0.9919449090957642], [0.9859045743942261, -0.15473154187202454, -0.0636403039097786], [-0.16100910305976868, -0.980857253074646, -0.10952259600162506], 1.7552112340927124, 'foot_r'), 'calf_twist_02_r': ([-12.954605102539062, -1.402963399887085, 36.259185791015625], [-12.956085205078125, -1.4128481149673462, 36.25886917114258], [-0.042632728815078735, 0.038361866027116776, -0.998356819152832], [-0.14801135659217834, -0.9884787797927856, -0.0316622257232666], [-0.9880664348602295, 0.14641784131526947, 0.04781821370124817], 1.1038106679916382, 'calf_r'), 'calf_twistCor_02_r': ([-12.954606056213379, -1.4029630422592163, 36.259185791015625], [-12.956086158752441, -1.4128477573394775, 36.25886917114258], [-0.042632877826690674, 0.03836188465356827, -0.998356819152832], [-0.14801135659217834, -0.9884787797927856, -0.0316622331738472], [-0.9880664944648743, 0.14641784131526947, 0.04781833291053772], 1.1038105487823486, 'calf_twist_02_r'), 'calf_twist_01_r': ([-13.576982498168945, -0.2007846087217331, 22.251741409301758], [-16.701906204223633, -21.066579818725586, 21.585620880126953], [-0.04406380653381348, 0.03846851363778114, -0.9982879161834717], [-0.14803744852542877, -0.9884781837463379, -0.03155623376369476], [-0.9879996180534363, 0.14639347791671753, 0.04925084114074707], 1.1038106679916382, 'calf_r'), 'calf_correctiveRoot_r': ([-12.332226753234863, -2.6051418781280518, 50.26662063598633], [-13.160930633544922, -8.040858268737793, 49.8369255065918], [-0.04422447085380554, 0.0854264423251152, -0.9953582286834717], [-0.15025575459003448, -0.9855725169181824, -0.07790981233119965], [-0.9876561164855957, 0.1461133062839508, 0.056426793336868286], 0.5633844137191772, 'calf_r'), 'calf_kneeBack_r': ([-11.050769805908203, 3.025921106338501, 50.53688430786133], [-5.642226696014404, 2.0222580432891846, 50.13848114013672], [-0.047797705978155136, 0.135270357131958, -0.9896551370620728], [0.9806456565856934, -0.18197833001613617, -0.07223614305257797], [-0.18986719846725464, -0.9739536643028259, -0.1239541620016098], 1.7823303937911987, 'calf_correctiveRoot_r'), 'calf_knee_r': ([-13.127237319946289, -7.759250640869141, 49.66874694824219], [-18.53577995300293, -6.755587577819824, 50.0671501159668], [-0.047797780483961105, 0.1352701038122177, -0.9896551966667175], [-0.9806456565856934, 0.18197835981845856, 0.07223615795373917], [0.18986722826957703, 0.9739538431167603, 0.12395387887954712], 1.506320595741272, 'calf_correctiveRoot_r'), 'thigh_twist_01_r': ([-10.756805419921875, -2.5683956146240234, 79.11779022216797], [-10.758277893066406, -2.5782856941223145, 79.1176528930664], [-0.044317469000816345, 0.020465949550271034, -0.9988165497779846], [-0.1472468078136444, -0.9890044927597046, -0.013732880353927612], [-0.9881065487861633, 0.1464625895023346, 0.04684168100357056], 1.338958501815796, 'thigh_r'), 'thigh_twistCor_01_r': ([-10.756804466247559, -2.5683956146240234, 79.11778259277344], [-10.75827693939209, -2.5782856941223145, 79.11764526367188], [-0.044317469000816345, 0.020465949550271034, -0.9988165497779846], [-0.1472468078136444, -0.9890044927597046, -0.013732880353927612], [-0.9881065487861633, 0.1464625895023346, 0.04684168100357056], 1.338958501815796, 'thigh_twist_01_r'), 'thigh_twist_02_r': ([-11.544516563415527, -2.586768388748169, 64.69220733642578], [-11.545989036560059, -2.59665846824646, 64.69207000732422], [-0.044317469000816345, 0.020465949550271034, -0.9988165497779846], [-0.1472468078136444, -0.9890044927597046, -0.013732880353927612], [-0.9881065487861633, 0.1464625895023346, 0.04684168100357056], 1.338958501815796, 'thigh_r'), 'thigh_twistCor_02_r': ([-11.544514656066895, -2.5867691040039062, 64.69222259521484], [-11.545987129211426, -2.5966591835021973, 64.69208526611328], [-0.044317469000816345, 0.020465949550271034, -0.9988165497779846], [-0.1472468078136444, -0.9890044927597046, -0.013732880353927612], [-0.9881065487861633, 0.1464625895023346, 0.04684168100357056], 1.338958501815796, 'thigh_twist_02_r'), 'thigh_correctiveRoot_r': ([-9.96909236907959, -2.550022840499878, 93.54337310791016], [-11.4305419921875, -12.451364517211914, 93.63578796386719], [-0.05452612787485123, -0.0012727025896310806, -0.9985188841819763], [-0.14601293206214905, -0.9892397522926331, 0.009233125485479832], [-0.987778902053833, 0.14629912376403809, 0.05375409126281738], 1.6426386833190918, 'thigh_r'), 'thigh_fwd_r': ([-9.98119068145752, -9.356941223144531, 97.79995727539062], [-19.86791229248047, -7.892627716064453, 98.33795928955078], [-0.054523952305316925, -0.0012715347111225128, -0.9985116720199585], [-0.9877790212631226, 0.1462990641593933, 0.053751617670059204], [0.14601296186447144, 0.989239513874054, -0.009232800453901291], 1.626444697380066, 'thigh_correctiveRoot_r'), 'thigh_bck_r': ([-6.724599361419678, 7.188129425048828, 98.39881896972656], [3.162121295928955, 5.723816394805908, 97.8608169555664], [-0.05452399328351021, -0.001271802932024002, -0.9985116720199585], [0.9877790212631226, -0.14629903435707092, -0.05375159904360771], [-0.14601293206214905, -0.9892396330833435, 0.009233057498931885], 1.6238737106323242, 'thigh_correctiveRoot_r'), 'thigh_out_r': ([-14.611981391906738, -1.2008516788482666, 99.62046813964844], [-13.150532722473145, 8.70048999786377, 99.5280532836914], [-0.05452391505241394, -0.0012719040969386697, -0.9985117316246033], [0.14601285755634308, 0.9892398118972778, -0.009233133867383003], [0.9877791404724121, -0.14629894495010376, -0.05375150218605995], 1.6252537965774536, 'thigh_correctiveRoot_r'), 'thigh_in_r': ([-2.203738212585449, -4.576478958129883, 82.17730712890625], [-3.6651880741119385, -14.47782039642334, 82.26972198486328], [-0.05452600121498108, -0.0012727752327919006, -0.998519241809845], [-0.14601294696331024, -0.9892397522926331, 0.009233124554157257], [-0.9877789616584778, 0.14629913866519928, 0.05375397950410843], 1.6426388025283813, 'thigh_correctiveRoot_r'), 'thigh_bck_lwr_r': ([-7.221047878265381, 7.757846832275391, 88.0961685180664], [2.665672779083252, 6.293533802032471, 87.55816650390625], [-0.05452399328351021, -0.001271802932024002, -0.9985116720199585], [0.9877790212631226, -0.14629903435707092, -0.05375159904360771], [-0.14601293206214905, -0.9892396330833435, 0.009233057498931885], 1.6238737106323242, 'thigh_correctiveRoot_r'), 'thigh_fwd_lwr_r': ([-10.317131996154785, -9.482711791992188, 92.5416488647461], [-20.203853607177734, -8.01839828491211, 93.07965087890625], [-0.054523952305316925, -0.0012715347111225128, -0.9985116720199585], [-0.9877790212631226, 0.1462990641593933, 0.053751617670059204], [0.14601296186447144, 0.989239513874054, -0.009232800453901291], 1.626444697380066, 'thigh_correctiveRoot_r'), 'thigh_l': ([9.969202995300293, -2.540597438812256, 93.54280090332031], [6.8050055503845215, 18.89688491821289, 93.34271240234375], [-0.05452391132712364, 0.0012719192309305072, 0.9985116124153137], [-0.14601291716098785, 0.9892396330833435, -0.009233148768544197], [-0.9877790808677673, -0.14629901945590973, -0.05375150218605995], -1.6252537965774536, 'pelvis'), 'calf_l': ([12.332341194152832, -2.5957157611846924, 50.2659797668457], [9.161500930786133, 18.208845138549805, 51.91062927246094], [-0.04157775267958641, -0.0850343182682991, 0.9955102205276489], [-0.15021288394927979, 0.9855790138244629, 0.0779123306274414], [-0.9877791404724121, -0.14629903435707092, -0.05375140905380249], -1.6188255548477173, 'thigh_l'), 'foot_l': ([14.08767032623291, 0.9942569136619568, 8.237628936767578], [12.841788291931152, 9.313282012939453, 8.503205299377441], [0.004649605602025986, -0.031211746856570244, 0.9995021224021912], [-0.14803744852542877, 0.9884781241416931, 0.031556136906147], [-0.9889708757400513, -0.14811044931411743, -2.4472828954458237e-05], -1.5684703588485718, 'calf_l'), 'ball_l': ([15.777871131896973, -13.928804397583008, 0.7508242726325989], [15.817001342773438, -14.191482543945312, 9.162625312805176], [0.14803741872310638, -0.9884783029556274, -0.03155618533492088], [0.004649505019187927, -0.031211785972118378, 0.9995020031929016], [-0.9889708757400513, -0.14811041951179504, -2.4667009711265564e-05], -1.4174141883850098, 'foot_l'), 'ankle_bck_l': ([14.067781448364258, 4.452132701873779, 8.958708763122559], [22.46796989440918, 4.582714080810547, 9.457383155822754], [-0.057188861072063446, -0.11035731434822083, 0.9922454357147217], [0.9981225728988647, 0.01551586389541626, 0.05925321206450462], [-0.021934576332569122, 0.9937708377838135, 0.10926277935504913], -1.5194954872131348, 'foot_l'), 'ankle_fwd_l': ([15.037823677062988, -3.563922166824341, 9.87154769897461], [6.63763427734375, -3.6945035457611084, 9.372873306274414], [-0.05718887597322464, -0.11035715043544769, 0.9922453165054321], [-0.9981226921081543, -0.015515871345996857, -0.05925320088863373], [0.021934546530246735, -0.9937708973884583, -0.10926258563995361], -1.7406871318817139, 'foot_l'), 'calf_twist_02_l': ([12.917449951171875, -1.3990588188171387, 36.25654220581055], [12.915969848632812, -1.3891741037368774, 36.256858825683594], [-0.044056955724954605, -0.03857322409749031, 0.998284101486206], [-0.14801135659217834, 0.9884788393974304, 0.0316622331738472], [-0.9880039095878601, -0.14636242389678955, -0.04925863817334175], -1.6177417039871216, 'calf_l'), 'calf_twistCor_02_l': ([12.917451858520508, -1.3990578651428223, 36.256534576416016], [12.915971755981445, -1.389173150062561, 36.25685119628906], [-0.044056955724954605, -0.03857322409749031, 0.998284101486206], [-0.14801135659217834, 0.9884788393974304, 0.0316622331738472], [-0.9880039095878601, -0.14636242389678955, -0.04925863817334175], -1.6177417039871216, 'calf_twist_02_l'), 'calf_twist_01_l': ([13.577085494995117, -0.19135770201683044, 22.251140594482422], [10.452165603637695, 20.67440414428711, 22.917259216308594], [-0.0440642386674881, -0.038468554615974426, 0.9982876181602478], [-0.14803744852542877, 0.988477885723114, 0.03155617415904999], [-0.9879994988441467, -0.14639350771903992, -0.04925132915377617], -1.6177417039871216, 'calf_l'), 'calf_correctiveRoot_l': ([12.332351684570312, -2.5957181453704834, 50.2660026550293], [11.503868103027344, 2.8385555744171143, 50.69558334350586], [-0.0442260205745697, -0.08542614430189133, 0.9953624606132507], [-0.1502557098865509, 0.9855725765228271, 0.07790976762771606], [-0.9876574277877808, -0.14611326158046722, -0.056423719972372055], -1.6215085983276367, 'calf_l'), 'calf_kneeBack_l': ([11.075106620788574, 3.034249782562256, 50.53401565551758], [16.482208251953125, 4.0376386642456055, 50.93240737915039], [-0.04781755059957504, -0.13525760173797607, 0.9896558523178101], [0.9806444644927979, 0.18197692930698395, 0.0722532644867897], [-0.18986734747886658, 0.9739557504653931, 0.1239379346370697], -1.5063508749008179, 'calf_correctiveRoot_l'), 'calf_knee_l': ([13.1307373046875, -7.752592086791992, 49.66517639160156], [7.723635673522949, -8.7559814453125, 49.26678466796875], [-0.04781753569841385, -0.13525745272636414, 0.9896559715270996], [-0.9806445240974426, -0.1819770187139511, -0.07225324958562851], [0.18986743688583374, -0.9739557504653931, -0.12393782287836075], -1.782334566116333, 'calf_correctiveRoot_l'), 'thigh_twist_01_l': ([10.756916999816895, -2.5589699745178223, 79.1171875], [10.755444526672363, -2.5490798950195312, 79.11732482910156], [-0.04671255871653557, -0.020822107791900635, 0.9986913204193115], [-0.14724679291248322, 0.9890044927597046, 0.01373287197202444], [-0.9879961609840393, -0.14641261100769043, -0.04926491901278496], -1.6190690994262695, 'thigh_l'), 'thigh_twistCor_01_l': ([10.756916046142578, -2.5589699745178223, 79.11719512939453], [10.755443572998047, -2.5490798950195312, 79.1173324584961], [-0.04671255871653557, -0.020822107791900635, 0.9986913204193115], [-0.14724679291248322, 0.9890044927597046, 0.01373287197202444], [-0.9879961609840393, -0.14641261100769043, -0.04926491901278496], -1.6190690994262695, 'thigh_twist_01_l'), 'thigh_twist_02_l': ([11.544638633728027, -2.577343702316284, 64.69158172607422], [11.543166160583496, -2.567453622817993, 64.69171905517578], [-0.04671255871653557, -0.020822107791900635, 0.9986913204193115], [-0.14724679291248322, 0.9890044927597046, 0.01373287197202444], [-0.9879961609840393, -0.14641261100769043, -0.04926491901278496], -1.6190690994262695, 'thigh_l'), 'thigh_twistCor_02_l': ([11.544637680053711, -2.577343702316284, 64.69158935546875], [11.54316520690918, -2.567453622817993, 64.69172668457031], [-0.04671255871653557, -0.020822107791900635, 0.9986913204193115], [-0.14724679291248322, 0.9890044927597046, 0.01373287197202444], [-0.9879961609840393, -0.14641261100769043, -0.04926491901278496], -1.6190690994262695, 'thigh_twist_02_l'), 'thigh_correctiveRoot_l': ([9.96920394897461, -2.5405969619750977, 93.54279327392578], [8.501510620117188, 7.403048515319824, 93.44998168945312], [-0.054523929953575134, 0.0012721001403406262, 0.998511791229248], [-0.14601291716098785, 0.9892399311065674, -0.009233348071575165], [-0.9877791404724121, -0.14629901945590973, -0.053751491010189056], -1.6252537965774536, 'thigh_l'), 'thigh_bck_l': ([6.751660346984863, 7.194935321807861, 98.39765930175781], [16.680625915527344, 8.665505409240723, 98.93795776367188], [-0.05452382564544678, 0.001271761953830719, 0.9985116124153137], [0.9877790808677673, 0.1462990641593933, 0.053751375526189804], [-0.14601296186447144, 0.989239513874054, -0.009233001619577408], -1.626444697380066, 'thigh_correctiveRoot_l'), 'thigh_fwd_l': ([9.989326477050781, -9.344791412353516, 97.78841400146484], [0.060362815856933594, -10.815361022949219, 97.24811553955078], [-0.05452393367886543, 0.0012715309858322144, 0.9985118508338928], [-0.9877791404724121, -0.14629903435707092, -0.053751390427351], [0.14601293206214905, -0.9892396330833435, 0.00923280417919159], -1.6238737106323242, 'thigh_correctiveRoot_l'), 'thigh_out_l': ([14.626700401306152, -1.2036268711090088, 99.61817932128906], [16.09439468383789, -11.147272109985352, 99.71099090576172], [-0.05452831834554672, 0.0012692753225564957, 0.9984979033470154], [0.14601293206214905, -0.9892394542694092, 0.009233306162059307], [0.9877786636352539, 0.14629894495010376, 0.053755275905132294], -1.6426386833190918, 'thigh_correctiveRoot_l'), 'thigh_bck_lwr_l': ([7.511082649230957, 7.651322364807129, 88.0892105102539], [17.440046310424805, 9.121891975402832, 88.62950897216797], [-0.05452394485473633, 0.0012717321515083313, 0.9985116720199585], [0.9877792596817017, 0.14629904925823212, 0.053751390427351], [-0.14601294696331024, 0.9892395734786987, -0.009232986718416214], -1.626444697380066, 'thigh_correctiveRoot_l'), 'thigh_in_l': ([2.231889247894287, -4.623826026916504, 82.2379150390625], [0.7641956806182861, 5.319819450378418, 82.14510345458984], [-0.054523929953575134, 0.001272100256755948, 0.998511791229248], [-0.14601293206214905, 0.9892399311065674, -0.009233348071575165], [-0.9877791404724121, -0.14629903435707092, -0.053751491010189056], -1.6252537965774536, 'thigh_correctiveRoot_l'), 'thigh_fwd_lwr_l': ([10.468570709228516, -9.945235252380371, 92.4569091796875], [0.539607048034668, -11.415804862976074, 91.91661071777344], [-0.05452393367886543, 0.0012715309858322144, 0.9985118508338928], [-0.9877791404724121, -0.14629903435707092, -0.053751390427351], [0.14601293206214905, -0.9892396330833435, 0.00923280417919159], -1.6238737106323242, 'thigh_correctiveRoot_l'), 'ik_foot_root': ([0.0, 0.0, 0.0], [0.0, 16.349424362182617, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], 0.0, None), 'ik_foot_l': ([14.087671279907227, 0.9942575097084045, 8.237637519836426], [11.667343139648438, 17.155305862426758, 8.753562927246094], [0.004649607464671135, -0.03121177665889263, 0.9995021224021912], [-0.14803752303123474, 0.9884781241416931, 0.03155616670846939], [-0.9889708757400513, -0.1481105089187622, -2.4475855752825737e-05], -1.5684703588485718, 'ik_foot_root'), 'ik_foot_r': ([-14.08755874633789, 0.9848345518112183, 8.238231658935547], [-16.50788688659668, -15.176214218139648, 7.722306251525879], [0.004649102687835693, 0.031211618334054947, -0.9995036721229553], [-0.14803753793239594, -0.9884781837463379, -0.03155619651079178], [-0.9889709949493408, 0.14811047911643982, 2.428889274597168e-05], 1.153082013130188, 'ik_foot_root'), 'ik_hand_root': ([0.0, 0.0, 0.0], [0.0, 115.93748474121094, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], 0.0, None), 'ik_hand_gun': ([-47.76831817626953, -15.69880485534668, 104.4664306640625], [-78.60330200195312, -28.040145874023438, 138.7989501953125], [0.45776593685150146, 0.6224100589752197, 0.6348668932914734], [-0.6455091238021851, -0.2583574652671814, 0.7187275886535645], [0.6113659143447876, -0.7388213872909546, 0.2835041880607605], -0.03169303387403488, 'ik_hand_root'), 'ik_hand_l': ([47.76860809326172, -15.698881149291992, 104.4665298461914], [16.933624267578125, -3.3575439453125, 70.13400268554688], [0.4577661454677582, -0.622409999370575, -0.6348668932914734], [-0.6455090045928955, 0.25835731625556946, -0.7187276482582092], [0.6113659143447876, 0.7388215065002441, -0.28350409865379333], 1.4318660497665405, 'ik_hand_gun'), 'ik_hand_r': ([-47.768310546875, -15.698801040649414, 104.4664306640625], [-78.60328674316406, -28.040142059326172, 138.7989501953125], [0.45776617527008057, 0.6224099397659302, 0.6348668336868286], [-0.6455090045928955, -0.2583574652671814, 0.7187276482582092], [0.6113658547401428, -0.7388214468955994, 0.28350409865379333], -0.03169303387403488, 'ik_hand_gun'), 'interaction': ([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], 0.0, None), 'center_of_mass': ([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], 0.0, None)
}
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

def is_clothes(obj)->ClothesMeta:
    name = obj.name
    if name.endswith(" Mesh"):
        name = name[:-len(" Mesh")]
    return CLOTHES.get(name)


def find_all_clothes(predicate=None):
    clothes = []
    for obj in bpy.data.objects:
        if isinstance(obj.data, bpy.types.Mesh):
            meta = is_clothes(obj)
            if meta is not None and (predicate is None or predicate(meta)):
                meta.obj = obj
                clothes.append(meta)
    return clothes

def is_skin_tight(x):
    return x.skin_tight>=0 and not x.panties

def is_not_skin_tight(x):
    return x.skin_tight < 0

def is_panties(x):
    return x.panties

def find_all_panties():
    return find_all_clothes(is_panties)

def find_all_skin_tight_clothes():
    return find_all_clothes(is_skin_tight)

def find_all_non_skin_tight_clothes():
    return find_all_clothes(is_not_skin_tight)

def is_hair(obj):
    return 'hair' in obj.name.lower()

def find_all_hair():
    return [obj for obj in bpy.data.objects if isinstance(obj.data, bpy.types.Mesh) and is_hair(obj)]

def is_cum(o):
    return o.name.startswith("Love Loads")

def find_cum():
    return [o for o in bpy.data.objects if isinstance(o.data, bpy.types.Mesh) and is_cum(o)]

def is_sub_rig(sub_rig, super_rig):
    for bone in sub_rig.bones:
        if bone.name not in super_rig.bones:
            return False
    return True

def find_object_containing(infix):
    for o in bpy.data.objects:
        if isinstance(o.data, bpy.types.Mesh) and infix in o.name:
            return o

def contains_group(vertex, group_index):
    for g in vertex.groups:
        if g.group == group_index:
            return g
    return None

def get_group_weight(vertex, group_index):
    for g in vertex.groups:
        if g.group == group_index:
            return g.weight
    return 0

def get_weights_as_array(mesh, vertex_group):
    if isinstance(vertex_group, str):
        vertex_group = mesh.vertex_groups[vertex_group]
    i1 = vertex_group.index
    return np.array([get_group_weight(v, i1) for v in mesh.data.vertices])

def get_weights_as_sparse(mesh, vertex_group):
    arr = get_weights_as_array(mesh, vertex_group)
    is_non_zero = arr>0
    weights = arr[is_non_zero]
    indices = rle_encode(is_non_zero)
    #indices, = np.where(is_non_zero)
    return weights, indices, is_non_zero

NUM_OF_VERTICES_IN_DAZ_BASE_MESH = 25182
ADDITIONAL_BONES = {
    'l_thigh_jiggle':({'head': [0.10246124863624573, 0.04263287037611008, 0.8381637334823608], 'tail': [0.10360853374004364, 0.07650366425514221, 0.7981241941452026], 'parent': 'l_thigh', 'connect': False, 'local': True},b'AAAA4CMHfz8AAADA9GTGPwAAAADUwsU/AAAAoPNWuD8AAADAigu+PwAAAMCRR7I/AAAAgHj0yj8AAACg7sLaPwAAAMAEE9k/AAAAAAGu0T8AAABgxu7BPwAAAMD5PoI/AAAAIN5o1T8AAACgBJ3WPwAAAEAUhOA/AAAAQKoN6D8AAACA5KfoPwAAAGBHrsY/AAAAoJ0RlD8AAACgfMPkPwAAAICc8tg/AAAAgF00xj8AAAAg+NrlPwAAAABHcZo/AAAAQPolaD8AAACA0QaYPwAAAKAuDdQ/AAAA4N9S2z8AAABAIxrXPwAAAEC6psw/AAAA4HCj3z8AAAAAb7XdPwAAAMA8D8c/AAAAYPzyuT8AAACgQGdSPwAAAMCTW7g/AAAAQNDqyT8AAABA0rnJPwAAAGDXZc0/AAAAgDvlwT8AAADg8beaPwAAAICQbsI/AAAAAKrs4z8AAABg6aPdPwAAAKBQ1eY/AAAAwNtv6D8AAABg463iPwAAAEDoq9U/AAAAYNV/wz8AAADAW1vTPwAAAEBLrL4/AAAAYEmAlz8AAADgIcacPwAAAKAZSOM+AAAAQG4xTT8AAAAAEBSsPwAAAEA1kbM/AAAAwORz1T8AAAAAVWbfPwAAAGBRIeg/AAAAoNce6D8AAAAgw+rDPwAAAEDg5kA/AAAA4Ll95D8AAACg5pHVPwAAAECsJMM/AAAAQFp/5D8AAAAAhY9EPwAAAIBEq9g/AAAAwHCZzT8AAABgy6riPwAAAGBXR+M/AAAA4ODZqT8AAABg86rPPwAAAKBlWds/AAAAILcWuj8AAABAiw3hPwAAAODViYQ/AAAAgHTOuj4AAACgNIvqPgAAAACS+5k+',np.array([25, 1, 1, 1, 1, 1, 426, 1, 839, 1, 53, 4, 2, 3, 9, 1, 4, 9, 9, 2, 7083, 1, 3, 1, 47, 16, 226, 9, 1, 1, 39, 1, 441, 1, 2, 1, 6, 1, 5, 1, 283, 9, 5, 2, 1571, 5, 1, 3, 5, 1, 50, 1, 3, 1, 1, 1, 2, 1, 13958],dtype=np.int64)),
    'r_thigh_jiggle':({'head': [-0.10246124863624573, 0.04263287037611008, 0.8381637334823608], 'tail': [-0.10360853374004364, 0.07650366425514221, 0.7981241941452026], 'parent': 'r_thigh', 'connect': False, 'local': True},b'AAAA4CMHfz8AAADA9GTGPwAAAADUwsU/AAAAoPNWuD8AAADAigu+PwAAAMCRR7I/AAAAgHj0yj8AAACg7sLaPwAAAMAEE9k/AAAAAAGu0T8AAABgxu7BPwAAAMD5PoI/AAAAIN5o1T8AAACgBJ3WPwAAAEAUhOA/AAAAQKoN6D8AAACA5KfoPwAAAGBHrsY/AAAAoJ0RlD8AAACgfMPkPwAAAICc8tg/AAAAgF00xj8AAAAg+NrlPwAAAABHcZo/AAAAQPolaD8AAACA0QaYPwAAAKAuDdQ/AAAA4N9S2z8AAABAIxrXPwAAAEC6psw/AAAA4HCj3z8AAAAAb7XdPwAAAMA8D8c/AAAAYPzyuT8AAACgQGdSPwAAAMCTW7g/AAAAQNDqyT8AAABA0rnJPwAAAGDXZc0/AAAAgDvlwT8AAADg8beaPwAAAICQbsI/AAAAAKrs4z8AAABg6aPdPwAAAKBQ1eY/AAAAwNtv6D8AAABg463iPwAAAEDoq9U/AAAAYNV/wz8AAADAW1vTPwAAAEBLrL4/AAAAYEmAlz8AAADgIcacPwAAAKAZSOM+AAAAQG4xTT8AAAAAEBSsPwAAAEA1kbM/AAAAwORz1T8AAAAAVWbfPwAAAGBRIeg/AAAAoNce6D8AAAAgw+rDPwAAAEDg5kA/AAAA4Ll95D8AAACg5pHVPwAAAECsJMM/AAAAQFp/5D8AAAAAhY9EPwAAAIBEq9g/AAAAwHCZzT8AAABgy6riPwAAAGBXR+M/AAAA4ODZqT8AAABg86rPPwAAAKBlWds/AAAAILcWuj8AAABAiw3hPwAAAODViYQ/AAAAgHTOuj4AAACgNIvqPgAAAACS+5k+',np.array([12029, 1, 1, 1, 1, 1, 426, 1, 838, 1, 53, 4, 2, 3, 9, 1, 4, 9, 9, 2, 6902, 1, 3, 1, 47, 16, 215, 9, 1, 1, 38, 1, 438, 1, 2, 1, 5, 1, 4, 1, 274, 9, 5, 2, 1568, 5, 1, 3, 5, 1, 50, 1, 2, 1, 1, 1, 2, 1, 2166],dtype=np.int64)),
    'l_thigh_jiggle2':({'head': [0.10460905730724335, 0.03490614518523216, 0.7691643238067627], 'tail': [0.10575634241104126, 0.068776935338974, 0.7291247844696045], 'parent': 'l_thigh', 'connect': False, 'local': True},b'AAAAYENsxD8AAABgjB3IPwAAAMDgBtg/AAAAICW14T8AAAAA8WPjPwAAAGB+QeE/AAAAQNBd3j8AAAAAx2+7PwAAAMCc9L0/AAAAIKs7sz8AAADgS57TPwAAAECcjAY/AAAA4ASUnD8AAABA5gGnPwAAAKBFj4A/AAAAAIyBcz8AAABgiU6xPwAAAKAeZ7M/AAAA4AGo2j8AAADgaXXNPwAAAICqCdU/AAAAgLWGtD8AAABAOcjHPwAAAACZXsc/AAAAIKPRtz8AAAAgdeBBPwAAAEAeltI/AAAAgFhB0T8AAADA+06ePwAAAADDAr4/AAAAoMu0yD8AAACgBV6zPwAAAKCUt0U/AAAAYNQppD8AAADA6KK8PwAAAEAqGOI/AAAAAKFY0D8AAAAAJbHdPwAAAKC4vSs/AAAAAAM+wT8AAADgz5zOPwAAAGA1xdw/AAAAIDSc2D8AAADgheamPwAAAOBLCNE/AAAAgG9dxT8AAADgb4m1PwAAAEAkadk/AAAAYOndnz8AAACgcxVhPwAAAMBtDVU/AAAA4EJcuz8AAAAA2XPIPwAAAGA6zto/AAAAQPxK4j8AAAAAiZvgPwAAAEAziLU/AAAA4GrGkj8AAAAgDz/ePwAAACAv2M4/AAAAwKkh4T8AAABg8wq7PwAAAKDkB0A/AAAAQLjX4T4AAADAkSCGPgAAAGCycUw+AAAAgM7eOz8AAADgP4GfPwAAAMBEXrc/AAAAIFTwtD8AAADgNJ+wPwAAAKABgKM/AAAAYHO5sz8AAABg/dfAPwAAACCALbE/AAAAIDMRyT8AAACAOtfKPwAAAIAFSVY/AAAAAJ7UuD8AAAAgAovDPwAAAEDz5KY/AAAAoDT9yD8AAAAAd9g8Pw==',np.array([23, 12, 1, 1, 1, 1, 21, 2, 361, 2, 31, 3, 4, 2, 68, 5, 5, 1, 2, 2, 139, 1, 1, 1, 3, 1, 600, 1, 1, 1, 2, 2, 13, 1, 35, 5, 1, 3, 9, 3, 8, 1, 636, 1, 335, 8, 10, 1, 1, 2, 6650, 1, 20, 1, 28, 1, 487, 4, 2, 2, 6, 1, 1572, 5, 1, 3, 5, 1, 1, 1, 14016],dtype=np.int64)),
    'l_thigh_jiggle_side':({'head': [0.14600594341754913, -0.007730054669082165, 0.8691152930259705], 'tail': [0.18229305744171143, -0.013294694945216179, 0.8209406733512878], 'parent': 'l_thigh', 'connect': False, 'local': True},b'AAAAwCZxRD8AAACAgPCSPwAAAAAo3Y4/AAAAwPQwpD8AAAAgqjJUPwAAAABZBp8/AAAAIBBurj8AAACAV+OxPwAAAMCnsaA/AAAAoPwCrT8AAADgJNN6PwAAAACPqss/AAAAICXetD8AAADgpJHEPwAAAADhctU/AAAAIHzi2T8AAABAlXU8PwAAAICXYsY+AAAAoH+YgT8AAABAita+PwAAAEDb1Lw/AAAA4LoNpD4AAABA2ZukPwAAAIDUE7Q/AAAAQEW0RT4AAAAAZRQ6PwAAACAxfIk/AAAAYEpMeD8AAABgPwJ7PwAAAGAWGKg/AAAA4BKRvj8AAADA03+LPwAAACBiFKg/AAAAgJ0/DT4AAABgeAKdPwAAAMAZRG8/AAAAwCUwjT8AAABAJj8kPwAAAOAU4dE/AAAAgGxCsz8AAABg+aGYPwAAAEDp+rI+AAAAAKKs1z8AAAAApiCoPwAAACB9Ecs/AAAA4F7l4z8AAAAgu6fKPwAAAOBz6+M/AAAAwP6k1z8AAADAw3FKPwAAAED0Ga8/AAAAoPh5UD8AAABguKbgPwAAAMDOa3I/AAAAgAAGtD8AAACA7uDSPwAAACCsat8/AAAAgDyZqT8AAAAgcv1JPwAAAOCf9Zw/AAAA4CXXxT8AAAAA2HDEPwAAAADTHaE/AAAAABkUfz8AAAAgIlI8PwAAAIC7wqY/AAAAoIDIyT8AAAAAfL27PwAAAABd/+E/AAAAwESz3T8AAABApiqyPwAAAECsG9E/AAAAgNjEwz8AAACgesmAPwAAAEC/yKo/AAAAgMTh0T8AAACgLH7WPwAAAOANK7M/AAAAQDCzlj8AAADARjPbPwAAAABKdAU/AAAAgDn34D8AAAAg9nVpPwAAAACJ5OI/AAAAoE7N5D8AAAAgBTC1PwAAAGAY0ak/AAAA4E5JpT8AAABArc64PwAAAABJ2bg/AAAAIDpdAz8AAADgfZB8PwAAAOD6/6Q+AAAA4M0plz8AAADgRlCoPwAAAEBztrQ/AAAAAPMAuj8AAADABJLBPwAAAGC7frA/AAAAADS7wD8AAABgQG3FPwAAACCxp8U/AAAAIFlCtz4AAACghKphPwAAAOA7rXk/AAAAgL39hD8AAABASaaOPwAAAGAw6Y4/AAAAAAsOWD8AAAAA/zKJPwAAAKCsXYQ/AAAA4CoszT4AAAAANQWgPwAAAKCJvbE/AAAAoDjcvz8AAADAlC/EPwAAAMBM1M8/AAAAQECGpj8AAABgVp+5PwAAACC3tck/AAAAIDwY0j8AAAAgTffTPwAAAOAeUtY+AAAAgPZ0mj8AAACA3R+VPwAAAID2SKA/AAAAQPTyQT8AAACAas6vPwAAACBLVbw/AAAAADLLzD8AAAAAjXrDPwAAACC+QLM/AAAAwN/Wxz8AAABAdErQPwAAAGCoptM/AAAAwBzaMz8AAACgulhqPwAAAOB2IrI/AAAA4ELPoT8AAABgTLa2PwAAAIAytsA/AAAAgNU9pT8AAACAcqy7PwAAAIAz7cM/AAAAwLGiwj8AAADgFX2KPwAAAGCLVdM+AAAAALncpD8AAABgLp63PwAAAADvHsY/AAAAQM7KzD8AAAAAsEbYPwAAAGDcCLA/AAAAQEgqwT8AAABgt07SPwAAACClW90/AAAAoGR+5D8AAACgupVPPw==',np.array([350, 1, 5, 2, 1003, 1, 7, 4, 8, 4, 7, 4, 967, 2, 3629, 2, 2495, 3, 1, 2, 7, 2, 5, 1, 19, 1, 3, 3, 4, 2, 26, 2, 1, 1, 24, 3, 6, 2, 1, 3, 17, 4, 9, 4, 5, 2, 1, 1, 1, 1, 33, 1, 2, 1, 4, 1, 27, 4, 44, 2, 2, 3, 16, 4, 11, 1, 156, 3, 3, 5, 4, 1, 3, 2, 2, 3, 2, 3, 116, 3, 1, 1, 2, 1, 5, 2, 2, 1, 5, 3, 144, 2, 4, 3, 9, 1, 1, 2, 34, 3, 1, 1, 1, 8, 2, 2, 209, 3, 4, 4, 8, 4, 1564, 1, 1, 1, 4, 4, 8, 4, 40, 3, 1, 1, 1, 8, 5, 1, 13942],dtype=np.int64)),
    'r_thigh_jiggle2':({'head': [-0.10460905730724335, 0.03490614518523216, 0.7691643238067627], 'tail': [-0.10575634241104126, 0.068776935338974, 0.7291247844696045], 'parent': 'r_thigh', 'connect': False, 'local': True},b'AAAAYENsxD8AAABgjB3IPwAAAMDgBtg/AAAAICW14T8AAAAA8WPjPwAAAGB+QeE/AAAAQNBd3j8AAAAAx2+7PwAAAMCc9L0/AAAAIKs7sz8AAADgS57TPwAAAECcjAY/AAAA4ASUnD8AAABA5gGnPwAAAKBFj4A/AAAAAIyBcz8AAABgiU6xPwAAAKAeZ7M/AAAA4AGo2j8AAADgaXXNPwAAAICqCdU/AAAAgLWGtD8AAABAOcjHPwAAAACZXsc/AAAAIKPRtz8AAAAgdeBBPwAAAEAeltI/AAAAgFhB0T8AAADA+06ePwAAAADDAr4/AAAAoMu0yD8AAACgBV6zPwAAAKCUt0U/AAAAYNQppD8AAADA6KK8PwAAAEAqGOI/AAAAAKFY0D8AAAAAJbHdPwAAAKC4vSs/AAAAAAM+wT8AAADgz5zOPwAAAGA1xdw/AAAAIDSc2D8AAADgheamPwAAAOBLCNE/AAAAgG9dxT8AAADgb4m1PwAAAEAkadk/AAAAYOndnz8AAACgcxVhPwAAAMBtDVU/AAAA4EJcuz8AAAAA2XPIPwAAAGA6zto/AAAAQPxK4j8AAAAAiZvgPwAAAEAziLU/AAAA4GrGkj8AAAAgDz/ePwAAACAv2M4/AAAAwKkh4T8AAABg8wq7PwAAAKDkB0A/AAAAQLjX4T4AAADAkSCGPgAAAGCycUw+AAAAgM7eOz8AAADgP4GfPwAAAMBEXrc/AAAAIFTwtD8AAADgNJ+wPwAAAKABgKM/AAAAYHO5sz8AAABg/dfAPwAAACCALbE/AAAAIDMRyT8AAACAOtfKPwAAAIAFSVY/AAAAAJ7UuD8AAAAgAovDPwAAAEDz5KY/AAAAoDT9yD8AAAAAd9g8Pw==',np.array([12027, 12, 1, 1, 1, 1, 21, 2, 361, 2, 31, 3, 4, 2, 68, 5, 5, 1, 2, 2, 139, 1, 1, 1, 3, 1, 599, 1, 1, 1, 2, 2, 13, 1, 35, 5, 1, 3, 9, 3, 8, 1, 630, 1, 331, 8, 10, 1, 1, 2, 6467, 1, 20, 1, 28, 1, 473, 4, 2, 2, 6, 1, 1569, 5, 1, 3, 5, 1, 1, 1, 2223],dtype=np.int64)),
    'r_thigh_jiggle_side':({'head': [-0.14600594341754913, -0.007730054669082165, 0.8691152930259705], 'tail': [-0.18229305744171143, -0.013294694945216179, 0.8209406733512878], 'parent': 'r_thigh', 'connect': False, 'local': True},b'AAAAwCZxRD8AAACAgPCSPwAAAAAo3Y4/AAAAwPQwpD8AAAAgqjJUPwAAAABZBp8/AAAAIBBurj8AAACAV+OxPwAAAMCnsaA/AAAAoPwCrT8AAADgJNN6PwAAAACPqss/AAAAICXetD8AAADgpJHEPwAAAADhctU/AAAAIHzi2T8AAABAlXU8PwAAAICXYsY+AAAAoH+YgT8AAABAita+PwAAAEDb1Lw/AAAA4LoNpD4AAABA2ZukPwAAAIDUE7Q/AAAAQEW0RT4AAAAAZRQ6PwAAACAxfIk/AAAAYEpMeD8AAABgPwJ7PwAAAGAWGKg/AAAA4BKRvj8AAADA03+LPwAAACBiFKg/AAAAgJ0/DT4AAABgeAKdPwAAAMAZRG8/AAAAwCUwjT8AAABAJj8kPwAAAOAU4dE/AAAAgGxCsz8AAABg+aGYPwAAAEDp+rI+AAAAAKKs1z8AAAAApiCoPwAAACB9Ecs/AAAA4F7l4z8AAAAgu6fKPwAAAOBz6+M/AAAAwP6k1z8AAADAw3FKPwAAAED0Ga8/AAAAoPh5UD8AAABguKbgPwAAAMDOa3I/AAAAgAAGtD8AAACA7uDSPwAAACCsat8/AAAAgDyZqT8AAAAgcv1JPwAAAOCf9Zw/AAAA4CXXxT8AAAAA2HDEPwAAAADTHaE/AAAAABkUfz8AAAAgIlI8PwAAAIC7wqY/AAAAoIDIyT8AAAAAfL27PwAAAABd/+E/AAAAwESz3T8AAABApiqyPwAAAECsG9E/AAAAgNjEwz8AAACgesmAPwAAAEC/yKo/AAAAgMTh0T8AAACgLH7WPwAAAOANK7M/AAAAQDCzlj8AAADARjPbPwAAAABKdAU/AAAAgDn34D8AAAAg9nVpPwAAAACJ5OI/AAAAoE7N5D8AAAAgBTC1PwAAAGAY0ak/AAAA4E5JpT8AAABArc64PwAAAABJ2bg/AAAAIDpdAz8AAADgfZB8PwAAAOD6/6Q+AAAA4M0plz8AAADgRlCoPwAAAEBztrQ/AAAAAPMAuj8AAADABJLBPwAAAGC7frA/AAAAADS7wD8AAABgQG3FPwAAACCxp8U/AAAAIFlCtz4AAACghKphPwAAAOA7rXk/AAAAgL39hD8AAABASaaOPwAAAGAw6Y4/AAAAAAsOWD8AAAAA/zKJPwAAAKCsXYQ/AAAA4CoszT4AAAAANQWgPwAAAKCJvbE/AAAAoDjcvz8AAADAlC/EPwAAAMBM1M8/AAAAQECGpj8AAABgVp+5PwAAACC3tck/AAAAIDwY0j8AAAAgTffTPwAAAOAeUtY+AAAAgPZ0mj8AAACA3R+VPwAAAID2SKA/AAAAQPTyQT8AAACAas6vPwAAACBLVbw/AAAAADLLzD8AAAAAjXrDPwAAACC+QLM/AAAAwN/Wxz8AAABAdErQPwAAAGCoptM/AAAAwBzaMz8AAACgulhqPwAAAOB2IrI/AAAA4ELPoT8AAABgTLa2PwAAAIAytsA/AAAAgNU9pT8AAACAcqy7PwAAAIAz7cM/AAAAwLGiwj8AAADgFX2KPwAAAGCLVdM+AAAAALncpD8AAABgLp63PwAAAADvHsY/AAAAQM7KzD8AAAAAsEbYPwAAAGDcCLA/AAAAQEgqwT8AAABgt07SPwAAACClW90/AAAAoGR+5D8AAACgupVPPw==',np.array([12354, 1, 5, 2, 1002, 1, 7, 4, 8, 4, 7, 4, 957, 2, 3470, 2, 2483, 3, 1, 2, 7, 2, 5, 1, 19, 1, 3, 3, 4, 2, 26, 2, 1, 1, 24, 3, 5, 2, 1, 3, 16, 4, 8, 4, 5, 2, 1, 1, 1, 1, 30, 1, 2, 1, 4, 1, 26, 4, 40, 2, 2, 3, 15, 4, 11, 1, 156, 3, 3, 5, 4, 1, 3, 2, 2, 3, 2, 3, 115, 3, 1, 1, 2, 1, 5, 2, 2, 1, 5, 3, 140, 2, 4, 3, 8, 1, 1, 2, 33, 3, 1, 1, 1, 8, 2, 2, 202, 3, 4, 4, 8, 4, 1561, 1, 1, 1, 4, 4, 8, 4, 39, 3, 1, 1, 1, 8, 5, 1, 2150],dtype=np.int64)),
    'r_glute':({'head': [-0.09685496240854263, 0.0867934301495552, 0.9802164435386658], 'tail': [-0.09942950308322906, 0.10083959251642227, 0.9657030701637268], 'parent': 'pelvis', 'connect': False, 'local': True},b'AAAAAO/Fpj4AAACgLicqPgAAAMDS9eI+AAAAoIrzBj8AAADgh0GgPgAAAED1kr0+AAAAAJXUKj4AAAAAg1S9PgAAAMBKn6Y9AAAAgE2Vvj4AAADATHlUPgAAAICYqmI+AAAAIAwDXj0AAABgKQtyPgAAAADXono+AAAAAFPZkz0AAABg5Fo2PgAAAGAiOQM+AAAAAL04Mz4AAAAAe77oPAAAAGDPsY09AAAAwPsQoj0AAAAA/dHVPQAAAMCYYaA9AAAAgONFij0AAABAUORsPQAAAMAEuJY+AAAAAO/2Yj4AAABgr4gmPgAAAEAMqBM+AAAAwLfrXz4AAADgCneHPgAAAKC9cMU9AAAAgBa7JT4AAABgY9HIPAAAAKBOzFw+AAAAYPGqXT4AAADgEt+RPwAAAIBDCrM/AAAAgH4Ryz8AAADAY63CPwAAAEBKG7c/AAAA4FV1NT8AAAAg2EjgPwAAAEDjDME/AAAAoPyFzT8AAAAAKVBZPwAAAKDyDnw/AAAAQDGjLD8AAABAH7u9PwAAAOCgVOE/AAAAIEjs1z8AAADA5rbnPwAAAABpC+U/AAAAgP4a5T8AAACA1gTnPwAAACCVdc0/AAAAoC+b1T8AAAAAP0W0PwAAAEDHjec/AAAAIL+p5j8AAADA+nLpPwAAAAB0JeM/AAAAYH6k4T8AAAAgEWvIPwAAAABVmFY/AAAAQNBteD8AAAAglAqAPwAAAOB0YWA/AAAAgJBpgj8AAACgYKmIPwAAAICfZ4k/AAAAwPfFjD8AAABAmmxzPwAAAEDZRyk/AAAAAI0J2z8AAADAmNidPwAAAMD7GdA/AAAA4DY34j8AAADgMWq+PwAAAGBQ1+I/AAAAIBuG4T8AAAAguPN+PwAAAEBqmNI/AAAAwONq4j8AAADAE13dPwAAAKC/e9Y/AAAA4IyeSD8AAABAGWKZPwAAAIDyTJo/AAAAAHN0iz8AAADgpR/hPwAAAACR29w/AAAAAMH31j8AAACgzeJfPwAAAAB8x+A/AAAA4MRw2z8AAAAg8FrVPwAAAEAiptU/AAAAwGu91T8AAACAxCfRPwAAAMDUkMk/AAAAAED2TD8AAACgVGrtPgAAAMChbS8+AAAAwPoUDj8AAAAApSW3PwAAACBMYa0/AAAAwPvuwT8AAADAnLi5PwAAACAi2ZU/AAAAQKQXhD8AAABgffJhPwAAAICMZdk/AAAAIK1L2D8AAACgbzPAPwAAAOCz69E/AAAAYPVqsD8AAABATuzXPwAAAKAW4Xk/AAAAAAag1T8AAACA8OTLPwAAAMAPUNk/AAAAQAFo1D8AAADgfQTOPwAAAACAL8Q/AAAA4AFMzj8AAADAQg8/PwAAAEAp0MI/AAAAoNZjpD8AAADAjDhgPgAAAOCoNI8/AAAAIF/IQD8AAADACemKPwAAAIC4WsQ+AAAAoJN1ij8AAADg8H6lPwAAAICNRek/AAAAQGIK6T8AAADAHlfnPwAAAGBtD80/AAAA4J3xuz8AAADgAPrhPwAAAODfINg/AAAAQKvp5T8AAAAA3RjiPwAAAEBt0aE/AAAAIAYwaD8AAABAN8vtPgAAAGAbrI8/AAAAIBVh2D8AAADg35GXPwAAAGBr7M4/AAAAQKkizT8AAACAJRS2PwAAAEBTrMY/AAAA4KbpmT8AAACgA8DOPwAAAKDS3yw/AAAAgDd9yD8AAAAApTS+PwAAAABEOLc/AAAAYMSMpz8AAADAUT/HPwAAAAAHRuQ/AAAA4C4g5D8AAABAzPvhPwAAAOArT7s/AAAAQMzRyT8AAACgyEbUPwAAACDuhNw/AAAA4IZi4j8AAADgRi/gPwAAAIAnCKA/AAAAgGiVdj8AAABgzZQXPwAAAGDHmVw/AAAAII7k0j8AAABgY8ONPwAAAMD5XSo/AAAA4Jgdpj8AAADAWrK7PwAAACDHdsA/AAAAgL5wvz8AAADAbY+1PwAAAACdtcI/AAAAQPiwwj8AAACgRdSzPwAAAABb4mQ/AAAAoJM3sT8AAADAxmGzPwAAAGBbP7g/AAAAQGEFkz8AAAAgZpaVPwAAAKCL8JY/AAAAoF5gkz8AAABAjoKSPwAAAAAXW40/AAAAQLynnD8AAACgpASgPwAAAMALFoY/AAAAYCuzwD8AAAAglyyzPwAAACAcTcc/AAAAQPIhkz8AAADAOC0tPgAAAIA2JM0/AAAAACTvwD8AAACgxnW7PwAAAKAZF6E/AAAA4L6DsD8AAADgNXc0PwAAAAA2m7o/AAAAQGnCuT8AAABg+FWkPwAAAMB7Bmw/AAAAAODJlT8AAABAJjOUPwAAAOB6aZ0/AAAAIDhrlD8AAABgdUfNPwAAAMA/6Lo/AAAAwFRfuj8AAAAAZSelPwAAAKDcEqQ/AAAAoKQRkz8AAAAg+VqnPwAAAKA3M54/AAAAwAgXhT8AAADA6udaPwAAAABq7uQ+AAAAYENOiT8AAACgdV78PgAAAECgNZA/AAAAoFgUqj8AAAAAXTyiPwAAAKAvLk8/AAAAALF9ZD8AAADgVaxpPwAAAIB4G+E+AAAAID3VEj8AAABA86BDPgAAAOBGGOk+AAAAoK82JD8=',np.array([8453, 2, 1, 1, 1, 1, 158, 1, 491, 3, 145, 1, 51, 2, 102, 2, 1771, 3, 3, 1, 2, 2, 48, 1, 1, 1, 1, 2, 3, 1, 3, 2, 2, 6, 2, 4, 1, 1, 9013, 1, 1, 1, 1, 2, 1, 2, 2, 1, 7, 2, 4, 2, 1, 17, 16, 2, 2, 2, 4, 4, 1, 1, 19, 1, 5, 7, 14, 6, 10, 3, 3, 11, 62, 1, 2, 1, 293, 1, 3, 1, 96, 2, 6, 2, 3, 3, 3, 9, 20, 8, 103, 1, 6, 2, 5, 2, 7, 14, 12, 2, 28, 9, 8, 2, 30, 14, 10, 2, 25, 1, 8, 2, 1, 6, 4, 1, 1708, 1, 1, 10, 2, 4, 12, 1, 1, 3, 1, 6, 1, 1, 8, 4, 2, 11, 3, 1, 2, 3, 2, 6, 1, 5, 2116],dtype=np.int64)),
    'r_glute2':({'head': [-0.08702501654624939, 0.08861622959375381, 0.9432905912399292], 'tail': [-0.08978187292814255, 0.10098331421613693, 0.927352786064148], 'parent': 'pelvis', 'connect': False, 'local': True},b'AAAAYHZrwj4AAADg767gPgAAACD6mTs/AAAAIAw92D8AAADAnB5vPwAAAOAqj4E/AAAAwFcDZj8AAADAKM4APwAAAGArunA/AAAAILwUoT8AAABATVTJPwAAAACxi8E/AAAAoNiyyz8AAAAAAUnFPwAAAECdyM0/AAAAIPK+zj8AAAAgJP6zPwAAAICJP9c/AAAA4PLquj8AAADARBXiPwAAAMDK9eE/AAAAIO6n4j8AAAAATWLfPwAAAMBHleA/AAAAIAPOyT8AAABAqgDDPwAAAIDIDcY/AAAA4K7ntz8AAADA3eqtPwAAAMDwWsY/AAAAgKTKwj8AAACg05W8PwAAAKCsQIQ/AAAA4LRawj8AAABgbKDSPwAAAOCKx9M/AAAAAAvJ1D8AAACgtfzQPwAAAMAQY7I/AAAAYPfuyz8AAADgNDfjPwAAAIAVbb8/AAAAAEJs2z8AAACgcz7nPwAAAMDHssw/AAAAYE6h6D8AAAAA1JHnPwAAAGBFiSg/AAAAII3xvz8AAABgnPHkPwAAAKACGOE/AAAAYLLXoT8AAAAA3sWGPwAAAMCo9fk+AAAAwFZgqT8AAABAGHlfPwAAACAwBJE/AAAA4HC0kT8AAADAw29+PgAAAEAgxis/AAAAAD7GGD8AAADgtUGyPwAAAKADja0/AAAAoI6FsT8AAACA9gOnPwAAAIBIZnQ/AAAAgAwEoD8AAACAJqRwPwAAACChgbU/AAAAwOvguj8AAADgVPtbPwAAAIARd+s/AAAAwPkW7T8AAAAgTqzaPwAAAECKTuU/AAAAQLnjzD8AAADAVLbrPwAAAKDTgr4/AAAAwN275z8AAAAAoyniPwAAACBziag/AAAA4CryhD8AAADgOSK4PwAAAGBvoNA/AAAAQN9w1T8AAACAfggAPwAAAEANEog/AAAAYFYF1z8AAADgL3gvPwAAAOBzhao/AAAAYHn0wT8AAAAgDtrJPwAAAODGoJM/AAAAgJUdsj8AAACAL5+9PgAAAMAkj7w/AAAAoG7O2D8AAAAAQi7bPwAAAOBVmds/AAAAYIl1wj8AAADARMqxPwAAAIBy5NY/AAAAQLWyzz8AAAAgLnzUPwAAAID/Vc8/AAAAgGG4kD8AAACgjGNkPwAAAGDtKQ4/AAAAAFpxxT8AAAAAaQa6PwAAAIC6SOw/AAAAAPXh6j8AAADA+ZbVPwAAAKCpR+E/AAAAYDaKyD8AAACAXrTmPwAAAOAFGbg/AAAAYGF/5D8AAABAfNugPwAAAGC41YM/AAAAwG4gvj4AAACADYrePwAAAOCjZdM/AAAAICSJWD4AAACAk32YPwAAAADsHbk/AAAAwINstT8AAACglsu2PwAAACC0QyA/AAAAQM/JVj8AAAAAi2yVPwAAAGDctrA/AAAAII5Crj8AAACA/gGtPwAAAAAiMQc/AAAAgD+X0z4AAACgvsKlPwAAAIA9K5Y/AAAAoHLQcD8AAABAcobUPgAAAODP82o/AAAAQPidyT4AAABALHBlPwAAAGCB+T0/AAAAoLgU1D4AAABAqEfiPgAAACAx68g/AAAAoCllyj8AAAAgCBjEPwAAAMAq9NM+AAAAgLaEtz8AAABAnDekPwAAAMAOlrs/AAAAgFOM4T8AAACAyZDgPwAAAKA+rc0/AAAAgHJy1z8AAABgcxLBPwAAAEDSnd8/AAAAYLMprT8AAABA7+TaPwAAAGDSP40/AAAAoHVhcz8AAABg5VTSPwAAAMCrP8s/AAAAYAw8xz8AAAAARq20PwAAAMCoX6I/AAAAIFYPwj8AAAAgX2SuPwAAAMB6T5s/AAAAwFtNrD8AAAAggiGgPwAAAEDqpYo/AAAAgBcSpj8AAADAdr5PPwAAAGAEwqY/AAAAQPA+qD8AAACAxPzuPgAAAGCfWBQ/AAAAQL5CXD8AAACgUZN3PwAAAGAvKHQ/AAAAIDJdej8=',np.array([13378, 1, 15, 1, 6898, 1, 6, 1, 1, 1, 10, 3, 1, 17, 16, 8, 1, 7, 25, 7, 15, 4, 11, 2, 1, 2, 1, 3, 4, 1, 9, 1, 58, 1, 70, 5, 2, 2, 315, 2, 6, 1, 10, 10, 15, 1, 3, 5, 1, 1, 109, 1, 1, 2, 5, 2, 3, 1, 2, 2, 1, 12, 13, 1, 28, 10, 4, 2, 1, 2, 30, 13, 12, 1, 1756, 2, 1, 4, 4, 1, 2, 4, 12, 5, 1, 9, 4, 1, 2, 4, 2, 11, 3, 1, 1, 3, 4, 2, 2125],dtype=np.int64)),
    'l_glute':({'head': [0.09685496240854263, 0.0867934301495552, 0.9802164435386658], 'tail': [0.09942950308322906, 0.10083959251642227, 0.9657030701637268], 'parent': 'pelvis', 'connect': False, 'local': True},b'AAAAAO/Fpj4AAACgLicqPgAAAMDS9eI+AAAAoIrzBj8AAADgEt+RPwAAAIBDCrM/AAAAgH4Ryz8AAADAY63CPwAAAEBKG7c/AAAA4FV1NT8AAAAg2EjgPwAAAEDjDME/AAAAoPyFzT8AAAAAKVBZPwAAAKDyDnw/AAAAQDGjLD8AAABAH7u9PwAAAOCgVOE/AAAAIEjs1z8AAADA5rbnPwAAAABpC+U/AAAAgP4a5T8AAACA1gTnPwAAACCVdc0/AAAAoC+b1T8AAAAAP0W0PwAAAEDHjec/AAAAIL+p5j8AAADA+nLpPwAAAAB0JeM/AAAAYH6k4T8AAAAgEWvIPwAAAABVmFY/AAAAQNBteD8AAAAglAqAPwAAAOB0YWA/AAAAgJBpgj8AAACgYKmIPwAAAICfZ4k/AAAAwPfFjD8AAABAmmxzPwAAAEDZRyk/AAAAAI0J2z8AAADAmNidPwAAAMD7GdA/AAAA4DY34j8AAADgMWq+PwAAAGBQ1+I/AAAAIBuG4T8AAAAguPN+PwAAAEBqmNI/AAAAwONq4j8AAADAE13dPwAAAKC/e9Y/AAAA4IyeSD8AAABAGWKZPwAAAIDyTJo/AAAAAHN0iz8AAADgpR/hPwAAAACR29w/AAAAAMH31j8AAACgzeJfPwAAAOCHQaA+AAAAAHzH4D8AAADgxHDbPwAAACDwWtU/AAAAQCKm1T8AAADAa73VPwAAAIDEJ9E/AAAAwNSQyT8AAAAAQPZMPwAAAKBUau0+AAAAwKFtLz4AAADA+hQOPwAAAAClJbc/AAAAIExhrT8AAADA++7BPwAAAMCcuLk/AAAAICLZlT8AAABApBeEPwAAAGB98mE/AAAAgIxl2T8AAAAgrUvYPwAAAKBvM8A/AAAA4LPr0T8AAABg9WqwPwAAAEBO7Nc/AAAAoBbheT8AAAAABqDVPwAAAIDw5Ms/AAAAwA9Q2T8AAABAAWjUPwAAAOB9BM4/AAAAAIAvxD8AAADgAUzOPwAAAMBCDz8/AAAAQCnQwj8AAACg1mOkPwAAAMCMOGA+AAAA4Kg0jz8AAAAgX8hAPwAAAMAJ6Yo/AAAAgLhaxD4AAACgk3WKPwAAAODwfqU/AAAAgI1F6T8AAABAYgrpPwAAAMAeV+c/AAAAYG0PzT8AAADgnfG7PwAAAOAA+uE/AAAA4N8g2D8AAABAq+nlPwAAAADdGOI/AAAAQG3RoT8AAAAgBjBoPwAAAEA3y+0+AAAAgE2Vvj4AAABgG6yPPwAAACAVYdg/AAAA4N+Rlz8AAABga+zOPwAAAECpIs0/AAAAgCUUtj8AAABAU6zGPwAAAOCm6Zk/AAAAoAPAzj8AAACg0t8sPwAAAIA3fcg/AAAAAKU0vj8AAAAARDi3PwAAAGDEjKc/AAAAwFE/xz8AAAAAB0bkPwAAAOAuIOQ/AAAAQMz74T8AAADgK0+7PwAAAEDM0ck/AAAAoMhG1D8AAAAg7oTcPwAAAOCGYuI/AAAA4EYv4D8AAACAJwigPwAAAIBolXY/AAAAYM2UFz8AAACAmKpiPgAAAGDHmVw/AAAAII7k0j8AAABgY8ONPwAAAMD5XSo/AAAA4Jgdpj8AAADAWrK7PwAAACDHdsA/AAAAgL5wvz8AAADAbY+1PwAAAACdtcI/AAAAQPiwwj8AAACgRdSzPwAAAABb4mQ/AAAAoJM3sT8AAADAxmGzPwAAAGBbP7g/AAAAQGEFkz8AAAAgZpaVPwAAAKCL8JY/AAAAoF5gkz8AAABAjoKSPwAAAAAXW40/AAAAQLynnD8AAACgpASgPwAAAMALFoY/AAAAYCuzwD8AAAAglyyzPwAAACAcTcc/AAAAQPIhkz8AAADAOC0tPgAAAIA2JM0/AAAAACTvwD8AAACgxnW7PwAAAKAZF6E/AAAA4L6DsD8AAADgNXc0PwAAAAA2m7o/AAAAQGnCuT8AAABg+FWkPwAAAMB7Bmw/AAAAAODJlT8AAABAJjOUPwAAAOB6aZ0/AAAAIDhrlD8AAABgdUfNPwAAAMA/6Lo/AAAAwFRfuj8AAAAAZSelPwAAAKDcEqQ/AAAAoKQRkz8AAAAg+VqnPwAAAKA3M54/AAAAwAgXhT8AAADA6udaPwAAAABq7uQ+AAAAYENOiT8AAACgdV78PgAAAECgNZA/AAAAoFgUqj8AAAAAXTyiPwAAAKAvLk8/AAAAALF9ZD8AAADgVaxpPwAAAIB4G+E+AAAAID3VEj8AAABA86BDPgAAAOBGGOk+AAAAoK82JD8AAABA9ZK9PgAAAACV1Co+AAAAAINUvT4AAADASp+mPQAAAMBMeVQ+AAAAIAwDXj0AAABgKQtyPgAAAADXono+AAAAAFPZkz0AAABg5Fo2PgAAAGAiOQM+AAAAAL04Mz4AAAAAe77oPAAAAGDPsY09AAAAwPsQoj0AAAAA/dHVPQAAAMCYYaA9AAAAgONFij0AAABAUORsPQAAAMAEuJY+AAAAAO/2Yj4AAABgr4gmPgAAAEAMqBM+AAAAwLfrXz4AAADgCneHPgAAAKC9cMU9AAAAgBa7JT4AAABgY9HIPAAAAKBOzFw+AAAAYPGqXT4=',np.array([8453, 2, 1, 1, 1, 1, 8, 1, 1, 1, 1, 2, 1, 2, 2, 1, 7, 2, 4, 2, 1, 17, 16, 2, 2, 2, 4, 4, 1, 1, 19, 1, 5, 7, 14, 6, 11, 3, 3, 12, 66, 1, 2, 1, 299, 1, 3, 1, 97, 2, 6, 2, 3, 3, 3, 9, 20, 8, 105, 1, 6, 2, 6, 2, 8, 14, 13, 3, 28, 9, 8, 2, 30, 14, 10, 3, 26, 1, 8, 2, 2, 6, 4, 1, 1715, 1, 1, 10, 2, 4, 12, 1, 1, 3, 2, 6, 1, 1, 8, 4, 2, 11, 3, 1, 2, 3, 2, 6, 2, 4, 1, 1, 9642, 3, 143, 1, 48, 1, 102, 1, 1762, 3, 3, 1, 2, 2, 47, 1, 1, 1, 1, 2, 3, 1, 3, 2, 2, 6, 1, 5, 2116],dtype=np.int64)),
    'l_glute2':({'head': [0.08702501654624939, 0.08861622959375381, 0.9432905912399292], 'tail': [0.08978187292814255, 0.10098331421613693, 0.927352786064148], 'parent': 'pelvis', 'connect': False, 'local': True},b'AAAAYHZrwj4AAADg767gPgAAACD6mTs/AAAAIAw92D8AAADAnB5vPwAAAOAqj4E/AAAAwFcDZj8AAADAKM4APwAAAGArunA/AAAAILwUoT8AAABATVTJPwAAAACxi8E/AAAAoNiyyz8AAAAAAUnFPwAAAECdyM0/AAAAIPK+zj8AAAAgJP6zPwAAAICJP9c/AAAA4PLquj8AAADARBXiPwAAAMDK9eE/AAAAIO6n4j8AAAAATWLfPwAAAMBHleA/AAAAIAPOyT8AAABAqgDDPwAAAIDIDcY/AAAA4K7ntz8AAADA3eqtPwAAAMDwWsY/AAAAgKTKwj8AAACg05W8PwAAAKCsQIQ/AAAA4LRawj8AAABgbKDSPwAAAOCKx9M/AAAAAAvJ1D8AAACgtfzQPwAAAMAQY7I/AAAAYPfuyz8AAADgNDfjPwAAAIAVbb8/AAAAAEJs2z8AAACgcz7nPwAAAMDHssw/AAAAYE6h6D8AAAAA1JHnPwAAAGBFiSg/AAAAII3xvz8AAABgnPHkPwAAAKACGOE/AAAAYLLXoT8AAAAA3sWGPwAAAMCo9fk+AAAAwFZgqT8AAABAGHlfPwAAACAwBJE/AAAA4HC0kT8AAADAw29+PgAAAEAgxis/AAAAAD7GGD8AAADgtUGyPwAAAKADja0/AAAAoI6FsT8AAACA9gOnPwAAAIBIZnQ/AAAAgAwEoD8AAACAJqRwPwAAACChgbU/AAAAwOvguj8AAADgVPtbPwAAAIARd+s/AAAAwPkW7T8AAAAgTqzaPwAAAECKTuU/AAAAQLnjzD8AAADAVLbrPwAAAKDTgr4/AAAAwN275z8AAAAAoyniPwAAACBziag/AAAA4CryhD8AAADgOSK4PwAAAGBvoNA/AAAAQN9w1T8AAACAfggAPwAAAEANEog/AAAAYFYF1z8AAADgL3gvPwAAAOBzhao/AAAAYHn0wT8AAAAgDtrJPwAAAODGoJM/AAAAgJUdsj8AAACAL5+9PgAAAMAkj7w/AAAAoG7O2D8AAAAAQi7bPwAAAOBVmds/AAAAYIl1wj8AAADARMqxPwAAAIBy5NY/AAAAQLWyzz8AAAAgLnzUPwAAAID/Vc8/AAAAgGG4kD8AAACgjGNkPwAAAGDtKQ4/AAAAAFpxxT8AAAAAaQa6PwAAAIC6SOw/AAAAAPXh6j8AAADA+ZbVPwAAAKCpR+E/AAAAYDaKyD8AAACAXrTmPwAAAOAFGbg/AAAAYGF/5D8AAABAfNugPwAAAGC41YM/AAAAwG4gvj4AAACADYrePwAAAOCjZdM/AAAAICSJWD4AAACAk32YPwAAAADsHbk/AAAAwINstT8AAACglsu2PwAAACC0QyA/AAAAQM/JVj8AAAAAi2yVPwAAAGDctrA/AAAAII5Crj8AAACA/gGtPwAAAAAiMQc/AAAAgD+X0z4AAACgvsKlPwAAAIA9K5Y/AAAAoHLQcD8AAABAcobUPgAAAODP82o/AAAAQPidyT4AAABALHBlPwAAAGCB+T0/AAAAoLgU1D4AAABAqEfiPgAAACAx68g/AAAAoCllyj8AAAAgCBjEPwAAAMAq9NM+AAAAgLaEtz8AAABAnDekPwAAAMAOlrs/AAAAgFOM4T8AAACAyZDgPwAAAKA+rc0/AAAAgHJy1z8AAABgcxLBPwAAAEDSnd8/AAAAYLMprT8AAABA7+TaPwAAAGDSP40/AAAAoHVhcz8AAABg5VTSPwAAAMCrP8s/AAAAYAw8xz8AAAAARq20PwAAAMCoX6I/AAAAIFYPwj8AAAAgX2SuPwAAAMB6T5s/AAAAwFtNrD8AAAAggiGgPwAAAEDqpYo/AAAAgBcSpj8AAADAdr5PPwAAAGAEwqY/AAAAQPA+qD8AAACAxPzuPgAAAGCfWBQ/AAAAQL5CXD8AAACgUZN3PwAAAGAvKHQ/AAAAIDJdej8=',np.array([1375, 1, 15, 1, 7079, 1, 6, 1, 1, 1, 10, 3, 1, 17, 16, 8, 1, 7, 25, 7, 15, 4, 12, 2, 1, 2, 1, 3, 5, 1, 9, 1, 62, 1, 75, 5, 2, 2, 317, 2, 6, 1, 10, 10, 15, 1, 3, 5, 1, 1, 111, 1, 1, 2, 6, 2, 4, 1, 2, 2, 1, 12, 15, 1, 28, 10, 4, 2, 1, 2, 30, 13, 13, 1, 1765, 2, 1, 4, 4, 1, 2, 4, 12, 5, 2, 9, 4, 1, 2, 4, 2, 11, 3, 1, 1, 3, 4, 2, 13917],dtype=np.int64)),
}

def first_non_zero(x):
    idx = x.view(bool).argmax() // x.itemsize
    return idx if x[idx] else -1

def get_disconnected_components(bm):
    not_visited = np.ones(len(bm.verts), dtype=bool)
    disconnected_components = np.zeros(len(bm.verts), dtype=np.int16)
    comp_num=0
    while True:
        i = first_non_zero(not_visited)
        if i < 0:
            break
        stack = [i]
        disconnected_components[i] = comp_num
        not_visited[i] = False
        while len(stack) > 0:
            next_idx = stack.pop()
            for e in bm.verts[next_idx].link_edges:
                for neighbour in e.verts:
                    i = neighbour.index
                    if not_visited[i]:
                        disconnected_components[i] = comp_num
                        not_visited[i] = False
                        stack.append(i)
        comp_num += 1
    return disconnected_components, comp_num


def count_vectors_in_dcomps(dcomps, dcomps_num):
    counts = np.bincount(dcomps, minlength=dcomps_num)
    return counts

def sum_vectors_in_dcomps(vectors, dcomps, dcomps_num):
    average_vector = [np.bincount(dcomps, weights=vectors[:, i], minlength=dcomps_num) for i in range(3)]
    average_vector = np.stack(average_vector)
    return average_vector

def average_vectors_in_dcomps(vectors, dcomps, dcomps_num):
    counts = count_vectors_in_dcomps(dcomps, dcomps_num)
    average_vector = sum_vectors_in_dcomps(vectors, dcomps, dcomps_num)
    average_vector /= counts
    return average_vector


def is_additional_bone(bone_name):
    return bone_name in ADDITIONAL_BONES

def is_daz_bone(bone_name):
    return bone_name in DAZ_G9_TO_UE5_BONES or bone_name in OTHER_DAZ_BONES

def is_known_bone(bone_name):
    return is_additional_bone(bone_name) or is_daz_bone(bone_name)

def serialize_bone_and_weights(obj, bone_names):
    import base64
    rig = get_rig_of(obj)
    select_object(rig)
    bpy.ops.object.mode_set(mode='EDIT')
    bones = {}
    for bone_name in bone_names:
        bone = rig.data.edit_bones[bone_name]
        parent_bone = None if bone.parent is None else bone.parent.name
        bone_head = list(bone.head)
        bone_tail = list(bone.tail)
        bones[bone_name] = {
            'head': bone_head,
            'tail': bone_tail,
            'parent': parent_bone,
            'connect': bone.use_connect,
            'local': bone.use_local_location
        }

    select_object(obj)
    for bone_name in bone_names:
        vg = obj.vertex_groups.get(bone_name)
        if vg is not None:
            bone = bones[bone_name]
            weights, indices, is_non_zero = get_weights_as_sparse(obj, vg)
            print(f"'{bone_name}':({bone},{base64.b64encode(weights)},np.array({indices.tolist()},dtype=np.{indices.dtype})),")


def apply_additional_bone(obj, bone_names):
    import base64
    rig = get_rig_of(obj)
    select_object(rig)
    bpy.ops.object.mode_set(mode='EDIT')
    for bone_name in bone_names:
        if bone_name in ADDITIONAL_BONES:
            bone = rig.data.edit_bones.get(bone_name)
            if bone is None:
                bone = rig.data.edit_bones.new(bone_name)
            bone_params, weights, indices = ADDITIONAL_BONES.get(bone_name)
            bone.head = bone_params['head']
            bone.tail = bone_params['tail']
            parent_bone = bone_params['parent']
            parent_bone = rig.data.edit_bones[parent_bone]
            bone.parent = parent_bone
            bone.use_connect = bone_params['connect']
            bone.use_local_location = bone_params['local']
            vg = obj.vertex_groups.get(bone_name)
            if vg is None:
                vg = obj.vertex_groups.new(name=bone_name)
            weights = base64.decodebytes(weights)
            weights = np.frombuffer(weights, dtype=np.float64)
            is_non_zero = rle_decode(indices, (NUM_OF_VERTICES_IN_DAZ_BASE_MESH,))
            indices, = np.where(is_non_zero)
            for val, idx in zip(weights.tolist(), indices.tolist()):
                vg.add(index=(idx,), weight=val, type='REPLACE')


def find_child_meshes(o):
    out = []
    def find_child_meshes_recursive(o):
        for c in o.children:
            if isinstance(c.data, bpy.types.Mesh):
                out.append(c)
            find_child_meshes_recursive(c)
    find_child_meshes_recursive(o)
    return out


def remove_unnecessary_shape_keys(objs=None, tolerance=0.001):
    if objs is None:
        objs = bpy.context.selected_objects
    if isinstance(objs, str):
        objs = bpy.data.objects[objs]
    if isinstance(objs, bpy.types.Object):
        objs = [objs]

    assert bpy.context.mode == 'OBJECT', "Must be in object mode!"

    for ob in objs:
        if ob.type != 'MESH': continue
        if not ob.data.shape_keys: continue
        if not ob.data.shape_keys.use_relative: continue

        kbs = ob.data.shape_keys.key_blocks
        nverts = len(ob.data.vertices)
        to_delete = []

        # Cache locs for rel keys since many keys have the same rel key
        cache = {}

        locs = np.empty(3 * nverts, dtype=np.float32)

        for kb in kbs:
            if kb == kb.relative_key: continue

            kb.data.foreach_get("co", locs)

            if kb.relative_key.name not in cache:
                rel_locs = np.empty(3 * nverts, dtype=np.float32)
                kb.relative_key.data.foreach_get("co", rel_locs)
                cache[kb.relative_key.name] = rel_locs
            rel_locs = cache[kb.relative_key.name]

            locs -= rel_locs
            if (np.abs(locs) < tolerance).all():
                to_delete.append(kb.name)

        for kb_name in to_delete:
            ob.shape_key_remove(ob.data.shape_keys.key_blocks[kb_name])


def transfer_weights_to_object(src_obj, dst_obj, vg_name=None, interp='POLYINTERP_NEAREST'):
    if vg_name is not None:
        if vg_name not in src_obj.vertex_groups:
            raise Exception(vg_name + " does not exist in " + src_obj)

    i = len(dst_obj.modifiers)
    m = dst_obj.modifiers.new('DataTransfer', 'DATA_TRANSFER')
    dst_obj.modifiers.move(i, 0)
    m.object = src_obj
    m.use_vert_data = True
    m.data_types_verts = {'VGROUP_WEIGHTS'}
    m.vert_mapping = interp
    if vg_name is None:
        for vg in dst_obj.vertex_groups:
            dst_obj.vertex_groups.remove(vg)
        for vg in src_obj.vertex_groups:
            dst_obj.vertex_groups.new(name=vg.name)
    else:
        if vg_name in dst_obj.vertex_groups:
            dst_obj.vertex_groups.remove(dst_obj.vertex_groups[vg_name])
        dst_obj.vertex_groups.new(name=vg_name)
        print("Transferring ", "all weights" if vg_name is None else vg_name, " for ", dst_obj)
        m.layers_vgroup_select_src = vg_name
    with bpy.context.temp_override(object=dst_obj):
        bpy.ops.object.modifier_apply(modifier=m.name)


def clean_up_unnecessary_groups(o):
    used_groups = [False] * len(o.vertex_groups)
    for v in o.data.vertices:
        for vg in v.groups:
            if vg.weight > 0.001:
                used_groups[vg.group] = True
    print(o.name, 'retained:')
    groups_to_remove = []
    for vgi, is_used in enumerate(used_groups):
        vg = o.vertex_groups[vgi]
        if is_used:
            print('    ', vg.name)
        else:
            groups_to_remove.append(vg)
    print(o.name, 'removed:')
    for vg in groups_to_remove:
        print('    ', vg.name)
        o.vertex_groups.remove(vg)


def clean_up_unnecessary_groups_for_all_objs(objs=None):
    if objs is None:
        objs = bpy.context.selected_objects
    if isinstance(objs, str):
        objs = bpy.data.objects[objs]
    if isinstance(objs, bpy.types.Object):
        objs = [objs]
    for o in objs:
        clean_up_unnecessary_groups(o)


def transfer_weights(src_obj, dst_objs, vg_names=None, interp='POLYINTERP_NEAREST'):
    if isinstance(dst_objs, str):
        dst_objs = bpy.data.objects[dst_objs]
    if isinstance(dst_objs, bpy.types.Object):
        dst_objs = [dst_objs]
    if vg_names is not None:
        if isinstance(vg_names, str):
            vg_names = [vg_names]
    for o in dst_objs:
        if isinstance(o, str):
            o = bpy.data.objects[o]
        if vg_names is None:
            transfer_weights_to_object(src_obj, o, interp=interp)
        else:
            for vg_name in vg_names:
                if vg_name.endswith("."):
                    transfer_weights_to_object(src_obj, o, vg_name + "L", interp=interp)
                    transfer_weights_to_object(src_obj, o, vg_name + "R", interp=interp)
                else:
                    transfer_weights_to_object(src_obj, o, vg_name, interp=interp)
        clean_up_unnecessary_groups(o)


def select_bone(bone):
    bone.select = True
    bone.select_head = True
    bone.select_tail = True


def find_original_body_rig():
    for o in bpy.data.objects:
        if o.name != 'root' and o.parent is None and isinstance(o.data, bpy.types.Armature):
            if o.daz_importer.DazRig == 'genesis9':
                return o


def find_body_rig():
    root = bpy.data.objects.get('root')
    if root is not None:
        return root
    for o in bpy.data.objects:
        if o.parent is None and isinstance(o.data, bpy.types.Armature):
            if o.daz_importer.DazRig == 'genesis9':
                return o
    return None


def find_body_mesh():
    body_rig = find_body_rig()
    return bpy.data.objects[body_rig.name + ' Mesh']


def is_toon(body_mesh):
    for mat in body_mesh.data.materials:
        for node_group in NodesUtils.find_all_by_type(mat.node_tree, bpy.types.ShaderNodeGroup):
            if 'Toon' in node_group.node_tree.name:
                return True
    return False


def hide_object(obj, hide=False):
    obj.hide_set(hide)
    obj.hide_viewport = hide
    obj.hide_render = hide


def select_object(obj):
    if bpy.context.view_layer.objects.active is not None:
        if bpy.context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = obj
    if obj is not None:
        hide_object(obj, False)
        obj.select_set(True)
        if bpy.context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')


def apply_vertex_group_weights(group:bpy.types.VertexGroup, weights:np.array, epsilon:float = 0.001, old_weights=None):
    new_weights_present = weights>epsilon
    if old_weights is not None:
        #old_weights_vanished = np.logical_and(old_weights > 0, np.logical_not(new_weights_present))
        new_weights_present = np.logical_or(new_weights_present, old_weights > 0)
    values = weights[new_weights_present]
    indices, = np.where(new_weights_present)
    for val, idx in zip(values.tolist(), indices.tolist()):
        group.add(index=(idx,), weight=val, type='REPLACE')


def subdivide_bone(cuts, mesh, rig, bone_name):
    if cuts < 1:
        return
    select_object(rig)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.armature.select_all(action='DESELECT')
    bone = rig.data.edit_bones[bone_name]
    select_bone(bone)
    bpy.ops.armature.subdivide(number_cuts=cuts)
    vertex_groups = []
    for i in range(0, cuts):
        subbone_name = bone_name + "." + str(i + 1).zfill(3)
        subbone = rig.data.edit_bones[subbone_name]
        subbone_name = subbone.name = bone_name + str(i + 1)
        group = mesh.vertex_groups.new(name=subbone_name)
        vertex_groups.append(group)


    old_group = mesh.vertex_groups[bone_name]
    select_object(mesh)
    bpy.ops.object.mode_set(mode='EDIT')

    old_weights = get_weights_as_array(mesh, old_group)
    bpy.ops.object.mode_set(mode='OBJECT')
    max_weight = np.max(old_weights)
    steps = len(vertex_groups) + 1
    step = max_weight / steps
    #pec_weights_normalised = pec_weights/max_weight
    subpec_groups_and_weights = [(old_group, old_weights.copy())]
    subpec_groups_and_weights.extend((subpec_group, (old_weights - step * (i + 1)).clip(0, 1)) for i, subpec_group in enumerate(vertex_groups))
    _, prev_weights = subpec_groups_and_weights[0]
    for _, next_weights in subpec_groups_and_weights[1:]:
        prev_weights[:] = (prev_weights[:]-next_weights[:]).clip(0, 1)
        prev_weights = next_weights
        # subpec_weights = subpec_weights.clip(0, step)
    subpec_group, subpec_weights = subpec_groups_and_weights[0]
    apply_vertex_group_weights(subpec_group, subpec_weights, old_weights=old_weights)
    for subpec_group, subpec_weights in subpec_groups_and_weights[1:]:
        apply_vertex_group_weights(subpec_group, subpec_weights)

    vertex_groups.append(old_group)
    return vertex_groups

def intersect_two_weight_groups(mesh, group1, group2, new_group, method="L0"):
    select_object(mesh)
    prev_mode = mesh.mode
    bpy.ops.object.mode_set(mode='EDIT')
    g1 = mesh.vertex_groups[group1]
    g2 = mesh.vertex_groups[group2]
    gn = mesh.vertex_groups.get(new_group)
    if gn is None:
        gn = mesh.vertex_groups.new(name=new_group)

    weights1 = get_weights_as_array(mesh, g1)
    weights2 = get_weights_as_array(mesh, g2)
    bpy.ops.object.mode_set(mode='OBJECT')
    if method=="GEOM":
        ##### Here is a method based on geometric mean
        intersection = np.sqrt(weights1*weights2)
        # from the AM-GM inequality (https://en.wikipedia.org/wiki/AM%E2%80%93GM_inequality) follows
        # np.sqrt(weights1*weights2) <= (weights1+weights2)/2 <= weights1+weights2
        # therefore
        # 2*intersection <= (weights1 + weights2)
        # 0 <= (weights1-intersection) + (weights2-intersection)
        frac = intersection/(weights1+weights2)
        new_weights1 = weights1-intersection
        new_weights2 = weights2-intersection
        intersection *= 2
    elif method=="MIN":
        ##### Here is a method based on lattice properties of min/max and intersection/union
        intersection = np.minimum(weights1, weights2)
        # min(weights1, weights2) <= weights1 and min(weights1, weights2) <= weights2
        # therefore
        # 2*min(weights1, weights2) <= weights1 + weights2
        # 0 <= (weights1-intersection) + (weights2-intersection)
        new_weights2 = weights2-intersection
        new_weights1 = weights1-intersection
        intersection *= 2
        print("min1=", np.min(new_weights1))
        print("min2=", np.min(new_weights2))
    elif method == "L0":
        ### This method is based on l0 norm
        a = weights2 - weights1
        # note that for all a the following two equations hold:
        # max(a,0) - max(-a,0) == a
        # max(a,0) + max(-a,0) == abs(a)
        # therefore
        # intersection == weights2+weights1 -  new_weights2 - new_weights1
        # intersection == weights2+weights1 -  max(weights2-weights1,0) - max(weights1-weights2,0)
        # intersection == weights2+weights1 - (max(weights2-weights1,0) + max(weights1-weights2,0))
        # intersection == weights2+weights1 - abs(weights2-weights1)
        intersection = weights2 + weights1 - np.absolute(a)
        new_weights2 = np.maximum(a, 0)  # new_weights2 = max(weights2-weights1,0)
        new_weights1 = np.maximum(-a, 0)  # new_weights1 = max(weights1-weights2,0)
    else:
        return
    apply_vertex_group_weights(gn, intersection)
    apply_vertex_group_weights(g1, new_weights1, old_weights=weights1)
    apply_vertex_group_weights(g2, new_weights2, old_weights=weights2)
    bpy.ops.object.mode_set(mode=prev_mode)
    return g1, g2, gn


def get_eyebrows_and_eyelashes_path():
    n = 'eyebrows_and_eyelashes.png'
    p = bpy.path.abspath('//'+n)
    if os.path.exists(p):
        return p
    p = bpy.path.abspath('//../'+n)
    if os.path.exists(p):
        return p

    from bpy.utils import resource_path
    from pathlib import Path
    USER = Path(resource_path('USER'))
    ADDON = "DazOptim"
    srcPath = USER / "scripts/addons" / ADDON / "assets" / n
    return str(srcPath)


def get_rig_of(obj):
    for m in obj.modifiers:
        if isinstance(m, bpy.types.ArmatureModifier):
            return m.object


def translate(obj, t):
    apply_recursive(obj, location=t)

def apply_recursive(obj, location=None, rotation=None, scale=None):
    select_object(obj)
    if location is not None:
        obj.location = location
    if rotation is not None:
        obj.rotation = rotation
    if scale is not None:
        obj.scale = scale
    stack = [obj]
    visited = {obj}
    while len(stack) > 0:
        bpy.ops.object.select_all(action='DESELECT')
        obj = stack.pop()
        try:
            select_object(obj)
        except RuntimeError:
            continue
        bpy.ops.object.transform_apply(location=location is not None, rotation=rotation is not None, scale=scale is not None)
        for child in obj.children:
            if child not in visited:
                visited.add(child)
                stack.append(child)

def scale_in_edit_mode(obj, z):
    stack = [obj]
    visited = {obj}
    while len(stack)>0:
        obj = stack.pop()
        select_object(obj)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.transform.resize(value=z)
        for child in obj.children:
            if child not in visited:
                visited.add(child)
                stack.append(child)


def scale(obj, z):
    apply_recursive(obj, scale=z)

def bind_surf_deform(body_obj, clothes_obj, name=None):
    if name is None:
        name = 'TransferShapeKeys'
    if clothes_obj.data.shape_keys is not None:
        for sk in clothes_obj.data.shape_keys.key_blocks:
            sk.value = 0
    if name in clothes_obj.modifiers:
        clothes_obj.modifiers.remove(clothes_obj.modifiers[name])
    i = len(clothes_obj.modifiers)
    m = clothes_obj.modifiers.new(name, 'SURFACE_DEFORM')
    clothes_obj.modifiers.move(i, 0)
    m.target = body_obj
    with bpy.context.temp_override(object=clothes_obj):
        bpy.ops.object.surfacedeform_bind(modifier=m.name)
    return m


def get_daz_content_dirs():
    settingsDir = bpy.context.preferences.addons['bl_ext.user_default.import_daz'].preferences.settingsDir
    settings = os.path.join(settingsDir, 'import_daz_settings.json')
    settings = io.open(settings, 'r', encoding='utf-8-sig')
    settings = json.load(settings)
    settings = settings['daz-settings']
    content_dirs = settings["contentDirs"]
    return content_dirs

def obj_daz_dir(obj):
    daz_url: str = obj.daz_importer.DazUrl
    daz_path = daz_url.rsplit('#', maxsplit=1)[0]
    daz_dir = os.path.dirname(daz_path)
    return daz_dir

def bind_to_objects(body_obj, clothes_objs=None, name=None):
    return [(bind_surf_deform(body_obj, o, name=name), o) for o in clothes_objs]

def transfer_all_shape_keys(body_obj, clothes_objs):
    for sk in body_obj.data.shape_keys.key_blocks:
        transfer_shape_key(body_obj, sk, clothes_objs)

def reset_shape_keys(obj):
    if obj.data.shape_keys is not None:
        for sk in obj.data.shape_keys.key_blocks:
            sk.value = 0

def transfer_shape_key(body_obj, sk, clothes_objs=None):
    reset_shape_keys(body_obj)
    if isinstance(sk, str):
        sk = body_obj.data.shape_keys.key_blocks[sk]
    ms = bind_to_objects(body_obj, clothes_objs, sk.name)
    sk.value = 1
    for m, o in ms:
        if o.data.shape_keys is not None:
            dest_sks = o.data.shape_keys.key_blocks
            if sk.name in dest_sks:
                old_sk = dest_sks[sk.name]
                o.shape_key_remove(old_sk)
        select_object(o)
        bpy.ops.object.modifier_apply_as_shapekey(modifier=m.name)
        add_driver(body_obj, sk.name, o)
    sk.value = 0


def serialize_nodes(ng):
    if isinstance(ng, str):
        ng = bpy.data.node_groups[ng]
    nodes = ng.nodes
    print()
    print()
    print("ng = bpy.data.node_groups.new('" + nodes.data.name + "', '" + nodes.data.__class__.__name__ + "')")

    def escape(s: str) -> str:
        return s.replace(".", "_").replace(" ", "_").lower()

    for soc in ng.interface.items_tree:
        print("ng.interface.new_socket(name='"+soc.name+"', in_out='"+soc.in_out+"', socket_type='"+soc.socket_type+"')")
    for node in nodes:
        nn = escape(node.name)
        print(nn + " = ng.nodes.new('" + node.__class__.__name__ + "')")
        print(nn + ".name = '" + node.name+ "'")
        print(nn + ".location = "+str(tuple(node.location)))

        for attr in ['data_type', 'operation', 'transform_space']:
            if hasattr(node, attr):
                print(escape(node.name) + "." + attr + " = " + repr(getattr(node, attr)))
    def soc_id(soc):
        if isinstance(soc.node, (bpy.types.NodeGroupOutput, bpy.types.NodeGroupInput)):
            return soc.name
        else:
            return soc.identifier
    for node in nodes:
        for inp in node.inputs:
            if not inp.is_unavailable:
                dst = escape(node.name) + ".inputs['" + soc_id(inp) + "']"
                if len(inp.links) > 0:
                    for lnk in inp.links:
                        src = escape(lnk.from_node.name) + ".outputs['" + soc_id(lnk.from_socket) + "']"
                        print("ng.links.new(" + dst + ", " + src + ")")
                elif hasattr(inp, 'default_value'):
                    v = inp.default_value
                    if isinstance(v, (int, float, bool, str)):
                        val = str(v)
                    else:
                        try:
                            val = str(tuple(v))
                        except TypeError:
                            val = repr(v)
                    print(dst + ".default_value = " + val)


def add_driver(body_obj, obj_sk, o):
    if body_obj == o:
        return
    if isinstance(obj_sk, str):
        obj_sk = o.data.shape_keys.key_blocks[obj_sk]
    obj_sk.driver_remove('value')
    driver = obj_sk.driver_add('value').driver
    driver.type = "SCRIPTED"
    driver.expression = "skw_var"
    driver_var = driver.variables.new()
    driver_var.name = "skw_var"
    driver_target = driver_var.targets[0]
    driver_target.id_type = 'OBJECT'
    driver_target.id = body_obj
    driver_target.data_path = 'data.shape_keys.key_blocks["' + obj_sk.name + '"].value'


def add_drivers_for_all_shape_keys(body_obj, objs=None):
    if objs is None:
        objs = bpy.context.selected_objects
    if isinstance(objs, str):
        objs = bpy.data.objects[objs]
    if isinstance(objs, bpy.types.Object):
        objs = [objs]
    for obj in objs:
        for sk in obj.data.shape_keys.key_blocks:
            add_driver(body_obj, sk, obj)


def remove_shape_key(body, sk_name):
    if sk_name in body.data.shape_keys.key_blocks:
        body.shape_key_remove(body.data.shape_keys.key_blocks[sk_name])

def do_edges_share_a_face(edge1, edge2):
    for linked_face in edge1.link_faces:
        if linked_face in edge2.link_faces:
            return True
    return False

def do_edges_share_a_vert(edge1, edge2):
    for v in edge1.verts:
        if v in edge2.verts:
            return True
    return False

def get_edges_that_share_a_face(edge, vert):
    for linked_edge in vert.link_edges:
        if do_edges_share_a_face(linked_edge, edge):
            yield linked_edge

def iterate_parallel_edges(edge):
    for face in edge.link_faces:
        parallel_edge = None
        perpendicular_egde = None
        for other_edge in face.edges:
            if do_edges_share_a_vert(edge, other_edge):
                perpendicular_egde = other_edge
                if parallel_edge is not None:
                    yield perpendicular_egde.calc_length(), parallel_edge
                    break
            else:
                parallel_edge = other_edge
                if perpendicular_egde is not None:
                    yield perpendicular_egde.calc_length(), parallel_edge
                    break


def iterate_edge_loop(start_edge, start_vert=None):
    current_edge = start_edge
    current_vert = current_edge.verts[0] if start_vert is None else start_vert
    while True:
        if len(current_vert.link_edges) != 4:
            return
        for next_edge in current_vert.link_edges:
            if not do_edges_share_a_face(current_edge, next_edge):
                yield next_edge
                current_edge = next_edge
                current_vert = current_edge.verts[1] if current_edge.verts[0] == current_vert else current_edge.verts[0]
                break
        if current_edge == start_edge:
            return

def iterate_edge_loop_over_allowed_verts(start_edge, allowed_verts: np.ndarray):
    current_edge = start_edge
    current_vert = current_edge.verts[0]
    while allowed_verts[current_vert.index]:
        if len(current_vert.link_edges) != 4:
            return
        for next_edge in current_vert.link_edges:
            if not do_edges_share_a_face(current_edge, next_edge):
                yield next_edge
                current_edge = next_edge
                current_vert = current_edge.verts[1] if current_edge.verts[0] == current_vert else current_edge.verts[0]
                break
        if current_edge == start_edge:
            return

def get_last_vert_of_edge_loop(edge_loop, start_vert=None):
    current_edge = edge_loop[-1]
    if len(edge_loop)==1:
        current_vert = current_edge.verts[0] if start_vert is None else start_vert
        return current_vert
    else:
        prev_edge = edge_loop[-2]
        return current_edge.verts[1] if current_edge.verts[0] in prev_edge.verts else current_edge.verts[0]

def collect_edge_loop(start_edge):
    one_end = list(iterate_edge_loop(start_edge, start_edge.verts[0]))
    if len(one_end)>0 and one_end[-1] == start_edge:
        return one_end, True
    second_end = list(iterate_edge_loop(start_edge, start_edge.verts[1]))
    second_end.reverse()
    second_end.append(start_edge)
    second_end.extend(one_end)
    return second_end, False

def select_edge_loop(edge_loop):
    for e in edge_loop:
        e.select_set(True)

def edge_loop_length(edge_loop):
    return sum(e.calc_length() for e in edge_loop)


def edge_vector(l):
    a, b = l.edge.verts
    return b.co - a.co

def rev_edge_vector(l):
    a, b = l.edge.verts
    return a.co - b.co


def edge_angle(l):
    a, b = l.edge.verts
    return b.co - a.co

def angle_between_3_verts(a,b,c):
    ab = b.co - a.co
    bc = c.co - b.co
    return ab.angle(bc), ab.length+bc.length, b

def angle_between_connected_edges(edge1, edge2):
    a, b = edge1.verts
    c, d = edge2.verts
    if a == c:
        return angle_between_3_verts(b, a, d)
    elif a == d:
        return angle_between_3_verts(b, a, c)
    elif b == c:
        return angle_between_3_verts(a, b, d)
    elif b == d:
        return angle_between_3_verts(a, b, c)
    else:
        return np.inf, 0, None

def edge_loop_find_sharp_points(edge_loop):
    prev_edge = edge_loop[-1]
    if len(edge_loop)==1:
        yield np.inf, 1, prev_edge.verts[0]
        yield np.inf, 1, prev_edge.verts[1]
    else:
        next_edge = edge_loop[0]
        angle, length, mid_vertex = angle_between_connected_edges(prev_edge, next_edge)
        if mid_vertex is None:
            next_next_edge = edge_loop[1]
            if next_edge.verts[0] in next_next_edge.verts:
                yield np.inf, 1, next_edge.verts[1]
            else:
                yield np.inf, 1, next_edge.verts[0]
            prev_prev_edge = edge_loop[-2]
            if prev_edge.verts[0] in prev_prev_edge.verts:
                yield np.inf, 1, prev_edge.verts[1]
            else:
                yield np.inf, 1, prev_edge.verts[0]
        else:
            yield angle, length, mid_vertex
        prev_edge = next_edge
        for next_edge in edge_loop[1:]:
            angle, length, mid_vertex = angle_between_connected_edges(prev_edge, next_edge)
            yield angle, length, mid_vertex
            prev_edge = next_edge

def edge_loop_find_sharpest_points(edge_loop, num_of_points=2):
    l = [(angle/length, mid_vertex) for angle, length, mid_vertex in edge_loop_find_sharp_points(edge_loop)]
    l.sort(key=lambda x:x[0], reverse=True)
    return l[:num_of_points]


def edge_loop_find_sharp_ends(edge_loop):
    total_len = edge_loop_length(edge_loop)
    l = [(angle / length, mid_vertex) for angle, length, mid_vertex in edge_loop_find_sharp_points(edge_loop)]
    l.sort(key=lambda x: x[0], reverse=True)
    _, first = l[0]
    _, second = l[1]
    _, third = l[2]
    if (first.co-second.co).length < total_len/10:
        return first, third
    else:
        return first, second

class DazOptimizer:

    def __init__(self, workdir=None, name=None):
        if workdir is None:
            workdir = os.path.dirname(bpy.data.filepath)
        if name is None:
            name = os.path.basename(bpy.data.filepath)
            name = name[:name.rindex('.')]
        self.name = name
        self.workdir = workdir
        self.body_mesh = None
        self.body_rig = None

    def get_fav_morphs_path(self):
        return os.path.join(self.workdir, self.name+"_fav_morphs.json")

    def gold_palace_dir(self):
        return os.path.join(self.workdir, "textures/original/meipex/m_goldenpalace/g9")

    def us_mask_path(self):
        return os.path.join(self.workdir, "uv_region_mask.png")

    def textures_dir(self):
        return os.path.join(self.workdir, "textures")

    def genesis_dir(self):
        return os.path.join(self.workdir, "textures/original/daz/characters/genesis9")

    def get_uv_mask(self):
        from PIL import Image
        p = self.us_mask_path()
        print("Reading UV mask ", p)
        uv_region_mask = np.array(Image.open(p), dtype=np.uint32)
        uv_region_mask = (uv_region_mask[:, :, 0] * 256 + uv_region_mask[:, :, 1]) * 256 + uv_region_mask[:, :, 2]
        return uv_region_mask

    def find_body(self):
        self.body_rig = find_body_rig()
        self.body_mesh = bpy.data.objects[self.body_rig.name + ' Mesh']

    def get_body_mesh(self):
        if self.body_mesh is None:
            self.find_body()
        return self.body_mesh

    def get_mouth_mesh(self):
        for b in bpy.data.objects:
            if b.daz_importer.DazMesh in ['Toon9-mouth', 'Mouth9']:
                return b

    def get_toon_mouth_mesh(self):
        for b in bpy.data.objects:
            if b.daz_importer.DazMesh == 'Toon9-mouth':
                return b

    def get_body_rig(self):
        if self.body_rig is None:
            self.find_body()
        return self.body_rig

    @staticmethod
    def get_daz_mesh(daz_mesh):
        for b in bpy.data.objects:
            if b.daz_importer.DazMesh == daz_mesh:
                return b

    def get_eyes_mesh(self):
        return DazOptimizer.get_daz_mesh('Eyes9')

    @staticmethod
    def get_toon_eye_socket_mesh():
        return DazOptimizer.get_daz_mesh('Toon9-eye-socket')

    @staticmethod
    def get_toon_floating_iris_mesh():
        return DazOptimizer.get_daz_mesh('Toon9-floating-iris')

    @staticmethod
    def get_mesh_by_name(suffix):
        for b in bpy.data.objects:
            if b.name.endswith(suffix):
                return b

    def get_eyelashes_mesh(self):
        return DazOptimizer.get_mesh_by_name('Eyelashes Mesh')

    def get_base_uv_layer(self, layer_name='Base Multi UDIM'):
        return self.get_body_mesh().data.uv_layers[layer_name]

    def get_base_uv_layer_np(self, layer_name='Base Multi UDIM'):
        return np.array([v.uv for v in self.get_base_uv_layer(layer_name=layer_name).data])

    def get_pixel_coords(self, layer_name='Base Multi UDIM'):
        return DazOptimizer.base_layer_to_pixel_coords(self.get_base_uv_layer_np(layer_name))

    def get_base_uv_layer_selection_np(self):
        return np.array([v.select for v in self.get_base_uv_layer().data], dtype=bool)

    def update_base_uv_layer(self, base_layer_np: np.ndarray):
        for v, new_uv in zip(self.get_base_uv_layer().data, base_layer_np):
            v.uv = new_uv

    def get_concat_image_path(self, map_type):
        return os.path.join(self.workdir, self.name + '_' + map_type + '.png')

    def get_simplified_eyes_image_path(self, map_type):
        return os.path.join(self.workdir, self.name + '_' + map_type + '_eyes.png')

    def get_eyebrows(self):
        for o in bpy.data.objects:
            if 'eyebrows' in o.name.lower() and o.name.endswith(" Mesh"):
                return o
        if bpy.context.scene.get('daz_optim_toon'):
            for o in bpy.data.objects:
                if 'toon brows' in o.name.lower() and o.name.endswith(" Mesh"):
                    return o

    def remove_old_eyebrows(self):
        for o in bpy.data.objects:
            if 'eyebrows' in o.name.lower() and o.name.endswith(" Mesh") and not o.name == 'Eyebrows Mesh':
                bpy.data.objects.remove(o)
                break

    def optimize_eyebrows(self):
        EYEBROWS_M = self.get_eyebrows()
        if bpy.context.scene.get('daz_optim_toon'):
            mats = [m for m in EYEBROWS_M.data.materials if not NodesUtils.contains_subgroup(m, "DAZ Transparent")]
            DazOptimizer.gen_simple_materials(mats)
        else:
            offset = 1.6009911569682034
            if EYEBROWS_M is not None:
                offset = 0
                sample_points = 10
                for i in np.random.randint(0, len(EYEBROWS_M.data.vertices), sample_points):
                    offset += EYEBROWS_M.data.vertices[i].co.z
                offset /= sample_points

            vertices = np.array([[0.020655272528529167, -0.09718257188796997, 0.0037765231999484783], [0.010495096445083618, -0.09829649329185486, 0.001051875677975822], [0.0246686190366745, -0.09323396533727646, -0.004316237839785408], [0.013008052483201027, -0.0945364311337471, -0.008372691544619393], [0.028579287230968475, -0.0950171947479248, 0.006468268958005119], [0.031095163896679878, -0.09265439212322235, -0.002452758225527596], [0.038159459829330444, -0.09091758728027344, 0.006650659171017814], [0.03818078339099884, -0.09061393141746521, -0.0010096105662258381], [0.04382877051830292, -0.08715396374464035, 0.006490084257992912], [0.043936073780059814, -0.08735480159521103, -0.0009910139170559162], [0.049159739166498184, -0.08238893747329712, 0.00556966933337133], [0.04852811247110367, -0.08362218737602234, -0.0018033060160549397], [0.05282459035515785, -0.07811952382326126, 0.0038766590031711345], [0.05235077813267708, -0.07928386330604553, -0.0033418211069973225], [0.055806536227464676, -0.07358748465776443, 0.0018968311223117595], [0.055413272231817245, -0.07450003176927567, -0.0054763826456936116], [0.058321163058280945, -0.06856732070446014, -0.0005002292719753498], [0.05764066427946091, -0.070110023021698, -0.007665303620425057], [0.061125967651605606, -0.06174656003713608, -0.003270057114687752], [0.06067896634340286, -0.06334099173545837, -0.010448244484988045], [0.01933024451136589, -0.09739914536476135, 0.007336351004513908], [0.010570534504950047, -0.09852366894483566, 0.004864307967099357], [-0.0203605554997921, -0.09722965955734253, 0.0038549629124728924], [-0.010192444548010826, -0.09832023829221725, 0.0010914531621066814], [-0.024334117770195007, -0.09329022467136383, -0.004222420128908944], [-0.012660950422286987, -0.09456589818000793, -0.008323696526614022], [-0.028289951384067535, -0.09508249908685684, 0.006577107039364982], [-0.030766494572162628, -0.09272542595863342, -0.00233438340100367], [-0.03786155581474304, -0.09100489318370819, 0.006796094504269767], [-0.037853024899959564, -0.09070125222206116, -0.000864175232973885], [-0.04352172836661339, -0.08725428581237793, 0.006657334891232658], [-0.04360099509358406, -0.08745533227920532, -0.0008235248652370686], [-0.04883836582303047, -0.08250148594379425, 0.0057573047551242595], [-0.04818148538470268, -0.08373325318098068, -0.0016181739893825764], [-0.05248706415295601, -0.07824047654867172, 0.004078361121091056], [-0.05198841169476509, -0.07940369844436646, -0.003142026337710213], [-0.055451150983572006, -0.07371526211500168, 0.0021098581227390056], [-0.05503189191222191, -0.0746268779039383, -0.005264905366030526], [-0.057945217937231064, -0.0687008649110794, -0.00027766552838404124], [-0.057240959256887436, -0.07024197280406952, -0.0074453624812038655], [-0.060723926872015, -0.0618865080177784, -0.003036883744326424], [-0.06025322154164314, -0.06347988545894623, -0.010216620835390877], [-0.019049562513828278, -0.09744320064783096, 0.0074096647175876384], [-0.01028292253613472, -0.09854759275913239, 0.004904123869809318]])
            vertex_normals = np.array([(0.21302460134029388, -0.965381383895874, -0.1505301296710968), (0.17133180797100067, -0.9650542140007019, -0.198281928896904), (0.23409877717494965, -0.9296879172325134, -0.28439071774482727), (0.20445850491523743, -0.9217665791511536, -0.3294588029384613), (0.2950233221054077, -0.9428872466087341, -0.15467630326747894), (0.30824264883995056, -0.9361860156059265, -0.16894443333148956), (0.43910646438598633, -0.8970659375190735, -0.04958169907331467), (0.4363127052783966, -0.8983418941497803, -0.05111850053071976), (0.5882634520530701, -0.8083396553993225, 0.02308816649019718), (0.5938681960105896, -0.8041653037071228, 0.025272265076637268), (0.7094771265983582, -0.703013002872467, 0.04914076626300812), (0.711438000202179, -0.7010304927825928, 0.04911404475569725), (0.8009960651397705, -0.5974375009536743, 0.03838849067687988), (0.8086886405944824, -0.5871158242225647, 0.036299120634794235), (0.8664449453353882, -0.49890807271003723, 0.019074566662311554), (0.8745542168617249, -0.48465055227279663, 0.01639372669160366), (0.9068731665611267, -0.42126020789146423, 0.010997472330927849), (0.9094046354293823, -0.4157572090625763, 0.011365870013833046), (0.9205265045166016, -0.3904625177383423, 0.01304092351347208), (0.9205264449119568, -0.3904625177383423, 0.013040922582149506), (0.1784271001815796, -0.9839159846305847, -0.008548013865947723), (0.1276978999376297, -0.9910843968391418, -0.03801281377673149), (-0.21146777272224426, -0.9655916690826416, -0.15137451887130737), (-0.17016936838626862, -0.9649455547332764, -0.19980597496032715), (-0.23305915296077728, -0.9295825362205505, -0.2855866551399231), (-0.20367898046970367, -0.9216205477714539, -0.3303488492965698), (-0.2939514219760895, -0.9430721998214722, -0.1555873155593872), (-0.30682215094566345, -0.9363527894020081, -0.17059792578220367), (-0.43836885690689087, -0.8973409533500671, -0.05110874027013779), (-0.4348205029964447, -0.8989534974098206, -0.05304456129670143), (-0.5879247784614563, -0.8086429834365845, 0.02100095897912979), (-0.593061625957489, -0.8048291206359863, 0.022978920489549637), (-0.7096368670463562, -0.7030275464057922, 0.046558331698179245), (-0.71131432056427, -0.7013322710990906, 0.046529170125722885), (-0.8014999628067017, -0.5969457626342773, 0.035406265407800674), (-0.8090589642524719, -0.5867817997932434, 0.03332577645778656), (-0.8670861124992371, -0.49790769815444946, 0.015799948945641518), (-0.8751322627067566, -0.4837063252925873, 0.013107089325785637), (-0.9075274467468262, -0.41992461681365967, 0.007562259677797556), (-0.9100339412689209, -0.41445812582969666, 0.007917601615190506), (-0.9211719632148743, -0.3890385329723358, 0.009553579613566399), (-0.9211719036102295, -0.3890385329723358, 0.009553579613566399), (-0.1766616404056549, -0.984230101108551, -0.009047266095876694), (-0.1256365329027176, -0.9913285374641418, -0.03851176053285599)])
            uvs = [[(0.10387720167636871, 0.15240783989429474), (0.20352177321910858, 0.002554043661803007), (0.4944729804992676, 0.05704062059521675), (0.35311999917030334, 0.2287999987602234)], [(0.10387720167636871, 0.15240783989429474), (0.35311999917030334, 0.2287999987602234), (0.2858409285545349, 0.32360079884529114), (0.019021285697817802, 0.27419033646583557)], [(0.019021285697817802, 0.27419033646583557), (0.2858409285545349, 0.32360079884529114), (0.23823584616184235, 0.4334968328475952), (0.010124947875738144, 0.4299057424068451)], [(0.010124947875738144, 0.4299057424068451), (0.23823584616184235, 0.4334968328475952), (0.23564350605010986, 0.5323020219802856), (0.012801339849829674, 0.5315198302268982)], [(0.012801339849829674, 0.5315198302268982), (0.23564350605010986, 0.5323020219802856), (0.2573564350605011, 0.6207517385482788), (0.03667948767542839, 0.6383661031723022)], [(0.03667948767542839, 0.6383661031723022), (0.2573564350605011, 0.6207517385482788), (0.299108624458313, 0.7074313759803772), (0.0829995721578598, 0.7227829694747925)], [(0.0829995721578598, 0.7227829694747925), (0.299108624458313, 0.7074313759803772), (0.357651025056839, 0.7929096221923828), (0.1371798813343048, 0.8043929934501648)], [(0.1371798813343048, 0.8043929934501648), (0.357651025056839, 0.7929096221923828), (0.41832488775253296, 0.8671479225158691), (0.20339392125606537, 0.8890230059623718)], [(0.20339392125606537, 0.8890230059623718), (0.41832488775253296, 0.8671479225158691), (0.4944729804992676, 0.978790283203125), (0.27912330627441406, 1.0)], [(0.20352177321910858, 0.002554043661803007), (0.10387720167636871, 0.15240783989429474), (1.2504191460038783e-08, 0.13043661415576935), (0.08955555409193039, 2.6969557254119536e-08)], [(0.10387720167636871, 0.15240783989429474), (0.019021285697817802, 0.27419033646583557), (1.2504191460038783e-08, 0.13043661415576935)], [(0.598351240158081, 0.8475814461708069), (0.8475865721702576, 0.7711809277534485), (0.98896723985672, 0.9429332613945007), (0.6980223059654236, 0.997435450553894)], [(0.598351240158081, 0.8475814461708069), (0.5134828090667725, 0.7257977724075317), (0.7803006172180176, 0.6763840317726135), (0.8475865721702576, 0.7711809277534485)], [(0.5134828090667725, 0.7257977724075317), (0.5045840740203857, 0.5700806975364685), (0.7326943874359131, 0.5664908289909363), (0.7803006172180176, 0.6763840317726135)], [(0.5045840740203857, 0.5700806975364685), (0.5072675943374634, 0.4684655964374542), (0.7301082611083984, 0.4676879048347473), (0.7326943874359131, 0.5664908289909363)], [(0.5072675943374634, 0.4684655964374542), (0.531157374382019, 0.361620157957077), (0.7518305778503418, 0.37924033403396606), (0.7301082611083984, 0.4676879048347473)], [(0.531157374382019, 0.361620157957077), (0.5774877667427063, 0.2772054672241211), (0.7935928702354431, 0.29256266355514526), (0.7518305778503418, 0.37924033403396606)], [(0.5774877667427063, 0.2772054672241211), (0.631676197052002, 0.19559861719608307), (0.8521435260772705, 0.20708619058132172), (0.7935928702354431, 0.29256266355514526)], [(0.631676197052002, 0.19559861719608307), (0.6978949308395386, 0.11097240447998047), (0.9128215909004211, 0.13284821808338165), (0.8521435260772705, 0.20708619058132172)], [(0.6978949308395386, 0.11097240447998047), (0.7736213803291321, -1.8817928548742202e-08), (0.9889672994613647, 0.021205546334385872), (0.9128215909004211, 0.13284821808338165)], [(0.6980223059654236, 0.997435450553894), (0.5840516686439514, 0.9999967813491821), (0.4944729506969452, 0.8695566654205322), (0.598351240158081, 0.8475814461708069)], [(0.598351240158081, 0.8475814461708069), (0.4944729506969452, 0.8695566654205322), (0.5134828090667725, 0.7257977724075317)]]
            loops = np.array([(0, 0, 1), (1, 1, 2), (3, 2, 3), (2, 3, 0), (0, 4, 0), (2, 5, 5), (5, 6, 4), (4, 7, 6), (4, 8, 4), (5, 9, 8), (7, 10, 7), (6, 11, 9), (6, 12, 7), (7, 13, 11), (9, 14, 10), (8, 15, 12), (8, 16, 10), (9, 17, 14), (11, 18, 13), (10, 19, 15), (10, 20, 13), (11, 21, 17), (13, 22, 16), (12, 23, 18), (12, 24, 16), (13, 25, 20), (15, 26, 19), (14, 27, 21), (14, 28, 19), (15, 29, 23), (17, 30, 22), (16, 31, 24), (16, 32, 22), (17, 33, 26), (19, 34, 25), (18, 35, 27), (1, 36, 1), (0, 37, 29), (20, 38, 28), (21, 39, 30), (0, 40, 6), (4, 41, 31), (20, 42, 29), (22, 43, 32), (24, 44, 35), (25, 45, 34), (23, 46, 33), (22, 47, 38), (26, 48, 36), (27, 49, 37), (24, 50, 32), (26, 51, 41), (28, 52, 39), (29, 53, 40), (27, 54, 36), (28, 55, 44), (30, 56, 42), (31, 57, 43), (29, 58, 39), (30, 59, 47), (32, 60, 45), (33, 61, 46), (31, 62, 42), (32, 63, 50), (34, 64, 48), (35, 65, 49), (33, 66, 45), (34, 67, 53), (36, 68, 51), (37, 69, 52), (35, 70, 48), (36, 71, 56), (38, 72, 54), (39, 73, 55), (37, 74, 51), (38, 75, 59), (40, 76, 57), (41, 77, 58), (39, 78, 54), (23, 79, 62), (43, 80, 60), (42, 81, 61), (22, 82, 33), (22, 83, 61), (42, 84, 63), (26, 85, 38)], dtype=np.int32)
            polygons = np.array([(0, 4), (4, 4), (8, 4), (12, 4), (16, 4), (20, 4), (24, 4), (28, 4), (32, 4), (36, 4), (40, 3), (43, 4), (47, 4), (51, 4), (55, 4), (59, 4), (63, 4), (67, 4), (71, 4), (75, 4), (79, 4), (83, 3)], dtype=np.int32)
            polygon_normals = np.array([(0.204458549618721, -0.9217666387557983, -0.329458624124527), (0.2573009133338928, -0.9340332746505737, -0.2477456033229828), (0.35233110189437866, -0.9309596419334412, -0.09579653292894363), (0.5236645340919495, -0.851923406124115, -0.0013364654732868075), (0.6546372175216675, -0.7543064951896667, 0.0497170016169548), (0.7625151872634888, -0.6451690793037415, 0.04824307933449745), (0.8438367247581482, -0.5359628200531006, 0.02614249475300312), (0.894795835018158, -0.446378618478775, 0.009297684766352177), (0.9205259084701538, -0.3904636800289154, 0.013040537014603615), (0.12769724428653717, -0.9910845160484314, -0.03801056370139122), (0.25233086943626404, -0.9670045971870422, 0.03509100154042244), (-0.20367690920829773, -0.9216209053993225, -0.3303491473197937), (-0.25626254081726074, -0.9340181350708008, -0.24887651205062866), (-0.3509964346885681, -0.9313123822212219, -0.09725535660982132), (-0.5227794647216797, -0.8524614572525024, -0.003333872416988015), (-0.6543564796447754, -0.7547026872634888, 0.04734565317630768), (-0.7627802491188049, -0.6450580358505249, 0.045457348227500916), (-0.8443940281867981, -0.5352294445037842, 0.022982647642493248), (-0.8954243659973145, -0.4451744854450226, 0.0059044249355793), (-0.9211748838424683, -0.38903138041496277, 0.009553337469696999), (-0.12563464045524597, -0.9913288950920105, -0.038512177765369415), (-0.2502712309360504, -0.9675723910331726, 0.0341765321791172)])
            mesh = bpy.data.meshes.new(name='Eyebrows Mesh')
            vertices[:,2] += offset
            # add the amount of vertices, in this case 4.
            mesh.vertices.add(len(vertices))

            # use the vertices numpy array
            mesh.vertices.foreach_set("co", vertices.reshape(-1))
            mesh.vertices.foreach_set("normal", vertex_normals.reshape(-1))

            # total indexes in vertex_index

            # add the amount of the vertex_index array, in this case 12
            mesh.loops.add(len(loops))

            # set the vertx_index
            mesh.loops.foreach_set("vertex_index", loops[:, 0])
            mesh.loops.foreach_set("index", loops[:, 1])
            mesh.loops.foreach_set("edge_index", loops[:, 2])

            # add the length of loop_start array
            mesh.polygons.add(len(polygons))

            # generate the polygons
            mesh.polygons.foreach_set("loop_start", polygons[:,0])
            mesh.polygons.foreach_set("loop_total", polygons[:,1])
            mesh.polygons.foreach_set("normal", polygon_normals.reshape(-1))

            mesh.update()
            mesh.validate()

            # create the object with the mesh just created
            obj = bpy.data.objects.new('Eyebrows Mesh', mesh)
            RIG = self.get_body_rig()
            BODY = self.get_body_mesh()
            obj.parent = RIG
            BODY.users_collection[0].objects.link(obj)
            # obj.select_set(True)
            # bpy.context.view_layer.objects.active = obj
            # bpy.ops.object.mode_set(mode='EDIT')
            mesh.uv_layers.new(name='UVMap')
            bm = bmesh.new()
            bm.from_mesh(mesh)
            bm.faces.ensure_lookup_table()
            uv_layer = bm.loops.layers.uv.verify()
            for bm_face, uv_face in zip(bm.faces, uvs):
                for bm_loop, (u,v) in zip(bm_face.loops, uv_face):
                    bm_loop[uv_layer].uv = (u/2,v)
            bm.to_mesh(mesh)
            #bmesh.update_edit_mesh(mesh)
            # bpy.ops.object.mode_set(mode='OBJECT')

            mat = bpy.data.materials.new('Eyebrows')
            mat.use_nodes = True
            obj.data.materials.append(mat)
            obj.active_material_index = len(obj.data.materials) - 1
            n = mat.node_tree.nodes
            l = mat.node_tree.links
            target_texture = n.new('ShaderNodeTexImage')
            img = bpy.data.images.load(get_eyebrows_and_eyelashes_path())
            target_texture.image = img
            target_texture.name = 'Eyebrows Texture'
            target_texture.location = (0, -300)
            bsdf = n['Principled BSDF']
            #l.new(bsdf.inputs['Base Color'], target_texture.outputs['Color'])
            bsdf.inputs['Base Color'].default_value = (0.0, 0.0, 0.0, 1.0)
            l.new(bsdf.inputs['Alpha'], target_texture.outputs['Color'])

            m = obj.modifiers.new(name='FitEyebrows', type="SHRINKWRAP")
            m.target = BODY
            m.offset = 0.003
            m.wrap_mode = 'OUTSIDE'

            ma = obj.modifiers.new(name='Armature', type="ARMATURE")
            ma.object = RIG

    def apply_optimized_eyebrows(self):
        obj = bpy.data.objects['Eyebrows Mesh']
        select_object(obj)
        bpy.ops.object.modifier_apply(modifier='FitEyebrows')
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        BODY = self.get_body_mesh()
        for g in ['head', 'centerbrow', 'r_browouter', 'l_browouter', 'r_browinner', 'l_browinner']:
            transfer_weights_to_object(BODY, obj, g)

    def optimize_eyelashes(self):
        EYELASHES_M = self.get_eyelashes_mesh()
        select_object(EYELASHES_M)
        if bpy.context.scene.get('daz_optim_toon'):
            mats = [m for m in EYELASHES_M.data.materials if not NodesUtils.contains_subgroup(m, "DAZ Transparent")]
            DazOptimizer.gen_simple_materials(mats)
        else:
            # uv_layer = EYES_M.data.uv_layers.active
            # uvs = np.array([v.uv for v in uv_layer.data], dtype=bool)
            # uvs[:, y] < 0.5
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.context.scene.tool_settings.use_uv_select_sync = False
            bpy.ops.uv.select_all(action='DESELECT')
            bpy.ops.mesh.select_all(action='DESELECT')

            me = bpy.context.object.data
            bm = bmesh.from_edit_mesh(me)
            uv_layer = bm.loops.layers.uv.verify()
            mask = rle_decode(EYELASHES_RLE, MASK_SHAPE)
            for v in bm.verts:
                v.select = False
            for face in bm.faces:
                for loop in face.loops:
                    loop_uv = loop[uv_layer]
                    u, v = loop_uv.uv
                    up, vp = int(u*MASK_SHAPE[0]), int((1-v)*MASK_SHAPE[0])
                    if mask[vp, up]:
                        u, v = u-0.5, v-0.5 # change center of rotation
                        u, v = v, -u # rotate 90 degrees clockwise
                        u, v = u + 0.5 + 0.25, v + 0.5 # undo center of rotation and move .25 to the right
                        loop_uv.uv = u,v
                    else:
                        loop.vert.select = True
            for v in bm.verts:
                if v.select:
                    bm.verts.remove(v)
            # bm.select_flush(True)
            bmesh.update_edit_mesh(me)
            bpy.ops.object.mode_set(mode='OBJECT')

            eyelashes_img = bpy.data.images.load(get_eyebrows_and_eyelashes_path())
            for mat in EYELASHES_M.data.materials:
                bsdf = NodesUtils.find_by_type(mat.node_tree, bpy.types.ShaderNodeBsdfPrincipled)
                texture_nodes = set()
                for channel in ["Base Color", "Alpha"]:
                    NodesUtils.from_socket_backwards_search_for(bsdf.inputs[channel], bpy.types.ShaderNodeTexImage, texture_nodes)
                texture_node, = texture_nodes
                texture_node.image = eyelashes_img

    def merge_eyebrows_and_eyelashes(self):
        eyelashes = self.get_eyelashes_mesh()
        eyebrows = self.get_eyebrows()
        if eyebrows is None or eyelashes is None:
            return
        if bpy.context.scene.get('daz_optim_toon'):
            def find_material(mesh, name):
                if len(mesh.data.materials)>1:
                    for m in mesh.data.materials:
                        if name in m.name.lower():
                            return m
                else:
                    return mesh.data.materials[0]
            eyebrow_mat = find_material(eyebrows, 'brow')
            eyelashes_mat = find_material(eyelashes, 'eyelash')
            select_object(eyelashes)
            eyebrows.select_set(True)
            bpy.ops.object.join()
            if bpy.context.scene.get('is_nirv_zero'):
                eyelashes.data.materials.clear()
                eyelashes.data.materials.append(eyelashes_mat)
            else:
                if eyelashes_mat is not None:
                    if NodesUtils.contains_subgroup(eyelashes_mat, "DAZ Transparent"):
                        eyelashes_mat.name = TRANSPARENT_TOON_EYELASHES_MAT_NAME
                if eyebrow_mat is not None:
                    if NodesUtils.contains_subgroup(eyebrow_mat, "DAZ Transparent"):
                        eyebrow_mat.name = TRANSPARENT_TOON_EYEBROWS_MAT_NAME
        else:
            eyelashes.data.uv_layers.active.name = 'Eyelashes UVs'
            eyebrows.data.uv_layers.active.name = 'Eyebrows UVs'
            select_object(eyelashes)
            eyebrows.select_set(True)
            bpy.ops.object.join()

            eyelashes_uvs = eyelashes.data.uv_layers['Eyelashes UVs']
            eyebrows_uvs = eyelashes.data.uv_layers['Eyebrows UVs']
            eyebrows_uvs_np = np.array([v.uv for v in eyebrows_uvs.data])
            is_eyebrows = np.any(eyebrows_uvs_np > 0, axis=1)
            eyelashes_np = np.array([v.uv for v in eyelashes_uvs.data])
            eyelashes_np[is_eyebrows] = eyebrows_uvs_np[is_eyebrows]
            for v, new_uv in zip(eyelashes_uvs.data, eyelashes_np):
                v.uv = new_uv
            eyelashes.data.uv_layers.remove(eyebrows_uvs)
            for mat in eyelashes.material_slots[1:]:
                eyelashes.active_material_index = mat.slot_index
                bpy.ops.object.material_slot_remove()
            eyelashes.data.materials[0].name = 'Facial hair'

    def merge_eyelashes_and_body(self, join_uvs):
        eyelashes = self.get_eyelashes_mesh()
        # eyelashes_mats = [m.name for m in eyelashes.material_slots]
        body = self.get_body_mesh()
        eyelashes_uv_layer = eyelashes.data.uv_layers.active.name
        body_uv_layer = body.data.uv_layers.active.name
        if not join_uvs:
            toon_uv_name = eyelashes.data.uv_layers.active.name = NEW_TOON_EYELASHES_UV_MAP
            for mat in eyelashes.data.materials:
                NodesUtils.add_explicit_uvs(mat, toon_uv_name)

        select_object(body)
        eyelashes.select_set(True)
        bpy.ops.object.join()

        if join_uvs:
            eyelashes_uvs = body.data.uv_layers[eyelashes_uv_layer]
            body_uvs = body.data.uv_layers[body_uv_layer]
            eyelashes_uvs_np = np.array([v.uv for v in eyelashes_uvs.data])
            is_eyelashes = np.any(eyelashes_uvs_np > 0, axis=1)
            body_np = np.array([v.uv for v in body_uvs.data])
            body_np[is_eyelashes] = eyelashes_uvs_np[is_eyelashes]
            for v, new_uv in zip(body_uvs.data, body_np):
                v.uv = new_uv
            body.data.uv_layers.remove(eyelashes_uvs)

    def give_erection(self):
        import mathutils
        dick = has_dick()
        if dick is not None:
            if dick.name.startswith('Genesis 9 Anatomical Elements Male'):
                length_factor = 0.1
                rig = get_rig_of(dick)
                select_object(rig)
                bpy.ops.object.mode_set(mode='POSE')
                genbase = rig.pose.bones['genbase']
                base_direction = genbase.tail - genbase.head
                movement_direction = base_direction.normalized()
                for i in range(1, 7):
                    gen = rig.pose.bones['gen'+str(i)]
                    gen_direction = gen.tail-gen.head
                    gen_len = gen_direction.length * length_factor
                    quat_diff = gen_direction.rotation_difference(base_direction)
                    if gen.rotation_mode == 'QUATERNION':
                        gen.rotation_quaternion = quat_diff
                    else:
                        gen.rotation_euler = quat_diff.to_euler(gen.rotation_mode)
                    base_direction = gen_direction

                select_object(dick)
                for mod in dick.modifiers:
                    if isinstance(mod, bpy.types.ArmatureModifier):
                        name = mod.name
                        bpy.ops.object.modifier_apply(modifier=name)
                        break
                select_object(rig)
                bpy.ops.object.mode_set(mode='POSE')
                bpy.ops.pose.armature_apply(selected=False)
                mod = dick.modifiers.new(name, type='ARMATURE')
                mod.object = rig
                    #gen.location.y = 0.08


    def enlarge_dick(self):
        import mathutils
        dick = has_dick()
        if dick is not None:
            if dick.name.startswith('Genesis 9 Anatomical Elements Male'):
                length_factor = 0.1
                rig = get_rig_of(dick)
                select_object(rig)
                bpy.ops.object.mode_set(mode='POSE')
                genbase = rig.pose.bones['genbase']
                base_direction = genbase.tail - genbase.head
                movement_direction = base_direction.normalized()
                for i in range(1, 7):
                    gen = rig.pose.bones['gen'+str(i)]
                    gen_direction = gen.tail-gen.head
                    gen_len = gen_direction.length * length_factor
                    quat_diff = gen_direction.rotation_difference(base_direction)
                    if gen.rotation_mode == 'QUATERNION':
                        gen.rotation_quaternion = quat_diff
                    else:
                        gen.rotation_euler = quat_diff.to_euler(gen.rotation_mode)
                    base_direction = gen_direction

                select_object(dick)
                for mod in dick.modifiers:
                    if isinstance(mod, bpy.types.ArmatureModifier):
                        name = mod.name
                        bpy.ops.object.modifier_apply(modifier=name)
                        break
                select_object(rig)
                bpy.ops.object.mode_set(mode='POSE')
                bpy.ops.pose.armature_apply(selected=False)
                mod = dick.modifiers.new(name, type='ARMATURE')
                mod.object = rig


    def fix_toon_eyes(self):
        eyes = self.get_eyes_mesh()
        rig = self.get_body_rig()
        daz_dir = obj_daz_dir(eyes).lower()
        if daz_dir == NIRV_ZERO_EYES_DAZ_DIR:
            select_object(rig)
            bpy.ops.object.mode_set(mode='POSE')

            def fix_iris(iris_bone_name, eye_bone_name):
                iris_bone = rig.pose.bones[iris_bone_name]
                iris_bone.constraints["Limit Rotation"].use_limit_y = False
                eye_bone = rig.pose.bones[eye_bone_name]
                rotation_mode = eye_bone.rotation_mode
                for i, axis in enumerate('XYZ'):
                    iris_bone.driver_remove('rotation_euler', i)
                    driver = iris_bone.driver_add('rotation_euler', i).driver
                    driver.type = "SCRIPTED"
                    driver.expression = '-0.267 * A'
                    driver_var = driver.variables.new()
                    driver_var.type = 'TRANSFORMS'
                    driver_var.name = "A"

                    driver_target = driver_var.targets[0]
                    driver_target.id = rig
                    driver_target.bone_target = eye_bone_name
                    driver_target.transform_type = 'ROT_'+axis
                    driver_target.rotation_mode = rotation_mode
                    driver_target.transform_space = 'LOCAL_SPACE'

            fix_iris('Left Iris', 'l_eye')
            fix_iris('Right Iris', 'r_eye')

    def optimize_eyes(self, optimize_for_toon=False, hard_toon_edges=False):

        EYES_M = self.get_eyes_mesh()
        is_toon = bpy.context.scene.get('daz_optim_toon')
        is_floating_iris = False
        if EYES_M is None:
            eye_socket = DazOptimizer.get_toon_eye_socket_mesh()
            if eye_socket is not None:
                bpy.data.objects.remove(eye_socket)

            floating_iris = DazOptimizer.get_toon_floating_iris_mesh()
            is_floating_iris = floating_iris is not None or eye_socket is not None
            shadow_plane = DazOptimizer.get_mesh_by_name('Toon Shadow Plane Mesh')
            if shadow_plane is not None:
                bpy.data.objects.remove(shadow_plane)
        else:
            select_object(EYES_M)

            # uv_layer = EYES_M.data.uv_layers.active
            # uvs = np.array([v.uv for v in uv_layer.data], dtype=bool)
            # uvs[:, y] < 0.5
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.context.scene.tool_settings.use_uv_select_sync = False
            bpy.ops.uv.select_all(action='DESELECT')
            bpy.ops.mesh.select_all(action='DESELECT')

            me = bpy.context.object.data
            bm = bmesh.from_edit_mesh(me)
            uv_layer = bm.loops.layers.uv.verify()

            for v in bm.verts:
                v.select = False
            optim_toon = optimize_for_toon and is_toon
            is_floating_iris = optim_toon
            max_dist = (0.116 if hard_toon_edges else 0.128) if optim_toon else 0.24
            min_dist = 0 if optim_toon else 0.038
            for face in bm.faces:
                for loop in face.loops:
                    loop_uv = loop[uv_layer]
                    uv = np.array(loop_uv.uv)
                    dist = np.linalg.norm(uv % 0.5 - 0.25)
                    if uv[1] < 0.5 or dist > max_dist or dist < min_dist:
                        loop.vert.select = True
            for v in bm.verts:
                if v.select:
                    bm.verts.remove(v)
            bmesh.update_edit_mesh(me)
            bpy.ops.object.mode_set(mode='OBJECT')
            bpy.ops.object.material_slot_remove_unused()

            if not is_toon:
                bpy.ops.object.mode_set(mode='EDIT')
                me = bpy.context.object.data
                bm = bmesh.from_edit_mesh(me)
                uv_layer = bm.loops.layers.uv.verify()
                def select_loop(center):
                    for face in bm.faces:
                        for loop in face.loops:
                            loop_uv = loop[uv_layer]
                            uv = np.array(loop_uv.uv)
                            dist = np.linalg.norm(uv - center)
                            loop.vert.select_set(dist < 0.044)

                select_loop([0.25, 0.75])
                bpy.ops.mesh.edge_face_add()
                select_loop([0.75, 0.75])
                bpy.ops.mesh.edge_face_add()
                bmesh.update_edit_mesh(me)
                bpy.ops.object.mode_set(mode='OBJECT')

        bpy.context.scene['is_floating_iris'] = is_floating_iris
        if is_floating_iris:
            body = self.get_body_mesh()
            select_object(body)
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.context.scene.tool_settings.use_uv_select_sync = False
            bpy.ops.uv.select_all(action='DESELECT')
            bpy.ops.mesh.select_all(action='DESELECT')

            uv_mask = rle_decode(EYE_SOCKET_RLE, MASK_SHAPE)
            me = bpy.context.object.data
            bm = bmesh.from_edit_mesh(me)
            uv_layer = bm.loops.layers.uv.verify()
            for face in bm.faces:
                for loop in face.loops:
                    loop_uv = loop[uv_layer]
                    if loop_uv.uv.x < 1:
                        uv = np.array(loop_uv.uv)
                        uv[1] = 1 - uv[1]
                        pixel_coord = (uv * MASK_SHAPE[0]).clip(0, MASK_SHAPE[0] - 1)
                        pixel_coord = np.int32(pixel_coord)
                        is_eye_socket = uv_mask[pixel_coord[1], pixel_coord[0]]
                        if is_eye_socket:
                            v0 = loop.edge.verts[0]
                            v1 = loop.edge.verts[1]
                            is_boundary = len(v0.link_edges)==3 and len(v1.link_edges)==3
                            if is_boundary:
                                loop.edge.select = True

            bmesh.update_edit_mesh(me)
            bpy.ops.mesh.edge_face_add()

    def merge_all_materials(self):
        for o in bpy.data.objects:
            if isinstance(o.data, bpy.types.Mesh):
                if not o.name.startswith("Love Loads"):
                    select_object(o)
                    bpy.ops.daz.merge_materials()

    def merge_cum_materials(self):
        mat_per_texture = {}
        for o in find_cum():
            cum_material = o.material_slots[0].material
            nt = cum_material.node_tree
            textures = NodesUtils.find_all_by_type(nt, bpy.types.ShaderNodeTexImage)
            if len(textures)>0:
                texture = textures[0]
                image = texture.image
                unified_mat = mat_per_texture.get(image)
                if unified_mat is None:
                    unified_mat = mat_per_texture[image] = cum_material
            else:
                unified_mat = cum_material
            o.material_slots[0].material = unified_mat

    def decimate_cum_meshes(self):
        ng = bpy.data.node_groups.get('DecimateCum')
        if ng is None:
            body = self.get_body_mesh()

            ng = bpy.data.node_groups.new('DecimateCum', 'GeometryNodeTree')
            ng.interface.new_socket(name='Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
            ng.interface.new_socket(name='Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
            group_input = ng.nodes.new('NodeGroupInput')
            group_input.name = 'Group Input'
            group_input.location = (-340.0, 0.0)
            group_output = ng.nodes.new('NodeGroupOutput')
            group_output.name = 'Group Output'
            group_output.location = (608.8350830078125, 127.03092193603516)
            delete_geometry = ng.nodes.new('GeometryNodeDeleteGeometry')
            delete_geometry.name = 'Delete Geometry'
            delete_geometry.location = (383.6963195800781, 3.5750083923339844)
            vector_math_001 = ng.nodes.new('ShaderNodeVectorMath')
            vector_math_001.name = 'Vector Math.001'
            vector_math_001.location = (-17.48244857788086, -212.68609619140625)
            vector_math_001.operation = 'DOT_PRODUCT'
            compare = ng.nodes.new('FunctionNodeCompare')
            compare.name = 'Compare'
            compare.location = (181.4677276611328, -146.98037719726562)
            compare.data_type = 'FLOAT'
            compare.operation = 'LESS_THAN'
            sample_nearest_surface = ng.nodes.new('GeometryNodeSampleNearestSurface')
            sample_nearest_surface.name = 'Sample Nearest Surface'
            sample_nearest_surface.location = (-271.0045166015625, -196.76165771484375)
            sample_nearest_surface.data_type = 'FLOAT_VECTOR'
            object_info = ng.nodes.new('GeometryNodeObjectInfo')
            object_info.name = 'Object Info'
            object_info.location = (-512.5869750976562, -178.63140869140625)
            object_info.transform_space = 'ORIGINAL'
            normal_001 = ng.nodes.new('GeometryNodeInputNormal')
            normal_001.name = 'Normal.001'
            normal_001.location = (-538.8834228515625, -440.8331604003906)
            ng.links.new(group_output.inputs['Geometry'], delete_geometry.outputs['Geometry'])
            ng.links.new(delete_geometry.inputs['Geometry'], group_input.outputs['Geometry'])
            ng.links.new(delete_geometry.inputs['Selection'], compare.outputs['Result'])
            ng.links.new(vector_math_001.inputs['Vector'], sample_nearest_surface.outputs['Value'])
            ng.links.new(vector_math_001.inputs['Vector_001'], normal_001.outputs['Normal'])
            ng.links.new(compare.inputs['A'], vector_math_001.outputs['Value'])
            compare.inputs['B'].default_value = 0.0
            ng.links.new(sample_nearest_surface.inputs['Mesh'], object_info.outputs['Geometry'])
            ng.links.new(sample_nearest_surface.inputs['Value'], normal_001.outputs['Normal'])
            sample_nearest_surface.inputs['Group ID'].default_value = 0
            sample_nearest_surface.inputs['Sample Position'].default_value = (0.0, 0.0, 0.0)
            sample_nearest_surface.inputs['Sample Group ID'].default_value = 0
            object_info.inputs['Object'].default_value = body
            object_info.inputs['As Instance'].default_value = False

        for o in find_cum():
            gn_mod = o.modifiers.get("Geometry Nodes")
            if gn_mod is not None:
                o.modifiers.remove(gn_mod)
            gn_mod = o.modifiers.new("Geometry Nodes", type='NODES')
            gn_mod.node_group = ng
            dec_mod = o.modifiers.get("Decimate")
            if dec_mod is not None:
                o.modifiers.remove(dec_mod)
            dec_mod = o.modifiers.new("Decimate", type='DECIMATE')
            dec_mod.decimate_type = 'UNSUBDIV'
            dec_mod.iterations = 3

    def apply_decimate_cum_meshes(self):
        for o in find_cum():
            select_object(o)
            bpy.ops.object.modifier_apply(modifier='Geometry Nodes')
            bpy.ops.object.modifier_apply(modifier='Decimate')

    def merge_multi_mesh_clothes(self):
        for trash in ['BaseShortsGeoGraft Mesh']:
            trash = bpy.data.objects.get(trash)
            if trash is not None:
                bpy.data.objects.remove(trash)
        for clothing_item in find_all_clothes():
            sub_clothes = find_child_meshes(clothing_item.obj)
            if len(sub_clothes)>0:
                select_object(clothing_item.obj)
                for sub in sub_clothes:
                    sub.hide_viewport = False
                    sub.hide_render = False
                    sub.hide_set(False)
                    sub.select_set(True)
                bpy.ops.object.join()

    def remove_all_subsurfs(self):
        for o in bpy.data.objects:
            for m in o.modifiers:
                if m.type == "SUBSURF":
                    o.modifiers.remove(m)

    def merge_all_rigs(self):
        body_rig = self.get_body_rig()
        select_object(body_rig)
        meshes = []
        for o in bpy.data.objects:
            if o == body_rig:
                continue
            if isinstance(o.data, bpy.types.Armature):
                if not is_sub_rig(o.data, body_rig.data) and (is_hair(o) or is_clothes(o) or is_cum(o)):
                    o.hide_viewport = True
                    o.hide_set(True)
                    print("Ignore", o)
                else:
                    o.hide_viewport = False
                    o.hide_render = False
                    o.hide_set(False)
                    # o.data.hide_set(False)
                    o.select_set(True)
                for c in o.children:
                    if isinstance(c.data, bpy.types.Mesh):
                        meshes.append(c.name)
        bpy.ops.daz.merge_rigs()
        for o in bpy.data.objects:
            o.hide_viewport = False
            o.hide_render = False
            o.hide_set(False)
        for mesh_name in meshes:
            mesh = bpy.data.objects[mesh_name]
            for mod in mesh.modifiers:
                if isinstance(mod, bpy.types.ArmatureModifier):
                    if mod.object is None:
                        mod.object = body_rig
            if mesh.parent is None:
                mesh.parent = body_rig

    def simplify_eyes_material(self):
        EYES_M = self.get_eyes_mesh()
        is_toon = EYES_M is None
        if is_toon:
            EYES_M = DazOptimizer.get_toon_floating_iris_mesh()
            all_filepaths = DazOptimizer.find_body_part_textures(EYES_M.data.materials)
            highlight_color = None
            highlight_body_part = None
            for body_part, channels in all_filepaths.items():
                bp = body_part.lower()
                if 'highlight' in bp:
                    highlight_color = channels['Base Color']
                    highlight_body_part = body_part
            if isinstance(highlight_color, np.ndarray):
                select_object(EYES_M)
                eyes_layer = EYES_M.data.uv_layers[0]
                eyes_layer.name = NEW_EYES_UV_MAP
                eyes_layer_np = np.array([v.uv for v in eyes_layer.data])
                is_highlight = eyes_layer_np[:, 0] > 1
                eyes_layer_np[is_highlight] = (eyes_layer_np[is_highlight] % 1) * 0.25 + (0.5-(1/8), 0.75)
                for v, new_uv in zip(eyes_layer.data, eyes_layer_np):
                    v.uv = new_uv
                for slot in EYES_M.material_slots:
                    mat = slot.material
                    if mat.name.rstrip('0123456789-_.') == highlight_body_part:
                        EYES_M.active_material_index = slot.slot_index
                        bpy.ops.object.material_slot_remove()

        else:
            all_filepaths = DazOptimizer.find_body_part_textures(EYES_M.data.materials)
        DazOptimizer.gen_simple_materials(EYES_M.data.materials, all_filepaths)


    def separate_iris_uvs(self):
        EYES_M = self.get_eyes_mesh()
        if EYES_M is None:
            EYES_M = self.get_toon_floating_iris_mesh()
        if EYES_M is None:
            return
        old_uv_layer = EYES_M.data.uv_layers.active
        new_uv_layer = EYES_M.data.uv_layers.new(name=NEW_EYES_UV_MAP)
        new_uv_layer.active = True
        new_uv_layer.active_render = True
        select_object(EYES_M)
        bpy.ops.object.mode_set(mode='EDIT')

        bpy.context.scene.tool_settings.use_uv_select_sync = False
        bpy.ops.uv.select_all(action='DESELECT')
        bpy.ops.mesh.select_all(action='DESELECT')

        me = bpy.context.object.data
        bm = bmesh.from_edit_mesh(me)
        uv_layer = bm.loops.layers.uv.verify()

        # for v in bm.verts:
        #    v.select_set(False)
        iris_uv_radius = 0.86577 - 0.75 + 0.001

        for face in bm.faces:
            full_loop = True
            for loop in face.loops:
                loop_uv = loop[uv_layer]
                uv = np.array(loop_uv.uv)
                is_iris = np.linalg.norm(np.mod(uv, 0.5) - 0.25) < iris_uv_radius
                full_loop = full_loop and is_iris
            face.select_set(full_loop)

        # bm.select_mode = {'VERT', 'EDGE', 'FACE'}
        bm.select_flush_mode()
        # bpy.context.tool_settings.mesh_select_mode = (False, False, True)
        bpy.ops.uv.select_all(action='SELECT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.select_split()

        bpy.ops.object.mode_set(mode='OBJECT')
        new_uv_layer_np = np.array([v.uv for v in EYES_M.data.uv_layers.active.data])
        # old_uv_layer_np = np.array([v.uv for v in old_uv_layer.data])
        selection = np.array([not v.select for v in EYES_M.data.uv_layers.active.data], dtype=bool)
        # if bpy.context.scene.get('daz_optim_toon'):
        #     selection = np.logical_not(selection)
        new_uv_layer_np[selection, 1] -= 0.5
        for v, new_uv in zip(EYES_M.data.uv_layers.active.data, new_uv_layer_np):
            v.uv = new_uv
        # += [0.043945, 0.006836] # top arm
        # += [-0.072266 , 0.085937] # obttom arm
        # += [0.008526, 0.019377] # torso
        # *= 0.25# nails
        # -= 0.5 # nails

    def find_body_parts_textures(self):
        BODY_M = self.get_body_mesh()
        mats = list(BODY_M.data.materials)
        for g in DICK_GEOGRAFTS:
            if g+' Mesh' in bpy.data.objects:
                mats.extend(bpy.data.objects[g+' Mesh'].data.materials)
        return DazOptimizer.find_body_part_textures(mats)

    @staticmethod
    def find_body_part_textures(mats):
        def find_textures(input_socket, outputs):
            for link in input_socket.links:
                node = link.from_node
                if node not in outputs:
                    if isinstance(node, bpy.types.ShaderNodeTexImage):
                        outputs.add(node.image)
                        continue
                    elif isinstance(node, bpy.types.ShaderNodeGroup):
                        nt = node.node_tree
                        fs = link.from_socket
                        for group_output in NodesUtils.find_all_by_type(nt, bpy.types.NodeGroupOutput):
                            path = bpy.path.abspath('//' + nt.name) + '.png'
                            inp_soc = group_output.inputs[fs.name]
                            if os.path.exists(path):
                                outputs.add(bpy.data.images.load(path))
                                continue
                            else:
                                print("Baking LIE group: ",nt)
                                baked_img = NodesUtils.bake_textures(inp_soc)
                                if isinstance(baked_img, BakedImg):
                                    outputs.add(baked_img.to_image(path))
                                    continue
                                else:
                                    find_textures(inp_soc, outputs)

                    for in_soc in node.inputs:
                        find_textures(in_soc, outputs)
            return outputs

        all_filepaths: {str: {str: [bpy.types.Image]}} = {}
        const_color_values = {}
        for mat in mats:
            output_node = NodesUtils.find_by_type(mat.node_tree, bpy.types.ShaderNodeOutputMaterial)
            body_part = mat.name.rstrip('0123456789-_.')
            body_part_filepaths = all_filepaths[body_part] = {'Base Color': set(), 'Roughness': set(), 'Normal': set()}
            const_color_value = None
            if output_node is not None:
                for bsdf in NodesUtils.from_socket_backwards_search_for(output_node.inputs['Surface'], (bpy.types.ShaderNodeBsdfPrincipled, bpy.types.ShaderNodeGroup), set()):
                    if isinstance(bsdf, bpy.types.ShaderNodeBsdfPrincipled):
                        for channel in ['Base Color', 'Roughness', 'Normal']:
                            for image in find_textures(bsdf.inputs[channel], set()):
                                body_part_filepaths[channel].add(image)
                                print(body_part, channel, image)
                    elif bsdf.node_tree.name == 'DAZ Dual Lobe PBR':
                        for image in find_textures(bsdf.inputs['Roughness 1'], set()):
                            body_part_filepaths['Roughness'].add(image)
                            print(body_part, "Roughness", image)
                        for image in find_textures(bsdf.inputs['Roughness 2'], set()):
                            body_part_filepaths['Roughness'].add(image)
                            print(body_part, "Roughness", image)
                        for image in find_textures(bsdf.inputs['Normal'], set()):
                            body_part_filepaths['Normal'].add(image)
                            print(body_part, "Normal", image)
                    elif bsdf.node_tree.name == 'DAZ Toon Diffuse':
                        clr_soc = bsdf.inputs['Color']
                        if len(clr_soc.links) == 0:
                            const_color_value = np.array(clr_soc.default_value)
                        else:
                            for image in find_textures(bsdf.inputs['Color'], set()):
                                body_part_filepaths['Base Color'].add(image)
                                print(body_part,"Base Color",image)
                        for image in find_textures(bsdf.inputs['Normal'], set()):
                            body_part_filepaths['Normal'].add(image)
                            print(body_part, "Normal", image)
                if len(body_part_filepaths['Base Color']) == 0 and const_color_value is not None:
                    const_color_values[body_part] = const_color_value
                    print(body_part, "Base Color", const_color_value)
        for body_part_name, body_part_filepaths in all_filepaths.items():
            occurrences = {}
            filenames = []
            for channel in body_part_filepaths.values():
                if len(channel)==1:
                    first = next(iter(channel))
                    filepath = first.filepath
                    filenames.append(filepath)
            lcp = os.path.commonprefix(filenames)
            for channel in body_part_filepaths.values():
                for image in channel:
                    if image not in occurrences:
                        occurrences[image] = 0
                    occurrences[image] += 1 + len(os.path.commonprefix([image.filepath, lcp]))

            for channel in body_part_filepaths:
                s = list(sorted(body_part_filepaths[channel], key=lambda x: -occurrences[x]))
                if channel == "Base Color" and len(s)==0 and body_part_name in const_color_values:
                    s = const_color_values[body_part_name]
                body_part_filepaths[channel] = s
        print(json.dumps({k: {k2: v2.tolist() if isinstance(v2, np.ndarray) else [v3.filepath+" "+str(tuple(v3.size)+(v3.channels,)) for v3 in v2] for k2, v2 in v.items()} for k, v in all_filepaths.items()}, indent=2))
        return all_filepaths

    @staticmethod
    def gen_simple_materials(mats, all_filepaths=None):
        if all_filepaths is None:
            all_filepaths = DazOptimizer.find_body_part_textures(mats)
        for mat in mats:
            body_part = mat.name.rstrip('0123456789-_.')
            body_part_filepaths = all_filepaths[body_part]
            mat.node_tree.nodes.clear()
            NodesUtils.gen_simple_material(mat.node_tree, body_part_filepaths)

    def simplify_materials(self):
        from PIL import Image
        BODY_M = self.get_body_mesh()
        MOUTH_M = self.get_mouth_mesh()
        gp = self.get_gp_mesh()
        is_toon = bpy.context.scene.get('daz_optim_toon')
        mats = list(BODY_M.data.materials)
        mats.extend(MOUTH_M.data.materials)
        if is_toon:
            mats.extend(gp.data.materials)
        for g in DICK_GEOGRAFTS:
            if g + ' Mesh' in bpy.data.objects:
                mats.extend(bpy.data.objects[g + ' Mesh'].data.materials)
        all_filepaths = DazOptimizer.find_body_part_textures(mats)


        nails_img = None
        head_img = None
        head_body_part = None
        mouth_cavity_color = None
        mouth_cavity_body_part = None
        fingernails_color = None
        fingernails_body_part = None
        teeth_color = None
        teeth_img = None
        teeth_body_part = None
        mouth_color = None
        mouth_img = None
        mouth_body_part = None
        toenails_color = None
        toenails_body_part = None
        eye_socket_color = None
        toon_eye_socket = DazOptimizer.get_toon_eye_socket_mesh()
        if toon_eye_socket is not None:
            eye_socket_color = np.array((1, 1, 1, 1))
            bpy.data.objects.remove(toon_eye_socket)

        for body_part, channels in all_filepaths.items():
            bc = channels['Base Color']
            bp = body_part.lower()
            if "nails" in bp:
                if "finger" in bp:
                    fingernails_body_part = body_part
                    if isinstance(bc, np.ndarray):
                        fingernails_color = linearrgb_to_srgb(bc)
                    elif len(bc)>0:
                        nails_img = bc[0]
                elif "toe" in bp:
                    toenails_body_part = body_part
                    if isinstance(bc, np.ndarray):
                        toenails_color = linearrgb_to_srgb(bc)
                    elif len(bc) > 0:
                        nails_img = bc[0]
            elif "mouth" in bp:
                if "cavity" in bp:
                    mouth_cavity_body_part = body_part
                    if isinstance(bc, np.ndarray):
                        mouth_cavity_color = linearrgb_to_srgb(bc)
                else:
                    mouth_body_part = body_part
                    if isinstance(bc, np.ndarray):
                        mouth_color = linearrgb_to_srgb(bc)
                    elif len(bc) > 0:
                        mouth_img = bc[0]
            elif 'head' in bp:
                head_body_part = body_part
                if not isinstance(bc, np.ndarray) and len(bc)>0:
                    head_img = bc[0]
            elif 'teeth' in bp:
                teeth_body_part = body_part
                if isinstance(bc, np.ndarray):
                    teeth_color = linearrgb_to_srgb(bc)
                elif len(bc) > 0:
                    teeth_img = bc[0]



        def to_channels(x, c):
            if len(x) < c:
                return np.append(x, 1)
            else:
                return x[:c]

        if mouth_color is not None or teeth_color is not None:
            dst_mouth_img_path = bpy.path.abspath("//mouth.png")
            if MOUTH_M.daz_importer.DazMesh == 'Toon9-mouth':
                mouth_mask = rle_decode(TOON_MOUTH_RLE, TOON_MOUTH_MASK_SHAPE)
                if not os.path.exists(dst_mouth_img_path):
                    if teeth_img is None:
                        if mouth_img is None:
                            mouth_img_np = np.zeros((TOON_MOUTH_MASK_SHAPE[0], TOON_MOUTH_MASK_SHAPE[1], 3))
                        else:
                            mouth_img_np = open_img_to_np(mouth_img.filepath)
                    else:
                        if mouth_img is None:
                            mouth_img_np = open_img_to_np(teeth_img.filepath)
                        else:
                            mouth_img_np = open_img_to_np(mouth_img.filepath)
                    h, w, c = mouth_img_np.shape
                    if [h,w] != TOON_MOUTH_MASK_SHAPE:
                        mouth_mask = Image.fromarray(mouth_mask * np.uint8(255))
                        mouth_mask = mouth_mask.resize((w,h))
                        mouth_mask = np.array(mouth_mask)>0
                    if mouth_color is not None:
                        mouth_color = to_channels(mouth_color, c)
                        print("Baking mouth color: ", mouth_color)
                        mouth_img_np[mouth_mask] = mouth_color
                    if teeth_color is not None:
                        teeth_mask = np.logical_not(mouth_mask)
                        teeth_color = to_channels(teeth_color, c)
                        print("Baking teeth color: ", teeth_color)
                        mouth_img_np[teeth_mask] = teeth_color
                    np_to_pil(mouth_img_np).save(dst_mouth_img_path)
                mouth_img = bpy.data.images.load(dst_mouth_img_path)
                if teeth_body_part is not None:
                    all_filepaths[teeth_body_part]['Base Color'] = [mouth_img]
                if mouth_body_part is not None:
                    all_filepaths[mouth_body_part]['Base Color'] = [mouth_img]
                select_object(MOUTH_M)
                mouth_layer = MOUTH_M.data.uv_layers[0]
                mouth_layer.name = NEW_EYES_UV_MAP
                mouth_layer_np = np.array([v.uv for v in mouth_layer.data])
                pixel_coords = DazOptimizer.base_layer_to_pixel_coords(mouth_layer_np, mask_shape=TOON_MOUTH_MASK_SHAPE)
                is_mouth = mouth_mask[pixel_coords[:, 1], pixel_coords[:, 0]]
                is_teeth = np.logical_not(is_mouth)
                current_height = 1-0.430743
                desired_height = 1-0.52
                scale = desired_height/current_height
                translate_y = (1-scale)
                translate_x = (1 - scale)/2
                mouth_layer_np[is_teeth] = mouth_layer_np[is_teeth] * scale + (translate_x, translate_y)
                for v, new_uv in zip(mouth_layer.data, mouth_layer_np):
                    v.uv = new_uv
        if head_img is not None and (mouth_cavity_color is not None or eye_socket_color is not None):
            dst_head_img_path = bpy.path.abspath("//head.png")
            if not os.path.exists(dst_head_img_path):
                if head_img is None:
                    head_img_np = np.zeros((4096, 4096, 4))
                else:
                    head_img_np = open_img_to_np(head_img.filepath)
                h, w, c = head_img_np.shape
                if mouth_cavity_color is not None:
                    mouth_cavity_rle = rle_decode(MOUTH_CAVITY_RLE, MASK_SHAPE)
                    mouth_cavity_color = to_channels(mouth_cavity_color, c)
                    print("Baking mouth cavity color: ", mouth_cavity_color)
                    head_img_np[mouth_cavity_rle] = mouth_cavity_color
                if eye_socket_color is not None:
                    eye_socket_rle = rle_decode(EYE_SOCKET_RLE, MASK_SHAPE)
                    eye_socket_color = to_channels(eye_socket_color, c)
                    print("Baking eye socket color: ", eye_socket_color)
                    head_img_np[eye_socket_rle] = eye_socket_color
                np_to_pil(head_img_np).save(dst_head_img_path)
            head_img = bpy.data.images.load(dst_head_img_path)
            if mouth_cavity_body_part is not None:
                all_filepaths[mouth_cavity_body_part]['Base Color'] = [head_img]
            if head_body_part is not None:
                all_filepaths[head_body_part]['Base Color'] = [head_img]
        if fingernails_color is not None or toenails_color is not None:
            dst_nails_img_path = bpy.path.abspath("//nails.png")
            if not os.path.exists(dst_nails_img_path):
                if nails_img is None:
                    nails_img_np = np.zeros((4096, 4096, 4))
                else:
                    nails_img_np = open_img_to_np(nails_img.filepath)
                h, w, c = nails_img_np.shape
                if fingernails_color is not None:
                    fingernails_color = to_channels(fingernails_color, c)
                    print("Baking finger nails color: ", fingernails_color)
                    nails_img_np[:h//2] = fingernails_color
                if toenails_color is not None:
                    toenails_color = to_channels(toenails_color, c)
                    print("Baking toe nails color: ", toenails_color)
                    nails_img_np[h//2:] = toenails_color
                np_to_pil(nails_img_np).save(dst_nails_img_path)
            nails_img = bpy.data.images.load(dst_nails_img_path)
            if fingernails_body_part is not None:
                all_filepaths[fingernails_body_part]['Base Color'] = [nails_img]
            if toenails_body_part is not None:
                all_filepaths[toenails_body_part]['Base Color'] = [nails_img]

        for n in BREAST_GEOGRAFTS + MALE_ONLY_GEOGRAFTS:
            n = n + ' Mesh'
            if n in bpy.data.objects:
                mats.extend(bpy.data.objects[n].data.materials)
        DazOptimizer.gen_simple_materials(mats, all_filepaths)

        body_part_filepaths = all_filepaths['Body']
        if gp is not None and not is_toon:
            for mat in gp.data.materials:
                print("mat=", mat)
                output_node = NodesUtils.find_by_type(mat.node_tree, bpy.types.ShaderNodeOutputMaterial)
                bsdf, = NodesUtils.from_socket_backwards_search_for(output_node.inputs['Surface'], bpy.types.ShaderNodeBsdfPrincipled, set())
                tail = output_node.inputs['Surface'].links[0].from_node
                while isinstance(tail, bpy.types.ShaderNodeGroup) and 'BSDF' in output_node.inputs:
                    output_node = tail
                    tail = output_node.inputs['BSDF'].links[0].from_node
                print("output_node=", output_node)
                out_socket = bsdf.outputs['BSDF'].links[0].to_socket
                NodesUtils.delete_all_before(mat.node_tree, bsdf)
                NodesUtils.gen_simple_material(mat.node_tree, body_part_filepaths, out_socket, shift_x=output_node.location[0]-300,uvs='Default UVs')

    def concat_textures(self):
        from PIL import Image
        import re

        all_filepaths = self.find_body_parts_textures()

        head_filepaths = {}
        arms_filepaths = {}
        legs_filepaths = {}
        nails_filepaths = {}
        body_filepaths = {}
        mouth_filepaths = {}
        eyes_filepaths = {}
        eyebrows_filepaths = {}
        eyelashes_filepaths = {}
        gp_filepaths = {}
        genitalia_filepaths = {}
        is_eyelashes_transparent_toon = False
        is_eyebrows_transparent_toon = False
        for body_part, body_part_filepaths in all_filepaths.items():
            body_part_l = body_part.lower()
            if 'head' in body_part_l:
                head_filepaths = body_part_filepaths
            elif 'arms' in body_part_l:
                arms_filepaths = body_part_filepaths
            elif 'body' in body_part_l:
                body_filepaths = body_part_filepaths
            elif 'legs' in body_part_l:
                legs_filepaths = body_part_filepaths
            elif 'nails' in body_part_l:
                nails_filepaths = body_part_filepaths
            elif 'mouth' in body_part_l:
                mouth_filepaths = body_part_filepaths
            elif 'eyebrows' in body_part_l:
                is_eyebrows_transparent_toon = body_part == TRANSPARENT_TOON_EYEBROWS_MAT_NAME
                eyebrows_filepaths = body_part_filepaths
            elif 'eyelashes' in body_part_l:
                is_eyelashes_transparent_toon = body_part == TRANSPARENT_TOON_EYELASHES_MAT_NAME
                eyelashes_filepaths = body_part_filepaths
            elif 'eye' in body_part_l:
                eyes_filepaths = body_part_filepaths
            elif body_part_l.startswith('gp_') or body_part.startswith('WK '):
                gp_filepaths = body_part_filepaths
            elif 'genital' in body_part_l:
                genitalia_filepaths = body_part_filepaths

        def open_img(filepaths, map_type, resize=None):
            fp = filepaths[map_type]
            if isinstance(fp, list):
                if len(fp)==0:
                    return None
                fp = fp[0]
            if isinstance(fp, bpy.types.Image):
                fp = bpy.path.abspath(fp.filepath)
            if not isinstance(fp, str) or len(fp)==0:
                raise Exception(str(fp)+" is not string")
            print("Reading ", fp, end='', flush=True)
            tile = Image.open(fp)
            print(" of size ", tile.size," and type ", tile.format)
            if resize is not None:
                tile = tile.resize(resize)
                print("Resized to ", tile.size, " ", tile.format)
            tile = np.array(tile)
            if map_type == "Roughness" and tile.ndim>2 and tile.shape[2]>1:
                tile = np.average(tile, axis=2)
                tile = tile.astype(np.uint8)
                print("Converted to greyscale ", tile.shape, " ",tile.dtype)

            return tile

        # from matplotlib import pyplot as plt
        is_toon = bpy.context.scene.get('daz_optim_toon')
        is_floating_iris = bpy.context.scene.get('is_floating_iris')

        for map_type in ["Base Color", "Roughness", "Normal"]:
            print("map_type=", map_type)
            head_tile = open_img(head_filepaths, map_type)
            body_tile = open_img(body_filepaths, map_type)
            arms_tile = open_img(arms_filepaths, map_type)
            legs_tile = open_img(legs_filepaths, map_type)
            if head_tile is None and body_tile is None and arms_tile is None and legs_tile is None:
                continue
            d = min(9 if head_tile is None else head_tile.ndim ,
                    9 if body_tile is None else body_tile.ndim,
                    9 if arms_tile is None else arms_tile.ndim,
                    9 if legs_tile is None else legs_tile.ndim)
            c = 1 if d < 3 else min(
                9 if head_tile is None else head_tile.shape[-1],
                9 if body_tile is None else body_tile.shape[-1],
                9 if arms_tile is None else arms_tile.shape[-1],
                9 if legs_tile is None else legs_tile.shape[-1]
            )
            s = max(
                0 if head_tile is None else head_tile.shape[0],
                0 if body_tile is None else body_tile.shape[0],
                0 if arms_tile is None else arms_tile.shape[0],
                0 if legs_tile is None else legs_tile.shape[0]
            )
            dtp = None
            if head_tile is not None:
                dtp = head_tile.dtype
            elif body_tile is not None:
                dtp = body_tile.dtype
            elif arms_tile is not None:
                dtp = arms_tile.dtype
            elif legs_tile is not None:
                dtp = legs_tile.dtype
            s2 = s * 2
            s4 = s // 4
            s8 = s // 8
            merged_shape = [s2, s2, c]
            mouth_tile = None
            if map_type in mouth_filepaths and len(mouth_filepaths[map_type]) > 0:
                mouth_tile = open_img(mouth_filepaths, map_type, [s4, s4])
            eyelashes_tile = None
            if is_toon:
                if not is_eyelashes_transparent_toon and map_type in eyelashes_filepaths and len(eyelashes_filepaths[map_type]) > 0:
                    eyelashes_tile = open_img(eyelashes_filepaths, map_type,  [s4, s4])
                elif not is_eyebrows_transparent_toon and map_type in eyebrows_filepaths and len(eyebrows_filepaths[map_type]) > 0:
                    eyelashes_tile = open_img(eyebrows_filepaths, map_type, [s4, s4])
            eyes_tile = None
            if map_type in eyes_filepaths and len(eyes_filepaths[map_type])>0:
                eyes_tile = open_img(eyes_filepaths, map_type,  [s4, s4])
            nails_tile = None
            if map_type in nails_filepaths and len(nails_filepaths[map_type])>0:
                nails_tile = open_img(nails_filepaths, map_type,  [s4, s4])
            genital_tile = None
            if map_type in gp_filepaths and len(gp_filepaths[map_type])>0:
                genital_tile = open_img(gp_filepaths, map_type, [s4, s4])
            elif map_type in genitalia_filepaths and len(genitalia_filepaths[map_type])>0:
                genital_tile = open_img(genitalia_filepaths, map_type, [s4, s4])

            def prepare_channels(img: np.ndarray):
                if img.ndim < 3:
                    img = np.expand_dims(img, 2)
                if c == 1:
                    if img.shape[2] > 1:
                        return np.mean(axis=2)
                elif c >= 3:
                    if img.shape[2] == 1:
                        img = img.repeat(c, 2)
                        if c == 4:
                            img[:, :, 3] = 1
                    elif img.shape[2] == 3:
                        if c > 3:
                            hwc = (img.shape[0], img.shape[1], 1)
                            img = np.dstack([img, np.ones(hwc)])
                    elif img.shape[2] == 4:
                        if c == 3:
                            img = img[:, :, :c]
                return img

            def shift_img(img: np.ndarray, y0, y1, x0, x1, mask: np.ndarray, translation: [float, float], hflip=False, scale=1):
                new_img = np.zeros(merged_shape, dtype=dtp)
                if img is None:
                    return new_img
                if hflip:
                    mask = np.flipud(mask)
                    img = np.flipud(img)
                img = prepare_channels(img)
                print("img.shape=",img.shape,"\nnew_img.shape=",new_img.shape,"\nmask.shape=",mask.shape, "\nnew_img[y0:y1, x0:x1].shape=", new_img[y0:y1, x0:x1].shape, "\nimg[mask].shape=", img[mask].shape, "\nnew_img[y0:y1, x0:x1][mask]=",new_img[y0:y1, x0:x1][mask].shape)
                new_img[y0:y1, x0:x1][mask] = img[mask]
                if scale!=1:
                    resized = Image.fromarray(new_img).resize([merged_shape[1]//scale, merged_shape[0]//scale])
                    resized = np.array(resized)
                    print("Resizing ", new_img.shape, " to ", resized.shape)
                    new_img.fill(0)
                    np.copyto(new_img[-resized.shape[0]:, :resized.shape[1]], resized)

                x, y = np.int32(np.array(translation) * s2)
                if x != 0 or y != 0:
                    new_img = np.roll(new_img, [-y, x], axis=[0, 1])
                return new_img

            def assign_img(img: np.ndarray, y0, y1, x0, x1):
                img = prepare_channels(img)
                packed[y0:y1, x0:x1] = img
            # Textures are concatenated as follows:
            #   Legs | Arms
            #  ------+-----
            #   Head | Body

            packed = shift_img(arms_tile, 0, s, s, s2, rle_decode(BOT_ARM_RLE, MASK_SHAPE), BOT_ARM_TRANS)
            packed = np.maximum(packed, shift_img(arms_tile, 0, s, s, s2, rle_decode(TOP_ARM_RLE, MASK_SHAPE), TOP_ARM_TRANS))
            packed = np.maximum(packed, shift_img(legs_tile, 0, s, 0, s, rle_decode(LEFT_LEG_RLE, MASK_SHAPE), [0, 0]))
            packed = np.maximum(packed, shift_img(legs_tile, 0, s, 0, s, rle_decode(RIGHT_LEG_RLE, MASK_SHAPE), [RIGHT_LEG_TRANS, 0], True))
            packed = np.maximum(packed, shift_img(body_tile, s, s2, s, s2, rle_decode(BODY_RLE, MASK_SHAPE), BODY_TRANS))
            packed = np.maximum(packed, shift_img(head_tile, s, s2, 0, s, rle_decode(LIP_RLE, MASK_SHAPE), LIP_TRANS))
            packed = np.maximum(packed, shift_img(head_tile, s, s2, 0, s, rle_decode(MOUTH_CAVITY_RLE, MASK_SHAPE), MOUTH_CAVITY_SCALED_TRANS, scale=2))
            if is_floating_iris:
                right_eye_socket_mask = rle_decode(EYE_SOCKET_RLE, MASK_SHAPE)
                left_eye_socket_mask = right_eye_socket_mask.copy()
                right_eye_socket_mask[:, :MASK_SHAPE[1]//2] = False
                left_eye_socket_mask[:, MASK_SHAPE[1] // 2:] = False
                packed = np.maximum(packed, shift_img(head_tile, s, s2, 0, s, left_eye_socket_mask, LEFT_EYE_SOCKET_TRANS))
                packed = np.maximum(packed, shift_img(head_tile, s, s2, 0, s, right_eye_socket_mask, RIGHT_EYE_SOCKET_TRANS))
            # packed += shift_img(head, s, s2, s, s2, head_region_mask == HEAD_COLOR, [0.008526, 0.019377])
            assign_img(head_tile, s, s2, 0, s)
            if nails_tile is not None:
                assign_img(nails_tile, s2 - s4, s2, s, s + s4)
            if mouth_tile is not None:
                assign_img(mouth_tile, s2 - s4, s2, s+s4, s + s4*2)
            if eyes_tile is not None:
                assign_img(eyes_tile[:s8], s2 - s4 - s8, s2 - s4, s + s4 * 1, s + s4 * 2)
                assign_img(eyes_tile[s8:], s2 - s4 - s8, s2 - s4, s + s4 * 2, s + s4 * 3)
            if genital_tile is not None:
                assign_img(genital_tile, s2 - s4, s2, s + s4 * 2, s + s4 * 3)
            if eyelashes_tile is not None:
                assign_img(eyelashes_tile, s2 - s4, s2, s + s4 * 3, s + s4 * 4)


            # packed[:s, :s] = legs_tile
            # packed[s:, s:] = body_tile
            # packed[:s, s:] = arms_tile
            print("packed.shape=", packed.shape)
            if packed.ndim > 2 and packed.shape[2] == 1:
                packed = np.squeeze(packed, 2)
            packed = Image.fromarray(packed)
            packed.save(self.get_concat_image_path(map_type))
            # plt.imshow(packed)
            # plt.show()

    def simplify_wet_kitty(self):
        BODY_M = self.get_body_mesh()
        BODY_RIG = self.get_body_rig()
        select_object(BODY_M)
        if 'Wet Kitty TOON Mesh' in bpy.data.objects:
            wk = bpy.data.objects['Wet Kitty TOON Mesh']
            labia_minora = wk.data.materials.get('Labia_Minora')
            vagina = wk.data.materials.get('Vagina')
            rectum = wk.data.materials.get('Rectum')
            wk_body = None
            for m in wk.data.materials:
                if m is not None:
                    if 'body' in m.name.lower():
                        wk_body = m
                    else:
                        all_filepaths = DazOptimizer.find_body_part_textures([m])
                        NodesUtils.gen_simple_material(m.node_tree, all_filepaths)
            mesh_body = None
            for m in BODY_M.data.materials:
                if 'body' in m.name.lower():
                    mesh_body = m
            wk.material_slots[wk_body.name].material = mesh_body
            # if mesh_body is not None:
            #     wk.data.materials.clear()
            #     wk.data.materials.append(mesh_body)
            wk.data.uv_layers.active.name = NEW_WK_UV_MAP
            for l in wk.data.uv_layers:
                if not l.active:
                    l.active = True
                    l.active_render = True
                    break
            vagina.name = 'WK Vagina'
            rectum.name = 'WK Rectum'
            labia_minora.name = 'WK Labia_Minora'

    def merge_geografts(self):
        BODY_M = self.get_body_mesh()
        BODY_RIG = self.get_body_rig()
        select_object(BODY_M)

        # merge meshes
        anything = False
        for g in GEOGRAFTS:
            if g not in DICK_GEOGRAFTS:
                if g+' Mesh' in bpy.data.objects:
                    g_m = bpy.data.objects[g+' Mesh']
                    g_m.select_set(True)
                    anything = True
        if anything:
            bpy.ops.daz.merge_geografts()

            # merge bones
            for g in GEOGRAFTS:
                if g in bpy.data.objects:
                    o = bpy.data.objects[g]
                    self.merge_two_rigs(BODY_RIG, o)
                    bpy.data.objects.remove(o)
                if g+' Mesh' in bpy.data.objects:
                    o = bpy.data.objects[g+" Mesh"]
                    bpy.data.objects.remove(o)

    def transfer_morphs_to_geografts(self):
        BODY_M = self.get_body_mesh()
        # BODY_RIG = self.get_body_rig()
        select_object(BODY_M)
        if BODY_M.data.shape_keys is not None:
            # merge meshes
            #selection = [shape for shapes in MORPHS['__base__']['shapes'].values() for shape in shapes]

            for g in GEOGRAFTS:
                if g + ' Mesh' in bpy.data.objects:
                    g_m = bpy.data.objects[g + ' Mesh']
                    g_m.select_set(True)
            bpy.ops.daz.transfer_shapekeys('INVOKE_DEFAULT', bodypart='NoFace', useOverwrite=False) #, selection=selection)

    def transfer_morphs_to_eyebrows(self):
        eyebrows = self.get_eyebrows()
        if eyebrows is not None:
            BODY_M = self.get_body_mesh()
            select_object(BODY_M)
            if BODY_M.data.shape_keys is not None:
                eyebrows.select_set(True)
                bpy.ops.daz.transfer_shapekeys('INVOKE_DEFAULT', bodypart='Face', useOverwrite=False)

    def merge_two_rigs(self, original, addon):
        select_object(original)
        addon.select_set(True)
        deform_bones = {bone.name for bone in addon.data.bones if bone.use_deform}
        parents = {bone.name: bone.parent.name for bone in addon.data.bones if bone.parent is not None}

        bpy.ops.object.mode_set(mode='EDIT')
        duplicates = [bone for bone in addon.data.edit_bones if bone.name in original.data.bones]
        for dup in duplicates:
            addon.data.edit_bones.remove(dup)

        addon.select_set(True)
        original.select_set(True)
        bpy.context.view_layer.objects.active = original
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.join()
        for bone in original.data.bones:
            if bone.name in deform_bones:
                bone.use_deform = True
        bpy.ops.object.mode_set(mode='EDIT')
        for bone in original.data.edit_bones:
            parent = parents.get(bone.name)
            if parent is not None:
                bone.parent = original.data.edit_bones[parent]
        bpy.ops.object.mode_set(mode='OBJECT')

    def get_allowed_morph_prefixes(self):
        if bpy.context.scene.get('is_nirv_zero'):
            return ["BaseAnime_", "Nirv_Zero_BaseAnim_", "Nirv_zero_", "Nirv_Zero_", "Nirv_"]
        elif bpy.context.scene.get('daz_optim_toon'):
            return ["BaseAnime_"]
        return []

    def collect_fav_shape_keys(self, categories_to_include=None, profiles_to_include=USED_PROFILES)->({str:MorphMeta},{str:{}}):
        is_fem = self.is_female()
        shape_keys = {}
        figure = FIGURE_TOON if bpy.context.scene.get('daz_optim_toon') else FIGURE_G9
        for obj in bpy.data.objects:
            if isinstance(obj.data, bpy.types.Mesh):
                if not GENERATE_MORPHS_FOR_CLOTHES and is_clothes(obj):
                    continue
                if not GENERATE_MORPHS_FOR_HAIR and is_hair(obj):
                    continue
                if is_cum(obj):
                    continue
                daz_dir = obj_daz_dir(obj).lower()
                morphs_for_daz_obj = MORPHS.get(daz_dir)
                if morphs_for_daz_obj is not None:
                    shapes = morphs_for_daz_obj['shapes']
                    for key in ['unisex', 'female' if is_fem else 'male']:
                        if key in shapes:
                            for shape_key, meta in shapes[key].items():
                                if figure in meta.figure and (
                                        categories_to_include is None or meta.category in categories_to_include) and (
                                        profiles_to_include is None or meta.usage_profiles in profiles_to_include):
                                    shape_keys[shape_key] = meta
        return shape_keys

    def make_fav_morphs_list(self, categories_to_include=None, load_all_conflicting_morphs=True):
        content_dirs = get_daz_content_dirs()
        content_dirs = [d[:-1] if d.endswith("/") or d.endswith("\\") else d for d in content_dirs]
        morph_prefixes = self.get_allowed_morph_prefixes()
        morph_prefixes_regex = re.compile(r"("+"|".join(morph_prefixes)+")?(.+)\.dsf")
        shape_keys = self.collect_fav_shape_keys(categories_to_include)
        fav_morphs = {
            "filetype": "favo_morphs",
            "root_paths": content_dirs,
        }
        for obj in bpy.data.objects:
            if not isinstance(obj.data, bpy.types.Mesh):
                continue
            if not GENERATE_MORPHS_FOR_CLOTHES and is_clothes(obj):
                continue
            if not GENERATE_MORPHS_FOR_HAIR and is_hair(obj):
                continue
            if is_cum(obj):
                continue
            daz_dir = obj_daz_dir(obj)
            collected_shape_keys = {}
            for contentDir in content_dirs:
                morphs_dir_path = contentDir+daz_dir+"/Morphs"
                if os.path.isdir(morphs_dir_path):
                    for root, dirs, files in os.walk(morphs_dir_path):
                        for file in files:
                            m = morph_prefixes_regex.match(file)
                            if m:
                                prefix = m.group(1)
                                shape_key_name = m.group(2)
                                meta: MorphMeta = shape_keys.get(shape_key_name)
                                if meta is not None:
                                    if load_all_conflicting_morphs and prefix is not None:
                                        shape_key_name = prefix+shape_key_name
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
            if len(collected_shape_keys)>0:
                morphs_dict = {}
                mesh = obj.data
                mesh_url = urllib.parse.quote(obj.daz_importer.DazUrl)
                fav_morphs[mesh_url] = {
                    "finger_print": mesh.daz_importer.DazFingerPrint,
                    "morphs": morphs_dict
                }
                for shape_key_name, collected_shape_key in collected_shape_keys.items():
                    filepath = collected_shape_key['filepath']
                    meta = collected_shape_key['meta']
                    category = "Face" if meta.category.startswith("Facs") else "Custom"
                    if meta.category not in morphs_dict:
                        shapes_list = morphs_dict[meta.category] = []
                    else:
                        shapes_list = morphs_dict[meta.category]
                    shapes_list.append([filepath, shape_key_name, category])
        fav_morphs_path = self.get_fav_morphs_path()
        with open(fav_morphs_path, 'w+') as f:
            json.dump(fav_morphs, f, indent=2)

    def load_fav_morphs(self):
        mesh = self.get_body_mesh()
        select_object(mesh)
        fav_morphs_path = self.get_fav_morphs_path()
        # with open(fav_morphs_path, 'r') as f:
        #     fav_morphs = json.load(f)
        # for obj in bpy.data.objects:
        #     if not isinstance(obj.data, bpy.types.Mesh):
        #         continue
        #     if not hasattr(obj.data, 'daz_importer'):
        #         continue
        #     mesh_url = urllib.parse.quote(obj.daz_importer.DazUrl)
        #     if mesh_url in fav_morphs:
        #         print("===== loading for", repr(obj), "=====")
        #         select_object(obj)
        bpy.ops.daz.load_favo_morphs(filepath=fav_morphs_path)

    def rebind_loaded_fav_morphs(self):
        body = self.get_body_mesh()
        rig = self.get_body_rig()
        facs_regex = re.compile('.*_((facs|head|body)_(c?bs)_.+)')
        ugly_hex_regex = re.compile('(.*)-0x[a-f0-9]+')
        data_path_regex = re.compile('key_blocks\["(.*)"\].value')
        body_drivers = {}
        if body.data.shape_keys is not None:
            for fcurve in body.data.shape_keys.animation_data.drivers:
                m = data_path_regex.fullmatch(fcurve.data_path)
                if m:
                    m = m.group(1)
                    body_drivers[m] = fcurve.driver.variables[0].targets[0].data_path
        for obj in bpy.data.objects:
            mesh = obj.data
            if isinstance(mesh, bpy.types.Mesh) and mesh.shape_keys is not None:
                print(mesh.name)
                to_replace = {}
                for sk in mesh.shape_keys.key_blocks:
                    ugly_name = sk.name
                    m = ugly_hex_regex.match(ugly_name)
                    if m:
                        nice_name = m.group(1)
                        m = facs_regex.match(nice_name)
                        if m:
                            nice_name_tmp = m.group(1)
                            if nice_name_tmp not in to_replace:
                                nice_name = nice_name_tmp
                        if nice_name in to_replace:
                            print("Conflicting shape keys", to_replace[nice_name], "and", ugly_name, "for", nice_name)
                        to_replace[nice_name] = ugly_name
                for nice_name, ugly_name in to_replace.items():
                    ugly_sk = mesh.shape_keys.key_blocks[ugly_name]
                    nice_sk = mesh.shape_keys.key_blocks.get(nice_name)
                    if nice_sk is not None:
                        obj.shape_key_remove(nice_sk)
                    ugly_sk.name = nice_name
                    driver_name = nice_name.replace('_cbs_', '_bs_') if '_cbs_' in nice_name else nice_name
                    driver_path = body_drivers.get(driver_name)
                    if driver_path is not None:
                        ugly_sk.driver_remove('value')
                        driver = ugly_sk.driver_add('value').driver
                        driver.type = "SCRIPTED"
                        driver.expression = "x"
                        driver_var = driver.variables.new()
                        driver_var.name = "x"
                        driver_target = driver_var.targets[0]
                        driver_target.id_type = 'ARMATURE'
                        driver_target.id = rig.data
                        driver_target.data_path = driver_path
                        print('    ',ugly_name, "->", nice_name, ',', driver_path)
                    else:
                        print('    ',ugly_name, "->", nice_name)


        # for facs in rig.daz_importer.DazFacs:
        #     fav_facs_name = facs.name
        #     meta = shape_keys.get(facs.text)
        #     if meta is not None:
        #         m = facs_regex.fullmatch(facs.text)
        #         if m:
        #             facs_name = m.group(1)
        #             print("attempting", fav_facs_name, "->", facs_name)
        #             if facs_name in rig.daz_importer.DazFacs:
        #                 for mesh in bpy.data.meshes:
        #                     if mesh.shape_keys is not None:
        #                         sk = mesh.shape_keys.key_blocks.get(fav_facs_name)
        #                         if sk is not None:
        #                             print("rebinding", fav_facs_name, "in", mesh)
        #                             sk.driver_remove('value')
        #                             driver = sk.driver_add('value').driver
        #                             driver.type = "SCRIPTED"
        #                             driver.expression = "x"
        #                             driver_var = driver.variables.new()
        #                             driver_var.name = "x"
        #                             driver_target = driver_var.targets[0]
        #                             driver_target.id_type = 'ARMATURE'
        #                             driver_target.id = rig.data
        #                             driver_target.data_path = '["'+facs_name+'(fin)"]'

    def merge_eyes(self):
        EYES_M = self.get_eyes_mesh()
        if EYES_M is None:
            EYES_M = DazOptimizer.get_toon_floating_iris_mesh()
        if EYES_M is not None:
            eyes_layer_name = NEW_EYES_UV_MAP if NEW_EYES_UV_MAP in EYES_M.data.uv_layers else EYES_M.data.uv_layers[0].name

            rig_name = EYES_M.name[:-len(" Mesh")]
            BODY_M = self.get_body_mesh()

            BODY_RIG = self.get_body_rig()
            select_object(BODY_M)
            EYES_M.select_set(True)
            old_uv_maps = [o.name for o in EYES_M.data.uv_layers]
            # merge meshes
            bpy.ops.object.join()

            # merge UV maps
            eyes_layer = BODY_M.data.uv_layers[eyes_layer_name]
            eyes_layer_np = np.array([v.uv for v in eyes_layer.data])
            is_eye = np.all(eyes_layer_np > 0, axis=1)
            base_layer_np = self.get_base_uv_layer_np()
            base_layer_np[is_eye] = eyes_layer_np[is_eye] + [5, 0]
            self.update_base_uv_layer(base_layer_np)
            for o in old_uv_maps:
                BODY_M.data.uv_layers.remove(BODY_M.data.uv_layers[o])

            # merge bones

            if rig_name in bpy.data.objects:
                EYES_RIG = bpy.data.objects[rig_name]
                self.merge_two_rigs(BODY_RIG, EYES_RIG)

    def merge_mouth(self):
        BODY_M = self.get_body_mesh()
        MOUTH_M = self.get_mouth_mesh()
        if MOUTH_M is not None:
            mouth_rig_name = MOUTH_M.name[:-len(' Mesh')]
            BODY_RIG = self.get_body_rig()
            select_object(BODY_M)
            uv_map_name = MOUTH_M.data.uv_layers.active.name
            # merge meshes
            MOUTH_M.select_set(True)
            bpy.ops.object.join()

            # merge UV maps
            mouth_layer = BODY_M.data.uv_layers[uv_map_name]
            mouth_layer_np = np.array([v.uv for v in mouth_layer.data])
            is_mouth = np.all(mouth_layer_np > 0, axis=1)
            base_layer_np = self.get_base_uv_layer_np()
            base_layer_np[is_mouth] = mouth_layer_np[is_mouth] + [6, 0]
            self.update_base_uv_layer(base_layer_np)
            BODY_M.data.uv_layers.remove(mouth_layer)
            if mouth_rig_name in bpy.data.objects:
                # merge bones
                MOUTH_RIG = bpy.data.objects[mouth_rig_name]
                self.merge_two_rigs(BODY_RIG, MOUTH_RIG)

    @staticmethod
    def base_layer_to_pixel_coords(base_layer_np, mask_shape=MASK_SHAPE):
        pixel_coords = np.mod(base_layer_np, 1)
        pixel_coords[:, 1] = 1 - pixel_coords[:, 1]
        pixel_coords = (pixel_coords * mask_shape[0]).clip(0, mask_shape[0] - 1)
        pixel_coords = np.int32(pixel_coords)
        return pixel_coords

    def pack_uvs(self, use_full_gp):

        # ========= Concat UVs =========
        BODY_M = self.get_body_mesh()
        # pack UVs
        select_object(BODY_M)
        base_layer_np = self.get_base_uv_layer_np()
        is_arms_legs_head_body = base_layer_np[:, 0] < 4
        is_head = np.logical_and(0 < base_layer_np[:, 0], base_layer_np[:, 0] < 1)
        is_body = np.logical_and(1 < base_layer_np[:, 0], base_layer_np[:, 0] < 2)
        is_legs = np.logical_and(2 < base_layer_np[:, 0], base_layer_np[:, 0] < 3)
        is_arms = np.logical_and(3 < base_layer_np[:, 0], base_layer_np[:, 0] < 4)
        is_nails = np.logical_and(4 < base_layer_np[:, 0], base_layer_np[:, 0] < 5)
        is_eyes = np.logical_and(5 < base_layer_np[:, 0], base_layer_np[:, 0] < 6)
        is_eyes_sclera = np.logical_and(is_eyes, base_layer_np[:, 1] > 0.5)
        is_eyes_iris = np.logical_and(is_eyes, base_layer_np[:, 1] < 0.5)
        is_mouth = np.logical_and(6 < base_layer_np[:, 0], base_layer_np[:, 0] < 7)
        pixel_coords = DazOptimizer.base_layer_to_pixel_coords(base_layer_np)

        uv_mask = rle_decode(RIGHT_LEG_RLE, MASK_SHAPE)
        is_right_leg = uv_mask[pixel_coords[:, 1], pixel_coords[:, 0]]
        is_right_leg = np.logical_and(is_legs, is_right_leg)
        # is_left_leg = np.logical_and(is_legs, np.logical_not(is_right_leg))

        uv_mask = rle_decode(BOT_ARM_RLE, MASK_SHAPE)
        is_bot_arm = uv_mask[pixel_coords[:, 1], pixel_coords[:, 0]]
        is_bot_arm = np.logical_and(is_arms, is_bot_arm)
        is_top_arm = np.logical_and(is_arms, np.logical_not(is_bot_arm))

        uv_mask = rle_decode(MOUTH_CAVITY_RLE, MASK_SHAPE)
        is_mouth_cavity = uv_mask[pixel_coords[:, 1], pixel_coords[:, 0]]
        is_mouth_cavity = np.logical_and(is_head, is_mouth_cavity)

        is_floating_iris = bpy.context.scene.get('is_floating_iris')
        if is_floating_iris:
            uv_mask = rle_decode(EYE_SOCKET_RLE, MASK_SHAPE)
            is_eye_socket = uv_mask[pixel_coords[:, 1], pixel_coords[:, 0]]
            is_eye_socket = np.logical_and(is_head, is_eye_socket)

        base_layer_np[is_arms_legs_head_body] *= 0.5
        out_of_bounds = np.logical_and(1 < base_layer_np[:, 0], is_arms_legs_head_body)
        base_layer_np[out_of_bounds] += [-1, 0.5]
        gp_np = None
        is_t = bpy.context.scene.get('daz_optim_toon')
        if NEW_GP_UV_MAP in BODY_M.data.uv_layers:
            gp_layer = BODY_M.data.uv_layers[NEW_GP_UV_MAP]
            gp_layer_np = np.array([v.uv for v in gp_layer.data])
            gp_layer_np = np.mod(gp_layer_np, 1)
            is_gp = gp_layer_np[:, 0] > 0
            if is_t or not use_full_gp:
                is_outer_gp = gp_layer_np[:, 0] > 0.5
                is_gp = np.logical_and(is_gp, np.logical_not(is_outer_gp))
            gp_np = gp_layer_np[is_gp]
        elif NEW_WK_UV_MAP in BODY_M.data.uv_layers:
            gp_layer = BODY_M.data.uv_layers[NEW_WK_UV_MAP]
            gp_layer_np = np.array([v.uv for v in gp_layer.data])
            is_gp = np.logical_and(0 < gp_layer_np[:, 0], gp_layer_np[:, 0] < 1)
            gp_np = gp_layer_np[is_gp]
        eyelashes_np = None
        if NEW_TOON_EYELASHES_UV_MAP in BODY_M.data.uv_layers:
            eyelashes_layer = BODY_M.data.uv_layers[NEW_TOON_EYELASHES_UV_MAP]
            eyelashes_layer_np = np.array([v.uv for v in eyelashes_layer.data])
            is_eyelashes = eyelashes_layer_np[:, 0] > 0
            eyelashes_np = eyelashes_layer_np[is_eyelashes]
        nails_np = base_layer_np[is_nails]
        sclera_np = base_layer_np[is_eyes_sclera]
        iris_np = base_layer_np[is_eyes_iris]
        mouth_np = base_layer_np[is_mouth]


        s2 = 1
        s = 0.5
        s4 = 1/8
        s8 = 1/16
        base_layer_np[is_bot_arm] += BOT_ARM_TRANS
        base_layer_np[is_top_arm] += TOP_ARM_TRANS
        base_layer_np[is_right_leg, 1] = 1.5 - base_layer_np[is_right_leg, 1]
        base_layer_np[is_right_leg, 0] += RIGHT_LEG_TRANS
        base_layer_np[is_body] += BODY_TRANS
        base_layer_np[is_nails] = np.mod(nails_np, 1) / 8 + [s, 0]
        base_layer_np[is_eyes_sclera] = np.mod(sclera_np, 1) / 8 + [s + s4 * 1, s4 - s8]
        base_layer_np[is_eyes_iris] = np.mod(iris_np, 1) / 8 + [s + s4 * 2, s4]
        base_layer_np[is_mouth_cavity] = base_layer_np[is_mouth_cavity]/2 + MOUTH_CAVITY_SCALED_TRANS # + np.add(MOUTH_CAVITY_CENTERING_TRANS, MOUTH_CAVITY_TRANS)
        if is_floating_iris:
            eye_socket_np = base_layer_np[is_eye_socket]
            is_left_eye_socket = eye_socket_np[:, 0] < 0.25
            is_right_eye_socket = np.logical_not(is_left_eye_socket)
            eye_socket_np[is_left_eye_socket] += LEFT_EYE_SOCKET_TRANS
            eye_socket_np[is_right_eye_socket] += RIGHT_EYE_SOCKET_TRANS
            base_layer_np[is_eye_socket] = eye_socket_np
        if gp_np is not None:
            base_layer_np[is_gp] = np.mod(gp_np, 1) / 8 + [s + s4 * 2, 0]
            if not use_full_gp:
                base_layer_np[is_outer_gp] += np.array([0.5,0]) + BODY_TRANS
        base_layer_np[is_mouth] = np.mod(mouth_np, 1) / 8 + [s + s4, 0]
        if eyelashes_np is not None:
            if TRANSPARENT_TOON_EYELASHES_MAT_NAME in BODY_M.material_slots or TRANSPARENT_TOON_EYEBROWS_MAT_NAME in BODY_M.material_slots:
                base_layer_np[is_eyelashes] = np.mod(eyelashes_np, 1)
            else:
                base_layer_np[is_eyelashes] = np.mod(eyelashes_np, 1) / 8 + [s + s4 * 3, 0]
        self.update_base_uv_layer(base_layer_np)

    def separate_lip_uvs(self):
        BODY_M = self.get_body_mesh()
        # pack UVs
        select_object(BODY_M)
        bpy.ops.object.mode_set(mode='EDIT')


        bpy.context.scene.tool_settings.use_uv_select_sync = False
        bpy.ops.uv.select_all(action='DESELECT')
        bpy.ops.mesh.select_all(action='DESELECT')

        me = bpy.context.object.data
        bm = bmesh.from_edit_mesh(me)
        uv_layer = bm.loops.layers.uv.verify()

        # for v in bm.verts:
        #    v.select_set(False)
        uv_mask = rle_decode(LIP_RLE, MASK_SHAPE)
        for face in bm.faces:
            full_loop = True
            for loop in face.loops:
                loop_uv = loop[uv_layer]
                uv = np.array(loop_uv.uv)
                uv *= 2
                uv[1] = 1 - uv[1]
                pixel_coord = (uv * MASK_SHAPE[0]).clip(0, MASK_SHAPE[0] - 1)
                pixel_coord = np.int32(pixel_coord)
                matched = uv_mask[pixel_coord[1], pixel_coord[0]]
                # loop_uv.select = matched
                full_loop = full_loop and matched
                # if matched:
                #    loop.vert.select_set(True)
            face.select_set(full_loop)

        # bm.select_mode = {'VERT', 'EDGE', 'FACE'}
        bm.select_flush_mode()
        # bpy.context.tool_settings.mesh_select_mode = (False, False, True)
        bpy.ops.uv.select_all(action='SELECT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.select_split()

        bpy.ops.object.mode_set(mode='OBJECT')
        #  def separate_lips(self):
        base_layer_np = self.get_base_uv_layer_np()
        # pixel_class = get_pixel_class()
        selection = self.get_base_uv_layer_selection_np()
        base_layer_np[selection] = base_layer_np[selection] + LIP_TRANS
        self.update_base_uv_layer(base_layer_np)
        # += [0.043945, 0.006836] # top arm
        # += [-0.072266 , 0.085937] # obttom arm
        # += [0.008526, 0.019377] # torso
        # *= 0.25# nails
        # -= 0.5 # nails

    def fit_panties(self):
        clothes = find_all_panties()
        bpy.context.scene.tool_settings.transform_pivot_point = 'MEDIAN_POINT'
        bpy.context.scene.transform_orientation_slots[0].type = 'GLOBAL'
        for c in clothes:
            s = 1+c.skin_tight
            s = (s, s, s)
            scale_in_edit_mode(c.obj, s)
        bpy.ops.object.mode_set(mode='OBJECT')

    def fit_clothes(self):
        body = self.get_body_mesh()
        select_object(body)
        remove_shape_key(body, EXTRUDED_SK_NAME)
        sk = body.shape_key_add(name=EXTRUDED_SK_NAME, from_mix=False)
        sk_idx = body.data.shape_keys.key_blocks.find(sk.name)
        body.active_shape_key_index = sk_idx
        sk.value = 1
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.transform.shrink_fatten(value=CLOTHES_MIN_DIST_TO_SKIN/bpy.context.scene.unit_settings.scale_length)
        bpy.ops.object.mode_set(mode='OBJECT')
        sk.value = 0
        clothes = find_all_non_skin_tight_clothes()
        for c in clothes:
            hide_object(c.obj, False)
            c.obj.select_set(True)
        bpy.ops.daz.transfer_shapekeys('INVOKE_DEFAULT', bodypart='NoFace', filter=EXTRUDED_SK_NAME, useOverwrite=False)

    def bind_clothes_to_extrude(self):
        body = self.get_body_mesh()
        select_object(body)
        remove_shape_key(body, EXTRUDED_SK_NAME)
        clothes = find_all_non_skin_tight_clothes()
        for c in clothes:
            sk = c.obj.data.shape_keys.key_blocks['extruded']
            sk.slider_max = 10
            sk.driver_remove('value')
            driver = sk.driver_add('value').driver
            driver.type = "SCRIPTED"
            driver.expression = "extruded_var"
            driver_var = driver.variables.new()
            driver_var.name = "extruded_var"
            driver_target = driver_var.targets[0]
            driver_target.id_type = 'SCENE'
            driver_target.id = bpy.context.scene
            driver_target.data_path = 'clothes_displacement'

        #
        # bind_to_objects(body, clothes, 'bind extruded')
        # sk.value = 1


    def apply_fit_clothes(self):
        body = self.get_body_mesh()
        remove_shape_key(body, EXTRUDED_SK_NAME)
        for c in find_all_non_skin_tight_clothes():
            select_object(c.obj)
            bpy.ops.object.shape_key_remove(all=True, apply_mix=True)

    def fit_skin_tight_clothes(self):
        BODY_M = self.get_body_mesh()
        m_name = 'FitSkinTightClothes'
        for meta in find_all_skin_tight_clothes():
            obj = meta.obj
            if m_name not in obj.modifiers:
                # m_len = len(obj.modifiers)
                m = obj.modifiers.new(name=m_name, type="SHRINKWRAP")
                m.target = BODY_M
                m.offset = meta.skin_tight
                m.wrap_mode = 'OUTSIDE'
                driver = m.driver_add('offset').driver
                driver.type = "SCRIPTED"
                driver.expression = "skin_tight_var+"+str(meta.skin_tight)
                driver_var = driver.variables.new()
                driver_var.name = "skin_tight_var"
                driver_target = driver_var.targets[0]
                driver_target.id_type = 'SCENE'
                driver_target.id = bpy.context.scene
                driver_target.data_path = 'skin_tight_displacement'
                # select_object(obj)
                # for _ in range(m_len):
                #     bpy.ops.object.modifier_move_up(modifier=m.name)

    def apply_fit_skin_tight_clothes(self):
        for obj in bpy.data.objects:
            if 'FitSkinTightClothes' in obj.modifiers:
                select_object(obj)
                if obj.data.shape_keys is not None and len(obj.data.shape_keys.key_blocks)>0:
                    bpy.ops.object.shape_key_remove(all=True, apply_mix=False)
                bpy.ops.object.modifier_apply(modifier='FitSkinTightClothes')

    def subdivide_breast_bones(self, cuts = 2):
        BODY_M = self.get_body_mesh()
        BODY_RIG = self.get_body_rig()
        vgs = subdivide_bone(cuts, BODY_M, BODY_RIG,'r_pectoral')
        vgs.extend(subdivide_bone(cuts, BODY_M, BODY_RIG,'l_pectoral'))
        vgs = [v.name for v in vgs]
        grafts = []
        for o_name in BREAST_GEOGRAFTS:
            o_name = o_name + ' Mesh'
            if o_name in bpy.data.objects:
                grafts.append(bpy.data.objects[o_name])
        transfer_weights(BODY_M, grafts, vgs)


    def save_textures(self):
        BODY_M = self.get_body_mesh()
        bpy.ops.object.select_all(action='DESELECT')
        BODY_M.select_set(True)
        bpy.context.view_layer.objects.active = BODY_M
        tex_dir = self.textures_dir()
        #if os.path.exists(tex_dir):
        #    shutil.rmtree(tex_dir)
        bpy.ops.daz.save_local_textures()

    def select_gp(self):
        mesh = bpy.data.objects['GoldenPalace_G9 Mesh']
        select_object(mesh)
        return mesh

    def select_body(self):
        mesh = self.get_body_mesh()
        select_object(mesh)
        return mesh

    def get_gp_or_body(self):
        if 'GoldenPalace_G9 Mesh' in bpy.data.objects:
            mesh = bpy.data.objects['GoldenPalace_G9 Mesh']
        else:
            mesh = self.get_body_mesh()
        return mesh

    def select_gp_or_body(self):
        mesh = self.get_gp_or_body()
        select_object(mesh)
        return mesh

    def unify_golden_palace_uvs(self):
        mesh = self.select_gp_or_body()
        if NEW_GP_UV_MAP not in mesh.data.uv_layers:
            gp_labia_majora = mesh.data.uv_layers.get('Golden Palace 2')
            gp_labia_minora = mesh.data.uv_layers['Golden Palace']
            new_uv_layer = mesh.data.uv_layers.new(name=NEW_GP_UV_MAP)
            gp_labia_minora_np = np.array([v.uv for v in gp_labia_minora.data])
            if gp_labia_majora is None:
                new_uv_layer_np = gp_labia_minora_np
            else:
                gp_labia_majora.active = True
                new_uv_layer_np = np.array([v.uv for v in new_uv_layer.data])
                gp_labia_majora_np = np.array([v.uv for v in gp_labia_majora.data])
                is_majora = np.all(gp_labia_majora_np > 0, axis=1)
                is_minora = np.all(gp_labia_minora_np > 0, axis=1)
                gp_labia_majora_np = np.mod(gp_labia_majora_np, 1)
                gp_labia_minora_np = np.mod(gp_labia_minora_np, 1)
                is_labia_majora = np.logical_and(0.285 < gp_labia_majora_np[:, 0], gp_labia_majora_np[:, 0] < 0.72)
                vagina_symmetry_line = 0.26598
                vagina_extent = 0.47191
                vagina_half_width = vagina_extent - vagina_symmetry_line
                p1A = np.array([0.31444 - vagina_symmetry_line, 0.912306])
                p2A = np.array([0.4164 - vagina_symmetry_line, 0.545495])
                vag_distance = np.absolute(gp_labia_minora_np[:, 0] - vagina_symmetry_line)
                slopeA = (p2A[1] - p1A[1]) / (p2A[0] - p1A[0])
                # p1[2] = p1[0] * slope + offset
                # p1[2] - p1[0] * slope = offset
                offsetA = p1A[1] - p1A[0] * slopeA
                offsetA += 0.05  # just for a good measure to avoid errors due to floating point precision
                p1B = np.array([0.43189 - vagina_symmetry_line, 0.317139])
                p2B = np.array([0.40303 - vagina_symmetry_line, 0.217741])
                slopeB = (p2B[1] - p1B[1]) / (p2B[0] - p1B[0])
                offsetB = p1B[1] - p1B[0] * slopeB
                offsetB -= 0.1  # just for a good measure to avoid errors due to floating point precision
                is_labia_minora = np.logical_and(vag_distance * slopeB + offsetB < gp_labia_minora_np[:, 1],
                                                 gp_labia_minora_np[:, 1] < vag_distance * slopeA + offsetA)
                is_labia_majora = np.logical_and(is_labia_majora, np.logical_not(is_labia_minora))
                is_anus = np.logical_and(is_minora, is_majora)
                is_anus = np.logical_and(is_anus, np.logical_not(np.logical_or(is_labia_majora, is_labia_minora)))
                p1 = (0.514551, 0.546842)  # point on circle boundary
                p2 = (0.499947, 0.550254)  # center
                p3 = (0.500005, 0.588271)  # oval top point
                vag_oval_longer_radius = np.linalg.norm(np.subtract(p3, p2))
                vag_radius = np.linalg.norm(np.subtract(p1, p2))
                vag_distance = gp_labia_majora_np - p2
                vag_distance[:, 1] *= vag_radius / vag_oval_longer_radius
                is_vagina = np.linalg.norm(vag_distance, axis=1) < vag_radius
                is_vagina = np.logical_and(is_vagina, np.logical_not(is_labia_minora))
                is_insides = np.logical_or(is_vagina, is_anus)
                new_uv_layer_np[:, :] = 0
                vagina_margin = 0.08
                new_uv_layer_np[is_labia_minora] = gp_labia_minora_np[is_labia_minora] - [vagina_margin, 0]
                new_uv_layer_np[is_labia_majora] = gp_labia_majora_np[is_labia_majora] + [1 - 0.72, 0]
                new_uv_layer_np[is_insides] = gp_labia_minora_np[is_insides] * (1 / 8) + [vagina_half_width * 2 - vagina_margin, 0]
            for v, new_uv in zip(new_uv_layer.data, new_uv_layer_np):
                v.uv = new_uv

    def simplify_golden_palace_material(self):
        mesh = self.select_gp_or_body()
        filepaths = {}
        for channel in ['Base Color', 'Roughness', 'Normal']:
            name = 'GP_Baked_' + channel
            if name in bpy.data.images:
                filepaths[channel] = bpy.data.images[name]
            else:
                p = os.path.join(self.workdir, self.name + "_" + channel + '_gp_baked.png')
                if os.path.exists(p):
                    filepaths[channel] = bpy.data.images.load(p)
        for mat in mesh.data.materials:
            if mat.name.startswith("GP_"):
                mat.node_tree.nodes.clear()
                NodesUtils.gen_simple_material(mat.node_tree, filepaths, uvs=NEW_GP_UV_MAP)


    def setup_golden_palace_for_baking(self):
        mesh = self.select_gp_or_body()

        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.device = 'GPU'
        bpy.context.scene.view_settings.view_transform = 'Standard'

        mesh.data.uv_layers[NEW_GP_UV_MAP].active = True
        baked_gp_imgs = {}
        for idx, channel in enumerate(['Base Color', 'Roughness', 'Normal']):
            name = 'GP_Baked_'+channel
            if 'GP_Baked' in bpy.data.images:
                baked_gp_img = bpy.data.images[name]
            else:
                gp_baked_path = os.path.join(self.workdir, self.name + "_" + channel + '_gp_baked.png')
                if os.path.exists(gp_baked_path):
                    baked_gp_img = bpy.data.images.load(gp_baked_path)
                    baked_gp_img.name = name
                else:
                    baked_gp_img = bpy.data.images.new(name, 1024 * 4, 1024 * 4)
                baked_gp_img.colorspace_settings.name = 'sRGB' if channel == 'Base Color' else 'Non-Color'
            baked_gp_imgs[channel] = baked_gp_img
        for mat in mesh.data.materials:
            if mat.name.startswith('GP_'):
                n = mat.node_tree.nodes
                l = mat.node_tree.links
                uv_map = n.new('ShaderNodeUVMap')
                uv_map.location = (-300, 200)
                uv_map.uv_map = NEW_GP_UV_MAP
                uv_map.name = 'GP_Baked_UVs'
                bsdf_node = n['simple_material_bsdf']
                diffuse_node = n.new('ShaderNodeBsdfDiffuse')
                diffuse_node.location = bsdf_node.location
                diffuse_node.location.x -= 300
                diffuse_node.location.y += 200
                diffuse_node.name = 'simple_material_diffuse'
                for idx, channel in enumerate(['Base Color', 'Roughness', 'Normal']):
                    bsdf_links = bsdf_node.inputs[channel].links
                    if len(bsdf_links)==0:
                        bpy.context.scene['gp_lacks_'+channel]=True
                    else:
                        before_bsdf = bsdf_links[0].from_socket
                        diffuse_socket_name = 'Color' if channel == 'Base Color' else channel
                        l.new(diffuse_node.inputs[diffuse_socket_name], before_bsdf)

                        name = 'GP_Baked_' + channel
                        target_texture = n.new('ShaderNodeTexImage')
                        target_texture.image = baked_gp_imgs[channel]
                        target_texture.name = name+' Texture'
                        target_texture.location = (0, 200 + 300 * idx)
                        l.new(target_texture.inputs['Vector'], uv_map.outputs['UV'])
                        n.active = target_texture
                l.new(bsdf_node.outputs['BSDF'].links[0].to_socket, diffuse_node.outputs['BSDF'])


    def select_golden_palace_for_bsdf_mode_baking(self, principled_bsdf):
        mesh = self.select_gp_or_body()
        for mat in mesh.data.materials:
            if mat.name.startswith('GP_'):
                n = mat.node_tree.nodes
                l = mat.node_tree.links
                bsdf_node = n['simple_material_bsdf']
                diffuse_node = n['simple_material_diffuse']
                good_node, bad_node = (bsdf_node, diffuse_node) if principled_bsdf else (diffuse_node, bsdf_node)
                bad_links = bad_node.outputs['BSDF'].links
                if len(bad_links) > 0:
                    l.new(bad_links[0].to_socket, good_node.outputs['BSDF'])


    def select_gp_color_for_baking(self):
        bpy.context.scene.cycles.bake_type = 'DIFFUSE'
        bpy.context.scene.render.bake.use_pass_direct = False
        bpy.context.scene.render.bake.use_pass_indirect = False
        self.select_golden_palace_for_baking('Base Color')
        self.select_golden_palace_for_bsdf_mode_baking(principled_bsdf=False)

    def select_gp_normals_for_baking(self):
        bpy.context.scene.cycles.bake_type = 'NORMAL'
        self.select_golden_palace_for_baking('Normal')
        self.select_golden_palace_for_bsdf_mode_baking(principled_bsdf=True)

    def select_gp_roughness_for_baking(self):
        bpy.context.scene.cycles.bake_type = 'ROUGHNESS'
        self.select_golden_palace_for_baking('Roughness')
        self.select_golden_palace_for_bsdf_mode_baking(principled_bsdf=True)

    def select_golden_palace_for_baking(self, channel):
        mesh = self.select_gp_or_body()
        for mat in mesh.data.materials:
            if mat.name.startswith('GP_'):
                n = mat.node_tree.nodes
                l = mat.node_tree.links
                target_texture = n.get('GP_Baked_' + channel+' Texture')
                if target_texture is not None:
                    target_texture.select = True
                    n.active = target_texture
                    # color_texture = n['simple_material_Base Color']
                    # bsdf_node = n['simple_material_bsdf']
                    # bsdf_out = bsdf_node.outputs['BSDF']
                    # color_out = color_texture.outputs['Color']
                    # is_bsdf_connected = len(bsdf_out.links)>0
                    # if channel == 'Base Color':
                    #     if is_bsdf_connected:
                    #         after_bsdf = bsdf_out.links[0].to_socket
                    #         l.remove(bsdf_out.links[0])
                    #         l.new(after_bsdf, color_out)
                    # else:
                    #     if not is_bsdf_connected:
                    #         for color_link in color_out.links:
                    #             if color_link.to_node != bsdf_node:
                    #                 after_bsdf = color_link.to_socket
                    #                 l.remove(color_link)
                    #                 l.new(after_bsdf, bsdf_out)
                    #                 break


    def get_gp_texture_path(self, channel):
        return os.path.join(self.workdir, self.name + "_" + channel + '_gp_baked.png')

    def save_gp_textures(self):

        for channel in ['Base Color', 'Roughness', 'Normal']:
            if not bpy.context.scene.get('gp_lacks_'+channel):
                gp_baked_path = self.get_gp_texture_path(channel)
                name = 'GP_Baked_' + channel
                if name in bpy.data.images:
                    bpy.data.images[name].save(filepath=gp_baked_path)
                    bpy.data.images[name].filepath = gp_baked_path


    def make_single_material(self):
        body_m = self.select_body()
        exceptions = ["Facial hair", TRANSPARENT_TOON_EYEBROWS_MAT_NAME, TRANSPARENT_TOON_EYELASHES_MAT_NAME]
        mat = NodesUtils.remove_all_mats(body_m, "UnifiedSkin", excpt=exceptions)
        filepaths = {}
        for channel in ['Base Color', 'Roughness', 'Normal']:
            fp = self.get_concat_image_path(channel)
            filepaths[channel] = [fp] if os.path.exists(fp) else []
        print("unified filepaths", filepaths)
        NodesUtils.gen_simple_material(mat.node_tree, filepaths)
        old_uv_maps = [o.name for o in body_m.data.uv_layers]
        base_uv_layer_name = 'Base Multi UDIM'
        for uv_layer_name in old_uv_maps:
            if uv_layer_name != base_uv_layer_name:
                l = body_m.data.uv_layers[uv_layer_name]
                body_m.data.uv_layers.remove(l)
        for e in exceptions:
            e_mat = body_m.material_slots.get(e)
            if e_mat is not None:
                nt = e_mat.material.node_tree
                for uv_node in NodesUtils.find_all_by_type(nt, bpy.types.ShaderNodeUVMap):
                    nt.nodes.remove(uv_node)
        for g in DICK_GEOGRAFTS:
            if g+' Mesh' in bpy.data.objects:
                dick = bpy.data.objects[g+' Mesh']
                bpy.context.view_layer.objects.active = dick
                l = dick.data.uv_layers[0]
                l.active = True
                l_np = np.array([v.uv for v in l.data])
                l_np = l_np / 8 + [0.75, 0]
                for v, new_uv in zip(l.data, l_np):
                    v.uv = new_uv
                bpy.context.object.data.materials.clear()
                dick.data.materials.append(mat)


    def add_thigh_bones(self):
        body_rig = self.get_body_rig()
        select_object(body_rig)
        bpy.ops.object.mode_set(mode='EDIT')
        r_thigh = body_rig.data.edit_bones['r_thigh']
        l_thigh = body_rig.data.edit_bones['l_thigh']
        l_thigh_jiggle = body_rig.data.edit_bones.new('l_thigh_jiggle')
        r_thigh_jiggle = body_rig.data.edit_bones.new('r_thigh_jiggle')
        l_thigh_jiggle.parent = l_thigh
        r_thigh_jiggle.parent = r_thigh
        p = 0.75
        d = 0.05
        l_thigh_jiggle.head = np.array(l_thigh.head)*p+np.array(l_thigh.tail)*(1-p)
        r_thigh_jiggle.head = np.array(r_thigh.head)*p+np.array(r_thigh.tail)*(1-p)
        r_thigh_jiggle.tail = r_thigh_jiggle.head
        r_thigh_jiggle.tail.y += d
        l_thigh_jiggle.tail = l_thigh_jiggle.head
        l_thigh_jiggle.tail.y += d

        body_mesh = self.get_body_mesh()
        select_object(body_mesh)
        bpy.ops.object.mode_set(mode='OBJECT')

        r_thigh = body_mesh.vertex_groups['r_thightwist1']
        l_thigh = body_mesh.vertex_groups['l_thightwist1']
        r_thigh_jiggle = body_mesh.vertex_groups.new(name='r_thigh_jiggle')
        l_thigh_jiggle = body_mesh.vertex_groups.new(name='l_thigh_jiggle')
        r_thigh_idx = r_thigh.index
        l_thigh_idx = l_thigh.index
        epsilon = 0.001
        diff = 0.2
        for idx, vert in enumerate(body_mesh.data.vertices):
            for g in vert.groups:
                if g.group == r_thigh_idx:
                    w = g.weight
                    if w > diff+epsilon:
                        r_thigh_jiggle.add(index=(idx,), weight=w-diff, type='REPLACE')
                elif g.group == l_thigh_idx:
                    w = g.weight
                    if w > diff + epsilon:
                        l_thigh_jiggle.add(index=(idx,), weight=w - diff, type='REPLACE')

    def apply_additional_bone(self, bone_names):
        body_mesh = self.get_body_mesh()
        apply_additional_bone(body_mesh, bone_names)

    def add_double_small_glute_bones(self):
        self.apply_additional_bone(['r_glute', 'r_glute2', 'l_glute', 'l_glute2'])

    def add_high_thigh_jiggle(self):
        self.apply_additional_bone(['l_thigh_jiggle', 'r_thigh_jiggle'])

    def add_low_thigh_jiggle(self):
        self.apply_additional_bone(['l_thigh_jiggle2', 'r_thigh_jiggle2'])

    def add_side_thigh_jiggle(self):
        self.apply_additional_bone(['l_thigh_jiggle_side', 'r_thigh_jiggle_side'])

    def add_single_big_glute_bones(self):
        body_mesh = self.get_body_mesh()
        body_rig = self.get_body_rig()
        select_object(body_rig)
        bpy.ops.object.mode_set(mode='EDIT')
        pelvis = body_rig.data.edit_bones['pelvis']
        r_thigh = body_rig.data.edit_bones['r_thigh']
        l_thigh = body_rig.data.edit_bones['l_thigh']
        l_glute = body_rig.data.edit_bones.new('l_glute')
        r_glute = body_rig.data.edit_bones.new('r_glute')
        l_glute.parent = pelvis
        r_glute.parent = pelvis
        l_glute.head = l_thigh.head
        r_glute.head = r_thigh.head
        l_glute.tail = l_thigh.head
        r_glute.tail = r_thigh.head
        dy = 0.05
        dz = -0.01
        r_glute.tail.y += dy
        l_glute.tail.y += dy
        r_glute.tail.z += dz
        l_glute.tail.z += dz

        select_object(body_mesh)

        bpy.ops.object.mode_set(mode='EDIT')

        bpy.context.scene.tool_settings.use_uv_select_sync = False
        me = body_mesh.data
        bm = bmesh.from_edit_mesh(me)
        uv_layer = bm.loops.layers.uv.verify()
        uv_mask = rle_decode(BUTT_RLE, MASK_SHAPE)
        vertex_mask = np.zeros((len(me.vertices), 2), dtype=np.float32)
        for face in bm.faces:
            for loop in face.loops:
                loop_uv = loop[uv_layer]
                uv = np.array(loop_uv.uv)
                if 1 < uv[0] < 2:
                    uv[0] -= 1
                    uv2 = np.array([uv[0], 1-uv[1]])
                    pixel_coord = (uv2 * MASK_SHAPE[0]).clip(0, MASK_SHAPE[0] - 1)
                    pixel_coord = np.int32(pixel_coord)
                    matched = uv_mask[pixel_coord[1], pixel_coord[0]]
                    if matched:
                        vertex_mask[loop.vert.index] = uv
                    else:
                        vertex_mask[loop.vert.index] = -uv
        bpy.ops.object.mode_set(mode='OBJECT')
        l_glute_group = body_mesh.vertex_groups.new(name="l_glute")
        r_glute_group = body_mesh.vertex_groups.new(name="r_glute")

        is_left = vertex_mask[:, 0] > 0.5
        is_cheek = vertex_mask[:, 1] > 0
        is_left_cheek = np.logical_and(is_left, is_cheek)
        is_right_cheek = np.logical_and(np.logical_not(is_left), is_cheek)
        r_cheek = vertex_mask[is_right_cheek]
        l_cheek = vertex_mask[is_left_cheek]
        r_center = (0.12788, 0.27)
        l_center = (0.87212, 0.27)
        r_cheek = np.linalg.norm(r_cheek - r_center, axis=1)
        l_cheek = np.linalg.norm(l_cheek - l_center, axis=1)
        max_radius = 0.13
        r_cheek = 1 - r_cheek/max_radius
        l_cheek = 1 - l_cheek/max_radius
        l_cheek_indices, = np.where(is_left_cheek)
        r_cheek_indices, = np.where(is_right_cheek)
        for val, idx in zip(l_cheek.tolist(), l_cheek_indices.tolist()):
            l_glute_group.add(index=(idx,), weight=val, type='REPLACE')
        for val, idx in zip(r_cheek.tolist(), r_cheek_indices.tolist()):
            r_glute_group.add(index=(idx,), weight=val, type='REPLACE')


    def subdivide_glute_bones(self, cuts=2):
        BODY_M = self.get_body_mesh()
        BODY_RIG = self.get_body_rig()
        subdivide_bone(cuts, BODY_M, BODY_RIG, 'l_glute')
        subdivide_bone(cuts, BODY_M, BODY_RIG, 'r_glute')


    def transfer_morphs_to_clothes(self):
        BODY_M = self.get_body_mesh()
        select_object(BODY_M)
        clothes = find_all_clothes()
        for c in clothes:
            hide_object(c.obj, False)
            c.obj.select_set(True)
        bpy.ops.daz.transfer_shapekeys('INVOKE_DEFAULT', bodypart='NoFace', useOverwrite=False)

    def transfer_morphs_to_cum(self):
        BODY_M = self.get_body_mesh()
        select_object(BODY_M)
        for c in find_cum():
            hide_object(c, False)
            c.select_set(True)
        bpy.ops.daz.transfer_shapekeys('INVOKE_DEFAULT', bodypart='NoFace', useOverwrite=False)

    def clean_up_morphs(self):
        pass

    def get_missing_bones(self):
        BODY_M = self.get_body_mesh()
        groups = set(b for b in ADDITIONAL_BONES.keys() if b in BODY_M.vertex_groups)
        def add_subdivided(name):
            i = 1
            while True:
                group = 'l_' + name + str(i)
                if group in BODY_M.vertex_groups:
                    groups.add(group)
                    groups.add('r_' + name + str(i))
                    i += 1
                else:
                    break

        if 'l_glute' in BODY_M.vertex_groups:
            add_subdivided('glute')
        add_subdivided('pectoral')
        return groups

    def rig_physics_bones(self):

        class AddBoneChainAlongEdgeLoop:
            def __init__(self, first_edge, first_vertex):
                self.first_edge = first_edge
                self.first_vertex = first_vertex


        additional_clothing_bones = {
            'Soaring Dragon Kung Fu Dress': AddBoneChainAlongEdgeLoop(16416, 2700),
        }

    def rig_physics_hair(self):


        class AnimeHair:

            def __init__(self, num_of_bones):
                self.num_of_bones = num_of_bones

            def run(self, mesh, rig):
                print("Rigging ",mesh," on ",rig)
                import scipy
                select_object(mesh)
                bpy.ops.object.mode_set(mode='EDIT')
                bpy.context.scene.tool_settings.use_uv_select_sync = False
                bpy.ops.mesh.select_all(action='DESELECT')
                me = mesh.data
                bm = bmesh.from_edit_mesh(me)
                bm.verts.ensure_lookup_table()
                dcomps, dcomps_num = get_disconnected_components(bm)
                sharp_ends_per_dcomp = [None] * dcomps_num
                for f in bm.faces:
                    if len(f.verts)==4:
                        vert = f.verts[0]
                        vert_idx = vert.index
                        dcomp_idx = dcomps[vert_idx]
                        if sharp_ends_per_dcomp[dcomp_idx] is None:
                            longest_edge_loop = None
                            longest_edge_loop_length = 0
                            for edge in f.edges:
                                if vert in edge.verts:
                                    edge_loop, is_circular = collect_edge_loop(edge)
                                    edge_loop_l = edge_loop_length(edge_loop)
                                    if not is_circular:
                                        edge_loop_l *= 2
                                    if longest_edge_loop_length < edge_loop_l:
                                        longest_edge_loop_length = edge_loop_l
                                        longest_edge_loop = edge_loop
                            if longest_edge_loop is not None:
                                # select_edge_loop(longest_edge_loop)
                                sharp_ends = edge_loop_find_sharp_ends(longest_edge_loop)
                                sharp_ends_per_dcomp[dcomp_idx] = sharp_ends
                                #select_edge_loop(sharp_ends)
                sharp_end_co_per_dcomp = np.array([end.co for ends in sharp_ends_per_dcomp for end in ends])
                point_count = len(sharp_end_co_per_dcomp)
                half_point_count = point_count/2

                def get_the_one_further_from_closer_to_most_points(e1, e2):
                    dist1 = np.linalg.norm(sharp_end_co_per_dcomp - e1.co, axis=1)
                    dist2 = np.linalg.norm(sharp_end_co_per_dcomp - e2.co, axis=1)
                    closer_to_e1 = dist1 < dist2
                    is_most_closer_to_e1 = closer_to_e1.sum()>half_point_count
                    return e2 if is_most_closer_to_e1 else e1


                hair_ends = [get_the_one_further_from_closer_to_most_points(end1, end2) for end1, end2 in sharp_ends_per_dcomp]
                hair_ends_coords = np.array([e.co for e in hair_ends])
                #select_edge_loop(hair_ends)
                _, cluster_labels = scipy.cluster.vq.kmeans2(hair_ends_coords, self.num_of_bones)
                verts = np.array([e.co for e in bm.verts])
                dcomps_sums = sum_vectors_in_dcomps(verts, dcomps, dcomps_num)
                dcomps_counts = count_vectors_in_dcomps(dcomps, dcomps_num)
                cluster_sums = np.zeros((self.num_of_bones,3))
                cluster_counts = np.bincount(cluster_labels, weights=dcomps_counts, minlength=self.num_of_bones)
                hair_ends_sum = np.zeros((self.num_of_bones, 3))
                hair_ends_count = np.zeros(self.num_of_bones)
                for label, dcomps_sum, hair_end_coords in zip(cluster_labels, dcomps_sums.T, hair_ends_coords):
                    cluster_sums[label] += dcomps_sum
                    hair_ends_sum[label] += hair_end_coords
                    hair_ends_count[label] += 1
                hair_ends_averaged = np.divide(hair_ends_sum.T, hair_ends_count).T
                controids = np.divide(cluster_sums.T, cluster_counts).T
                bm.select_flush(True)
                bmesh.update_edit_mesh(me)

                select_object(rig)
                bpy.ops.object.mode_set(mode='EDIT')
                head_bone = rig.data.edit_bones.get('head')
                for i, (hair_end_averaged, controid) in enumerate(zip(hair_ends_averaged, controids)):
                    hair_bone = rig.data.edit_bones.new('hair_'+str(i))
                    hair_bone.head = controid
                    hair_bone.tail = hair_end_averaged
                    hair_bone.use_connect = False
                    hair_bone.parent = head_bone
                #edge_lengths = np.array([e.calc_length() for e in bm.edges])
                # edge_vert0_indices = np.array([e.verts[0].index for e in bm.edges])
                # edge_vert1_indices = np.array([e.verts[1].index for e in bm.edges])

                # edge_vectors = verts[edge_vert1_indices]-verts[edge_vert0_indices]
                # edge_dcomps = dcomps[edge_vert0_indices]

                #average_vector = sum_vectors_in_dcomps(edge_vectors, edge_dcomps, dcomps_num)
                # k_means = scipy.cluster.vq.kmeans(centroids, self.num_of_bones)





        additional_hair_bones = {
            #'Toon Style Side Part Bob Hair': AddBoneChainAlongEdgeLoop(16416, 2700),
            #'MB Avani Toon Character for Genesis 9': AnimeHair(8),
            'MAH Mirai Anime Hair': AnimeHair(8),
        }
        for hair in find_all_hair():
            rig = get_rig_of(hair)
            method = additional_hair_bones.get(rig.name)
            if method is not None:
                method.run(hair, rig)

    def transfer_missing_bones_to_clothes(self):
        BODY_M = self.get_body_mesh()
        groups = self.get_missing_bones()
        clothes = [c.obj for c in find_all_clothes()]
        transfer_weights(BODY_M, clothes, groups)

    def transfer_missing_bones_to_cum(self):
        BODY_M = self.get_body_mesh()
        groups = self.get_missing_bones()
        cum = find_cum()
        transfer_weights(BODY_M, cum, groups)

    def is_female(self):
        return bool(bpy.context.scene['daz_optim_female'])

    def get_hierarchy(self):
        quinn = self.is_female()
        return UE5_QUINN_BONE_HIERARCHY if quinn else UE5_MANNY_BONE_HIERARCHY

    def compare_daz_to_ue5_skeleton(self):
        body_rig = self.get_body_rig()
        select_object(body_rig)
        bpy.ops.object.mode_set(mode='OBJECT')
        print(str(body_rig.data))
        hierarchy = self.get_hierarchy()
        for bone in body_rig.data.bones:
            if bone.name in hierarchy:
                parent = hierarchy[bone.name][-1]
                if bone.parent is None:
                    match = parent is None
                    if not match:
                        print(bone.name + ".parent == None != " + str(parent))
                else:
                    match = parent == bone.parent.name
                    if not match:
                        print(bone.name + ".parent == ", bone.parent.name + " != " + str(parent))

    def convert_daz_to_ue5_skeleton(self):
        import mathutils
        body_rig = self.get_body_rig()
        body_mesh = self.get_body_mesh()
        hierarchy = self.get_hierarchy()
        def convert_rig(rig):
            select_object(rig)
            bpy.ops.object.mode_set(mode='EDIT')
            pelvis = rig.data.edit_bones.get('pelvis')
            hip = rig.data.edit_bones.get('hip')
            if hip is not None and pelvis is not None:
                pelvis_children = list(pelvis.children)
                for c in pelvis_children:
                    c.parent = hip
            if 'spine1' in rig.data.edit_bones and pelvis is not None:
                rig.data.edit_bones['spine1'].parent = pelvis
            for daz_name, ue5_name in DAZ_G9_TO_UE5_BONES.items():
                if daz_name in rig.data.edit_bones:
                    bone = rig.data.edit_bones[daz_name]
                    bone.name = bone.name + "_tmp_suffix"
            for daz_name, ue5_name in DAZ_G9_TO_UE5_BONES.items():
                daz_name = daz_name + "_tmp_suffix"
                if daz_name in rig.data.edit_bones:
                    bone = rig.data.edit_bones[daz_name]
                    bone.name = ue5_name
            for bone_name in ['pelvis', 'spine_01', 'spine_02', 'spine_03', 'spine_04', 'spine_05', 'neck_01', 'neck_02']: #
                if bone_name in rig.data.edit_bones:
                    daz_bone = rig.data.edit_bones.get(bone_name)
                    ue5_bone = hierarchy[bone_name][0]
                    ue5_bone = mathutils.Vector(ue5_bone)/100
                    old_daz_bone = daz_bone.head.copy()
                    daz_bone.head = ue5_bone
                    daz_bone.tail += ue5_bone-old_daz_bone
            bpy.ops.object.mode_set(mode='POSE')
            for bone in rig.data.bones:
                bone.inherit_scale = 'FULL'

            # r_thigh = rig.data.edit_bones.get('thigh_r')
            # spine_01 = rig.data.edit_bones.get('spine_01')
            # if r_thigh is not None and pelvis is not None:
            #     new_z = r_thigh.head.z + ue5_pelvis_height
            #     old_z = pelvis.head.z
            #     pelvis.head.z = new_z
            #     pelvis.tail.z += (new_z-old_z)
            # if spine_01 is not None and pelvis is not None:
            #     new_z = pelvis.head.z + ue5_spine_01_height
            #     old_z = spine_01.head.z
            #     spine_01.head.z = new_z
            #     spine_01.tail.z += (new_z-old_z)

        convert_rig(body_rig)
        # body_rig.name = 'root'
        # body_mesh.name = 'root Mesh'
        children = list(body_rig.children)
        while len(children)>0:
            o = children.pop()
            if isinstance(o.data, bpy.types.Armature):
                convert_rig(o)
            children.extend(o.children)
        for obj in bpy.data.objects:
            if isinstance(obj.data, bpy.types.Mesh):
                spine1 = obj.vertex_groups.get('spine_01')
                if spine1 is not None:
                    spine1.name = 'pelvis'

    @staticmethod
    def get_gp_mesh():
        gp = bpy.data.objects.get('GoldenPalace_G9 Mesh')
        return gp

    def remove_tentacles(self):
        gp_mesh = self.get_gp_or_body()
        gp_rig = get_rig_of(gp_mesh)
        select_object(gp_rig)
        tentacle_bones = [bone.name for bone in gp_rig.data.bones if 'tentacle' in bone.name.lower()]
        select_object(gp_mesh)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.context.scene.tool_settings.use_uv_select_sync = False
        bpy.ops.uv.select_all(action='DESELECT')
        bpy.ops.mesh.select_all(action='DESELECT')
        me = bpy.context.object.data
        bm = bmesh.from_edit_mesh(me)
        bm.verts.ensure_lookup_table()
        vagina_vertex_group = gp_mesh.vertex_groups['vagina']
        vagina_weights = get_weights_as_array(gp_mesh, vagina_vertex_group)
        for bone_name in tentacle_bones:
            tentacle_weights = get_weights_as_array(gp_mesh, bone_name)
            is_tentacle_vert = tentacle_weights > 0
            tentacle_vert_indices, = np.where(is_tentacle_vert)
            for vert_idx in tentacle_vert_indices:
                start_vert = bm.verts[vert_idx]
                for start_edge in start_vert.link_edges:
                    edges_in_this_loop = list(iterate_edge_loop_over_allowed_verts(start_edge, is_tentacle_vert))
                    if len(edges_in_this_loop) > 0 and edges_in_this_loop[-1] == start_edge:
                        for edge in edges_in_this_loop:
                            is_tentacle_vert[edge.verts[0].index] = False
                            is_tentacle_vert[edge.verts[1].index] = False
                            edge.select_set(True)
            vagina_weights += tentacle_weights
        bmesh.update_edit_mesh(me)
        select_object(gp_rig)
        bpy.ops.object.mode_set(mode='EDIT')
        for bone_name in tentacle_bones:
            gp_rig.data.edit_bones.remove(gp_rig.data.edit_bones[bone_name])
        apply_vertex_group_weights(gp_mesh.vertex_groups['vagina'], vagina_weights)
        select_object(gp_mesh)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.dissolve_edges()
        for bone_name in tentacle_bones:
            vg = gp_mesh.vertex_groups.get(bone_name)
            if vg is not None:
                gp_mesh.vertex_groups.remove(vg)
        #uv_layer = bm.loops.layers.uv.verify()

    def remove_rectum_bones(self):
        gp_mesh = self.get_gp_or_body()
        gp_rig = get_rig_of(gp_mesh)
        select_object(gp_mesh)
        bone_names = ['rectum_'+str(i).zfill(2) for i in range(1, 8)]
        vgs = [gp_mesh.vertex_groups[i] for i in bone_names]
        vg_indices = np.array([i.index for i in vgs], dtype=int)

        def rectum_weight(vertex):
            return sum(g.weight for g in vertex.groups if g.group in vg_indices)

        rectum_weights = np.array([rectum_weight(v) for v in gp_mesh.data.vertices])

        select_object(gp_rig)
        bpy.ops.object.mode_set(mode='EDIT')
        for bone_name in bone_names[1:]:
            bone = gp_rig.data.edit_bones[bone_name]
            gp_rig.data.edit_bones.remove(bone)
        gp_rig.data.edit_bones[bone_names[0]].name = 'rectum'

        select_object(gp_mesh)
        apply_vertex_group_weights(vgs[0], rectum_weights)
        for vg in vgs[1:]:
            gp_mesh.vertex_groups.remove(vg)

    def remove_clitzilla(self):
        clit_center_uv = np.array((0.264921, 0.645213))
        clit_end_uv = np.array((0.265918, 0.614961))
        clit_radius = np.linalg.norm(clit_end_uv-clit_center_uv)
        gp_mesh = self.get_gp_or_body()
        gp_rig = get_rig_of(gp_mesh)

        select_object(gp_mesh)

        bpy.ops.object.mode_set(mode='EDIT')
        #bpy.context.scene.tool_settings.use_uv_select_sync = False
        bpy.ops.uv.select_all(action='DESELECT')
        bpy.ops.mesh.select_all(action='DESELECT')

        me = gp_mesh.data
        clitzilla_vgs = [gp_mesh.vertex_groups['clitzilla'+str(i).zfill(2)] for i in range(4, 15)]
        clitzilla_vg_indices = np.array([cvg.index for cvg in clitzilla_vgs], dtype=int)

        def is_clitzilla_weight(vertex):
            clitzilla_sum = 0
            for g in vertex.groups:
                if g.group in clitzilla_vg_indices:
                    clitzilla_sum += g.weight
            return clitzilla_sum > 0.01

        is_clitzilla_vertex = np.array([is_clitzilla_weight(v) for v in me.vertices], dtype=bool)
        # bpy.ops.mesh.select_mode(type='VERT')
        bm = bmesh.from_edit_mesh(me)
        # for v in bm.verts:
        #     v.select_set(is_clitzilla_vertex[v.index])
        #     #print(v.index)
        # bm.select_flush(True)
        # bmesh.update_edit_mesh(me)
        # return
        # uv_layer = bm.loops.layers.uv.verify()
        #for v in bm.verts:
        #    v.select = False

        suspected_edges = set()
        for face in bm.faces:
            if len(face.loops) == 4:
                is_clit = True
                for v in face.verts:
                    if not is_clitzilla_vertex[v.index]:
                        is_clit = False
                        break
                if is_clit:
                    v0, v1, v2, v3 = face.loops
                    e0 = edge_vector(v0).length
                    e1 = edge_vector(v1).length
                    e2 = edge_vector(v2).length
                    e3 = edge_vector(v3).length
                    e02 = e0+e2
                    e13 = e1+e3
                    if e02 > e13*2:
                        v0.edge.select_set(True)
                        v2.edge.select_set(True)
                        suspected_edges.add(v0.edge.index)
                        suspected_edges.add(v2.edge.index)
                    elif e13 > e02*2:
                        v1.edge.select_set(True)
                        v3.edge.select_set(True)
                        suspected_edges.add(v1.edge.index)
                        suspected_edges.add(v3.edge.index)
                        #face.select_set(True)
        # bmesh.update_edit_mesh(me)
        # return
        visited_edges = {}
        max_non_selected_edges = 15
        edge_loops = []
        bm.edges.ensure_lookup_table()
        class Loop:
            def __init__(self, edge_list):
                self.edge_list = edge_list
                self.next_loop = None
                self.prev_loop = None
                self.dist_to_next = -1
                self.neighbour_loops = {}
                self.is_valid=False
        for edge_index in suspected_edges:
            if edge_index not in visited_edges:
                start_edge = bm.edges[edge_index]
                edges_in_this_loop = Loop(list(iterate_edge_loop(start_edge)))
                visited_edges.update({e.index: edges_in_this_loop for e in edges_in_this_loop.edge_list})
                if len(edges_in_this_loop.edge_list)>0 and edges_in_this_loop.edge_list[-1]==start_edge:
                    number_of_non_selected_edges_in_loop = sum(not e.select for e in edges_in_this_loop.edge_list)
                    if number_of_non_selected_edges_in_loop < max_non_selected_edges:
                        edges_in_this_loop.is_valid = True
                        edge_loops.append(edges_in_this_loop)
        for edge_loop in edge_loops:
            edge = edge_loop.edge_list[0]
            for distance, parallel_edge in iterate_parallel_edges(edge):
                neighbour_loop = visited_edges.get(parallel_edge.index)
                if neighbour_loop is not None and neighbour_loop.is_valid:
                    edge_loop.neighbour_loops[neighbour_loop] = distance
        for edge_loop in edge_loops:
            if len(edge_loop.neighbour_loops)==1:
                starting_loop = edge_loop
                break
        prev_loop = starting_loop
        (next_loop, next_loop_dist), = prev_loop.neighbour_loops.items()
        while True:
            prev_loop.next_loop = next_loop
            prev_loop.dist_to_next = next_loop_dist
            next_loop.prev_loop = prev_loop
            if len(next_loop.neighbour_loops)==1:
                break
            else:
                (a, ad), (b, bd) = next_loop.neighbour_loops.items()
                if a != prev_loop:
                    prev_loop = next_loop
                    next_loop = a
                    next_loop_dist = ad
                elif b != prev_loop:
                    prev_loop = next_loop
                    next_loop = b
                    next_loop_dist = bd
                else:
                    break
        idx = 0
        edge_loop = starting_loop
        bpy.ops.mesh.select_all(action='DESELECT')
        min_distance = max(edge_loop.dist_to_next for edge_loop in edge_loops)*2
        cumulative_distance = 0
        while edge_loop is not None:
            cumulative_distance += edge_loop.dist_to_next
            if cumulative_distance > min_distance:
                cumulative_distance = 0
            else:
                for edge in edge_loop.edge_list:
                    edge.select_set(True)
            edge_loop = edge_loop.next_loop
            idx += 1

        #for v in bm.verts:
        #    if v.select:
        #        bm.verts.remove(v)
        bmesh.update_edit_mesh(me)
        bpy.ops.mesh.dissolve_edges()

        select_object(gp_rig)
        bpy.ops.object.mode_set(mode='EDIT')
        clitzilla_bones = [bone for bone in gp_rig.data.edit_bones if 'clitzilla' in bone.name.lower()]
        clit_vertex_group = gp_mesh.vertex_groups['clitoris']
        clit_weights = get_weights_as_array(gp_mesh, clit_vertex_group)
        for bone in clitzilla_bones:
            clit_weights += get_weights_as_array(gp_mesh, bone.name)
            gp_rig.data.edit_bones.remove(bone)
        apply_vertex_group_weights(clit_vertex_group, clit_weights)
        for cvg in clitzilla_vgs:
            gp_mesh.vertex_groups.remove(cvg)
        #bpy.ops.object.mode_set(mode='OBJECT')

    @staticmethod
    def remove_daz_bone_drivers():
        for o in bpy.data.objects:
            if isinstance(o.data, bpy.types.Mesh):
                if o.data.shape_keys is not None:
                    keys = o.data.shape_keys.key_blocks
                    for key in keys:
                        key.driver_remove('value')

    @staticmethod
    def remove_daz_bone_constraints():
        for rig in bpy.data.objects:
            if isinstance(rig.data, bpy.types.Armature):
                for bone in rig.pose.bones:
                    bone.lock_rotation = (False, False, False)
                    bone.lock_location = (False, False, False)
                    if not bone.name.endswith("(drv)"):
                        bone.rotation_mode = 'QUATERNION'
                    for c in list(bone.constraints):
                        if isinstance(c, bpy.types.LimitRotationConstraint):
                            bone.constraints.remove(c)

    def align_pose_to_ue5(self):
        import mathutils
        body_rig = self.get_body_rig()
        body_rig.location.y = -0.02
        select_object(body_rig)
        bpy.ops.object.mode_set(mode='POSE')
        bpy.ops.pose.select_all(action='SELECT')
        bpy.ops.pose.transforms_clear()
        hierarchy = self.get_hierarchy()

        def recursion(bone, parent_rotation):
            bone_name = bone.name
            if bone_name in DAZ_TO_UE5_POSE_ROTATIONS:
                _, _, x_axis, y_axis, z_axis, _, _ = hierarchy[bone_name]
                ue5_y_axis = mathutils.Vector(y_axis)
                daz_y_axis = mathutils.Vector(bone.y_axis)
                daz_y_axis.rotate(parent_rotation)
                quat = daz_y_axis.rotation_difference(ue5_y_axis)
                l, r, s = bone.matrix.decompose()
                r.rotate(quat)
                bone.matrix = mathutils.Matrix.LocRotScale(l, r, s)
                sum_quat = parent_rotation @ quat
                for child_bone in bone.children:
                    recursion(child_bone, sum_quat)
        bn = 'upperarm_'
        recursion(body_rig.pose.bones[bn+'l'], mathutils.Quaternion())
        recursion(body_rig.pose.bones[bn+'r'], mathutils.Quaternion())

    def apply_pose(self):
        body_rig = self.get_body_rig()
        select_object(body_rig)
        bpy.ops.object.mode_set(mode='POSE')
        bpy.ops.pose.armature_apply(selected=True)
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)


    def add_toon_outline(self):
        pass

    # def reorient_bones(self):
    #     import mathutils
    #     body_rig = self.get_body_rig()
    #     select_object(body_rig)
    #     bpy.ops.object.mode_set(mode='POSE')
    #     bone_transforms = {b.name: b.matrix.copy() for b in body_rig.pose.bones}
    #     bpy.ops.object.mode_set(mode='EDIT')
    #     for bone in body_rig.data.edit_bones:
    #         bone_name = bone.name
    #         if bone_name in UE5_BONE_HIERARCHY:
    #             matrix = bone_transforms[bone_name]
    #             matrix = matrix.inverted()
    #             matrix = np.array(matrix)
    #             print("matrix=", matrix.shape)
    #             ue5_start, ue5_tail, x_axis, y_axis, z_axis, _ = UE5_BONE_HIERARCHY[bone_name]
    #             ue5_orientation = np.empty(4)
    #             ue5_orientation[:3] = np.subtract(ue5_tail, ue5_start) / 100
    #             ue5_orientation[3] = 1
    #             print("ue5_orientation=", ue5_orientation.shape)
    #             daz_orientation = ue5_orientation @ matrix
    #             print("daz_orientation=", daz_orientation.shape)
    #             bone.tail = np.add(bone.head, daz_orientation[:3])
    #     bpy.ops.object.mode_set(mode='OBJECT')

    # def reorient_bones(self):
    #     import mathutils
    #     body_rig = self.get_body_rig()
    #     select_object(body_rig)
    #     bpy.ops.object.mode_set(mode='EDIT')
    #
    #     def recursion(bone, quat):
    #         bone_name = bone.name
    #         if bone_name in UE5_BONE_HIERARCHY:
    #             if bone_name in DAZ_TO_UE5_POSE_ROTATIONS:
    #                 rotation = DAZ_TO_UE5_POSE_ROTATIONS[bone_name]
    #                 rotation = mathutils.Euler(rotation, 'YZX')
    #                 rotation = rotation.to_quaternion()
    #                 # quat2 = quat.copy()
    #                 # quat2.rotate(rotation)
    #                 # rotation = quat2
    #                 # rotation.rotate(quat)
    #             else:
    #                 rotation = quat
    #             ue5_start, ue5_tail, x_axis, y_axis, z_axis, _ = UE5_BONE_HIERARCHY[bone_name]
    #             # ue5_orientation = np.empty(4)
    #             # ue5_orientation[:3] = np.subtract(ue5_tail, ue5_start) / 100
    #             # ue5_orientation[3] = 1
    #             ue5_orientation = mathutils.Vector(ue5_tail) - mathutils.Vector(ue5_start)
    #             ue5_orientation /= 100
    #             ue5_orientation.rotate(rotation.inverted())
    #             bone.tail = bone.head+ue5_orientation
    #             for child in bone.children:
    #                 recursion(child, rotation)
    #
    #     recursion(body_rig.data.edit_bones['pelvis'], mathutils.Quaternion())
    #     bpy.ops.object.mode_set(mode='OBJECT')

    @staticmethod
    def bake_animation_to_root():
        original_rig = find_original_body_rig()
        original_action = original_rig.animation_data.action
        root = bpy.data.objects.get('root')
        start_frame, end_frame = original_action.curve_frame_range
        select_object(root)
        new_action = bpy.data.actions.new(original_action.name+' ue5')
        root.animation_data.action = new_action
        bpy.ops.nla.bake(
            frame_start=int(start_frame),
            frame_end=int(end_frame),
            step=1,
            only_selected=False,
            visual_keying=True,
            clear_constraints=False,
            clear_parents=False,
            use_current_action=True,
            clean_curves=False,
            bake_types={'POSE'},
            channel_types={'LOCATION', 'ROTATION'}
        )

    @staticmethod
    def attach_duplicate_skeleton(attach):
        root = bpy.data.objects.get('root')
        if root is not None:
            for bone in root.pose.bones:
                t = bone.constraints.get("Transformation")
                if t is not None:
                    t.enabled = attach

    @staticmethod
    def has_duplicate_skeleton():
        return 'root' in bpy.data.objects

    @staticmethod
    def is_duplicate_skeleton_attached():
        root = bpy.data.objects.get('root')
        if root is not None:
            for bone in root.pose.bones:
                t = bone.constraints.get("Transformation")
                if t is not None:
                    return t.enabled
        return None

    def duplicate_skeleton(self):
        body_rig = self.get_body_rig()
        body_mesh = self.get_body_mesh()

        def dup(rig_obj, new_name, parent):
            rig = rig_obj.data
            select_object(rig_obj)
            bpy.ops.object.mode_set(mode='EDIT')
            class B:
                def __init__(self, b):
                    self.head = tuple(b.head)
                    self.roll = b.roll
                    self.tail = tuple(b.tail)
                    self.parent = None if b.parent is None else b.parent.name
                    self.rotation_mode = None
                    self.use_deform = b.use_deform

            bones_to_copy = {b.name: B(b) for b in rig.edit_bones if '(' not in b.name}
            bpy.ops.object.mode_set(mode='POSE')
            for bone in rig_obj.pose.bones:
                b = bones_to_copy.get(bone.name)
                if b is not None:
                    b.rotation_mode = bone.rotation_mode
            bpy.ops.object.mode_set(mode='OBJECT')
            new_rig_obj = rig_obj.copy()
            new_rig_obj.name = new_name
            rig_obj.users_collection[0].objects.link(new_rig_obj)
            if parent is not None:
                new_rig_obj.parent = parent
            new_rig_obj.data = bpy.data.armatures.new(new_name)

            new_rig = new_rig_obj.data
            new_rig.display_type = 'STICK'
            select_object(new_rig_obj)
            bpy.ops.object.mode_set(mode='EDIT')

            for bone_name, bone in bones_to_copy.items():
                new_bone = new_rig.edit_bones.new(bone_name)
                new_bone.head = bone.head
                new_bone.tail = bone.tail
                new_bone.roll = bone.roll
                new_bone.use_deform = bone.use_deform
            for new_bone in new_rig.edit_bones:
                bone = bones_to_copy[new_bone.name]
                if bone.parent is not None:
                    new_parent = new_rig.edit_bones[bone.parent]
                    new_bone.parent = new_parent
            bpy.ops.object.mode_set(mode='POSE')
            for new_bone in new_rig_obj.pose.bones:
                bone = bones_to_copy[new_bone.name]
                new_bone.rotation_mode = bone.rotation_mode

                c = new_bone.constraints.new('TRANSFORM')
                c.name = 'Transformation'
                c.target = rig_obj
                c.subtarget = new_bone.name
                c.map_from = 'ROTATION'
                c.map_to = 'ROTATION'
                if bone.rotation_mode != 'QUATERNION':
                    c.to_euler_order = bone.rotation_mode
                a = 3.14159
                c.to_max_x_rot = a
                c.to_max_y_rot = a
                c.to_max_z_rot = a
                c.to_min_x_rot = -a
                c.to_min_y_rot = -a
                c.to_min_z_rot = -a
                c.from_max_x_rot = a
                c.from_max_y_rot = a
                c.from_max_z_rot = a
                c.from_min_x_rot = -a
                c.from_min_y_rot = -a
                c.from_min_z_rot = -a
                c.map_to_x_from = 'X'
                c.map_to_y_from = 'Y'
                c.map_to_z_from = 'Z'
                c.target_space = 'LOCAL' # 'POSE'
                c.owner_space = 'LOCAL' # 'POSE'

            return new_rig_obj

        new_body_rig = dup(body_rig, 'root', None)
        body_mesh.name = 'root Mesh'
        old_to_new_rig = {body_rig: new_body_rig}
        for o in body_rig.children:
            if isinstance(o.data, bpy.types.Armature):
                new_o = dup(o, 'ue5_'+o.name, new_body_rig)
                old_to_new_rig[o] = new_o
        for o in bpy.data.objects:
            if isinstance(o.data, bpy.types.Mesh):
                new_o = old_to_new_rig.get(o.parent)
                if new_o is not None:
                    o.parent = new_o
            for m in o.modifiers:
                if isinstance(m, bpy.types.ArmatureModifier):
                    new_o = old_to_new_rig.get(m.object)
                    if new_o is not None:
                        m.object = new_o

    def compute_ue5_bone_oerientation(self):
        import mathutils
        body_rig = self.get_body_rig()
        select_object(body_rig)
        bpy.ops.object.mode_set(mode='EDIT')

        class B:
            def __init__(self, bone):
                self.new_y_axis = bone.z_axis.copy()
                self.new_z_axis = bone.x_axis.copy()
                self.length = 0
                self.axes = ['X', 'Y', 'Z']
                self.directions = [1,1,1]

        bones: {str: B} = {}
        hierarchy = self.get_hierarchy()
        for bone in body_rig.data.edit_bones:
            bone_name = bone.name
            bones[bone_name] = B(bone)
        for side in ['l', 'r']:
            for bone_name in ['hand_', 'clavicle_']:
                bone_name = bone_name+side
                b = bones[bone_name]
                bone = body_rig.data.edit_bones[bone_name]
                b.new_y_axis = bone.x_axis.copy()
                b.new_z_axis = bone.z_axis.copy()
            bones['foot_'+side].new_y_axis = bones['calf_'+side].new_y_axis
        for spine_bone in ['pelvis', 'spine_01', 'spine_02', 'spine_03', 'spine_04', 'spine_05', 'neck_01', 'neck_02']:
            _, _, x_axis, y_axis, z_axis, _, _ = hierarchy[spine_bone]
            b = bones[spine_bone]
            b.new_y_axis = mathutils.Vector(y_axis)
            b.new_z_axis = mathutils.Vector(z_axis)
        for bone in body_rig.data.edit_bones:
            bone_name = bone.name
            if bone_name in hierarchy:
                b = bones[bone_name]
                new_z_axis = b.new_z_axis
                new_y_axis = b.new_y_axis
                ue5_start, ue5_tail, x_axis, y_axis, z_axis, roll, parent_name = hierarchy[bone_name]
                z_axis = mathutils.Vector(z_axis)
                y_axis = mathutils.Vector(y_axis)
                ue5_orientation = mathutils.Vector(ue5_tail)-mathutils.Vector(ue5_start)
                ue5_orientation /= 100
                #if bone_name in DAZ_TO_UE5_POSE_ROTATIONS:
                length = ue5_orientation.length
                if y_axis.dot(new_y_axis) < 0:
                    new_y_axis = -new_y_axis
                if z_axis.dot(new_z_axis) < 0:
                    new_z_axis = -new_z_axis
                b.new_z_axis = new_z_axis
                b.new_y_axis = new_y_axis
                b.length = length

        return hierarchy, bones

    def reweight_pelvis(self):
        body_mesh = self.get_body_mesh()
        spine1, spine2, hip = intersect_two_weight_groups(body_mesh, 'spine_01', 'spine_02', 'tmp')
        spine1_name = spine1.name
        spine1.name = 'hip'
        hip.name = spine1_name

    def reorient_bones(self):
        import mathutils
        hierarchy, bones = self.compute_ue5_bone_oerientation()

        def reorient(rig):
            select_object(rig)
            bpy.ops.object.mode_set(mode='EDIT')
            for bone in rig.data.edit_bones:
                bone_name = bone.name
                if bone_name in hierarchy:
                    b = bones[bone_name]
                    old_x = bone.x_axis.copy()
                    old_y = bone.y_axis.copy()
                    old_z = bone.z_axis.copy()
                    bone.tail = bone.head + b.new_y_axis * b.length
                    bone.align_roll(b.new_z_axis)
                    new_x = bone.x_axis.copy()
                    new_y = bone.y_axis.copy()
                    new_z = bone.z_axis.copy()
                    old = mathutils.Matrix((old_x, old_y, old_z))
                    new = mathutils.Matrix((new_x, new_y, new_z))
                    print("bone_name=", bone_name)
                    print("old=", old)
                    print("new=", new)
                    dot_products = old @ new.transposed()
                    print("dots=", dot_products)
                    old_axes = "XYZ"
                    used_axes = ""
                    for old_i in range(3):
                        most_similar_new_d = 0
                        most_similar_new_i = -1
                        for new_i in range(3):
                            if old_axes[new_i] not in used_axes:
                                d = dot_products[old_i][new_i]
                                if abs(d) > abs(most_similar_new_d):
                                    most_similar_new_d = d
                                    most_similar_new_i = new_i
                        used_axes += old_axes[most_similar_new_i]
                        print("b.axes[",most_similar_new_i,"]=",b.axes[most_similar_new_i]," := old_axes[",old_i,"]=",old_axes[old_i])
                        b.axes[most_similar_new_i] = old_axes[old_i]
                        b.directions[most_similar_new_i] = 1 if most_similar_new_d >= 0 else -1
                    print("new_axes=", b.axes)

            bpy.ops.object.mode_set(mode='POSE')
            for bone in rig.pose.bones:
                bone_name = bone.name
                if bone_name in hierarchy:
                    t = bone.constraints.get("Transformation")
                    if t is not None:
                        b = bones[bone_name]
                        t.map_to_x_from = b.axes[0]
                        t.map_to_y_from = b.axes[1]
                        t.map_to_z_from = b.axes[2]
                        if b.directions[0] == -1:
                            t.to_min_x_rot = -t.to_min_x_rot
                            t.to_max_x_rot = -t.to_max_x_rot
                        if b.directions[1] == -1:
                            t.to_min_y_rot = -t.to_min_y_rot
                            t.to_max_y_rot = -t.to_max_y_rot
                        if b.directions[2] == -1:
                            t.to_min_z_rot = -t.to_min_z_rot
                            t.to_max_z_rot = -t.to_max_z_rot
                # else:
                #     bone.tail = bone.head + ue5_orientation
            bpy.ops.object.mode_set(mode='OBJECT')

        body_rig = self.get_body_rig()
        reorient(body_rig)
        children = list(body_rig.children)
        while len(children) > 0:
            o = children.pop()
            if isinstance(o.data, bpy.types.Armature):
                reorient(o)
            children.extend(o.children)

    def add_ue5_ik_bones(self):
        rig = self.get_body_rig()
        select_object(rig)
        bpy.ops.object.mode_set(mode='EDIT')
        hierarchy = self.get_hierarchy()
        for ik_bone_name, fk_bone_name in UE5_IK_BONES.items():
            ik_bone = rig.data.edit_bones.new(ik_bone_name)
            ik_bone.use_deform = False
            ik_bone.use_connect = False
            if fk_bone_name != '':
                head = np.array(rig.data.edit_bones[fk_bone_name].head)
                ik_bone.head = head
            ik_bone.tail = np.array(ik_bone.head)
            ik_bone.tail.y += 0.2
        for ik_bone_name in UE5_IK_BONES:
            parent_ik_bone_name = hierarchy[ik_bone_name][-1]
            if parent_ik_bone_name is not None and parent_ik_bone_name != '':
                rig.data.edit_bones[ik_bone_name].parent = rig.data.edit_bones[parent_ik_bone_name]

    def scale_to_ue5_units(self):
        s = bpy.context.scene.unit_settings.scale_length
        bpy.context.scene.unit_settings.scale_length = 0.01
        z = s / 0.01
        self.scale(z)
        for workspace in bpy.data.workspaces:
            for screen in workspace.screens:
                for area in screen.areas:
                    if area.type == 'VIEW_3D':
                        for space in area.spaces:
                            if space.type == 'VIEW_3D':
                                space.clip_start = 1
                                space.clip_end = 100000

    def scale_to_quinn(self):
        mesh = self.get_body_mesh()
        height = mesh.dimensions[2]
        ue5_height = QUINN_HEIGHT if self.is_female() else MANNY_HEIGHT
        rig = self.get_body_rig()
        select_object(rig)
        if 'hip' in rig.data.bones:
            root = rig.data.bones['hip']
        elif 'pelvis' in rig.data.bones:
            root = rig.data.bones['pelvis']
        else:
            root = None
        if root is None:
            loc = None
        else:
            hierarchy = self.get_hierarchy()
            ue5_pevis_pos = hierarchy['pelvis'][0]
            loc = (0,ue5_pevis_pos[1]/100 - root.head.y,0)
        scl = ue5_height / height
        scl = (scl, scl, scl)
        apply_recursive(rig, location=loc, scale=scl)

    def scale(self, z):
        rig = self.get_body_rig()
        scale(rig, (z, z, z))
        original_rig = find_original_body_rig()
        if original_rig is not None:
            scale(original_rig, (z, z, z))

    def translate(self, t):
        rig = self.get_body_rig()
        translate(rig, t)
        original_rig = find_original_body_rig()
        if original_rig is not None:
            translate(original_rig, t)

    def detach_hair_from_skeleton(self):
        for hair in find_all_hair():
            rig = get_rig_of(hair)
            select_object(rig)
            bpy.ops.object.mode_set(mode='EDIT')
            if 'head' in rig.data.edit_bones:
                head = rig.data.edit_bones['head']
                head_pos = head.head.copy()
                bones_to_keep = set(b.name for b in head.children_recursive)
                bones_to_keep.add('head')
                for bone in list(rig.data.edit_bones):
                    if bone.name not in bones_to_keep:
                        rig.data.edit_bones.remove(bone)
                translate(rig, -head_pos)

    def export_body_to_fbx(self):
        body = self.get_body_mesh()
        rig = self.get_body_rig()
        self.export_to_fbx(rig, body, os.path.join(self.workdir, self.name + '.fbx'))

    def export_grafts_to_fbx(self):
        p = os.path.join(self.workdir, self.name + "_grafts")
        if not os.path.exists(p):
            os.mkdir(p)
        for g in GEOGRAFTS:
            o = bpy.data.objects.get(g+" Mesh")
            if o is not None:
                rig = get_rig_of(o)
                self.export_to_fbx(rig, o, os.path.join(p, o.name + '.fbx'))

    def export_animation_to_fbx(self):
        body = self.get_body_mesh()
        rig = self.get_body_rig()

        select_object(rig)
        body.select_set(True)
        p = os.path.join(self.workdir, self.name + "_anims")
        if not os.path.exists(p):
            os.mkdir(p)
        action = rig.animation_data.action
        start_frame, end_frame = action.curve_frame_range
        path = os.path.join(p, action.name + '.fbx')
        bpy.ops.export_scene.fbx(filepath=path,
                                 check_existing=False,
                                 filter_glob='*.fbx',
                                 use_selection=True,
                                 use_visible=False,
                                 use_active_collection=False,
                                 collection='',
                                 global_scale=1.0,
                                 apply_unit_scale=True,
                                 apply_scale_options='FBX_SCALE_NONE',
                                 use_space_transform=True,
                                 bake_space_transform=False,
                                 object_types={'ARMATURE', 'CAMERA', 'EMPTY', 'LIGHT', 'MESH', 'OTHER'},
                                 use_mesh_modifiers=True,
                                 use_mesh_modifiers_render=True,
                                 mesh_smooth_type='FACE',
                                 colors_type='SRGB',
                                 prioritize_active_color=False,
                                 use_subsurf=False,
                                 use_mesh_edges=False,
                                 use_tspace=False,
                                 use_triangles=False,
                                 use_custom_props=False,
                                 add_leaf_bones=False,
                                 primary_bone_axis='Y',
                                 secondary_bone_axis='X',
                                 use_armature_deform_only=False,
                                 armature_nodetype='NULL',
                                 bake_anim=True,
                                 bake_anim_use_all_bones=True,
                                 bake_anim_use_nla_strips=False,
                                 bake_anim_use_all_actions=False,
                                 bake_anim_force_startend_keying=True,
                                 bake_anim_step=1.0,
                                 bake_anim_simplify_factor=1.0,
                                 path_mode='AUTO',
                                 embed_textures=False,
                                 batch_mode='OFF',
                                 use_batch_own_dir=True,
                                 use_metadata=True,
                                 axis_forward='-Z',
                                 axis_up='Y')

    def serialize_extra_bones(self):
        body = self.get_body_mesh()
        rig = self.get_body_rig()
        select_object(rig)
        bones = [bone.name for bone in rig.data.bones if not is_known_bone(bone.name)]
        serialize_bone_and_weights(body, bones)

    def serialize_extra_clothes(self):
        for o in bpy.data.objects:
            if not o.hide_get() and isinstance(o.data, bpy.types.Mesh) and o.name.endswith(" Mesh"):
                print("'"+o.name[:-len(" Mesh")]+"': ClothesMeta('"+o.data.daz_importer.DazFingerPrint+"', -1, "+str('panties' in o.name.lower())+"),")

    def export_hair_to_fbx(self):
        root = bpy.data.objects.get('root')
        if root is not None:
            root.name = 'root_tmp'
        p = os.path.join(self.workdir, self.name + "_hair")
        if not os.path.exists(p):
            os.mkdir(p)
        for hair in find_all_hair():
            hair_rig = get_rig_of(hair)
            prev_name = hair_rig.name
            hair_rig.name = 'root'
            name = hair.name[:-len(' Mesh')]
            hide_object(hair, False)
            hide_object(hair_rig, False)
            self.export_to_fbx(None, hair, os.path.join(p, name + '.fbx'))
            hair_rig.name = prev_name
        if root is not None:
            root.name = 'root'

    def export_clothes_to_fbx(self):
        root = bpy.data.objects.get('root')
        if root is not None:
            root.name = 'root_tmp'
        p = os.path.join(self.workdir, self.name+"_clothes")
        if not os.path.exists(p):
            os.mkdir(p)
        for clothes in find_all_clothes():
            clothes = clothes.obj
            name = clothes.name[:-len(' Mesh')] if clothes.name.endswith(' Mesh') else clothes.name
            clothes_rig = get_rig_of(clothes)
            prev_name = clothes.parent.name
            clothes_rig.name = 'root'
            self.export_to_fbx(clothes_rig, clothes, os.path.join(p, name + '.fbx'))
            clothes_rig.name = prev_name
        if root is not None:
            root.name = 'root'

    def export_cum_to_fbx(self):
        rig = self.get_body_rig()
        p = os.path.join(self.workdir, self.name + "_cum")
        if not os.path.exists(p):
            os.mkdir(p)
        for cum in find_cum():
            name = cum.name[:-len(' Mesh')] if cum.name.endswith(' Mesh') else cum.name
            self.export_to_fbx(rig, cum, os.path.join(p, name + '.fbx'))

    def export_to_fbx(self, rig, obj, path):
        if "Subsurf" in obj.modifiers:
            obj.modifiers.remove(obj.modifiers["Subsurf"])
        if rig is None:
            rig = get_rig_of(obj)
        select_object(rig)
        obj.select_set(True)
        bpy.ops.export_scene.fbx(filepath=path,
                                 check_existing=False,
                                 filter_glob='*.fbx',
                                 use_selection=True,
                                 use_visible=False,
                                 use_active_collection=False,
                                 collection='',
                                 global_scale=1.0,
                                 apply_unit_scale=True,
                                 apply_scale_options='FBX_SCALE_NONE',
                                 use_space_transform=True,
                                 bake_space_transform=False,
                                 object_types={'ARMATURE', 'CAMERA', 'EMPTY', 'LIGHT', 'MESH', 'OTHER'},
                                 use_mesh_modifiers=True,
                                 use_mesh_modifiers_render=True,
                                 mesh_smooth_type='FACE',
                                 colors_type='SRGB',
                                 prioritize_active_color=False,
                                 use_subsurf=False,
                                 use_mesh_edges=False,
                                 use_tspace=False,
                                 use_triangles=False,
                                 use_custom_props=False,
                                 add_leaf_bones=False,
                                 primary_bone_axis='Y',
                                 secondary_bone_axis='X',
                                 use_armature_deform_only=False,
                                 armature_nodetype='NULL',
                                 bake_anim=False,
                                 bake_anim_use_all_bones=True,
                                 bake_anim_use_nla_strips=True,
                                 bake_anim_use_all_actions=True,
                                 bake_anim_force_startend_keying=True,
                                 bake_anim_step=1.0,
                                 bake_anim_simplify_factor=1.0,
                                 path_mode='AUTO',
                                 embed_textures=False,
                                 batch_mode='OFF',
                                 use_batch_own_dir=True,
                                 use_metadata=True,
                                 axis_forward='-Z',
                                 axis_up='Y')

    def print_morphs_csv(self):
        cat_mapping = {
            CAT_BREAST: "BREASTS",
            CAT_HEAD: "HEAD",
            CAT_ARMS: "ARMS",
            CAT_LEGS: "LEGS",
            CAT_BODY: "BODY",
            CAT_ASS: "ASS",
            CAT_GENITALS: "GENITALS",
            CAT_SPECIAL: "SPECIAL",
            CAT_FACS: "FACS",
            CAT_FACSEXPR: "FACS",
            CAT_FACSDET: "FACS",
            CAT_JCM: "JCM",
        }
        shape_key_categories = {}
        for mesh_name, morphs in MORPHS.items():
            for gender, shapes in morphs['shapes'].items():
                if gender == 'female':
                    is_female = True
                    is_male = False
                elif gender == 'male':
                    is_female = False
                    is_male = True
                elif gender == 'unisex':
                    is_female = True
                    is_male = True
                for shape, meta in shapes.items():
                    shape_key_categories[shape] = is_female, is_male, meta
        body = self.get_body_mesh()
        print("---,Name,MorphName,BodyPart,IsForFemales,IsForMales,Min,Max,Default")
        for b in body.data.shape_keys.key_blocks:
            n = b.name
            if n in shape_key_categories:
                assert isinstance(n, str)
                is_female, is_male, meta = shape_key_categories[n]
                if meta.category not in CATS_FACS:
                    n = n.replace(' ', '_')
                    row = [n, meta.title, n, meta.category, is_female, is_male, b.slider_min, b.slider_max, b.value]
                    print(",".join(map(str,row)))

def save_blend_file(duf_filepath):
    duf_filepath = os.path.abspath(duf_filepath)
    workdir = os.path.dirname(duf_filepath)
    name = os.path.basename(duf_filepath)[:-len(".duf")]
    blend_filepath = os.path.join(workdir, name + ".blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_filepath)



def run_outside_blender():
    import subprocess

    file_path = os.path.realpath(__file__)

    if len(sys.argv) < 2 or not sys.argv[1].endswith(".duf"):
        print("Specify path to .duf")
        exit()
    duf_path: str = os.path.abspath(sys.argv[1])
    if not os.path.exists(duf_path):
        print("File not exists:", duf_path)
        exit()
    subprocess.run(["blender", "-P", file_path, "--", duf_path])

    # Image("")


bl_info = {
    "name": "Daz Optimizer",
    "blender": (2, 80, 0),
    "category": "Object",
}


class EasyImportPanel(bpy.types.Panel):
    bl_label = "Panel"
    bl_idname = "dazoptim_easy_import_panel"
    bl_space_type = 'FILE_BROWSER'
    bl_region_type = 'TOOLS'
    filepath = None
    directory = None

    @classmethod
    def poll(cls, context):
        op = context.active_operator
        if op and op.bl_idname == "DAZ_OT_easy_import_daz":
            cls.directory = op.directory
            cls.filepath = op.filepath
            #context.scene['duf_filepath'] = op.filepath
        return False

    # needs a draw method
    def draw(self, context):
        pass


UNLOCK = False


class DazDelCube_operator(bpy.types.Operator):
    """ Delete default cube """
    bl_idname = "dazoptim.delcube"
    bl_label = "Delete default cube"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = None

    @classmethod
    def poll(cls, context):
        return UNLOCK or 'daz_optim_stage' not in context.scene

    def execute(self, context):
        for x in list(bpy.data.objects):
            bpy.data.objects.remove(x)
        for x in list(bpy.data.collections):
            bpy.data.collections.remove(x)
        return {'FINISHED'}



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
    return 'GoldenPalace_G9 Mesh' in bpy.data.objects

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

class DazAddGluteOneBigBones_operator(bpy.types.Operator):
    """ Add glute bones """
    bl_idname = "dazoptim.add_glute_single_big_bone"
    bl_label = "Optimize UVs"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = '7'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazMergeAllRigs_operator], [DazAddGluteOneBigBones_operator])

    def execute(self, context):
        DazOptimizer().add_single_big_glute_bones()
        pass_stage(self)
        return {'FINISHED'}

class DazAddGluteTwoSmallerBones_operator(bpy.types.Operator):
    """ Add glute bones """
    bl_idname = "dazoptim.add_glute_two_small_bones"
    bl_label = "Optimize UVs"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = '7'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazMergeAllRigs_operator], [DazAddGluteTwoSmallerBones_operator])

    def execute(self, context):
        DazOptimizer().add_double_small_glute_bones()
        pass_stage(self)
        return {'FINISHED'}

class DazAddThighUpperBones_operator(bpy.types.Operator):
    """ Add glute bones """
    bl_idname = "dazoptim.add_thigh_upper_bones"
    bl_label = "Optimize UVs"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = '#'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazMergeAllRigs_operator], [DazAddThighUpperBones_operator])

    def execute(self, context):
        DazOptimizer().add_high_thigh_jiggle()
        pass_stage(self)
        return {'FINISHED'}

class DazAddThighLowerBones_operator(bpy.types.Operator):
    """ Add glute bones """
    bl_idname = "dazoptim.add_thigh_lower_bones"
    bl_label = "Optimize UVs"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = '%'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazMergeAllRigs_operator], [DazAddThighLowerBones_operator])

    def execute(self, context):
        DazOptimizer().add_low_thigh_jiggle()
        pass_stage(self)
        return {'FINISHED'}

class DazAddThighSideBones_operator(bpy.types.Operator):
    """ Add glute bones """
    bl_idname = "dazoptim.add_thigh_side_bones"
    bl_label = "Add thigh side bones"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = '^'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazMergeAllRigs_operator], [DazAddThighSideBones_operator])

    def execute(self, context):
        DazOptimizer().add_side_thigh_jiggle()
        pass_stage(self)
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
            DazAddGluteOneBigBones_operator, DazAddBreastBones_operator, DazAddGluteTwoSmallerBones_operator,
            DazAddThighLowerBones_operator, DazAddThighUpperBones_operator, DazAddThighSideBones_operator
]
class DazTransferMissingBonesToClothes_operator(bpy.types.Operator):
    """ transfer new bones to clothes """
    bl_idname = "dazoptim.transfer_new_bones_to_clothes"
    bl_label = "transfer new bones to clothes"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = '9'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage_any(context, BONE_ADDING_OPS, [DazTransferMissingBonesToClothes_operator, RigPhysicsBones])

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
        return UNLOCK or (has_gp() and check_stage(context, [DazMaleLoad_operator], [DazOptimizeGoldenPalaceUVs]))

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
        return UNLOCK or check_stage(context, [DazScaleToQuinn], [DazScaleToUnreal])

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


class SaveMorphsOnlyFACS(bpy.types.Operator):
    """ Generates favourite morphs """
    bl_idname = "dazoptim.save_fav_morphs_only_facs"
    bl_label = "Save favourite morphs with facs"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 'Q'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazSaveBlend_operator], [])

    def execute(self, context):
        DazOptimizer().make_fav_morphs_list(CATS_FACS)
        pass_stage(self)
        return {'FINISHED'}

class SaveMorphsOnlyBody(bpy.types.Operator):
    """ Generates favourite morphs """
    bl_idname = "dazoptim.save_fav_morphs_only_body"
    bl_label = "Save favourite morphs with facs"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 'Q'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazSaveBlend_operator], [])

    def execute(self, context):
        DazOptimizer().make_fav_morphs_list(CATS_BODY)
        pass_stage(self)
        return {'FINISHED'}

class SaveMorphsOnlyGenitals(bpy.types.Operator):
    """ Generates favourite morphs """
    bl_idname = "dazoptim.save_fav_morphs_only_genitals"
    bl_label = "Save favourite morphs with facs"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 'Q'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazSaveBlend_operator], [])

    def execute(self, context):
        DazOptimizer().make_fav_morphs_list({CAT_GENITALS})
        pass_stage(self)
        return {'FINISHED'}

class SaveMorphsOnlySpecial(bpy.types.Operator):
    """ Generates favourite morphs """
    bl_idname = "dazoptim.save_fav_morphs_only_special"
    bl_label = "Save favourite morphs with special"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 'Q'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazSaveBlend_operator], [])

    def execute(self, context):
        DazOptimizer().make_fav_morphs_list({CAT_SPECIAL})
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
        return UNLOCK or check_stage(context, [SaveMorphs], [DazMergeGrografts_operator])

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


class RigPhysicsBones(bpy.types.Operator):
    """ rig physics bones """
    bl_idname = "dazoptim.rig_physics_bones"
    bl_label = "Rig Physics Bones"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = "6"

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazMaleLoad_operator], [RigPhysicsBones, DazMergeAllRigs_operator])

    def execute(self, context):
        DazOptimizer().rig_physics_bones()
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
        for c in find_all_clothes():
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
        for c in find_all_hair():
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
        for c in find_all_clothes():
            c.obj.hide_set(False)
        return {'FINISHED'}

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

class EntryLabel:
    def __init__(self, s: str, idx: int):
        self.s = s
        self.idx = idx

    def draw(self, p: DazOptimize_sidebar, col, context, idx: int):
        p.layout.label(text=self.s)
        col = p.layout.column(align=True)
        return self.idx, col


class EntryOp:
    def __init__(self, op_class: type, op_text: str):
        self.op_text = op_text
        self.op_class = op_class

    def draw(self, p: DazOptimize_sidebar, col, context, idx: int):
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
    def __init__(self, name: str):
        self.name = name

    def draw(self, p: DazOptimize_sidebar, col, context, idx: int):
        p.props[self.name] = col.prop(context.scene, self.name)
        return idx, col


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
    EntryOp(DazMergeAllMaterials_operator, "Merge all materials"),
    EntryOp(DazMergeCumMaterials_operator, "Merge cum materials"),
    EntryOp(DazDecimateCumMeshes_operator, "Decimate cum meshes"),
    EntryOp(DazApplyDecimateCumMeshes_operator, "Apply decimate cum"),
    EntryOp(DazMergeMultiMeshClothes_operator, "Merge clothes sub-meshes"),
    EntryOp(FixToonEyes, "Fix Nirv Zero eyes"),
    EntryOp(DazGiveErection, "Give erection"),
    EntryOp(SaveMorphs, "Generate fav morphs (all)"),
    EntryOp(SaveMorphsOnlyFACS, "Generate fav morphs (only FACS)"),
    EntryOp(SaveMorphsOnlyBody, "Generate fav morphs (only body)"),
    EntryOp(SaveMorphsOnlyGenitals, "Generate fav morphs (only genitals)"),
    EntryOp(SaveMorphsOnlySpecial, "Generate fav morphs (only special)"),
    EntryOp(LoadMorphs, "Load fav morphs"),
    EntryOp(RebindFavMorphs, "Rebind fav morphs"),
    EntryOp(TransferMorphsToGeografts, "Transfer morphs to geografts"),
    EntryOp(DazAddBreastBones_operator, "Subdivide breast bones"),
    EntryOp(DazAddGluteOneBigBones_operator, "Add glute bones (one big)"),
    EntryOp(DazAddGluteTwoSmallerBones_operator, "Add glute bones (two smaller)"),
    EntryOp(DazAddThighUpperBones_operator, "Add thigh bones (upper)"),
    EntryOp(DazAddThighLowerBones_operator, "Add thigh bones (lower)"),
    EntryOp(DazAddThighSideBones_operator, "Add thigh bones (sides)"),
    EntryOp(DazSimplifyMaterials_operator, "Simplify materials"),
    EntryOp(DazOptimizeEyes_operator, "Optimize eyes mesh"),
    EntryOp(DazOptimizeEyesForToon_operator, "Optimize eyes for toon"),
    EntryOp(DazOptimizeEyelashes_operator, "Optimize eyelashes"),
    EntryOp(DazOptimizeEyebrows_operator, "Optimize eyebrows"),
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
    EntryProp("clothes_displacement"),
    EntryOp(DazApplyFitClothes_operator, "Apply displacement"),
    EntryOp(DazFitPanties_operator, "Fit panties"),
    EntryOp(DazFitSkinTightClothes_operator, "Fit skin-tight clothes"),
    EntryProp("skin_tight_displacement"),
    EntryOp(DazApplyFitSkinTightClothes_operator, "Apply skin-tight clothes"),
    EntryOp(DazTransferMissingBonesToClothes_operator, "Transfer new bones to clothes"),
    EntryOp(DazTransferMissingBonesToCum_operator, "Transfer new bones to cum"),
    EntryOp(TransferMorphsToClothes, "Transfer morphs to clothes"),
    EntryOp(TransferMorphsToCum, "Transfer morphs to cum"),
    EntryOp(DazScaleToQuinn, "Scale to Manny height"),
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
    bpy.types.Scene.clothes_displacement = bpy.props.FloatProperty(name="clothes_displacement", default=1)
    bpy.types.Scene.skin_tight_displacement = bpy.props.FloatProperty(name="skin_tight_displacement", default=0)


def unregister():
    for c in classes:
        bpy.utils.unregister_class(c)
    del bpy.types.Scene.clothes_displacement
    del bpy.types.Scene.skin_tight_displacement


if __name__ == '__main__':
    register()
