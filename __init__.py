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
    def __init__(self, nump, path, img: bpy.types.Image=None):
        self.np = nump
        self.image = img
        self.path = path

    def to_numpy(self):
        return self.np

    def to_image(self, path):
        if self.image is None:
            img = np_to_pil(self.np)
            img.save(path)
            self.image = bpy.data.images.load(path)
        return self.image

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

def gamma(color, gamma_value):
    return np.power(color, gamma_value)

def lerp(a,b,alpha):
    return a + (b - a) * alpha

def screen(a,b,alpha):
    facm = 1.0 - alpha
    return 1.0 - (facm + alpha * (1 - b)) * (1 - a)

def alpha_multiply(a,b,alpha):
    return a * (b * alpha + 1 - alpha)

def overlay(a,b,alpha):
    facm = 1.0 - alpha
    mul = a * (facm + 2 * alpha * b)
    screen = 1.0 - (facm + 2.0 * alpha * (1.0 - b)) * (1.0 - a)
    mask = a < 0.5
    screen[mask] = mul[mask]
    return screen

def reset_alpha_channel(output, original):
    if output.shape[2] > 3:
        output[:,:,3] = original[:,:,3]

def burn(a,b,alpha):
    raise Exception("burn not implemented yet")

def dodge(a,b,alpha):
    raise Exception("dodge not implemented yet")



class BakedNode:
    def __init(self, node):
        self.node = node
        self.usages = 0
        for outp in node.outputs:
            self.usages += len(outp.links)


class MaterialBaker:
    @staticmethod
    def tonp(x):
        return x if isinstance(x, np.ndarray) or np.isscalar(x) else x.to_numpy()

    @staticmethod
    def topa(x):
        return [x.path] if isinstance(x, BakedImg) else []

    @staticmethod
    def channels(x):
        if x.ndim == 0:
            return 1
        elif x.ndim == 1:
            if len(x) == 4 and x[3] == 1:
                return 3
            return len(x)
        elif x.ndim < 3:
            return 1
        else:
            return x.shape[2]

    @staticmethod
    def size(x, axis):
        if x.ndim <= 1:
            return 1
        else:
            return x.shape[axis]

    @staticmethod
    def to_size(x, w, h):
        if x.ndim < 3:
            return x
        h2, w2, c2 = x.shape
        if h2 == h and w2 == w:
            return x
        zeros = np.zeros((h, w, c2))
        zeros[:h2, :w2] = x
        return zeros

    @staticmethod
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

    @staticmethod
    def common(*args):
        a = []
        for arg in args:
            a.extend(MaterialBaker.topa(arg))
        return os.path.commonprefix(a)

    @staticmethod
    def is_leaf(node):
        for inp in node.inputs:
            if inp.is_linked:
                return False
        return True

    @staticmethod
    def is_final(node):
        for inp in node.outputs:
            if inp.is_linked:
                return False
        return True

    @staticmethod
    def get_all_final(node_tree):
        return [n for n in node_tree.nodes if MaterialBaker.is_final(n)]

    @staticmethod
    def sort_topologically(node_tree):
        sorted_nodes = []
        visited = set()

        def sort_topologically_recursion(node):
            for i in node.inputs:
                for l in i.links:
                    node = l.from_node
                    if node not in visited:
                        visited.add(node)
                        sort_topologically_recursion(node)
            sorted_nodes.append(node)


        for end in MaterialBaker.get_all_final(node_tree):
            sort_topologically_recursion(end)

        return sorted_nodes

    def __int__(self, node_tree):
        self.node_tree = node_tree
        self.evaluated = {}
        self.inputs = None
        self.outputs = None

    def bake(self):
        nodes = MaterialBaker.sort_topologically(self.node_tree)
        for node in nodes:
            self.evaluate(node)

    def get(self, node, socket, default_value=None):
        if isinstance(socket, (str,int)):
            socket = node.inputs.get(socket)
        if socket is None or socket.is_inactive:
            return default_value
        if socket.is_linked:
            return self.evaluated.get(socket)
        else:
            return socket.default_value

    def evaluate(self, node):
        if isinstance(node, bpy.types.ShaderNodeRGB):
            o = node.outputs[0]
            rgb = linearrgb_to_srgb(o.default_value)
            self.evaluated[o] = rgb
            return True
        elif isinstance(node, bpy.types.ShaderNodeMath):
            a_i = self.get(node, 0)
            b_i = self.get(node, 1)
            if a_i is None or b_i is None:
                return False
            p = MaterialBaker.common(a_i, b_i)
            a = MaterialBaker.tonp(a_i)
            b = MaterialBaker.tonp(b_i)
            if node.operation == "MULTIPLY":
                c = a * b
            elif node.operation == "ADD":
                c = a + b
            elif node.operation == "SUBTRACT":
                c = a - b
            elif node.operation == "DIVIDE":
                c = a / b
            elif node.operation == "MODULO":
                c = a % b
            self.evaluated[node.outputs[0]] = BakedImg(c, p)
            return True
        elif isinstance(node, bpy.types.NodeGroupInput):
            if self.inputs is not None:
                for out in node.outputs:
                    self.evaluated[out] = self.inputs[out.name]
            return False # the input node is never removed after optimisation
        elif isinstance(node, bpy.types.NodeGroupOutput):
            if self.outputs is not None:
                for i in node.inputs:
                    optimised_i = self.get(node, i)
                    if optimised_i is not None:
                        self.outputs[i.name] = optimised_i
            return False # the input node is never removed after optimisation
        elif isinstance(node, bpy.types.ShaderNodeGroup):
            mb = MaterialBaker(node.node_tree)
            mb.inputs = {}
            for i in node.inputs:
                mb.inputs[i.name] = self.get(node, i)
            mb.outputs = {}
            mb.bake()
            all_optimised = True
            for o in node.outputs:
                optimised_o = mb.outputs.get(o.name)
                if optimised_o is None:
                    all_optimised = False
                else:
                    self.evaluated[o] = optimised_o
            return all_optimised
        elif isinstance(node, bpy.types.ShaderNodeMapping):
            l_i = self.get(node, 'Location', 0)
            s_i = self.get(node, 'Scale', 1)
            r_i = self.get(node, 'Rotation', 0)
            if l_i is None or s_i is None or r_i is None:
                return False
            l = MaterialBaker.tonp(l_i)
            s = MaterialBaker.tonp(s_i)
            r = MaterialBaker.tonp(r_i)
            if not np.all(r == 0):
                raise Exception("Encountered "+ repr(node)+ ' with  rotation! Not implemented yet!!!!')
            v_i = self.get(node, 'Vector', 0)
            v = MaterialBaker.tonp(v_i)
            out = v * s + l
            p = MaterialBaker.common(r_i, s_i, l_i, v_i)
            self.evaluated[node.outputs[0]] = BakedImg(out, p)
            return True
        elif isinstance(node, bpy.types.ShaderNodeGamma):
            c_i = self.get(node, 'Color')
            g_i = self.get(node, 'Gamma')
            if c_i is None or g_i is None:
                return False
            c = MaterialBaker.tonp(c_i)
            g = MaterialBaker.tonp(g_i)
            img_c = gamma(c, g)
            p = MaterialBaker.common(c_i, g_i)
            self.evaluated[node.outputs[0]] = BakedImg(img_c, p)
            return True
        elif isinstance(node, bpy.types.ShaderNodeTexImage):
            normal = node.inputs['Vector']
            if normal.is_linked:
                return False
            a_soc = node.outputs["Alpha"]
            c_soc = node.outputs['Color']
            from PIL import Image
            path = bpy.path.abspath(node.image.filepath)
            i = np.array(Image.open(path)) / np.float32(255)
            col = i[:, :, :3]
            if i.shape[2] > 3:
                alpha = i[:, :, 3]
            else:
                alpha = 1
            path = os.path.basename(node.image.filepath)
            self.evaluated[c_soc] = BakedImg(col, path, node.image)
            self.evaluated[a_soc] = BakedImg(alpha, path, node.image)
            return True
        elif isinstance(node, bpy.types.ShaderNodeMix):
            a_i = self.get(node, 'A')
            b_i = self.get(node, 'B')
            alpha_i = self.get(node, 'Factor')
            if a_i is None or b_i is None or alpha_i is None:
                return False
            p = MaterialBaker.common(a_i,b_i,alpha_i)
            a = MaterialBaker.tonp(a_i)
            b = MaterialBaker.tonp(b_i)
            alpha = MaterialBaker.tonp(alpha_i)
            t0, t1 = 3000, 1000
            # print(node, "[0] a=",a[t0, t1] if a.ndim>=2 else a, "b=",b[t0, t1] if b.ndim>=2 else b, "alpha=",alpha[t0, t1] if alpha.ndim>=2 else alpha)
            max_channels = max(MaterialBaker.channels(a), MaterialBaker.channels(b), MaterialBaker.channels(alpha))
            a = MaterialBaker.to_channels(a, max_channels)
            b = MaterialBaker.to_channels(b, max_channels)
            alpha = MaterialBaker.to_channels(alpha, max_channels)
            max_height = max(MaterialBaker.size(a, 0), MaterialBaker.size(b, 0), MaterialBaker.size(alpha, 0))
            max_width = max(MaterialBaker.size(a, 1), MaterialBaker.size(b, 1), MaterialBaker.size(alpha, 1))
            a = MaterialBaker.to_size(a, max_height, max_width)
            b = MaterialBaker.to_size(b, max_height, max_width)
            alpha = MaterialBaker.to_size(alpha, max_height, max_width)
            # original implementation
            # https://projects.blender.org/blender/blender/src/branch/main/source/blender/gpu/shaders/material/gpu_shader_material_mix_color.glsl
            if node.blend_type == 'MIX':
                img_c = lerp(a, b, alpha)
                self.evaluated[node.outputs[0]] = BakedImg(img_c, p)
                return True
            elif node.blend_type == 'DARKEN':
                img_c = lerp(a, np.minimum(a, b), alpha)
                reset_alpha_channel(img_c, a)
                self.evaluated[node.outputs[0]] =  BakedImg(img_c, p)
                return True
            elif node.blend_type == 'LIGHTEN':
                img_c = lerp(a, np.maximum(a, b), alpha)
                reset_alpha_channel(img_c, a)
                self.evaluated[node.outputs[0]] =  BakedImg(img_c, p)
                return True
            elif node.blend_type == 'DODGE':
                img_c = dodge(a, b)
                reset_alpha_channel(img_c, a)
                self.evaluated[node.outputs[0]] =  BakedImg(img_c, p)
                return True
            elif node.blend_type == 'BURN':
                img_c = burn(a,b,alpha)
                reset_alpha_channel(img_c, a)
                self.evaluated[node.outputs[0]] =  BakedImg(img_c, p)
                return True
            elif node.blend_type == 'SCREEN':
                outcol = screen(a,b,alpha)
                reset_alpha_channel(outcol, a)
                self.evaluated[node.outputs[0]] =  BakedImg(outcol, p)
                return True
            elif node.blend_type == 'OVERLAY':
                img_c = overlay(a,b,alpha)
                self.evaluated[node.outputs[0]] =  BakedImg(img_c, p)
                return True
            elif node.blend_type == 'ADD':
                img_c = a + b * alpha
                reset_alpha_channel(img_c, a)
                self.evaluated[node.outputs[0]] =  BakedImg(img_c, p)
                return True
            elif node.blend_type == 'MULTIPLY':
                img_c = alpha_multiply(a,b,alpha)
                reset_alpha_channel(img_c, a)
                self.evaluated[node.outputs[0]] =  BakedImg(img_c, p)
                return True
        return False

class NodesUtils:
    @staticmethod
    def walk_backwards(input_socket, visited, lambda_func, used_inp_socks=None):
        for link in input_socket.links:
            node = link.from_node
            if node not in visited:
                visited.add(node)
                if isinstance(node, bpy.types.ShaderNodeGroup):
                    for group_output in NodesUtils.find_all_by_type(node.node_tree, bpy.types.NodeGroupOutput):
                        used_inp_socks_nested = set()
                        visited_nested = set()
                        for i in group_output.inputs:
                            NodesUtils.walk_backwards(i, visited_nested, lambda_func, used_inp_socks_nested)
                elif isinstance(node, bpy.types.NodeGroupInput):
                    used_inp_socks.add(link.from_socket.name)
                else:
                    lambda_func(node, link.from_socket)
                    for i in node.inputs:
                        NodesUtils.walk_backwards(i, visited, lambda_func, used_inp_socks)

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


NIRV_ZERO_EYES_DAZ_DIR = "/data/nirvana/nirv zero/nirv zero eyes"


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
        return self.check_gender(is_female) and self.check_figure(figure) and self.check_category(cat) and self.check_profile(prof)

class MorphsStore:

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
            'BREAST': "Custom/Breasts",
            'HEAD': "Custom/Head",
            'ARMS': "Custom/Arms",
            'LEGS': "Custom/Legs",
            'BODY': "Custom/Body",
            'ASS': "Custom/Ass",
            'GENITALS': "Custom/Genitals",
            'SPECIAL': "Custom/Special",
            'FACS': "Facs",
            'FACSEXPR': "Facsexpr",
            'FACSDET': "Facsdetails",
            'JCM': "JCM",
        }
        self.CAT_SETS = {
            'BODY': {'BREAST', 'HEAD', 'ARMS', 'LEGS','BODY', 'ASS', 'GENITALS', 'SPECIAL'},
            'FACS': {'FACS', 'FACSDET', 'FACSEXPR'},
            'GENITALS': {'GENITALS'},
            'JCM': {'JCM'},
            'SPECIAL': {'SPECIAL'},
            'ALL': None,
        }

        self.profile = 10
        self.file_name = ''
        self.morphs: {str: {str: Morph}} = {}
        self.GENERATE_MORPHS_FOR_CLOTHES = True
        self.GENERATE_MORPHS_FOR_HAIR = True

    def load_file(self, file_name=None):
        def process_gender(morphs, m, is_male=False, is_female=False):
            for morph_name, morph_meta in m.items():
                if morph_name not in morphs:
                    morph = morphs[morph_name] = Morph(morph_name)
                else:
                    morph = morphs[morph_name]
                morph.is_male = morph.is_male or is_male
                morph.is_female = morph.is_female or is_female
                profile_meta = morph_meta.get('profile', 9999)
                morph.profile = min(profile_meta, morph.profile)
                fig_meta = self.FIGURES.get(morph_meta.get('figure'), '')
                morph.figure = ''.join(set(fig_meta+morph.figure))
                morph.title = morph_meta.get('name', morph.title)
                morph.category = morph_meta.get('category', morph.category)
        if file_name is None:
            file_name = bpy.types.Scene.morphs_file
        if self.file_name != file_name:
            p = get_morphs_path(file_name)
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

    def get_allowed_morph_prefixes(self):
        if bpy.context.scene.get('is_nirv_zero'):
            return ["BaseAnime_", "Nirv_Zero_BaseAnim_", "Nirv_zero_", "Nirv_Zero_", "Nirv_"]
        elif bpy.context.scene.get('daz_optim_toon'):
            return ["BaseAnime_"]
        return []

    def get_figure(self):
        return self.FIGURES.get('FIGURE_TOON' if bpy.context.scene.get('daz_optim_toon') else 'FIGURE_G9')

    def collect_fav_shape_keys(self, categories_to_include: {str}, profiles_to_include: int)->{str:Morph}:
        is_fem = DazOptimizer.is_female()
        shape_keys = {}
        figure = self.get_figure()
        for obj in bpy.data.objects:
            if isinstance(obj.data, bpy.types.Mesh):
                if not self.GENERATE_MORPHS_FOR_CLOTHES and is_clothes(obj):
                    continue
                if not self.GENERATE_MORPHS_FOR_HAIR and is_hair(obj):
                    continue
                if is_cum(obj):
                    continue
                daz_dir = obj_daz_dir(obj).lower()
                morphs_for_daz_obj: {str: Morph} = self.morphs.get(daz_dir)
                if morphs_for_daz_obj is not None:
                    for morph in morphs_for_daz_obj.values():
                        if morph.check(is_fem, figure, categories_to_include, profiles_to_include):
                            shape_keys[morph.shape_key] = morph
        return shape_keys

    def make_fav_morphs_list(self, fav_morphs_path, categories_to_include: {str}, profiles_to_include: int, load_all_conflicting_morphs=True):
        content_dirs = get_daz_content_dirs()
        content_dirs = [d[:-1] if d.endswith("/") or d.endswith("\\") else d for d in content_dirs]
        morph_prefixes = self.get_allowed_morph_prefixes()
        morph_prefixes_regex = re.compile(r"("+"|".join(morph_prefixes)+")?(.+)\.dsf")
        shape_keys = self.collect_fav_shape_keys(categories_to_include, profiles_to_include)
        fav_morphs = {
            "filetype": "favo_morphs",
            "root_paths": content_dirs,
        }
        for obj in bpy.data.objects:
            if not isinstance(obj.data, bpy.types.Mesh):
                continue
            if not self.GENERATE_MORPHS_FOR_CLOTHES and is_clothes(obj):
                continue
            if not self.GENERATE_MORPHS_FOR_HAIR and is_hair(obj):
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
                                meta: Morph = shape_keys.get(shape_key_name)
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
        with open(fav_morphs_path, 'w+') as f:
            json.dump(fav_morphs, f, indent=2)

class ClothesMeta:

    def __init__(self, item, s:'ClothesStore'):
        self.fingerprint = item['fingerprint']
        self.skin_tight = item.get('skin_tight', -1)
        self.skin_tight = s.constants.get(self.skin_tight, self.skin_tight)
        self.panties = item.get('panties', False)
        self.obj = None
        self.dont_resize = item.get('dont_resize', False)

class ClothesStore:
    def __init__(self):
        self.EXTRUDED_SK_NAME = 'extruded'
        self.constants = {
            'CLOTHES_MIN_DIST_TO_SKIN': 0.004,
            'PANTIE_SCALING': 0.02,
        }
        clothes_path = get_clothes_path('all')
        with open(clothes_path, 'r') as f:
            clothes = json.load(f)
        self.metas = {k:ClothesMeta(v, self) for k,v in clothes.items()}

    def get(self, name):
        return self.metas.get(name)

    def items(self):
        return self.metas.items()

    def __getitem__(self, item):
        return self.metas[item]


#

class BoneRelation:
    def __init__(self, j):
        self.start = j['start']
        self.tail = j['tail']
        self.x_axis = j['x_axis']
        self.y_axis = j['y_axis']
        self.z_axis = j['z_axis']
        self.roll = j['roll']
        self.parent_name = j.get('parent_name')

def load_bone_hierarchy(name):
    p = get_resource_path(name+'.json', 'boneh')
    with open(p, 'r') as f:
        j = json.load(f)
        return {k: BoneRelation(v) for k,v in j.items()}

def print_bone_hierarchy(armature='root'):
    if isinstance(armature, str):
        armature = bpy.data.armatures[armature]
    return {b.name: {
        'head':list(b.head),
        'tail':list(b.tail),
        'x_axis': list(b.x_axis),
        'y_axis': list(b.y_axis),
        'z_axis': list(b.z_axis),
        'roll': b.roll,
        'parent_name': b.parent.name if b.parent is not None else None
    } for b in armature.edit_bones}

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
    return not x.dont_resize and x.skin_tight>=0 and not x.panties

def is_not_skin_tight(x):
    return not x.dont_resize and x.skin_tight < 0

def is_panties(x):
    return not x.dont_resize and x.panties

def find_all_panties():
    return find_all_clothes(is_panties)

def find_all_skin_tight_clothes():
    return find_all_clothes(is_skin_tight)

def find_all_non_skin_tight_clothes():
    return find_all_clothes(is_not_skin_tight)

def is_hair(obj):
    l = obj.name.lower()
    return 'hair' in l or 'ponytail' in l

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


class AdditionalBone:
    def __init__(self, j):
        self.head = j["head"]
        self.tail = j["tail"]
        self.parent = j["parent"]
        self.connect = j["connect"]
        self.local = j["local"]
        self.weights = j["weights"]
        self.indices = j["indices"]

def load_additional_bones():
    ab = collect_resource_paths('additional_bones', '.json')
    abs = {}
    for f in ab:
        file_name = os.path.basename(f)[:-len(".json")]
        with open(f, 'r') as f:
            abs[file_name] = AdditionalBone(json.load(f))
    return abs


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

def get_resource_path(file_name, dir_name):
    n = file_name
    p = bpy.path.abspath('//'+dir_name+'/' + n)
    if os.path.exists(p):
        return p
    p = bpy.path.abspath('//../'+dir_name+'/' + n)
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
    dirs = [bpy.path.abspath('//'+dir_name+'/'),
         bpy.path.abspath('//../'+dir_name+'/'),
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


def get_masks_path(file_name):
    return get_resource_path(file_name+'.npy', 'masks')


def get_morphs_path(file_name):
    return get_resource_path(file_name+'.json', 'morphs')


def get_clothes_path(file_name):
    return get_resource_path(file_name + '.json', 'clothes')


def get_additional_bones_path(file_name):
    return get_resource_path(file_name + '.json', 'additional_bones')


def get_eyebrows_and_eyelashes_path():
    return get_resource_path('eyebrows_and_eyelashes.png', 'assets')


def load_mask(file_name):
    return np.load(get_masks_path(file_name))


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
    if body.data.shape_keys is not None:
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


HEAD_RLE = load_mask('head_rle')
BODY_RLE = load_mask('body_rle')
LEFT_LEG_RLE = load_mask('left_leg_rle')
RIGHT_LEG_RLE = load_mask('right_leg_rle')
BUTT_RLE = load_mask('butt_rle')
BOT_ARM_RLE = load_mask('bot_arm_rle')
LIP_RLE = load_mask('lip_rle')
TOP_ARM_RLE = load_mask('top_arm_rle')
EYELASHES_RLE = load_mask('eyelashes_rle')
EYE_SOCKET_RLE = load_mask('eye_socket_rle')
MOUTH_CAVITY_RLE = load_mask('mouth_cavity_rle')
TOON_MOUTH_RLE = load_mask('toon_mouth_rle')

TRANSPARENT_TOON_EYEBROWS_MAT_NAME = "ToonEyebrows"
TRANSPARENT_TOON_EYELASHES_MAT_NAME = "ToonEyelashes"

BREAST_GEOGRAFTS = ['BreastacularG9', 'Body Geo', 'STX Gen 9 Nipples Feminine']
DICK_GEOGRAFTS = ['Genesis 9 Anatomical Elements Male']
MALE_ONLY_GEOGRAFTS = DICK_GEOGRAFTS
FEMALE_ONLY_GEOGRAFTS = ['GoldenPalace_G9', 'Wet Kitty TOON'] + BREAST_GEOGRAFTS
GEOGRAFTS = FEMALE_ONLY_GEOGRAFTS + MALE_ONLY_GEOGRAFTS


MORPHS = MorphsStore()

CLOTHES = ClothesStore()

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

UE5_QUINN_BONE_HIERARCHY = load_bone_hierarchy('quinn')
UE5_MANNY_BONE_HIERARCHY = load_bone_hierarchy('manny')
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
ADDITIONAL_BONES = load_additional_bones()



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
        body = self.get_body_mesh()
        for trash in ['BaseShortsGeoGraft Mesh']:
            trash = bpy.data.objects.get(trash)
            if trash is not None:
                bpy.data.objects.remove(trash)
        for clothing_item in find_all_clothes():
            sub_clothes = find_child_meshes(clothing_item.obj)
            if len(sub_clothes)>0:
                transfer_weights(body, sub_clothes)
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

        def find_textures(i):
            visited = set()
            outputs = []
            def callback(node):
                if isinstance(node, bpy.types.ShaderNodeTexImage):
                    outputs.add(node.image)
            NodesUtils.walk_backwards(i, visited, callback)
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
                            for image in find_textures(bsdf.inputs[channel]):
                                body_part_filepaths[channel].add(image)
                                print(body_part, channel, image)
                    elif bsdf.node_tree.name == 'DAZ Dual Lobe PBR':
                        for image in find_textures(bsdf.inputs['Roughness 1']):
                            body_part_filepaths['Roughness'].add(image)
                            print(body_part, "Roughness", image)
                        for image in find_textures(bsdf.inputs['Roughness 2']):
                            body_part_filepaths['Roughness'].add(image)
                            print(body_part, "Roughness", image)
                        for image in find_textures(bsdf.inputs['Normal']):
                            body_part_filepaths['Normal'].add(image)
                            print(body_part, "Normal", image)
                    elif bsdf.node_tree.name == 'DAZ Toon Diffuse':
                        clr_soc = bsdf.inputs['Color']
                        if len(clr_soc.links) == 0:
                            const_color_value = np.array(clr_soc.default_value)
                        else:
                            for image in find_textures(bsdf.inputs['Color']):
                                body_part_filepaths['Base Color'].add(image)
                                print(body_part,"Base Color",image)
                        for image in find_textures(bsdf.inputs['Normal']):
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


    def load_fav_morphs(self):
        mesh = self.get_body_mesh()
        select_object(mesh)

        #
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
        if body.data.shape_keys is None:
            bpy.ops.object.shape_key_add(from_mix=False)
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

    @staticmethod
    def is_female():
        return bool(bpy.context.scene['daz_optim_female'])

    @staticmethod
    def get_hierarchy():
        quinn = DazOptimizer.is_female()
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
        height = body_mesh.dimensions[2]
        ue5_height = QUINN_HEIGHT if self.is_female() else MANNY_HEIGHT
        scl = height / ue5_height


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
                    daz_bone.head.z *= scl
                    daz_bone.tail.z *= scl
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
        for workspace in bpy.data.workspaces:
            for screen in workspace.screens:
                for area in screen.areas:
                    if area.type == 'VIEW_3D':
                        for space in area.spaces:
                            if space.type == 'VIEW_3D':
                                space.clip_start = 1
                                space.clip_end = 100000
        self.scale(z)

    def scale_to_quinn(self):
        mesh = self.get_body_mesh()
        height = mesh.dimensions[2]
        ue5_height = QUINN_HEIGHT if self.is_female() else MANNY_HEIGHT
        rig = self.get_body_rig()
        select_object(rig)
        scl = ue5_height / height
        scl = (scl, scl, scl)
        apply_recursive(rig, scale=scl)

    def translate_to_quinn(self):
        mesh = self.get_body_mesh()
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
            loc = (0, ue5_pevis_pos[1] / 100 - root.head.y, 0)
        apply_recursive(rig, location=loc)

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

    @staticmethod
    def save_weights_and_bones_of_selected_obj():
        bpy.context.selected_objects


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


class LoadMorphs(bpy.types.Operator):
    """ load morphs """
    bl_idname = "dazoptim.load_morphs"
    bl_label = "Load morphs"
    bl_options = {"REGISTER", "UNDO"}
    stage_id = 'R'

    @classmethod
    def poll(cls, context):
        return UNLOCK or check_stage(context, [DazSaveBlend_operator], [DazMergeGrografts_operator])

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

    def on_register(self):
        pass

    def on_unregister(self):
        pass

    def draw(self, p: DazOptimize_sidebar, col, context, idx: int):
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
    def __init__(self, name: str, prop_type: type, default_value):
        self.name = name
        self.prop_type = prop_type
        self.default_value = default_value

    def on_register(self):
        if self.prop_type is float:
            cons = bpy.props.FloatProperty
        elif self.prop_type is str:
            cons = bpy.props.StringProperty
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

    def draw(self, p: DazOptimize_sidebar, col, context, idx: int):
        p.props[self.name] = col.prop(context.scene, self.name)
        return idx, col


class EntryFileEnumProp:
    def __init__(self, name: str, dir_name: str, extension = ".json"):
        self.name = name
        self.extension = extension
        self.dir_name = dir_name



    def on_register(self):
        self.prop = bpy.props.EnumProperty(
            name=self.name,
            items=lambda x, y:[(i, i, i) for i in collect_resource_files(self.dir_name, self.extension)],
        )
        setattr(bpy.types.Scene, self.name, self.prop)

    def on_unregister(self):
        delattr(bpy.types.Scene, self.name)

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
    ]),
    EntryProp('morph_level', list,[
        ("ALL",'full', "Loads morphs at all profile levels"),
        ("MID",'medium', "Loads morphs at medium profile level"),
        ("MIN",'minimal', "Loads only morphs at minimal profile level"),
    ]),
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
