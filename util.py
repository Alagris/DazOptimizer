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
    def __init__(self, nump, path, color_space, img: bpy.types.Image=None):
        self.np = nump
        self.image = img
        self.path = path
        self.uv_map = None
        self.color_space = color_space

    def __repr__(self):
        return (self.path if self.image is None else self.image.filepath) + ':' + self.color_space

    def to_numpy(self):
        return self.np

    def to_image(self, path):
        if self.image is None:
            if self.color_space == "sRGB":
                raw = linearrgb_to_srgb(self.np)
            else:
                raw = self.np
            img = np_to_pil(raw)
            img.save(path)
            self.image = bpy.data.images.load(path)
            self.image.colorspace_settings.name = self.color_space
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


def linearrgb_to_srgb(c):
    assert c.ndim == 2 or 3 <= c.shape[2] <= 4, c.shape
    rgb = c[:,:,:3] if c.ndim > 2 else c
    mask = rgb < 0.0031308
    neg_mask = rgb < 0
    rgb[mask] = 12.92 * rgb[mask]
    not_mask = ~mask
    rgb[not_mask] = 1.055 * (rgb[not_mask] ** (1 / 2.4)) - 0.055
    rgb[neg_mask] = 0
    return c


def srgb_to_linearrgb(c):
    assert c.ndim == 2 or 3 <= c.shape[2] <= 4, c.shape
    rgb = c[:,:,:3] if c.ndim > 2 else c
    mask = rgb >= 0.04045
    rgb[mask] = ((rgb[mask] + 0.055) / 1.055) ** 2.4
    rgb[~mask] = rgb[~mask] / 12.92
    return rgb

def gamma(color, gamma_value):
    return np.power(color, gamma_value)

def lerp(a,b,alpha):
    return a * (1-alpha) + b * alpha

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
    def __init__(self, node_tree, debug=0):
        if isinstance(node_tree, str):
            node_tree = bpy.data.materials[node_tree]
        if isinstance(node_tree, bpy.types.Material):
            self.material = node_tree
            node_tree = node_tree.node_tree
        else:
            self.material = None
        assert isinstance(node_tree, bpy.types.ShaderNodeTree)
        self.debug = debug
        self.node_tree = node_tree
        self.evaluated = {}
        self.inputs = None
        self.outputs = None
        self.whitelist = None

    def __repr__(self):
        return repr(self.node_tree) if self.material is None else self.material.name

    @staticmethod
    def tonp(x):
        assert x is not None
        if isinstance(x, BakedImg):
            return x.to_numpy()
        return np.array(x)

    @staticmethod
    def topa(x):
        assert x is not None
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
    def common_cs(*args):
        for e in args:
            if isinstance(e, BakedImg) and e.color_space != 'Non-Color':
                return e.color_space
        return 'Non-Color'

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
                    if l.from_node not in visited:
                        visited.add(l.from_node)
                        sort_topologically_recursion(l.from_node)
            sorted_nodes.append(node)

        for end in MaterialBaker.get_all_final(node_tree):
            sort_topologically_recursion(end)

        return sorted_nodes

    def bake(self, debug=False):
        if debug:
            print('    '*self.debug + 'Baking ', self)
        nodes = MaterialBaker.sort_topologically(self.node_tree)
        for node in nodes:
            if self.whitelist is not None and node not in self.whitelist:
                continue
            if debug:
                print('    '*(self.debug+1)+repr(node))
                for i in node.inputs:
                    if not i.is_unavailable:
                        print('    '*(self.debug+2) +i.name+"="+repr(self.get(node, i)))
            baked = self.evaluate(node, debug)
            if debug:
                print('    '*(self.debug+2)+'baked:'+str(baked))

    def apply(self):
        to_remove = set()
        print("evaluated=", {e.node.name+'->'+e.name for e in self.evaluated.keys()})
        for node in self.node_tree.nodes:
            if self.whitelist is not None and node not in self.whitelist:
                continue
            if len(node.outputs)>0: # this if guards against removing Material Output node
                all_outs_baked = True
                for o in node.outputs:
                    if not o.is_unavailable and o not in self.evaluated:
                        all_outs_baked = False
                        print("Keeping "+node.name+" because of "+o.name+" ("+repr(o)+")")
                        break
                if all_outs_baked:
                    to_remove.add(node)
        print("to_remove=", {e.name for e in to_remove})
        to_insert_baked = set()
        for node in self.node_tree.nodes:
            for o in node.outputs:
                if o in self.evaluated:
                    any_link_needed = False
                    for l in o.links:
                        if l.to_node not in to_remove:
                            any_link_needed = True
                            break
                    if any_link_needed:
                        to_insert_baked.add(o)
        print("to_insert_baked=", {e.node.name+"->"+e.name for e in to_insert_baked})
        for o in to_insert_baked:
            baked_o = self.evaluated.get(o)
            i_socs = [l.to_socket for l in o.links]
            if isinstance(baked_o, BakedImg):
                tex_node = self.node_tree.nodes.new('ShaderNodeTexImage')
                tex_node.name = 'baked '+o.node.name+' '+o.name
                tex_node.location = o.node.location
                mat_name = self.node_tree.name if self.material is None else self.material.name
                path = bpy.path.abspath('//baked/' + mat_name+"_"+o.node.name+' '+o.name + '.png')
                try:
                    os.makedirs(os.path.dirname(path))
                except OSError as e:
                    pass
                tex_node.image = baked_o.to_image(path)
                for i_soc in i_socs:
                    self.node_tree.links.new(i_soc, tex_node.outputs['Color'])
            else:
                for i_soc in i_socs:
                    i_soc.default_value = baked_o
        for node in to_remove:
            self.node_tree.nodes.remove(node)

    def get(self, node, socket, default_value=None):
        if isinstance(socket, (str,int)):
            try:
                socket = node.inputs[socket]
            except KeyError:
                return None
            except IndexError:
                return None
        if socket.is_unavailable:
            return default_value
        if socket.is_linked:
            if len(socket.links)==1:
                return self.evaluated.get(socket.links[0].from_socket)
            else:
                return None
        else:
            try:
                return np.array(socket.default_value)
            except AttributeError:
                return None

    def evaluate(self, node, debug=True):

        if isinstance(node, bpy.types.ShaderNodeRGB):
            o = node.outputs[0]
            rgb = linearrgb_to_srgb(o.default_value)
            self.evaluated[o] = rgb
            return True
        elif isinstance(node, bpy.types.ShaderNodeMath):
            a_i = self.get(node, 0)
            b_i = self.get(node, 1)
            c_i = self.get(node, 2)
            if a_i is None or b_i is None:
                return False
            p = MaterialBaker.common(a_i, b_i)
            cs = MaterialBaker.common_cs(a_i, b_i, c_i)
            a = MaterialBaker.tonp(a_i)
            b = MaterialBaker.tonp(b_i)
            if node.operation == "MULTIPLY":
                res = a * b
            elif node.operation == "MULTIPLY_ADD":
                if c_i is None:
                    return False
                c = MaterialBaker.tonp(c_i)
                res = a * b + c
            elif node.operation == "ADD":
                res = a + b
            elif node.operation == "SUBTRACT":
                res = a - b
            elif node.operation == "DIVIDE":
                res = a / b
            elif node.operation == "MODULO":
                res = a % b
            elif node.operation == "POWER":
                res = a ** b
            elif node.operation == "LOGARITHM":
                res = np.log(a, b)
            else:
                raise Exception('unimplemented bpy.types.ShaderNodeMath with operation '+node.operation)
            self.evaluated[node.outputs['Value']] = BakedImg(res, p, cs)
            return True
        elif isinstance(node, bpy.types.NodeGroupInput):
            if self.inputs is not None:
                for out in node.outputs:
                    self.evaluated[out] = self.inputs.get(out.name)
            return False # the input node is never removed after optimisation
        elif isinstance(node, bpy.types.NodeGroupOutput):
            if self.outputs is not None:
                for i in node.inputs:
                    optimised_i = self.get(node, i)
                    if optimised_i is not None:
                        self.outputs[i.name] = optimised_i
            return False # the input node is never removed after optimisation
        elif isinstance(node, bpy.types.ShaderNodeGroup):
            mb = MaterialBaker(node.node_tree, debug=self.debug+2)
            mb.inputs = {}
            for i in node.inputs:
                mb.inputs[i.name] = self.get(node, i)
            mb.outputs = {}
            mb.bake(debug=debug)
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
            v_i = self.get(node, 'Vector', 0)
            if l_i is None or s_i is None or r_i is None or v_i is None:
                return False
            l = MaterialBaker.tonp(l_i)
            s = MaterialBaker.tonp(s_i)
            r = MaterialBaker.tonp(r_i)
            if not np.all(r == 0):
                raise Exception("Encountered "+ repr(node)+ ' with  rotation! Not implemented yet!!!!')
            v = MaterialBaker.tonp(v_i)
            out = v * s + l
            p = MaterialBaker.common(r_i, s_i, l_i, v_i)
            cs = MaterialBaker.common_cs(r_i, s_i, l_i, v_i)
            self.evaluated[node.outputs['Vector']] = BakedImg(out, p, cs)
            return True
        elif isinstance(node, bpy.types.ShaderNodeValue):
            self.evaluated[node.outputs['Value']] = np.array(node.outputs[0].default_value)
        elif isinstance(node, bpy.types.ShaderNodeGamma):
            c_i = self.get(node, 'Color')
            g_i = self.get(node, 'Gamma')
            if c_i is None or g_i is None:
                return False
            c = MaterialBaker.tonp(c_i)
            g = MaterialBaker.tonp(g_i)
            img_c = gamma(c, g)
            p = MaterialBaker.common(c_i, g_i)
            cs = MaterialBaker.common_cs(c_i, g_i)
            self.evaluated[node.outputs['Color']] = BakedImg(img_c, p, cs)
            return True
        elif isinstance(node, bpy.types.ShaderNodeTexImage):
            if node.image is None:
                raise Exception(repr(node)+" has no image")
            normal = self.get(node, 'Vector')
            if normal is not None:
                if not np.all(normal==0):
                    return False
            a_soc = node.outputs["Alpha"]
            c_soc = node.outputs['Color']
            from PIL import Image
            path = bpy.path.abspath(node.image.filepath)
            i = np.array(Image.open(path)) / np.float32(255)
            if i.ndim > 2:
                col = i[:, :, :3]
                if i.shape[2] > 3:
                    alpha = i[:, :, 3]
                else:
                    alpha = 1
            else:
                col = i
                alpha = 1
            path = os.path.basename(node.image.filepath)
            c_space = node.image.colorspace_settings.name
            if c_space == 'sRGB':
                col = srgb_to_linearrgb(col)
            elif c_space == 'Non-Color':
                pass
            else:
                raise Exception('unsupported color space ' + c_space + ' in ' + repr(node))
            self.evaluated[c_soc] = BakedImg(col, path, c_space, node.image)
            self.evaluated[a_soc] = BakedImg(alpha, path, c_space, node.image)
            return True
        elif isinstance(node, bpy.types.ShaderNodeMix):
            a_i = self.get(node, 'A')
            b_i = self.get(node, 'B')
            alpha_i = self.get(node, 'Factor')
            if a_i is None or b_i is None or alpha_i is None:
                return False
            p = MaterialBaker.common(a_i,b_i,alpha_i)
            cs = MaterialBaker.common_cs(a_i, b_i, alpha_i)
            a = MaterialBaker.tonp(a_i)
            b = MaterialBaker.tonp(b_i)
            alpha = MaterialBaker.tonp(alpha_i)
            max_channels = max(MaterialBaker.channels(a), MaterialBaker.channels(b), MaterialBaker.channels(alpha))
            a = MaterialBaker.to_channels(a, max_channels)
            b = MaterialBaker.to_channels(b, max_channels)
            alpha = MaterialBaker.to_channels(alpha, max_channels)
            max_height = max(MaterialBaker.size(a, 0), MaterialBaker.size(b, 0), MaterialBaker.size(alpha, 0))
            max_width = max(MaterialBaker.size(a, 1), MaterialBaker.size(b, 1), MaterialBaker.size(alpha, 1))
            a = MaterialBaker.to_size(a, max_height, max_width)
            b = MaterialBaker.to_size(b, max_height, max_width)
            alpha = MaterialBaker.to_size(alpha, max_height, max_width)
            o = node.outputs['Result']
            # original implementation
            # https://projects.blender.org/blender/blender/src/branch/main/source/blender/gpu/shaders/material/gpu_shader_material_mix_color.glsl
            if node.blend_type == 'MIX':
                #import ipdb; ipdb.set_trace()
                img_c = lerp(a, b, alpha)
                self.evaluated[o] = BakedImg(img_c, p, cs)
                return True
            elif node.blend_type == 'DARKEN':
                img_c = lerp(a, np.minimum(a, b), alpha)
                reset_alpha_channel(img_c, a)
                self.evaluated[o] = BakedImg(img_c, p, cs)
                return True
            elif node.blend_type == 'LIGHTEN':
                img_c = lerp(a, np.maximum(a, b), alpha)
                reset_alpha_channel(img_c, a)
                self.evaluated[o] = BakedImg(img_c, p, cs)
                return True
            elif node.blend_type == 'DODGE':
                img_c = dodge(a, b)
                reset_alpha_channel(img_c, a)
                self.evaluated[o] = BakedImg(img_c, p, cs)
                return True
            elif node.blend_type == 'BURN':
                img_c = burn(a, b, alpha)
                reset_alpha_channel(img_c, a)
                self.evaluated[o] = BakedImg(img_c, p, cs)
                return True
            elif node.blend_type == 'SCREEN':
                outcol = screen(a, b, alpha)
                reset_alpha_channel(outcol, a)
                self.evaluated[o] = BakedImg(outcol, p, cs)
                return True
            elif node.blend_type == 'OVERLAY':
                img_c = overlay(a, b, alpha)
                self.evaluated[o] = BakedImg(img_c, p, cs)
                return True
            elif node.blend_type == 'ADD':
                img_c = a + b * alpha
                reset_alpha_channel(img_c, a)
                self.evaluated[o] = BakedImg(img_c, p, cs)
                return True
            elif node.blend_type == 'MULTIPLY':
                img_c = alpha_multiply(a, b, alpha)
                reset_alpha_channel(img_c, a)
                self.evaluated[o] = BakedImg(img_c, p, cs)
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
    def find_textures(inp_socket):
        visited = set()
        outputs = []
        def callback(node, input_soc):
            if isinstance(node, bpy.types.ShaderNodeTexImage):
                outputs.append(node.image)
        NodesUtils.walk_backwards(inp_socket, visited, callback)
        return outputs

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
    def collect_all_before_socket(input_socket, outputs):
        for link in input_socket.links:
            NodesUtils.collect_all_before(link.from_node, outputs)

    @staticmethod
    def is_used(node):
        for o in node.outputs:
            if o.is_linked:
                return True
        return False

    @staticmethod
    def delete_unused(node_tree):
        to_visit = set()
        unused = set()
        for node in node_tree.nodes:
            if not isinstance(node, (bpy.types.ShaderNodeOutputMaterial, bpy.types.ShaderNodeOutputAOV)) and not NodesUtils.is_used(node):
                unused.add(node)
                to_visit.add(node)
        while len(to_visit)>0:
            node = to_visit.pop()
            is_used = False
            for o in node.outputs:
                for l in o.links:
                    if l.to_node not in unused:
                        is_used = True
                        break
            if not is_used:
                for input_socket in node.inputs:
                    for l in input_socket.links:
                        to_visit.add(l.from_node)
                unused.add(node)
        for unused_node in unused:
            node_tree.nodes.remove(unused_node)



    @staticmethod
    def delete_all_before(node_tree, node, inclusive=True):
        nodes_before = set()
        NodesUtils.collect_all_before(node, nodes_before)
        if not inclusive:
            nodes_before.remove(node)
        for node in nodes_before:
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
    def gen_simple_material(node_tree, filepaths, shift_x=0, uvs=None, clear_all=False, keep_shells=True):
        print("Simplifying ", node_tree)
        if isinstance(node_tree, bpy.types.Material):
            node_tree = node_tree.node_tree
        ns = node_tree.nodes
        ls = node_tree.links

        output_node = NodesUtils.find_by_type(node_tree, bpy.types.ShaderNodeOutputMaterial)
        if output_node is None:
            output_node = ns.new('ShaderNodeOutputMaterial')
        output_node.location = (shift_x+400, 0)
        output_socket = output_node.inputs['Surface']
        if keep_shells:
            while output_socket.is_linked:
                prev_node = output_socket.links[0].from_node
                if isinstance(prev_node, bpy.types.ShaderNodeGroup) and 'Shell' in prev_node.node_tree.name and 'BSDF' in prev_node.inputs:
                    output_node = prev_node
                    output_socket = prev_node.inputs['BSDF']
                else:
                    break
        else:
            NodesUtils.remove_all_input_links(node_tree, output_node)

        is_toon = bpy.context.scene.get('daz_optim_toon')
        if clear_all:
            if is_toon:
                bsdf_node = None
                ns.clear()
                NodesUtils.delete_all_before(node_tree, output_node, inclusive=False)
            else:
                bsdf_node = NodesUtils.find_by_type(node_tree, bpy.types.ShaderNodeBsdfPrincipled)
                if bsdf_node is None:
                    NodesUtils.delete_all_before(node_tree, output_node, inclusive=False)
                else:
                    NodesUtils.delete_all_before(node_tree, bsdf_node, inclusive=False)


        if is_toon:
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
            if bsdf_node is None:
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
        NodesUtils.delete_unused(node_tree)

    @staticmethod
    def remove_all_input_links(node_tree, node):
        for i in  node.inputs:
            for l in i.links:
                node_tree.links.remove(l)


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

def get_fingerprint(o):
    if isinstance(o, bpy.types.Object):
        o = o.data
    if isinstance(o, bpy.types.Mesh):
        return o.daz_importer.DazFingerPrint
    return None

def find_by_fingerprint(fingerprint):
    for o in bpy.data.objects:
        if isinstance(o.data, bpy.types.Mesh):
            mesh = o.data
            if mesh.daz_importer.DazFingerPrint == fingerprint:
                return o
    return None

def find_all_fingerprints():
    s = {}
    for o in bpy.data.meshes:
        try:
            fp = o.daz_importer.DazFingerPrint
            s[fp] = o
        except AttributeError:
            pass
    return s

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


