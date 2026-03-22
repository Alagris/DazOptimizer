
from .util import *
from .assets import *
from .constants import *
import io
import os
import re
import shutil
import sys

import base64
import urllib
import mathutils
import numpy as np
import bpy
import bmesh
import json
from collections import namedtuple

class DazOptimizer:

    def __init__(self, workdir=None, name=None):
        if workdir is None:
            workdir = os.path.dirname(bpy.data.filepath)
        if name is None:
            name = os.path.basename(bpy.data.filepath)
            assert '.' in name, '. not in '+name
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
            n = re.sub(r' \([0-9]+\) ',' ',b.name)
            if n.endswith(suffix):
                return b

    def get_eyelashes_mesh(self):
        return DazOptimizer.get_mesh_by_name('Eyelashes Mesh')

    def get_base_uv_layer(self, layer_name='Base Multi UDIM'):
        return self.get_body_mesh().data.uv_layers[layer_name]

    def get_base_uv_layer_np(self, layer_name='Base Multi UDIM'):
        return np.array([v.uv for v in self.get_base_uv_layer(layer_name=layer_name).data])

    def get_pixel_coords(self, layer_name='Base Multi UDIM'):
        return DazOptimizer.base_layer_to_pixel_coords(self.get_base_uv_layer_np(layer_name))

    def update_base_uv_layer(self, base_layer_np: np.ndarray):
        for v, new_uv in zip(self.get_base_uv_layer().data, base_layer_np):
            v.uv = new_uv

    def get_concat_image_path(self, map_type):
        return os.path.join(self.workdir, self.name + '_' + map_type + '.png')

    def get_simplified_eyes_image_path(self, map_type):
        return os.path.join(self.workdir, self.name + '_' + map_type + '_eyes.png')

    def get_eyebrows(self, but_not_eyebrows_mesh=False):
        brows = None
        for o in bpy.data.objects:
            if but_not_eyebrows_mesh and o.name == 'Eyebrows Mesh':
                continue
            if o.name.endswith(" Mesh"):
                n = o.name.lower()
                if 'eyebrows' in n:
                    return o
                if 'brows' in n:
                    brows = o
        if brows is not None:
            return brows
        if bpy.context.scene.get('daz_optim_toon'):
            for o in bpy.data.objects:
                if 'toon brows' in o.name.lower() and o.name.endswith(" Mesh"):
                    return o

    def remove_old_eyebrows(self):
        m = self.get_eyebrows(True)
        if m is not None:
            bpy.data.objects.remove(m)

    def optimize_eyebrows(self):

        EYEBROWS_M = self.get_eyebrows()
        if bpy.context.scene.get('daz_optim_toon'):
            mats = [m for m in EYEBROWS_M.data.materials if not NodesUtils.contains_subgroup(m, "DAZ Transparent")]
            DazOptimizer.gen_simple_materials(mats)
        else:
            offset = 1.6009911569682034
            #assert EYEBROWS_M is not None
            if EYEBROWS_M is not None:
                offset = 0
                sample_points = 10
                for i in np.random.randint(0, len(EYEBROWS_M.data.vertices), sample_points):
                    offset += EYEBROWS_M.data.vertices[i].co.z
                offset /= sample_points
                offset_vector = mathutils.Vector((0,0,offset))
                offset_vector = EYEBROWS_M.matrix_world @ offset_vector
                print('offset=', offset_vector)
            else:
                offset_vector = mathutils.Vector((0, offset, 0))

            vertices = np.array([[0.020655272528529167, -0.09718257188796997, 0.0037765231999484783], [0.010495096445083618, -0.09829649329185486, 0.001051875677975822], [0.0246686190366745, -0.09323396533727646, -0.004316237839785408], [0.013008052483201027, -0.0945364311337471, -0.008372691544619393], [0.028579287230968475, -0.0950171947479248, 0.006468268958005119], [0.031095163896679878, -0.09265439212322235, -0.002452758225527596], [0.038159459829330444, -0.09091758728027344, 0.006650659171017814], [0.03818078339099884, -0.09061393141746521, -0.0010096105662258381], [0.04382877051830292, -0.08715396374464035, 0.006490084257992912], [0.043936073780059814, -0.08735480159521103, -0.0009910139170559162], [0.049159739166498184, -0.08238893747329712, 0.00556966933337133], [0.04852811247110367, -0.08362218737602234, -0.0018033060160549397], [0.05282459035515785, -0.07811952382326126, 0.0038766590031711345], [0.05235077813267708, -0.07928386330604553, -0.0033418211069973225], [0.055806536227464676, -0.07358748465776443, 0.0018968311223117595], [0.055413272231817245, -0.07450003176927567, -0.0054763826456936116], [0.058321163058280945, -0.06856732070446014, -0.0005002292719753498], [0.05764066427946091, -0.070110023021698, -0.007665303620425057], [0.061125967651605606, -0.06174656003713608, -0.003270057114687752], [0.06067896634340286, -0.06334099173545837, -0.010448244484988045], [0.01933024451136589, -0.09739914536476135, 0.007336351004513908], [0.010570534504950047, -0.09852366894483566, 0.004864307967099357], [-0.0203605554997921, -0.09722965955734253, 0.0038549629124728924], [-0.010192444548010826, -0.09832023829221725, 0.0010914531621066814], [-0.024334117770195007, -0.09329022467136383, -0.004222420128908944], [-0.012660950422286987, -0.09456589818000793, -0.008323696526614022], [-0.028289951384067535, -0.09508249908685684, 0.006577107039364982], [-0.030766494572162628, -0.09272542595863342, -0.00233438340100367], [-0.03786155581474304, -0.09100489318370819, 0.006796094504269767], [-0.037853024899959564, -0.09070125222206116, -0.000864175232973885], [-0.04352172836661339, -0.08725428581237793, 0.006657334891232658], [-0.04360099509358406, -0.08745533227920532, -0.0008235248652370686], [-0.04883836582303047, -0.08250148594379425, 0.0057573047551242595], [-0.04818148538470268, -0.08373325318098068, -0.0016181739893825764], [-0.05248706415295601, -0.07824047654867172, 0.004078361121091056], [-0.05198841169476509, -0.07940369844436646, -0.003142026337710213], [-0.055451150983572006, -0.07371526211500168, 0.0021098581227390056], [-0.05503189191222191, -0.0746268779039383, -0.005264905366030526], [-0.057945217937231064, -0.0687008649110794, -0.00027766552838404124], [-0.057240959256887436, -0.07024197280406952, -0.0074453624812038655], [-0.060723926872015, -0.0618865080177784, -0.003036883744326424], [-0.06025322154164314, -0.06347988545894623, -0.010216620835390877], [-0.019049562513828278, -0.09744320064783096, 0.0074096647175876384], [-0.01028292253613472, -0.09854759275913239, 0.004904123869809318]])
            vertex_normals = np.array([(0.21302460134029388, -0.965381383895874, -0.1505301296710968), (0.17133180797100067, -0.9650542140007019, -0.198281928896904), (0.23409877717494965, -0.9296879172325134, -0.28439071774482727), (0.20445850491523743, -0.9217665791511536, -0.3294588029384613), (0.2950233221054077, -0.9428872466087341, -0.15467630326747894), (0.30824264883995056, -0.9361860156059265, -0.16894443333148956), (0.43910646438598633, -0.8970659375190735, -0.04958169907331467), (0.4363127052783966, -0.8983418941497803, -0.05111850053071976), (0.5882634520530701, -0.8083396553993225, 0.02308816649019718), (0.5938681960105896, -0.8041653037071228, 0.025272265076637268), (0.7094771265983582, -0.703013002872467, 0.04914076626300812), (0.711438000202179, -0.7010304927825928, 0.04911404475569725), (0.8009960651397705, -0.5974375009536743, 0.03838849067687988), (0.8086886405944824, -0.5871158242225647, 0.036299120634794235), (0.8664449453353882, -0.49890807271003723, 0.019074566662311554), (0.8745542168617249, -0.48465055227279663, 0.01639372669160366), (0.9068731665611267, -0.42126020789146423, 0.010997472330927849), (0.9094046354293823, -0.4157572090625763, 0.011365870013833046), (0.9205265045166016, -0.3904625177383423, 0.01304092351347208), (0.9205264449119568, -0.3904625177383423, 0.013040922582149506), (0.1784271001815796, -0.9839159846305847, -0.008548013865947723), (0.1276978999376297, -0.9910843968391418, -0.03801281377673149), (-0.21146777272224426, -0.9655916690826416, -0.15137451887130737), (-0.17016936838626862, -0.9649455547332764, -0.19980597496032715), (-0.23305915296077728, -0.9295825362205505, -0.2855866551399231), (-0.20367898046970367, -0.9216205477714539, -0.3303488492965698), (-0.2939514219760895, -0.9430721998214722, -0.1555873155593872), (-0.30682215094566345, -0.9363527894020081, -0.17059792578220367), (-0.43836885690689087, -0.8973409533500671, -0.05110874027013779), (-0.4348205029964447, -0.8989534974098206, -0.05304456129670143), (-0.5879247784614563, -0.8086429834365845, 0.02100095897912979), (-0.593061625957489, -0.8048291206359863, 0.022978920489549637), (-0.7096368670463562, -0.7030275464057922, 0.046558331698179245), (-0.71131432056427, -0.7013322710990906, 0.046529170125722885), (-0.8014999628067017, -0.5969457626342773, 0.035406265407800674), (-0.8090589642524719, -0.5867817997932434, 0.03332577645778656), (-0.8670861124992371, -0.49790769815444946, 0.015799948945641518), (-0.8751322627067566, -0.4837063252925873, 0.013107089325785637), (-0.9075274467468262, -0.41992461681365967, 0.007562259677797556), (-0.9100339412689209, -0.41445812582969666, 0.007917601615190506), (-0.9211719632148743, -0.3890385329723358, 0.009553579613566399), (-0.9211719036102295, -0.3890385329723358, 0.009553579613566399), (-0.1766616404056549, -0.984230101108551, -0.009047266095876694), (-0.1256365329027176, -0.9913285374641418, -0.03851176053285599)])
            uvs = [[(0.10387720167636871, 0.15240783989429474), (0.20352177321910858, 0.002554043661803007), (0.4944729804992676, 0.05704062059521675), (0.35311999917030334, 0.2287999987602234)], [(0.10387720167636871, 0.15240783989429474), (0.35311999917030334, 0.2287999987602234), (0.2858409285545349, 0.32360079884529114), (0.019021285697817802, 0.27419033646583557)], [(0.019021285697817802, 0.27419033646583557), (0.2858409285545349, 0.32360079884529114), (0.23823584616184235, 0.4334968328475952), (0.010124947875738144, 0.4299057424068451)], [(0.010124947875738144, 0.4299057424068451), (0.23823584616184235, 0.4334968328475952), (0.23564350605010986, 0.5323020219802856), (0.012801339849829674, 0.5315198302268982)], [(0.012801339849829674, 0.5315198302268982), (0.23564350605010986, 0.5323020219802856), (0.2573564350605011, 0.6207517385482788), (0.03667948767542839, 0.6383661031723022)], [(0.03667948767542839, 0.6383661031723022), (0.2573564350605011, 0.6207517385482788), (0.299108624458313, 0.7074313759803772), (0.0829995721578598, 0.7227829694747925)], [(0.0829995721578598, 0.7227829694747925), (0.299108624458313, 0.7074313759803772), (0.357651025056839, 0.7929096221923828), (0.1371798813343048, 0.8043929934501648)], [(0.1371798813343048, 0.8043929934501648), (0.357651025056839, 0.7929096221923828), (0.41832488775253296, 0.8671479225158691), (0.20339392125606537, 0.8890230059623718)], [(0.20339392125606537, 0.8890230059623718), (0.41832488775253296, 0.8671479225158691), (0.4944729804992676, 0.978790283203125), (0.27912330627441406, 1.0)], [(0.20352177321910858, 0.002554043661803007), (0.10387720167636871, 0.15240783989429474), (1.2504191460038783e-08, 0.13043661415576935), (0.08955555409193039, 2.6969557254119536e-08)], [(0.10387720167636871, 0.15240783989429474), (0.019021285697817802, 0.27419033646583557), (1.2504191460038783e-08, 0.13043661415576935)], [(0.598351240158081, 0.8475814461708069), (0.8475865721702576, 0.7711809277534485), (0.98896723985672, 0.9429332613945007), (0.6980223059654236, 0.997435450553894)], [(0.598351240158081, 0.8475814461708069), (0.5134828090667725, 0.7257977724075317), (0.7803006172180176, 0.6763840317726135), (0.8475865721702576, 0.7711809277534485)], [(0.5134828090667725, 0.7257977724075317), (0.5045840740203857, 0.5700806975364685), (0.7326943874359131, 0.5664908289909363), (0.7803006172180176, 0.6763840317726135)], [(0.5045840740203857, 0.5700806975364685), (0.5072675943374634, 0.4684655964374542), (0.7301082611083984, 0.4676879048347473), (0.7326943874359131, 0.5664908289909363)], [(0.5072675943374634, 0.4684655964374542), (0.531157374382019, 0.361620157957077), (0.7518305778503418, 0.37924033403396606), (0.7301082611083984, 0.4676879048347473)], [(0.531157374382019, 0.361620157957077), (0.5774877667427063, 0.2772054672241211), (0.7935928702354431, 0.29256266355514526), (0.7518305778503418, 0.37924033403396606)], [(0.5774877667427063, 0.2772054672241211), (0.631676197052002, 0.19559861719608307), (0.8521435260772705, 0.20708619058132172), (0.7935928702354431, 0.29256266355514526)], [(0.631676197052002, 0.19559861719608307), (0.6978949308395386, 0.11097240447998047), (0.9128215909004211, 0.13284821808338165), (0.8521435260772705, 0.20708619058132172)], [(0.6978949308395386, 0.11097240447998047), (0.7736213803291321, -1.8817928548742202e-08), (0.9889672994613647, 0.021205546334385872), (0.9128215909004211, 0.13284821808338165)], [(0.6980223059654236, 0.997435450553894), (0.5840516686439514, 0.9999967813491821), (0.4944729506969452, 0.8695566654205322), (0.598351240158081, 0.8475814461708069)], [(0.598351240158081, 0.8475814461708069), (0.4944729506969452, 0.8695566654205322), (0.5134828090667725, 0.7257977724075317)]]
            loops = np.array([(0, 0, 1), (1, 1, 2), (3, 2, 3), (2, 3, 0), (0, 4, 0), (2, 5, 5), (5, 6, 4), (4, 7, 6), (4, 8, 4), (5, 9, 8), (7, 10, 7), (6, 11, 9), (6, 12, 7), (7, 13, 11), (9, 14, 10), (8, 15, 12), (8, 16, 10), (9, 17, 14), (11, 18, 13), (10, 19, 15), (10, 20, 13), (11, 21, 17), (13, 22, 16), (12, 23, 18), (12, 24, 16), (13, 25, 20), (15, 26, 19), (14, 27, 21), (14, 28, 19), (15, 29, 23), (17, 30, 22), (16, 31, 24), (16, 32, 22), (17, 33, 26), (19, 34, 25), (18, 35, 27), (1, 36, 1), (0, 37, 29), (20, 38, 28), (21, 39, 30), (0, 40, 6), (4, 41, 31), (20, 42, 29), (22, 43, 32), (24, 44, 35), (25, 45, 34), (23, 46, 33), (22, 47, 38), (26, 48, 36), (27, 49, 37), (24, 50, 32), (26, 51, 41), (28, 52, 39), (29, 53, 40), (27, 54, 36), (28, 55, 44), (30, 56, 42), (31, 57, 43), (29, 58, 39), (30, 59, 47), (32, 60, 45), (33, 61, 46), (31, 62, 42), (32, 63, 50), (34, 64, 48), (35, 65, 49), (33, 66, 45), (34, 67, 53), (36, 68, 51), (37, 69, 52), (35, 70, 48), (36, 71, 56), (38, 72, 54), (39, 73, 55), (37, 74, 51), (38, 75, 59), (40, 76, 57), (41, 77, 58), (39, 78, 54), (23, 79, 62), (43, 80, 60), (42, 81, 61), (22, 82, 33), (22, 83, 61), (42, 84, 63), (26, 85, 38)], dtype=np.int32)
            polygons = np.array([(0, 4), (4, 4), (8, 4), (12, 4), (16, 4), (20, 4), (24, 4), (28, 4), (32, 4), (36, 4), (40, 3), (43, 4), (47, 4), (51, 4), (55, 4), (59, 4), (63, 4), (67, 4), (71, 4), (75, 4), (79, 4), (83, 3)], dtype=np.int32)
            polygon_normals = np.array([(0.204458549618721, -0.9217666387557983, -0.329458624124527), (0.2573009133338928, -0.9340332746505737, -0.2477456033229828), (0.35233110189437866, -0.9309596419334412, -0.09579653292894363), (0.5236645340919495, -0.851923406124115, -0.0013364654732868075), (0.6546372175216675, -0.7543064951896667, 0.0497170016169548), (0.7625151872634888, -0.6451690793037415, 0.04824307933449745), (0.8438367247581482, -0.5359628200531006, 0.02614249475300312), (0.894795835018158, -0.446378618478775, 0.009297684766352177), (0.9205259084701538, -0.3904636800289154, 0.013040537014603615), (0.12769724428653717, -0.9910845160484314, -0.03801056370139122), (0.25233086943626404, -0.9670045971870422, 0.03509100154042244), (-0.20367690920829773, -0.9216209053993225, -0.3303491473197937), (-0.25626254081726074, -0.9340181350708008, -0.24887651205062866), (-0.3509964346885681, -0.9313123822212219, -0.09725535660982132), (-0.5227794647216797, -0.8524614572525024, -0.003333872416988015), (-0.6543564796447754, -0.7547026872634888, 0.04734565317630768), (-0.7627802491188049, -0.6450580358505249, 0.045457348227500916), (-0.8443940281867981, -0.5352294445037842, 0.022982647642493248), (-0.8954243659973145, -0.4451744854450226, 0.0059044249355793), (-0.9211748838424683, -0.38903138041496277, 0.009553337469696999), (-0.12563464045524597, -0.9913288950920105, -0.038512177765369415), (-0.2502712309360504, -0.9675723910331726, 0.0341765321791172)])
            mesh = bpy.data.meshes.new(name='Eyebrows Mesh')
            vertices += offset_vector
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
        assert EYELASHES_M is not None
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
            mask = rle_decode(MaskStore.get_store().eyelashes_rle(), MASK_SHAPE)
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

            uv_mask = rle_decode(MaskStore().get_store().eye_socket_rle(), MASK_SHAPE)
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
        for clothing_item in ClothesStore.get_store().find_all_clothes():
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
                if not is_sub_rig(o.data, body_rig.data) and (is_hair(o) or ClothesStore.get_store().is_clothes(o) or is_cum(o)):
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
        new_uv_layer = EYES_M.data.uv_layers.new(name=NEW_EYES_UV_MAP)
        new_uv_layer.active = True
        new_uv_layer.active_render = True
        select_object(EYES_M)
        selection = np.zeros(len(EYES_M.data.uv_layers.active.data), dtype=bool)
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
            if full_loop:
                for loop in face.loops:
                    selection[loop.index] = True
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

        # if bpy.context.scene.get('daz_optim_toon'):
        #     selection = np.logical_not(selection)
        new_uv_layer_np[selection, 1] += 1
        for v, new_uv in zip(EYES_M.data.uv_layers.active.data, new_uv_layer_np):
            v.uv = new_uv

    def find_body_parts_textures(self):
        BODY_M = self.get_body_mesh()
        mats = list(BODY_M.data.materials)
        for g in DICK_GEOGRAFTS:
            if g+' Mesh' in bpy.data.objects:
                mats.extend(bpy.data.objects[g+' Mesh'].data.materials)
        return DazOptimizer.find_body_part_textures(mats)

    @staticmethod
    def find_sockets_of_each_map_type(mat):
        output_node = NodesUtils.find_by_type(mat.node_tree, bpy.types.ShaderNodeOutputMaterial)
        sockets = {'Base Color': set(), 'Roughness': set(), 'Normal': set()}
        if output_node is not None:
            for bsdf in NodesUtils.from_socket_backwards_search_for(output_node.inputs['Surface'], (bpy.types.ShaderNodeBsdfPrincipled, bpy.types.ShaderNodeGroup), set()):
                if isinstance(bsdf, bpy.types.ShaderNodeBsdfPrincipled):
                    for channel in ['Base Color', 'Roughness', 'Normal']:
                        sockets[channel].add(bsdf.inputs[channel])
                elif bsdf.node_tree.name == 'DAZ Dual Lobe PBR':
                    sockets['Roughness'].add(bsdf.inputs['Roughness 1'])
                    sockets['Roughness'].add(bsdf.inputs['Roughness 2'])
                    sockets['Normal'].add(bsdf.inputs['Normal'])
                elif bsdf.node_tree.name == 'DAZ Toon Diffuse':
                    sockets['Base Color'].add(bsdf.inputs['Color'])
                    sockets['Normal'].add(bsdf.inputs['Normal'])
        return sockets

    class FindImagesResult:
        def __init__(self):
            self.images: {bpy.types.Image} = set()
            self.const = None

    @staticmethod
    def find_images_of_each_map_type(mat):
        images = {}
        for channel, sockets in DazOptimizer.find_sockets_of_each_map_type(mat).items():
            r = images[channel] = DazOptimizer.FindImagesResult()
            for soc in sockets:
                if len(soc.links) == 0:
                    r.const_value = np.array(soc.default_value)
                r.images.update(NodesUtils.find_textures(soc))
        return images

    @staticmethod
    def find_body_part_textures(mats):
        all_filepaths: {str: {str: DazOptimizer.FindImagesResult}} = {}
        for mat in mats:
            body_part = mat.name.rstrip('0123456789-_.')
            assert body_part not in all_filepaths
            all_filepaths[body_part] = DazOptimizer.find_images_of_each_map_type(mat)

        for body_part_name, body_part_filepaths in all_filepaths.items():
            occurrences = {}
            filenames = []
            for channel_images in body_part_filepaths.values():
                assert isinstance(channel_images, DazOptimizer.FindImagesResult)
                if len(channel_images.images) == 1:
                    first = next(iter(channel_images.images))
                    filepath = first.filepath
                    filenames.append(filepath)
            lcp = os.path.commonprefix(filenames)
            for channel_name, channel_images in body_part_filepaths.items():
                for image in channel_images.images:
                    if image not in occurrences:
                        fp: str = image.filepath
                        score = len(os.path.commonprefix([fp, lcp]))
                        no_ext = fp.rsplit('.', maxsplit=1)[0]
                        has_magic_word = False
                        if channel_name == 'Base Color':
                            has_magic_word = 'Diffuse' in no_ext or no_ext.endswith(' D')
                        elif channel_name == 'Normal':
                            has_magic_word = 'Normal' in no_ext or no_ext.endswith(' N')
                        elif channel_name == 'Roughness':
                            has_magic_word = 'Rough' in no_ext or no_ext.endswith(' R') or no_ext.endswith(' S')
                        if has_magic_word:
                            score += 10
                        occurrences[image] = score
                    occurrences[image] += 1

            for channel_name, channel_images in body_part_filepaths.items():
                assert isinstance(channel_images, DazOptimizer.FindImagesResult)
                s = list(sorted(channel_images.images, key=lambda x: -occurrences[x]))
                if len(s)==0 and channel_images.const is not None:
                    s = channel_images.images
                body_part_filepaths[channel_name] = s
        print(json.dumps({k: {k2: v2.tolist() if isinstance(v2, np.ndarray) else [v3.filepath+" "+str(tuple(v3.size)+(v3.channels,)) for v3 in v2] for k2, v2 in v.items()} for k, v in all_filepaths.items()}, indent=2))
        return all_filepaths

    @staticmethod
    def gen_simple_materials(mats, all_filepaths=None):
        if all_filepaths is None:
            all_filepaths = DazOptimizer.find_body_part_textures(mats)
        for mat in mats:
            body_part = mat.name.rstrip('0123456789-_.')
            body_part_filepaths = all_filepaths[body_part]
            NodesUtils.gen_simple_material(mat, body_part_filepaths, clear_all=True)

    def collect_bakeable_mats(self):
        BODY_M = self.get_body_mesh()
        MOUTH_M = self.get_mouth_mesh()
        eyes = self.get_eyes_mesh()
        gp = self.get_gp_mesh()
        is_toon = bpy.context.scene.get('daz_optim_toon')
        mats = list(BODY_M.data.materials)
        if MOUTH_M is not None:
            mats.extend(MOUTH_M.data.materials)
        if eyes is not None:
            mats.extend(eyes.data.materials)
        if is_toon:
            mats.extend(gp.data.materials)
        for g in DICK_GEOGRAFTS:
            if g + ' Mesh' in bpy.data.objects:
                mats.extend(bpy.data.objects[g + ' Mesh'].data.materials)
        return set(mats)

    def bake_materials(self, active_only=False):
        if active_only:
            mats = [bpy.context.object.active_material]
        else:
            mats = self.collect_bakeable_mats()
        diffuse = bpy.context.scene.bake_diffuse
        norm = bpy.context.scene.bake_normal_maps
        rough = bpy.context.scene.bake_roughness_maps
        all_maps = diffuse and norm and rough
        for mat in mats:
            print("Baking material: ", mat.name)
            b = MaterialBaker(mat)
            if not all_maps:
                sockets = DazOptimizer.find_sockets_of_each_map_type(mat)
                b.whitelist = set()
                if diffuse:
                    for s in sockets['Base Color']:
                        NodesUtils.collect_all_before_socket(s, b.whitelist)
                if norm:
                    for s in sockets['Normal']:
                        NodesUtils.collect_all_before_socket(s, b.whitelist)
                if norm:
                    for s in sockets['Roughness']:
                        NodesUtils.collect_all_before_socket(s, b.whitelist)
            b.bake()
            b.apply()

    def simplify_materials(self):
        from PIL import Image
        is_toon = bpy.context.scene.get('daz_optim_toon')
        BODY_M = self.get_body_mesh()
        MOUTH_M = self.get_mouth_mesh()
        gp = self.get_gp_mesh()
        mats = self.collect_bakeable_mats()
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
                mouth_mask = rle_decode(MaskStore.get_store().toon_mouth_rle(), TOON_MOUTH_MASK_SHAPE)
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
                    mouth_cavity_rle = rle_decode(MaskStore.get_store().mouth_cavity_rle(), MASK_SHAPE)
                    mouth_cavity_color = to_channels(mouth_cavity_color, c)
                    print("Baking mouth cavity color: ", mouth_cavity_color)
                    head_img_np[mouth_cavity_rle] = mouth_cavity_color
                if eye_socket_color is not None:
                    eye_socket_rle = rle_decode(MaskStore.get_store().eye_socket_rle(), MASK_SHAPE)
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
                NodesUtils.gen_simple_material(mat, body_part_filepaths, uvs='Default UVs', clear_all=True)

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
            stor = MaskStore.get_store()
            packed = shift_img(arms_tile, 0, s, s, s2, rle_decode(stor.bot_arm_rle(), MASK_SHAPE), BOT_ARM_TRANS)
            packed = np.maximum(packed, shift_img(arms_tile, 0, s, s, s2, rle_decode(stor.top_arm_rle(), MASK_SHAPE), TOP_ARM_TRANS))
            packed = np.maximum(packed, shift_img(legs_tile, 0, s, 0, s, rle_decode(stor.left_leg_rle(), MASK_SHAPE), [0, 0]))
            packed = np.maximum(packed, shift_img(legs_tile, 0, s, 0, s, rle_decode(stor.right_leg_rle(), MASK_SHAPE), [RIGHT_LEG_TRANS, 0], True))
            packed = np.maximum(packed, shift_img(body_tile, s, s2, s, s2, rle_decode(stor.body_rle(), MASK_SHAPE), BODY_TRANS))
            packed = np.maximum(packed, shift_img(head_tile, s, s2, 0, s, rle_decode(stor.lip_rle(), MASK_SHAPE), LIP_TRANS))
            packed = np.maximum(packed, shift_img(head_tile, s, s2, 0, s, rle_decode(stor.mouth_cavity_rle(), MASK_SHAPE), MOUTH_CAVITY_SCALED_TRANS, scale=2))
            if is_floating_iris:
                right_eye_socket_mask = rle_decode(stor.eye_socket_rle(), MASK_SHAPE)
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
                assign_img(eyes_tile[:s8], s2 - s4 - s8, s2 - s4, s + s4 * 2, s + s4 * 3)
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
                        NodesUtils.gen_simple_material(m, all_filepaths)
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


    def make_fav_morphs_list(self):
        s = MorphsStore.get_store()
        s.clear()
        s.load_current()
        cats = s.CAT_SETS[bpy.context.scene.morph_profile]
        prof = s.PROFILES[bpy.context.scene.morph_level]
        s.make_fav_morphs_list(self.get_fav_morphs_path(),categories_to_include=cats,profiles_to_include=prof)


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
        is_eyes_sclera = np.logical_and(is_eyes, base_layer_np[:, 1] < 1)
        is_eyes_iris = np.logical_and(is_eyes, base_layer_np[:, 1] > 1)
        is_mouth = np.logical_and(6 < base_layer_np[:, 0], base_layer_np[:, 0] < 7)
        pixel_coords = DazOptimizer.base_layer_to_pixel_coords(base_layer_np)

        stor = MaskStore.get_store()
        uv_mask = rle_decode(stor.right_leg_rle(), MASK_SHAPE)
        is_right_leg = uv_mask[pixel_coords[:, 1], pixel_coords[:, 0]]
        is_right_leg = np.logical_and(is_legs, is_right_leg)
        # is_left_leg = np.logical_and(is_legs, np.logical_not(is_right_leg))

        uv_mask = rle_decode(stor.bot_arm_rle(), MASK_SHAPE)
        is_bot_arm = uv_mask[pixel_coords[:, 1], pixel_coords[:, 0]]
        is_bot_arm = np.logical_and(is_arms, is_bot_arm)
        is_top_arm = np.logical_and(is_arms, np.logical_not(is_bot_arm))

        uv_mask = rle_decode(stor.mouth_cavity_rle(), MASK_SHAPE)
        is_mouth_cavity = uv_mask[pixel_coords[:, 1], pixel_coords[:, 0]]
        is_mouth_cavity = np.logical_and(is_head, is_mouth_cavity)

        is_floating_iris = bpy.context.scene.get('is_floating_iris')
        if is_floating_iris:
            uv_mask = rle_decode(stor.eye_socket_rle(), MASK_SHAPE)
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
        base_layer_np[is_eyes_iris] = np.mod(iris_np, 1) / 8 + [s + s4 * 2, s4 - s8]
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
            #if not use_full_gp:
            #    base_layer_np[is_outer_gp] += np.array([0.5,0]) + BODY_TRANS
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
        base_layer_np = self.get_base_uv_layer_np()
        bpy.ops.object.mode_set(mode='EDIT')

        bpy.context.scene.tool_settings.use_uv_select_sync = False
        bpy.ops.uv.select_all(action='DESELECT')
        bpy.ops.mesh.select_all(action='DESELECT')

        me = bpy.context.object.data
        bm = bmesh.from_edit_mesh(me)
        uv_layer = bm.loops.layers.uv.verify()
        selection = np.zeros(len(base_layer_np), dtype=bool)
        # for v in bm.verts:
        #    v.select_set(False)
        uv_mask = rle_decode(MaskStore.get_store().lip_rle(), MASK_SHAPE)

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
            if full_loop:
                for loop in face.loops:
                    selection[loop.index] = True
            face.select_set(full_loop)

        # bm.select_mode = {'VERT', 'EDGE', 'FACE'}
        bm.select_flush_mode()
        # bpy.context.tool_settings.mesh_select_mode = (False, False, True)
        bpy.ops.uv.select_all(action='SELECT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.select_split()

        bpy.ops.object.mode_set(mode='OBJECT')
        #  def separate_lips(self):

        # pixel_class = get_pixel_class()
        base_layer_np[selection] = base_layer_np[selection] + LIP_TRANS
        self.update_base_uv_layer(base_layer_np)
        # += [0.043945, 0.006836] # top arm
        # += [-0.072266 , 0.085937] # obttom arm
        # += [0.008526, 0.019377] # torso
        # *= 0.25# nails
        # -= 0.5 # nails

    def fit_panties(self):
        clothes = ClothesStore.get_store().find_all_panties()
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
        clothes = ClothesStore.get_store().find_all_non_skin_tight_clothes()
        for c in clothes:
            hide_object(c.obj, False)
            c.obj.select_set(True)
        bpy.ops.daz.transfer_shapekeys('INVOKE_DEFAULT', bodypart='NoFace', filter=EXTRUDED_SK_NAME, useOverwrite=False)

    def bind_clothes_to_extrude(self):
        body = self.get_body_mesh()
        select_object(body)
        remove_shape_key(body, EXTRUDED_SK_NAME)
        clothes = ClothesStore.get_store().find_all_non_skin_tight_clothes()
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
        for c in ClothesStore.get_store().find_all_non_skin_tight_clothes():
            select_object(c.obj)
            bpy.ops.object.shape_key_remove(all=True, apply_mix=True)

    def fit_skin_tight_clothes(self):
        BODY_M = self.get_body_mesh()
        m_name = 'FitSkinTightClothes'
        for meta in ClothesStore.get_store().find_all_skin_tight_clothes():
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
            print("Creating new Golden Palace UVs")
            gp_labia_majora = mesh.data.uv_layers.get('Golden Palace 2')
            gp_labia_minora = mesh.data.uv_layers['Golden Palace']
            new_uv_layer = mesh.data.uv_layers.new(name=NEW_GP_UV_MAP)
            print("Reading old Golden Palace labia UVs")
            gp_labia_minora_np = np.array([v.uv for v in gp_labia_minora.data])
            if gp_labia_majora is None:
                new_uv_layer_np = gp_labia_minora_np
            else:
                gp_labia_majora.active = True
                print("Reading new_uv_layer.data")
                new_uv_layer_np = np.array([v.uv for v in new_uv_layer.data])
                print("Reading gp_labia_majora.data")
                gp_labia_majora_np = np.array([v.uv for v in gp_labia_majora.data])
                print("Calculating new UVs")
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
            print("Applying new Golden Palace UVs")
            for v, new_uv in zip(new_uv_layer.data, new_uv_layer_np):
                v.uv = new_uv

    def simplify_golden_palace_material(self):
        mesh = self.select_gp_or_body()
        filepaths = {}
        for channel in ['Base Color', 'Roughness', 'Normal']:
            if not bpy.context.scene.get('gp_lacks_' + channel, False):
                name = 'GP_Baked_' + channel
                if name in bpy.data.images:
                    filepaths[channel] = bpy.data.images[name]
                else:
                    p = os.path.join(self.workdir, self.name + "_" + channel + '_gp_baked.png')
                    if os.path.exists(p):
                        filepaths[channel] = bpy.data.images.load(p)
        for mat in mesh.data.materials:
            if mat.name.startswith("GP_"):
                NodesUtils.gen_simple_material(mat, filepaths, uvs=NEW_GP_UV_MAP, clear_all=True, keep_shells=False)


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
        NodesUtils.gen_simple_material(mat, filepaths)
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

    @staticmethod
    def apply_additional_bone(bone_names=None):
        AdditionalBones(bone_names).apply()

    @staticmethod
    def save_custom_rig():
        CustomRig.save_custom_rig()

    @staticmethod
    def apply_custom_rig():
        CustomRig.apply_custom_rig()

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
        uv_mask = rle_decode(MaskStore.get_store().butt_rle(), MASK_SHAPE)
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
        clothes = ClothesStore.get_store().find_all_clothes()
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
        applied_additional_bones = AdditionalBones.get_applied_bones()
        groups = set(b for b in applied_additional_bones if b in BODY_M.vertex_groups)
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
        clothes = [c.obj for c in ClothesStore.get_store().find_all_clothes()]
        transfer_weights(BODY_M, clothes, groups)

    def transfer_missing_bones_to_cum(self):
        BODY_M = self.get_body_mesh()
        groups = self.get_missing_bones()
        cum = find_cum()
        transfer_weights(BODY_M, cum, groups)


    def compare_daz_to_ue5_skeleton(self):
        body_rig = self.get_body_rig()
        select_object(body_rig)
        bpy.ops.object.mode_set(mode='OBJECT')
        print(str(body_rig.data))
        hierarchy = BoneHierarchy.get_hierarchy()
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
        hierarchy = BoneHierarchy.get_hierarchy()
        height = body_mesh.dimensions[2]
        ue5_height = QUINN_HEIGHT if is_female() else MANNY_HEIGHT
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
                    ue5_bone = hierarchy[bone_name].start
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
        hierarchy = BoneHierarchy.get_hierarchy()

        def recursion(bone, parent_rotation):
            bone_name = bone.name
            if bone_name in DAZ_TO_UE5_POSE_ROTATIONS:
                ue5_bone = hierarchy[bone_name]
                ue5_y_axis = mathutils.Vector(ue5_bone.y_axis)
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
        hierarchy = BoneHierarchy.get_hierarchy()
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
            ue5_bone = hierarchy[spine_bone]
            b = bones[spine_bone]
            b.new_y_axis = mathutils.Vector(ue5_bone.y_axis)
            b.new_z_axis = mathutils.Vector(ue5_bone.z_axis)
        for bone in body_rig.data.edit_bones:
            bone_name = bone.name
            if bone_name in hierarchy:
                b = bones[bone_name]
                new_z_axis = b.new_z_axis
                new_y_axis = b.new_y_axis
                ue5_bone = hierarchy[bone_name]
                assert isinstance(ue5_bone, BoneRelation)
                z_axis = mathutils.Vector(ue5_bone.z_axis)
                y_axis = mathutils.Vector(ue5_bone.y_axis)
                ue5_orientation = mathutils.Vector(ue5_bone.tail)-mathutils.Vector(ue5_bone.start)
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
        hierarchy = BoneHierarchy.get_hierarchy()
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
            ue5_bone = hierarchy[ik_bone_name]
            assert isinstance(ue5_bone, BoneRelation)
            parent_ik_bone_name = ue5_bone.parent_name
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

    def scale_to_quinn(self, relative_to_base_daz=True):
        if relative_to_base_daz:
            height = 1.7000360488891602
        else:
            mesh = self.get_body_mesh()
            height = mesh.dimensions[2]
        ue5_height = QUINN_HEIGHT if is_female() else MANNY_HEIGHT
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
            hierarchy = BoneHierarchy.get_hierarchy()
            ue5_pevis_pos = hierarchy['pelvis'].start
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
            remove_unnecessary_bones()

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
        applied_additional_bones = AdditionalBones.get_applied_bones()
        bones = [bone.name for bone in rig.data.bones if not is_known_bone(bone.name, applied_additional_bones)]
        AdditionalBone.serialize_bone_and_weights(body, bones).save()


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
        for clothes in ClothesStore.get_store().find_all_clothes():
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
        hide_object(obj, False)
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
        s = MorphsStore.get_store()
        for mesh_name, morphs in s.morphs.items():
            assert isinstance(morphs, dict)
            for shape, meta in morphs.items():
                assert isinstance(meta, Morph)
                shape_key_categories[shape] = meta.is_female, meta.is_male, meta
        body = self.get_body_mesh()
        print("---,Name,MorphName,BodyPart,IsForFemales,IsForMales,Min,Max,Default")
        for b in body.data.shape_keys.key_blocks:
            n = b.name
            if n in shape_key_categories:
                assert isinstance(n, str)
                is_female, is_male, meta = shape_key_categories[n]
                if meta.category not in s.CAT_SETS['FACS']:
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



