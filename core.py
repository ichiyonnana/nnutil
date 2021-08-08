# coding:utf-8

"""
単独で機能になっていないもの
基本的にはほかのモジュールが呼び出すもの
"""

import maya.cmds as cmds

from itertools import *

import math
import re
import copy

def distance(p1, p2):
    """
    p1,p2の三次元距離を返す
    """

    return math.sqrt(distance_sq(p1, p2))

def distance_sq(p1, p2):
    """
    p1,p2の三次元距離の二乗を返す
    """

    if not isinstance(p1, list):
        p1 = point_from_vertex(p1)

    if not isinstance(p2, list):
        p2 = point_from_vertex(p2)

    return (p2[0]-p1[0])**2 + (p2[1]-p1[1])**2 + (p2[2]-p1[2])**2

def dot(v1, v2):
    """三次元ベクトルの内積"""
    return sum([x1 * x2 for x1,x2 in zip(v1,v2)])

def cross(a, b):
    """三次元ベクトルの外積"""
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])

def add(v1, v2):
    return [x1 + x2 for x1,x2 in zip(v1,v2)]

def diff(v2, v1):
    return [x2 - x1 for x1,x2 in zip(v1,v2)]

def mul(v1, f):
    return [x * f for x in v1]

def div(v1, f):
    return [x / f for x in v1]

def angle(v1, v2, degrees=False):
    """2ベクトル間のなす角"""
    rad = math.acos(nu.dot(v1, v2))

    if degrees:
        return math.degrees(rad)
    else:
        return rad

def edge_to_vector(edge, normalize=False):
    p1, p2 = [get_vtx_coord(x) for x in to_vtx(edge)]
    v = diff(p2, p1)

    if normailze:
        return normalize(v)
    else:
        return v


def get_farthest_point_pair(points):
    """
    points に含まれる点のうち一番遠い二点を返す
    """

    farthest_point_pair = []
    max_distance = -1

    point_pairs = combinations(points, 2)

    for a,b in point_pairs:
        d = distance_sq(a, b)

        if d > max_distance:
            most_distant_vtx_pair = vtx_pair
            max_distance = d

    return most_distant_vtx_pair


    return get_most_distant_vts(points)


def get_nearest_point_from_point(point, target_points):
    """
    target_points のうち point に一番近い点の座標を返す
    """

    ret_point = []
    min_distance = -1

    for target_point in target_points:
        d = distance(point, target_point)
        if d < min_distance or min_distance < 0:
            min_distance = d
            ret_point = target_point

    return ret_point

def nearest_point_on_line(p1, p2, p3):
    """
    p1 p2 を通る直線を構成する頂点うち p3 に最も近いものの座標を返す (垂線を下ろした交点)
    """

    v = normalize(vector(p1, p2))
    p = vector(p1, p3)

    return add(p1, mul(v, dot(p, v)))

def vector(p1, p2):
    """
    p1->p2ベクトルを返す
    """
    return (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])

def normalize(v):
    """
    ベクトルを渡して正規化したベクトルを返す
    """
    norm = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    if norm != 0:
        return (v[0]/norm, v[1]/norm, v[2]/norm)
    else:
        return (0,0,0)

def point_from_vertex(vtx):
    """
    maya.cmds のコンポーネント文字列から座標を取得
    """
    return cmds.xform(vtx, q=True, ws=True, t=True)

def get_vtx_coord(vtx):
    """
    maya.cmds のコンポーネント文字列から座標を取得
    """
    return cmds.xform(vtx, q=True, ws=True, t=True)

def set_vtx_coord(vtx, point):
        cmds.xform(vtx, ws=True, t=point)

def get_uv_coord(uv):
    uv_coord = cmds.polyEditUV(uv, query=True)
    return uv_coord

def get_end_vtx_e(edges):
    """
    edges に含まれる端の頂点を返す
    """
    endvtx = []

    vts = cmds.filterExpand(cmds.polyListComponentConversion(
        edges, fe=True, tv=True), sm=31)

    for vtx in vts:
        neighbors = set(cmds.filterExpand(cmds.polyListComponentConversion(
            vtx, fv=True, te=True), sm=32)) & set(edges)
        if len(neighbors) == 1:
            endvtx.append(vtx)

    return endvtx

def get_end_vtx_v(vts):
    """
    連続したエッジで接続される頂点の端の頂点を返す
    引数は頂点列
    """
    conv_edges = to_edge(vts)
    edges = [e for e in conv_edges if len(set(to_vtx(e)) & set(vts)) == 2]

    return get_end_vtx_e(edges)

def get_most_distant_vts(vts):
    """
    引数で渡した頂点集合のうち最も離れた2点を返す
    """
    most_distant_vtx_pair = []
    max_distance = -1

    vtx_point_dic = {vtx:point_from_vertex(vtx) for vtx in vts}
    vtx_pairs = combinations(vts, 2)

    for vtx_pair in vtx_pairs:
        d = abs(distance_sq(vtx_point_dic[vtx_pair[0]] , vtx_point_dic[vtx_pair[1]]))

        if d > max_distance:
            most_distant_vtx_pair = vtx_pair
            max_distance = d

    return most_distant_vtx_pair

def sortVtx(edges, vts):
    """
    指定した点から順にエッジたどって末尾まで到達する頂点の列を返す
    """
    def partVtxList(partEdges, startVtx):
        """
        部分エッジ集合と開始頂点から再帰的に頂点列を求める
        """
        neighbors = cmds.filterExpand(
            cmds.polyListComponentConversion(startVtx, fv=True, te=True), sm=32)
        nextEdges = list(set(neighbors) & set(partEdges))

        if len(nextEdges) == 1:
            nextEdge = nextEdges[0]
            vset1 = set(cmds.filterExpand(
                cmds.polyListComponentConversion(nextEdge, fe=True, tv=True), sm=31))
            vset2 = {startVtx}
            nextVtx = list(vset1-vset2)[0]
            restEdges = list(set(partEdges) - set(nextEdges))
            partial_vts = partVtxList(restEdges, nextVtx)
            partial_vts.insert(0, startVtx)
            return partial_vts
        else:
            return [startVtx]

    first_vtx = get_end_vtx_e(edges)[0]
    return partVtxList(edges, first_vtx)

def isStart(vtx, curve):
    """
    vts[0] とカーブの 0.0, 1.0 の距離比較して 0.0 の方が近ければ true 返す
    """
    curve_start = cmds.pointOnCurve(curve, pr=0, p=True)
    curve_end = cmds.pointOnCurve(curve, pr=1, p=True)
    pnt = get_vtx_coord(vtx)
    d1 = distance(pnt, curve_start)
    d2 = distance(pnt, curve_end)
    if d1 < d2:
        return True
    else:
        return False

def vtxListPath(vts, n=None):
    """
    vts で渡された頂点の i 番目までの距離を返す
    i を省略した場合は道のり全長を返す
    """
    if n == None or n > len(vts):
        n = len(vts)-1

    path = 0.0

    for i in range(n):
        pnt1 = get_vtx_coord(vts[i])
        pnt2 = get_vtx_coord(vts[i+1])
        path += distance(pnt1, pnt2)

    return path


def get_object(target):
    """
    targetからオブジェクトを取得する
    コンポーネントの場合
        そのコンポーネントが所属するオブジェクトを返す
    コンポーネントリストの場合
        リストの先頭のコンポーネントの所属するオブジェクトを返す
    オブジェクトの場合
        オブジェクトをそのまま返す
    オブジェクトリストの場合
        リストの先頭のオブジェクトをそのまま返す
    """

    # コンポーネント文字列や配列でない場合は終了
    if not isinstance(target, str) and not isinstance(target, list) and not isinstance(target, unicode):
        return None

    # リストなら先頭要素を使用する
    target_str = None
    if isinstance(target, list):
        target_str = target[0]
    else:
        target_str = target

    if "." in target_str:
        return re.sub(r"\..*", "", target_str)
    else:
        return target_str

def to_vtx(components):
    ret = cmds.filterExpand(cmds.polyListComponentConversion(
    components, tv=True), sm=31)
    return ret

def to_edge(components):
    ret = cmds.filterExpand(cmds.polyListComponentConversion(
    components, te=True), sm=32)
    return ret

def to_face(components):
    ret = cmds.filterExpand(cmds.polyListComponentConversion(
    components, tf=True), sm=34)
    return ret

def to_uv(components):
    ret = cmds.filterExpand(cmds.polyListComponentConversion(
    components, tuv=True), sm=35)
    return ret

def to_vtxface(components):
    ret = cmds.filterExpand(cmds.polyListComponentConversion(
    components, tvf=True), sm=70)
    return ret

def type_of_component(comp):
    """
    component の種類を返す
    "e","f","v",None
    """
    if ".vtx[" in comp:
        return "vtx"
    elif ".e[" in comp:
        return "edge"
    elif ".f[" in comp:
        return "face"
    elif ".map[" in comp:
        return "uv"
    elif ".vtxFace[" in comp:
        return "vf"
    else:
        return None


def flatten(a):
    from itertools import chain
    return list(chain.from_iterable(a))

def uniq(a):
    elemtype = type(a[0])
    string_types = [type(""), type(u"")] # Supports python2.x and python3.x
    if elemtype in string_types:
        elements_tuple_list = list(set([tuple(x) for x in a]))
        return ["".join(elements_tuple) for elements_tuple in elements_tuple_list]
    else:
        return list(set(a))


def round_vector(v, fraction):
    v = [round(x, fraction) for x in v]
    return v

def get_poly_line(edges):
    """
    edges を連続するエッジのまとまりとしてエッジリストを一つ返す
    """

    polyline = []
    polyline_list = []
    processed_edges = []
    processed_vts = []
    rest_edges = copy.copy(edges)
    vtx_queue = []

    vtx_queue = [to_vtx(rest_edges[0])[0]]

    while len(vtx_queue) > 0:
        # edges[0]のvts[0] 取得

        for vtx in vtx_queue:

            # 処理済み頂点にvtx 追加
            processed_vts.append(vtx)
            vtx_queue = list(set(vtx_queue) - set(processed_vts))

            # vtx をエッジ変換して edges に含まれるものだけ取得する
            adjacent_edges = list(set(to_edge(vtx)) & set(rest_edges))

            if len(adjacent_edges) > 0:
                # 隣接エッジがあれば連続エッジに追加
                polyline.extend(adjacent_edges)
                polyline = uniq(polyline)

                # 処理済みエッジに追加
                processed_edges.extend(adjacent_edges)
                rest_edges = list(set(rest_edges) - set(adjacent_edges))

                # 隣接エッジの構成頂点のうち未処理のものをキューに追加する
                vtx_queue.extend(list(set(to_vtx(adjacent_edges)) - set(processed_vts)))
            else:
                # 隣接エッジなし (一つの連続エッジの完了)
                pass

    return polyline

def get_all_polylines(edges):
    """
    edges で指定したエッジ列を連続するエッジ列の集まりに分割してリストを返す
    """
    polylines = []
    rest_edges = copy.copy(edges)

    while len(rest_edges) > 0:
        polyline = get_poly_line(rest_edges)
        polylines.append(polyline)
        rest_edges = list(set(rest_edges) - set(polyline))

    return polylines

def name_to_uuid(name):
    uuid_list = cmds.ls(name, uuid=True)
    if len(uuid_list) == 1:
        return uuid_list[0]
    else:
        raise("name: " + name + "is not unique. try fullpath")

def uuid_to_name(uuid):
    return cmds.ls(uuid)[0]

def get_fullpath(name):
    """
    オブジェクト名のフルパス
    """
    return ls(name, l=True)

def get_basename(name):
    fullpath = get_fullpath(name)
    return re(r"^.*\|", "", name)

def get_active_camera():
    active_panel = cmds.getPanel(wf=True)
    camera = cmds.modelPanel(active_panel, q=True, camera=True)

    return camera