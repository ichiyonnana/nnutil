# coding:utf-8

"""
単独で機能になっていないもの
基本的にはほかのモジュールが呼び出すもの
戻り値のある物
"""
import datetime
import math
import re
import copy

import itertools as it
import functools


import maya.cmds as cmds
import maya.mel as mel
import pymel.core as pm
import pymel.core.nodetypes as nt

DEBUG = False


def undo_chunk(function):
    """ Undo チャンク用デコレーター """
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        cmds.undoInfo(ock=True)
        ret = function(*args, **kwargs)
        cmds.undoInfo(cck=True)
        return ret

    return wrapper


if DEBUG:
    def timer(function):
        """時間計測デコレーター"""
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            start = datetime.datetime.today()
            ret = function(*args, **kwargs)
            end = datetime.datetime.today()
            delta = (end - start)
            sec = delta.seconds + delta.microseconds/1000000.0
            print('time(sec): ' + str(sec) + " " + str(function))
            return ret

        return wrapper

else:
    def timer(function):
        """デバッグ無効時の時間計測デコレーター"""
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            return function(*args, **kwargs)

        return wrapper


def no_warning(function):
    """警告抑止デコレーター"""
    def wrapper(*args, **kwargs):
        warning_flag = cmds.scriptEditorInfo(q=True, suppressWarnings=True)
        info_flag = cmds.scriptEditorInfo(q=True, suppressInfo=True)
        cmds.scriptEditorInfo(e=True, suppressWarnings=True, suppressInfo=True)
        ret = function(*args, **kwargs)
        cmds.scriptEditorInfo(e=True, suppressWarnings=warning_flag, suppressInfo=info_flag)
        return ret

    return wrapper


def idstr(pynode):
    """ [pm] PyNode を cmds 用の文字列に変換する """
    if type(pynode) == list:
        return [idstr(x) for x in pynode]
    else:
        return str(pynode)


def pynode(object_name):
    """ [pm/cmds] cmds 用の文字列 を PyNode に変換する """
    if type(object_name) == list:
        return [pynode(x) for x in object_name]
    else:
        return pm.PyNode(object_name)


def list_add(l1, l2):
    """ リスト同士の和 (l1 + l2 ) を返す

    重複削除はせずただ結合しただけのリストが返る
    TODO: *args で可変長 (l1 + l2 + l3 + ...) に対応して
    
    """
    return l1 + l2


def list_diff(l1, l2):
    """ リスト同士の差集合 (l1 - l2) を返す
    
    list(set() - set()) よりは軽い
    TODO: *args で可変長 (l1 - l2 - l3 - ... )に対応して

    """
    return filter(lambda x: x not in l2, l1)


def list_intersection(l1, l2):
    """ リストの積集合 (l1 & l2) を返す
    
    list(set() & set()) よりは軽い
    TODO: *args で可変長 (l1 & l2 & l3 & ...) に対応して
    
    """
    return filter(lambda x: x in l2, l1)


def distance(p1, p2):
    """ [cmds] p1,p2の三次元空間での距離を返す

    Args:
        p1 (list[float, float, float] or str): 三次元座標のリスト or cmds のコンポーネント文字列
        p2 (list[float, float, float] or str): 三次元座標のリスト or cmds のコンポーネント文字列

    Returns:
        list[float, float, float]:
    """

    return math.sqrt(distance_sq(p1, p2))


def distance_sq(p1, p2):
    """ [cmds] p1,p2の三次元距離の二乗を返す

    Args:
        p1 (list[float, float, float] or str): 三次元座標のリスト or cmds のコンポーネント文字列
        p2 (list[float, float, float] or str): 三次元座標のリスト or cmds のコンポーネント文字列

    Returns:
        list[float, float, float]:
    """
    if not isinstance(p1, list):
        p1 = point_from_vertex(p1)

    if not isinstance(p2, list):
        p2 = point_from_vertex(p2)

    return (p2[0]-p1[0])**2 + (p2[1]-p1[1])**2 + (p2[2]-p1[2])**2


def dot(v1, v2):
    """ 三次元ベクトルの内積
    
    Args:
        v1 (list[float, float]): 
        v2 (list[float, float]): 
        
    Returns:
        float:
    """
    return sum([x1 * x2 for x1, x2 in zip(v1, v2)])


def cross(a, b):
    """ 三次元ベクトルの外積

    Args:
        a (list[float, float, float]):
        b (list[float, float, float]):
        
    Returns:
        list[float, float, float]:
    """
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def add(v1, v2):
    """ 三次元ベクトルの加算
    
    Args:
        v1 (list[float, float, float]):
        v2 (list[float, float, float]):
        
    Returns:
        list[float, float, float]:
    """
    return [x1 + x2 for x1, x2 in zip(v1, v2)]


def diff(v2, v1):
    """ 三次元ベクトルの減算
    
    Args:
        v1 (list[float, float, float]):
        v2 (list[float, float, float]):
        
    Returns:
        list[float, float, float]:
    """
    return [x2 - x1 for x1, x2 in zip(v1, v2)]


def mul(v1, f):
    """ 三次元ベクトルと実数の積
    
    Args:
        v1 (list[float, float, float]):
        f (float):
        
    Returns:
        list[float, float, float]:
    """
    return [x * f for x in v1]


def div(v1, f):
    """ 三次元ベクトルと実数の商
    
    Args:
        v1 (list[float, float, float]):
        f (float):
        
    Returns:
        list[float, float, float]:
    """
    return [x / f for x in v1]


def angle(v1, v2, degrees=False):
    """ 2ベクトル間のなす角
    
    Args:
        v1 (list[float, float, float]):
        v2 (list[float, float, float]):
        
    Returns:
        float:
    """
    rad = math.acos(dot(v1, v2))

    if degrees:
        return math.degrees(rad)
    else:
        return rad


def edge_to_vector(edge, normalize=False):
    """ [cmds] エッジからベクトルを取得する｡ベクトルの前後は不定

    Args:
        edge (str): エッジを表すcmdsコンポーネント文字列
        normalize (bool, optional): Trueの場合正規化を行う. Defaults to False.

    Returns:
        list[float, float, float]: エッジから作成されたベクトル
    """
    p1, p2 = [get_vtx_coord(x) for x in to_vtx(edge)]
    v = diff(p2, p1)

    if normalize:
        return normalize(v)
    else:
        return v


def get_farthest_point_pair(points):
    """ points に含まれる点のうち一番遠い二点を返す
    
    Args:
        points (list[list[float, float, float]]):

    Returns:
        list[list[float, float, float]]
    """
    point_pairs = it.combinations(points, 2)
    max_distance = -1

    for a, b in point_pairs:
        d = distance_sq(a, b)

        if d > max_distance:
            most_distant_vtx_pair = [a, b]
            max_distance = d

    return most_distant_vtx_pair


def get_nearest_point_from_point(point, target_points):
    """ target_points のうち point に一番近い点の座標を返す
    
    Args:
        point (list[float, float, float]):
        target_points (list[list[float, float, float]]):

    Returns:
        list[float, float, float]
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
    """ p1 p2 を通る直線を構成する頂点うち p3 に最も近いものの座標を返す (垂線を下ろした交点)

    Args:
        p1 (list[float, float, float]): 直線の通過する点1
        p2 (list[float, float, float]): 直線の通過する点2
        p2 (list[float, float, float]): 垂線を下ろす点

    Returns:
        list[float, float, float]
    """

    v = normalize(vector(p1, p2))
    p = vector(p1, p3)

    return add(p1, mul(v, dot(p, v)))


def vector(p1, p2):
    """ p1->p2ベクトルを返す

    Args:
        p1 (list[float, float, float]): ベクトルの起点座標
        p2 (list[float, float, float]): ベクトルの終点座標
    """
    return (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])


def normalize(v):
    """ ベクトルを渡して正規化したベクトルを返す

    Args:
        v (list[float, float, float]): 正規化するベクトル
    """
    norm = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    if norm != 0:
        return (v[0]/norm, v[1]/norm, v[2]/norm)
    else:
        return (0, 0, 0)


def point_from_vertex(vtx):
    """ [cmds] maya.cmds のコンポーネント文字列から座標を取得

    Args:
        vtx (str): 頂点を表すコンポーネント文字列

    Returns:
        list[float, float, float]: vtx のワールド空間座標
    """
    return cmds.xform(vtx, q=True, ws=True, t=True)


def get_vtx_coord(vtx):
    """ [cmds] maya.cmds のコンポーネント文字列から座標を取得

    Args:
        vtx (str): 頂点を表すコンポーネント文字列

    Returns:
        list[float, float, float]: vtx のワールド空間座標
    """
    return cmds.xform(vtx, q=True, ws=True, t=True)


def set_vtx_coord(vtx, point):
    """ [cmds] maya.cmds のコンポーネント文字列から座標を設定

    Args:
        vtx (str): 頂点を表すコンポーネント文字列
        point(list[float, float, float]): 設定するワールド座標

    Returns:
        None
    """
    cmds.xform(vtx, ws=True, t=point)


def get_uv_coord(uv):
    """ [cmds] maya.cmds のコンポーネント文字列からUV座標を取得
    
    Args:
        uv (str):
    
    Return:
        list[float, float]: 
    
    """
    uv_coord = cmds.polyEditUV(uv, query=True)
    return uv_coord


def get_connected_vertices(comp):
    if type(comp) in [type(""), type(u"")]:
        # comp 自体を取り除く (obj, objShape 対策でインデックスのみ比較)
        return [x for x in to_vtx(to_edge(comp)) if re.search(r"(\[\d+\])", comp).groups()[0] not in x]
    
    elif type(comp) in [pm.MeshEdge, pm.MeshVertex]:
        return pynode(get_connected_vertices(idstr(comp)))

    else:
        raise
    

def get_end_vtx_e(edges):
    """ [cmds] edges に含まれる端の頂点をすべて返す

    Args:
        edges(list[str]): エッジを表すcmdsコンポーネント文字列のリスト

    Returns:
        list[str]: 頂点を表すcmdsコンポーネント文字列のリスト
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
    """ [cmds] 連続したエッジで接続される頂点の端の頂点をすべて返す

    Args:
        vts (list[str]): 頂点を表すcmdsコンポーネント文字列のリスト
    
    Returns:
        list[str]: 頂点を表すcmdsコンポーネント文字列のリスト
    """
    conv_edges = to_edge(vts)
    edges = [e for e in conv_edges if len(set(to_vtx(e)) & set(vts)) == 2]

    return get_end_vtx_e(edges)


def get_most_distant_vts(vts):
    """ [cmds] 引数で渡した頂点集合のうち最も離れた2点を返す
    
    Args:
        vts (list[str]): 頂点を表すcmdsコンポーネント文字列のリスト

    Returns:
        (list[str, str]):
    """
    most_distant_vtx_pair = []
    max_distance = -1

    vtx_point_dic = {vtx: point_from_vertex(vtx) for vtx in vts}
    vtx_pairs = it.combinations(vts, 2)

    for vtx_pair in vtx_pairs:
        d = abs(distance_sq(vtx_point_dic[vtx_pair[0]], vtx_point_dic[vtx_pair[1]]))

        if d > max_distance:
            most_distant_vtx_pair = vtx_pair
            max_distance = d

    return most_distant_vtx_pair


def sortVtx(edges, vts):
    """ [cmds] 指定した点から順にエッジたどって末尾まで到達する頂点の列を返す

    Args:
        edges(list[str]): エッジを表すcmdsコンポーネント文字列のリスト
        vts (list[str]): 頂点を表すcmdsコンポーネント文字列のリスト
    
    Returns:
        list[str]: 頂点を表すcmdsコンポーネント文字列のリスト
    
    """
    def partVtxList(partEdges, startVtx):
        """ 部分エッジ集合と開始頂点から再帰的に頂点列を求める """
        neighbors = cmds.filterExpand(
            cmds.polyListComponentConversion(startVtx, fv=True, te=True), sm=32)
        nextEdges = list_intersection(neighbors, partEdges)

        if len(nextEdges) == 1:
            nextEdge = nextEdges[0]
            vset1 = set(cmds.filterExpand(cmds.polyListComponentConversion(nextEdge, fe=True, tv=True), sm=31))
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
    """ [cmds] vts[0] とカーブの始点･終点の距離を比較して始点の方が近ければ True 返す

    Args:

    Returns:

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
    """ [cmds] vts で渡された頂点の i 番目までの距離を返す｡ n を省略した場合は道のり全長を返す

    Args:

    Returns:

    """
    if n is None or n > len(vts):
        n = len(vts)-1

    path = 0.0

    for i in range(n):
        pnt1 = get_vtx_coord(vts[i])
        pnt2 = get_vtx_coord(vts[i+1])
        path += distance(pnt1, pnt2)

    return path


def length_each_vertices(vertices, space="world"):
    """ [pm] 頂点間の距離をリストで返す
    
    戻り値リストの n 番目は vertices[n] と vertices[n+1] の距離

    Args:
        vertices (list[MeshVertex]):
        space (str):

    Returns:
        list[float]: 
    
    """
    length_list = []

    for i in range(len(vertices)-1):        
        pnt1 = vertices[i].getPosition(space=space)
        pnt2 = vertices[i+1].getPosition(space=space)
        length_list.append((pnt1 - pnt2).length())

    return length_list


def vertices_path_length(vertices, n=None, space="world"):
    """ [pm] vertices で渡された頂点の i 番目までの距離を返す｡ n を省略した場合は道のり全長を返す

    すべての頂点間の距離を調べるときは length_each_vertices() 推奨｡ピンポイント 1 点とかならこっちでも｡

    Args:
        vertices (list[MeshVertex]):
        space (str):

    Returns:
        float:

    """
    if n is None or n > len(vertices):
        n = len(vertices)-1

    path = 0.0

    for i in range(n):        
        pnt1 = vertices[i].getPosition(space=space)
        pnt2 = vertices[i+1].getPosition(space=space)
        path += (pnt1 - pnt2).length()

    return path


def get_object(component, pn=False):
    """ [pm/cmds] component の所属するオブジェクトを取得する

    TODO: 引数の型で pm/cmds 判断して pn オプション廃止する (指定し忘れのクラッシュ事故が多い)

    Args:

    Returns:
        str or PyNode:
    """
    if pn:
        return pynode(pm.polyListComponentConversion(component)[0])
    else:
        return cmds.polyListComponentConversion(component)[0]


def to_vtx(components, pn=False):
    """ [pm/cmds] 
    
    TODO: 引数の型で pm/cmds 判断して pn オプション廃止する (指定し忘れのクラッシュ事故が多い)

    Args:
        components ([type]): [description]
        pn (bool, optional): [description]. Defaults to False.

    Returns:
        str or PyNode:
    """
    if pn:
        return uniq(pynode(pm.filterExpand(pm.polyListComponentConversion(components, tv=True), sm=31)))
    else:
        return uniq(cmds.filterExpand(cmds.polyListComponentConversion(components, tv=True), sm=31))


def to_edge(components, pn=False):
    """ [pm/cmds] 
    
    TODO: 引数の型で pm/cmds 判断して pn オプション廃止する (指定し忘れのクラッシュ事故が多い)

    Args:
        components ([type]): [description]
        pn (bool, optional): [description]. Defaults to False.

    Returns:
        str or PyNode:
    """
    if pn:
        return uniq(pynode(pm.filterExpand(pm.polyListComponentConversion(components, te=True), sm=32)))
    else:
        return uniq(cmds.filterExpand(cmds.polyListComponentConversion(components, te=True), sm=32))


def to_face(components, pn=False):
    """ [pm/cmds] 
    
    TODO: 引数の型で pm/cmds 判断して pn オプション廃止する (指定し忘れのクラッシュ事故が多い)

    Args:
        components ([type]): [description]
        pn (bool, optional): [description]. Defaults to False.

    Returns:
        str or PyNode:
    """
    if pn:
        return uniq(pynode(pm.filterExpand(pm.polyListComponentConversion(components, tf=True), sm=34)))
    else:
        return uniq(cmds.filterExpand(cmds.polyListComponentConversion(components, tf=True), sm=34))


def to_uv(components, pn=False):  
    """ [pm/cmds] 
    
    TODO: 引数の型で pm/cmds 判断して pn オプション廃止する (指定し忘れのクラッシュ事故が多い)

    Args:
        components ([type]): [description]
        pn (bool, optional): [description]. Defaults to False.

    Returns:
        str or PyNode:
    """  
    if pn:
        return uniq(pynode(pm.filterExpand(pm.polyListComponentConversion(components, tuv=True), sm=35)))
    else:
        return uniq(cmds.filterExpand(cmds.polyListComponentConversion(components, tuv=True), sm=35))


def to_vtxface(components, pn=False):
    """ [pm/cmds] 
    
    TODO: 引数の型で pm/cmds 判断して pn オプション廃止する (指定し忘れのクラッシュ事故が多い)

    Args:
        components ([type]): [description]
        pn (bool, optional): [description]. Defaults to False.

    Returns:
        str or PyNode:
    """
    if pn:
        return uniq(pynode(pm.filterExpand(pm.polyListComponentConversion(components, tvf=True), sm=70)))
    else:
        return uniq(cmds.filterExpand(cmds.polyListComponentConversion(components, tvf=True), sm=70))


def type_of_component(comp):
    """ [cmds] component の種類を返す

    TODO: 大雑把すぎるので修正する｡種類の網羅｡

    Args:
        comp (str):
    
    Returns:
        str or None: "e", "f", "v", None
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
    """ 多次元配列を一次元配列にする """
    from itertools import chain
    return list(chain.from_iterable(a))


def uniq(a):
    """ 配列の重複要素を取り除く """
    elemtype = type(a[0])
    string_types = [type(""), type(u"")]  # Supports python2.x and python3.x
    if elemtype in string_types:
        elements_tuple_list = list(set([tuple(x) for x in a]))
        return ["".join(elements_tuple) for elements_tuple in elements_tuple_list]
    else:
        return list(set(a))


def round_vector(v, fraction):
    """ ベクトルの各要素をそれぞれ round する
    
    Args:
        v (list[float, float, float]): 

    Returns:
        list[float, float, float]:
    """
    v = [round(x, fraction) for x in v]
    return v


def get_poly_line(edges, intersections=[]):
    """ [pm/cmds] edges を連続するエッジのまとまりとしてエッジリストを一つ返す

    intersections を指定することで実際には連続しているエッジ同士を分離する事が可能
    edges の型で pm/cmds を判断する｡
    
    Args:
        edges (list[str]):
        intersections (list[str or MeshVertex]): 実際にはトポロジーが連続していても連続していないと見なす点のリスト

    Returns:
        list[str]: 

    """
    if isinstance(edges[0], pm.MeshEdge):
        return _get_poly_line_pm(edges, intersections=intersections)

    first_edge = edges[0]
    rest_edges = edges[1:]
    processed_edges = [first_edge]
    processed_vts = []    
    vtx_queue = []
    polyline = [first_edge]

    # edges[0]のvts[0] から開始
    vtx_queue = list_diff(to_vtx(first_edge), intersections)

    while len(vtx_queue) > 0:
        for vtx in vtx_queue:
            # 処理済み頂点にvtx 追加
            processed_vts.append(vtx)
            vtx_queue = list_diff(list_diff(vtx_queue, processed_vts), intersections)  # TODO:本当に set set set より早いか

            # 未処理の隣接エッジの取得
            adjacent_edges = list_intersection(to_edge(vtx), rest_edges)

            if len(adjacent_edges) > 0:
                # 隣接エッジがあれば連続エッジに追加
                polyline.extend(adjacent_edges)

                # 処理済みエッジに追加
                processed_edges.extend(adjacent_edges)
                rest_edges = list_diff(rest_edges, adjacent_edges)

                # 隣接エッジの構成頂点のうち未処理のものをキューに追加する
                vtx_queue.extend(list_diff(list_diff(to_vtx(adjacent_edges), processed_vts), intersections))
            else:
                # 隣接エッジなし
                pass

    return polyline


def get_all_polylines(edges):
    """ [cmds] edges で指定したエッジ列を連続するエッジ列の集まりに分割してリストを返す

    edges の型で pm/cmds を判断する｡

    Args:
        edges (list[str]):

    Returns:
        list[list[str]]: 

    """
    if isinstance(edges[0], pm.MeshEdge):
        return _get_all_polylines_pm(edges)

    polylines = []
    rest_edges = edges
    intersections = [v for v in to_vtx(edges) if len(set(to_edge(v)) & set(edges)) > 2]

    while len(rest_edges) > 0:
        polyline = get_poly_line(rest_edges, intersections)
        polylines.append(polyline)
        rest_edges = list_diff(rest_edges, polyline)

    return polylines


def _get_poly_line_pm(edges, intersections=[]):
    """ [pm] edges を連続するエッジのまとまりとしてエッジリストを一つ返す
    
    get_poly_line() から呼ばれる Pymel 版の実装｡基本的には直接呼ばず get_poly_line() を使う｡
    intersections を指定することで実際には連続しているエッジ同士を分離する事が可能

    Args:
        edges (list[MeshEdge]):
        intersections (list[MeshVertex]): 複数のエッジの交点と見なす頂点

    Returns:
        list[MeshEdge]: 
    """
    first_edge = edges[0]    
    rest_edges = edges[1:]  # 未処理エッジ
    processed_edges = [first_edge]  # 処理済みエッジ
    processed_vts = []  # 処理済み頂点
    polyline = [first_edge]  # 返値
    vtx_queue = list_diff(get_connected_vertices(first_edge), intersections)
        
    while len(vtx_queue) > 0:
        for vtx in vtx_queue:
            # 処理済み頂点にvtx 追加
            processed_vts.append(vtx)
            vtx_queue = list_diff(vtx_queue, processed_vts)

            # 隣接する未処理エッジの取得
            adjacent_edges = list_intersection(vtx.connectedEdges(), rest_edges)

            if len(adjacent_edges) > 0:
                # 隣接エッジがあれば連続エッジに追加
                polyline.extend(adjacent_edges)

                # 処理済みエッジに追加
                processed_edges.extend(adjacent_edges)
                rest_edges = list_diff(rest_edges, adjacent_edges)

                # 隣接エッジの構成頂点のうち未処理のものをキューに追加する
                vtx_queue.extend(list_diff(list_diff(to_vtx(adjacent_edges, pn=True), processed_vts), intersections))
            else:
                # 隣接エッジなし
                pass

    return polyline


def _get_all_polylines_pm(edges):
    """ [pm] edges で指定したエッジ列を連続するエッジ列の集まりに分割してリストを返す

    get_all_polylines() から呼ばれる Pymel 版の実装｡基本的には直接呼ばず get_all_polylines() を使う｡

    Args:
        edges (list[MeshEdge]):

    Returns:
        list[list[MeshEdge]]: 

    """
    polylines = []
    rest_edges = edges
    intersections = [v for v in to_vtx(edges, pn=True) if len(set(to_edge(v, pn=True)) & set(edges)) > 2]

    while len(rest_edges) > 0:
        polyline = _get_poly_line_pm(edges=rest_edges, intersections=intersections)
        polylines.append(polyline)
        rest_edges = list_diff(rest_edges, polyline)

    return polylines


def name_to_uuid(name):
    """ [cmds] ノード名からUUIDを取得する """
    uuid_list = cmds.ls(name, uuid=True)
    if len(uuid_list) == 1:
        return uuid_list[0]
    else:
        raise("name: " + name + "is not unique. try fullpath")


def uuid_to_name(uuid):
    """ [cmds] UUIDからノード名を取得する """
    return cmds.ls(uuid)[0]


def get_fullpath(name):
    """ [cmds] オブジェクト名のフルパスを取得する """
    return cmds.ls(name, l=True)


def get_basename(name):
    """ [pm/cmds] オブジェクト名からベースネームを取得する """
    fullpath = get_fullpath(name)
    return re(r"^.*\|", "", fullpath)


def get_active_camera():
    """" 関数を呼んだ時点でアクティブなパネルでのアクティブカメラの取得 """
    active_panel = cmds.getPanel(wf=True)
    camera = cmds.modelPanel(active_panel, q=True, camera=True)

    return camera


def is_supported_weight_format_option():
    """ ウェイト関連機能の format オプションに対応しているかどうか

    Returns:
        bool:
    """
    ver = int(cmds.about(version=True))
    if ver > 2018:
        return True
    else:
        return False


def get_end_vertices_e(edges):
    """ [pm] 連続するエッジの端の頂点をすべて返す

    Args:
        edges(list[MeshEdge]): エッジリスト (順不同)
    
    Returns:
        list[MeshVertex]:
    """
   
    return get_end_vertices_v(to_vtx(edges, pn=True))


def get_end_vertices_v(vertices):
    """ [pm] 連続したエッジで接続される頂点の端の頂点を返す

    Args:
        vertices (list[MeshVertx]): 頂点リスト (順不同)

    Returns:
        list[MeshVertx]
    """
    end_vertices = []

    for vertex in vertices:
        connected_vertices = get_connected_vertices(vertex)
        if len(set(connected_vertices) & set(vertices)) == 1:
            end_vertices.append(vertex)

    return end_vertices


def sort_edges(edges):
    """ [pm] エッジをトポロジーの連続性でソートする

    Args:
        edges (list[MeshEdge]): 未ソートエッジリスト

    Returns:
        list[MeshEdge]: ソート済エッジリスト
    """
    def part_vertex_list(edges, first_vertex):
        """ 部分エッジ集合と開始頂点から再帰的に頂点列を求める """
        neighbors = to_edge(first_vertex)
        next_edges = list_intersection(neighbors, edges)

        if len(next_edges) == 1:
            next_edge = next_edges[0]
            vset1 = set(to_vtx(next_edge))
            vset2 = {first_vertex}
            next_vertex = list(vset1-vset2)[0]
            rest_edges = list_diff(edges, next_edges)
            partial_vts = part_vertex_list(rest_edges, next_vertex)
            partial_vts.insert(0, first_vertex)
            return partial_vts
        else:
            return [first_vertex]

    first_vertex = get_end_vertices_e(edges)[0]

    return part_vertex_list(edges, first_vertex)


def sort_vertices(vertices):
    """ [pm] 頂点をトポロジーの連続性でソートする

    やや重いのでソート済エッジがすでの存在するならば sorted_edges_to_vertices() 使う

    Args:
        vertices (list[MeshVertex): 未ソート頂点リスト

    Returns:
        list[MeshVertex]: ソート済頂点リスト
    """
    def part_vertex_list(vertices, first_vertex):
        """ 部分エッジ集合と開始頂点から再帰的に頂点列を求める """
        neighbors = first_vertex.connectedVertices()
        next_vertices = list_intersection(neighbors, vertices)

        if len(next_vertices) == 1:
            next_vertex = next_vertices[0]
            rest_vertices = list_diff(vertices, next_vertices)
            partial_vertices = part_vertex_list(rest_vertices, next_vertex)
            partial_vertices.insert(0, first_vertex)
            return partial_vertices
        else:
            return [first_vertex]

    first_vertex = get_end_vertices_v(vertices)[0]
    rest_vertices = list_diff(vertices, [first_vertex])

    return part_vertex_list(rest_vertices, first_vertex)


def sorted_edges_to_vertices(edges):
    """ [pm] ソートされたエッジを同じ順序でソートされた頂点に変換する

    Args:
        edges (list[MeshEdge]): ソート済エッジリスト

    Returns:
        list[MeshVertex]: ソート済頂点リスト
    """
    sorted_vertices = []
    first_vertices = list_diff(edges[0].vertices(), edges[1].vertices())[0]
    sorted_vertices.append(first_vertices)
   
    for i in range(len(edges)-1):
        edge = edges[i]
        next_edge = edges[i+1]
        shared_vertex = list_intersection(edge.vertices(), next_edge.vertices())[0]
        sorted_vertices.append(shared_vertex)

    return sorted_vertices


def sorted_vertices_to_edges(vertices):
    """ [pm] ソートされた頂点を同じ順序でソートされたエッジに変換する
    
    Args:
        list[MeshVertex]: ソート済頂点リスト

    Returns:
        edges (list[MeshEdge]): ソート済エッジリスト
    """
    sorted_edges = []

    for i in range(len(vertices)-1):
        vertex = vertices[i]
        next_vertex = vertices[i+1]
        shared_edge = list_intersection(vertex.connectedEdges(), next_vertex.connectedEdges())[0]
        sorted_edges.append(shared_edge)
    
    return sorted_edges

