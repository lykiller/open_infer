# -*- coding: utf-8 -*-
import os.path
import json
from typing import Dict
from enum import Enum, unique
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from typing import List
import pandas as pd
from dbfread import DBF
from av2.map.map_api import LaneSegment
import av2.geometry.polyline_utils as polyline_utils
import av2.rendering.vector as vector_plotting_utils
from pyproj import Transformer
import matplotlib


PURPLE_RGB = [201, 71, 245]
PURPLE_RGB_MPL = np.array(PURPLE_RGB) / 255

DARK_GRAY_RGB = [40, 39, 38]
DARK_GRAY_RGB_MPL = np.array(DARK_GRAY_RGB) / 255

_COLORS = (
    np.array(
        [0.000, 0.447, 0.741, 0.850, 0.325, 0.098, 0.929, 0.694, 0.125, 0.494, 0.184, 0.556, 0.466,
         0.674, 0.188, 0.301,
         0.745, 0.933, 0.635, 0.078, 0.184, 0.300, 0.300, 0.300, 0.600, 0.600, 0.600, 1.000, 0.000,
         0.000, 1.000, 0.500,
         0.000, 0.749, 0.749, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 1.000, 0.667, 0.000, 1.000,
         0.333, 0.333, 0.000,
         0.333, 0.667, 0.000, 0.333, 1.000, 0.000, 0.667, 0.333, 0.000, 0.667, 0.667, 0.000, 0.667,
         1.000, 0.000, 1.000,
         0.333, 0.000, 1.000, 0.667, 0.000, 1.000, 1.000, 0.000, 0.000, 0.333, 0.500, 0.000, 0.667,
         0.500, 0.000, 1.000,
         0.500, 0.333, 0.000, 0.500, 0.333, 0.333, 0.500, 0.333, 0.667, 0.500, 0.333, 1.000, 0.500,
         0.667, 0.000, 0.500,
         0.667, 0.333, 0.500, 0.667, 0.667, 0.500, 0.667, 1.000, 0.500, 1.000, 0.000, 0.500, 1.000,
         0.333, 0.500, 1.000,
         0.667, 0.500, 1.000, 1.000, 0.500, 0.000, 0.333, 1.000, 0.000, 0.667, 1.000, 0.000, 1.000,
         1.000, 0.333, 0.000,
         1.000, 0.333, 0.333, 1.000, 0.333, 0.667, 1.000, 0.333, 1.000, 1.000, 0.667, 0.000, 1.000,
         0.667, 0.333, 1.000,
         0.667, 0.667, 1.000, 0.667, 1.000, 1.000, 1.000, 0.000, 1.000, 1.000, 0.333, 1.000, 1.000,
         0.667, 1.000, 0.333,
         0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000, 0.000, 1.000, 0.000,
         0.000, 0.000, 0.167,
         0.000, 0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000,
         0.000, 1.000, 0.000,
         0.000, 0.000, 0.167, 0.000, 0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000,
         0.000, 0.833, 0.000,
         0.000, 1.000, 0.000, 0.000, 0.000, 0.143, 0.143, 0.143, 0.286, 0.286, 0.286, 0.429, 0.429,
         0.429, 0.571, 0.571,
         0.571, 0.714, 0.714, 0.714, 0.857, 0.857, 0.857, 0.000, 0.447, 0.741, 0.314, 0.717, 0.741,
         0.50, 0.5, 0,
         ]
    )
    .astype(np.float32)
    .reshape(-1, 3)
)


def create_lane_segment_json(lane_link: pd.DataFrame) -> (list, list):
    """
    根据HAD_LANE_LINK.shp读取road的id、交叉路口车道类型信息
    :param lane_link:
    :return:
    """
    lane_segment_json = []
    lane_link_id = list(lane_link.LaneLinkID)
    for ind, curr_lane_id in enumerate(lane_link_id):
        json_data = {"id": int(float(curr_lane_id)), "lane_type": "VEHICLE",
                     "left_lane_mark_type": "None",
                     "right_lane_mark_type": "None", "left_neighbor_id": None,
                     "right_neighbor_id": None, "is_intersection": False, "predecessors": [],
                     "successors": []}
        if lane_link.IsInters[ind] == 1:
            json_data["is_intersection"] = True
        else:
            json_data["is_intersection"] = False

        lane_type = lane_link.LANE_TYPE[ind]
        lane_type_bin = bin(lane_type)[2:]  # 删除“0b”
        if len(lane_type_bin) < 32:
            padding_zeros = '0' * (32 - len(lane_type_bin))
            lane_type_bin = padding_zeros + lane_type_bin
        if lane_type_bin[0] == '1':
            json_data["lane_type"] = "VEHICLE"
        elif lane_type_bin[18] == '1':
            json_data["lane_type"] = "BUS"
        elif lane_type_bin[19] == '1':
            json_data["lane_type"] = "BIKE"
        lane_segment_json.append(json_data)
    return lane_segment_json, lane_link_id


def collect_lane_info(lane_segment_json: list, lane_neighbor: dict, lane_boundary: dict,
                      boundary_coord: dict, boundary_attri: dict):
    """
    收集各关系表，并以此索引转换为用于初始化 LaneSegment 的 json 数据
    :param lane_segment_json: 初始化的车道信息
    :param lane_neighbor:  车道左右邻对应关系
    :param lane_boundary: 车道与左右车道线id对应关系
    :param boundary_coord: 车道线坐标
    :param boundary_attri:车道线属性，颜色，形状
    :return:
    """
    new_lane_segment_json = []
    for ind in range(len(lane_segment_json)):
        store_flag = True
        lane_link_id = lane_segment_json[ind]["id"]

        # 添加车道左右邻
        if lane_link_id in lane_neighbor:
            lane_segment_json[ind]["left_neighbor_id"] = lane_neighbor[lane_link_id][
                "left_neighbor_id"]
            lane_segment_json[ind]["right_neighbor_id"] = lane_neighbor[lane_link_id][
                "right_neighbor_id"]

        if lane_link_id in lane_boundary:
            left_boundary_id = lane_boundary[lane_link_id]["left_boundary_id"]
            right_boundary_id = lane_boundary[lane_link_id]["right_boundary_id"]
            # 添加左右车道线属性
            if left_boundary_id in boundary_attri:
                lane_segment_json[ind]["left_lane_mark_type"] = boundary_attri[left_boundary_id]
            else:
                print("doesn't exist boundary_id in boundary_attri:{}".format(left_boundary_id))
                lane_segment_json[ind]["left_lane_mark_type"] = "NONE"

            if right_boundary_id in boundary_attri:
                lane_segment_json[ind]["right_lane_mark_type"] = boundary_attri[right_boundary_id]
            else:
                print("doesn't exist boundary_id in boundary_attri:{}".format(right_boundary_id))
                lane_segment_json[ind]["right_lane_mark_type"] = "NONE"

            # 添加左右车道线坐标
            if left_boundary_id in boundary_coord:
                lane_segment_json[ind]["left_lane_boundary"] = boundary_coord[left_boundary_id]
            else:
                print("failed to find left_lane_boundary:{}".format(left_boundary_id))
                store_flag = False

            if right_boundary_id in boundary_coord:
                lane_segment_json[ind]["right_lane_boundary"] = boundary_coord[right_boundary_id]
            else:
                print("failed to find right_lane_boundary:{}".format(right_boundary_id))
                store_flag = False
        else:
            print("failed to find lane link id:{}".format(lane_link_id))
            store_flag = False

        if store_flag:
            new_lane_segment_json.append(lane_segment_json[ind])
    return new_lane_segment_json


def find_successor_predecessors(lane_topo_table: pd.DataFrame, lane_segment_json: list):
    """
    此处id表示连接ID，需要根据 ，LANE_MARK_REL找到左右邻，
    :param lane_topo_table:车道前后继topo表
    :param lane_segment_json:车道段的字典信息
    :return:
    """
    successor_lane_topo = dict()
    predecessors_lane_topo = dict()
    for ind in range(len(lane_topo_table.InLaneID)):
        in_lane_id = int(float(lane_topo_table.InLaneID[ind]))
        out_lane_id = int(float(lane_topo_table.OutLaneID[ind]))
        if out_lane_id in predecessors_lane_topo:
            predecessors_lane_topo[out_lane_id].append(in_lane_id)
        else:
            predecessors_lane_topo[out_lane_id] = [in_lane_id]

        if in_lane_id in successor_lane_topo:
            successor_lane_topo[in_lane_id].append(out_lane_id)
        else:
            successor_lane_topo[in_lane_id] = [out_lane_id]

    for ind in range(len(lane_segment_json)):
        lane_id = lane_segment_json[ind]["id"]
        if lane_id in predecessors_lane_topo:
            lane_segment_json[ind]["predecessors"] = predecessors_lane_topo[lane_id]
        if lane_id in successor_lane_topo:
            lane_segment_json[ind]["successors"] = successor_lane_topo[lane_id]
    return lane_segment_json


def find_boundary_coord(lane_mark_link: pd.DataFrame):
    """
    获取车道边界线的经纬度信息
    :param lane_mark_link:
    :return 车道线id，对应经纬度
    """
    # 参数1：WGS84地理坐标系统 对应 4326
    # 参数2：坐标系WKID 广州市 WGS_1984_UTM_Zone_49N 对应 32649
    transformer = Transformer.from_crs("epsg:4326", "epsg:32650")
    # test 以第一个点为坐标原点
    offset_x = 0 # 228219.8316448916
    offset_y = 0 # 3377537.1485753627
    boundary_coord = {}
    for ind in range(len(lane_mark_link.BoundaryID)):
        boundary_id = int(float(lane_mark_link.BoundaryID[ind]))
        pts = []
        for pt in lane_mark_link.geometry[ind].coords:
            lat = pt[1]
            lon = pt[0]
            x, y, z = transformer.transform(lat, lon, 0)
            pts.append({"x": (x - offset_x), "y": (y - offset_y), "z": z})
        boundary_coord[boundary_id] = pts
    return boundary_coord


def find_boundary_attri(lane_mark_link_marking: pd.DataFrame):
    """
    获取车道线的颜色、形状属性
    :param lane_mark_link_marking: 打开的dbf文件
    :return: 车道线id对应的属性
    """
    boundary_attri = {}
    for ind in range(len(lane_mark_link_marking.BoundaryID)):
        boundary_id = int(float(lane_mark_link_marking.BoundaryID[ind]))
        color = lane_mark_link_marking.MARK_COLOR[ind]
        type = lane_mark_link_marking.MARK_TYPE[ind]
        if 1 == type and 1 == color:
            lane_mark_type = "SOLID_WHITE"
        elif 1 == type and 2 == color:
            lane_mark_type = "SOLID_YELLOW"
        elif 1 == type and 6 == color:
            lane_mark_type = "SOLID_BLUE"
        elif 2 == type and 1 == color:
            lane_mark_type = "DASHED_WHITE"
        elif 2 == type and 2 == color:
            lane_mark_type = "DASHED_YELLOW"
        else:
            lane_mark_type = "NONE"
        boundary_attri[boundary_id] = lane_mark_type
    return boundary_attri


def find_lane_boundary(lane_mark_rel_table: pd.DataFrame) -> (dict, dict):
    """
    根据关系表找到车道的左右车道线
    :param lane_mark_rel_table:
    :return:
    """
    lane_boundary = {}
    for ind in range(len(lane_mark_rel_table.LaneLinkID)):
        lane_link_id = int(float(lane_mark_rel_table.LaneLinkID[ind]))
        boundary_id = int(float(lane_mark_rel_table.BoundaryID[ind]))
        side = int(float(lane_mark_rel_table.SIDE[ind]))
        direct = int(float(lane_mark_rel_table.DIRECT[ind]))
        if lane_link_id not in lane_boundary:
            lane_boundary[lane_link_id] = {}
        if 2 == side:
            lane_boundary[lane_link_id]["left_boundary_id"] = boundary_id
        elif 1 == side:
            lane_boundary[lane_link_id]["right_boundary_id"] = boundary_id

    return lane_boundary


def find_lane_neighbor(lane_boundary: dict):
    # 根据左右车道线判断车道相邻的关系
    lane_neighbor = {}
    vector_lane_boundary = lane_boundary.items()
    for lb in vector_lane_boundary:
        lane_id = lb[0]
        left_boundary_id = lb[1]["left_boundary_id"]
        right_boundary_id = lb[1]["right_boundary_id"]

        lane_neighbor[lane_id] = {}
        for search_lane in vector_lane_boundary:
            search_lane_id = search_lane[0]
            search_left_boundary_id = search_lane[1]["left_boundary_id"]
            search_right_boundary_id = search_lane[1]["right_boundary_id"]
            if lane_id == search_lane_id:
                continue
            if right_boundary_id == search_left_boundary_id:
                lane_neighbor[lane_id]["right_neighbor_id"] = search_lane_id
            if left_boundary_id == search_right_boundary_id:
                lane_neighbor[lane_id]["left_neighbor_id"] = search_lane_id
        check_neighbor = lane_neighbor[lane_id]
        if "right_neighbor_id" not in check_neighbor:
            lane_neighbor[lane_id]["right_neighbor_id"] = None
        if "left_neighbor_id" not in check_neighbor:
            lane_neighbor[lane_id]["left_neighbor_id"] = None
    return lane_neighbor


def convert_argoversev2(root_path, saved_path):
    # 获取车道id，类型，交叉口信息
    lane_link_shp_file = os.path.join(root_path, "HAD_LANE_LINK.shp")
    gdf = gpd.read_file(lane_link_shp_file)
    lane_segment_json, lane_link_id = create_lane_segment_json(gdf)

    # 获取车道前后继关系
    lane_topo = os.path.join(root_path, "HAD_LANE_TOPO_DETAIL.dbf")
    lane_topo_table = DBF(lane_topo, encoding='GBK')
    lane_topo_table_df = pd.DataFrame(iter(lane_topo_table))
    lane_segment_json = find_successor_predecessors(lane_topo_table_df,
                                                    lane_segment_json)
    # 获取车道线信息-经纬度
    lane_mark_link = os.path.join(root_path, "HAD_LANE_MARK_LINK.shp")
    gdf = gpd.read_file(lane_mark_link)
    boundary_coord = find_boundary_coord(gdf)
    # 获取车道线信息-属性
    lane_mark_link_marking = os.path.join(root_path, "HAD_LANE_MARK_LINK_MARKING.dbf")
    lane_mark_link_marking_table = DBF(lane_mark_link_marking, encoding='GBK')
    lane_mark_link_marking_df = pd.DataFrame(iter(lane_mark_link_marking_table))
    boundary_attri = find_boundary_attri(lane_mark_link_marking_df)

    # 获取车道左右车道线id
    lane_mark_rel_dbf = os.path.join(root_path, "HAD_LANE_MARK_REL.dbf")
    lane_mark_rel_table = DBF(lane_mark_rel_dbf, encoding='GBK')
    lane_mark_rel_table_df = pd.DataFrame(iter(lane_mark_rel_table))
    lane_boundary = find_lane_boundary(lane_mark_rel_table_df)

    # 获取车道左右邻车道
    lane_neighbor = find_lane_neighbor(lane_boundary)

    lane_segment_json = collect_lane_info(lane_segment_json, lane_neighbor, lane_boundary,
                                          boundary_coord, boundary_attri)
    lane_segment_json = check_lane_topo(lane_segment_json)
    # test
    lane_segment_write_json("test.json", lane_segment_json)
    # init LaneSegment
    lane_segments = []
    for json_data in lane_segment_json:
        ls = LaneSegment.from_dict(json_data)
        lane_segments.append(ls)
    return lane_segments


def display_elem(root_path):
    shp_files = [
        "HAD_LANE_LINK.shp",
        # "HAD_LANE_MARK_LINK.shp",
        # "HAD_LANE_MARK_NODE.shp",
        "HAD_LANE_NODE.shp",
        "HAD_LINK.shp",
        # "HAD_MESH.shp",
        "HAD_NODE.shp",
        # "HAD_OBJECT_ARROW.shp",
        "HAD_OBJECT_STOP_LOCATION.shp",
        # "HAD_PA.shp",
        "HAD_OBJECT_CROSS_WALK.shp"
    ]

    # 可视化地图
    fig, ax = plt.subplots(figsize=(30, 30))
    for ind, shp_file in enumerate(shp_files):
        file = os.path.join(root_path, shp_file)
        gdf = gpd.read_file(file)
        color = (_COLORS[ind % 80]).tolist()
        gdf.plot(ax=ax, color=(color[0], color[1], color[2]))

    plt.show()


def check_lane_topo(lane_segment_json: list):
    """
    确保车道段中topo关系涉及的ID都存在
    :param lane_segment_json:
    :return:
    """
    for ind, ls in enumerate(lane_segment_json):
        curr_id = ls["id"]
        left_neighbor_id = ls["left_neighbor_id"]
        right_neighbor_id = ls["right_neighbor_id"]
        # 确定左右邻车道是否存在
        left_flag = False
        right_flag = False
        # 确定前后继车道是否存在
        successor_flag = [False] * len(ls["successors"])
        predecessor_flag = [False] * len(ls["predecessors"])
        for search_ls in lane_segment_json:
            search_id = search_ls["id"]
            if search_id != curr_id:
                if left_neighbor_id == search_id or left_neighbor_id is None:
                    left_flag = True
                if right_neighbor_id == search_id or right_neighbor_id is None:
                    right_flag = True
                for index, tmp_id in enumerate(ls["predecessors"]):
                    if tmp_id == search_id:
                        predecessor_flag[index] = True
                for index, tmp_id in enumerate(ls["successors"]):
                    if tmp_id == search_id:
                        successor_flag[index] = True

        if not left_flag:
            print("doesn't exist lane:{}".format(ls["left_neighbor_id"]))
            lane_segment_json[ind]["left_neighbor_id"] = None
        if not right_flag:
            print("doesn't exist lane:{}".format(ls["right_neighbor_id"]))
            lane_segment_json[ind]["right_neighbor_id"] = None
        # 确定前后继车道是否存在
        cnt = 0
        for tmp_ind, flag in enumerate(successor_flag):
            if not flag:
                tmp = lane_segment_json[ind]["successors"].pop(tmp_ind + cnt)
                print("doesn't exist lane:{}".format(tmp))
                cnt -= 1
        cnt = 0
        for tmp_ind, flag in enumerate(predecessor_flag):
            if not flag:
                tmp = lane_segment_json[ind]["predecessors"].pop(tmp_ind + cnt)
                print("doesn't exist lane:{}".format(tmp))
                cnt -= 1
    return lane_segment_json


def lane_segment_write_json(json_name: str, lane_segments: list):
    fp = open(json_name, "w")
    json.dump(lane_segments, fp, sort_keys=False, indent=2)
    fp.close()
    return


def plot_lane_segments(
        ax: matplotlib.axes.Axes, lane_segments: List[LaneSegment],
        lane_color: np.ndarray = DARK_GRAY_RGB_MPL
) -> None:
    """

    Args:
        ax:
        lane_segments:
    """
    for ls in lane_segments:
        pts_city = ls.polygon_boundary
        ALPHA = 1.0  # 0.1
        vector_plotting_utils.plot_polygon_patch_mpl(
            polygon_pts=pts_city, ax=ax, color=lane_color, alpha=ALPHA, zorder=1
        )

        for bound_type, bound_city in zip(
                [ls.left_mark_type, ls.right_mark_type],
                [ls.left_lane_boundary, ls.right_lane_boundary]
        ):
            if "YELLOW" in bound_type:
                mark_color = "y"
            elif "WHITE" in bound_type:
                mark_color = "w"
            else:
                mark_color = "grey"  # "b" lane_color #

            LOOSELY_DASHED = (0, (5, 10))

            if "DASHED" in bound_type:
                linestyle = LOOSELY_DASHED
            else:
                linestyle = "solid"

            if "DOUBLE" in bound_type:
                left, right = polyline_utils.get_double_polylines(
                    polyline=bound_city.xyz[:, :2], width_scaling_factor=0.1
                )
                ax.plot(left[:, 0], left[:, 1], mark_color, alpha=ALPHA, linestyle=linestyle,
                        zorder=2)
                ax.plot(right[:, 0], right[:, 1], mark_color, alpha=ALPHA, linestyle=linestyle,
                        zorder=2)
            else:
                ax.plot(
                    bound_city.xyz[:, 0],
                    bound_city.xyz[:, 1],
                    mark_color,
                    alpha=ALPHA,
                    linestyle=linestyle,
                    zorder=2,
                )


def display_lane_segment(lane_segments: list):
    fig = plt.figure(1, figsize=(10, 10))
    ax = fig.add_subplot(111)
    view_radius_m = 400
    xlims = [-view_radius_m, view_radius_m]
    ylims = [-view_radius_m, view_radius_m]
    plot_lane_segments(ax=ax, lane_segments=lane_segments)

    plt.axis("equal")
    plt.xlim(*xlims)
    plt.ylim(*ylims)
    plt.title("hdmap")
    plt.axis("off")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    # 读取 Shapefile 文件
    root_path = r"G:\hdmap\0425\SHP_XY"
    saved_path = r"G:\hdmap\0425"
    display_elem(root_path)
    vector_lane_segment = convert_argoversev2(root_path, saved_path)
    display_lane_segment(vector_lane_segment)
