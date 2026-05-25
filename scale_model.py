#!/usr/bin/env python3
"""
Scale a MuJoCo arm XML to kinematic shoulder->elbow and elbow->paw lengths
WITHOUT rewriting STL files. Fixed v14: v13 baseline plus compiled-iteration solve for upper axial scale; keeps v12 hand mesh-position compensation.

This version scales visible meshes by duplicating <mesh> assets that point to the
same STL file and setting XML mesh scale="sx sy sz". Original mesh files/paths are
not overwritten.

Important limitation: MuJoCo XML mesh scale is diagonal in mesh coordinates. It
cannot represent an arbitrary axial scale about an arbitrary 3D direction unless
that direction is aligned with a mesh/body coordinate axis. This script therefore
uses the dominant local coordinate axis for each bone/body and applies the same
coordinate-axis scale to sites, joints, child-body positions, inertials, and mesh
scale. That keeps the visible mesh and MuJoCo attachment points consistent.
"""
from __future__ import annotations

import argparse
import csv
import struct
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

try:
    import mujoco
except Exception as e:  # pragma: no cover
    mujoco = None
    MUJOCO_IMPORT_ERROR = e
else:
    MUJOCO_IMPORT_ERROR = None


def opensim_to_mujoco(v: Sequence[float]) -> np.ndarray:
    x, y, z = v
    return np.array([z, x, y], dtype=np.float64)


def parse_vec(text: str, n: Optional[int] = None) -> np.ndarray:
    vals = np.array([float(x) for x in text.replace(',', ' ').split()], dtype=np.float64)
    if n is not None and vals.size != n:
        raise ValueError(f"expected {n} values, got {vals.size}: {text!r}")
    return vals


def fmt_float(x: float) -> str:
    if abs(x) < 5e-15:
        x = 0.0
    return f"{float(x):.12g}"


def fmt_vec(v: Sequence[float]) -> str:
    return ' '.join(fmt_float(float(x)) for x in v)


def quat_to_mat(q: Sequence[float]) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    if q.size != 4:
        raise ValueError('quat must have 4 values')
    n = np.linalg.norm(q)
    if n <= 1e-15:
        return np.eye(3)
    w, x, y, z = q / n
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y],
    ], dtype=np.float64)


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    # Not currently used, kept for extension.
    raise NotImplementedError


@dataclass
class Frame:
    xpos: np.ndarray
    xmat: np.ndarray

    def local_to_world(self, p: np.ndarray) -> np.ndarray:
        return self.xpos + self.xmat @ p

    def world_to_local(self, p: np.ndarray) -> np.ndarray:
        return self.xmat.T @ (p - self.xpos)


@dataclass
class DiagScaleLocal:
    origin: np.ndarray
    diag: np.ndarray

    def apply(self, p: np.ndarray) -> np.ndarray:
        return self.origin + self.diag * (p - self.origin)


def iter_subtree(body: ET.Element) -> Iterable[ET.Element]:
    yield body
    for child in body.findall('body'):
        yield from iter_subtree(child)


def build_body_map(root: ET.Element) -> Dict[str, ET.Element]:
    return {b.attrib['name']: b for b in root.findall('.//body') if 'name' in b.attrib}


def build_mesh_map(root: ET.Element) -> Dict[str, ET.Element]:
    return {m.attrib['name']: m for m in root.findall('./asset/mesh') if 'name' in m.attrib}


def read_kinematic_lengths(csv_path: Path, statistic: str, frame: int) -> Tuple[float, float, int]:
    uppers, lowers = [], []
    with csv_path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        req = ['shoulder_x','shoulder_y','shoulder_z','elbow_x','elbow_y','elbow_z','paw_x','paw_y','paw_z']
        missing = [c for c in req if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f'Missing kinematic columns: {missing}')
        for i, row in enumerate(reader):
            if statistic == 'frame' and i != frame:
                continue
            shoulder = opensim_to_mujoco([float(row['shoulder_x']), float(row['shoulder_y']), float(row['shoulder_z'])])
            elbow = opensim_to_mujoco([float(row['elbow_x']), float(row['elbow_y']), float(row['elbow_z'])])
            paw = opensim_to_mujoco([float(row['paw_x']), float(row['paw_y']), float(row['paw_z'])])
            uppers.append(float(np.linalg.norm(elbow - shoulder)))
            lowers.append(float(np.linalg.norm(paw - elbow)))
            if statistic == 'frame':
                break
    if not uppers:
        raise ValueError('No kinematic frames read')
    u = np.asarray(uppers, dtype=np.float64)
    l = np.asarray(lowers, dtype=np.float64)
    if statistic == 'median':
        return float(np.median(u)), float(np.median(l)), len(uppers)
    if statistic == 'mean':
        return float(np.mean(u)), float(np.mean(l)), len(uppers)
    if statistic == 'frame':
        return float(u[0]), float(l[0]), 1
    raise ValueError(statistic)


def require_mujoco():
    if mujoco is None:
        raise RuntimeError(f'Could not import mujoco: {MUJOCO_IMPORT_ERROR}')


def get_id(model, objtype, name: str) -> int:
    idx = mujoco.mj_name2id(model, objtype, name)
    if idx < 0:
        raise ValueError(f'Could not find {objtype} named {name!r}')
    return int(idx)


def set_qpos_angles(model, data, angles: Dict[str, float]) -> None:
    for name, val in angles.items():
        jid = get_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[int(model.jnt_qposadr[jid])] = float(val)


def compile_model_points(xml_path: Path, angles: Dict[str, float]):
    require_mujoco()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    set_qpos_angles(model, data, angles)
    if model.nu:
        data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    shoulder_jid = get_id(model, mujoco.mjtObj.mjOBJ_JOINT, 'elv_angle')
    elbow_jid = get_id(model, mujoco.mjtObj.mjOBJ_JOINT, 'elbow_flex')
    hand_sid = get_id(model, mujoco.mjtObj.mjOBJ_SITE, 'handm')
    shoulder = np.array(data.xanchor[shoulder_jid], dtype=np.float64).copy()
    elbow = np.array(data.xanchor[elbow_jid], dtype=np.float64).copy()
    hand = np.array(data.site_xpos[hand_sid], dtype=np.float64).copy()
    frames: Dict[str, Frame] = {}
    for bid in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if name:
            frames[name] = Frame(
                np.array(data.xpos[bid], dtype=np.float64).copy(),
                np.array(data.xmat[bid], dtype=np.float64).reshape(3, 3).copy(),
            )
    return model, data, shoulder, elbow, hand, frames


def shape_scales(length_scale: float, mode: str, transverse_exponent: float) -> Tuple[float, float]:
    if mode == 'axial':
        return length_scale, 1.0
    if mode == 'uniform':
        return length_scale, length_scale
    if mode == 'allometric':
        return length_scale, length_scale ** transverse_exponent
    raise ValueError(mode)


def dominant_axis_diag(local_axis: np.ndarray, axial: float, transverse: float) -> Tuple[np.ndarray, int]:
    local_axis = np.asarray(local_axis, dtype=np.float64)
    local_axis = local_axis / np.linalg.norm(local_axis)
    idx = int(np.argmax(np.abs(local_axis)))
    diag = np.full(3, transverse, dtype=np.float64)
    diag[idx] = axial
    return diag, idx


def exact_axial_for_target_length(local_delta: np.ndarray, target_length: float, transverse: float) -> Tuple[float, int]:
    """
    Pick the dominant local coordinate axis and solve the axial scale that makes
    the transformed 3D endpoint distance equal target_length exactly.

    If d is the endpoint vector in the scaling frame and axis i is chosen, then

        target_length**2 = (a*d[i])**2 + sum((transverse*d[j])**2 for j != i)

    so a can be solved directly.  This avoids the old approximation of setting
    a = target_length / ||d||, which is only exact when d lies perfectly on axis i.
    """
    d = np.asarray(local_delta, dtype=np.float64)
    if d.size != 3:
        raise ValueError('local_delta must have 3 values')
    idx = int(np.argmax(np.abs(d)))
    axial_component2 = float(d[idx] * d[idx])
    if axial_component2 <= 1e-24:
        raise ValueError('Cannot solve axial scale because dominant component is zero')

    transverse_component2 = float(sum((transverse * d[j]) ** 2 for j in range(3) if j != idx))
    needed2 = float(target_length * target_length) - transverse_component2
    if needed2 < -1e-12:
        raise ValueError(
            'Target length is smaller than the fixed transverse components; '
            'cannot hit it with this axial-only XML scale'
        )
    needed2 = max(0.0, needed2)
    return float(np.sqrt(needed2 / axial_component2)), idx


def transform_pos_attr(elem: ET.Element, attr: str, xf: DiagScaleLocal, report: Dict[str, int]) -> None:
    if attr in elem.attrib:
        elem.attrib[attr] = fmt_vec(xf.apply(parse_vec(elem.attrib[attr], 3)))
        report[f'transformed_{elem.tag}_{attr}'] = report.get(f'transformed_{elem.tag}_{attr}', 0) + 1


def scale_inertial(inertial: ET.Element, axial: float, transverse: float, report: Dict[str, int]) -> None:
    volume_scale = axial * transverse * transverse
    if 'mass' in inertial.attrib:
        inertial.attrib['mass'] = fmt_float(float(inertial.attrib['mass']) * volume_scale)
        report['scaled_inertial_mass'] = report.get('scaled_inertial_mass', 0) + 1
    if 'diaginertia' in inertial.attrib:
        vals = parse_vec(inertial.attrib['diaginertia'], 3)
        inertia_scale = volume_scale * (axial * axial + 2 * transverse * transverse) / 3.0
        inertial.attrib['diaginertia'] = fmt_vec(vals * inertia_scale)
        report['scaled_inertial_diaginertia'] = report.get('scaled_inertial_diaginertia', 0) + 1


def transform_body_contents(body: ET.Element, xf: DiagScaleLocal, axial: float, transverse: float,
                            no_inertials: bool, report: Dict[str, int]) -> None:
    inertial = body.find('inertial')
    if inertial is not None:
        transform_pos_attr(inertial, 'pos', xf, report)
        if not no_inertials:
            scale_inertial(inertial, axial, transverse, report)

    for elem in body.findall('site'):
        transform_pos_attr(elem, 'pos', xf, report)
    for elem in body.findall('joint'):
        transform_pos_attr(elem, 'pos', xf, report)

    # Critical: mesh geoms often omit pos, which means pos=0.
    # If we scale the mesh asset but leave geom pos absent, MuJoCo scales the
    # visible vertices around the body origin, while the sites/joints are scaled
    # around the anatomical anchor.  That makes bones look the right size but
    # shifted away from the skeleton.  Therefore geom pos must always be written,
    # even when it was implicit zero.
    for elem in body.findall('geom'):
        old_pos = parse_vec(elem.attrib.get('pos', '0 0 0'), 3)
        elem.attrib['pos'] = fmt_vec(xf.apply(old_pos))
        report['transformed_geom_pos'] = report.get('transformed_geom_pos', 0) + 1

    for tag in ('camera', 'light'):
        for elem in body.findall(tag):
            transform_pos_attr(elem, 'pos', xf, report)


def child_joint_local(child: ET.Element, joint_name: str) -> Optional[np.ndarray]:
    for j in child.findall('joint'):
        if j.attrib.get('name') == joint_name:
            return parse_vec(j.attrib.get('pos', '0 0 0'), 3)
    return None


def set_child_body_pos_to_put_local_point_at_world(child: ET.Element, local_point: np.ndarray,
                                                   target_world: np.ndarray, parent_frame: Frame,
                                                   child_frame_original: Frame, report: Dict[str, int]) -> Frame:
    new_child_world_origin = target_world - child_frame_original.xmat @ local_point
    child.attrib['pos'] = fmt_vec(parent_frame.world_to_local(new_child_world_origin))
    report['anchored_child_body_pos'] = report.get('anchored_child_body_pos', 0) + 1
    return Frame(new_child_world_origin, child_frame_original.xmat.copy())


def propagate_translation_to_subtree_frames(body: ET.Element, frames: Dict[str, Frame], old_frame: Frame, new_frame: Frame) -> None:
    delta = new_frame.xpos - old_frame.xpos
    for b in iter_subtree(body):
        name = b.attrib.get('name')
        if name in frames:
            frames[name] = Frame(frames[name].xpos + delta, frames[name].xmat.copy())



def refresh_child_frames_from_xml(parent: ET.Element, frames: Dict[str, Frame], report: Dict[str, int]) -> None:
    """
    After a parent-local transform changes child <body pos>, update cached frames
    for those children before later processing their own sites/geoms/meshes.

    Orientations are unchanged by this scaler, so only world origins translate.
    Descendant frames are shifted by the same delta until their own body pos is
    explicitly transformed later.
    """
    parent_name = parent.attrib.get('name')
    if not parent_name or parent_name not in frames:
        return
    parent_frame = frames[parent_name]
    for child in parent.findall('body'):
        child_name = child.attrib.get('name')
        if not child_name or child_name not in frames:
            continue
        old_frame = frames[child_name]
        child_pos = parse_vec(child.attrib.get('pos', '0 0 0'), 3)
        new_origin = parent_frame.local_to_world(child_pos)
        new_frame = Frame(new_origin, old_frame.xmat.copy())
        propagate_translation_to_subtree_frames(child, frames, old_frame, new_frame)
        report['refreshed_child_frames'] = report.get('refreshed_child_frames', 0) + 1

def transform_child_body_positions(parent: ET.Element, xf: DiagScaleLocal, report: Dict[str, int]) -> None:
    for child in parent.findall('body'):
        old_pos = parse_vec(child.attrib.get('pos', '0 0 0'), 3)
        child.attrib['pos'] = fmt_vec(xf.apply(old_pos))
        report['transformed_child_body_pos'] = report.get('transformed_child_body_pos', 0) + 1


def transform_descendant_body_origins_in_world(
    root_body: ET.Element,
    frames: Dict[str, Frame],
    world_xf,
    report: Dict[str, int],
) -> None:
    """
    Move every descendant body origin through one shared world-space transform.

    This is used for the lower limb after the ulna has been anchored at the
    transformed elbow.  It keeps nested bodies such as radius -> hand placed
    consistently relative to the ulna, instead of only moving direct children
    of ulna or reinterpreting each child-body offset in a different local frame.
    """
    old_frames = {
        b.attrib['name']: Frame(frames[b.attrib['name']].xpos.copy(), frames[b.attrib['name']].xmat.copy())
        for b in iter_subtree(root_body)
        if 'name' in b.attrib and b.attrib['name'] in frames
    }

    def recurse(parent: ET.Element) -> None:
        parent_name = parent.attrib.get('name')
        if not parent_name or parent_name not in frames:
            return
        parent_frame_new = frames[parent_name]

        for child in parent.findall('body'):
            child_name = child.attrib.get('name')
            if not child_name or child_name not in old_frames:
                continue

            old_child_frame = old_frames[child_name]
            new_child_world_origin = world_xf(old_child_frame.xpos)
            child.attrib['pos'] = fmt_vec(parent_frame_new.world_to_local(new_child_world_origin))
            frames[child_name] = Frame(new_child_world_origin, old_child_frame.xmat.copy())
            report['transformed_descendant_body_origin'] = report.get('transformed_descendant_body_origin', 0) + 1
            recurse(child)

    recurse(root_body)



def copy_frames(frames: Dict[str, Frame]) -> Dict[str, Frame]:
    return {name: Frame(fr.xpos.copy(), fr.xmat.copy()) for name, fr in frames.items()}


def transform_local_pos_by_world_map(
    old_local: np.ndarray,
    old_frame: Frame,
    new_frame: Frame,
    world_xf,
) -> np.ndarray:
    """Map a local point by a world-space transform and express it in the new body frame."""
    return new_frame.world_to_local(world_xf(old_frame.local_to_world(old_local)))


def transform_pos_attr_by_world_map(
    elem: ET.Element,
    attr: str,
    old_frame: Frame,
    new_frame: Frame,
    world_xf,
    report: Dict[str, int],
    force: bool = False,
) -> None:
    if attr in elem.attrib or force:
        old_local = parse_vec(elem.attrib.get(attr, '0 0 0'), 3)
        elem.attrib[attr] = fmt_vec(transform_local_pos_by_world_map(old_local, old_frame, new_frame, world_xf))
        report[f'transformed_{elem.tag}_{attr}_worldmap'] = report.get(f'transformed_{elem.tag}_{attr}_worldmap', 0) + 1


def transform_body_contents_by_world_map(
    body: ET.Element,
    old_frame: Frame,
    new_frame: Frame,
    world_xf,
    axial: float,
    transverse: float,
    no_inertials: bool,
    report: Dict[str, int],
) -> None:
    """
    Transform sites, joints, geoms, inertials, cameras, and lights exactly once by
    mapping their old world positions through world_xf and writing the result in
    the updated body frame.
    """
    inertial = body.find('inertial')
    if inertial is not None:
        transform_pos_attr_by_world_map(inertial, 'pos', old_frame, new_frame, world_xf, report)
        if not no_inertials:
            scale_inertial(inertial, axial, transverse, report)

    for elem in body.findall('site'):
        transform_pos_attr_by_world_map(elem, 'pos', old_frame, new_frame, world_xf, report)
    for elem in body.findall('joint'):
        transform_pos_attr_by_world_map(elem, 'pos', old_frame, new_frame, world_xf, report)

    # Always write geom pos, even if implicit zero, so mesh vertices and geom frame
    # are displaced consistently with the same world transform.
    for elem in body.findall('geom'):
        transform_pos_attr_by_world_map(elem, 'pos', old_frame, new_frame, world_xf, report, force=True)

    for tag in ('camera', 'light'):
        for elem in body.findall(tag):
            transform_pos_attr_by_world_map(elem, 'pos', old_frame, new_frame, world_xf, report)


def relocate_descendant_body_origins_by_world_map(
    root_body: ET.Element,
    old_frames: Dict[str, Frame],
    frames: Dict[str, Frame],
    world_xf,
    report: Dict[str, int],
) -> None:
    """
    Move every descendant body origin by the same world-space transform, then
    rewrite each child <body pos> in its already-updated parent frame.
    """
    def recurse(parent: ET.Element) -> None:
        parent_name = parent.attrib.get('name')
        if not parent_name or parent_name not in frames:
            return
        parent_frame_new = frames[parent_name]

        for child in parent.findall('body'):
            child_name = child.attrib.get('name')
            if not child_name or child_name not in old_frames:
                continue
            old_child_frame = old_frames[child_name]
            new_child_origin = world_xf(old_child_frame.xpos)
            child.attrib['pos'] = fmt_vec(parent_frame_new.world_to_local(new_child_origin))
            frames[child_name] = Frame(new_child_origin, old_child_frame.xmat.copy())
            report['relocated_body_origin_worldmap'] = report.get('relocated_body_origin_worldmap', 0) + 1
            recurse(child)

    recurse(root_body)


def find_body_containing_site(body: ET.Element, site_name: str) -> Optional[ET.Element]:
    """Return the first body in this subtree that directly contains site_name."""
    for b in iter_subtree(body):
        for site in b.findall('site'):
            if site.attrib.get('name') == site_name:
                return b
    return None


def collect_geom_original_info(root: ET.Element) -> Dict[Tuple[str, int], Tuple[str, np.ndarray, Optional[np.ndarray]]]:
    """
    Store original mesh name, geom pos, and optional geom quat for each geom by
    (body_name, geom_index) before any XML positions are rewritten.
    """
    out: Dict[Tuple[str, int], Tuple[str, np.ndarray, Optional[np.ndarray]]] = {}
    for body in root.findall('.//body'):
        bname = body.attrib.get('name')
        if not bname:
            continue
        for gi, geom in enumerate(body.findall('geom')):
            mesh_name = geom.attrib.get('mesh')
            if not mesh_name:
                continue
            quat = parse_vec(geom.attrib['quat'], 4) if 'quat' in geom.attrib else None
            out[(bname, gi)] = (
                mesh_name,
                parse_vec(geom.attrib.get('pos', '0 0 0'), 3),
                quat,
            )
    return out


def resolve_mesh_file(xml_path: Path, root: ET.Element, mesh: ET.Element) -> Optional[Path]:
    fname = mesh.attrib.get('file')
    if not fname:
        return None
    f = Path(fname)
    if f.is_absolute():
        return f
    meshdir = ''
    compiler = root.find('./compiler')
    if compiler is not None:
        meshdir = compiler.attrib.get('meshdir', '')
    return (xml_path.parent / meshdir / f).resolve()


def read_stl_vertices(path: Path) -> np.ndarray:
    """Read vertices from a binary or ASCII STL file. Used only for mesh-center compensation."""
    data = path.read_bytes()
    if len(data) >= 84:
        ntri = struct.unpack('<I', data[80:84])[0]
        expected = 84 + 50 * ntri
        if expected == len(data):
            verts = []
            off = 84
            for _ in range(ntri):
                # normal: 12 bytes, then 3 vertices, then attr bytes
                off += 12
                for _ in range(3):
                    verts.append(struct.unpack('<fff', data[off:off+12]))
                    off += 12
                off += 2
            return np.asarray(verts, dtype=np.float64)
    # ASCII fallback.
    verts = []
    text = data.decode('utf-8', errors='ignore')
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) == 4 and parts[0].lower() == 'vertex':
            try:
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            except ValueError:
                pass
    if not verts:
        raise ValueError(f'Could not read STL vertices from {path}')
    return np.asarray(verts, dtype=np.float64)


def mesh_bbox_center(mesh_file: Path) -> np.ndarray:
    verts = read_stl_vertices(mesh_file)
    return 0.5 * (np.min(verts, axis=0) + np.max(verts, axis=0))


def compensate_mesh_geom_position_for_center(
    geom: ET.Element,
    mesh_elem: ET.Element,
    old_mesh_name: str,
    old_geom_pos: np.ndarray,
    old_geom_quat: Optional[np.ndarray],
    old_frame: Frame,
    new_frame: Frame,
    world_xf,
    root: ET.Element,
    xml_path: Path,
    mesh_map: Dict[str, ET.Element],
    diag: np.ndarray,
    report: Dict[str, int],
    warnings: list[str],
) -> None:
    """
    Keep the visual mesh placement consistent with the same world-space map used
    for sites/COM/tendon points.

    MuJoCo mesh scale happens about the STL/mesh coordinate origin.  If that
    origin is offset from the anatomical/body frame, scaling the mesh asset can
    make the visible mesh translate even when the body/sites are correct.  This
    adjusts geom pos so the mesh bbox center follows world_xf exactly.
    """
    mesh_file = resolve_mesh_file(xml_path, root, mesh_elem)
    if mesh_file is None or not mesh_file.exists():
        warnings.append(f"could not find mesh file for hand placement compensation: {old_mesh_name!r}")
        return
    try:
        center_file = mesh_bbox_center(mesh_file)
    except Exception as e:
        warnings.append(f"could not read mesh center for hand placement compensation {mesh_file}: {e}")
        return

    old_mesh = mesh_map.get(old_mesh_name)
    if old_mesh is None:
        warnings.append(f"could not find original mesh asset for hand placement compensation: {old_mesh_name!r}")
        return

    old_scale = mesh_scale_vec(old_mesh)
    new_scale = old_scale * diag
    Rg = quat_to_mat(old_geom_quat) if old_geom_quat is not None else np.eye(3)

    old_center_body = old_geom_pos + Rg @ (old_scale * center_file)
    target_center_new_body = new_frame.world_to_local(world_xf(old_frame.local_to_world(old_center_body)))
    corrected_geom_pos = target_center_new_body - Rg @ (new_scale * center_file)
    geom.attrib['pos'] = fmt_vec(corrected_geom_pos)
    report['compensated_hand_mesh_geom_pos'] = report.get('compensated_hand_mesh_geom_pos', 0) + 1


def compensate_hand_mesh_positions(
    hand_body: ET.Element,
    old_frame: Frame,
    new_frame: Frame,
    world_xf,
    root: ET.Element,
    xml_path: Path,
    mesh_map: Dict[str, ET.Element],
    original_geom_info: Dict[Tuple[str, int], Tuple[str, np.ndarray, Optional[np.ndarray]]],
    diag: np.ndarray,
    report: Dict[str, int],
    warnings: list[str],
) -> None:
    """Apply mesh-center compensation only to the body that owns site handm."""
    bname = hand_body.attrib.get('name')
    if not bname:
        return
    for gi, geom in enumerate(hand_body.findall('geom')):
        current_mesh_name = geom.attrib.get('mesh')
        if not current_mesh_name:
            continue
        old_info = original_geom_info.get((bname, gi))
        if old_info is None:
            continue
        old_mesh_name, old_geom_pos, old_geom_quat = old_info
        old_mesh_elem = mesh_map.get(old_mesh_name)
        if old_mesh_elem is None:
            continue
        compensate_mesh_geom_position_for_center(
            geom, old_mesh_elem, old_mesh_name, old_geom_pos, old_geom_quat,
            old_frame, new_frame, world_xf, root, xml_path, mesh_map, diag,
            report, warnings,
        )

def ensure_asset(root: ET.Element) -> ET.Element:
    asset = root.find('./asset')
    if asset is None:
        asset = ET.Element('asset')
        root.insert(0, asset)
    return asset


def mesh_scale_vec(mesh: ET.Element) -> np.ndarray:
    if 'scale' not in mesh.attrib:
        return np.ones(3, dtype=np.float64)
    s = parse_vec(mesh.attrib['scale'])
    if s.size == 1:
        return np.repeat(s, 3)
    if s.size == 3:
        return s
    raise ValueError(f"Unsupported mesh scale on {mesh.attrib.get('name')}: {mesh.attrib.get('scale')}")


def duplicate_and_scale_mesh_assets(root: ET.Element, body: ET.Element, body_name: str, mesh_map: Dict[str, ET.Element],
                                    diag: np.ndarray, suffix: str, report: Dict[str, int], warnings: list[str]) -> None:
    asset = ensure_asset(root)
    for gi, geom in enumerate(body.findall('geom')):
        mesh_name = geom.attrib.get('mesh')
        if not mesh_name:
            continue
        mesh = mesh_map.get(mesh_name)
        if mesh is None:
            raise ValueError(f"Body {body_name!r} references missing mesh asset {mesh_name!r}")
        # This XML's bone geoms are identity-oriented. If someone later adds geom quat, XML mesh scale
        # is in geom/mesh coordinates and will no longer exactly match body-local coordinate scaling.
        if 'quat' in geom.attrib:
            warnings.append(f"geom mesh {mesh_name!r} in body {body_name!r} has quat; mesh scale is XML-diagonal, not full axial transform")
        old_scale = mesh_scale_vec(mesh)
        new_mesh = ET.Element('mesh', dict(mesh.attrib))
        new_name = f"{mesh_name}_{suffix}_{body_name}_{gi}"
        new_mesh.attrib['name'] = new_name
        new_mesh.attrib['scale'] = fmt_vec(old_scale * diag)
        asset.append(new_mesh)
        geom.attrib['mesh'] = new_name
        report['duplicated_scaled_mesh_assets'] = report.get('duplicated_scaled_mesh_assets', 0) + 1


def parse_angles(items: Sequence[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for item in items:
        if '=' not in item:
            raise argparse.ArgumentTypeError('--angle must be joint=value')
        k, v = item.split('=', 1)
        out[k.strip()] = float(v)
    return out


def validate_xml_compile(path: Path) -> None:
    require_mujoco()
    mujoco.MjModel.from_xml_path(str(path))



def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description='Scale MuJoCo arm XML to kinematic shoulder->elbow and elbow->paw lengths.'
    )
    p.add_argument('input_xml', type=Path)
    p.add_argument('kinematics_csv', type=Path)
    p.add_argument('output_xml', type=Path)
    args = p.parse_args(argv)

    # Fixed pipeline choices. No CLI flags.
    statistic = 'median'
    frame = 0
    shape_mode = 'axial'
    transverse_exponent = 0.5
    no_inertials = False
    angles = {
        'elv_angle': 1.27,
        'shoulder_ext': -0.404,
        'shoulder_rot': 0.204,
        'elbow_flex': -0.011,
    }

    kin_upper, kin_lower, nframes = read_kinematic_lengths(args.kinematics_csv, statistic, frame)
    model, data, shoulder_w, elbow_w, hand_w, frames0 = compile_model_points(args.input_xml, angles)
    model_upper = float(np.linalg.norm(elbow_w - shoulder_w))
    model_lower = float(np.linalg.norm(hand_w - elbow_w))
    upper_scale = kin_upper / model_upper
    lower_scale = kin_lower / model_lower
    upper_axial_nominal, upper_trans = shape_scales(upper_scale, shape_mode, transverse_exponent)
    lower_axial_nominal, lower_trans = shape_scales(lower_scale, shape_mode, transverse_exponent)

    # Algebraic first guesses.  These solve the diagonal-axis length equation before
    # MuJoCo recompilation.  The lower chain tends to hit exactly; the upper can move
    # slightly after the ulna child body is re-anchored, so v14 refines upper_axial by
    # compiling candidate XMLs and measuring the actual shoulder->elbow length.
    humerus_frame0 = frames0['humerus']
    upper_delta_l0 = humerus_frame0.world_to_local(elbow_w) - humerus_frame0.world_to_local(shoulder_w)
    upper_axial_initial, upper_axis_i_initial = exact_axial_for_target_length(upper_delta_l0, kin_upper, upper_trans)

    def build_candidate(output_path: Path, upper_axial_override: Optional[float] = None, keep_output: bool = True):
        tree = ET.parse(args.input_xml)
        root = tree.getroot()
        bodies = build_body_map(root)
        meshes = build_mesh_map(root)
        original_geom_info = collect_geom_original_info(root)
        report: Dict[str, int] = {}
        warnings: list[str] = []
        frames = copy_frames(frames0)

        humerus = bodies['humerus']
        ulna = bodies['ulna']
        handm_body = find_body_containing_site(ulna, 'handm')
        humerus_frame = frames['humerus']
        ulna_frame_old = frames['ulna']

        upper_delta_l = humerus_frame.world_to_local(elbow_w) - humerus_frame.world_to_local(shoulder_w)
        solved_upper_axial, upper_axis_i = exact_axial_for_target_length(upper_delta_l, kin_upper, upper_trans)
        upper_axial = float(upper_axial_override) if upper_axial_override is not None else solved_upper_axial
        upper_diag = np.full(3, upper_trans, dtype=np.float64)
        upper_diag[upper_axis_i] = upper_axial
        upper_origin_l = humerus_frame.world_to_local(shoulder_w)
        upper_xf = DiagScaleLocal(upper_origin_l, upper_diag)

        transform_body_contents(humerus, upper_xf, upper_axial, upper_trans, no_inertials, report)
        duplicate_and_scale_mesh_assets(root, humerus, 'humerus', meshes, upper_diag, 'upper', report, warnings)

        # Move ulna body so elbow_flex anchor lands at the transformed elbow.
        elbow_new_w = humerus_frame.local_to_world(upper_xf.apply(humerus_frame.world_to_local(elbow_w)))
        elbow_local_ulna = child_joint_local(ulna, 'elbow_flex')
        if elbow_local_ulna is None:
            raise ValueError('Could not find elbow_flex joint directly inside ulna body')
        ulna_frame_new = set_child_body_pos_to_put_local_point_at_world(
            ulna, elbow_local_ulna, elbow_new_w, humerus_frame, ulna_frame_old, report
        )
        propagate_translation_to_subtree_frames(ulna, frames, ulna_frame_old, ulna_frame_new)

        # Lower chain: use ONE consistent kinematic scaling transform in the ulna frame.
        lower_axis_w = (hand_w - elbow_w) / model_lower
        lower_axis_l_ulna = ulna_frame_new.xmat.T @ lower_axis_w
        lower_delta_l_ulna = lower_axis_l_ulna * model_lower
        lower_axial, lower_axis_i = exact_axial_for_target_length(lower_delta_l_ulna, kin_lower, lower_trans)
        lower_diag_ulna = np.full(3, lower_trans, dtype=np.float64)
        lower_diag_ulna[lower_axis_i] = lower_axial
        lower_anchor_l_ulna = ulna_frame_new.world_to_local(elbow_new_w)
        lower_xf_ulna = DiagScaleLocal(lower_anchor_l_ulna, lower_diag_ulna)

        lower_axis_info = {}

        def lower_world_xf(p_world: np.ndarray) -> np.ndarray:
            return ulna_frame_new.local_to_world(
                lower_xf_ulna.apply(ulna_frame_new.world_to_local(p_world))
            )

        lower_pre_scale_frames = copy_frames(frames)

        relocate_descendant_body_origins_by_world_map(
            ulna, lower_pre_scale_frames, frames, lower_world_xf, report
        )

        for b in iter_subtree(ulna):
            bname = b.attrib.get('name')
            if not bname:
                continue

            old_frame_b = lower_pre_scale_frames[bname]
            new_frame_b = frames[bname]

            lower_axis_l = new_frame_b.xmat.T @ lower_axis_w
            diag, axis_i = dominant_axis_diag(lower_axis_l, lower_axial, lower_trans)

            lower_axis_info[bname] = (axis_i, diag.copy())
            transform_body_contents_by_world_map(
                b, old_frame_b, new_frame_b, lower_world_xf,
                lower_axial, lower_trans, no_inertials, report
            )
            if handm_body is not None and b is handm_body:
                compensate_hand_mesh_positions(
                    b, old_frame_b, new_frame_b, lower_world_xf, root, args.input_xml,
                    meshes, original_geom_info, diag, report, warnings
                )
            duplicate_and_scale_mesh_assets(root, b, bname, meshes, diag, 'lower', report, warnings)

        ET.indent(tree, space='  ')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        validate_xml_compile(output_path)

        _, _, shoulder2, elbow2, hand2, _ = compile_model_points(output_path, angles)
        out_upper = float(np.linalg.norm(elbow2 - shoulder2))
        out_lower = float(np.linalg.norm(hand2 - elbow2))

        if not keep_output:
            try:
                output_path.unlink()
            except OSError:
                pass

        return {
            'upper_axial': upper_axial,
            'upper_axis_i': upper_axis_i,
            'lower_axial': lower_axial,
            'lower_axis_i': lower_axis_i,
            'lower_axis_info': lower_axis_info,
            'shoulder': shoulder2,
            'elbow': elbow2,
            'hand': hand2,
            'out_upper': out_upper,
            'out_lower': out_lower,
            'report': report,
            'warnings': warnings,
        }

    # Measure function for compiled upper length.  Candidate XMLs are written next
    # to the final output so relative mesh paths resolve the same way as the final XML.
    tmp_paths: list[Path] = []

    def eval_upper(axial: float, idx: int):
        tmp_path = args.output_xml.parent / f'.{args.output_xml.stem}.upper_iter_{idx}.xml'
        tmp_paths.append(tmp_path)
        result = build_candidate(tmp_path, axial, keep_output=False)
        return result['out_upper'] - kin_upper, result

    # Bracket the compiled solution around the algebraic v13 solution.
    f0, r0 = eval_upper(upper_axial_initial, 0)
    best_axial = upper_axial_initial
    best_result = r0
    best_abs = abs(f0)

    if abs(f0) > 1e-12:
        if f0 > 0:
            hi_a, hi_f = upper_axial_initial, f0
            lo_a = upper_axial_initial * 0.95
            lo_f, lo_r = eval_upper(lo_a, 1)
            k = 2
            while lo_f > 0 and k < 12:
                hi_a, hi_f = lo_a, lo_f
                lo_a *= 0.90
                lo_f, lo_r = eval_upper(lo_a, k)
                k += 1
        else:
            lo_a, lo_f = upper_axial_initial, f0
            hi_a = upper_axial_initial * 1.05
            hi_f, hi_r = eval_upper(hi_a, 1)
            k = 2
            while hi_f < 0 and k < 12:
                lo_a, lo_f = hi_a, hi_f
                hi_a *= 1.10
                hi_f, hi_r = eval_upper(hi_a, k)
                k += 1

        # If a bracket was found, do bisection against the compiled MuJoCo length.
        if lo_f <= 0 <= hi_f:
            for i in range(12):
                mid_a = 0.5 * (lo_a + hi_a)
                mid_f, mid_r = eval_upper(mid_a, 100 + i)
                if abs(mid_f) < best_abs:
                    best_abs = abs(mid_f)
                    best_axial = mid_a
                    best_result = mid_r
                if mid_f < 0:
                    lo_a, lo_f = mid_a, mid_f
                else:
                    hi_a, hi_f = mid_a, mid_f
                if abs(mid_f) <= 1e-12:
                    break
        else:
            # No clean bracket; keep the best measured candidate seen so far.
            for a, f, r in [(upper_axial_initial, f0, r0)]:
                if abs(f) < best_abs:
                    best_abs = abs(f)
                    best_axial = a
                    best_result = r
    
    # Build the final output using the best compiled upper axial scale.
    final = build_candidate(args.output_xml, best_axial, keep_output=True)

    # Clean up any temporary candidate files that might remain.
    for tp in tmp_paths:
        try:
            tp.unlink()
        except OSError:
            pass

    shoulder2 = final['shoulder']
    elbow2 = final['elbow']
    hand2 = final['hand']
    upper_axis_i = final['upper_axis_i']
    lower_axis_info = final['lower_axis_info']
    lower_axial = final['lower_axial']
    warnings = final['warnings']

    print(f'  output upper length: {final["out_upper"]:.12g}')
    print(f'  target upper length: {kin_upper:.12g}')
    print(f'  output lower length: {final["out_lower"]:.12g}')
    print(f'  target lower length: {kin_lower:.12g}')
    print(f'  output shoulder: {fmt_vec(shoulder2)}')
    print(f'  output elbow:    {fmt_vec(elbow2)}')
    print(f'  output hand:     {fmt_vec(hand2)}')

    print(f'Wrote: {args.output_xml}')
    print('No unit conversion. Median kinematic bone lengths. Axial XML mesh scaling.')
    print(f'  frames used: {nframes}')
    print(f'  nominal upper scale: {upper_scale:.12g}')
    print(f'  solved upper axial scale: {upper_axial_initial:.12g}')
    print(f'  refined compiled upper axial scale: {final["upper_axial"]:.12g}')
    print(f'  nominal lower scale: {lower_scale:.12g}')
    print(f'  solved lower axial scale: {lower_axial:.12g}')
    print(f'  humerus XML axis: {"xyz"[upper_axis_i]}')
    for bname, (axis_i, diag) in lower_axis_info.items():
        print(f'  {bname} XML axis: {"xyz"[axis_i]}, mesh/site scale={fmt_vec(diag)}')
    if warnings:
        print('Warnings:', file=sys.stderr)
        for w in warnings:
            print(f'  - {w}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
