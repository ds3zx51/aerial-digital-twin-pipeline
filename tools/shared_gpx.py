#!/usr/bin/env python3
"""Backend-neutral GPX 1.1 writer and strict parser."""
from __future__ import annotations

from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any, Iterable
import xml.etree.ElementTree as ET


GPX_VERSION = "1.1"
GPX_NAMESPACE = "http://www.topografix.com/GPX/1/1"
GPX_SCHEMA_LOCATION = "http://www.topografix.com/GPX/1/1 http://www.topografix.com/GPX/1/1/gpx.xsd"
XSI_NAMESPACE = "http://www.w3.org/2001/XMLSchema-instance"


def format_gpx_time(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("GPX timestamps must be timezone-aware")
    value = value.astimezone(timezone.utc)
    return value.strftime("%Y-%m-%dT%H:%M:%S.") + f"{value.microsecond // 1000:03d}Z"


def parse_gpx_time(value: str) -> datetime:
    if not value or not value.endswith("Z"):
        raise ValueError("GPX timestamp must be an explicit UTC value")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"invalid GPX timestamp: {value!r}") from exc
    return parsed.astimezone(timezone.utc)


def write_gpx(path: Path, trajectory_id: str, ue_id: str, points: Iterable[dict[str, Any]]) -> None:
    ET.register_namespace("", GPX_NAMESPACE)
    ET.register_namespace("xsi", XSI_NAMESPACE)
    q = lambda name: f"{{{GPX_NAMESPACE}}}{name}"
    root = ET.Element(q("gpx"), {
        "version": GPX_VERSION,
        "creator": "aodt-experiment-pipeline",
        f"{{{XSI_NAMESPACE}}}schemaLocation": GPX_SCHEMA_LOCATION,
    })
    metadata = ET.SubElement(root, q("metadata"))
    ET.SubElement(metadata, q("name")).text = trajectory_id
    ET.SubElement(metadata, q("desc")).text = f"Shared deterministic trajectory for {ue_id}"
    track = ET.SubElement(root, q("trk"))
    ET.SubElement(track, q("name")).text = f"{trajectory_id}:{ue_id}"
    segment = ET.SubElement(track, q("trkseg"))
    count = 0
    for point in points:
        latitude = float(point["latitude_deg"])
        longitude = float(point["longitude_deg"])
        elevation = float(point["gpx_ele_m"])
        if not all(math.isfinite(value) for value in (latitude, longitude, elevation)):
            raise ValueError("GPX coordinates must be finite")
        trkpt = ET.SubElement(segment, q("trkpt"), {"lat": f"{latitude:.10f}", "lon": f"{longitude:.10f}"})
        ET.SubElement(trkpt, q("ele")).text = f"{elevation:.6f}"
        ET.SubElement(trkpt, q("time")).text = str(point["timestamp"])
        count += 1
    if count == 0:
        raise ValueError("cannot write an empty GPX track")
    ET.indent(root, space="  ")
    path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root)
    tree.write(path, encoding="utf-8", xml_declaration=True, short_empty_elements=False)
    with path.open("ab") as handle:
        handle.write(b"\n")


def read_gpx(path: Path) -> dict[str, Any]:
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise ValueError(f"invalid GPX XML: {path}") from exc
    if root.tag != f"{{{GPX_NAMESPACE}}}gpx" or root.attrib.get("version") != GPX_VERSION:
        raise ValueError("GPX must use the GPX 1.1 namespace and version")
    q = lambda name: f"{{{GPX_NAMESPACE}}}{name}"
    tracks = root.findall(q("trk"))
    if len(tracks) != 1:
        raise ValueError("GPX must contain exactly one track")
    segments = tracks[0].findall(q("trkseg"))
    if len(segments) != 1:
        raise ValueError("GPX track must contain exactly one segment")
    metadata_name = root.findtext(f"{q('metadata')}/{q('name')}")
    points: list[dict[str, Any]] = []
    for index, node in enumerate(segments[0].findall(q("trkpt"))):
        time_text = node.findtext(q("time"))
        elevation_text = node.findtext(q("ele"))
        if time_text is None:
            raise ValueError(f"GPX point {index} is missing timestamp")
        if elevation_text is None:
            raise ValueError(f"GPX point {index} is missing elevation")
        try:
            latitude = float(node.attrib["lat"])
            longitude = float(node.attrib["lon"])
            elevation = float(elevation_text)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"GPX point {index} has invalid coordinates") from exc
        if not all(math.isfinite(value) for value in (latitude, longitude, elevation)):
            raise ValueError(f"GPX point {index} has non-finite coordinates")
        points.append({
            "sample_id": index,
            "latitude_deg": latitude,
            "longitude_deg": longitude,
            "gpx_ele_m": elevation,
            "timestamp": time_text,
            "datetime": parse_gpx_time(time_text),
        })
    return {
        "version": root.attrib["version"],
        "creator": root.attrib.get("creator"),
        "trajectory_id": metadata_name,
        "points": points,
    }

