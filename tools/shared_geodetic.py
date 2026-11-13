#!/usr/bin/env python3
"""WGS84, ECEF, ENU, and scene-grid coordinate conversions.

The Tokyo scene's established horizontal contract is an EPSG:32654 UTM grid:
scene X adds to UTM easting and scene Y adds to UTM northing.  The ECEF/ENU
functions are provided for backend-neutral interoperability; GPX generation
uses the established scene-grid contract so it remains consistent with the A
pipeline's device geolocation export.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


CONVERTER_VERSION = "shared-geodetic-1.0.0"
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
UTM_K0 = 0.9996


@dataclass(frozen=True)
class SceneGeoreference:
    anchor_id: str
    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    utm_zone: int
    northern_hemisphere: bool
    anchor_easting_m: float
    anchor_northing_m: float


def _finite(*values: float) -> None:
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("coordinates must be finite")


def geodetic_to_ecef(latitude_deg: float, longitude_deg: float, altitude_m: float) -> tuple[float, float, float]:
    _finite(latitude_deg, longitude_deg, altitude_m)
    latitude = math.radians(latitude_deg)
    longitude = math.radians(longitude_deg)
    sin_lat = math.sin(latitude)
    cos_lat = math.cos(latitude)
    radius = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (radius + altitude_m) * cos_lat * math.cos(longitude)
    y = (radius + altitude_m) * cos_lat * math.sin(longitude)
    z = (radius * (1.0 - WGS84_E2) + altitude_m) * sin_lat
    return x, y, z


def ecef_to_geodetic(x_m: float, y_m: float, z_m: float) -> tuple[float, float, float]:
    _finite(x_m, y_m, z_m)
    longitude = math.atan2(y_m, x_m)
    horizontal = math.hypot(x_m, y_m)
    if horizontal < 1e-9:
        latitude = math.copysign(math.pi / 2.0, z_m)
        semi_minor = WGS84_A * (1.0 - WGS84_F)
        return math.degrees(latitude), math.degrees(longitude), abs(z_m) - semi_minor
    latitude = math.atan2(z_m, horizontal * (1.0 - WGS84_E2))
    altitude = 0.0
    for _ in range(12):
        sin_lat = math.sin(latitude)
        radius = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        altitude = horizontal / max(math.cos(latitude), 1e-15) - radius
        next_latitude = math.atan2(z_m, horizontal * (1.0 - WGS84_E2 * radius / (radius + altitude)))
        if abs(next_latitude - latitude) < 1e-14:
            latitude = next_latitude
            break
        latitude = next_latitude
    return math.degrees(latitude), math.degrees(longitude), altitude


def enu_to_ecef(
    east_m: float,
    north_m: float,
    up_m: float,
    anchor_latitude_deg: float,
    anchor_longitude_deg: float,
    anchor_altitude_m: float,
) -> tuple[float, float, float]:
    _finite(east_m, north_m, up_m)
    latitude = math.radians(anchor_latitude_deg)
    longitude = math.radians(anchor_longitude_deg)
    sin_lat, cos_lat = math.sin(latitude), math.cos(latitude)
    sin_lon, cos_lon = math.sin(longitude), math.cos(longitude)
    anchor_x, anchor_y, anchor_z = geodetic_to_ecef(anchor_latitude_deg, anchor_longitude_deg, anchor_altitude_m)
    dx = -sin_lon * east_m - sin_lat * cos_lon * north_m + cos_lat * cos_lon * up_m
    dy = cos_lon * east_m - sin_lat * sin_lon * north_m + cos_lat * sin_lon * up_m
    dz = cos_lat * north_m + sin_lat * up_m
    return anchor_x + dx, anchor_y + dy, anchor_z + dz


def ecef_to_enu(
    x_m: float,
    y_m: float,
    z_m: float,
    anchor_latitude_deg: float,
    anchor_longitude_deg: float,
    anchor_altitude_m: float,
) -> tuple[float, float, float]:
    latitude = math.radians(anchor_latitude_deg)
    longitude = math.radians(anchor_longitude_deg)
    sin_lat, cos_lat = math.sin(latitude), math.cos(latitude)
    sin_lon, cos_lon = math.sin(longitude), math.cos(longitude)
    anchor_x, anchor_y, anchor_z = geodetic_to_ecef(anchor_latitude_deg, anchor_longitude_deg, anchor_altitude_m)
    dx, dy, dz = x_m - anchor_x, y_m - anchor_y, z_m - anchor_z
    east = -sin_lon * dx + cos_lon * dy
    north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    up = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
    return east, north, up


def enu_to_wgs84(
    east_m: float,
    north_m: float,
    up_m: float,
    anchor_latitude_deg: float,
    anchor_longitude_deg: float,
    anchor_altitude_m: float,
) -> tuple[float, float, float]:
    return ecef_to_geodetic(*enu_to_ecef(
        east_m, north_m, up_m,
        anchor_latitude_deg, anchor_longitude_deg, anchor_altitude_m,
    ))


def wgs84_to_enu(
    latitude_deg: float,
    longitude_deg: float,
    altitude_m: float,
    anchor_latitude_deg: float,
    anchor_longitude_deg: float,
    anchor_altitude_m: float,
) -> tuple[float, float, float]:
    return ecef_to_enu(
        *geodetic_to_ecef(latitude_deg, longitude_deg, altitude_m),
        anchor_latitude_deg, anchor_longitude_deg, anchor_altitude_m,
    )


def epsg_to_utm(crs: str) -> tuple[int, bool]:
    try:
        epsg = int(str(crs).split(":", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"unsupported scene CRS: {crs!r}") from exc
    if 32601 <= epsg <= 32660:
        return epsg - 32600, True
    if 32701 <= epsg <= 32760:
        return epsg - 32700, False
    raise ValueError(f"scene CRS must be WGS84 UTM, got {crs!r}")


def wgs84_to_utm(latitude_deg: float, longitude_deg: float, zone: int, northern: bool = True) -> tuple[float, float]:
    _finite(latitude_deg, longitude_deg)
    if not 1 <= zone <= 60:
        raise ValueError(f"UTM zone must be in 1..60, got {zone}")
    latitude = math.radians(latitude_deg)
    longitude = math.radians(longitude_deg)
    central_meridian = math.radians((zone - 1) * 6 - 180 + 3)
    eccentricity_prime_sq = WGS84_E2 / (1.0 - WGS84_E2)
    sin_lat, cos_lat = math.sin(latitude), math.cos(latitude)
    tan_lat = math.tan(latitude)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    t = tan_lat * tan_lat
    c = eccentricity_prime_sq * cos_lat * cos_lat
    a = cos_lat * (longitude - central_meridian)
    m = WGS84_A * (
        (1.0 - WGS84_E2 / 4.0 - 3.0 * WGS84_E2**2 / 64.0 - 5.0 * WGS84_E2**3 / 256.0) * latitude
        - (3.0 * WGS84_E2 / 8.0 + 3.0 * WGS84_E2**2 / 32.0 + 45.0 * WGS84_E2**3 / 1024.0) * math.sin(2.0 * latitude)
        + (15.0 * WGS84_E2**2 / 256.0 + 45.0 * WGS84_E2**3 / 1024.0) * math.sin(4.0 * latitude)
        - (35.0 * WGS84_E2**3 / 3072.0) * math.sin(6.0 * latitude)
    )
    easting = UTM_K0 * n * (
        a + (1.0 - t + c) * a**3 / 6.0
        + (5.0 - 18.0 * t + t**2 + 72.0 * c - 58.0 * eccentricity_prime_sq) * a**5 / 120.0
    ) + 500000.0
    northing = UTM_K0 * (
        m + n * tan_lat * (
            a**2 / 2.0
            + (5.0 - t + 9.0 * c + 4.0 * c**2) * a**4 / 24.0
            + (61.0 - 58.0 * t + t**2 + 600.0 * c - 330.0 * eccentricity_prime_sq) * a**6 / 720.0
        )
    )
    if not northern:
        northing += 10000000.0
    return easting, northing


def utm_to_wgs84(easting_m: float, northing_m: float, zone: int, northern: bool = True) -> tuple[float, float]:
    _finite(easting_m, northing_m)
    if not 1 <= zone <= 60:
        raise ValueError(f"UTM zone must be in 1..60, got {zone}")
    x = float(easting_m) - 500000.0
    y = float(northing_m) - (0.0 if northern else 10000000.0)
    eccentricity_prime_sq = WGS84_E2 / (1.0 - WGS84_E2)
    meridional_arc = y / UTM_K0
    mu = meridional_arc / (WGS84_A * (1.0 - WGS84_E2 / 4.0 - 3.0 * WGS84_E2**2 / 64.0 - 5.0 * WGS84_E2**3 / 256.0))
    e1 = (1.0 - math.sqrt(1.0 - WGS84_E2)) / (1.0 + math.sqrt(1.0 - WGS84_E2))
    phi1 = (
        mu + (3.0 * e1 / 2.0 - 27.0 * e1**3 / 32.0) * math.sin(2.0 * mu)
        + (21.0 * e1**2 / 16.0 - 55.0 * e1**4 / 32.0) * math.sin(4.0 * mu)
        + (151.0 * e1**3 / 96.0) * math.sin(6.0 * mu)
        + (1097.0 * e1**4 / 512.0) * math.sin(8.0 * mu)
    )
    sin_phi, cos_phi, tan_phi = math.sin(phi1), math.cos(phi1), math.tan(phi1)
    n1 = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_phi**2)
    r1 = WGS84_A * (1.0 - WGS84_E2) / (1.0 - WGS84_E2 * sin_phi**2) ** 1.5
    t1 = tan_phi**2
    c1 = eccentricity_prime_sq * cos_phi**2
    d = x / (n1 * UTM_K0)
    latitude = phi1 - (n1 * tan_phi / r1) * (
        d**2 / 2.0
        - (5.0 + 3.0 * t1 + 10.0 * c1 - 4.0 * c1**2 - 9.0 * eccentricity_prime_sq) * d**4 / 24.0
        + (61.0 + 90.0 * t1 + 298.0 * c1 + 45.0 * t1**2 - 252.0 * eccentricity_prime_sq - 3.0 * c1**2) * d**6 / 720.0
    )
    central_meridian = math.radians((zone - 1) * 6 - 180 + 3)
    longitude = central_meridian + (
        d - (1.0 + 2.0 * t1 + c1) * d**3 / 6.0
        + (5.0 - 2.0 * c1 + 28.0 * t1 - 3.0 * c1**2 + 8.0 * eccentricity_prime_sq + 24.0 * t1**2) * d**5 / 120.0
    ) / cos_phi
    return math.degrees(latitude), math.degrees(longitude)


def georeference_from_metadata(metadata: dict[str, Any], anchor_id: str) -> SceneGeoreference:
    source = metadata.get("geodetic_anchor")
    if not isinstance(source, dict):
        raise ValueError("missing geodetic_anchor metadata")
    required = ("crs", "center_lat", "center_lon", "center_z", "anchor_x", "anchor_y", "vertical_datum")
    missing = [key for key in required if key not in source]
    if missing:
        raise ValueError(f"geodetic anchor is missing: {', '.join(missing)}")
    if source["vertical_datum"] != "ellipsoid":
        raise ValueError("geodetic anchor altitude must use the ellipsoid vertical datum")
    zone, northern = epsg_to_utm(str(source["crs"]))
    georef = SceneGeoreference(
        anchor_id=anchor_id,
        latitude_deg=float(source["center_lat"]),
        longitude_deg=float(source["center_lon"]),
        altitude_m=float(source["center_z"]),
        utm_zone=zone,
        northern_hemisphere=northern,
        anchor_easting_m=float(source["anchor_x"]),
        anchor_northing_m=float(source["anchor_y"]),
    )
    computed_lat, computed_lon = utm_to_wgs84(georef.anchor_easting_m, georef.anchor_northing_m, zone, northern)
    if max(abs(computed_lat - georef.latitude_deg), abs(computed_lon - georef.longitude_deg)) > 1e-6:
        raise ValueError("UTM anchor does not reproduce the stored WGS84 scene center")
    return georef


def scene_enu_to_wgs84(east_m: float, north_m: float, up_m: float, georef: SceneGeoreference) -> tuple[float, float, float]:
    latitude, longitude = utm_to_wgs84(
        georef.anchor_easting_m + east_m,
        georef.anchor_northing_m + north_m,
        georef.utm_zone,
        georef.northern_hemisphere,
    )
    return latitude, longitude, georef.altitude_m + up_m


def wgs84_to_scene_enu(latitude_deg: float, longitude_deg: float, altitude_m: float, georef: SceneGeoreference) -> tuple[float, float, float]:
    easting, northing = wgs84_to_utm(
        latitude_deg, longitude_deg, georef.utm_zone, georef.northern_hemisphere,
    )
    return (
        easting - georef.anchor_easting_m,
        northing - georef.anchor_northing_m,
        altitude_m - georef.altitude_m,
    )
