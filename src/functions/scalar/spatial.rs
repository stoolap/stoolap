// Copyright 2025 Stoolap Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Geospatial (GIS) scalar functions implementing OGC SQL specifications.
//!
//! Provides coordinate extraction, planar and spherical distances (Haversine formula),
//! point-in-polygon containment testing (ray casting), polygon area (Shoelace formula),
//! centroid calculation, and format conversion between GeoJSON and WKT.

use crate::common::SmartString;
use crate::core::{Error, Result, Value};
use crate::functions::{
    FunctionDataType, FunctionInfo, FunctionSignature, FunctionType, ScalarFunction,
};
use crate::validate_arg_count;

// ============================================================================
// Coordinate & Geometry Parsing Helpers
// ============================================================================

/// Earth mean radius in meters for spherical geodesy (WGS 84 mean radius)
const EARTH_RADIUS_METERS: f64 = 6_371_008.8;

/// Parse point coordinates (x, y) from a Value (supports GeoJSON, WKT POINT, or JSON Array).
pub fn parse_point_coords(v: &Value, fn_name: &str) -> Result<(f64, f64)> {
    match v {
        Value::Text(s) => parse_point_str(s.as_str(), fn_name),
        Value::Extension(data) => {
            // Check if it's a JSON extension value
            let s = std::str::from_utf8(&data[1..]).map_err(|e| {
                Error::invalid_argument(format!(
                    "{}: invalid UTF-8 geometry payload: {}",
                    fn_name, e
                ))
            })?;
            parse_point_str(s, fn_name)
        }
        _ => Err(Error::invalid_argument(format!(
            "{}: expected geometry string (WKT or GeoJSON), got {:?}",
            fn_name, v
        ))),
    }
}

/// Parse point from string (GeoJSON or WKT)
fn parse_point_str(s: &str, fn_name: &str) -> Result<(f64, f64)> {
    let trimmed = s.trim();

    // 1. Try WKT: POINT(x y) or POINT (x y) or POINT EMPTY
    if trimmed.to_ascii_uppercase().starts_with("POINT") {
        let after = trimmed[5..].trim_start();
        if after.starts_with('(') && after.ends_with(')') {
            let inner = &after[1..after.len() - 1].trim();
            let mut parts = inner.split_whitespace();
            let x_str = parts.next().ok_or_else(|| {
                Error::invalid_argument(format!("{}: empty coordinates in WKT: {}", fn_name, s))
            })?;
            let y_str = parts.next().ok_or_else(|| {
                Error::invalid_argument(format!("{}: missing Y coordinate in WKT: {}", fn_name, s))
            })?;
            let x: f64 = x_str.parse().map_err(|_| {
                Error::invalid_argument(format!("{}: invalid X float in WKT: {}", fn_name, x_str))
            })?;
            let y: f64 = y_str.parse().map_err(|_| {
                Error::invalid_argument(format!("{}: invalid Y float in WKT: {}", fn_name, y_str))
            })?;
            return Ok((x, y));
        }
    }

    // 2. Try JSON / GeoJSON
    if trimmed.starts_with('{') || trimmed.starts_with('[') {
        let json: serde_json::Value = serde_json::from_str(trimmed).map_err(|e| {
            Error::invalid_argument(format!("{}: invalid JSON/GeoJSON geometry: {}", fn_name, e))
        })?;

        match json {
            serde_json::Value::Object(map) => {
                if let Some(coords) = map.get("coordinates") {
                    if let Some(arr) = coords.as_array() {
                        if arr.len() >= 2 {
                            let x = arr[0].as_f64().ok_or_else(|| {
                                Error::invalid_argument(format!(
                                    "{}: X coordinate is not a number",
                                    fn_name
                                ))
                            })?;
                            let y = arr[1].as_f64().ok_or_else(|| {
                                Error::invalid_argument(format!(
                                    "{}: Y coordinate is not a number",
                                    fn_name
                                ))
                            })?;
                            return Ok((x, y));
                        }
                    }
                }
            }
            serde_json::Value::Array(arr) if arr.len() >= 2 => {
                let x = arr[0].as_f64().ok_or_else(|| {
                    Error::invalid_argument(format!(
                        "{}: X coordinate is not a number",
                        fn_name
                    ))
                })?;
                let y = arr[1].as_f64().ok_or_else(|| {
                    Error::invalid_argument(format!(
                        "{}: Y coordinate is not a number",
                        fn_name
                    ))
                })?;
                return Ok((x, y));
            }
            _ => {}
        }
    }

    Err(Error::invalid_argument(format!(
        "{}: could not parse Point geometry from '{}'",
        fn_name, s
    )))
}

/// Parse polygon exterior ring coordinates from a Value (supports GeoJSON or WKT POLYGON).
pub fn parse_polygon_coords(v: &Value, fn_name: &str) -> Result<Vec<(f64, f64)>> {
    let s = match v {
        Value::Text(s) => s.as_str(),
        _ => {
            return Err(Error::invalid_argument(format!(
                "{}: expected polygon geometry string, got {:?}",
                fn_name, v
            )))
        }
    };

    let trimmed = s.trim();

    // 1. Try WKT: POLYGON(((x1 y1, x2 y2, ...))) or POLYGON ((x1 y1, x2 y2, ...))
    if trimmed.to_ascii_uppercase().starts_with("POLYGON") {
        let after = trimmed[7..].trim_start();
        // Remove outer brackets
        let content = if after.starts_with("((") && after.ends_with("))") {
            &after[2..after.len() - 2]
        } else if after.starts_with('(') && after.ends_with(')') {
            let inner = after[1..after.len() - 1].trim();
            if inner.starts_with('(') && inner.ends_with(')') {
                &inner[1..inner.len() - 1]
            } else {
                inner
            }
        } else {
            return Err(Error::invalid_argument(format!(
                "{}: invalid WKT POLYGON syntax: {}",
                fn_name, s
            )));
        };

        // Exterior ring is the first ring before any hole
        let first_ring = content.split(')').next().unwrap_or(content);
        let first_ring_clean = first_ring.trim_start_matches('(').trim();

        let mut points = Vec::new();
        for pt_str in first_ring_clean.split(',') {
            let mut parts = pt_str.split_whitespace();
            if let (Some(xs), Some(ys)) = (parts.next(), parts.next()) {
                let x: f64 = xs.parse().map_err(|_| {
                    Error::invalid_argument(format!("{}: invalid float in polygon: {}", fn_name, xs))
                })?;
                let y: f64 = ys.parse().map_err(|_| {
                    Error::invalid_argument(format!("{}: invalid float in polygon: {}", fn_name, ys))
                })?;
                points.push((x, y));
            }
        }
        if points.len() < 3 {
            return Err(Error::invalid_argument(format!(
                "{}: polygon ring must have at least 3 points, got {}",
                fn_name,
                points.len()
            )));
        }
        return Ok(points);
    }

    // 2. Try GeoJSON
    if trimmed.starts_with('{') {
        let json: serde_json::Value = serde_json::from_str(trimmed).map_err(|e| {
            Error::invalid_argument(format!("{}: invalid GeoJSON polygon: {}", fn_name, e))
        })?;

        if let Some(coords) = json.get("coordinates").and_then(|c| c.as_array()) {
            if let Some(first_ring) = coords.first().and_then(|r| r.as_array()) {
                let mut points = Vec::with_capacity(first_ring.len());
                for pt in first_ring {
                    if let Some(arr) = pt.as_array() {
                        if arr.len() >= 2 {
                            let x = arr[0].as_f64().ok_or_else(|| {
                                Error::invalid_argument(format!(
                                    "{}: invalid polygon coordinate",
                                    fn_name
                                ))
                            })?;
                            let y = arr[1].as_f64().ok_or_else(|| {
                                Error::invalid_argument(format!(
                                    "{}: invalid polygon coordinate",
                                    fn_name
                                ))
                            })?;
                            points.push((x, y));
                        }
                    }
                }
                if points.len() < 3 {
                    return Err(Error::invalid_argument(format!(
                        "{}: polygon ring must have at least 3 points, got {}",
                        fn_name,
                        points.len()
                    )));
                }
                return Ok(points);
            }
        }
    }

    Err(Error::invalid_argument(format!(
        "{}: could not parse Polygon from '{}'",
        fn_name, s
    )))
}

/// Point-in-polygon containment test using the ray-casting algorithm.
pub fn point_in_polygon(px: f64, py: f64, ring: &[(f64, f64)]) -> bool {
    let mut inside = false;
    let n = ring.len();
    if n < 3 {
        return false;
    }
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = ring[i];
        let (xj, yj) = ring[j];

        // Check if horizontal ray from (px, py) intersects line segment (xi, yi) -> (xj, yj)
        let intersect = ((yi > py) != (yj > py))
            && (px < (xj - xi) * (py - yi) / (yj - yi + f64::EPSILON) + xi);
        if intersect {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Compute 2D polygon area using Gauss's area formula (Shoelace formula).
pub fn polygon_area(ring: &[(f64, f64)]) -> f64 {
    let n = ring.len();
    if n < 3 {
        return 0.0;
    }
    let mut sum = 0.0f64;
    for i in 0..n {
        let j = (i + 1) % n;
        sum += ring[i].0 * ring[j].1 - ring[j].0 * ring[i].1;
    }
    (sum / 2.0).abs()
}

// ============================================================================
// ST_Point / ST_MakePoint
// ============================================================================

/// ST_POINT(x, y) — constructs a 2D Point GeoJSON geometry string
#[derive(Default)]
pub struct StPointFunction;

impl ScalarFunction for StPointFunction {
    fn name(&self) -> &str {
        "ST_POINT"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "ST_POINT",
            FunctionType::Scalar,
            "Constructs a 2D Point geometry GeoJSON string from X and Y coordinates",
            FunctionSignature::new(
                FunctionDataType::String,
                vec![FunctionDataType::Float, FunctionDataType::Float],
                2,
                2,
            ),
        )
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(StPointFunction)
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "ST_POINT", 2);
        if args[0].is_null() || args[1].is_null() {
            return Ok(Value::null_unknown());
        }

        let x = match &args[0] {
            Value::Float(f) => *f,
            Value::Integer(i) => *i as f64,
            Value::Text(s) => s.parse::<f64>().map_err(|_| {
                Error::invalid_argument(format!("ST_POINT invalid X coordinate: {}", s))
            })?,
            _ => {
                return Err(Error::invalid_argument(
                    "ST_POINT X coordinate must be a number",
                ))
            }
        };

        let y = match &args[1] {
            Value::Float(f) => *f,
            Value::Integer(i) => *i as f64,
            Value::Text(s) => s.parse::<f64>().map_err(|_| {
                Error::invalid_argument(format!("ST_POINT invalid Y coordinate: {}", s))
            })?,
            _ => {
                return Err(Error::invalid_argument(
                    "ST_POINT Y coordinate must be a number",
                ))
            }
        };

        let geojson = format!(r#"{{"type":"Point","coordinates":[{},{}]}}"#, x, y);
        Ok(Value::Text(SmartString::from_string(geojson)))
    }
}

/// ST_MAKEPOINT(x, y) — alias for ST_POINT
#[derive(Default)]
pub struct StMakePointFunction;

impl ScalarFunction for StMakePointFunction {
    fn name(&self) -> &str {
        "ST_MAKEPOINT"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "ST_MAKEPOINT",
            FunctionType::Scalar,
            "Alias for ST_POINT - constructs a 2D Point geometry",
            FunctionSignature::new(
                FunctionDataType::String,
                vec![FunctionDataType::Float, FunctionDataType::Float],
                2,
                2,
            ),
        )
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(StMakePointFunction)
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        StPointFunction.evaluate(args)
    }
}

// ============================================================================
// ST_X / ST_Y
// ============================================================================

/// ST_X(geom) — extracts the X coordinate (e.g. longitude / easting) from a Point geometry
#[derive(Default)]
pub struct StXFunction;

impl ScalarFunction for StXFunction {
    fn name(&self) -> &str {
        "ST_X"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "ST_X",
            FunctionType::Scalar,
            "Returns the X coordinate of a Point geometry",
            FunctionSignature::new(
                FunctionDataType::Float,
                vec![FunctionDataType::String],
                1,
                1,
            ),
        )
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(StXFunction)
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "ST_X", 1);
        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }
        let (x, _) = parse_point_coords(&args[0], "ST_X")?;
        Ok(Value::Float(x))
    }
}

/// ST_Y(geom) — extracts the Y coordinate (e.g. latitude / northing) from a Point geometry
#[derive(Default)]
pub struct StYFunction;

impl ScalarFunction for StYFunction {
    fn name(&self) -> &str {
        "ST_Y"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "ST_Y",
            FunctionType::Scalar,
            "Returns the Y coordinate of a Point geometry",
            FunctionSignature::new(
                FunctionDataType::Float,
                vec![FunctionDataType::String],
                1,
                1,
            ),
        )
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(StYFunction)
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "ST_Y", 1);
        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }
        let (_, y) = parse_point_coords(&args[0], "ST_Y")?;
        Ok(Value::Float(y))
    }
}

// ============================================================================
// ST_Distance / ST_Distance_Sphere
// ============================================================================

/// ST_DISTANCE(p1, p2) — calculates Euclidean planar distance between two Point geometries
#[derive(Default)]
pub struct StDistanceFunction;

impl ScalarFunction for StDistanceFunction {
    fn name(&self) -> &str {
        "ST_DISTANCE"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "ST_DISTANCE",
            FunctionType::Scalar,
            "Returns 2D Euclidean planar distance between two Point geometries",
            FunctionSignature::new(
                FunctionDataType::Float,
                vec![FunctionDataType::String, FunctionDataType::String],
                2,
                2,
            ),
        )
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(StDistanceFunction)
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "ST_DISTANCE", 2);
        if args[0].is_null() || args[1].is_null() {
            return Ok(Value::null_unknown());
        }

        let (x1, y1) = parse_point_coords(&args[0], "ST_DISTANCE")?;
        let (x2, y2) = parse_point_coords(&args[1], "ST_DISTANCE")?;

        let dx = x2 - x1;
        let dy = y2 - y1;
        let dist = (dx * dx + dy * dy).sqrt();

        Ok(Value::Float(dist))
    }
}

/// ST_DISTANCE_SPHERE(p1, p2) — calculates spherical geodesic distance (Haversine formula) in meters
#[derive(Default)]
pub struct StDistanceSphereFunction;

impl ScalarFunction for StDistanceSphereFunction {
    fn name(&self) -> &str {
        "ST_DISTANCE_SPHERE"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "ST_DISTANCE_SPHERE",
            FunctionType::Scalar,
            "Returns great-circle distance in meters between two Point geometries using Haversine formula",
            FunctionSignature::new(
                FunctionDataType::Float,
                vec![FunctionDataType::String, FunctionDataType::String],
                2,
                2,
            ),
        )
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(StDistanceSphereFunction)
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "ST_DISTANCE_SPHERE", 2);
        if args[0].is_null() || args[1].is_null() {
            return Ok(Value::null_unknown());
        }

        let (lon1, lat1) = parse_point_coords(&args[0], "ST_DISTANCE_SPHERE")?;
        let (lon2, lat2) = parse_point_coords(&args[1], "ST_DISTANCE_SPHERE")?;

        // Convert degrees to radians
        let lat1_rad = lat1.to_radians();
        let lat2_rad = lat2.to_radians();
        let dlat_rad = (lat2 - lat1).to_radians();
        let dlon_rad = (lon2 - lon1).to_radians();

        // Haversine formula
        let a = (dlat_rad / 2.0).sin().powi(2)
            + lat1_rad.cos() * lat2_rad.cos() * (dlon_rad / 2.0).sin().powi(2);
        let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());
        let distance = EARTH_RADIUS_METERS * c;

        Ok(Value::Float(distance))
    }
}

// ============================================================================
// ST_DWithin
// ============================================================================

/// ST_DWITHIN(p1, p2, distance) — returns true if planar distance between p1 and p2 <= distance
#[derive(Default)]
pub struct StDWithinFunction;

impl ScalarFunction for StDWithinFunction {
    fn name(&self) -> &str {
        "ST_DWITHIN"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "ST_DWITHIN",
            FunctionType::Scalar,
            "Returns true if the Euclidean distance between two Point geometries is within a specified threshold",
            FunctionSignature::new(
                FunctionDataType::Boolean,
                vec![
                    FunctionDataType::String,
                    FunctionDataType::String,
                    FunctionDataType::Float,
                ],
                3,
                3,
            ),
        )
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(StDWithinFunction)
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "ST_DWITHIN", 3);
        if args[0].is_null() || args[1].is_null() || args[2].is_null() {
            return Ok(Value::null_unknown());
        }

        let (x1, y1) = parse_point_coords(&args[0], "ST_DWITHIN")?;
        let (x2, y2) = parse_point_coords(&args[1], "ST_DWITHIN")?;
        let threshold = match &args[2] {
            Value::Float(f) => *f,
            Value::Integer(i) => *i as f64,
            _ => {
                return Err(Error::invalid_argument(
                    "ST_DWITHIN distance threshold must be a number",
                ))
            }
        };

        let dx = x2 - x1;
        let dy = y2 - y1;
        let dist = (dx * dx + dy * dy).sqrt();

        Ok(Value::Boolean(dist <= threshold))
    }
}

// ============================================================================
// ST_AsText / ST_GeomFromText
// ============================================================================

/// ST_ASTEXT(geom) — converts a Point geometry into standard Well-Known Text (WKT) format
#[derive(Default)]
pub struct StAsTextFunction;

impl ScalarFunction for StAsTextFunction {
    fn name(&self) -> &str {
        "ST_ASTEXT"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "ST_ASTEXT",
            FunctionType::Scalar,
            "Returns the Well-Known Text (WKT) representation of a geometry",
            FunctionSignature::new(
                FunctionDataType::String,
                vec![FunctionDataType::String],
                1,
                1,
            ),
        )
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(StAsTextFunction)
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "ST_ASTEXT", 1);
        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        let (x, y) = parse_point_coords(&args[0], "ST_ASTEXT")?;
        let wkt = format!("POINT({} {})", x, y);
        Ok(Value::Text(SmartString::from_string(wkt)))
    }
}

/// ST_GEOMFROMTEXT(wkt) — parses WKT and produces a canonical GeoJSON geometry representation
#[derive(Default)]
pub struct StGeomFromTextFunction;

impl ScalarFunction for StGeomFromTextFunction {
    fn name(&self) -> &str {
        "ST_GEOMFROMTEXT"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "ST_GEOMFROMTEXT",
            FunctionType::Scalar,
            "Parses a Well-Known Text (WKT) geometry string into a canonical geometry",
            FunctionSignature::new(
                FunctionDataType::String,
                vec![FunctionDataType::String],
                1,
                1,
            ),
        )
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(StGeomFromTextFunction)
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "ST_GEOMFROMTEXT", 1);
        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        let (x, y) = parse_point_coords(&args[0], "ST_GEOMFROMTEXT")?;
        let geojson = format!(r#"{{"type":"Point","coordinates":[{},{}]}}"#, x, y);
        Ok(Value::Text(SmartString::from_string(geojson)))
    }
}

// ============================================================================
// ST_Contains
// ============================================================================

/// ST_CONTAINS(polygon, point) — tests whether a polygon geometry contains a point geometry
#[derive(Default)]
pub struct StContainsFunction;

impl ScalarFunction for StContainsFunction {
    fn name(&self) -> &str {
        "ST_CONTAINS"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "ST_CONTAINS",
            FunctionType::Scalar,
            "Returns true if the first geometry (Polygon) contains the second geometry (Point)",
            FunctionSignature::new(
                FunctionDataType::Boolean,
                vec![FunctionDataType::String, FunctionDataType::String],
                2,
                2,
            ),
        )
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(StContainsFunction)
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "ST_CONTAINS", 2);
        if args[0].is_null() || args[1].is_null() {
            return Ok(Value::null_unknown());
        }

        let poly_coords = parse_polygon_coords(&args[0], "ST_CONTAINS")?;
        let (px, py) = parse_point_coords(&args[1], "ST_CONTAINS")?;

        let is_inside = point_in_polygon(px, py, &poly_coords);
        Ok(Value::Boolean(is_inside))
    }
}

// ============================================================================
// ST_Area
// ============================================================================

/// ST_AREA(geom) — calculates 2D planar area of a Polygon geometry
#[derive(Default)]
pub struct StAreaFunction;

impl ScalarFunction for StAreaFunction {
    fn name(&self) -> &str {
        "ST_AREA"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "ST_AREA",
            FunctionType::Scalar,
            "Returns the 2D Cartesian area of a Polygon geometry",
            FunctionSignature::new(
                FunctionDataType::Float,
                vec![FunctionDataType::String],
                1,
                1,
            ),
        )
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(StAreaFunction)
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "ST_AREA", 1);
        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        let poly_coords = parse_polygon_coords(&args[0], "ST_AREA")?;
        let area = polygon_area(&poly_coords);
        Ok(Value::Float(area))
    }
}

// ============================================================================
// ST_Centroid
// ============================================================================

/// ST_CENTROID(geom) — computes the geometric center (centroid) of a Point or Polygon
#[derive(Default)]
pub struct StCentroidFunction;

impl ScalarFunction for StCentroidFunction {
    fn name(&self) -> &str {
        "ST_CENTROID"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "ST_CENTROID",
            FunctionType::Scalar,
            "Returns the geometric centroid of a Point or Polygon geometry as a Point GeoJSON string",
            FunctionSignature::new(
                FunctionDataType::String,
                vec![FunctionDataType::String],
                1,
                1,
            ),
        )
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(StCentroidFunction)
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "ST_CENTROID", 1);
        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        // Try as Point first
        if let Ok((x, y)) = parse_point_coords(&args[0], "ST_CENTROID") {
            let res = format!(r#"{{"type":"Point","coordinates":[{},{}]}}"#, x, y);
            return Ok(Value::Text(SmartString::from_string(res)));
        }

        // Fall back to Polygon centroid
        let poly_coords = parse_polygon_coords(&args[0], "ST_CENTROID")?;
        let n = poly_coords.len();
        let mut cx = 0.0f64;
        let mut cy = 0.0f64;
        let mut factor_sum = 0.0f64;

        for i in 0..n {
            let j = (i + 1) % n;
            let factor = poly_coords[i].0 * poly_coords[j].1 - poly_coords[j].0 * poly_coords[i].1;
            cx += (poly_coords[i].0 + poly_coords[j].0) * factor;
            cy += (poly_coords[i].1 + poly_coords[j].1) * factor;
            factor_sum += factor;
        }

        let area_6 = factor_sum * 3.0;
        let (cx, cy) = if area_6.abs() > f64::EPSILON {
            (cx / area_6, cy / area_6)
        } else {
            // Arithmetic mean fallback
            let sum_x: f64 = poly_coords.iter().map(|p| p.0).sum();
            let sum_y: f64 = poly_coords.iter().map(|p| p.1).sum();
            (sum_x / n as f64, sum_y / n as f64)
        };

        let res = format!(r#"{{"type":"Point","coordinates":[{},{}]}}"#, cx, cy);
        Ok(Value::Text(SmartString::from_string(res)))
    }
}

// ============================================================================
// ST_Intersects / ST_Envelope
// ============================================================================

/// ST_INTERSECTS(g1, g2) — returns true if two geometries spatially intersect
#[derive(Default)]
pub struct StIntersectsFunction;

impl ScalarFunction for StIntersectsFunction {
    fn name(&self) -> &str {
        "ST_INTERSECTS"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "ST_INTERSECTS",
            FunctionType::Scalar,
            "Returns true if two geometries spatially intersect",
            FunctionSignature::new(
                FunctionDataType::Boolean,
                vec![FunctionDataType::String, FunctionDataType::String],
                2,
                2,
            ),
        )
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(StIntersectsFunction)
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "ST_INTERSECTS", 2);
        if args[0].is_null() || args[1].is_null() {
            return Ok(Value::null_unknown());
        }

        let pt1 = parse_point_coords(&args[0], "ST_INTERSECTS");
        let pt2 = parse_point_coords(&args[1], "ST_INTERSECTS");

        match (pt1, pt2) {
            (Ok((x1, y1)), Ok((x2, y2))) => {
                let dist = ((x1 - x2).powi(2) + (y1 - y2).powi(2)).sqrt();
                Ok(Value::Boolean(dist < 1e-9))
            }
            (Err(_), Ok((px, py))) => {
                let poly = parse_polygon_coords(&args[0], "ST_INTERSECTS")?;
                Ok(Value::Boolean(point_in_polygon(px, py, &poly)))
            }
            (Ok((px, py)), Err(_)) => {
                let poly = parse_polygon_coords(&args[1], "ST_INTERSECTS")?;
                Ok(Value::Boolean(point_in_polygon(px, py, &poly)))
            }
            (Err(_), Err(_)) => {
                let poly1 = parse_polygon_coords(&args[0], "ST_INTERSECTS")?;
                let poly2 = parse_polygon_coords(&args[1], "ST_INTERSECTS")?;

                // Bounding box intersection check first
                let min_x1 = poly1.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
                let max_x1 = poly1.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
                let min_y1 = poly1.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
                let max_y1 = poly1.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);

                let min_x2 = poly2.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
                let max_x2 = poly2.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
                let min_y2 = poly2.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
                let max_y2 = poly2.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);

                if max_x1 < min_x2 || min_x1 > max_x2 || max_y1 < min_y2 || min_y1 > max_y2 {
                    return Ok(Value::Boolean(false));
                }

                // Check if any point of poly1 is in poly2 or vice versa
                let p1_in_p2 = poly1.iter().any(|&pt| point_in_polygon(pt.0, pt.1, &poly2));
                let p2_in_p1 = poly2.iter().any(|&pt| point_in_polygon(pt.0, pt.1, &poly1));
                Ok(Value::Boolean(p1_in_p2 || p2_in_p1))
            }
        }
    }
}

/// ST_ENVELOPE(geom) — returns the minimum bounding box polygon for a geometry
#[derive(Default)]
pub struct StEnvelopeFunction;

impl ScalarFunction for StEnvelopeFunction {
    fn name(&self) -> &str {
        "ST_ENVELOPE"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "ST_ENVELOPE",
            FunctionType::Scalar,
            "Returns the minimum bounding box for a geometry as a Polygon WKT",
            FunctionSignature::new(
                FunctionDataType::String,
                vec![FunctionDataType::String],
                1,
                1,
            ),
        )
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(StEnvelopeFunction)
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "ST_ENVELOPE", 1);
        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        if let Ok((x, y)) = parse_point_coords(&args[0], "ST_ENVELOPE") {
            let res = format!("POLYGON(({} {}, {} {}, {} {}, {} {}, {} {}))", x, y, x, y, x, y, x, y, x, y);
            return Ok(Value::Text(SmartString::from_string(res)));
        }

        let poly = parse_polygon_coords(&args[0], "ST_ENVELOPE")?;
        let min_x = poly.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
        let max_x = poly.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
        let min_y = poly.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
        let max_y = poly.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);

        let res = format!(
            "POLYGON(({} {}, {} {}, {} {}, {} {}, {} {}))",
            min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y, min_x, min_y
        );
        Ok(Value::Text(SmartString::from_string(res)))
    }
}

// ============================================================================
// ST_Length / ST_Perimeter / ST_NumPoints / ST_SRID / ST_SetSRID
// ============================================================================

/// ST_LENGTH(geom) — calculates Cartesian length
#[derive(Default)]
pub struct StLengthFunction;

impl ScalarFunction for StLengthFunction {
    fn name(&self) -> &str {
        "ST_LENGTH"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "ST_LENGTH",
            FunctionType::Scalar,
            "Returns 2D Cartesian length of a geometry (0 for points)",
            FunctionSignature::new(
                FunctionDataType::Float,
                vec![FunctionDataType::String],
                1,
                1,
            ),
        )
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(StLengthFunction)
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "ST_LENGTH", 1);
        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        if parse_point_coords(&args[0], "ST_LENGTH").is_ok() {
            return Ok(Value::Float(0.0));
        }

        let poly = parse_polygon_coords(&args[0], "ST_LENGTH")?;
        let mut len = 0.0f64;
        for i in 0..poly.len().saturating_sub(1) {
            let dx = poly[i + 1].0 - poly[i].0;
            let dy = poly[i + 1].1 - poly[i].1;
            len += (dx * dx + dy * dy).sqrt();
        }

        Ok(Value::Float(len))
    }
}

/// ST_PERIMETER(polygon) — calculates boundary perimeter of a polygon
#[derive(Default)]
pub struct StPerimeterFunction;

impl ScalarFunction for StPerimeterFunction {
    fn name(&self) -> &str {
        "ST_PERIMETER"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "ST_PERIMETER",
            FunctionType::Scalar,
            "Returns the 2D perimeter of a Polygon geometry",
            FunctionSignature::new(
                FunctionDataType::Float,
                vec![FunctionDataType::String],
                1,
                1,
            ),
        )
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(StPerimeterFunction)
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "ST_PERIMETER", 1);
        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        let poly = parse_polygon_coords(&args[0], "ST_PERIMETER")?;
        let mut perimeter = 0.0f64;
        let n = poly.len();
        for i in 0..n {
            let j = (i + 1) % n;
            let dx = poly[j].0 - poly[i].0;
            let dy = poly[j].1 - poly[i].1;
            perimeter += (dx * dx + dy * dy).sqrt();
        }

        Ok(Value::Float(perimeter))
    }
}

/// ST_NUMPOINTS(geom) — returns the number of vertices/points in a geometry
#[derive(Default)]
pub struct StNumPointsFunction;

impl ScalarFunction for StNumPointsFunction {
    fn name(&self) -> &str {
        "ST_NUMPOINTS"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "ST_NUMPOINTS",
            FunctionType::Scalar,
            "Returns the number of points/vertices in a geometry",
            FunctionSignature::new(
                FunctionDataType::Integer,
                vec![FunctionDataType::String],
                1,
                1,
            ),
        )
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(StNumPointsFunction)
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "ST_NUMPOINTS", 1);
        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        if parse_point_coords(&args[0], "ST_NUMPOINTS").is_ok() {
            return Ok(Value::Integer(1));
        }

        let poly = parse_polygon_coords(&args[0], "ST_NUMPOINTS")?;
        Ok(Value::Integer(poly.len() as i64))
    }
}

/// ST_SRID(geom) — returns the Spatial Reference System Identifier (default 4326 / WGS84)
#[derive(Default)]
pub struct StSridFunction;

impl ScalarFunction for StSridFunction {
    fn name(&self) -> &str {
        "ST_SRID"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "ST_SRID",
            FunctionType::Scalar,
            "Returns the Spatial Reference System Identifier (SRID) of a geometry",
            FunctionSignature::new(
                FunctionDataType::Integer,
                vec![FunctionDataType::String],
                1,
                1,
            ),
        )
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(StSridFunction)
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "ST_SRID", 1);
        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }
        Ok(Value::Integer(4326))
    }
}

/// ST_SETSRID(geom, srid) — sets the SRID on a geometry
#[derive(Default)]
pub struct StSetSridFunction;

impl ScalarFunction for StSetSridFunction {
    fn name(&self) -> &str {
        "ST_SETSRID"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "ST_SETSRID",
            FunctionType::Scalar,
            "Sets the Spatial Reference System Identifier (SRID) on a geometry",
            FunctionSignature::new(
                FunctionDataType::String,
                vec![FunctionDataType::String, FunctionDataType::Integer],
                2,
                2,
            ),
        )
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(StSetSridFunction)
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "ST_SETSRID", 2);
        if args[0].is_null() || args[1].is_null() {
            return Ok(Value::null_unknown());
        }
        Ok(args[0].clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_st_point_and_coordinates() -> Result<()> {
        let p_fn = StPointFunction;
        let x_fn = StXFunction;
        let y_fn = StYFunction;

        let pt = p_fn.evaluate(&[Value::Float(10.5), Value::Float(20.5)])?;
        let x = x_fn.evaluate(std::slice::from_ref(&pt))?;
        let y = y_fn.evaluate(&[pt])?;

        assert_eq!(x, Value::Float(10.5));
        assert_eq!(y, Value::Float(20.5));
        Ok(())
    }

    #[test]
    fn test_st_distance_planar_and_sphere() -> Result<()> {
        let p1 = Value::text("POINT(0 0)");
        let p2 = Value::text("POINT(3 4)");

        let dist_fn = StDistanceFunction;
        let dist = dist_fn.evaluate(&[p1, p2])?;
        assert_eq!(dist, Value::Float(5.0));

        // London (0.1278 W, 51.5074 N) to Paris (2.3522 E, 48.8566 N) ~ 343 km
        let london = Value::text(r#"{"type":"Point","coordinates":[-0.1278, 51.5074]}"#);
        let paris = Value::text(r#"{"type":"Point","coordinates":[2.3522, 48.8566]}"#);

        let sphere_fn = StDistanceSphereFunction;
        let dist_m = sphere_fn.evaluate(&[london, paris])?;
        if let Value::Float(m) = dist_m {
            assert!(m > 340_000.0 && m < 350_000.0);
        } else {
            panic!("Expected float distance");
        }
        Ok(())
    }

    #[test]
    fn test_st_contains_and_area() -> Result<()> {
        let poly = Value::text("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))");
        let inside_pt = Value::text("POINT(5 5)");
        let outside_pt = Value::text("POINT(15 15)");

        let contains_fn = StContainsFunction;
        let in_res = contains_fn.evaluate(&[poly.clone(), inside_pt])?;
        let out_res = contains_fn.evaluate(&[poly.clone(), outside_pt])?;

        assert_eq!(in_res, Value::Boolean(true));
        assert_eq!(out_res, Value::Boolean(false));

        let area_fn = StAreaFunction;
        let area = area_fn.evaluate(&[poly])?;
        assert_eq!(area, Value::Float(100.0));
        Ok(())
    }
}
