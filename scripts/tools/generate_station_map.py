#!/usr/bin/env python3
"""Generate interactive station distribution map with distance slider using plotly and mapbox."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import List, Tuple

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Error: plotly is required. Install with: pip install plotly")
    sys.exit(1)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
METADATA_PATH = PROJECT_ROOT / "data" / "external" / "cimis_station_metadata.json"
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.json"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "figures"
OUTPUT_HTML = OUTPUT_DIR / "station_distribution_map.html"


def load_mapbox_token() -> str | None:
    """Load mapbox token from settings.json."""
    if not SETTINGS_PATH.exists():
        print(f"Warning: {SETTINGS_PATH} not found. Map will use default tiles.")
        return None
    
    try:
        with open(SETTINGS_PATH, 'r', encoding='utf-8') as f:
            settings = json.load(f)
            token = settings.get("mapbox_token")
            if token and token != "<YOUR_MAPBOX_ACCESS_TOKEN>":
                return token
    except Exception as e:
        print(f"Warning: Failed to load mapbox token: {e}")
    
    return None


def load_station_metadata() -> list[dict]:
    """Load station metadata from JSON file."""
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Station metadata not found at {METADATA_PATH}. "
            "Run scripts/tools/fetch_station_metadata.py first."
        )
    
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Filter out stations with missing coordinates
    import math
    valid_stations = [
        s for s in metadata
        if not (math.isnan(s.get("Latitude", float('nan'))) 
                or math.isnan(s.get("Longitude", float('nan'))))
    ]
    
    return valid_stations


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two points on Earth (in km).
    
    Args:
        lat1, lon1: Latitude and longitude of first point (in degrees).
        lat2, lon2: Latitude and longitude of second point (in degrees).
    
    Returns:
        Distance in kilometers.
    """
    # Earth radius in km
    R = 6371.0
    
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c


def compute_distance_matrix(stations: list[dict]) -> Tuple[List[List[float]], List[dict]]:
    """Compute pairwise distance matrix between all stations.
    
    Args:
        stations: List of station metadata dictionaries.
    
    Returns:
        Tuple of (distance_matrix, station_info_list).
        distance_matrix[i][j] is distance from station i to station j in km.
        station_info_list contains dict with 'id', 'name', 'lat', 'lon' for each station.
    """
    n = len(stations)
    distance_matrix = [[0.0] * n for _ in range(n)]
    station_info = []
    
    for i, s in enumerate(stations):
        station_info.append({
            'id': s['Stn Id'],
            'name': s.get('Stn Name', f"Station {s['Stn Id']}"),
            'lat': s['Latitude'],
            'lon': s['Longitude']
        })
        
        for j in range(i + 1, n):
            dist = haversine_distance(
                s['Latitude'], s['Longitude'],
                stations[j]['Latitude'], stations[j]['Longitude']
            )
            distance_matrix[i][j] = dist
            distance_matrix[j][i] = dist
    
    return distance_matrix, station_info


def get_edges_within_distance(
    distance_matrix: List[List[float]], 
    station_info: List[dict], 
    max_distance: float
) -> List[dict]:
    """Get edges (connections) between stations within max_distance.
    
    Args:
        distance_matrix: Pairwise distance matrix.
        station_info: List of station information dicts.
        max_distance: Maximum distance threshold in km.
    
    Returns:
        List of edge dictionaries with 'lat1', 'lon1', 'lat2', 'lon2', 'distance'.
    """
    edges = []
    n = len(station_info)
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = distance_matrix[i][j]
            if dist <= max_distance:
                edges.append({
                    'lat1': station_info[i]['lat'],
                    'lon1': station_info[i]['lon'],
                    'lat2': station_info[j]['lat'],
                    'lon2': station_info[j]['lon'],
                    'distance': dist
                })
    
    return edges


def generate_circle_coords(lat_center: float, lon_center: float, radius_km: float, n_points: int = 64) -> Tuple[List[float], List[float]]:
    """Generate coordinates for a circle around a center point.
    
    Args:
        lat_center: Center latitude in degrees.
        lon_center: Center longitude in degrees.
        radius_km: Radius in kilometers.
        n_points: Number of points to generate (more = smoother circle).
    
    Returns:
        Tuple of (latitudes, longitudes) lists for the circle.
    """
    import math
    R = 6371.0  # Earth radius in km
    
    # Convert center to radians
    lat_center_rad = math.radians(lat_center)
    lon_center_rad = math.radians(lon_center)
    
    # Angular radius in radians
    angular_radius = radius_km / R
    
    lats = []
    lons = []
    
    for i in range(n_points + 1):  # +1 to close the circle
        angle = 2 * math.pi * i / n_points
        
        # Calculate point on circle
        lat_point_rad = math.asin(
            math.sin(lat_center_rad) * math.cos(angular_radius) +
            math.cos(lat_center_rad) * math.sin(angular_radius) * math.cos(angle)
        )
        
        lon_point_rad = lon_center_rad + math.atan2(
            math.sin(angle) * math.sin(angular_radius) * math.cos(lat_center_rad),
            math.cos(angular_radius) - math.sin(lat_center_rad) * math.sin(lat_point_rad)
        )
        
        lats.append(math.degrees(lat_point_rad))
        lons.append(math.degrees(lon_point_rad))
    
    return lats, lons


def create_station_map(
    stations: list[dict], 
    mapbox_token: str | None = None,
    initial_distance: float = 50.0,
    map_style: str = "carto-positron",  # Clean map style
    map_height: int = 1200  # Increased height
) -> go.Figure:
    """Create interactive station distribution map with distance slider.
    
    Args:
        stations: List of station metadata dictionaries.
        mapbox_token: Optional mapbox access token for better tiles.
        initial_distance: Initial distance threshold in km for connections.
    
    Returns:
        Plotly figure object with slider.
    """
    # Compute distance matrix
    distance_matrix, station_info = compute_distance_matrix(stations)
    
    # Extract coordinates and names
    lats = [s["Latitude"] for s in stations]
    lons = [s["Longitude"] for s in stations]
    names = [s["name"] for s in station_info]
    ids = [s["id"] for s in station_info]
    
    # Find distance range for slider
    max_dist = 0.0
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix[i])):
            if distance_matrix[i][j] > max_dist:
                max_dist = distance_matrix[i][j]
    
    # Create slider steps (every 10km up to max, then some larger steps)
    # Limit max step to 150km for better visualization (no need for 500km circles)
    slider_steps = []
    step_values = []
    
    # Small steps: 0 to 100km every 10km
    for d in range(0, min(100, int(max_dist) + 10), 10):
        step_values.append(float(d))
    
    # Medium steps: 100 to 150km every 10km (more granular)
    for d in range(100, min(150, int(max_dist) + 10), 10):
        if d not in step_values:
            step_values.append(float(d))
    
    # Optional larger steps: 150km+ every 25km (if needed for connections)
    # But limit to reasonable range for visualization
    max_step_for_circles = min(150, max_dist)  # Limit circles to 150km
    for d in range(150, int(max_step_for_circles) + 25, 25):
        if d not in step_values and d <= max_step_for_circles:
            step_values.append(float(d))
    
    # Always include max distance if reasonable (but won't create circles for it)
    # Only add max distance for connections, not for circles
    if max_dist not in step_values and max_dist <= 200:
        step_values.append(math.ceil(max_dist / 10) * 10)
    
    step_values = sorted(set(step_values))
    
    # Note: We'll generate circles only for radii <= 150km in the loop
    # Connections can use all step values
    
    # Find initial step index (before loop)
    initial_step_idx = 0
    for i, dist in enumerate(step_values):
        if dist >= initial_distance:
            initial_step_idx = i
            break
    
    # Create figure with multiple traces (one for each slider step)
    fig = go.Figure()
    
    # First, add station markers BEFORE everything else so they appear on top
    # Calculate connections for each station at initial distance
    initial_edges = get_edges_within_distance(distance_matrix, station_info, step_values[initial_step_idx])
    # Count connections per station
    station_connections = [0] * len(station_info)
    for edge in initial_edges:
        # Find station indices for this edge
        for i, st in enumerate(station_info):
            if (abs(st['lat'] - edge['lat1']) < 0.0001 and abs(st['lon'] - edge['lon1']) < 0.0001):
                station_connections[i] += 1
            elif (abs(st['lat'] - edge['lat2']) < 0.0001 and abs(st['lon'] - edge['lon2']) < 0.0001):
                station_connections[i] += 1
    
    hover_text = [
        f"<b>{name}</b><br>"
        f"Station ID: {sid}<br>"
        f"Lat: {lat:.4f}°<br>"
        f"Lon: {lon:.4f}°<br>"
        f"Neighbors ({step_values[initial_step_idx]:.0f} km): {station_connections[i]}"
        for i, (name, sid, lat, lon) in enumerate(zip(names, ids, lats, lons))
    ]
    
    # Add station markers with white border (using two layers for border effect)
    # First layer: white outer circle (border)
    fig.add_trace(go.Scattermapbox(
        lat=lats,
        lon=lons,
        mode='markers',
        marker=dict(
            size=30,  # Larger white circle as border
            color='white',
            opacity=1.0,
            symbol='circle'
        ),
        hoverinfo='skip',
        name='Station Borders',
        showlegend=False,
        visible=True,
    ))
    
    # Second layer: red outer circle (larger)
    fig.add_trace(go.Scattermapbox(
        lat=lats,
        lon=lons,
        mode='markers',
        marker=dict(
            size=25,  # Larger red circle
            color='#FF0000',  # Bright red
            opacity=1.0,
            symbol='circle',
        ),
        customdata=hover_text,
        hovertemplate='%{customdata}<extra></extra>',
        name='CIMIS Stations',
        showlegend=True,
        visible=True,
    ))
    
    # Third layer: white inner circle (center point)
    fig.add_trace(go.Scattermapbox(
        lat=lats,
        lon=lons,
        mode='markers',
        marker=dict(
            size=12,  # White inner circle
            color='white',
            opacity=1.0,
            symbol='circle'
        ),
        hoverinfo='skip',
        name='Station Centers',
        showlegend=False,
        visible=True
    ))
    
    # Optional station ID labels (toggled via buttons)
    text_trace_idx = len(fig.data)
    id_labels = [f"#{sid}" for sid in ids]
    fig.add_trace(go.Scattermapbox(
        lat=lats,
        lon=lons,
        mode='text',
        text=id_labels,
        textposition='top center',
        textfont=dict(color='#1b4965', size=12),
        name='Station IDs',
        showlegend=False,
        hoverinfo='skip',
        visible=False
    ))
    
    # Maximum radius for coverage circles (to avoid huge circles)
    MAX_CIRCLE_RADIUS = 150  # km - don't show circles larger than this
    
    # Add traces for each distance threshold
    for step_idx, max_dist_step in enumerate(step_values):
        # Get edges within this distance
        edges = get_edges_within_distance(distance_matrix, station_info, max_dist_step)
        
        # Add station markers (always visible)
        visible = (step_idx == initial_step_idx)  # Only initial step visible initially
        
        # Only add coverage circles if radius is within reasonable range
        # For large radii (>150km), skip circles but still show connections
        show_circles = max_dist_step <= MAX_CIRCLE_RADIUS
        
        # Add coverage circles for each station (radius visualization)
        # IMPORTANT: All stations use the SAME radius (max_dist_step) from the slider
        # Only generate circles if radius is within reasonable range
        for station_idx, station in enumerate(station_info):
            # Only generate circle coordinates if we're going to show them
            if show_circles:
                # Generate circle with radius = max_dist_step (from slider)
                # ALL stations use the SAME radius value for this step
                circle_lats, circle_lons = generate_circle_coords(
                    station['lat'], 
                    station['lon'], 
                    max_dist_step,  # Same radius for all stations
                    n_points=64
                )
                
                # Add coverage circle (same color for all distances, same radius for all stations)
                fig.add_trace(go.Scattermapbox(
                    lat=circle_lats,
                    lon=circle_lons,
                    mode='lines',
                    line=dict(
                        width=2,
                        color='rgba(100, 150, 255, 0.5)',  # Same color for all distances
                    ),
                    fill='toself',
                    fillcolor='rgba(100, 150, 255, 0.15)',  # Same color for all distances
                    name=f'Coverage ({max_dist_step:.0f} km)',  # Shows the radius value
                    showlegend=False,
                    visible=visible,
                    hoverinfo='skip'
                ))
            else:
                # For large radii, don't generate circles
                # Add empty trace to maintain trace order
                fig.add_trace(go.Scattermapbox(
                    lat=[None],
                    lon=[None],
                    mode='lines',
                    name=f'Coverage ({max_dist_step:.0f} km - hidden)',
                    showlegend=False,
                    visible=False,
                    hoverinfo='skip'
                ))
        
        # Connection lines removed as requested
    
    # Create slider
    steps = []
    n_stations = len(station_info)
    
    # Trace order: 
    # - First 4 traces: station markers + centers + labels
    # - Then: coverage circles (n_stations per step)
    text_initial_visible = False
    
    for step_idx, max_dist_step in enumerate(step_values):
        # Build visibility list
        # First 3 traces always visible; label trace toggled via button
        visible_list = [True, True, True, text_initial_visible]
        
        # Coverage circles: n_stations per step
        for cs_idx in range(len(step_values)):
            for st_idx in range(n_stations):
                visible_list.append(cs_idx == step_idx)  # Circle
        
        step = dict(
            method='update',
            args=[
                {'visible': visible_list},
                {'title': f'Station Network (Distance ≤ {max_dist_step:.0f} km)'}
            ],
            label=f'{max_dist_step:.0f} km'
        )
        steps.append(step)
    
    # Set initial visibility
    # Each step has: n_stations circles (no connection lines)
    n_coverage_traces = len(step_values) * n_stations  # Circles only
    n_total_traces = 4 + n_coverage_traces  # 3 station layers + label + coverage circles
    
    initial_visible = [True, True, True, text_initial_visible]
    initial_visible.extend([False] * n_coverage_traces)  # Coverage circles
    
    # Show initial step coverage circles
    initial_coverage_start = 4 + initial_step_idx * n_stations  # After base layers
    initial_coverage_end = initial_coverage_start + n_stations  # Circles
    for i in range(initial_coverage_start, initial_coverage_end):
        initial_visible[i] = True
    
    # Update trace visibility
    for i in range(len(fig.data)):
        if i < len(initial_visible):
            fig.data[i].visible = initial_visible[i]
    
    # Map style options (clean maps without roads/labels)
    clean_map_styles = {
        "carto-positron": "carto-positron",  # Very clean, light gray background
        "carto-darkmatter": "carto-darkmatter",  # Clean dark background
        "white-bg": "open-street-map",  # Fallback to OSM
        "open-street-map": "open-street-map",  # Standard OSM
        "satellite": "satellite",
        "streets": "streets"
    }
    
    selected_style = clean_map_styles.get(map_style, "carto-positron")
    
    # Add slider
    sliders = [dict(
        active=initial_step_idx,
        currentvalue={"prefix": "Max Distance: "},
        pad={"t": 50},
        steps=steps,
        len=0.9,
        x=0.05,
        xanchor="left",
        y=0,
        yanchor="bottom"
    )]
    
    # Toggle buttons for station ID labels
    id_toggle_menu = dict(
        type="buttons",
        direction="right",
        buttons=[
            dict(
                label="Show IDs",
                method="restyle",
                args=[{"visible": True}, [text_trace_idx]]
            ),
            dict(
                label="Hide IDs",
                method="restyle",
                args=[{"visible": False}, [text_trace_idx]]
            ),
        ],
        showactive=True,
        x=0.5,
        y=1.08,
        xanchor="center",
        yanchor="top",
        pad={"t": 20}
    )
    
    # Calculate appropriate zoom level based on coverage radius
    # Larger radius needs lower zoom to fit all circles
    initial_radius = step_values[initial_step_idx]
    if initial_radius <= 30:
        zoom_level = 8  # Closer zoom for small radii
    elif initial_radius <= 50:
        zoom_level = 7  # Medium zoom for medium radii
    elif initial_radius <= 100:
        zoom_level = 6  # Wider zoom for large radii
    else:
        zoom_level = 5  # Very wide zoom for very large radii
    
    # Set map style with increased height and zoom enabled
    layout_dict = dict(
        mapbox=dict(
            center=dict(
                lat=sum(lats) / len(lats),
                lon=sum(lons) / len(lons)
            ),
            zoom=zoom_level,
            style=selected_style,
            # Enable mouse zoom and pan
            bearing=0,
            pitch=0
        ),
        title=dict(
            text=f'CIMIS Station Network (Distance ≤ {step_values[initial_step_idx]:.0f} km)',
            x=0.5,
            font=dict(size=20)
        ),
        height=map_height,  # Increased height
        margin=dict(l=0, r=0, t=120, b=100),
        sliders=sliders,
        updatemenus=[id_toggle_menu]
    )
    
    # Add mapbox token if available
    if mapbox_token and selected_style not in ["carto-positron", "carto-darkmatter", "white-bg"]:
        layout_dict["mapbox"]["accesstoken"] = mapbox_token
    
    fig.update_layout(**layout_dict)
    
    return fig


def main():
    """Generate station distribution map."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate interactive station distribution map")
    parser.add_argument(
        "--map-style",
        type=str,
        default="carto-positron",
        choices=["carto-positron", "carto-darkmatter", "white-bg", "open-street-map", "satellite", "streets"],
        help="Map style (default: carto-positron - clean map without roads/labels)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1200,
        help="Map height in pixels (default: 1200)"
    )
    parser.add_argument(
        "--initial-distance",
        type=float,
        default=50.0,
        help="Initial distance threshold in km (default: 50.0)"
    )
    
    args = parser.parse_args()
    
    print("Generating station distribution map...")
    print(f"Map style: {args.map_style}")
    print(f"Map height: {args.height}px")
    
    # Load mapbox token
    mapbox_token = load_mapbox_token()
    if mapbox_token:
        print("✓ Loaded mapbox token")
    else:
        print("⚠ Using default map tiles (mapbox token not available)")
    
    # Load station metadata
    try:
        stations = load_station_metadata()
        print(f"✓ Loaded metadata for {len(stations)} stations")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nTo generate station metadata, run:")
        print("  python scripts/tools/fetch_station_metadata.py")
        sys.exit(1)
    
    # Create map with distance slider
    fig = create_station_map(
        stations, 
        mapbox_token, 
        initial_distance=args.initial_distance,
        map_style=args.map_style,
        map_height=args.height
    )
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save HTML with zoom enabled
    fig.write_html(
        OUTPUT_HTML,
        include_plotlyjs='cdn',  # Use CDN for smaller file size
        config={
            'displayModeBar': True,
            'scrollZoom': True,  # Enable mouse wheel zoom
            'doubleClick': 'reset+autosize',  # Double click to reset
            'modeBarButtonsToRemove': []  # Keep all toolbar buttons
        }
    )
    
    print(f"✓ Saved map to {OUTPUT_HTML}")
    print(f"\nOpen the file in a web browser to view the interactive map.")
    print(f"\nFeatures:")
    print(f"  - Station markers (red outer + white inner circles)")
    print(f"  - Coverage circles (center=station, radius=slider distance)")
    print(f"  - Distance slider to adjust radius")
    print(f"  - Mouse zoom and pan enabled")
    print(f"  - Clean map style: {args.map_style}")


if __name__ == "__main__":
    main()

