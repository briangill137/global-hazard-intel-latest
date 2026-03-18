from pathlib import Path
from typing import List, Dict

try:
    import folium
except Exception as exc:  # noqa: BLE001
    folium = None
    _folium_error = exc
else:
    _folium_error = None

try:
    from folium.plugins import HeatMap
except Exception:  # noqa: BLE001
    HeatMap = None

MAP_PATH = Path("map.html")


def build_map(events: List[Dict]) -> Path:
    """Generate folium map with hazard markers and optional heatmap."""
    if folium is None:
        # Create placeholder HTML when folium is unavailable
        placeholder = "<html><body><h3>Folium not installed.</h3><p>Install folium to view maps.</p></body></html>"
        MAP_PATH.write_text(placeholder, encoding="utf-8")
        return MAP_PATH

    base_map = folium.Map(location=[20, 0], zoom_start=2, tiles="OpenStreetMap", control_scale=True)
    folium.TileLayer(
        "CartoDB dark_matter",
        name="Dark",
        attr="© OpenStreetMap contributors © CartoDB",
    ).add_to(base_map)
    folium.TileLayer(
        "Stamen Terrain",
        name="Terrain",
        attr="Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL.",
    ).add_to(base_map)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Esri Satellite",
        overlay=False,
        control=True,
    ).add_to(base_map)

    heat_points = []
    for event in events:
        lat = event.get("lat")
        lon = event.get("lon")
        location = event.get("location", "Unknown")
        severity = float(event.get("severity", 10))
        popup = f"{event.get('type')} – {location} | Sev {severity:.1f}"
        if lat is not None and lon is not None:
            color = "red" if severity > 70 else "orange" if severity > 40 else "green"
            icon = "fire" if str(event.get("type", "")).lower().startswith("wildfire") else "exclamation-sign"
            folium.Marker(
                [lat, lon],
                popup=popup,
                icon=folium.Icon(color=color, icon=icon),
            ).add_to(base_map)
            heat_points.append([lat, lon, severity])

    if HeatMap and heat_points:
        HeatMap(heat_points, radius=14, blur=18, min_opacity=0.3).add_to(base_map)

    folium.LayerControl(collapsed=False).add_to(base_map)
    base_map.save(str(MAP_PATH))
    return MAP_PATH
