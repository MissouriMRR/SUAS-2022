import plotter
import search_path


search_area = [
    {
        "altitudeMin": 100.0,
        "altitudeMax": 400.0,
        "boundaryPoints": [
            {"latitude": 37.95120673695754, "longitude": -91.78693406657685},
            {"latitude": 37.951248325955994, "longitude": -91.78207299965484},
            {"latitude": 37.948226725559984, "longitude": -91.78191292975686},
            {"latitude": 37.94711526778687, "longitude": -91.78116415681103},
            {"latitude": 37.94623824627026, "longitude": -91.78490802154013},
            {"latitude": 37.94797658615526, "longitude": -91.78697801835733},
            {"latitude": 37.95120673695754, "longitude": -91.78693406657685},
        ],
    }
][0]

if __name__ == "__main__":
    search_area_points = search_area["boundaryPoints"]

    # Add utm coordinates to all
    search_area_points = search_path.all_latlon_to_utm(search_area_points)

    # Generate search path
    buffer_distance = -50
    search_paths = search_path.generate_search_path(search_area_points, buffer_distance)

    # Plot data
    plotter.plot_data(search_paths)
