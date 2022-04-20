import plotter
import point_finder


flyZones = [
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

obstacles = [
    {
        "latitude": 37.94986244280119,
        "longitude": -91.78386816205061,
        "radius": 20,
        "height": 400,
    },
    {
        "latitude": 37.949388706571064,
        "longitude": -91.78569200437985,
        "radius": 30,
        "height": 200,
    },
    {
        "latitude": 37.94828884469052,
        "longitude": -91.78546673913256,
        "radius": 50,
        "height": 150,
    },
    {
        "latitude": 37.94922429469271,
        "longitude": -91.7825514554602,
        "radius": 100,
        "height": 200,
    },
]

odlc = {"latitude": 37.94873549190636, "longitude": -91.78086677968352}

if __name__ == "__main__":
    boundary_points = flyZones["boundaryPoints"]

    # Add utm coordinates to all
    boundary_points = point_finder.all_latlon_to_utm(boundary_points)
    obstacles = point_finder.all_latlon_to_utm(obstacles)
    odlc = point_finder.latlon_to_utm(odlc)

    # Find the closest point
    (closest_point, shrunk_boundary) = point_finder.find_closest_point(
        odlc, boundary_points, obstacles
    )

    # Plot data
    plotter.plot_data(odlc, closest_point, boundary_points, shrunk_boundary, obstacles)
