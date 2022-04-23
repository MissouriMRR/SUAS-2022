import rrt
import helpers
import time
import plotter

flyZones = {
    "altitudeMin": 100.0,
    "altitudeMax": 750.0,
    "boundaryPoints": [
        {"latitude": 38.1462694444444, "longitude": -76.4281638888889},
        {"latitude": 38.151625, "longitude": -76.4286833333333},
        {"latitude": 38.1518888888889, "longitude": -76.4314666666667},
        {"latitude": 38.1505944444444, "longitude": -76.4353611111111},
        {"latitude": 38.1475666666667, "longitude": -76.4323416666667},
        {"latitude": 38.1446666666667, "longitude": -76.4329472222222},
        {"latitude": 38.1432555555556, "longitude": -76.4347666666667},
        {"latitude": 38.1404638888889, "longitude": -76.4326361111111},
        {"latitude": 38.1407194444444, "longitude": -76.4260138888889},
        {"latitude": 38.1437611111111, "longitude": -76.4212055555556},
        {"latitude": 38.1473472222222, "longitude": -76.4232111111111},
        {"latitude": 38.1461305555556, "longitude": -76.4266527777778},
        {"latitude": 38.1462694444444, "longitude": -76.4281638888889},
    ],
}

waypoints = [
    {"latitude": 38.1446916666667, "longitude": -76.4279944444445, "altitude": 200.0},
    {"latitude": 38.1461944444444, "longitude": -76.4237138888889, "altitude": 300.0},
    {"latitude": 38.1438972222222, "longitude": -76.42255, "altitude": 400.0},
    {"latitude": 38.1417722222222, "longitude": -76.4251083333333, "altitude": 400.0},
    {"latitude": 38.14535, "longitude": -76.428675, "altitude": 300.0},
    {"latitude": 38.1508972222222, "longitude": -76.4292972222222, "altitude": 300.0},
    {"latitude": 38.1514944444444, "longitude": -76.4313833333333, "altitude": 300.0},
    {"latitude": 38.1505333333333, "longitude": -76.434175, "altitude": 300.0},
    {"latitude": 38.1479472222222, "longitude": -76.4316055555556, "altitude": 200.0},
    {"latitude": 38.1443333333333, "longitude": -76.4322888888889, "altitude": 200.0},
    {"latitude": 38.1433166666667, "longitude": -76.4337111111111, "altitude": 300.0},
    {"latitude": 38.1410944444444, "longitude": -76.4321555555556, "altitude": 400.0},
    {"latitude": 38.1415777777778, "longitude": -76.4252472222222, "altitude": 400.0},
    {"latitude": 38.1446083333333, "longitude": -76.4282527777778, "altitude": 200.0},
]

obstacles = [
    {"latitude": 38.146689, "longitude": -76.426475, "radius": 150.0, "height": 750.0},
    {"latitude": 38.142914, "longitude": -76.430297, "radius": 300.0, "height": 300.0},
    {"latitude": 38.149504, "longitude": -76.43311, "radius": 100.0, "height": 750.0},
    {"latitude": 38.148711, "longitude": -76.429061, "radius": 300.0, "height": 750.0},
    {"latitude": 38.144203, "longitude": -76.426155, "radius": 50.0, "height": 400.0},
    {"latitude": 38.146003, "longitude": -76.430733, "radius": 225.0, "height": 500.0},
]

if __name__ == "__main__":
    # Add utm coordinates to all
    boundary = helpers.all_latlon_to_utm(flyZones["boundaryPoints"])
    obstacles = helpers.all_latlon_to_utm(obstacles)
    waypoints = helpers.all_latlon_to_utm(waypoints)

    zone_num, zone_char = helpers.get_zone_info(boundary)

    # Convert silly units to proper units
    obstacles = helpers.all_feet_to_meters(obstacles)

    # Create shapely representations of everything for use in algorithm
    boundary_shape = helpers.coords_to_shape(boundary)
    obstacle_shapes = helpers.circles_to_shape(obstacles)
    waypoints_points = helpers.coords_to_points(waypoints)

    # Magic
    start = waypoints_points[0]  # 4
    goal = waypoints_points[1]  # 5
    start_time = time.time()
    G, ellr, informed_boundary = rrt.RRT_star(start, goal, boundary_shape, obstacle_shapes)
    print(f"rrt runtime = {(time.time()-start_time):.3f}s")

    if G.success:
        path = rrt.dijkstra(G)
        path = rrt.relax_path(path, obstacle_shapes)
        plotter.plot(obstacles, boundary, G, path, ellr, informed_boundary)
        gps_path = helpers.path_to_latlon(path, zone_num, zone_char)
        print(gps_path)
    else:
        print("major error! could not find a path!")
        plotter.plot(obstacles, boundary, G, ellr, informed_boundary)
