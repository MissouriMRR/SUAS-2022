import rrt
import helpers
import plotter


OBSTACLE_BUFFER = 10  # meters

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

waypoints = [
    # {"latitude": 37.94883876837826, "longitude": -91.78443677093598, "altitude": 100.0},
    # {
    #     "latitude": 37.949016435151485,
    #     "longitude": -91.78364283711778,
    #     "altitude": 150.0,
    # },
    # {"latitude": 37.94957481364174, "longitude": -91.78369648129468, "altitude": 300.0},
    # {"latitude": 37.95004012580848, "longitude": -91.78468353414975, "altitude": 250.0},
    # {
    #     "latitude": 37.949735558177984,
    #     "longitude": -91.78542382379104,
    #     "altitude": 150.0,
    # },
    # {
    #     "latitude": 37.948790418686826,
    #     "longitude": -91.78605823636349,
    #     "altitude": 200.0,
    # },
    # {
    #     "latitude": 37.948576497594445,
    #     "longitude": -91.78460843230208,
    #     "altitude": 150.0,
    # },
    {
        "latitude": 37.947286878699536,
        "longitude": -91.78228398250506,
        "altitude": 150.0,
    },
    {
        "latitude": 37.95086689181623,
        "longitude": -91.78656411699848,
        "altitude": 150.0,
    },
]

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
        "radius": 40,
        "height": 200,
    },
    {
        "latitude": 37.94828884469052,
        "longitude": -91.78546673913256,
        "radius": 40,
        "height": 150,
    },
    {
        "latitude": 37.948861243306304,
        "longitude": -91.7830360828154,
        "radius": 30,
        "height": 300,
    },
    {
        "latitude": 37.947987798615436,
        "longitude": -91.7835146921038,
        "radius": 35,
        "height": 300,
    },
    {
        "latitude": 37.95007973521904,
        "longitude": -91.78522401099094,
        "radius": 50,
        "height": 300,
    },
]

if __name__ == "__main__":
    # Add utm coordinates to all
    boundary = helpers.all_latlon_to_utm(flyZones["boundaryPoints"])
    obstacles = helpers.all_latlon_to_utm(obstacles)
    waypoints = helpers.all_latlon_to_utm(waypoints)

    boundary_shape = helpers.coords_to_shape(boundary)
    obstacle_shapes = helpers.circles_to_shape(obstacles, OBSTACLE_BUFFER)
    waypoints_points = helpers.coords_to_points(waypoints)

    # Find new safe path between all waypoints
    start = waypoints_points[0]
    goal = waypoints_points[1]
    G = rrt.rrt(start, goal, boundary_shape, obstacle_shapes)

    if G.success:
        path = rrt.dijkstra(G)
        print(path)
        plotter.plot(G, obstacles, boundary, start, goal, path)
    else:
        plotter.plot(G, obstacles, boundary, start, goal)
