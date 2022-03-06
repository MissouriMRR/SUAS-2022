import obstacle_avoidance
import plotter

SAFETY_MARGIN = 10  # meters
MAX_DISTANCE = 0  # meters

waypoints = [
    {"latitude": 37.94883876837826, "longitude": -91.78443677093598, "altitude": 100.0},
    {
        "latitude": 37.949016435151485,
        "longitude": -91.78364283711778,
        "altitude": 150.0,
    },
    {"latitude": 37.94957481364174, "longitude": -91.78369648129468, "altitude": 300.0},
    {"latitude": 37.95004012580848, "longitude": -91.78468353414975, "altitude": 250.0},
    {
        "latitude": 37.949735558177984,
        "longitude": -91.78542382379104,
        "altitude": 150.0,
    },
    {
        "latitude": 37.948790418686826,
        "longitude": -91.78605823636349,
        "altitude": 200.0,
    },
    {
        "latitude": 37.948576497594445,
        "longitude": -91.78460843230208,
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
        "radius": 30,
        "height": 200,
    },
    {
        "latitude": 37.94828884469052,
        "longitude": -91.78546673913256,
        "radius": 50,
        "height": 150,
    },
]

if __name__ == "__main__":
    # Convert latlon data to UTM projection
    waypoints = obstacle_avoidance.all_latlon_to_utm(waypoints)
    obstacles = obstacle_avoidance.all_latlon_to_utm(obstacles)

    # Plot data before processing
    # plotter.plot_data(waypoints, obstacles, SAFETY_MARGIN)

    # Find new safe path between all waypoints
    waypoints = obstacle_avoidance.get_safe_route(
        waypoints, obstacles, SAFETY_MARGIN, MAX_DISTANCE, debugging=False
    )

    # Plot data after processing
    plotter.plot_data(waypoints, obstacles, SAFETY_MARGIN, flight_path_color="bo-")
