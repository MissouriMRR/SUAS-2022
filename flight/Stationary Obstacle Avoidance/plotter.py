import matplotlib.pyplot as plt


def plot_data(waypoints, obstacles, padding, flight_path_color="bo-"):
    # plot obstacles
    for obstacle in obstacles:
        x = obstacle["utm_x"]
        y = obstacle["utm_y"]
        radius = obstacle["radius"]

        plt.gca().add_patch(
            plt.Circle((x, y), radius + padding, color="gray")
        )  # padded obstacle
        plt.gca().add_patch(plt.Circle((x, y), radius, color="red"))  # true obstacle

    # plot waypoints
    x, y = [], []
    for waypoint in waypoints:
        x.append(waypoint["utm_x"])
        y.append(waypoint["utm_y"])
    plt.plot(x, y, flight_path_color)

    plt.gca().set_aspect(1)
    plt.show()


def plot_debug(
    pointA,
    pointB,
    obstacle,
    tangentA1,
    tangentA2,
    tangentB1,
    tangentB2,
    new_point,
    alt_point,
    temp_point,
    extra_1,
    extra_2,
    special_case,
    padding,
):
    radius = obstacle[2]

    # plot obstacle
    plt.gca().add_patch(
        plt.Circle((obstacle[0], obstacle[1]), radius + padding, color="gray")
    )  # padded obstacle
    plt.gca().add_patch(
        plt.Circle((obstacle[0], obstacle[1]), radius, color="red")
    )  # true obstacle

    # prepare point data
    x = [
        pointA[0],
        pointB[0],
        obstacle[0],
        tangentA1[0],
        tangentA2[0],
        tangentB1[0],
        tangentB2[0],
        new_point[0],
        alt_point[0],
        temp_point[0],
        extra_1[0],
        extra_2[0],
    ]
    y = [
        pointA[1],
        pointB[1],
        obstacle[1],
        tangentA1[1],
        tangentA2[1],
        tangentB1[1],
        tangentB2[1],
        new_point[1],
        alt_point[1],
        temp_point[1],
        extra_1[1],
        extra_2[1],
    ]
    col = [
        "green",
        "green",
        "black",
        "black",
        "black",
        "black",
        "black",
        "blue",
        "blue",
        "blue",
        "blue",
        "blue",
    ]
    txt = [
        "Start",
        "End",
        "Obstacle",
        "A1",
        "A2",
        "B1",
        "B2",
        "New Waypoint",
        "Alt Waypoint",
        "Temp",
        "Extra 1",
        "Extra 2",
    ]

    # plot points
    for i in range(len(x)):
        plt.scatter(x[i], y[i], c=col[i])

    # plot text labels
    for i, v in enumerate(txt):
        plt.gca().annotate(v, (x[i], y[i]))

    if special_case:
        plt.plot(
            [pointA[0], extra_1[0], extra_2[0], pointB[0]],
            [pointA[1], extra_1[1], extra_2[1], pointB[1]],
            "go-",
        )
    else:
        plt.plot(
            [pointA[0], new_point[0], pointB[0]],
            [pointA[1], new_point[1], pointB[1]],
            "go-",
        )

    plt.gca().set_aspect(1)
    plt.show()
