import cv2 as cv

GRID_SIZE = 5
MOTION_THRESHOLD = 25
MIN_ACTIVE_AREA_RATIO = 0.01
RESET_INTERVAL = 10.0
BACKGROUND_UPDATE_RATE = 0.02
BLUR_KERNEL_SIZE = (5, 5)
MORPH_KERNEL_SIZE = (5, 5)


# Get the pixel boundaries of a grid cell
def get_cell_bounds(row, col, width, height):
    """
    Return the pixel boundaries of a grid cell.
    """

    cell_width = width // GRID_SIZE
    cell_height = height // GRID_SIZE

    x1 = col * cell_width
    y1 = row * cell_height

    # Make the final column/row reach the actual
    # image boundary in case width/height isn't
    # perfectly divisible by 5.
    if col == GRID_SIZE - 1:
        x2 = width
    else:
        x2 = (col + 1) * cell_width

    if row == GRID_SIZE - 1:
        y2 = height
    else:
        y2 = (row + 1) * cell_height

    return x1, y1, x2, y2


# Find cells containing significant motion
def find_motion_cells(motion_mask, width, height):
    """
    Determine which of the 25 cells contain
    enough significant motion.
    """

    motion_cells = set()

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):

            x1, y1, x2, y2 = get_cell_bounds(
                row,
                col,
                width,
                height
            )

            cell_mask = motion_mask[y1:y2, x1:x2]

            # Number of pixels classified as motion.
            motion_pixels = cv.countNonZero(cell_mask)

            # Total number of pixels in this cell.
            cell_area = cell_mask.shape[0] * cell_mask.shape[1]

            # Percentage of cell containing motion.
            motion_ratio = motion_pixels / cell_area

            if motion_ratio >= MIN_ACTIVE_AREA_RATIO:
                motion_cells.add((row, col))

    return motion_cells


# Get cells immediately adjacent to active cells
def get_eligible_cells(active_cells):
    """
    Return cells immediately adjacent to active cells.

    Diagonal neighbours are included.
    """

    eligible_cells = set()

    directions = [
        (-1, -1),
        (-1,  0),
        (-1,  1),

        ( 0, -1),
        ( 0,  1),

        ( 1, -1),
        ( 1,  0),
        ( 1,  1)
    ]

    for row, col in active_cells:

        for dr, dc in directions:

            new_row = row + dr
            new_col = col + dc

            # Make sure neighbour is inside 5x5 grid.
            if (
                0 <= new_row < GRID_SIZE
                and
                0 <= new_col < GRID_SIZE
            ):
                eligible_cells.add((new_row, new_col))

    # A cell that is already active doesn't need
    # to be considered eligible again.
    eligible_cells -= active_cells

    return eligible_cells
