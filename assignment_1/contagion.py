import cv2 as cv
import numpy as np
import time

from helpers import (get_eligible_cells, get_cell_bounds, find_motion_cells)


GRID_SIZE = 5
MOTION_THRESHOLD = 25
MIN_ACTIVE_AREA_RATIO = 0.01
RESET_INTERVAL = 10.0
BACKGROUND_UPDATE_RATE = 0.02
BLUR_KERNEL_SIZE = (5, 5)
MORPH_KERNEL_SIZE = (5, 5)


cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

ret, frame = cap.read()

if not ret:
    print("Error: Could not read first frame.")
    cap.release()
    exit()


# Get dimensions
height, width = frame.shape[:2]

print(f"Camera resolution: {width} x {height}")


gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

gray = cv.GaussianBlur(
    gray,
    BLUR_KERNEL_SIZE,
    0
)

# We keep the background as float32 because
# cv.accumulateWeighted() requires it.
background = gray.astype(np.float32)

# Set containing active cell coordinates.
active_cells = set()

# Time at which significant motion was last detected.
last_motion_time = time.time()

# Used for FPS calculation.
previous_time = time.time()

fps = 0.0


# ============================================================
# HELPER FUNCTION:
# GET CELL BOUNDARIES
# ============================================================

# def get_cell_bounds(row, col, width, height):
#     """
#     Return the pixel boundaries of a grid cell.
#     """

#     cell_width = width // GRID_SIZE
#     cell_height = height // GRID_SIZE

#     x1 = col * cell_width
#     y1 = row * cell_height

#     # Make the final column/row reach the actual
#     # image boundary in case width/height isn't
#     # perfectly divisible by 5.
#     if col == GRID_SIZE - 1:
#         x2 = width
#     else:
#         x2 = (col + 1) * cell_width

#     if row == GRID_SIZE - 1:
#         y2 = height
#     else:
#         y2 = (row + 1) * cell_height

#     return x1, y1, x2, y2


# ============================================================
# HELPER FUNCTION:
# FIND MOTION CELLS
# ============================================================

# def find_motion_cells(motion_mask, width, height):
#     """
#     Determine which of the 25 cells contain
#     enough significant motion.
#     """

#     motion_cells = set()

#     for row in range(GRID_SIZE):
#         for col in range(GRID_SIZE):

#             x1, y1, x2, y2 = get_cell_bounds(
#                 row,
#                 col,
#                 width,
#                 height
#             )

#             cell_mask = motion_mask[y1:y2, x1:x2]

#             # Number of pixels classified as motion.
#             motion_pixels = cv.countNonZero(cell_mask)

#             # Total number of pixels in this cell.
#             cell_area = cell_mask.shape[0] * cell_mask.shape[1]

#             # Percentage of cell containing motion.
#             motion_ratio = motion_pixels / cell_area

#             if motion_ratio >= MIN_ACTIVE_AREA_RATIO:
#                 motion_cells.add((row, col))

#     return motion_cells


# ============================================================
# HELPER FUNCTION:
# FIND ELIGIBLE NEIGHBOURS
# ============================================================

# def get_eligible_cells(active_cells):
#     """
#     Return cells immediately adjacent to active cells.

#     Diagonal neighbours are included.
#     """

#     eligible_cells = set()

#     directions = [
#         (-1, -1),
#         (-1,  0),
#         (-1,  1),

#         ( 0, -1),
#         ( 0,  1),

#         ( 1, -1),
#         ( 1,  0),
#         ( 1,  1)
#     ]

#     for row, col in active_cells:

#         for dr, dc in directions:

#             new_row = row + dr
#             new_col = col + dc

#             # Make sure neighbour is inside 5x5 grid.
#             if (
#                 0 <= new_row < GRID_SIZE
#                 and
#                 0 <= new_col < GRID_SIZE
#             ):
#                 eligible_cells.add((new_row, new_col))

#     # A cell that is already active doesn't need
#     # to be considered eligible again.
#     eligible_cells -= active_cells

#     return eligible_cells


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read frame.")
        break


    # FPS calculation.
    current_time = time.time()

    elapsed = current_time - previous_time

    if elapsed > 0:
        fps = 1.0 / elapsed

    previous_time = current_time


    # PREPROCESS FRAME
    gray = cv.cvtColor(
        frame,
        cv.COLOR_BGR2GRAY
    )

    # Blur slightly to reduce tiny pixel fluctuations.
    gray = cv.GaussianBlur(
        gray,
        BLUR_KERNEL_SIZE,
        0
    )

    # Convert our floating-point background back to
    # an 8-bit image so that we can compare it with gray.
    background_uint8 = cv.convertScaleAbs(background)

    # Calculate absolute difference between current frame and background.
    difference = cv.absdiff(
        gray,
        background_uint8
    )


    # THRESHOLD
    _, motion_mask = cv.threshold(
        difference,
        MOTION_THRESHOLD,
        255,
        cv.THRESH_BINARY
    )


    # MORPHOLOGICAL FILTERING
    kernel = cv.getStructuringElement(
        cv.MORPH_ELLIPSE,
        MORPH_KERNEL_SIZE
    )

    # OPENING:
    # Removes small isolated white regions.
    motion_mask = cv.morphologyEx(
        motion_mask,
        cv.MORPH_OPEN,
        kernel
    )

    # CLOSING:
    # Connects nearby regions and fills small holes.
    motion_mask = cv.morphologyEx(
        motion_mask,
        cv.MORPH_CLOSE,
        kernel
    )


    # FIND CELLS WITH SIGNIFICANT MOTION
    motion_cells = find_motion_cells(
        motion_mask,
        width,
        height
    )


    # RESET TIMER
    if len(motion_cells) > 0:

        # We detected significant motion somewhere
        # in the frame.
        last_motion_time = current_time


    # CHECK 10-SECOND RESET
    time_since_motion = (
        current_time - last_motion_time
    )

    if time_since_motion >= RESET_INTERVAL:

        # Complete reset.
        active_cells.clear()

        # Restart the timer so that we don't repeatedly
        # execute the reset.
        last_motion_time = current_time


    # CONTAGION LOGIC
    if len(active_cells) == 0:

        # ----------------------------------------------------
        # NO ACTIVE REGION YET
        # ----------------------------------------------------
        #
        # Any cell with significant motion can become
        # the first seed.

        if len(motion_cells) > 0:

            # Choose one deterministic seed.
            #
            # We use sorted() so that the result is
            # predictable for testing.
            seed = sorted(motion_cells)[0]

            active_cells.add(seed)

    else:

        # ----------------------------------------------------
        # ACTIVE REGION ALREADY EXISTS
        # ----------------------------------------------------

        # Find cells touching the existing active region.
        eligible_cells = get_eligible_cells(
            active_cells
        )

        # Only eligible cells are allowed to activate.
        new_active_cells = (
            motion_cells
            &
            eligible_cells
        )

        active_cells.update(
            new_active_cells
        )


    # UPDATE BACKGROUND

    # Slowly adapt the background to gradual changes
    # in lighting and the environment.
    cv.accumulateWeighted(
        gray,
        background,
        BACKGROUND_UPDATE_RATE
    )


    # RENDER RESULT
    output = frame.copy()

    # Find currently eligible cells for display.
    if len(active_cells) > 0:
        eligible_cells = get_eligible_cells(
            active_cells
        )
    else:
        # Before a seed exists, every cell is eligible.
        eligible_cells = {
            (row, col)
            for row in range(GRID_SIZE)
            for col in range(GRID_SIZE)
        }


    # INVERT ACTIVE CELLS
    for row, col in active_cells:

        x1, y1, x2, y2 = get_cell_bounds(
            row,
            col,
            width,
            height
        )

        cell = output[y1:y2, x1:x2]

        # Negative / colour inversion.
        output[y1:y2, x1:x2] = cv.bitwise_not(
            cell
        )


    # DRAW GRID
    for row in range(GRID_SIZE):

        for col in range(GRID_SIZE):

            x1, y1, x2, y2 = get_cell_bounds(
                row,
                col,
                width,
                height
            )

            # Draw grid rectangle.
            cv.rectangle(
                output,
                (x1, y1),
                (x2 - 1, y2 - 1),
                (255, 255, 255),
                1
            )


            # ------------------------------------------------
            # CELL STATUS
            # ------------------------------------------------

            if (row, col) in active_cells:

                # Active cell.
                label = "ACTIVE"

                cv.putText(
                    output,
                    label,
                    (x1 + 5, y1 + 20),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1,
                    cv.LINE_AA
                )

            elif (row, col) in eligible_cells:

                # Cell can potentially become active.
                label = "E"

                cv.putText(
                    output,
                    label,
                    (x1 + 5, y1 + 20),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                    cv.LINE_AA
                )


    # RESET COUNTDOWN
    if len(active_cells) > 0:

        reset_remaining = max(
            0,
            RESET_INTERVAL - time_since_motion
        )

        reset_text = (
            f"Reset in: {reset_remaining:.1f}s"
        )

    else:

        reset_text = "Waiting for seed"


    # ========================================================
    # INFORMATION PANEL
    # ========================================================

    cv.putText(
        output,
        f"FPS: {fps:.1f}",
        (10, 25),
        cv.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv.LINE_AA
    )

    cv.putText(
        output,
        f"Active cells: {len(active_cells)}/25",
        (10, 50),
        cv.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv.LINE_AA
    )

    cv.putText(
        output,
        f"Motion cells: {len(motion_cells)}",
        (10, 73),
        cv.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv.LINE_AA
    )

    cv.putText(
        output,
        reset_text,
        (10, 96),
        cv.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv.LINE_AA
    )


    # ========================================================
    # DISPLAY

    cv.imshow(
        "Motion Contagion Grid",
        output
    )

    # Optional debugging window.
    # cv.imshow(
    #     "Motion Mask",
    #     motion_mask
    # )


    key = cv.waitKey(1) & 0xFF

    if key == 27:       # ESC
        break


cap.release()
cv.destroyAllWindows()
