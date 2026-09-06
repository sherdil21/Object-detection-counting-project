import cv2
import numpy as np

MIN_AREA_DEFAULT = 80  # minimum contour area (in pixels) to count as an object
MAX_AREA_FRACTION = 0.25  # ignore contours bigger than 25% of the frame (background/edges, not real objects)


def detect_objects(frame, min_area, invert):
    """
    Returns (annotated_frame, count) — the frame with detected objects
    outlined and numbered, and the total count of valid objects found.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh_type = cv2.THRESH_BINARY if invert else cv2.THRESH_BINARY_INV
    _, thresh = cv2.threshold(blurred, 0, 255, thresh_type + cv2.THRESH_OTSU)

    # A smaller kernel avoids erasing genuinely small objects (like pen dots)
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    # RETR_LIST finds every contour regardless of nesting level (e.g. a
    # dot sitting inside a paper-shaped "hole" within the background) —
    # RETR_EXTERNAL would have missed such nested objects entirely.
    contours, _ = cv2.findContours(cleaned, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    frame_area = frame.shape[0] * frame.shape[1]
    max_area = frame_area * MAX_AREA_FRACTION

    # Keep only contours that are big enough to be a real object, but not
    # so big that they're actually the background/table edge/shadow.
    valid_contours = [
        c for c in contours
        if min_area < cv2.contourArea(c) < max_area
    ]

    annotated = frame.copy()
    for i, c in enumerate(valid_contours, start=1):
        (x, y), radius = cv2.minEnclosingCircle(c)
        center = (int(x), int(y))
        cv2.circle(annotated, center, int(radius) + 6, (0, 255, 0), 2)
        cv2.putText(
            annotated, str(i), (center[0] - 8, center[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
        )

    return annotated, len(valid_contours)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not access the webcam. Check your camera connection/permissions.")

    min_area = MIN_AREA_DEFAULT
    invert = False

    print("Live object counter started.")
    print("Show objects (dots, coins, buttons, etc.) on a plain background.")
    print("Controls: '+' more sensitive | '-' less sensitive | 'i' invert | ESC quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera.")
            break

        annotated, count = detect_objects(frame, min_area, invert)

        cv2.putText(
            annotated, f"Count: {count}", (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3
        )
        cv2.putText(
            annotated, f"Sensitivity (min area): {min_area}  |  Inverted: {invert}",
            (10, annotated.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

        cv2.imshow("AI Object Counter", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('+'):
            min_area = max(20, min_area - 50)  # lower area = more sensitive
        elif key == ord('-'):
            min_area += 50
        elif key == ord('i'):
            invert = not invert

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
