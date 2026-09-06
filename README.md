# AI-Powered Live Object Counter

A computer vision project that uses your webcam to detect and count
objects live — dots, coins, buttons, seeds, or any small objects that
stand out clearly from their background. Each detected object gets
outlined with a circle and numbered, and the total count updates live
on screen.

## How It Works

This project uses classical computer vision (via OpenCV), not a
trained neural network — which makes it fast, lightweight, and
100% offline (no dataset, no training, no internet needed).

The pipeline:
1. **Grayscale conversion** — simplifies the image to light/dark values.
2. **Thresholding (Otsu's method)** — automatically separates objects
   from the background based on contrast.
3. **Noise cleanup (morphological operations)** — removes small specks
   of dust/noise and fills small gaps in detected shapes.
4. **Contour detection** — finds each separate connected blob; each
   blob is treated as one object.
5. **Filtering** — contours smaller than a minimum size are ignored
   (this is what the sensitivity controls adjust).
6. **Annotation** — each valid object gets a green circle and a number
   drawn around it, plus a live running total shown on screen.

## Project Structure

```
object_counter/
├── live_counter.py     # Main script — run this
├── requirements.txt
└── outputs/             # (created when you save test images, optional)
```

## How to Run

```
pip install -r requirements.txt
python live_counter.py
```

## Controls

| Key | Action |
|-----|--------|
| `+` | Increase sensitivity (detects smaller objects, but may pick up noise) |
| `-` | Decrease sensitivity (ignores smaller objects/noise) |
| `i` | Invert detection (switch between dark-objects-on-light-background and light-objects-on-dark-background) |
| `ESC` | Quit |

## Tips for Best Results

- Use a **plain, single-colored background** — a white sheet of paper
  works great for dark dots, coins, or buttons.
- Make sure objects **don't touch or overlap** — touching objects can
  be counted as a single blob. Spread them out.
- Use **even lighting** and avoid shadows falling across the objects.
- If it misses very small objects, press `+` to increase sensitivity.
- If it's picking up dust/noise as objects, press `-` to decrease sensitivity.

## Tested Behavior

This logic was tested with generated sample images before being
packaged:
- 12 clearly separated dots → correctly counted as 12/12.
- Objects of mixed sizes with one very small object near the
  sensitivity threshold → the small object was missed until
  sensitivity was increased. This is expected, adjustable behavior,
  not a bug — real-world objects vary in size, so the sensitivity
  controls exist for exactly this situation.

## Limitations & Possible Extensions

- This method counts **any distinct blob**, not specific object
  *types* — it can't tell the difference between "5 coins" and
  "5 buttons" if they're both counted together; it just sees 5 shapes.
- For recognizing *what* the objects are (not just counting them),
  a trained object-detection model (like YOLO) would be needed —
  a possible upgrade path if you want to extend this project further.

## Author

Sher Dil — BS Computer Science, Minhaj University Lahore

