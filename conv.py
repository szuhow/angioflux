from pathlib import Path
import numpy as np
from PIL import Image
import argparse


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """
    Normalizuje dane do zakresu 0-255 i konwertuje do uint8.
    """
    arr = arr.astype(np.float32)

    min_val = arr.min()
    max_val = arr.max()

    if max_val - min_val == 0:
        return np.zeros_like(arr, dtype=np.uint8)

    arr = (arr - min_val) / (max_val - min_val)
    arr = (arr * 255).clip(0, 255)

    return arr.astype(np.uint8)


def convert_npy_to_png(npy_path: Path, output_path: Path):
    arr = np.load(npy_path)

    # Obsługa różnych wymiarów
    if arr.ndim == 2:
        img = normalize_to_uint8(arr)

    elif arr.ndim == 3:
        # Możliwe formaty:
        # HWC albo CHW
        if arr.shape[0] in [1, 3]:
            # CHW -> HWC
            arr = np.transpose(arr, (1, 2, 0))

        if arr.shape[-1] == 1:
            arr = arr.squeeze(-1)

        img = normalize_to_uint8(arr)

    else:
        raise ValueError(f"Nieobsługiwany wymiar tablicy: {arr.shape}")

    Image.fromarray(img).save(output_path)
    print(f"Zapisano: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Konwerter NPY -> PNG")
    parser.add_argument("input", help="Plik .npy lub katalog")
    parser.add_argument(
        "--output-dir",
        default="output_png",
        help="Katalog wyjściowy"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        output_path = output_dir / f"{input_path.stem}.png"
        convert_npy_to_png(input_path, output_path)

    elif input_path.is_dir():
        for npy_file in input_path.glob("*.npy"):
            output_path = output_dir / f"{npy_file.stem}.png"
            convert_npy_to_png(npy_file, output_path)

    else:
        raise FileNotFoundError(input_path)


if __name__ == "__main__":
    main()