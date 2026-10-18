"""Extract figure assets from assignment PDFs.

For each PDF: saves raster images (deduped, size-filtered) and renders
vector-drawing clusters as PNGs, then writes manifest.json grouped by page.
"""
import sys, os, json
import fitz

MIN_RASTER = 150      # min width/height in px for embedded raster images
MIN_VEC_W = 80        # min width in pt for vector clusters
MIN_VEC_H = 50        # min height in pt for vector clusters


def extract(pdf_path: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    manifest = {}
    seen_xrefs = set()
    fig_count = 0

    for pno in range(len(doc)):
        page = doc[pno]
        entries = []

        # --- raster images embedded in the page ---
        for img in page.get_images(full=True):
            xref = img[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)
            try:
                pix = fitz.Pixmap(doc, xref)
            except Exception:
                continue
            if pix.width < MIN_RASTER or pix.height < MIN_RASTER:
                continue
            if pix.colorspace is None:
                continue
            if pix.n - pix.alpha > 3:  # CMYK etc -> RGB
                pix = fitz.Pixmap(fitz.csRGB, pix)
            fig_count += 1
            name = f"p{pno+1:02d}_r{fig_count:02d}.png"
            pix.save(os.path.join(out_dir, name))
            entries.append({"file": name, "kind": "raster", "px": [pix.width, pix.height]})

        # --- vector drawing clusters (TikZ-style figures) ---
        try:
            try:
                clusters = page.cluster_drawings(x_tolerance=15, y_tolerance=15)
            except TypeError:
                clusters = page.cluster_drawings()
        except Exception:
            clusters = []
        for rect in clusters:
            if rect.width < MIN_VEC_W or rect.height < MIN_VEC_H:
                continue
            if rect.width > page.rect.width + 5 or rect.height > page.rect.height + 5:
                continue
            fig_count += 1
            name = f"p{pno+1:02d}_v{fig_count:02d}.png"
            pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), clip=rect)
            pix.save(os.path.join(out_dir, name))
            entries.append({
                "file": name, "kind": "vector",
                "px": [pix.width, pix.height],
                "bbox_pt": [round(v, 1) for v in rect],
            })

        if entries:
            manifest[str(pno + 1)] = entries

    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    total = sum(len(v) for v in manifest.values())
    print(f"{os.path.basename(pdf_path)}: {total} assets on {len(manifest)} pages -> {out_dir}")


if __name__ == "__main__":
    extract(sys.argv[1], sys.argv[2])
