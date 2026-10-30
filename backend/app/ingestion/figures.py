from pathlib import Path

import pymupdf

MIN_FIGURE_SIZE = 100  # points : on ignore les petits traits décoratifs
PADDING = 8            # marge ajoutée autour de la figure (pour les légendes)


def extract_vector_figures(page, page_number, images_dir, table_rects, base_meta):
    figures = []
    page_area = page.rect.get_area()

    for index, rect in enumerate(page.cluster_drawings(), start=1):
        if rect.width < MIN_FIGURE_SIZE or rect.height < MIN_FIGURE_SIZE:
            continue
        if rect.get_area() > 0.9 * page_area:           # cadre de page entière
            continue
        if any((rect & t).get_area() > 0.5 * rect.get_area() for t in table_rects):
            continue                                    # ce sont les traits d'un tableau

        clip = pymupdf.Rect(
            rect.x0 - PADDING, rect.y0 - PADDING, rect.x1 + PADDING, rect.y1 + PADDING
        ) & page.rect
        pix = page.get_pixmap(clip=clip, dpi=150)

        path = Path(images_dir) / f"p{page_number}_fig{index}.png"
        pix.save(path)
        figures.append(
            {**base_meta, "type": "image", "file_path": str(path),
             "width": pix.width, "height": pix.height}
        )
    return figures