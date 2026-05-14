from __future__ import annotations

import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from PIL import Image, ImageOps

from psl_workflow.processing.models import DocumentMetadata, IngestedDocument, PageText

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


class IngestionError(RuntimeError):
    """Raised when a document cannot be processed into structured text."""


class DocumentIngestor:
    """Extract text from legal PDFs/images with OCR fallback for scanned pages."""

    def __init__(self, min_text_chars: int = 60, ocr_language: str = "eng") -> None:
        self.min_text_chars = min_text_chars
        self.ocr_language = ocr_language

    def ingest(self, source_path: str | Path) -> IngestedDocument:
        path = Path(source_path)
        if not path.exists():
            raise IngestionError(f"Input document does not exist: {path}")

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._ingest_pdf(path)
        if suffix in IMAGE_EXTENSIONS:
            return self._ingest_image(path)
        raise IngestionError(f"Unsupported document type: {suffix or '<none>'}")

    def _ingest_pdf(self, path: Path) -> IngestedDocument:
        try:
            import fitz
        except ImportError as exc:  # pragma: no cover - dependency installed in normal setup
            raise IngestionError("PyMuPDF is required for PDF ingestion.") from exc

        try:
            doc = fitz.open(path)
        except Exception as exc:  # noqa: BLE001
            raise IngestionError(f"Unable to open PDF {path}: {exc}") from exc

        metadata = DocumentMetadata(
            source_path=str(path),
            document_type="pdf",
            page_count=len(doc),
            title=doc.metadata.get("title") or path.stem,
            author=doc.metadata.get("author") or None,
            created_at=doc.metadata.get("creationDate") or None,
            extra={"format": doc.metadata.get("format")},
        )
        pages: list[PageText] = []
        with TemporaryDirectory() as temp_dir:
            for index, page in enumerate(doc, start=1):
                text = self._normalize_text(page.get_text("text") or "")
                page_meta: dict[str, Any] = {
                    "width": page.rect.width,
                    "height": page.rect.height,
                    "source_kind": "embedded-text",
                }
                if len(text) >= self.min_text_chars:
                    pages.append(PageText(index, text, "pymupdf", page_meta))
                    continue

                pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                image_path = Path(temp_dir) / f"page-{index}.png"
                pixmap.save(image_path)
                ocr_text, ocr_meta = self._ocr_image(image_path)
                extraction_method = "ocr" if ocr_text.strip() else "ocr-unavailable"
                page_meta.update(ocr_meta)
                page_meta["source_kind"] = "rendered-page"
                pages.append(
                    PageText(
                        index,
                        self._normalize_text(ocr_text),
                        extraction_method,
                        page_meta,
                    )
                )

        return IngestedDocument(metadata=metadata, pages=pages)

    def _ingest_image(self, path: Path) -> IngestedDocument:
        ocr_text, ocr_meta = self._ocr_image(path)
        extraction_method = "ocr" if ocr_text.strip() else "ocr-unavailable"
        metadata = DocumentMetadata(
            source_path=str(path),
            document_type="image",
            page_count=1,
            title=path.stem,
            extra={"image_suffix": path.suffix.lower()},
        )
        return IngestedDocument(
            metadata=metadata,
            pages=[
                PageText(
                    page_number=1,
                    text=self._normalize_text(ocr_text),
                    extraction_method=extraction_method,
                    metadata=ocr_meta,
                )
            ],
        )

    def _ocr_image(self, image_path: Path) -> tuple[str, dict[str, Any]]:
        prepared = self._prepare_image_for_ocr(image_path)
        if shutil.which("tesseract"):
            try:
                import pytesseract

                text = pytesseract.image_to_string(prepared, lang=self.ocr_language)
                return text, {"ocr_engine": "tesseract", "ocr_language": self.ocr_language}
            except Exception as exc:  # noqa: BLE001
                return "", {"ocr_engine": "tesseract", "ocr_error": str(exc)}

        easyocr_result = self._try_easyocr(prepared)
        if easyocr_result is not None:
            return easyocr_result, {"ocr_engine": "easyocr", "ocr_language": self.ocr_language}

        return "", {
            "ocr_engine": "unavailable",
            "ocr_error": (
                "Install the tesseract binary or optional EasyOCR dependency for OCR text."
            ),
        }

    def _try_easyocr(self, image: Image.Image) -> str | None:
        try:
            import easyocr  # type: ignore[import-not-found]
        except ImportError:
            return None

        reader = easyocr.Reader([self.ocr_language[:2]], gpu=False)
        lines = reader.readtext(image, detail=0, paragraph=True)
        return "\n".join(str(line) for line in lines)

    @staticmethod
    def _prepare_image_for_ocr(image_path: Path) -> Image.Image:
        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image).convert("L")
        image = ImageOps.autocontrast(image)
        return image

    @staticmethod
    def _normalize_text(text: str) -> str:
        lines = [" ".join(line.split()) for line in text.replace("\x00", "").splitlines()]
        return "\n".join(line for line in lines if line).strip()
