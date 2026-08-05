from __future__ import annotations

import json
import math
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any, ClassVar

import numpy as np
from PIL import Image, ImageTk

from .analysis import (
    classify_fibers,
    fiber_level_summary,
    fiber_statistics,
    format_length_m,
    one_click_measurement,
    section_level_summary,
    snap_two_click_edges,
    validate_measurement_geometry,
)
from .exporters import export_annotated, export_csv, export_html_report
from .model import Measurement, Project
from .project_io import (
    SourceVerificationStatus,
    load_project,
    save_project,
    verify_project_source,
)
from .zeiss import file_sha256, load_image_document, load_pixels


class FiberQuickApp(tk.Tk):
    GROUP_HEX: ClassVar[list[str]] = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
    SELECTED = "#FFBF00"


    def __init__(self, initial_path: str | None = None) -> None:
        super().__init__()
        self.title("Fathom Fibers Quick 0.1 — Manual-first SEM measurement")
        self.geometry("1500x900")
        self.minsize(1100, 700)

        self.project: Project | None = None
        self.source_image: Image.Image | None = None
        self.gray: np.ndarray | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self._has_fit = False
        self._is_dirty = False
        self.pending_point: tuple[float, float] | None = None
        self.pending_canvas_item: int | None = None
        self.selected_measurement_id: str | None = None
        self.drag_endpoint: int | None = None
        self.dragging_measurement_id: str | None = None
        self.pan_anchor: tuple[float, float] | None = None
        self.pan_offset_anchor: tuple[float, float] | None = None

        self.tool_var = tk.StringVar(value="select")
        self.fiber_var = tk.StringVar(value="F001")
        self.defect_var = tk.StringVar(value="None")
        self.search_radius_var = tk.DoubleVar(value=60.0)
        self.classification_var = tk.StringVar(value="Auto")
        self.footer_visible_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Abre un TIFF Zeiss para comenzar.")
        self.calibration_var = tk.StringVar(value="Sin imagen")
        self.selected_info_var = tk.StringVar(value="Sin selección")

        self._build_ui()
        self._bind_shortcuts()
        self.protocol("WM_DELETE_WINDOW", self._on_app_closing)
        self.after(100, lambda: self.open_path(initial_path) if initial_path else None)

    # ---------- dirty state & title ----------

    def _update_title(self) -> None:
        name = Path(self.project.image.path).name if self.project else ""
        dirty = " *" if self._is_dirty else ""
        if name:
            self.title(f"Fathom Fibers Quick — {name}{dirty}")
        else:
            self.title(f"Fathom Fibers Quick 0.1 — Manual-first SEM measurement{dirty}")

    def _mark_dirty(self) -> None:
        self._is_dirty = True
        self._update_title()

    def _clear_dirty(self) -> None:
        self._is_dirty = False
        self._update_title()

    def _confirm_discard_changes(self) -> bool:
        if not self._is_dirty:
            return True
        answer = messagebox.askyesnocancel(
            "Cambios no guardados",
            "El proyecto actual tiene cambios no guardados. ¿Deseas descartarlos?",
            parent=self,
        )
        return answer is True

    def _on_app_closing(self) -> None:
        if self._confirm_discard_changes():
            self.destroy()

    # ---------- UI construction ----------

    def _build_ui(self) -> None:
        self._build_menu()
        root = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        root.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(root, padding=8)
        center = ttk.Frame(root)
        right = ttk.Frame(root, padding=6)
        root.add(left, weight=0)
        root.add(center, weight=1)
        root.add(right, weight=0)

        self._build_left_panel(left)
        self._build_canvas(center)
        self._build_right_panel(right)

        status = ttk.Label(self, textvariable=self.status_var, anchor=tk.W, padding=(8, 4))
        status.pack(fill=tk.X, side=tk.BOTTOM)

    def _build_menu(self) -> None:
        menu = tk.Menu(self)
        file_menu = tk.Menu(menu, tearoff=False)
        file_menu.add_command(label="Abrir imagen…", command=self.open_image_dialog, accelerator="Ctrl+O")
        file_menu.add_command(label="Abrir proyecto…", command=self.open_project_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Guardar proyecto…", command=self.save_project_dialog, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="Exportar CSV…", command=self.export_csv_dialog)
        file_menu.add_command(label="Exportar imagen anotada…", command=self.export_annotated_dialog)
        file_menu.add_command(label="Exportar informe HTML…", command=self.export_report_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Salir", command=self._on_app_closing)
        menu.add_cascade(label="Archivo", menu=file_menu)

        help_menu = tk.Menu(menu, tearoff=False)
        help_menu.add_command(label="Protocolo rápido", command=self.show_protocol)
        help_menu.add_command(label="Acerca de", command=self.show_about)
        menu.add_cascade(label="Ayuda", menu=help_menu)
        self.config(menu=menu)

    def _build_left_panel(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="FATHOM FIBERS", font=("TkDefaultFont", 14, "bold")).pack(anchor=tk.W)
        ttk.Label(parent, text="MVP manual y asistido", foreground="#666").pack(anchor=tk.W, pady=(0, 8))

        file_box = ttk.LabelFrame(parent, text="Proyecto", padding=6)
        file_box.pack(fill=tk.X, pady=4)
        ttk.Button(file_box, text="Abrir TIFF / imagen", command=self.open_image_dialog).pack(fill=tk.X, pady=2)
        ttk.Button(file_box, text="Guardar proyecto", command=self.save_project_dialog).pack(fill=tk.X, pady=2)
        ttk.Label(file_box, textvariable=self.calibration_var, wraplength=225).pack(anchor=tk.W, pady=(5, 0))

        fiber_box = ttk.LabelFrame(parent, text="Fibra actual", padding=6)
        fiber_box.pack(fill=tk.X, pady=4)
        ttk.Entry(fiber_box, textvariable=self.fiber_var).pack(fill=tk.X)
        ttk.Button(fiber_box, text="Nueva fibra", command=self.new_fiber).pack(fill=tk.X, pady=(4, 0))

        tools = ttk.LabelFrame(parent, text="Herramienta", padding=6)
        tools.pack(fill=tk.X, pady=4)
        choices = [
            ("Seleccionar / editar (V)", "select"),
            ("Manual 2 clics (M)", "manual"),
            ("Ajustar bordes 2 clics (S)", "snap"),
            ("Propuesta local 1 clic (A)", "auto"),
        ]
        for label, value in choices:
            ttk.Radiobutton(tools, text=label, variable=self.tool_var, value=value, command=self._clear_pending).pack(anchor=tk.W)
        radius_row = ttk.Frame(tools)
        radius_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(radius_row, text="Radio auto (px):").pack(side=tk.LEFT)
        ttk.Spinbox(radius_row, from_=10, to=300, increment=5, textvariable=self.search_radius_var, width=7).pack(side=tk.RIGHT)

        review = ttk.LabelFrame(parent, text="Revisión", padding=6)
        review.pack(fill=tk.X, pady=4)
        ttk.Label(review, text="Defecto / observación:").pack(anchor=tk.W)
        defect_combo = ttk.Combobox(
            review,
            textvariable=self.defect_var,
            state="readonly",
            values=("None", "Bead", "Constriction", "Fused", "Debris", "Ribbon-like", "Ambiguous", "Other"),
        )
        defect_combo.pack(fill=tk.X, pady=2)
        ttk.Button(review, text="Aplicar a medición", command=self.apply_defect).pack(fill=tk.X, pady=2)
        ttk.Button(review, text="Aceptar / rechazar", command=self.toggle_selected_acceptance).pack(fill=tk.X, pady=2)
        ttk.Button(review, text="Eliminar seleccionada", command=self.delete_selected).pack(fill=tk.X, pady=2)

        classify = ttk.LabelFrame(parent, text="Familias de tamaño", padding=6)
        classify.pack(fill=tk.X, pady=4)
        ttk.Combobox(
            classify,
            textvariable=self.classification_var,
            state="readonly",
            values=("Auto", "1", "2", "3", "4"),
            width=8,
        ).pack(fill=tk.X)
        ttk.Button(classify, text="Clasificar fibras", command=self.classify).pack(fill=tk.X, pady=(4, 0))
        ttk.Label(classify, text="Usa la mediana de cada fibra; el color no prueba una familia física.", wraplength=225, foreground="#666").pack(anchor=tk.W, pady=(4, 0))

        display = ttk.LabelFrame(parent, text="Vista", padding=6)
        display.pack(fill=tk.X, pady=4)
        ttk.Checkbutton(display, text="Mostrar zona del footer", variable=self.footer_visible_var, command=self.render).pack(anchor=tk.W)
        ttk.Button(display, text="Ajustar imagen", command=self.fit_image).pack(fill=tk.X, pady=2)

        ttk.Label(parent, textvariable=self.selected_info_var, wraplength=230).pack(anchor=tk.W, pady=8)

    def _build_canvas(self, parent: ttk.Frame) -> None:
        self.canvas = tk.Canvas(parent, background="#161616", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind("<ButtonPress-1>", self._on_left_press)
        self.canvas.bind("<B1-Motion>", self._on_left_motion)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_release)
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<ButtonPress-3>", self._start_pan)
        self.canvas.bind("<B3-Motion>", self._pan_motion)
        self.canvas.bind("<ButtonRelease-3>", self._end_pan)
        self.canvas.bind("<MouseWheel>", self._wheel_zoom)
        self.canvas.bind("<Button-4>", lambda event: self._zoom_at(event.x, event.y, 1.2))
        self.canvas.bind("<Button-5>", lambda event: self._zoom_at(event.x, event.y, 1 / 1.2))

    def _build_right_panel(self, parent: ttk.Frame) -> None:
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)

        measurement_tab = ttk.Frame(notebook, padding=4)
        stats_tab = ttk.Frame(notebook, padding=6)
        metadata_tab = ttk.Frame(notebook, padding=6)
        notebook.add(measurement_tab, text="Mediciones")
        notebook.add(stats_tab, text="Estadística")
        notebook.add(metadata_tab, text="Metadata")

        columns = ("id", "fiber", "width", "method", "group", "defect", "ok")
        self.tree = ttk.Treeview(measurement_tab, columns=columns, show="headings", height=28)
        labels = {
            "id": "ID",
            "fiber": "Fibra",
            "width": "Ancho proyectado",
            "method": "Método",
            "group": "Grupo",
            "defect": "Defecto",
            "ok": "OK",
        }
        widths = {"id": 62, "fiber": 65, "width": 105, "method": 94, "group": 52, "defect": 88, "ok": 35}
        for column in columns:
            self.tree.heading(column, text=labels[column])
            self.tree.column(column, width=widths[column], anchor=tk.CENTER, stretch=column in {"method", "defect"})
        scroll = ttk.Scrollbar(measurement_tab, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.bind("<<TreeviewSelect>>", self._tree_selection)
        self.tree.bind("<Double-1>", lambda _event: self.tool_var.set("select"))

        self.stats_text = tk.Text(stats_tab, width=44, height=40, wrap=tk.WORD, state=tk.DISABLED)
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        self.metadata_text = tk.Text(metadata_tab, width=44, height=40, wrap=tk.NONE, state=tk.DISABLED)
        self.metadata_text.pack(fill=tk.BOTH, expand=True)

    def _bind_shortcuts(self) -> None:
        self.bind_all("<Control-o>", lambda _event: self.open_image_dialog())
        self.bind_all("<Control-s>", lambda _event: self.save_project_dialog())
        self.bind_all("<Key-v>", lambda _event: self.tool_var.set("select"))
        self.bind_all("<Key-m>", lambda _event: self.tool_var.set("manual"))
        self.bind_all("<Key-s>", lambda _event: self.tool_var.set("snap"))
        self.bind_all("<Key-a>", lambda _event: self.tool_var.set("auto"))
        self.bind_all("<Delete>", lambda _event: self.delete_selected())
        self.bind_all("<Escape>", lambda _event: self._clear_pending())

    # ---------- loading and saving ----------

    def open_image_dialog(self) -> None:
        if not self._confirm_discard_changes():
            return
        path = filedialog.askopenfilename(
            title="Abrir micrografía",
            filetypes=[("Micrografías", "*.tif *.tiff *.png *.jpg *.jpeg *.bmp"), ("Todos", "*.*")],
        )
        if path:
            self.open_path(path)

    def open_path(self, path: str | None) -> None:
        if not path:
            return
        try:
            try:
                document, source_image, gray = load_image_document(path)
            except ValueError as exc:
                if "calibration" not in str(exc).lower():
                    raise
                nm_per_px = simpledialog.askfloat(
                    "Calibración requerida",
                    "La imagen no contiene calibración Zeiss compatible. Ingresa nm por píxel:",
                    minvalue=1e-6,
                    parent=self,
                )
                if nm_per_px is None:
                    return
                document, source_image, gray = load_image_document(path, manual_pixel_size_m=nm_per_px * 1e-9)
            self.project = Project(schema_version=1, image=document)
            self.source_image = source_image
            self.gray = gray
            self.selected_measurement_id = None
            self.pending_point = None
            self._has_fit = False
            self._clear_dirty()
            self.calibration_var.set(
                f"{document.calibration.pixel_size_x_m * 1e9:.4f} nm/px\n"
                f"Fuente: {document.calibration.source}\n"
                f"Footer: {document.footer_bounds or 'no detectado'}"
            )
            self._refresh_all()
            self.after_idle(self.fit_image)
            self.status_var.set("Imagen cargada. Mide perpendicularmente al eje de la fibra.")
        except Exception as exc:
            messagebox.showerror("No se pudo abrir", str(exc), parent=self)

    def open_project_dialog(self) -> None:
        if not self._confirm_discard_changes():
            return
        path = filedialog.askopenfilename(filetypes=[("FiberQuick project", "*.fiberquick.json"), ("JSON", "*.json")])
        if not path:
            return
        try:
            project = load_project(path)
            verification = verify_project_source(project)

            if verification.status == SourceVerificationStatus.MISSING:
                answer = messagebox.askyesno(
                    "Imagen fuente no encontrada",
                    f"{verification.message}\n\n¿Deseas localizar la imagen fuente?",
                    parent=self,
                )
                if not answer:
                    return
                replacement = filedialog.askopenfilename(title="Localiza la imagen original", parent=self)
                if not replacement:
                    return
                new_path = str(Path(replacement).resolve())
                new_hash = file_sha256(Path(new_path))
                if verification.expected_sha256 and new_hash != verification.expected_sha256:
                    confirm = messagebox.askyesno(
                        "El hash no coincide",
                        f"La imagen seleccionada no coincide con el hash original del proyecto.\n\n"
                        f"Esperado: {verification.expected_sha256}\nObtenido: {new_hash}\n\n"
                        "¿Deseas abrir esta imagen de todos modos bajo tu responsabilidad?",
                        parent=self,
                    )
                    if not confirm:
                        return
                project.image.path = new_path

            elif verification.status == SourceVerificationStatus.MISMATCH:
                choice = messagebox.askyesnocancel(
                    "Advertencia de hash de imagen",
                    f"¡ADVERTENCIA: La imagen fuente no coincide con el hash de referencia del proyecto!\n\n"
                    f"Hash esperado: {verification.expected_sha256}\n"
                    f"Hash obtenido:  {verification.actual_sha256}\n\n"
                    "• Haz clic en SÍ para abrir la imagen explícitamente bajo tu responsabilidad.\n"
                    "• Haz clic en NO para localizar otra imagen en el disco.\n"
                    "• Haz clic en CANCELAR para abortar la apertura.",
                    parent=self,
                )
                if choice is None:
                    return
                if choice is False:
                    replacement = filedialog.askopenfilename(title="Localiza la imagen original", parent=self)
                    if not replacement:
                        return
                    new_path = str(Path(replacement).resolve())
                    new_hash = file_sha256(Path(new_path))
                    if verification.expected_sha256 and new_hash != verification.expected_sha256:
                        confirm = messagebox.askyesno(
                            "El hash no coincide",
                            f"La imagen elegida no coincide con el hash de referencia.\n\n"
                            f"Esperado: {verification.expected_sha256}\nObtenido: {new_hash}\n\n"
                            "¿Deseas usar esta imagen de todos modos bajo tu responsabilidad?",
                            parent=self,
                        )
                        if not confirm:
                            return
                    project.image.path = new_path

            source_image, gray = load_pixels(project.image.path)
            self.project = project
            self.source_image = source_image
            self.gray = gray
            self.selected_measurement_id = None
            self._has_fit = False
            self._clear_dirty()
            self.calibration_var.set(
                f"{project.image.calibration.pixel_size_x_m * 1e9:.4f} nm/px\n"
                f"Fuente: {project.image.calibration.source}\n"
                f"Footer: {project.image.footer_bounds or 'no detectado'}"
            )
            self._refresh_all()
            self.after_idle(self.fit_image)
            self.status_var.set(f"Proyecto abierto: {Path(path).name}")
        except Exception as exc:
            messagebox.showerror("No se pudo abrir el proyecto", str(exc), parent=self)

    def save_project_dialog(self) -> None:
        if self.project is None:
            return
        initial = self.project.project_path or f"{Path(self.project.image.path).stem}.fiberquick.json"
        path = filedialog.asksaveasfilename(
            title="Guardar proyecto",
            initialfile=Path(initial).name,
            defaultextension=".fiberquick.json",
            filetypes=[("FiberQuick project", "*.fiberquick.json")],
        )
        if path:
            try:
                saved = save_project(self.project, path)
                self._clear_dirty()
                self.status_var.set(f"Proyecto guardado: {saved}")
            except Exception as exc:
                messagebox.showerror("Error al guardar", str(exc), parent=self)

    # ---------- export ----------

    def export_csv_dialog(self) -> None:
        if self.project is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="diameter_measurements.csv")
        if path:
            export_csv(self.project, path)
            self.status_var.set(f"CSV exportado: {path}")

    def export_annotated_dialog(self) -> None:
        if self.project is None or self.source_image is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", initialfile="fibers_annotated.png")
        if path:
            export_annotated(self.project, self.source_image, path)
            self.status_var.set(f"Imagen anotada exportada: {path}")

    def export_report_dialog(self) -> None:
        if self.project is None or self.source_image is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".html", initialfile="fiber_report.html")
        if not path:
            return
        report_path = Path(path)
        annotated_path = report_path.with_name(report_path.stem + "_annotated.png")
        export_annotated(self.project, self.source_image, annotated_path)
        export_html_report(self.project, annotated_path.name, report_path)
        self.status_var.set(f"Informe exportado: {report_path}")

    # ---------- transforms and rendering ----------

    def _on_canvas_configure(self, _event: tk.Event[Any]) -> None:
        if self.source_image is not None:
            if not self._has_fit:
                self.fit_image()
            else:
                self.render()

    def fit_image(self) -> None:
        if self.source_image is None:
            return
        self.update_idletasks()
        cw = max(self.canvas.winfo_width(), 10)
        ch = max(self.canvas.winfo_height(), 10)
        width, height = self.source_image.size
        self.scale = min(cw / width, ch / height) * 0.98
        self.offset_x = (cw - width * self.scale) / 2
        self.offset_y = (ch - height * self.scale) / 2
        self._has_fit = True
        self.render()

    def image_to_canvas(self, point: tuple[float, float]) -> tuple[float, float]:
        return self.offset_x + point[0] * self.scale, self.offset_y + point[1] * self.scale

    def canvas_to_image(self, point: tuple[float, float]) -> tuple[float, float]:
        return (point[0] - self.offset_x) / self.scale, (point[1] - self.offset_y) / self.scale

    def _inside_image(self, point: tuple[float, float]) -> bool:
        if self.project is None:
            return False
        return 0 <= point[0] < self.project.image.width_px and 0 <= point[1] < self.project.image.height_px

    def _inside_footer(self, point: tuple[float, float]) -> bool:
        if self.project is None or self.project.image.footer_bounds is None:
            return False
        y0, y1 = self.project.image.footer_bounds
        return y0 <= point[1] <= y1

    def render(self) -> None:
        self.canvas.delete("all")
        if self.source_image is None or self.project is None:
            self.canvas.create_text(30, 30, anchor=tk.NW, fill="#ddd", text="Abre un TIFF Zeiss para comenzar.", font=("TkDefaultFont", 16))
            return
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        width, height = self.source_image.size
        ix0 = max(0.0, (0 - self.offset_x) / self.scale)
        iy0 = max(0.0, (0 - self.offset_y) / self.scale)
        ix1 = min(float(width), (cw - self.offset_x) / self.scale)
        iy1 = min(float(height), (ch - self.offset_y) / self.scale)
        if ix1 > ix0 and iy1 > iy0:
            crop_box = (math.floor(ix0), math.floor(iy0), math.ceil(ix1), math.ceil(iy1))
            crop = self.source_image.crop(crop_box)
            display_size = (
                max(1, round((crop_box[2] - crop_box[0]) * self.scale)),
                max(1, round((crop_box[3] - crop_box[1]) * self.scale)),
            )
            crop = crop.resize(display_size, Image.Resampling.BILINEAR)
            self.photo = ImageTk.PhotoImage(crop)
            cx, cy = self.image_to_canvas((crop_box[0], crop_box[1]))
            self.canvas.create_image(cx, cy, anchor=tk.NW, image=self.photo)

        if self.footer_visible_var.get() and self.project.image.footer_bounds:
            y0, y1 = self.project.image.footer_bounds
            x0c, y0c = self.image_to_canvas((0, y0))
            x1c, y1c = self.image_to_canvas((width, y1))
            self.canvas.create_rectangle(x0c, y0c, x1c, y1c, outline="#ff6b6b", width=2, fill="#555555", stipple="gray50")
            self.canvas.create_text(x0c + 8, y0c + 8, text="FOOTER EXCLUIDO", anchor=tk.NW, fill="#ffdfdf", font=("TkDefaultFont", 10, "bold"))

        self._draw_measurements()

    def _draw_measurements(self) -> None:
        if self.project is None:
            return
        selected = self._selected_measurement()
        selected_fiber = selected.fiber_id if selected else None
        extrema: dict[str, list[str]] = {}
        if selected_fiber:
            fiber_measurements = [m for m in self.project.measurements if m.fiber_id == selected_fiber and m.accepted]
            if fiber_measurements:
                ordered = sorted(fiber_measurements, key=lambda item: item.width_m)
                median_value = float(np.median([item.width_m for item in ordered]))
                median_item = min(ordered, key=lambda item: abs(item.width_m - median_value))
                for item, label in ((ordered[0], "MIN"), (median_item, "MED"), (ordered[-1], "MAX")):
                    extrema.setdefault(item.measurement_id, []).append(label)

        for measurement in self.project.measurements:
            p1c = self.image_to_canvas(measurement.p1)
            p2c = self.image_to_canvas(measurement.p2)
            if measurement.measurement_id == self.selected_measurement_id:
                color = self.SELECTED
                width = 4
            elif not measurement.accepted:
                color = "#888888"
                width = 2
            elif measurement.group is not None:
                color = self.GROUP_HEX[measurement.group % len(self.GROUP_HEX)]
                width = 3
            else:
                color = "#00D7FF"
                width = 3
            self.canvas.create_line(*p1c, *p2c, fill=color, width=width)
            radius = 5 if measurement.measurement_id == self.selected_measurement_id else 3
            for point in (p1c, p2c):
                self.canvas.create_oval(point[0]-radius, point[1]-radius, point[0]+radius, point[1]+radius, fill=color, outline="white")
            center = self.image_to_canvas(measurement.center)
            if measurement.measurement_id == self.selected_measurement_id or self.scale > 0.5:
                self.canvas.create_text(
                    center[0] + 7,
                    center[1] + 7,
                    text=f"{measurement.fiber_id} {measurement.width_m * 1e6:.3f} µm",
                    anchor=tk.NW,
                    fill="#fff6a8",
                    font=("TkDefaultFont", 9, "bold"),
                )
            if measurement.measurement_id in extrema:
                label = "/".join(extrema[measurement.measurement_id])
                self.canvas.create_text(center[0], center[1]-12, text=label, fill="#FFFFFF", font=("TkDefaultFont", 9, "bold"))

    # ---------- canvas interactions ----------

    def _wheel_zoom(self, event: tk.Event[Any]) -> None:
        factor = 1.2 if event.delta > 0 else 1 / 1.2
        self._zoom_at(event.x, event.y, factor)

    def _zoom_at(self, x: float, y: float, factor: float) -> None:
        if self.source_image is None:
            return
        image_point = self.canvas_to_image((x, y))
        self.scale = max(0.03, min(12.0, self.scale * factor))
        self.offset_x = x - image_point[0] * self.scale
        self.offset_y = y - image_point[1] * self.scale
        self.render()

    def _start_pan(self, event: tk.Event[Any]) -> None:
        self.pan_anchor = (event.x, event.y)
        self.pan_offset_anchor = (self.offset_x, self.offset_y)

    def _pan_motion(self, event: tk.Event[Any]) -> None:
        if self.pan_anchor is None or self.pan_offset_anchor is None:
            return
        self.offset_x = self.pan_offset_anchor[0] + event.x - self.pan_anchor[0]
        self.offset_y = self.pan_offset_anchor[1] + event.y - self.pan_anchor[1]
        self.render()

    def _end_pan(self, _event: tk.Event[Any]) -> None:
        self.pan_anchor = None
        self.pan_offset_anchor = None

    def _on_left_press(self, event: tk.Event[Any]) -> None:
        if self.project is None or self.gray is None:
            return
        point = self.canvas_to_image((event.x, event.y))
        if not self._inside_image(point):
            return
        tool = self.tool_var.get()
        if tool == "select":
            measurement, endpoint = self._hit_test(event.x, event.y)
            if measurement:
                self.select_measurement(measurement.measurement_id)
                if endpoint is not None:
                    self.drag_endpoint = endpoint
                    self.dragging_measurement_id = measurement.measurement_id
            else:
                self.select_measurement(None)
            return
        if self._inside_footer(point):
            self.status_var.set("La zona del footer está excluida. Elige una región de micrografía.")
            self.bell()
            return
        if tool == "auto":
            self._create_one_click(point)
            return
        if tool in {"manual", "snap"}:
            if self.pending_point is None:
                self.pending_point = point
                self.status_var.set("Primer borde fijado. Haz clic en el borde opuesto.")
            else:
                first = self.pending_point
                self.pending_point = None
                self._create_two_click(first, point, snap=(tool == "snap"))
                self.render()

    def _on_left_motion(self, event: tk.Event[Any]) -> None:
        if self.drag_endpoint is None or self.dragging_measurement_id is None or self.project is None:
            return
        measurement = self._measurement_by_id(self.dragging_measurement_id)
        if measurement is None:
            return
        point = self.canvas_to_image((event.x, event.y))
        point = (
            min(max(point[0], 0), self.project.image.width_px - 1),
            min(max(point[1], 0), self.project.image.height_px - 1),
        )
        if self.drag_endpoint == 0:
            measurement.p1 = point
        else:
            measurement.p2 = point
        measurement.width_m = self.project.image.calibration.distance_m(measurement.p1, measurement.p2)
        self.render()

    def _on_left_release(self, _event: tk.Event[Any]) -> None:
        if self.dragging_measurement_id:
            self.drag_endpoint = None
            self.dragging_measurement_id = None
            self._mark_dirty()
            self._refresh_all()

    def _on_motion(self, event: tk.Event[Any]) -> None:
        if self.pending_point is None:
            return
        self.render()
        start = self.image_to_canvas(self.pending_point)
        self.canvas.create_line(start[0], start[1], event.x, event.y, fill="#FFBF00", width=2, dash=(6, 3))

    def _create_two_click(self, p1: tuple[float, float], p2: tuple[float, float], snap: bool) -> None:
        assert self.project is not None and self.gray is not None
        method = "MANUAL_FREE_CALIPER"
        confidence = 1.0
        if snap:
            try:
                p1, p2, confidence = snap_two_click_edges(
                    self.gray, p1, p2, footer_bounds=self.project.image.footer_bounds
                )
                method = "ASSISTED_EDGE_SNAP"
            except Exception as exc:
                self.status_var.set(f"Ajuste no confiable; se conservaron los clics manuales: {exc}")
                method = "MANUAL_FREE_CALIPER"
                confidence = 0.5
        self._add_measurement(p1, p2, method, confidence)

    def _create_one_click(self, point: tuple[float, float]) -> None:
        assert self.project is not None and self.gray is not None
        try:
            p1, p2, confidence = one_click_measurement(
                self.gray,
                point,
                self.search_radius_var.get(),
                footer_bounds=self.project.image.footer_bounds,
            )
            self._add_measurement(p1, p2, "ASSISTED_LOCAL_ONE_CLICK", confidence)
            self.status_var.set(
                f"Propuesta local creada (confianza heurística {confidence:.2f}). Revísala y ajusta sus extremos."
            )
        except Exception as exc:
            messagebox.showwarning("Propuesta asistida no válida", str(exc), parent=self)

    def _add_measurement(self, p1: tuple[float, float], p2: tuple[float, float], method: str, confidence: float | None) -> None:
        assert self.project is not None
        validation = validate_measurement_geometry(
            p1,
            p2,
            width_px=self.project.image.width_px,
            height_px=self.project.image.height_px,
            footer_bounds=self.project.image.footer_bounds,
        )
        if not validation.valid:
            messagebox.showwarning("Medición no válida", validation.reason, parent=self)
            return

        width_m = self.project.image.calibration.distance_m(p1, p2)
        measurement = Measurement(
            measurement_id=self.project.next_measurement_id(),
            fiber_id=self.fiber_var.get().strip() or "F001",
            p1=p1,
            p2=p2,
            width_m=width_m,
            method=method,
            confidence=confidence,
        )
        self.project.measurements.append(measurement)
        self.select_measurement(measurement.measurement_id)
        self._mark_dirty()
        self._refresh_all()
        self.status_var.set(f"{measurement.measurement_id}: {format_length_m(width_m)}")

    def _hit_test(self, canvas_x: float, canvas_y: float) -> tuple[Measurement | None, int | None]:
        if self.project is None:
            return None, None
        threshold = 9.0
        best: tuple[float, Measurement, int | None] | None = None
        for measurement in self.project.measurements:
            p1 = np.asarray(self.image_to_canvas(measurement.p1))
            p2 = np.asarray(self.image_to_canvas(measurement.p2))
            point = np.asarray((canvas_x, canvas_y))
            d1 = float(np.linalg.norm(point - p1))
            d2 = float(np.linalg.norm(point - p2))
            for distance, endpoint in ((d1, 0), (d2, 1)):
                if distance <= threshold and (best is None or distance < best[0]):
                    best = (distance, measurement, endpoint)
            segment = p2 - p1
            denom = float(np.dot(segment, segment))
            if denom > 0:
                t = max(0.0, min(1.0, float(np.dot(point - p1, segment) / denom)))
                projection = p1 + t * segment
                line_distance = float(np.linalg.norm(point - projection))
                if line_distance <= threshold and (best is None or line_distance < best[0]):
                    best = (line_distance, measurement, None)
        return (best[1], best[2]) if best else (None, None)

    # ---------- measurements and statistics ----------

    def _measurement_by_id(self, measurement_id: str | None) -> Measurement | None:
        if self.project is None or measurement_id is None:
            return None
        return next((m for m in self.project.measurements if m.measurement_id == measurement_id), None)

    def _selected_measurement(self) -> Measurement | None:
        return self._measurement_by_id(self.selected_measurement_id)

    def select_measurement(self, measurement_id: str | None) -> None:
        self.selected_measurement_id = measurement_id
        if measurement_id and self.tree.exists(measurement_id):
            self.tree.selection_set(measurement_id)
            self.tree.see(measurement_id)
        measurement = self._selected_measurement()
        if measurement:
            self.fiber_var.set(measurement.fiber_id)
            self.defect_var.set(measurement.defect)
            confidence = "—" if measurement.confidence is None else f"{measurement.confidence:.2f}"
            self.selected_info_var.set(
                f"{measurement.measurement_id} · {measurement.fiber_id}\n"
                f"{format_length_m(measurement.width_m)}\n"
                f"{measurement.method}\nConfianza: {confidence}"
            )
        else:
            self.selected_info_var.set("Sin selección")
        self.render()

    def _tree_selection(self, _event: tk.Event[Any]) -> None:
        selected = self.tree.selection()
        if selected:
            self.selected_measurement_id = selected[0]
            measurement = self._selected_measurement()
            if measurement:
                self.fiber_var.set(measurement.fiber_id)
                self.defect_var.set(measurement.defect)
            self.render()
            self._update_selected_info()

    def _update_selected_info(self) -> None:
        measurement = self._selected_measurement()
        if not measurement:
            self.selected_info_var.set("Sin selección")
            return
        confidence = "—" if measurement.confidence is None else f"{measurement.confidence:.2f}"
        self.selected_info_var.set(
            f"{measurement.measurement_id} · {measurement.fiber_id}\n"
            f"{format_length_m(measurement.width_m)}\n{measurement.method}\n"
            f"Grupo: {measurement.group if measurement.group is not None else '—'} · Confianza: {confidence}"
        )

    def new_fiber(self) -> None:
        if self.project is None:
            self.fiber_var.set("F001")
            return
        numbers = []
        for measurement in self.project.measurements:
            if measurement.fiber_id.startswith("F") and measurement.fiber_id[1:].isdigit():
                numbers.append(int(measurement.fiber_id[1:]))
        self.fiber_var.set(f"F{(max(numbers, default=0) + 1):03d}")
        self.status_var.set(f"Nueva fibra actual: {self.fiber_var.get()}")

    def apply_defect(self) -> None:
        measurement = self._selected_measurement()
        if measurement is None:
            return
        measurement.defect = self.defect_var.get()
        self._mark_dirty()
        self._refresh_all()

    def toggle_selected_acceptance(self) -> None:
        measurement = self._selected_measurement()
        if measurement is None:
            return
        measurement.accepted = not measurement.accepted
        self._mark_dirty()
        self._refresh_all()
        self.status_var.set(f"{measurement.measurement_id}: {'aceptada' if measurement.accepted else 'rechazada'}")

    def delete_selected(self) -> None:
        if self.project is None or self.selected_measurement_id is None:
            return
        self.project.measurements = [m for m in self.project.measurements if m.measurement_id != self.selected_measurement_id]
        self.selected_measurement_id = None
        self._mark_dirty()
        self._refresh_all()

    def classify(self) -> None:
        if self.project is None or not self.project.measurements:
            return
        requested = None if self.classification_var.get() == "Auto" else int(self.classification_var.get())
        mapping = classify_fibers(self.project.measurements, requested_k=requested)
        groups = len(set(mapping.values())) if mapping else 0
        self._mark_dirty()
        self._refresh_all()
        self.status_var.set(f"Clasificación aplicada: {groups} grupo(s), usando la mediana por fibra.")

    def _refresh_all(self) -> None:
        self._refresh_tree()
        self._refresh_stats()
        self._refresh_metadata()
        self._update_selected_info()
        self.render()

    def _refresh_tree(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        if self.project is None:
            return
        for measurement in self.project.measurements:
            group = "—" if measurement.group is None else str(measurement.group + 1)
            self.tree.insert(
                "",
                tk.END,
                iid=measurement.measurement_id,
                values=(
                    measurement.measurement_id,
                    measurement.fiber_id,
                    f"{measurement.width_m * 1e6:.3f} µm",
                    measurement.method.replace("ASSISTED_", "A:").replace("MANUAL_", "M:"),
                    group,
                    measurement.defect,
                    "✓" if measurement.accepted else "✗",
                ),
            )
        if self.selected_measurement_id and self.tree.exists(self.selected_measurement_id):
            self.tree.selection_set(self.selected_measurement_id)

    def _refresh_stats(self) -> None:
        measurements = self.project.measurements if self.project else []
        f_stats = fiber_level_summary(measurements)
        s_stats = section_level_summary(measurements)
        fibers = fiber_statistics(measurements)

        lines = ["RESUMEN POR FIBRA — medianas por fibra", ""]
        lines.append(f"Fibras identificadas: {f_stats['n_fibers']}")
        lines.append(f"Mediciones válidas totales: {f_stats['n_measurements']}")
        for label, key in (
            ("Media de medianas", "mean_m"),
            ("Mediana global", "median_m"),
            ("Mínimo crudo", "min_m"),
            ("Máximo crudo", "max_m"),
            ("Desv. estándar", "std_m"),
            ("P05", "p05_m"),
            ("P95", "p95_m"),
        ):
            val = f_stats[key]
            lines.append(f"{label}: {'—' if val is None else format_length_m(float(val))}")

        lines.extend(["", "DISTRIBUCIÓN DE SECCIONES LOCALES", ""])
        lines.append(f"Secciones totales: {s_stats['n_measurements']}")
        for label, key in (
            ("Media por sección", "mean_m"),
            ("Mediana por sección", "median_m"),
            ("Mínimo crudo", "min_m"),
            ("Máximo crudo", "max_m"),
            ("Desv. estándar", "std_m"),
            ("P05", "p05_m"),
            ("P95", "p95_m"),
        ):
            val = s_stats[key]
            lines.append(f"{label}: {'—' if val is None else format_length_m(float(val))}")

        lines.extend(["", "DETALLE POR FIBRA", ""])
        for fiber_id, values in sorted(fibers.items()):
            groups = {m.group for m in measurements if m.fiber_id == fiber_id and m.group is not None}
            group_text = f" · grupo {next(iter(groups)) + 1}" if len(groups) == 1 else ""
            lines.append(
                f"{fiber_id}: N={values['n']} · mediana={format_length_m(float(values['median_m']))}"
                f" · min={format_length_m(float(values['min_m']))} · max={format_length_m(float(values['max_m']))}{group_text}"
            )
        lines.extend([
            "",
            "INTERPRETACIÓN",
            "La aplicación mide ancho proyectado en la micrografía 2D. El resumen principal se calcula sobre las medianas de cada fibra para evitar que fibras con más mediciones sesguen la distribución.",
        ])
        self.stats_text.configure(state=tk.NORMAL)
        self.stats_text.delete("1.0", tk.END)
        self.stats_text.insert("1.0", "\n".join(lines))
        self.stats_text.configure(state=tk.DISABLED)

    def _refresh_metadata(self) -> None:
        self.metadata_text.configure(state=tk.NORMAL)
        self.metadata_text.delete("1.0", tk.END)
        if self.project:
            payload = {
                "path": self.project.image.path,
                "shape": [self.project.image.height_px, self.project.image.width_px],
                "pixel_size_nm": self.project.image.calibration.pixel_size_x_m * 1e9,
                "calibration_source": self.project.image.calibration.source,
                "footer_bounds": self.project.image.footer_bounds,
                "sha256": self.project.image.source_sha256,
                "instrument_metadata": self.project.image.metadata,
            }
            self.metadata_text.insert("1.0", json.dumps(payload, indent=2, ensure_ascii=False))
        self.metadata_text.configure(state=tk.DISABLED)

    def _clear_pending(self) -> None:
        self.pending_point = None
        self.drag_endpoint = None
        self.dragging_measurement_id = None
        self.render()

    # ---------- help ----------

    def show_protocol(self) -> None:
        messagebox.showinfo(
            "Protocolo rápido",
            "1. Abre el TIFF Zeiss y confirma la escala mostrada.\n"
            "2. Crea una ID por fibra.\n"
            "3. Mide siempre perpendicular al eje. Usa 3–5 secciones limpias por fibra.\n"
            "4. Evita cruces, beads, bordes de imagen y zonas fusionadas; o márcalas como defecto.\n"
            "5. Las herramientas asistidas entregan puntajes heurísticos locales, no probabilidades calibradas. Selecciónala con V y arrastra sus extremos.\n"
            "6. Clasifica después de medir varias fibras y revisa cada grupo sobre la imagen.\n"
            "7. Guarda el proyecto y exporta CSV + informe.",
            parent=self,
        )

    def show_about(self) -> None:
        messagebox.showinfo(
            "Acerca de",
            "Fathom Fibers Quick 0.1\n\n"
            "MVP manual-first para ancho proyectado de fibras SEM.\n"
            "Lee metadata CZ_SEM de TIFF Zeiss y conserva cada medición con coordenadas físicas.\n\n"
            "Los puntajes de herramientas asistidas son heurísticos locales. No sustituyen la revisión científica de cruces, inclinación o regiones fusionadas.",
            parent=self,
        )


def main() -> None:
    initial_path = sys.argv[1] if len(sys.argv) > 1 else None
    app = FiberQuickApp(initial_path)
    app.mainloop()


if __name__ == "__main__":
    main()
