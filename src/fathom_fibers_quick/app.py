from __future__ import annotations

import json
import math
import sys
import tkinter as tk
from dataclasses import asdict
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any, ClassVar

import numpy as np
from PIL import Image, ImageTk

from .analysis import (
    classify_fibers,
    compute_histogram_data,
    fiber_statistics,
    format_length_m,
    one_click_measurement,
    snap_two_click_edges,
)
from .auto_roi import (
    PRESET_HIGH_MAG_FINE,
    PRESET_LOW_MAG_NETWORK,
    PRESET_MID_MAG_GENERAL,
    AutoFiberCandidate,
    AutoROISummary,
    analyze_roi,
    get_preset_for_calibration,
)
from .autosave import check_has_autosave, clear_autosave, load_autosave, perform_atomic_autosave
from .exporters import export_annotated, export_csv, export_html_report
from .history import Command, HistoryManager
from .measurement_geometry import (
    compute_angle_geometry,
    compute_area_roi_geometry,
    compute_line_geometry,
    compute_polyline_geometry,
    compute_profile_geometry,
)
from .measurement_records import (
    MeasurementKind,
    MeasurementRecord,
    MeasurementSource,
    MeasurementStatus,
    normalize_tags,
)
from .model import Project
from .project_io import (
    SourceVerificationStatus,
    load_project,
    save_project,
    verify_project_source,
)
from .zeiss import load_image_document, load_pixels


class FiberQuickApp(tk.Tk):
    GROUP_HEX: ClassVar[list[str]] = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
    SELECTED = "#FFBF00"
    CANDIDATE_COLOR = "#00E5FF"

    def __init__(self, initial_path: str | None = None) -> None:
        super().__init__()
        self.title("Fathom Fibers Quick 0.2 — Scientific Measurement Workspace")
        self.geometry("1600x960")
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

        # History manager
        self.history = HistoryManager()
        self.history.register_on_change(self._on_history_change)
        self._action_counter = 0

        # Selection state
        self.selected_record_id: str | None = None
        self.active_fiber_id: str = "F001"
        self.drag_endpoint: int | None = None
        self.dragging_record_id: str | None = None

        # Multi-click drawing state for Polylines, Angles, Polygons
        self.drawing_points: list[tuple[float, float]] = []
        self.pending_point: tuple[float, float] | None = None
        self.histogram_filter: tuple[str, int, list[str]] | None = None

        # Auto ROI state
        self.roi_bbox: tuple[int, int, int, int] | None = None
        self.pending_roi_start: tuple[float, float] | None = None
        self.auto_candidates: list[AutoFiberCandidate] = []
        self.auto_summary: AutoROISummary | None = None
        self.selected_candidate_id: str | None = None

        self.pan_anchor: tuple[float, float] | None = None
        self.pan_offset_anchor: tuple[float, float] | None = None

        # Variables
        self.tool_var = tk.StringVar(value="select")
        self.target_sections_var = tk.StringVar(value="5 secciones")
        self.auto_advance_var = tk.BooleanVar(value=True)
        self.show_all_sections_var = tk.BooleanVar(value=True)
        self.defect_var = tk.StringVar(value="None")
        self.search_radius_var = tk.DoubleVar(value=60.0)
        self.profile_bandwidth_var = tk.IntVar(value=3)
        self.classification_var = tk.StringVar(value="Auto")
        self.footer_visible_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Abre un TIFF Zeiss para comenzar.")
        self.calibration_var = tk.StringVar(value="Sin imagen")
        self.protocol_status_var = tk.StringVar(value="Protocolo: 0 / 5 (Incompleto)")

        # Results table filter variables
        self.filter_text_var = tk.StringVar(value="")
        self.filter_kind_var = tk.StringVar(value="Todos")
        self.filter_status_var = tk.StringVar(value="Todos")

        # Auto ROI variables
        self.show_auto_candidates_var = tk.BooleanVar(value=True)
        self.roi_preset_var = tk.StringVar(value="MID_MAG_GENERAL")
        self.roi_thresh_var = tk.StringVar(value="Automático")
        self.roi_polarity_var = tk.StringVar(value="Automática")
        self.roi_min_area_var = tk.IntVar(value=40)
        self.roi_min_elong_var = tk.DoubleVar(value=2.2)
        self.roi_min_width_var = tk.DoubleVar(value=3.0)
        self.roi_n_sections_var = tk.IntVar(value=3)
        self.roi_curved_var = tk.BooleanVar(value=True)
        self.roi_summary_var = tk.StringVar(value="ROI: Sin analizar")
        self.roi_feedback_var = tk.StringVar(value="Dibuja una ROI sobre la imagen.")

        # Histogram variables
        self.hist_mode_var = tk.StringVar(value="fiber")
        self.hist_bins_var = tk.IntVar(value=10)

        self._build_ui()
        self._bind_shortcuts()
        self.protocol("WM_DELETE_WINDOW", self._on_app_closing)
        self._schedule_autosave_timer()

        self.after(100, lambda: self.open_path(initial_path) if initial_path else None)

    # ---------- Dirty & Title & History Callback ----------

    def _update_title(self) -> None:
        name = Path(self.project.image.path).name if self.project else ""
        dirty = " *" if self._is_dirty else ""
        if name:
            self.title(f"Fathom Fibers Quick — {name}{dirty}")
        else:
            self.title(f"Fathom Fibers Quick 0.2 — Scientific Measurement Workspace{dirty}")

    def _mark_dirty(self) -> None:
        self._is_dirty = True
        self._action_counter += 1
        self._update_title()
        if self._action_counter >= 10 and self.project:
            self._trigger_autosave()

    def _clear_dirty(self) -> None:
        self._is_dirty = False
        self._action_counter = 0
        self._update_title()
        if self.project:
            clear_autosave(self.project)

    def _on_history_change(self) -> None:
        self._refresh_history_tab()

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

    # ---------- Autosave ----------

    def _schedule_autosave_timer(self) -> None:
        self.after(30000, self._periodic_autosave_check)

    def _periodic_autosave_check(self) -> None:
        if self._is_dirty and self.project:
            self._trigger_autosave()
        self._schedule_autosave_timer()

    def _trigger_autosave(self) -> None:
        if self.project:
            try:
                perform_atomic_autosave(self.project)
                self.status_var.set("Autosave guardado en segundo plano.")
            except Exception as exc:
                print(f"Error in autosave: {exc}")

    def _check_and_prompt_autosave(self) -> None:
        if not self.project:
            return
        has_auto, path, _mtime = check_has_autosave(self.project)
        if has_auto and path:
            answer = messagebox.askyesnocancel(
                "Autosave recuperable",
                f"Se detectó un autosave más reciente para esta imagen:\n{path.name}\n\n"
                "• Haz clic en SÍ para recuperar la sesión guardada.\n"
                "• Haz clic en NO para ignorar y descartar el autosave.\n"
                "• Haz clic en CANCELAR para continuar sin cambios.",
                parent=self,
            )
            if answer is True:
                try:
                    self.project = load_autosave(path)
                    self.status_var.set("Sesión recuperada desde autosave.")
                    self._mark_dirty()
                except Exception as exc:
                    messagebox.showerror("Error al cargar autosave", str(exc), parent=self)
            elif answer is False:
                clear_autosave(self.project)

    # ---------- UI construction ----------

    def _build_ui(self) -> None:
        self._build_menu()

        # Scientific Notice Banner (Section 18)
        notice_frame = tk.Frame(self, background="#003542", height=24)
        notice_frame.pack(fill=tk.X, side=tk.TOP)
        notice_lbl = tk.Label(
            notice_frame,
            text="ⓘ Las mediciones representan geometría proyectada 2D. La interpretación física depende de calibración, resolución y geometría de la muestra.",
            foreground="#00E5FF",
            background="#003542",
            font=("TkDefaultFont", 8, "italic"),
        )
        notice_lbl.pack(side=tk.LEFT, padx=8, pady=2)

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
        file_menu.add_command(label="Abrir imagen… (Ctrl+O)", command=self.open_image_dialog, accelerator="Ctrl+O")
        file_menu.add_command(label="Abrir proyecto…", command=self.open_project_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Guardar proyecto… (Ctrl+S)", command=self.save_project_dialog, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="Exportar CSV unificado…", command=self.export_csv_dialog)
        file_menu.add_command(label="Exportar imagen anotada… (Ctrl+E)", command=self.export_annotated_dialog, accelerator="Ctrl+E")
        file_menu.add_command(label="Exportar informe HTML…", command=self.export_report_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Salir", command=self._on_app_closing)
        menu.add_cascade(label="Archivo", menu=file_menu)

        edit_menu = tk.Menu(menu, tearoff=False)
        edit_menu.add_command(label="Deshacer (Ctrl+Z)", command=self.undo, accelerator="Ctrl+Z")
        edit_menu.add_command(label="Rehacer (Ctrl+Y)", command=self.redo, accelerator="Ctrl+Y")
        menu.add_cascade(label="Edición", menu=edit_menu)

        help_menu = tk.Menu(menu, tearoff=False)
        help_menu.add_command(label="Protocolo rápido", command=self.show_protocol)
        help_menu.add_command(label="Acerca de", command=self.show_about)
        menu.add_cascade(label="Ayuda", menu=help_menu)
        self.config(menu=menu)

    def _build_left_panel(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="FATHOM FIBERS", font=("TkDefaultFont", 14, "bold")).pack(anchor=tk.W)
        ttk.Label(parent, text="Scientific Measurement Workspace 0.2", foreground="#666").pack(anchor=tk.W, pady=(0, 4))

        # Protocol Panel
        proto_box = ttk.LabelFrame(parent, text="PROTOCOLO DE FIBRA", padding=6)
        proto_box.pack(fill=tk.X, pady=4)

        row_target = ttk.Frame(proto_box)
        row_target.pack(fill=tk.X, pady=2)
        ttk.Label(row_target, text="Objetivo:").pack(side=tk.LEFT)
        target_cb = ttk.Combobox(
            row_target,
            textvariable=self.target_sections_var,
            values=("3 secciones", "5 secciones", "Libre (0)"),
            state="readonly",
            width=12,
        )
        target_cb.pack(side=tk.RIGHT)
        target_cb.bind("<<ComboboxSelected>>", self._on_target_sections_changed)

        row_fiber = ttk.Frame(proto_box)
        row_fiber.pack(fill=tk.X, pady=4)
        ttk.Label(row_fiber, text="Fibra activa:", font=("TkDefaultFont", 10, "bold")).pack(side=tk.LEFT)
        self.lbl_active_fiber = ttk.Label(row_fiber, text="F001", font=("TkDefaultFont", 11, "bold"), foreground="#0072B2")
        self.lbl_active_fiber.pack(side=tk.LEFT, padx=6)
        ttk.Button(row_fiber, text="+ Nueva (N)", command=self.new_fiber, width=10).pack(side=tk.RIGHT)

        ttk.Label(proto_box, textvariable=self.protocol_status_var, font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W, pady=2)

        nav_row = ttk.Frame(proto_box)
        nav_row.pack(fill=tk.X, pady=4)
        ttk.Button(nav_row, text="◀ Anter. (PgUp)", command=self.prev_fiber, width=13).pack(side=tk.LEFT, expand=True)
        ttk.Button(nav_row, text="Siguiente ▶ (PgDn)", command=self.next_fiber, width=13).pack(side=tk.RIGHT, expand=True)

        ttk.Checkbutton(proto_box, text="Avanzar auto al completar", variable=self.auto_advance_var).pack(anchor=tk.W, pady=2)

        # Tools Panel (Expanded with 5 Scientific Measurement Tools)
        tools = ttk.LabelFrame(parent, text="Herramientas de Medición", padding=6)
        tools.pack(fill=tk.X, pady=4)
        choices = [
            ("Seleccionar / editar (V)", "select"),
            ("Ancho proyectado (M)", "manual"),
            ("Línea distancia (D)", "distance"),
            ("Polilínea longitud (P)", "polyline"),
            ("Ángulo 3-puntos (G)", "angle"),
            ("ROI Rectángulo área (R)", "rect_area"),
            ("ROI Polígono área (Y)", "poly_area"),
            ("Perfil de intensidad (L)", "profile"),
            ("ROI automática (I)", "auto_roi"),
            ("Ajustar bordes snap (S)", "snap"),
            ("Propuesta local (A)", "auto"),
        ]
        for label, value in choices:
            ttk.Radiobutton(tools, text=label, variable=self.tool_var, value=value, command=self._clear_pending).pack(anchor=tk.W)

        prof_row = ttk.Frame(tools)
        prof_row.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(prof_row, text="Ancho banda perfil:").pack(side=tk.LEFT)
        ttk.Spinbox(prof_row, from_=1, to=21, increment=2, textvariable=self.profile_bandwidth_var, width=5).pack(side=tk.RIGHT)

        # Auto ROI Control Panel
        roi_box = ttk.LabelFrame(parent, text="Análisis Automático ROI", padding=6)
        roi_box.pack(fill=tk.X, pady=4)

        preset_row = ttk.Frame(roi_box)
        preset_row.pack(fill=tk.X, pady=2)
        ttk.Label(preset_row, text="Perfil:").pack(side=tk.LEFT)
        preset_cb = ttk.Combobox(
            preset_row,
            textvariable=self.roi_preset_var,
            values=("HIGH_MAG_FINE", "MID_MAG_GENERAL", "LOW_MAG_NETWORK", "Personalizado"),
            state="readonly",
            width=14,
        )
        preset_cb.pack(side=tk.RIGHT)
        preset_cb.bind("<<ComboboxSelected>>", self._on_preset_changed)

        thresh_row = ttk.Frame(roi_box)
        thresh_row.pack(fill=tk.X, pady=2)
        ttk.Label(thresh_row, text="Umbral:").pack(side=tk.LEFT)
        ttk.Combobox(
            thresh_row,
            textvariable=self.roi_thresh_var,
            values=("Automático", "Otsu", "Percentil", "Local Adaptativo"),
            state="readonly",
            width=14,
        ).pack(side=tk.RIGHT)

        roi_btns = ttk.Frame(roi_box)
        roi_btns.pack(fill=tk.X, pady=(4, 2))
        ttk.Button(roi_btns, text="Analizar ROI", command=self.run_auto_roi).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        ttk.Button(roi_btns, text="Limpiar ROI", command=self.clear_roi).pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=1)

        ttk.Label(roi_box, textvariable=self.roi_feedback_var, font=("TkDefaultFont", 8, "italic"), foreground="#444", wraplength=210).pack(anchor=tk.W, pady=2)

        # Display & View Panel
        display = ttk.LabelFrame(parent, text="Vista y Canvas", padding=6)
        display.pack(fill=tk.X, pady=4)
        ttk.Checkbutton(display, text="Mostrar todas las fibras", variable=self.show_all_sections_var, command=self.render).pack(anchor=tk.W)
        ttk.Checkbutton(display, text="Mostrar candidatos ROI", variable=self.show_auto_candidates_var, command=self.render).pack(anchor=tk.W)

        btn_row = ttk.Frame(display)
        btn_row.pack(fill=tk.X, pady=2)
        ttk.Button(btn_row, text="Ajustar (F)", command=self.fit_image).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        ttk.Button(btn_row, text="Centrar (C)", command=self.center_selection).pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=1)

        # Classification Panel
        classify = ttk.LabelFrame(parent, text="Familias de Tamaño", padding=6)
        classify.pack(fill=tk.X, pady=4)
        ttk.Combobox(classify, textvariable=self.classification_var, state="readonly", values=("Auto", "1", "2", "3", "4"), width=8).pack(fill=tk.X, pady=2)
        ttk.Button(classify, text="Clasificar fibras", command=self.classify).pack(fill=tk.X, pady=(2, 0))

    def _build_canvas(self, parent: ttk.Frame) -> None:
        self.canvas = tk.Canvas(parent, background="#161616", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind("<ButtonPress-1>", self._on_left_press)
        self.canvas.bind("<B1-Motion>", self._on_left_motion)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_release)
        self.canvas.bind("<Double-Button-1>", self._on_double_click_canvas)
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<ButtonPress-3>", self._start_pan)
        self.canvas.bind("<B3-Motion>", self._pan_motion)
        self.canvas.bind("<ButtonRelease-3>", self._end_pan)
        self.canvas.bind("<MouseWheel>", self._wheel_zoom)
        self.canvas.bind("<Button-4>", lambda event: self._zoom_at(event.x, event.y, 1.2))
        self.canvas.bind("<Button-5>", lambda event: self._zoom_at(event.x, event.y, 1 / 1.2))

    def _build_right_panel(self, parent: ttk.Frame) -> None:
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        results_tab = ttk.Frame(self.notebook, padding=4)
        fibers_tab = ttk.Frame(self.notebook, padding=4)
        auto_tab = ttk.Frame(self.notebook, padding=4)
        histogram_tab = ttk.Frame(self.notebook, padding=4)
        inspector_tab = ttk.Frame(self.notebook, padding=6)
        history_tab = ttk.Frame(self.notebook, padding=4)
        metadata_tab = ttk.Frame(self.notebook, padding=6)

        self.notebook.add(results_tab, text="RESULTADOS")
        self.notebook.add(fibers_tab, text="Fibras")
        self.notebook.add(auto_tab, text="Revisión ROI")
        self.notebook.add(histogram_tab, text="Histograma")
        self.notebook.add(inspector_tab, text="Inspector")
        self.notebook.add(history_tab, text="Historial")
        self.notebook.add(metadata_tab, text="Metadata")

        # Tab 1: Unified Scientific Results Table (Section 9)
        self._build_results_table_tab(results_tab)

        # Tab 2: Fibers Tree
        fiber_cols = ("id", "sections", "median", "min", "max", "status", "group")
        self.fiber_tree = ttk.Treeview(fibers_tab, columns=fiber_cols, show="headings", height=28)
        fiber_headers = {"id": "ID Fibra", "sections": "Secciones", "median": "Mediana", "min": "Mín.", "max": "Máx.", "status": "Estado", "group": "Grupo"}
        fiber_widths = {"id": 75, "sections": 70, "median": 90, "min": 85, "max": 85, "status": 95, "group": 55}
        for col in fiber_cols:
            self.fiber_tree.heading(col, text=fiber_headers[col])
            self.fiber_tree.column(col, width=fiber_widths[col], anchor=tk.CENTER)
        f_scroll = ttk.Scrollbar(fibers_tab, orient=tk.VERTICAL, command=self.fiber_tree.yview)
        self.fiber_tree.configure(yscrollcommand=f_scroll.set)
        self.fiber_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        f_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.fiber_tree.bind("<<TreeviewSelect>>", self._on_fiber_tree_select)

        # Tab 3: Auto-ROI Review Queue
        self._build_auto_review_tab(auto_tab)

        # Tab 4: Interactive Histogram
        self._build_histogram_tab(histogram_tab)

        # Tab 5: Inspector
        self.inspector_text = tk.Text(inspector_tab, width=46, height=40, wrap=tk.WORD, state=tk.DISABLED)
        self.inspector_text.pack(fill=tk.BOTH, expand=True)

        # Tab 6: History Tab (Section 13)
        self._build_history_tab(history_tab)

        # Tab 7: Metadata
        self.metadata_text = tk.Text(metadata_tab, width=46, height=40, wrap=tk.NONE, state=tk.DISABLED)
        self.metadata_text.pack(fill=tk.BOTH, expand=True)

    def _build_results_table_tab(self, parent: ttk.Frame) -> None:
        filter_bar = ttk.Frame(parent, padding=2)
        filter_bar.pack(fill=tk.X, side=tk.TOP)

        ttk.Label(filter_bar, text="Filtrar:").pack(side=tk.LEFT)
        ttk.Entry(filter_bar, textvariable=self.filter_text_var, width=14).pack(side=tk.LEFT, padx=2)
        self.filter_text_var.trace_add("write", lambda *_: self._refresh_results_table())

        ttk.Label(filter_bar, text="Tipo:").pack(side=tk.LEFT, padx=(4, 0))
        kind_cb = ttk.Combobox(filter_bar, textvariable=self.filter_kind_var, values=("Todos", "PROJECTED_WIDTH", "DISTANCE", "POLYLINE_LENGTH", "ANGLE", "RECTANGLE_AREA", "POLYGON_AREA", "INTENSITY_PROFILE"), state="readonly", width=13)
        kind_cb.pack(side=tk.LEFT, padx=2)
        kind_cb.bind("<<ComboboxSelected>>", lambda _: self._refresh_results_table())

        columns = ("id", "name", "kind", "fiber", "primary_val", "unit", "source", "status", "tags")
        self.results_tree = ttk.Treeview(parent, columns=columns, show="headings", height=24)
        labels = {"id": "ID", "name": "Nombre", "kind": "Tipo", "fiber": "Fibra", "primary_val": "Valor", "unit": "Unidad", "source": "Método", "status": "Estado", "tags": "Tags"}
        widths = {"id": 65, "name": 110, "kind": 110, "fiber": 60, "primary_val": 95, "unit": 50, "source": 85, "status": 80, "tags": 100}

        for col in columns:
            self.results_tree.heading(col, text=labels[col], command=lambda _c=col: self._sort_results_tree(_c))
            self.results_tree.column(col, width=widths[col], anchor=tk.CENTER, stretch=(col in {"name", "tags"}))

        scroll = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scroll.set)
        self.results_tree.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_tree.bind("<<TreeviewSelect>>", self._on_results_tree_select)
        self.results_tree.bind("<Double-1>", self._on_results_tree_double_click)

        btn_bar = ttk.Frame(parent, padding=4)
        btn_bar.pack(fill=tk.X, side=tk.BOTTOM)
        ttk.Button(btn_bar, text="✓ Aceptar", command=self.accept_selected_records).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_bar, text="✗ Rechazar", command=self.reject_selected_records).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_bar, text="✎ Editar Metadata", command=self.edit_selected_metadata).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_bar, text="Eliminar (Del)", command=self.delete_selected_records).pack(side=tk.RIGHT, padx=2)

    def _build_auto_review_tab(self, parent: ttk.Frame) -> None:
        summary_lbl = ttk.Label(parent, textvariable=self.roi_summary_var, font=("TkDefaultFont", 9, "bold"), padding=4)
        summary_lbl.pack(anchor=tk.W)

        auto_cols = ("id", "status", "sections", "median", "cv_width", "discrepancy", "confidence", "flags")
        self.auto_tree = ttk.Treeview(parent, columns=auto_cols, show="headings", height=18)
        auto_headers = {"id": "ID Candidate", "status": "Estado", "sections": "Secciones", "median": "Mediana", "cv_width": "CV Ancho", "discrepancy": "Discrepancia", "confidence": "Confianza", "flags": "Quality Flags"}
        auto_widths = {"id": 85, "status": 75, "sections": 65, "median": 80, "cv_width": 75, "discrepancy": 85, "confidence": 95, "flags": 140}
        for col in auto_cols:
            self.auto_tree.heading(col, text=auto_headers[col])
            self.auto_tree.column(col, width=auto_widths[col], anchor=tk.CENTER, stretch=(col == "flags"))

        a_scroll = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.auto_tree.yview)
        self.auto_tree.configure(yscrollcommand=a_scroll.set)
        self.auto_tree.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        a_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.auto_tree.bind("<<TreeviewSelect>>", self._on_auto_tree_select)

        ctrl_frame = ttk.Frame(parent, padding=4)
        ctrl_frame.pack(fill=tk.X, side=tk.BOTTOM)
        row1 = ttk.Frame(ctrl_frame)
        row1.pack(fill=tk.X, pady=2)
        ttk.Button(row1, text="✓ Aceptar candidato", command=self.accept_candidate).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        ttk.Button(row1, text="✗ Rechazar candidato", command=self.reject_candidate).pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=1)

    def _build_histogram_tab(self, parent: ttk.Frame) -> None:
        ctrls = ttk.Frame(parent, padding=4)
        ctrls.pack(fill=tk.X)

        ttk.Label(ctrls, text="Modo:").pack(side=tk.LEFT, padx=(0, 4))
        r1 = ttk.Radiobutton(ctrls, text="Por fibra", variable=self.hist_mode_var, value="fiber", command=self._refresh_histogram)
        r2 = ttk.Radiobutton(ctrls, text="Por sección", variable=self.hist_mode_var, value="section", command=self._refresh_histogram)
        r1.pack(side=tk.LEFT, padx=2)
        r2.pack(side=tk.LEFT, padx=2)

        ttk.Label(ctrls, text=" Bins:").pack(side=tk.LEFT, padx=(8, 2))
        spin = ttk.Spinbox(ctrls, from_=5, to=30, increment=1, textvariable=self.hist_bins_var, width=5, command=self._refresh_histogram)
        spin.pack(side=tk.LEFT, padx=2)

        self.hist_info_lbl = ttk.Label(parent, text="N = 0", font=("TkDefaultFont", 9, "bold"))
        self.hist_info_lbl.pack(anchor=tk.W, padx=6, pady=2)

        self.hist_canvas = tk.Canvas(parent, background="#222222", height=320, highlightthickness=0)
        self.hist_canvas.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    def _build_history_tab(self, parent: ttk.Frame) -> None:
        ctrls = ttk.Frame(parent, padding=4)
        ctrls.pack(fill=tk.X, side=tk.TOP)

        ttk.Button(ctrls, text="↶ Deshacer (Ctrl+Z)", command=self.undo).pack(side=tk.LEFT, padx=4)
        ttk.Button(ctrls, text="↷ Rehacer (Ctrl+Y)", command=self.redo).pack(side=tk.LEFT, padx=4)

        self.history_listbox = tk.Listbox(parent, background="#1e1e1e", foreground="#eeeeee", selectbackground="#0072B2")
        self.history_listbox.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    def _refresh_history_tab(self) -> None:
        self.history_listbox.delete(0, tk.END)
        for entry in self.history.get_log_entries():
            self.history_listbox.insert(tk.END, entry)

    def undo(self) -> None:
        cmd = self.history.undo()
        if cmd:
            self.status_var.set(f"Deshecho: {cmd.description}")
            self._mark_dirty()
            self._refresh_all()

    def redo(self) -> None:
        cmd = self.history.redo()
        if cmd:
            self.status_var.set(f"Rehecho: {cmd.description}")
            self._mark_dirty()
            self._refresh_all()

    def _bind_shortcuts(self) -> None:
        self.bind_all("<Control-o>", lambda _event: self.open_image_dialog())
        self.bind_all("<Control-s>", lambda _event: self.save_project_dialog())
        self.bind_all("<Control-e>", lambda _event: self.export_annotated_dialog())
        self.bind_all("<Control-z>", lambda _event: self.undo())
        self.bind_all("<Control-y>", lambda _event: self.redo())
        self.bind_all("<Control-Shift-Z>", lambda _event: self.redo())

        self.bind_all("<Key-v>", lambda event: self._shortcut_guard(event, lambda: self.tool_var.set("select")))
        self.bind_all("<Key-i>", lambda event: self._shortcut_guard(event, lambda: self.tool_var.set("auto_roi")))
        self.bind_all("<Key-m>", lambda event: self._shortcut_guard(event, lambda: self.tool_var.set("manual")))
        self.bind_all("<Key-d>", lambda event: self._shortcut_guard(event, lambda: self.tool_var.set("distance")))
        self.bind_all("<Key-p>", lambda event: self._shortcut_guard(event, lambda: self.tool_var.set("polyline")))
        self.bind_all("<Key-g>", lambda event: self._shortcut_guard(event, lambda: self.tool_var.set("angle")))
        self.bind_all("<Key-r>", lambda event: self._shortcut_guard(event, lambda: self.tool_var.set("rect_area")))
        self.bind_all("<Key-y>", lambda event: self._shortcut_guard(event, lambda: self.tool_var.set("poly_area")))
        self.bind_all("<Key-l>", lambda event: self._shortcut_guard(event, lambda: self.tool_var.set("profile")))
        self.bind_all("<Key-n>", lambda event: self._shortcut_guard(event, self.new_fiber))
        self.bind_all("<Prior>", lambda _event: self.prev_fiber())
        self.bind_all("<Next>", lambda _event: self.next_fiber())
        self.bind_all("<Return>", lambda event: self._shortcut_guard(event, self._on_enter_pressed))
        self.bind_all("<Delete>", lambda _event: self._on_delete_pressed())
        self.bind_all("<Key-f>", lambda event: self._shortcut_guard(event, self.fit_image))
        self.bind_all("<Key-c>", lambda event: self._shortcut_guard(event, self.center_selection))
        self.bind_all("<Escape>", lambda _event: self._clear_pending())

    def _shortcut_guard(self, event: tk.Event[Any], action: Any) -> None:
        widget = self.focus_get()
        if isinstance(widget, (tk.Entry, ttk.Entry, tk.Text, ttk.Spinbox, ttk.Combobox)):
            return
        action()

    def _on_preset_changed(self, _event: tk.Event[Any]) -> None:
        preset_name = self.roi_preset_var.get()
        if preset_name == "HIGH_MAG_FINE":
            preset = PRESET_HIGH_MAG_FINE
        elif preset_name == "LOW_MAG_NETWORK":
            preset = PRESET_LOW_MAG_NETWORK
        else:
            preset = PRESET_MID_MAG_GENERAL

        self.roi_min_area_var.set(preset.min_area_px)
        self.roi_min_elong_var.set(preset.min_elongation)
        self.roi_min_width_var.set(preset.min_width_px)
        self.roi_n_sections_var.set(preset.n_sections)

        self.roi_feedback_var.set(f"Perfil seleccionado: {preset.name}\n{preset.description}")

    def _on_target_sections_changed(self, _event: tk.Event[Any]) -> None:
        val = self.target_sections_var.get()
        target = 3 if "3" in val else (5 if "5" in val else 0)
        if self.project:
            self.project.target_sections = target
            self._mark_dirty()
        self._update_protocol_display()
        self._refresh_all()

    # ---------- Opening and Saving Projects ----------

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
            document, source_image, gray = load_image_document(path)
            self.project = Project(schema_version=2, image=document)
            self.source_image = source_image
            self.gray = gray
            self.selected_record_id = None
            self.active_fiber_id = "F001"
            self.drawing_points = []
            self.pending_point = None
            self.roi_bbox = None
            self.auto_candidates = []
            self.auto_summary = None
            self.history.clear()
            self._has_fit = False

            auto_preset = get_preset_for_calibration(document.calibration)
            self.roi_preset_var.set(auto_preset.name)
            self._on_preset_changed(None)

            self._clear_dirty()
            self._check_and_prompt_autosave()

            self.calibration_var.set(
                f"{document.calibration.pixel_size_x_m * 1e9:.4f} nm/px\n"
                f"Fuente: {document.calibration.source}\n"
                f"Footer: {document.footer_bounds or 'no detectado'}"
            )
            self._refresh_all()
            self.after_idle(self.fit_image)
            self.status_var.set(f"Imagen cargada: {Path(path).name}")
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
                answer = messagebox.askyesno("Imagen no encontrada", f"{verification.message}\n\n¿Deseas localizar la imagen fuente?", parent=self)
                if not answer:
                    return
                replacement = filedialog.askopenfilename(title="Localiza la imagen original", parent=self)
                if not replacement:
                    return
                project.image.path = str(Path(replacement).resolve())

            source_image, gray = load_pixels(project.image.path)
            self.project = project
            self.source_image = source_image
            self.gray = gray
            self.selected_record_id = None
            self.active_fiber_id = project.active_fiber_id or "F001"
            self.drawing_points = []
            self.roi_bbox = None
            self.auto_candidates = []
            self.auto_summary = None
            self.history.clear()
            self._has_fit = False

            self._clear_dirty()
            self._check_and_prompt_autosave()
            self._refresh_all()
            self.after_idle(self.fit_image)
            self.status_var.set(f"Proyecto abierto: {Path(path).name} ({len(project.records)} registros).")
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
                self.project.active_fiber_id = self.active_fiber_id
                saved = save_project(self.project, path)
                self._clear_dirty()
                self.status_var.set(f"Proyecto guardado: {saved}")
            except Exception as exc:
                messagebox.showerror("Error al guardar", str(exc), parent=self)

    # ---------- Canvas Zoom & Render ----------

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

    def center_selection(self) -> None:
        if self.source_image is None or self.project is None:
            return
        target_center: tuple[float, float] | None = None
        record = self._selected_record()
        if record:
            target_center = record.center
        elif self.active_fiber_id:
            acts = self.project.accepted_fiber_measurements(self.active_fiber_id)
            if acts:
                target_center = (
                    float(np.mean([m.center[0] for m in acts])),
                    float(np.mean([m.center[1] for m in acts])),
                )

        if target_center:
            cw = max(self.canvas.winfo_width(), 10)
            ch = max(self.canvas.winfo_height(), 10)
            self.offset_x = cw / 2 - target_center[0] * self.scale
            self.offset_y = ch / 2 - target_center[1] * self.scale
            self.render()

    def image_to_canvas(self, point: tuple[float, float]) -> tuple[float, float]:
        return self.offset_x + point[0] * self.scale, self.offset_y + point[1] * self.scale

    def canvas_to_image(self, point: tuple[float, float]) -> tuple[float, float]:
        return (point[0] - self.offset_x) / self.scale, (point[1] - self.offset_y) / self.scale

    def _inside_image(self, point: tuple[float, float]) -> bool:
        if self.project is None:
            return False
        return 0 <= point[0] < self.project.image.width_px and 0 <= point[1] < self.project.image.height_px

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

        self._draw_roi_box()
        self._draw_auto_candidates()
        self._draw_records()
        self._draw_drawing_in_progress()

    def _draw_roi_box(self) -> None:
        if self.roi_bbox is None or self.project is None:
            return
        x0, y0, x1, y1 = self.roi_bbox
        p0c = self.image_to_canvas((x0, y0))
        p1c = self.image_to_canvas((x1, y1))
        self.canvas.create_rectangle(*p0c, *p1c, outline="#00D7FF", width=2, dash=(6, 4))

    def _draw_auto_candidates(self) -> None:
        if not self.show_auto_candidates_var.get() or not self.auto_candidates:
            return
        for cand in self.auto_candidates:
            is_selected = cand.candidate_id == self.selected_candidate_id
            color = self.SELECTED if is_selected else (self.CANDIDATE_COLOR if cand.status != "REJECTED" else "#555555")
            for pm in cand.proposed_measurements:
                p1c = self.image_to_canvas(pm.p1)
                p2c = self.image_to_canvas(pm.p2)
                self.canvas.create_line(*p1c, *p2c, fill=color, width=2, dash=(4, 3))

    def _draw_records(self) -> None:
        if self.project is None:
            return

        for record in self.project.records:
            is_selected = record.measurement_id == self.selected_record_id
            color = self.SELECTED if is_selected else ("#0072B2" if record.is_included_in_statistics else "#777777")
            width = 4 if is_selected else 2

            if record.kind in {MeasurementKind.PROJECTED_WIDTH, MeasurementKind.DISTANCE}:
                p1c = self.image_to_canvas(record.p1)
                p2c = self.image_to_canvas(record.p2)
                dash = None if record.kind == MeasurementKind.PROJECTED_WIDTH else (4, 2)
                self.canvas.create_line(*p1c, *p2c, fill=color, width=width, dash=dash)
                for pt in (p1c, p2c):
                    self.canvas.create_oval(pt[0] - 4, pt[1] - 4, pt[0] + 4, pt[1] + 4, fill=color, outline="white")

            elif record.kind == MeasurementKind.POLYLINE_LENGTH:
                pts = [self.image_to_canvas(pt) for pt in record.geometry.get("points", [])]
                if len(pts) >= 2:
                    flat_pts = [c for pt in pts for c in pt]
                    self.canvas.create_line(*flat_pts, fill=color, width=width)
                    for pt in pts:
                        self.canvas.create_oval(pt[0] - 3, pt[1] - 3, pt[0] + 3, pt[1] + 3, fill=color, outline="white")

            elif record.kind == MeasurementKind.ANGLE:
                a = self.image_to_canvas(record.geometry.get("pt_a", (0, 0)))
                b = self.image_to_canvas(record.geometry.get("pt_b", (0, 0)))
                c = self.image_to_canvas(record.geometry.get("pt_c", (0, 0)))
                self.canvas.create_line(*a, *b, fill=color, width=width)
                self.canvas.create_line(*b, *c, fill=color, width=width)
                self.canvas.create_oval(b[0] - 5, b[1] - 5, b[0] + 5, b[1] + 5, fill="#E69F00", outline="white")

            elif record.kind in {MeasurementKind.RECTANGLE_AREA, MeasurementKind.POLYGON_AREA}:
                if record.geometry.get("bbox"):
                    bx0, by0, bx1, by1 = record.geometry["bbox"]
                    p0c = self.image_to_canvas((bx0, by0))
                    p1c = self.image_to_canvas((bx1, by1))
                    self.canvas.create_rectangle(*p0c, *p1c, outline=color, width=width, dash=(4, 2))
                elif "polygon" in record.geometry and len(record.geometry["polygon"]) >= 3:
                    pts = [self.image_to_canvas(pt) for pt in record.geometry["polygon"]]
                    flat_pts = [c for pt in pts for c in pt]
                    self.canvas.create_polygon(*flat_pts, outline=color, fill="", width=width)

            elif record.kind == MeasurementKind.INTENSITY_PROFILE:
                p1c = self.image_to_canvas(record.p1)
                p2c = self.image_to_canvas(record.p2)
                self.canvas.create_line(*p1c, *p2c, fill="#00E5FF", width=3, dash=(6, 2))

    def _draw_drawing_in_progress(self) -> None:
        tool = self.tool_var.get()
        if tool in {"polyline", "poly_area"} and self.drawing_points:
            pts = [self.image_to_canvas(pt) for pt in self.drawing_points]
            for pt in pts:
                self.canvas.create_oval(pt[0] - 4, pt[1] - 4, pt[0] + 4, pt[1] + 4, fill="#FFBF00", outline="white")
            if len(pts) >= 2:
                flat = [c for pt in pts for c in pt]
                self.canvas.create_line(*flat, fill="#FFBF00", width=2, dash=(4, 2))
        elif tool == "angle" and self.drawing_points:
            pts = [self.image_to_canvas(pt) for pt in self.drawing_points]
            for pt in pts:
                self.canvas.create_oval(pt[0] - 4, pt[1] - 4, pt[0] + 4, pt[1] + 4, fill="#E69F00", outline="white")
            if len(pts) == 2:
                self.canvas.create_line(pts[0][0], pts[0][1], pts[1][0], pts[1][1], fill="#E69F00", width=2)

    # ---------- Mouse Event Handlers & 5 Measurement Tools ----------

    def _on_left_press(self, event: tk.Event[Any]) -> None:
        if self.project is None or self.gray is None:
            return
        point = self.canvas_to_image((event.x, event.y))
        if not self._inside_image(point):
            return

        tool = self.tool_var.get()

        if tool == "select":
            record = self._hit_test_record(event.x, event.y)
            if record:
                self.select_record(record.measurement_id)
            else:
                self.select_record(None)
            return

        if tool == "auto_roi":
            self.pending_roi_start = point
            self.status_var.set("Arrastra para definir el rectángulo de la ROI...")
            return

        if tool in {"manual", "distance", "rect_area", "profile", "snap", "auto"}:
            if tool == "auto":
                self._create_one_click(point)
                return
            if self.pending_point is None:
                self.pending_point = point
                self.status_var.set("Primer punto fijado. Clic en el punto final.")
            else:
                p1 = self.pending_point
                self.pending_point = None
                self._finish_two_point_tool(tool, p1, point)

        elif tool == "polyline":
            self.drawing_points.append(point)
            self.status_var.set(f"Polilínea: {len(self.drawing_points)} punto(s). Clic para añadir, Doble Clic / Enter para finalizar.")
            self.render()

        elif tool == "poly_area":
            self.drawing_points.append(point)
            self.status_var.set(f"Polígono área: {len(self.drawing_points)} vértice(s). Doble Clic / Enter para cerrar.")
            self.render()

        elif tool == "angle":
            self.drawing_points.append(point)
            if len(self.drawing_points) == 1:
                self.status_var.set("Ángulo: Punto A fijado. Haz clic en el Vértice B.")
            elif len(self.drawing_points) == 2:
                self.status_var.set("Ángulo: Vértice B fijado. Haz clic en el Punto C.")
            elif len(self.drawing_points) == 3:
                pts = self.drawing_points
                self.drawing_points = []
                self._create_angle_record(pts[0], pts[1], pts[2])
            self.render()

    def _on_double_click_canvas(self, _event: tk.Event[Any]) -> None:
        tool = self.tool_var.get()
        if tool == "polyline" and len(self.drawing_points) >= 2:
            pts = list(self.drawing_points)
            self.drawing_points = []
            self._create_polyline_record(pts)
        elif tool == "poly_area" and len(self.drawing_points) >= 3:
            pts = list(self.drawing_points)
            self.drawing_points = []
            self._create_polygon_area_record(pts)

    def _on_enter_pressed(self) -> None:
        tool = self.tool_var.get()
        if tool == "polyline" and len(self.drawing_points) >= 2:
            pts = list(self.drawing_points)
            self.drawing_points = []
            self._create_polyline_record(pts)
        elif tool == "poly_area" and len(self.drawing_points) >= 3:
            pts = list(self.drawing_points)
            self.drawing_points = []
            self._create_polygon_area_record(pts)

    def _on_delete_pressed(self) -> None:
        if self.drawing_points:
            self.drawing_points.pop()
            self.render()
            self.status_var.set(f"Vértice eliminado ({len(self.drawing_points)} restantes).")
        else:
            self.delete_selected_records()

    def _on_left_motion(self, event: tk.Event[Any]) -> None:
        if self.tool_var.get() == "auto_roi" and self.pending_roi_start is not None:
            p_curr = self.canvas_to_image((event.x, event.y))
            x0 = int(min(self.pending_roi_start[0], p_curr[0]))
            y0 = int(min(self.pending_roi_start[1], p_curr[1]))
            x1 = int(max(self.pending_roi_start[0], p_curr[0]))
            y1 = int(max(self.pending_roi_start[1], p_curr[1]))
            self.roi_bbox = (x0, y0, x1, y1)
            self.render()

    def _on_left_release(self, _event: tk.Event[Any]) -> None:
        if self.tool_var.get() == "auto_roi" and self.pending_roi_start is not None:
            self.pending_roi_start = None
            if self.roi_bbox:
                w = self.roi_bbox[2] - self.roi_bbox[0]
                h = self.roi_bbox[3] - self.roi_bbox[1]
                self.status_var.set(f"ROI definida ({w}x{h} px). Haz clic en 'Analizar ROI'.")

    def _on_motion(self, event: tk.Event[Any]) -> None:
        if self.pending_point is None and not self.drawing_points:
            return
        self.render()
        if self.pending_point:
            start = self.image_to_canvas(self.pending_point)
            self.canvas.create_line(start[0], start[1], event.x, event.y, fill="#FFBF00", width=2, dash=(6, 3))
        elif self.drawing_points:
            last = self.image_to_canvas(self.drawing_points[-1])
            self.canvas.create_line(last[0], last[1], event.x, event.y, fill="#FFBF00", width=2, dash=(4, 2))

    # ---------- Creating 5 Scientific Tools via Commands ----------

    def _finish_two_point_tool(self, tool: str, p1: tuple[float, float], p2: tuple[float, float]) -> None:
        assert self.project is not None and self.gray is not None

        if tool == "manual":
            w_m = self.project.image.calibration.distance_m(p1, p2)
            m_id = self.project.next_measurement_id()
            rec = MeasurementRecord(
                measurement_id=m_id,
                kind=MeasurementKind.PROJECTED_WIDTH,
                name=f"Ancho {m_id}",
                status=MeasurementStatus.ACCEPTED,
                source=MeasurementSource.MANUAL,
                fiber_id=self.active_fiber_id,
                geometry={"p1": p1, "p2": p2},
                values={"width_m": w_m, "length_m": w_m},
                calibration_snapshot=asdict(self.project.image.calibration),
            )
            self._add_record_with_command(rec, f"Añadido Ancho Proyectado {m_id}")

        elif tool == "snap":
            try:
                p1_s, p2_s, conf = snap_two_click_edges(self.gray, p1, p2, footer_bounds=self.project.image.footer_bounds)
                w_m = self.project.image.calibration.distance_m(p1_s, p2_s)
                m_id = self.project.next_measurement_id()
                rec = MeasurementRecord(
                    measurement_id=m_id,
                    kind=MeasurementKind.PROJECTED_WIDTH,
                    name=f"Ancho Snap {m_id}",
                    status=MeasurementStatus.ACCEPTED,
                    source=MeasurementSource.ASSISTED,
                    fiber_id=self.active_fiber_id,
                    confidence=conf,
                    geometry={"p1": p1_s, "p2": p2_s},
                    values={"width_m": w_m, "length_m": w_m},
                    calibration_snapshot=asdict(self.project.image.calibration),
                )
                self._add_record_with_command(rec, f"Añadido Ancho Snap {m_id}")
            except Exception as exc:
                messagebox.showwarning("Ajuste fallido", str(exc), parent=self)

        elif tool == "distance":
            info = compute_line_geometry(p1, p2, self.project.image.calibration)
            m_id = self.project.next_measurement_id()
            rec = MeasurementRecord(
                measurement_id=m_id,
                kind=MeasurementKind.DISTANCE,
                name=f"Línea {m_id}",
                status=MeasurementStatus.ACCEPTED,
                source=MeasurementSource.MANUAL,
                geometry={"p1": p1, "p2": p2},
                values=info,
                calibration_snapshot=asdict(self.project.image.calibration),
            )
            self._add_record_with_command(rec, f"Añadida Línea Distancia {m_id}")

        elif tool == "rect_area":
            x0, x1 = min(p1[0], p2[0]), max(p1[0], p2[0])
            y0, y1 = min(p1[1], p2[1]), max(p1[1], p2[1])
            bbox = (int(x0), int(y0), int(x1), int(y1))
            info = compute_area_roi_geometry(self.gray, self.project.image.calibration, bbox=bbox, footer_bounds=self.project.image.footer_bounds)
            m_id = self.project.next_measurement_id()
            rec = MeasurementRecord(
                measurement_id=m_id,
                kind=MeasurementKind.RECTANGLE_AREA,
                name=f"Área Rect {m_id}",
                status=MeasurementStatus.ACCEPTED,
                source=MeasurementSource.MANUAL,
                geometry={"bbox": bbox},
                values=info,
                calibration_snapshot=asdict(self.project.image.calibration),
            )
            self._add_record_with_command(rec, f"Añadida ROI Rectángulo {m_id}")

        elif tool == "profile":
            bw = self.profile_bandwidth_var.get()
            info = compute_profile_geometry(self.gray, p1, p2, self.project.image.calibration, bandwidth_px=bw)
            m_id = self.project.next_measurement_id()
            rec = MeasurementRecord(
                measurement_id=m_id,
                kind=MeasurementKind.INTENSITY_PROFILE,
                name=f"Perfil {m_id}",
                status=MeasurementStatus.ACCEPTED,
                source=MeasurementSource.MANUAL,
                geometry={"p1": p1, "p2": p2, "bandwidth_px": bw},
                values=info,
                calibration_snapshot=asdict(self.project.image.calibration),
            )
            self._add_record_with_command(rec, f"Añadido Perfil Intensidad {m_id}")

    def _create_polyline_record(self, points: list[tuple[float, float]]) -> None:
        assert self.project is not None
        info = compute_polyline_geometry(points, self.project.image.calibration)
        m_id = self.project.next_measurement_id()
        rec = MeasurementRecord(
            measurement_id=m_id,
            kind=MeasurementKind.POLYLINE_LENGTH,
            name=f"Polilínea {m_id}",
            status=MeasurementStatus.ACCEPTED,
            source=MeasurementSource.MANUAL,
            geometry={"points": points},
            values=info,
            calibration_snapshot=asdict(self.project.image.calibration),
        )
        self._add_record_with_command(rec, f"Añadida Polilínea {m_id}")

    def _create_polygon_area_record(self, points: list[tuple[float, float]]) -> None:
        assert self.project is not None and self.gray is not None
        info = compute_area_roi_geometry(self.gray, self.project.image.calibration, polygon=points, footer_bounds=self.project.image.footer_bounds)
        m_id = self.project.next_measurement_id()
        rec = MeasurementRecord(
            measurement_id=m_id,
            kind=MeasurementKind.POLYGON_AREA,
            name=f"Área Polígono {m_id}",
            status=MeasurementStatus.ACCEPTED,
            source=MeasurementSource.MANUAL,
            geometry={"polygon": points},
            values=info,
            calibration_snapshot=asdict(self.project.image.calibration),
        )
        self._add_record_with_command(rec, f"Añadida ROI Polígono {m_id}")

    def _create_angle_record(self, p_a: tuple[float, float], p_b: tuple[float, float], p_c: tuple[float, float]) -> None:
        assert self.project is not None
        info = compute_angle_geometry(p_a, p_b, p_c, self.project.image.calibration)
        m_id = self.project.next_measurement_id()
        rec = MeasurementRecord(
            measurement_id=m_id,
            kind=MeasurementKind.ANGLE,
            name=f"Ángulo {m_id}",
            status=MeasurementStatus.ACCEPTED,
            source=MeasurementSource.MANUAL,
            geometry={"pt_a": p_a, "pt_b": p_b, "pt_c": p_c},
            values=info,
            calibration_snapshot=asdict(self.project.image.calibration),
        )
        self._add_record_with_command(rec, f"Añadido Ángulo {m_id}")

    def _create_one_click(self, point: tuple[float, float]) -> None:
        assert self.project is not None and self.gray is not None
        try:
            p1, p2, confidence = one_click_measurement(
                self.gray, point, self.search_radius_var.get(), footer_bounds=self.project.image.footer_bounds
            )
            w_m = self.project.image.calibration.distance_m(p1, p2)
            m_id = self.project.next_measurement_id()
            rec = MeasurementRecord(
                measurement_id=m_id,
                kind=MeasurementKind.PROJECTED_WIDTH,
                name=f"Ancho Auto {m_id}",
                status=MeasurementStatus.ACCEPTED,
                source=MeasurementSource.ASSISTED,
                fiber_id=self.active_fiber_id,
                confidence=confidence,
                geometry={"p1": p1, "p2": p2},
                values={"width_m": w_m, "length_m": w_m},
                calibration_snapshot=asdict(self.project.image.calibration),
            )
            self._add_record_with_command(rec, f"Añadida Propuesta Auto {m_id}")
        except Exception as exc:
            messagebox.showwarning("Propuesta asistida no válida", str(exc), parent=self)

    def _add_record_with_command(self, record: MeasurementRecord, desc: str) -> None:
        assert self.project is not None
        rec = record

        def execute() -> None:
            self.project.records.append(rec)
            self.select_record(rec.measurement_id)

        def undo() -> None:
            self.project.records = [r for r in self.project.records if r.measurement_id != rec.measurement_id]
            if self.selected_record_id == rec.measurement_id:
                self.selected_record_id = None

        cmd = Command(description=desc, execute_fn=execute, undo_fn=undo, affected_ids=[rec.measurement_id])
        self.history.push_and_execute(cmd)
        self._mark_dirty()

        if rec.kind == MeasurementKind.PROJECTED_WIDTH and self.project.is_fiber_complete(self.active_fiber_id) and self.auto_advance_var.get():
            self.status_var.set(f"¡Protocolo {self.active_fiber_id} completo! Avanzando automáticamente.")
            self.next_fiber()
        else:
            self._refresh_all()
            val_str = format_length_m(rec.primary_value) if rec.primary_value is not None else "OK"
            self.status_var.set(f"{rec.name}: {val_str}")

    def _hit_test_record(self, canvas_x: float, canvas_y: float) -> MeasurementRecord | None:
        if self.project is None:
            return None
        threshold = 12.0
        best: tuple[float, MeasurementRecord] | None = None

        for record in self.project.records:
            if record.kind in {MeasurementKind.PROJECTED_WIDTH, MeasurementKind.DISTANCE}:
                p1 = np.asarray(self.image_to_canvas(record.p1))
                p2 = np.asarray(self.image_to_canvas(record.p2))
                point = np.asarray((canvas_x, canvas_y))
                d1 = float(np.linalg.norm(point - p1))
                d2 = float(np.linalg.norm(point - p2))
                min_d = min(d1, d2)
                if min_d <= threshold and (best is None or min_d < best[0]):
                    best = (min_d, record)

        return best[1] if best else None

    # ---------- Linked Selection & Results Table ----------

    def select_record(self, record_id: str | None) -> None:
        self.selected_record_id = record_id
        record = self._selected_record()
        if record:
            if record.fiber_id:
                self.active_fiber_id = record.fiber_id
                self.lbl_active_fiber.configure(text=record.fiber_id)
            if self.results_tree.exists(record_id):
                self.results_tree.selection_set(record_id)
                self.results_tree.see(record_id)

        self._refresh_inspector()
        self.render()

    def _on_results_tree_select(self, _event: tk.Event[Any]) -> None:
        sel = self.results_tree.selection()
        if sel:
            self.select_record(sel[0])

    def _on_results_tree_double_click(self, _event: tk.Event[Any]) -> None:
        self.center_selection()

    def _sort_results_tree(self, col: str) -> None:
        items = [(self.results_tree.set(k, col), k) for k in self.results_tree.get_children("")]
        items.sort()
        for idx, (_val, k) in enumerate(items):
            self.results_tree.move(k, "", idx)

    def select_fiber(self, fiber_id: str) -> None:
        self.active_fiber_id = fiber_id
        if self.project:
            self.project.active_fiber_id = fiber_id
        self.lbl_active_fiber.configure(text=fiber_id)
        self._update_protocol_display()
        self.render()

    def new_fiber(self) -> None:
        if self.project:
            new_id = self.project.get_next_fiber_id()
            self._mark_dirty()
            self.select_fiber(new_id)

    def prev_fiber(self) -> None:
        if not self.project:
            return
        fibers = sorted({r.fiber_id for r in self.project.records if r.fiber_id} | {self.active_fiber_id})
        idx = fibers.index(self.active_fiber_id) if self.active_fiber_id in fibers else 0
        self.select_fiber(fibers[max(0, idx - 1)])

    def next_fiber(self) -> None:
        if not self.project:
            return
        fibers = sorted({r.fiber_id for r in self.project.records if r.fiber_id} | {self.active_fiber_id})
        idx = fibers.index(self.active_fiber_id) if self.active_fiber_id in fibers else 0
        if idx + 1 < len(fibers):
            self.select_fiber(fibers[idx + 1])
        else:
            self.new_fiber()

    def _update_protocol_display(self) -> None:
        if not self.project:
            return
        target = self.project.target_sections
        accepted = len(self.project.accepted_fiber_measurements(self.active_fiber_id))
        t_str = "∞" if target <= 0 else str(target)
        status = "✓ Completo" if self.project.is_fiber_complete(self.active_fiber_id) else "Incompleto"
        self.protocol_status_var.set(f"Protocolo: {accepted} / {t_str} ({status})")

    # ---------- Batch Table Actions & Metadata Editing ----------

    def accept_selected_records(self) -> None:
        sel = list(self.results_tree.selection())
        if not sel:
            return
        old_states = {r_id: self._record_by_id(r_id).status for r_id in sel if self._record_by_id(r_id)}

        def execute() -> None:
            for r_id in sel:
                r = self._record_by_id(r_id)
                if r:
                    r.status = MeasurementStatus.ACCEPTED

        def undo() -> None:
            for r_id, st in old_states.items():
                r = self._record_by_id(r_id)
                if r:
                    r.status = st

        cmd = Command(description=f"Aceptadas {len(sel)} mediciones", execute_fn=execute, undo_fn=undo, affected_ids=sel)
        self.history.push_and_execute(cmd)
        self._mark_dirty()
        self._refresh_all()

    def reject_selected_records(self) -> None:
        sel = list(self.results_tree.selection())
        if not sel:
            return
        old_states = {r_id: self._record_by_id(r_id).status for r_id in sel if self._record_by_id(r_id)}

        def execute() -> None:
            for r_id in sel:
                r = self._record_by_id(r_id)
                if r:
                    r.status = MeasurementStatus.REJECTED

        def undo() -> None:
            for r_id, st in old_states.items():
                r = self._record_by_id(r_id)
                if r:
                    r.status = st

        cmd = Command(description=f"Rechazadas {len(sel)} mediciones", execute_fn=execute, undo_fn=undo, affected_ids=sel)
        self.history.push_and_execute(cmd)
        self._mark_dirty()
        self._refresh_all()

    def delete_selected_records(self) -> None:
        sel = list(self.results_tree.selection())
        if not sel and self.selected_record_id:
            sel = [self.selected_record_id]
        if not sel or not self.project:
            return

        deleted_records = [r for r in self.project.records if r.measurement_id in sel]

        def execute() -> None:
            self.project.records = [r for r in self.project.records if r.measurement_id not in sel]
            self.selected_record_id = None

        def undo() -> None:
            self.project.records.extend(deleted_records)

        cmd = Command(description=f"Eliminadas {len(deleted_records)} mediciones", execute_fn=execute, undo_fn=undo, affected_ids=sel)
        self.history.push_and_execute(cmd)
        self._mark_dirty()
        self._refresh_all()

    def edit_selected_metadata(self) -> None:
        record = self._selected_record()
        if not record:
            return

        new_name = simpledialog.askstring("Editar Nombre", "Nombre:", initialvalue=record.name, parent=self)
        if new_name is None:
            return
        new_tags_str = simpledialog.askstring("Editar Tags", "Tags (separados por coma):", initialvalue=", ".join(record.tags), parent=self)
        if new_tags_str is None:
            return
        new_notes = simpledialog.askstring("Editar Notas", "Notas:", initialvalue=record.notes, parent=self)
        if new_notes is None:
            return

        old_name, old_tags, old_notes = record.name, list(record.tags), record.notes
        new_tags = normalize_tags([new_tags_str])

        def execute() -> None:
            record.name = new_name
            record.tags = new_tags
            record.notes = new_notes

        def undo() -> None:
            record.name = old_name
            record.tags = old_tags
            record.notes = old_notes

        cmd = Command(description=f"Metadata editada en {record.measurement_id}", execute_fn=execute, undo_fn=undo, affected_ids=[record.measurement_id])
        self.history.push_and_execute(cmd)
        self._mark_dirty()
        self._refresh_all()

    # ---------- Auto-ROI Pipeline Controls ----------

    def run_auto_roi(self) -> None:
        if not self.project or self.gray is None or not self.roi_bbox:
            messagebox.showwarning("Sin ROI", "Dibuja una ROI antes de analizar.", parent=self)
            return

        pol_map = {"Automática": "auto", "Clara": "bright", "Oscura": "dark"}
        polarity = pol_map.get(self.roi_polarity_var.get(), "auto")

        try:
            candidates, summary = analyze_roi(
                gray=self.gray,
                roi_bbox=self.roi_bbox,
                calibration=self.project.image.calibration,
                footer_bounds=self.project.image.footer_bounds,
                polarity=polarity,
                threshold_method=self.roi_thresh_var.get(),
                min_area_px=self.roi_min_area_var.get(),
                min_elongation=self.roi_min_elong_var.get(),
                min_width_px=self.roi_min_width_var.get(),
                n_sections=self.roi_n_sections_var.get(),
                allow_curved_trace=self.roi_curved_var.get(),
            )
            self.auto_candidates = candidates
            self.auto_summary = summary
            self.selected_candidate_id = candidates[0].candidate_id if candidates else None

            self.roi_summary_var.set(
                f"Thresh: {summary.threshold_method_used} | Res: {summary.resolution_status}\n"
                f"Comp: {summary.total_components} | Medibles: {summary.measurable_candidates} | Alta Conf: {summary.high_confidence}"
            )
            self._refresh_auto_tree()
            self.notebook.select(2)
            self.render()
            self.status_var.set(f"Análisis ROI completado: {len(candidates)} candidatos.")
        except Exception as exc:
            messagebox.showerror("Error ROI", str(exc), parent=self)

    def clear_roi(self) -> None:
        self.roi_bbox = None
        self.auto_candidates = []
        self.auto_summary = None
        self.selected_candidate_id = None
        self._refresh_auto_tree()
        self.render()
        self.status_var.set("ROI limpiada.")

    def accept_candidate(self) -> None:
        if not self.project or not self.selected_candidate_id:
            return
        cand = next((c for c in self.auto_candidates if c.candidate_id == self.selected_candidate_id), None)
        if not cand or cand.status == "ACCEPTED":
            return

        fiber_id = self.project.get_next_fiber_id()
        added_recs = []

        for pm in cand.proposed_measurements:
            m_id = self.project.next_measurement_id()
            w_m = pm.width_m
            rec = MeasurementRecord(
                measurement_id=m_id,
                kind=MeasurementKind.PROJECTED_WIDTH,
                name=f"Ancho Auto {m_id}",
                status=MeasurementStatus.ACCEPTED,
                source=MeasurementSource.AUTO_ROI_COMPONENT,
                fiber_id=fiber_id,
                confidence=cand.confidence_score,
                geometry={"p1": pm.p1, "p2": pm.p2},
                values={"width_m": w_m, "length_m": w_m},
                calibration_snapshot=asdict(self.project.image.calibration),
            )
            added_recs.append(rec)

        def execute() -> None:
            self.project.records.extend(added_recs)
            cand.status = "ACCEPTED"

        def undo() -> None:
            added_ids = {r.measurement_id for r in added_recs}
            self.project.records = [r for r in self.project.records if r.measurement_id not in added_ids]
            cand.status = "PENDING"

        cmd = Command(description=f"Candidato {cand.candidate_id} aceptado como {fiber_id}", execute_fn=execute, undo_fn=undo, affected_ids=[r.measurement_id for r in added_recs])
        self.history.push_and_execute(cmd)
        self.select_fiber(fiber_id)
        self._mark_dirty()
        self._refresh_all()

    def reject_candidate(self) -> None:
        cand = next((c for c in self.auto_candidates if c.candidate_id == self.selected_candidate_id), None)
        if cand:
            cand.status = "REJECTED"
            self._refresh_auto_tree()
            self.render()

    def classify(self) -> None:
        if not self.project or not self.project.records:
            return
        mode = self.classification_var.get()
        k = None if mode == "Auto" else int(mode)
        mapping = classify_fibers(self.project.measurements, requested_k=k)
        self._mark_dirty()
        self._refresh_all()
        self.status_var.set(f"Clasificación aplicada: {len(set(mapping.values()))} grupo(s).")

    # ---------- Exports ----------

    def export_csv_dialog(self) -> None:
        if not self.project:
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="measurements_unified.csv")
        if path:
            export_csv(self.project, path)
            self.status_var.set(f"CSV exportado: {path}")

    def export_annotated_dialog(self) -> None:
        if not self.project or self.source_image is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", initialfile="annotated.png")
        if path:
            export_annotated(self.project, self.source_image, path)
            self.status_var.set(f"Imagen anotada exportada: {path}")

    def export_report_dialog(self) -> None:
        if not self.project or self.source_image is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".html", initialfile="report.html")
        if path:
            r_path = Path(path)
            a_path = r_path.with_name(r_path.stem + "_annotated.png")
            export_annotated(self.project, self.source_image, a_path)
            export_html_report(self.project, a_path.name, r_path)
            self.status_var.set(f"Informe HTML exportado: {r_path}")

    # ---------- Refreshes ----------

    def _refresh_all(self) -> None:
        self._update_protocol_display()
        self._refresh_results_table()
        self._refresh_fiber_tree()
        self._refresh_auto_tree()
        self._refresh_inspector()
        self._refresh_histogram()
        self._refresh_metadata()
        self.render()

    def _refresh_results_table(self) -> None:
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        if not self.project:
            return

        text_filter = self.filter_text_var.get().lower().strip()
        kind_filter = self.filter_kind_var.get()

        for r in self.project.records:
            if kind_filter != "Todos" and r.kind != kind_filter:
                continue
            if text_filter and text_filter not in r.name.lower() and text_filter not in r.measurement_id.lower() and text_filter not in (r.fiber_id or "").lower():
                continue

            val_str = format_length_m(r.primary_value) if r.primary_value is not None else "—"
            self.results_tree.insert(
                "",
                tk.END,
                iid=r.measurement_id,
                values=(
                    r.measurement_id,
                    r.name,
                    r.kind.value,
                    r.fiber_id or "—",
                    val_str,
                    r.primary_unit,
                    r.source.value,
                    r.status.value,
                    ", ".join(r.tags),
                ),
            )
        if self.selected_record_id and self.results_tree.exists(self.selected_record_id):
            self.results_tree.selection_set(self.selected_record_id)

    def _refresh_fiber_tree(self) -> None:
        for item in self.fiber_tree.get_children():
            self.fiber_tree.delete(item)
        if not self.project:
            return

        fibers_stats = fiber_statistics(self.project.measurements)
        all_fiber_ids = sorted({r.fiber_id for r in self.project.records if r.fiber_id} | {self.active_fiber_id})

        for fid in all_fiber_ids:
            tot = len(self.project.fiber_measurements(fid))
            accepted = len(self.project.accepted_fiber_measurements(fid))
            info = fibers_stats.get(fid)

            med_str = f"{info['median_m'] * 1e6:.3f} µm" if info and accepted > 0 else "—"
            min_str = f"{info['min_m'] * 1e6:.3f} µm" if info and accepted > 0 else "—"
            max_str = f"{info['max_m'] * 1e6:.3f} µm" if info and accepted > 0 else "—"
            status = "✓ Completo" if self.project.is_fiber_complete(fid) else "Incompleto"

            self.fiber_tree.insert("", tk.END, iid=fid, values=(fid, f"{accepted}/{tot}", med_str, min_str, max_str, status, "—"))

    def _refresh_auto_tree(self) -> None:
        for item in self.auto_tree.get_children():
            self.auto_tree.delete(item)
        for cand in self.auto_candidates:
            med_str = format_length_m(cand.median_width_m) if cand.median_width_m else "—"
            cv_str = f"{cand.cv_width * 100:.1f}%" if cand.cv_width is not None else "—"
            disc_str = f"{cand.mean_discrepancy * 100:.1f}%" if cand.mean_discrepancy is not None else "—"
            self.auto_tree.insert("", tk.END, iid=cand.candidate_id, values=(cand.candidate_id, cand.status, len(cand.proposed_measurements), med_str, cv_str, disc_str, f"{cand.confidence_score:.2f}", ", ".join(sorted(cand.quality_flags))))

    def _refresh_inspector(self) -> None:
        lines = []
        lines.append("INSPECTOR CIENTÍFICO DE REGISTRO")
        lines.append("=" * 40)
        rec = self._selected_record()

        if rec:
            lines.append(f"ID: {rec.measurement_id}")
            lines.append(f"Nombre: {rec.name}")
            lines.append(f"Tipo: {rec.kind.value}")
            lines.append(f"Estado: {rec.status.value}")
            lines.append(f"Fuente: {rec.source.value}")
            lines.append(f"Fibra: {rec.fiber_id or '—'}")
            lines.append(f"Tags: {', '.join(rec.tags) if rec.tags else 'ninguno'}")
            lines.append(f"Notas: {rec.notes or '—'}")
            lines.append(f"Quality Flags: {', '.join(rec.quality_flags) if rec.quality_flags else 'ninguno'}")
            lines.append(f"Creado: {rec.created_at}")
            lines.append("")
            lines.append("MAGNITUDES DERIVADAS")
            lines.append("-" * 40)

            for k, v in rec.values.items():
                if isinstance(v, float):
                    lines.append(f"  {k}: {v:.6g}")
                else:
                    lines.append(f"  {k}: {v}")
        else:
            lines.append("Ningún registro seleccionado.")

        self.inspector_text.configure(state=tk.NORMAL)
        self.inspector_text.delete("1.0", tk.END)
        self.inspector_text.insert("1.0", "\n".join(lines))
        self.inspector_text.configure(state=tk.DISABLED)

    def _refresh_histogram(self) -> None:
        if not self.project:
            self.hist_canvas.delete("all")
            return
        hist_data = compute_histogram_data(self.project.measurements, mode=self.hist_mode_var.get(), n_bins=self.hist_bins_var.get())
        vals = hist_data["values"]
        self.hist_info_lbl.configure(text=f"N = {vals.size} | Mediana: {format_length_m(hist_data['median_m']) if hist_data['median_m'] else '—'}")

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
            }
            self.metadata_text.insert("1.0", json.dumps(payload, indent=2, ensure_ascii=False))
        self.metadata_text.configure(state=tk.DISABLED)

    def _clear_pending(self) -> None:
        self.pending_point = None
        self.drawing_points = []
        self.render()

    def _selected_record(self) -> MeasurementRecord | None:
        return self._record_by_id(self.selected_record_id)

    def _record_by_id(self, r_id: str | None) -> MeasurementRecord | None:
        if not self.project or not r_id:
            return None
        return next((r for r in self.project.records if r.measurement_id == r_id), None)

    def _on_fiber_tree_select(self, _event: tk.Event[Any]) -> None:
        sel = self.fiber_tree.selection()
        if sel:
            self.select_fiber(sel[0])

    def _on_auto_tree_select(self, _event: tk.Event[Any]) -> None:
        sel = self.auto_tree.selection()
        if sel:
            self.selected_candidate_id = sel[0]

    def show_protocol(self) -> None:
        messagebox.showinfo("Protocolo", "Selecciona una herramienta (M, D, P, G, R, Y, L, I) para medir sobre la imagen.", parent=self)

    def show_about(self) -> None:
        messagebox.showinfo("Acerca de", "Fathom Fibers Quick 0.2 — Scientific Measurement Workspace", parent=self)


def main() -> None:
    initial_path = sys.argv[1] if len(sys.argv) > 1 else None
    app = FiberQuickApp(initial_path)
    app.mainloop()


if __name__ == "__main__":
    main()
