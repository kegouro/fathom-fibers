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
    compute_histogram_data,
    fiber_statistics,
    format_length_m,
    get_fiber_extrema,
    one_click_measurement,
    snap_two_click_edges,
    validate_measurement_geometry,
)
from .auto_roi import (
    PRESET_HIGH_MAG_FINE,
    PRESET_LOW_MAG_NETWORK,
    PRESET_MID_MAG_GENERAL,
    AutoFiberCandidate,
    AutoROISummary,
    analyze_roi,
    check_resolution_resolvability,
    get_preset_for_calibration,
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
    CANDIDATE_COLOR = "#00E5FF"

    def __init__(self, initial_path: str | None = None) -> None:
        super().__init__()
        self.title("Fathom Fibers Quick 0.1 — Manual-first SEM measurement")
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

        # Interactive selection state
        self.selected_measurement_id: str | None = None
        self.active_fiber_id: str = "F001"
        self.drag_endpoint: int | None = None
        self.dragging_measurement_id: str | None = None
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
        self.classification_var = tk.StringVar(value="Auto")
        self.footer_visible_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Abre un TIFF Zeiss para comenzar.")
        self.calibration_var = tk.StringVar(value="Sin imagen")
        self.protocol_status_var = tk.StringVar(value="Protocolo: 0 / 5 (Incompleto)")

        # Auto ROI variables
        self.show_auto_candidates_var = tk.BooleanVar(value=True)
        self.roi_preset_var = tk.StringVar(value="MID_MAG_GENERAL")
        self.roi_thresh_var = tk.StringVar(value="Automático")
        self.roi_polarity_var = tk.StringVar(value="Automática")
        self.roi_min_area_var = tk.IntVar(value=35)
        self.roi_min_elong_var = tk.DoubleVar(value=2.2)
        self.roi_min_width_var = tk.DoubleVar(value=2.0)
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
        file_menu.add_command(label="Abrir imagen… (Ctrl+O)", command=self.open_image_dialog, accelerator="Ctrl+O")
        file_menu.add_command(label="Abrir proyecto…", command=self.open_project_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Guardar proyecto… (Ctrl+S)", command=self.save_project_dialog, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="Exportar CSV…", command=self.export_csv_dialog)
        file_menu.add_command(label="Exportar imagen anotada… (Ctrl+E)", command=self.export_annotated_dialog, accelerator="Ctrl+E")
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
        ttk.Label(parent, text="MVP manual y asistido + ROI Auto", foreground="#666").pack(anchor=tk.W, pady=(0, 4))

        # Protocol Panel
        proto_box = ttk.LabelFrame(parent, text="PROTOCOLO DE MEDICIÓN", padding=6)
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
        ttk.Label(row_fiber, text="Fibra actual:", font=("TkDefaultFont", 10, "bold")).pack(side=tk.LEFT)
        self.lbl_active_fiber = ttk.Label(row_fiber, text="F001", font=("TkDefaultFont", 11, "bold"), foreground="#0072B2")
        self.lbl_active_fiber.pack(side=tk.LEFT, padx=6)
        ttk.Button(row_fiber, text="+ Nueva (N)", command=self.new_fiber, width=10).pack(side=tk.RIGHT)

        ttk.Label(proto_box, textvariable=self.protocol_status_var, font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W, pady=2)

        nav_row = ttk.Frame(proto_box)
        nav_row.pack(fill=tk.X, pady=4)
        ttk.Button(nav_row, text="◀ Anter. (PgUp)", command=self.prev_fiber, width=13).pack(side=tk.LEFT, expand=True)
        ttk.Button(nav_row, text="Siguiente ▶ (PgDn)", command=self.next_fiber, width=13).pack(side=tk.RIGHT, expand=True)

        ttk.Checkbutton(
            proto_box,
            text="Avanzar auto al completar",
            variable=self.auto_advance_var,
        ).pack(anchor=tk.W, pady=2)

        # Tools Panel
        tools = ttk.LabelFrame(parent, text="Herramienta", padding=6)
        tools.pack(fill=tk.X, pady=4)
        choices = [
            ("Seleccionar / editar (V)", "select"),
            ("ROI automática (I)", "auto_roi"),
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

        # Auto ROI Control Panel (Updated with Presets & Parameters)
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

        pol_row = ttk.Frame(roi_box)
        pol_row.pack(fill=tk.X, pady=2)
        ttk.Label(pol_row, text="Polaridad:").pack(side=tk.LEFT)
        ttk.Combobox(
            pol_row,
            textvariable=self.roi_polarity_var,
            values=("Automática", "Clara", "Oscura"),
            state="readonly",
            width=14,
        ).pack(side=tk.RIGHT)

        ttk.Checkbutton(roi_box, text="Permitir trazado curvo", variable=self.roi_curved_var).pack(anchor=tk.W, pady=2)

        preset_btn_row = ttk.Frame(roi_box)
        preset_btn_row.pack(fill=tk.X, pady=2)
        ttk.Button(preset_btn_row, text="Guardar Preset", command=self.save_preset_json).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        ttk.Button(preset_btn_row, text="Cargar Preset", command=self.load_preset_json).pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=1)

        roi_btns = ttk.Frame(roi_box)
        roi_btns.pack(fill=tk.X, pady=(4, 2))
        ttk.Button(roi_btns, text="Analizar ROI", command=self.run_auto_roi).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        ttk.Button(roi_btns, text="Limpiar ROI", command=self.clear_roi).pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=1)

        # Feedback box
        ttk.Label(roi_box, textvariable=self.roi_feedback_var, font=("TkDefaultFont", 8, "italic"), foreground="#444", wraplength=210).pack(anchor=tk.W, pady=2)

        # Display & View Panel
        display = ttk.LabelFrame(parent, text="Vista y Canvas", padding=6)
        display.pack(fill=tk.X, pady=4)
        ttk.Checkbutton(display, text="Mostrar todas las fibras", variable=self.show_all_sections_var, command=self.render).pack(anchor=tk.W)
        ttk.Checkbutton(display, text="Mostrar candidatos ROI", variable=self.show_auto_candidates_var, command=self.render).pack(anchor=tk.W)
        ttk.Checkbutton(display, text="Mostrar zona del footer", variable=self.footer_visible_var, command=self.render).pack(anchor=tk.W)

        btn_row = ttk.Frame(display)
        btn_row.pack(fill=tk.X, pady=2)
        ttk.Button(btn_row, text="Ajustar (F)", command=self.fit_image).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        ttk.Button(btn_row, text="Centrar (C)", command=self.center_selection).pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=1)

        # Review Panel
        review = ttk.LabelFrame(parent, text="Revisión de Medición", padding=6)
        review.pack(fill=tk.X, pady=4)
        ttk.Label(review, text="Defecto / observación:").pack(anchor=tk.W)
        defect_combo = ttk.Combobox(
            review,
            textvariable=self.defect_var,
            state="readonly",
            values=("None", "Bead", "Constriction", "Fused", "Debris", "Ribbon-like", "Ambiguous", "Other"),
        )
        defect_combo.pack(fill=tk.X, pady=2)
        ttk.Button(review, text="Aplicar defecto", command=self.apply_defect).pack(fill=tk.X, pady=2)

        rev_btns = ttk.Frame(review)
        rev_btns.pack(fill=tk.X, pady=2)
        ttk.Button(rev_btns, text="Aceptar / Rechazar (R)", command=self.toggle_selected_acceptance).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        ttk.Button(rev_btns, text="Eliminar (Del)", command=self.delete_selected).pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=1)

        # Classification Panel (Updated per Section 15)
        classify = ttk.LabelFrame(parent, text="Familias de Tamaño", padding=6)
        classify.pack(fill=tk.X, pady=4)
        ttk.Label(classify, text="Agrupamiento automático experimental", font=("TkDefaultFont", 8, "italic"), foreground="#666").pack(anchor=tk.W)
        ttk.Combobox(
            classify,
            textvariable=self.classification_var,
            state="readonly",
            values=("Auto", "1", "2", "3", "4"),
            width=8,
        ).pack(fill=tk.X, pady=2)
        ttk.Button(classify, text="Clasificar fibras", command=self.classify).pack(fill=tk.X, pady=(2, 0))

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
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        fibers_tab = ttk.Frame(self.notebook, padding=4)
        measurements_tab = ttk.Frame(self.notebook, padding=4)
        auto_tab = ttk.Frame(self.notebook, padding=4)
        histogram_tab = ttk.Frame(self.notebook, padding=4)
        inspector_tab = ttk.Frame(self.notebook, padding=6)
        metadata_tab = ttk.Frame(self.notebook, padding=6)

        self.notebook.add(fibers_tab, text="Fibras")
        self.notebook.add(measurements_tab, text="Mediciones")
        self.notebook.add(auto_tab, text="Revisión ROI")
        self.notebook.add(histogram_tab, text="Histograma")
        self.notebook.add(inspector_tab, text="Inspector")
        self.notebook.add(metadata_tab, text="Metadata")

        # Tab 1: Fibers Tree
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

        # Tab 2: Measurements Tree
        columns = ("id", "fiber", "width", "method", "group", "defect", "ok")
        self.tree = ttk.Treeview(measurements_tab, columns=columns, show="headings", height=28)
        labels = {"id": "ID", "fiber": "Fibra", "width": "Ancho proyectado", "method": "Método", "group": "Grupo", "defect": "Defecto", "ok": "OK"}
        widths = {"id": 60, "fiber": 60, "width": 105, "method": 90, "group": 50, "defect": 85, "ok": 35}
        for column in columns:
            self.tree.heading(column, text=labels[column])
            self.tree.column(column, width=widths[column], anchor=tk.CENTER, stretch=column in {"method", "defect"})
        scroll = ttk.Scrollbar(measurements_tab, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.bind("<<TreeviewSelect>>", self._tree_selection)
        self.tree.bind("<Double-1>", self._on_tree_double_click)

        # Tab 3: Auto-ROI Review Queue (Enhanced per Section L)
        self._build_auto_review_tab(auto_tab)

        # Tab 4: Interactive Histogram
        self._build_histogram_tab(histogram_tab)

        # Tab 5: Inspector
        self.inspector_text = tk.Text(inspector_tab, width=44, height=40, wrap=tk.WORD, state=tk.DISABLED)
        self.inspector_text.pack(fill=tk.BOTH, expand=True)

        # Tab 6: Metadata
        self.metadata_text = tk.Text(metadata_tab, width=44, height=40, wrap=tk.NONE, state=tk.DISABLED)
        self.metadata_text.pack(fill=tk.BOTH, expand=True)

    def _build_auto_review_tab(self, parent: ttk.Frame) -> None:
        summary_lbl = ttk.Label(parent, textvariable=self.roi_summary_var, font=("TkDefaultFont", 9, "bold"), padding=4)
        summary_lbl.pack(anchor=tk.W)

        auto_cols = ("id", "status", "sections", "median", "cv_width", "discrepancy", "confidence", "flags")
        self.auto_tree = ttk.Treeview(parent, columns=auto_cols, show="headings", height=18)
        auto_headers = {
            "id": "ID Candidate",
            "status": "Estado",
            "sections": "Secciones",
            "median": "Mediana",
            "cv_width": "CV Ancho",
            "discrepancy": "Discrepancia",
            "confidence": "Confianza",
            "flags": "Quality Flags",
        }
        auto_widths = {"id": 85, "status": 75, "sections": 65, "median": 80, "cv_width": 75, "discrepancy": 85, "confidence": 95, "flags": 140}
        for col in auto_cols:
            self.auto_tree.heading(col, text=auto_headers[col], command=lambda _c=col: self._sort_auto_tree(_c))
            self.auto_tree.column(col, width=auto_widths[col], anchor=tk.CENTER, stretch=(col == "flags"))

        a_scroll = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.auto_tree.yview)
        self.auto_tree.configure(yscrollcommand=a_scroll.set)
        self.auto_tree.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        a_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.auto_tree.bind("<<TreeviewSelect>>", self._on_auto_tree_select)

        ctrl_frame = ttk.Frame(parent, padding=4)
        ctrl_frame.pack(fill=tk.X, side=tk.BOTTOM)

        row_nav = ttk.Frame(ctrl_frame)
        row_nav.pack(fill=tk.X, pady=2)
        ttk.Button(row_nav, text="Ir al candidato", command=self.goto_selected_candidate).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        ttk.Button(row_nav, text="ROI alrededor", command=self.roi_around_candidate).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        ttk.Button(row_nav, text="Medir manualmente aquí", command=self.manual_measure_here).pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=1)

        row1 = ttk.Frame(ctrl_frame)
        row1.pack(fill=tk.X, pady=2)
        ttk.Button(row1, text="✓ Aceptar candidato", command=self.accept_candidate).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        ttk.Button(row1, text="✗ Rechazar candidato", command=self.reject_candidate).pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=1)

        row2 = ttk.Frame(ctrl_frame)
        row2.pack(fill=tk.X, pady=2)
        ttk.Button(row2, text="Aceptar candidatos de Alta Confianza", command=self.accept_high_confidence_candidates).pack(fill=tk.X, pady=1)
        ttk.Button(row2, text="Rechazar restantes", command=self.reject_remaining_candidates).pack(fill=tk.X, pady=1)

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

        btn_clear = ttk.Button(ctrls, text="Limpiar filtro", command=self.clear_histogram_filter)
        btn_clear.pack(side=tk.RIGHT, padx=4)

        self.hist_info_lbl = ttk.Label(parent, text="N = 0", font=("TkDefaultFont", 9, "bold"))
        self.hist_info_lbl.pack(anchor=tk.W, padx=6, pady=2)

        self.hist_canvas = tk.Canvas(parent, background="#222222", height=320, highlightthickness=0)
        self.hist_canvas.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.hist_canvas.bind("<ButtonPress-1>", self._on_histogram_click)

    def _bind_shortcuts(self) -> None:
        self.bind_all("<Control-o>", lambda _event: self.open_image_dialog())
        self.bind_all("<Control-s>", lambda _event: self.save_project_dialog())
        self.bind_all("<Control-e>", lambda _event: self.export_annotated_dialog())
        self.bind_all("<Key-v>", lambda event: self._shortcut_guard(event, lambda: self.tool_var.set("select")))
        self.bind_all("<Key-i>", lambda event: self._shortcut_guard(event, lambda: self.tool_var.set("auto_roi")))
        self.bind_all("<Key-m>", lambda event: self._shortcut_guard(event, lambda: self.tool_var.set("manual")))
        self.bind_all("<Key-s>", lambda event: self._shortcut_guard(event, lambda: self.tool_var.set("snap")))
        self.bind_all("<Key-a>", lambda event: self._shortcut_guard(event, lambda: self.tool_var.set("auto")))
        self.bind_all("<Key-n>", lambda event: self._shortcut_guard(event, self.new_fiber))
        self.bind_all("<Prior>", lambda _event: self.prev_fiber())
        self.bind_all("<Next>", lambda _event: self.next_fiber())
        self.bind_all("<Return>", lambda event: self._shortcut_guard(event, self._on_enter_pressed))
        self.bind_all("<Key-r>", lambda event: self._shortcut_guard(event, self.toggle_selected_acceptance))
        self.bind_all("<Delete>", lambda _event: self.delete_selected())
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

    def save_preset_json(self) -> None:
        path = filedialog.asksaveasfilename(title="Guardar Preset JSON", defaultextension=".json", filetypes=[("JSON", "*.json")])
        if path:
            data = {
                "name": self.roi_preset_var.get(),
                "min_area_px": self.roi_min_area_var.get(),
                "min_elongation": self.roi_min_elong_var.get(),
                "min_width_px": self.roi_min_width_var.get(),
                "n_sections": self.roi_n_sections_var.get(),
                "threshold_method": self.roi_thresh_var.get(),
                "polarity": self.roi_polarity_var.get(),
            }
            Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
            self.status_var.set(f"Preset guardado: {path}")

    def load_preset_json(self) -> None:
        path = filedialog.askopenfilename(title="Cargar Preset JSON", filetypes=[("JSON", "*.json")])
        if path:
            try:
                data = json.loads(Path(path).read_text(encoding="utf-8"))
                self.roi_preset_var.set(data.get("name", "Personalizado"))
                self.roi_min_area_var.set(data.get("min_area_px", 35))
                self.roi_min_elong_var.set(data.get("min_elongation", 2.2))
                self.roi_min_width_var.set(data.get("min_width_px", 2.0))
                self.roi_n_sections_var.set(data.get("n_sections", 3))
                if "threshold_method" in data:
                    self.roi_thresh_var.set(data["threshold_method"])
                if "polarity" in data:
                    self.roi_polarity_var.set(data["polarity"])
                self.status_var.set(f"Preset cargado desde: {path}")
            except Exception as exc:
                messagebox.showerror("Error al cargar preset", str(exc), parent=self)

    def _on_target_sections_changed(self, _event: tk.Event[Any]) -> None:
        val = self.target_sections_var.get()
        if "3" in val:
            target = 3
        elif "5" in val:
            target = 5
        else:
            target = 0
        if self.project:
            self.project.target_sections = target
            self._mark_dirty()
        self._update_protocol_display()
        self._refresh_all()

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
            self.active_fiber_id = "F001"
            self.pending_point = None
            self.roi_bbox = None
            self.auto_candidates = []
            self.auto_summary = None
            self.histogram_filter = None
            self._has_fit = False

            # Auto preset selection
            auto_preset = get_preset_for_calibration(document.calibration)
            self.roi_preset_var.set(auto_preset.name)
            self._on_preset_changed(None)

            self._clear_dirty()
            self.calibration_var.set(
                f"{document.calibration.pixel_size_x_m * 1e9:.4f} nm/px\n"
                f"Fuente: {document.calibration.source}\n"
                f"Footer: {document.footer_bounds or 'no detectado'}"
            )
            self._refresh_all()
            self.after_idle(self.fit_image)
            self.status_var.set(f"Imagen cargada. Perfil recomendado: {auto_preset.name}")
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
            self.active_fiber_id = project.active_fiber_id or "F001"
            self.roi_bbox = None
            self.auto_candidates = []
            self.auto_summary = None
            self.histogram_filter = None
            self._has_fit = False

            if project.target_sections == 3:
                self.target_sections_var.set("3 secciones")
            elif project.target_sections == 5:
                self.target_sections_var.set("5 secciones")
            else:
                self.target_sections_var.set("Libre (0)")

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
                self.project.active_fiber_id = self.active_fiber_id
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
            export_annotated(
                self.project,
                self.source_image,
                path,
                show_ids=True,
                show_values=True,
                show_extrema=True,
                show_defects=True,
                show_legend=True,
                show_candidates=self.show_auto_candidates_var.get(),
                candidates=self.auto_candidates,
            )
            self.status_var.set(f"Imagen anotada exportada: {path}")

    def export_report_dialog(self) -> None:
        if self.project is None or self.source_image is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".html", initialfile="fiber_report.html")
        if not path:
            return
        report_path = Path(path)
        annotated_path = report_path.with_name(report_path.stem + "_annotated.png")
        export_annotated(
            self.project,
            self.source_image,
            annotated_path,
            show_ids=True,
            show_values=True,
            show_extrema=True,
            show_defects=True,
            show_legend=True,
        )
        export_html_report(self.project, annotated_path.name, report_path)
        self.status_var.set(f"Informe exportado: {report_path}")

    # ---------- transforms, zoom & rendering ----------

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
        if self.selected_measurement_id:
            m = self._selected_measurement()
            if m:
                target_center = m.center
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

        self._draw_roi_box()
        self._draw_auto_candidates()
        self._draw_measurements()

    def _draw_roi_box(self) -> None:
        if self.roi_bbox is None or self.project is None:
            return
        x0, y0, x1, y1 = self.roi_bbox
        p0c = self.image_to_canvas((x0, y0))
        p1c = self.image_to_canvas((x1, y1))

        self.canvas.create_rectangle(*p0c, *p1c, outline="#00D7FF", width=2, dash=(6, 4))

        w_px = x1 - x0
        h_px = y1 - y0
        w_m = w_px * self.project.image.calibration.pixel_size_x_m
        h_m = h_px * self.project.image.calibration.pixel_size_y_m
        roi_txt = f"ROI: {w_px}x{h_px} px | {format_length_m(w_m)} x {format_length_m(h_m)}"

        self.canvas.create_rectangle(p0c[0], p0c[1] - 20, p0c[0] + 280, p0c[1], fill="#003542", outline="#00D7FF")
        self.canvas.create_text(p0c[0] + 6, p0c[1] - 10, text=roi_txt, fill="#00D7FF", font=("TkDefaultFont", 8, "bold"), anchor=tk.W)

    def _draw_auto_candidates(self) -> None:
        if not self.show_auto_candidates_var.get() or not self.auto_candidates:
            return

        for cand in self.auto_candidates:
            is_selected = cand.candidate_id == self.selected_candidate_id
            if cand.status == "REJECTED":
                color = "#555555"
                width = 1
                dash = (2, 4)
            elif cand.status == "ACCEPTED":
                color = "#00FF7F"
                width = 2
                dash = None
            elif is_selected:
                color = self.SELECTED
                width = 4
                dash = None
            else:
                color = self.CANDIDATE_COLOR
                width = 2
                dash = (4, 3)

            for pm in cand.proposed_measurements:
                p1c = self.image_to_canvas(pm.p1)
                p2c = self.image_to_canvas(pm.p2)
                self.canvas.create_line(*p1c, *p2c, fill=color, width=width, dash=dash)
                for pt in (p1c, p2c):
                    self.canvas.create_oval(pt[0] - 3, pt[1] - 3, pt[0] + 3, pt[1] + 3, fill=color, outline="white")

            if cand.proposed_measurements:
                c_center = self.image_to_canvas(cand.proposed_measurements[0].center)
                label_txt = f"{cand.candidate_id} ({cand.confidence_level})"
                self.canvas.create_text(
                    c_center[0] + 6,
                    c_center[1] - 6,
                    text=label_txt,
                    fill=color,
                    font=("TkDefaultFont", 8, "bold" if is_selected else "normal"),
                    anchor=tk.SW,
                )

    def _draw_measurements(self) -> None:
        if self.project is None:
            return

        extrema_by_m = get_fiber_extrema(self.project.measurements, self.active_fiber_id)
        show_all = self.show_all_sections_var.get()

        filter_items = set(self.histogram_filter[2]) if self.histogram_filter else None
        filter_mode = self.histogram_filter[0] if self.histogram_filter else None

        for measurement in self.project.measurements:
            is_active_fiber = measurement.fiber_id == self.active_fiber_id
            is_selected = measurement.measurement_id == self.selected_measurement_id

            if not show_all and not is_active_fiber:
                continue

            if filter_items is not None:
                if filter_mode == "fiber" and measurement.fiber_id not in filter_items:
                    continue
                if filter_mode == "section" and measurement.measurement_id not in filter_items:
                    continue

            p1c = self.image_to_canvas(measurement.p1)
            p2c = self.image_to_canvas(measurement.p2)

            if is_selected:
                color = self.SELECTED
                width = 5
                dash = None
            elif not measurement.accepted:
                color = "#666666"
                width = 2
                dash = (4, 4)
            elif is_active_fiber:
                color = self.GROUP_HEX[(measurement.group or 0) % len(self.GROUP_HEX)]
                width = 3
                dash = None
            else:
                color = "#005a8d" if show_all else "#333333"
                width = 2
                dash = None

            self.canvas.create_line(*p1c, *p2c, fill=color, width=width, dash=dash)
            radius = 6 if is_selected else 3
            for point in (p1c, p2c):
                self.canvas.create_oval(
                    point[0] - radius,
                    point[1] - radius,
                    point[0] + radius,
                    point[1] + radius,
                    fill=color,
                    outline="white",
                )

            center = self.image_to_canvas(measurement.center)

            if is_selected or (is_active_fiber and self.scale > 0.4):
                method_icon = " [A]" if "ASSISTED" in measurement.method or "AUTO" in measurement.method else ""
                label_txt = f"{measurement.fiber_id}: {measurement.width_m * 1e6:.3f} µm{method_icon}"
                self.canvas.create_text(
                    center[0] + 8,
                    center[1] + 8,
                    text=label_txt,
                    anchor=tk.NW,
                    fill="#fff6a8" if is_selected else "#ffffff",
                    font=("TkDefaultFont", 9, "bold" if is_selected else "normal"),
                )

            if is_active_fiber and measurement.measurement_id in extrema_by_m:
                labels = "/".join(extrema_by_m[measurement.measurement_id])
                self.canvas.create_rectangle(
                    center[0] - 18,
                    center[1] - 18,
                    center[0] + 18,
                    center[1] - 4,
                    fill="#0072B2",
                    outline="#ffffff",
                )
                self.canvas.create_text(
                    center[0],
                    center[1] - 11,
                    text=labels,
                    fill="#FFFFFF",
                    font=("TkDefaultFont", 8, "bold"),
                )

    # ---------- canvas interactions & hit test ----------

    def _wheel_zoom(self, event: tk.Event[Any]) -> None:
        factor = 1.2 if event.delta > 0 else 1 / 1.2
        self._zoom_at(event.x, event.y, factor)

    def _zoom_at(self, x: float, y: float, factor: float) -> None:
        if self.source_image is None:
            return
        image_point = self.canvas_to_image((x, y))
        self.scale = max(0.03, min(15.0, self.scale * factor))
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
                self.select_fiber(measurement.fiber_id)
                if endpoint is not None:
                    self.drag_endpoint = endpoint
                    self.dragging_measurement_id = measurement.measurement_id
            else:
                self.select_measurement(None)
            return

        if tool == "auto_roi":
            self.pending_roi_start = point
            self.status_var.set("Arrastra para definir el rectángulo de la ROI...")
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
        if self.tool_var.get() == "auto_roi" and self.pending_roi_start is not None:
            p_curr = self.canvas_to_image((event.x, event.y))
            x0 = int(min(self.pending_roi_start[0], p_curr[0]))
            y0 = int(min(self.pending_roi_start[1], p_curr[1]))
            x1 = int(max(self.pending_roi_start[0], p_curr[0]))
            y1 = int(max(self.pending_roi_start[1], p_curr[1]))
            self.roi_bbox = (x0, y0, x1, y1)
            self._update_roi_pre_analysis_feedback()
            self.render()
            return

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
        self._refresh_inspector()
        self.render()

    def _on_left_release(self, _event: tk.Event[Any]) -> None:
        if self.tool_var.get() == "auto_roi" and self.pending_roi_start is not None:
            self.pending_roi_start = None
            if self.roi_bbox:
                w = self.roi_bbox[2] - self.roi_bbox[0]
                h = self.roi_bbox[3] - self.roi_bbox[1]
                self.status_var.set(f"ROI definida ({w}x{h} px). Haz clic en 'Analizar ROI'.")
            return

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

    def _update_roi_pre_analysis_feedback(self) -> None:
        if self.roi_bbox is None or self.project is None or self.gray is None:
            return
        x0, y0, x1, y1 = self.roi_bbox
        w_px, h_px = x1 - x0, y1 - y0
        patch = self.gray[y0:y1, x0:x1]
        if patch.size == 0:
            return

        res_status, _res_msg = check_resolution_resolvability(patch, self.project.image.calibration)
        std_val = float(patch.std())

        preset_name = self.roi_preset_var.get()
        self.roi_feedback_var.set(
            f"ROI: {w_px}x{h_px} px | Contraste std: {std_val:.1f}\n"
            f"Resolución: {res_status}\n"
            f"Perfil: {preset_name}"
        )

    def run_auto_roi(self) -> None:
        if self.project is None or self.gray is None:
            messagebox.showwarning("Sin imagen", "Abre una micrografía antes de analizar una ROI.", parent=self)
            return

        if self.roi_bbox is None:
            messagebox.showwarning("Sin ROI definida", "Usa la herramienta 'ROI automática (I)' para definir un rectángulo sobre la imagen.", parent=self)
            return

        pol_map = {"Automática": "auto", "Clara": "bright", "Oscura": "dark"}
        polarity = pol_map.get(self.roi_polarity_var.get(), "auto")
        thresh_method = self.roi_thresh_var.get()
        n_sec = self.roi_n_sections_var.get()

        self.status_var.set("Analizando ROI...")
        self.update_idletasks()

        try:
            candidates, summary = analyze_roi(
                gray=self.gray,
                roi_bbox=self.roi_bbox,
                calibration=self.project.image.calibration,
                footer_bounds=self.project.image.footer_bounds,
                polarity=polarity,
                threshold_method=thresh_method,
                min_area_px=self.roi_min_area_var.get(),
                min_elongation=self.roi_min_elong_var.get(),
                min_width_px=self.roi_min_width_var.get(),
                n_sections=n_sec,
                allow_curved_trace=self.roi_curved_var.get(),
            )

            self.auto_candidates = candidates
            self.auto_summary = summary
            self.selected_candidate_id = candidates[0].candidate_id if candidates else None

            recommendation = ""
            if summary.resolution_status == "RESOLUTION_INSUFFICIENT":
                recommendation = " | Recomendación: Resolución insuficiente para diámetros automáticos; usar magnificación mayor o caliper manual."
            elif summary.total_components > 45:
                recommendation = " | Recomendación: ROI muy densa; pruebe una ROI más pequeña para evitar fusiones."

            self.roi_summary_var.set(
                f"Threshold: {summary.threshold_method_used} | Res: {summary.resolution_status}\n"
                f"Comp: {summary.total_components} | Medibles: {summary.measurable_candidates} | "
                f"Alta conf: {summary.high_confidence} | Revisión: {summary.needs_review} | Excluidos: {summary.excluded}"
                f"{recommendation}"
            )

            self._refresh_auto_tree()
            self.notebook.select(2)
            self.render()

            self.status_var.set(
                f"Análisis ROI completado ({summary.threshold_method_used}): {summary.measurable_candidates} candidatos en ROI ({summary.high_confidence} Alta confianza)."
            )
        except Exception as exc:
            messagebox.showerror("Error al analizar ROI", str(exc), parent=self)

    def clear_roi(self) -> None:
        self.roi_bbox = None
        self.pending_roi_start = None
        self.auto_candidates = []
        self.auto_summary = None
        self.selected_candidate_id = None
        self.roi_summary_var.set("ROI: Sin analizar")
        self.roi_feedback_var.set("Dibuja una ROI sobre la imagen.")
        self._refresh_auto_tree()
        self.render()
        self.status_var.set("ROI y candidatos automáticos limpiados.")

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
            fiber_id=self.active_fiber_id,
            p1=p1,
            p2=p2,
            width_m=width_m,
            method=method,
            confidence=confidence,
        )
        self.project.measurements.append(measurement)
        self.select_measurement(measurement.measurement_id)
        self._mark_dirty()

        if self.project.is_fiber_complete(self.active_fiber_id) and self.auto_advance_var.get():
            self.status_var.set(f"¡Protocolo de {self.active_fiber_id} completado! Avanzando automáticamente.")
            self.next_fiber()
        else:
            self._refresh_all()
            self.status_var.set(f"{measurement.measurement_id}: {format_length_m(width_m)}")

    def _hit_test(self, canvas_x: float, canvas_y: float) -> tuple[Measurement | None, int | None]:
        if self.project is None:
            return None, None
        endpoint_threshold = 12.0
        line_threshold = 9.0
        best: tuple[float, Measurement, int | None] | None = None

        for measurement in self.project.measurements:
            p1 = np.asarray(self.image_to_canvas(measurement.p1))
            p2 = np.asarray(self.image_to_canvas(measurement.p2))
            point = np.asarray((canvas_x, canvas_y))
            d1 = float(np.linalg.norm(point - p1))
            d2 = float(np.linalg.norm(point - p2))

            priority_penalty = 0.0 if measurement.fiber_id == self.active_fiber_id else 5.0

            for distance, endpoint in ((d1, 0), (d2, 1)):
                adj_dist = distance + priority_penalty
                if distance <= endpoint_threshold and (best is None or adj_dist < best[0]):
                    best = (adj_dist, measurement, endpoint)

            segment = p2 - p1
            denom = float(np.dot(segment, segment))
            if denom > 0:
                t = max(0.0, min(1.0, float(np.dot(point - p1, segment) / denom)))
                projection = p1 + t * segment
                line_distance = float(np.linalg.norm(point - projection))
                adj_line_dist = line_distance + priority_penalty
                if line_distance <= line_threshold and (best is None or adj_line_dist < best[0]):
                    best = (adj_line_dist, measurement, None)

        return (best[1], best[2]) if best else (None, None)

    # ---------- Protocol & Linked Selection ----------

    def select_fiber(self, fiber_id: str) -> None:
        self.active_fiber_id = fiber_id
        if self.project:
            self.project.active_fiber_id = fiber_id
        self.lbl_active_fiber.configure(text=fiber_id)
        self._update_protocol_display()

        if self.fiber_tree.exists(fiber_id):
            self.fiber_tree.selection_set(fiber_id)
            self.fiber_tree.see(fiber_id)

        self._refresh_inspector()
        self.render()

    def select_measurement(self, measurement_id: str | None) -> None:
        self.selected_measurement_id = measurement_id
        if measurement_id and self.tree.exists(measurement_id):
            self.tree.selection_set(measurement_id)
            self.tree.see(measurement_id)
        measurement = self._selected_measurement()
        if measurement:
            self.select_fiber(measurement.fiber_id)
            self.defect_var.set(measurement.defect)
        self._refresh_inspector()
        self.render()

    def _tree_selection(self, _event: tk.Event[Any]) -> None:
        selected = self.tree.selection()
        if selected:
            self.select_measurement(selected[0])

    def _on_tree_double_click(self, _event: tk.Event[Any]) -> None:
        self.tool_var.set("select")
        self.center_selection()

    def _on_fiber_tree_select(self, _event: tk.Event[Any]) -> None:
        selected = self.fiber_tree.selection()
        if selected:
            self.select_fiber(selected[0])

    def _on_auto_tree_select(self, _event: tk.Event[Any]) -> None:
        selected = self.auto_tree.selection()
        if selected:
            self.selected_candidate_id = selected[0]
            self._refresh_inspector()
            self.render()

    def _sort_auto_tree(self, col: str) -> None:
        items = [(self.auto_tree.set(k, col), k) for k in self.auto_tree.get_children("")]
        items.sort()
        for index, (_val, k) in enumerate(items):
            self.auto_tree.move(k, "", index)

    def goto_selected_candidate(self) -> None:
        cand = self._selected_candidate()
        if cand and cand.proposed_measurements:
            cw = max(self.canvas.winfo_width(), 10)
            ch = max(self.canvas.winfo_height(), 10)
            c_center = cand.proposed_measurements[0].center
            self.offset_x = cw / 2 - c_center[0] * self.scale
            self.offset_y = ch / 2 - c_center[1] * self.scale
            self.render()

    def roi_around_candidate(self) -> None:
        cand = self._selected_candidate()
        if cand:
            x0, y0, x1, y1 = cand.roi_bbox
            pad = 40
            self.roi_bbox = (max(0, x0 - pad), max(0, y0 - pad), x1 + pad, y1 + pad)
            self.render()

    def manual_measure_here(self) -> None:
        self.goto_selected_candidate()
        self.tool_var.set("manual")
        self.status_var.set("Herramienta manual activada sobre el candidato.")

    def new_fiber(self) -> None:
        if self.project is None:
            new_id = "F001"
        else:
            new_id = self.project.get_next_fiber_id()
            self._mark_dirty()
        self.select_fiber(new_id)
        self.status_var.set(f"Nueva fibra activa: {new_id}")

    def prev_fiber(self) -> None:
        if self.project is None:
            return
        fibers = sorted({m.fiber_id for m in self.project.measurements} | {self.active_fiber_id})
        if self.active_fiber_id in fibers:
            idx = fibers.index(self.active_fiber_id)
            prev_id = fibers[max(0, idx - 1)]
            self.select_fiber(prev_id)

    def next_fiber(self) -> None:
        if self.project is None:
            return
        fibers = sorted({m.fiber_id for m in self.project.measurements} | {self.active_fiber_id})
        if self.active_fiber_id in fibers:
            idx = fibers.index(self.active_fiber_id)
            if idx + 1 < len(fibers):
                next_id = fibers[idx + 1]
            else:
                next_id = self.project.get_next_fiber_id()
                self._mark_dirty()
            self.select_fiber(next_id)

    def _update_protocol_display(self) -> None:
        if self.project is None:
            self.protocol_status_var.set("Protocolo: 0 / 5 (Incompleto)")
            return
        target = self.project.target_sections
        accepted = len(self.project.accepted_fiber_measurements(self.active_fiber_id))
        target_str = "∞" if target <= 0 else str(target)

        if self.project.is_fiber_complete(self.active_fiber_id):
            self.protocol_status_var.set(f"Protocolo: {accepted} / {target_str} (✓ Completo)")
        else:
            self.protocol_status_var.set(f"Protocolo: {accepted} / {target_str} (Incompleto)")

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

    def _on_enter_pressed(self) -> None:
        measurement = self._selected_measurement()
        if measurement and not measurement.accepted:
            measurement.accepted = True
            self._mark_dirty()
            self._refresh_all()

    def classify(self) -> None:
        if self.project is None or not self.project.measurements:
            return
        mode = self.classification_var.get()
        if mode == "Auto":
            mapping = classify_fibers(self.project.measurements)
        else:
            k = int(mode)
            mapping = classify_fibers(self.project.measurements, requested_k=k)

        groups = len(set(mapping.values())) if mapping else 0
        self._mark_dirty()
        self._refresh_all()
        self.status_var.set(f"Clasificación aplicada: {groups} grupo(s) compatibles con la distribución.")

    # ---------- Candidate Review Actions ----------

    def _selected_candidate(self) -> AutoFiberCandidate | None:
        if not self.selected_candidate_id:
            return None
        return next((c for c in self.auto_candidates if c.candidate_id == self.selected_candidate_id), None)

    def accept_candidate(self) -> None:
        if self.project is None or self.selected_candidate_id is None:
            return
        cand = self._selected_candidate()
        if cand is None or cand.status == "ACCEPTED":
            return

        fiber_id = self.project.get_next_fiber_id()
        for pm in cand.proposed_measurements:
            m = Measurement(
                measurement_id=self.project.next_measurement_id(),
                fiber_id=fiber_id,
                p1=pm.p1,
                p2=pm.p2,
                width_m=pm.width_m,
                method="AUTO_ROI_COMPONENT",
                confidence=cand.confidence_score,
            )
            self.project.measurements.append(m)

        cand.status = "ACCEPTED"
        self.select_fiber(fiber_id)
        self._mark_dirty()
        self._refresh_all()
        self._refresh_auto_tree()
        self.status_var.set(f"Candidato {cand.candidate_id} aceptado como nueva fibra {fiber_id} ({len(cand.proposed_measurements)} secciones).")

    def reject_candidate(self) -> None:
        if self.selected_candidate_id is None:
            return
        cand = self._selected_candidate()
        if cand:
            cand.status = "REJECTED"
            self._refresh_auto_tree()
            self.render()
            self.status_var.set(f"Candidato {cand.candidate_id} rechazado.")

    def accept_high_confidence_candidates(self) -> None:
        if self.project is None or not self.auto_candidates:
            return
        high_conf = [c for c in self.auto_candidates if c.status == "PENDING" and c.confidence_level == "Alta"]
        if not high_conf:
            messagebox.showinfo("Sin candidatos", "No hay candidatos pendientes de Alta Confianza.", parent=self)
            return

        n_sec = sum(len(c.proposed_measurements) for c in high_conf)
        confirm = messagebox.askyesno(
            "Confirmación de aceptación masiva",
            f"¿Deseas aceptar {len(high_conf)} candidatos de Alta Confianza con un total de {n_sec} secciones?\n\n"
            "Solo se incorporarán candidatos sin flags críticos ni contacto con bordes.",
            parent=self,
        )
        if not confirm:
            return

        for cand in high_conf:
            fiber_id = self.project.get_next_fiber_id()
            for pm in cand.proposed_measurements:
                m = Measurement(
                    measurement_id=self.project.next_measurement_id(),
                    fiber_id=fiber_id,
                    p1=pm.p1,
                    p2=pm.p2,
                    width_m=pm.width_m,
                    method="AUTO_ROI_COMPONENT",
                    confidence=cand.confidence_score,
                )
                self.project.measurements.append(m)
            cand.status = "ACCEPTED"

        self._mark_dirty()
        self._refresh_all()
        self._refresh_auto_tree()
        self.status_var.set(f"Aceptados masivamente {len(high_conf)} candidatos de alta confianza.")

    def reject_remaining_candidates(self) -> None:
        if not self.auto_candidates:
            return
        pending = [c for c in self.auto_candidates if c.status == "PENDING"]
        for c in pending:
            c.status = "REJECTED"
        self._refresh_auto_tree()
        self.render()
        self.status_var.set(f"Rechazados {len(pending)} candidatos pendientes.")

    # ---------- Interactive Histogram ----------

    def _refresh_histogram(self) -> None:
        if self.project is None:
            self.hist_canvas.delete("all")
            self.hist_info_lbl.configure(text="N = 0")
            return

        mode = self.hist_mode_var.get()
        n_bins = self.hist_bins_var.get()
        hist_data = compute_histogram_data(self.project.measurements, mode=mode, n_bins=n_bins)

        self.hist_canvas.delete("all")
        counts = hist_data["counts"]
        edges = hist_data["bin_edges"]
        vals = hist_data["values"]

        if vals.size == 0:
            self.hist_info_lbl.configure(text="N = 0 (Sin datos suficientes)")
            return

        self.hist_info_lbl.configure(
            text=f"N = {vals.size} | Media: {format_length_m(hist_data['mean_m'])} | Mediana: {format_length_m(hist_data['median_m'])}"
        )

        w = max(self.hist_canvas.winfo_width(), 300)
        h = max(self.hist_canvas.winfo_height(), 200)

        padding_x = 40
        padding_y = 30
        plot_w = w - 2 * padding_x
        plot_h = h - 2 * padding_y

        max_count = max(counts) if counts.size > 0 else 1
        n = len(counts)
        bar_w = plot_w / max(1, n)

        self._hist_bin_boxes = []

        for i in range(n):
            c = counts[i]
            bar_h = (c / max_count) * plot_h
            x0 = padding_x + i * bar_w
            y0 = h - padding_y - bar_h
            x1 = x0 + bar_w - 2
            y1 = h - padding_y

            is_filtered_bin = (
                self.histogram_filter is not None
                and self.histogram_filter[0] == mode
                and self.histogram_filter[1] == i
            )
            color = "#FFBF00" if is_filtered_bin else "#0072B2"

            self.hist_canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="#ffffff")
            if c > 0:
                self.hist_canvas.create_text((x0 + x1) / 2, y0 - 8, text=str(c), fill="#ffffff", font=("TkDefaultFont", 8))

            self._hist_bin_boxes.append((x0, y0, x1, y1, i, edges[i], edges[i + 1]))

        mean_val = hist_data["mean_m"]
        med_val = hist_data["median_m"]
        min_e, max_e = edges[0], edges[-1]
        span = max_e - min_e if max_e > min_e else 1.0

        if mean_val is not None:
            x_mean = padding_x + ((mean_val - min_e) / span) * plot_w
            self.hist_canvas.create_line(x_mean, padding_y, x_mean, h - padding_y, fill="#E69F00", width=2, dash=(4, 2))
            self.hist_canvas.create_text(x_mean, padding_y - 10, text="Media", fill="#E69F00", font=("TkDefaultFont", 8, "bold"))

        if med_val is not None:
            x_med = padding_x + ((med_val - min_e) / span) * plot_w
            self.hist_canvas.create_line(x_med, padding_y, x_med, h - padding_y, fill="#009E73", width=2)
            self.hist_canvas.create_text(x_med, padding_y - 20, text="Mediana", fill="#009E73", font=("TkDefaultFont", 8, "bold"))

    def _on_histogram_click(self, event: tk.Event[Any]) -> None:
        if not hasattr(self, "_hist_bin_boxes") or not self.project:
            return
        cx = event.x
        mode = self.hist_mode_var.get()
        n_bins = self.hist_bins_var.get()
        hist_data = compute_histogram_data(self.project.measurements, mode=mode, n_bins=n_bins)

        for x0, _y0, x1, _y1, bin_idx, e0, e1 in self._hist_bin_boxes:
            if x0 <= cx <= x1:
                matching_ids = []
                for item_id, val in hist_data["items"]:
                    if e0 <= val <= e1:
                        matching_ids.append(item_id)
                self.histogram_filter = (mode, bin_idx, matching_ids)
                self._refresh_histogram()
                self.render()
                self.status_var.set(
                    f"Filtro aplicado en histograma ({mode}): {len(matching_ids)} elementos en rango {format_length_m(e0)} - {format_length_m(e1)}"
                )
                return

    def clear_histogram_filter(self) -> None:
        self.histogram_filter = None
        self._refresh_histogram()
        self.render()
        self.status_var.set("Filtro de histograma limpiado.")

    # ---------- refreshes ----------

    def _refresh_all(self) -> None:
        self._update_protocol_display()
        self._refresh_fiber_tree()
        self._refresh_tree()
        self._refresh_auto_tree()
        self._refresh_inspector()
        self._refresh_histogram()
        self._refresh_metadata()
        self.render()

    def _refresh_auto_tree(self) -> None:
        for item in self.auto_tree.get_children():
            self.auto_tree.delete(item)
        for cand in self.auto_candidates:
            med_str = format_length_m(cand.median_width_m) if cand.median_width_m else "—"
            cv_str = f"{cand.cv_width * 100:.1f}%" if cand.cv_width is not None else "—"
            disc_str = f"{cand.mean_discrepancy * 100:.1f}%" if cand.mean_discrepancy is not None else "—"
            flags_str = ", ".join(sorted(cand.quality_flags)) if cand.quality_flags else "ninguno"
            conf_str = f"{cand.confidence_score:.2f} ({cand.confidence_level})"
            self.auto_tree.insert(
                "",
                tk.END,
                iid=cand.candidate_id,
                values=(
                    cand.candidate_id,
                    cand.status,
                    len(cand.proposed_measurements),
                    med_str,
                    cv_str,
                    disc_str,
                    conf_str,
                    flags_str,
                ),
            )
        if self.selected_candidate_id and self.auto_tree.exists(self.selected_candidate_id):
            self.auto_tree.selection_set(self.selected_candidate_id)

    def _refresh_fiber_tree(self) -> None:
        for item in self.fiber_tree.get_children():
            self.fiber_tree.delete(item)
        if self.project is None:
            return

        fibers_stats = fiber_statistics(self.project.measurements)
        all_fiber_ids = sorted({m.fiber_id for m in self.project.measurements} | {self.active_fiber_id})

        for fid in all_fiber_ids:
            tot = len(self.project.fiber_measurements(fid))
            accepted = len(self.project.accepted_fiber_measurements(fid))
            info = fibers_stats.get(fid)

            if info and accepted > 0:
                med_str = f"{info['median_m'] * 1e6:.3f} µm"
                min_str = f"{info['min_m'] * 1e6:.3f} µm"
                max_str = f"{info['max_m'] * 1e6:.3f} µm"
            else:
                med_str = "—"
                min_str = "—"
                max_str = "—"

            status = "✓ Completo" if self.project.is_fiber_complete(fid) else "Incompleto"
            if accepted == 0:
                status = "Sin datos"

            groups = {m.group for m in self.project.fiber_measurements(fid) if m.group is not None}
            grp_str = str(next(iter(groups)) + 1) if len(groups) == 1 else "—"

            self.fiber_tree.insert(
                "",
                tk.END,
                iid=fid,
                values=(
                    fid,
                    f"{accepted}/{tot}",
                    med_str,
                    min_str,
                    max_str,
                    status,
                    grp_str,
                ),
            )

        if self.active_fiber_id and self.fiber_tree.exists(self.active_fiber_id):
            self.fiber_tree.selection_set(self.active_fiber_id)

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
                    measurement.method.replace("ASSISTED_", "A:").replace("MANUAL_", "M:").replace("AUTO_ROI_COMPONENT", "ROI:"),
                    group,
                    measurement.defect,
                    "✓" if measurement.accepted else "✗",
                ),
            )
        if self.selected_measurement_id and self.tree.exists(self.selected_measurement_id):
            self.tree.selection_set(self.selected_measurement_id)

    def _refresh_inspector(self) -> None:
        lines = []
        fid = self.active_fiber_id
        lines.append(f"INSPECTOR DE FIBRA — {fid}")
        lines.append("=" * 35)

        if self.project:
            tot = len(self.project.fiber_measurements(fid))
            accepted = len(self.project.accepted_fiber_measurements(fid))
            rejected = tot - accepted
            is_comp = self.project.is_fiber_complete(fid)

            lines.append(f"Estado del Protocolo: {'✓ Protocolo completo' if is_comp else 'Protocolo incompleto'}")
            lines.append(f"Secciones totales: {tot} (Aceptadas: {accepted}, Rechazadas: {rejected})")

            fib_stats = fiber_statistics(self.project.measurements).get(fid)
            if fib_stats and accepted > 0:
                lines.append(f"Media: {format_length_m(float(fib_stats['mean_m']))}")
                lines.append(f"Mediana: {format_length_m(float(fib_stats['median_m']))}")
                lines.append(f"Desviación estándar: {format_length_m(float(fib_stats['std_m']))}")
                lines.append(f"Mínimo crudo: {format_length_m(float(fib_stats['min_m']))}")
                lines.append(f"Máximo crudo: {format_length_m(float(fib_stats['max_m']))}")
            else:
                lines.append("Media: —\nMediana: —\nDesviación estándar: —\nMínimo crudo: —\nMáximo crudo: —")

        lines.append("")
        lines.append("CANDIDATO SELECCIONADO")
        lines.append("=" * 35)
        cand = self._selected_candidate()
        if cand:
            lines.append(f"ID Candidato: {cand.candidate_id}")
            lines.append(f"Estado: {cand.status}")
            lines.append(f"Umbral: {cand.threshold_method}")
            lines.append(f"Confianza: {cand.confidence_score:.2f} ({cand.confidence_level})")
            lines.append(f"Mediana propuesta: {format_length_m(cand.median_width_m) if cand.median_width_m else '—'}")

            if cand.proposed_measurements:
                pm0 = cand.proposed_measurements[0]
                m_str = format_length_m(pm0.mask_width_m) if pm0.mask_width_m else "—"
                p_str = format_length_m(pm0.profile_width_m) if pm0.profile_width_m else "—"
                d_str = f"{pm0.discrepancy * 100:.1f}%" if pm0.discrepancy is not None else "—"
                lines.append(f"Ancho máscara: {m_str}")
                lines.append(f"Ancho perfil: {p_str}")
                lines.append(f"Discrepancia: {d_str}")

            lines.append(f"Flags: {', '.join(sorted(cand.quality_flags)) if cand.quality_flags else 'ninguno'}")
        else:
            lines.append("Ningún candidato seleccionado.")

        lines.append("")
        lines.append("MEDICIÓN SELECCIONADA")
        lines.append("=" * 35)
        m = self._selected_measurement()
        if m:
            confidence = "—" if m.confidence is None else f"{m.confidence:.2f}"
            lines.append(f"ID: {m.measurement_id}")
            lines.append(f"Ancho proyectado: {format_length_m(m.width_m)}")
            lines.append(f"Método: {m.method}")
            lines.append(f"Confianza: {confidence}")
            lines.append(f"Grupo: {m.group + 1 if m.group is not None else '—'}")
            lines.append(f"Defecto: {m.defect}")
            lines.append(f"Aceptada: {'Sí' if m.accepted else 'No'}")
        else:
            lines.append("Ninguna medición seleccionada.")

        self.inspector_text.configure(state=tk.NORMAL)
        self.inspector_text.delete("1.0", tk.END)
        self.inspector_text.insert("1.0", "\n".join(lines))
        self.inspector_text.configure(state=tk.DISABLED)

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
        self.pending_roi_start = None
        self.drag_endpoint = None
        self.dragging_measurement_id = None
        self.render()

    def _selected_measurement(self) -> Measurement | None:
        return self._measurement_by_id(self.selected_measurement_id)

    def _measurement_by_id(self, measurement_id: str | None) -> Measurement | None:
        if self.project is None or measurement_id is None:
            return None
        return next((m for m in self.project.measurements if m.measurement_id == measurement_id), None)

    # ---------- help ----------

    def show_protocol(self) -> None:
        messagebox.showinfo(
            "Protocolo de Medición de Fibras SEM",
            "1. Selecciona o crea una ID de fibra (ej. F001).\n"
            "2. Realiza 3 o 5 secciones perpendiculares por fibra o usa 'ROI automática (I)'.\n"
            "3. Revisa el indicador de progreso del protocolo (ej. 5/5 Completo).\n"
            "4. Utiliza la pestaña 'Revisión ROI' para validar y aceptar candidatos automáticos.\n"
            "5. Usa los atajos de teclado para un flujo ultra-rápido:\n"
            "   • I: Herramienta ROI automática\n"
            "   • N: Nueva fibra\n"
            "   • PageUp / PageDown: Navegar entre fibras\n"
            "   • M: Medición manual | S: Snap | A: Propuesta auto\n"
            "   • R: Aceptar/Rechazar | Del: Eliminar\n"
            "   • F: Ajustar imagen | C: Centrar selección",
            parent=self,
        )

    def show_about(self) -> None:
        messagebox.showinfo(
            "Acerca de Fathom Fibers Quick",
            "Fathom Fibers Quick 0.1 — Endurecido\n\n"
            "Herramienta interactiva de medición manual, asistida y automática en ROI para micrografías SEM.\n"
            "Optimizado para metadatos Zeiss CZ_SEM con verificación de integridad SHA-256.\n\n"
            "Los candidatos automáticos son propuestas heurísticas en ROI que requieren validación humana.",
            parent=self,
        )


def main() -> None:
    initial_path = sys.argv[1] if len(sys.argv) > 1 else None
    app = FiberQuickApp(initial_path)
    app.mainloop()


if __name__ == "__main__":
    main()
