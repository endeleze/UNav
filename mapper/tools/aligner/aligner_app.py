"""
UNav Floorplan-SLAM Aligner GUI

This module implements the main GUI for registering 3D SLAM keyframes with a 2D floorplan.
Handles all user interactions, layout, visualization, and delegates alignment logic to AlignerLogic.
"""

import re
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from PIL import Image

from mapper.tools.aligner.canvas_keyframe_feature import KeyframeViewer
from mapper.tools.aligner.canvas_floorplan_point import FloorplanViewer
from mapper.tools.aligner.aligner_logic import AlignerLogic
from mapper.tools.aligner.computation import (
    project_point3d_to_floor2d, lists_equal, pose_dicts_equal
)
from mapper.tools.aligner.aligner_utils import natural_sort_key, compute_floorplan_display_size

class AlignerApp(ttk.Frame):
    """
    Main GUI application for 3D keyframe / 2D floorplan alignment.
    Handles widget layout, events, and visual overlay.
    Core logic and mapping is delegated to AlignerLogic.
    """
    LEFT_PANEL_WIDTH = 280
    INFO_PANEL_WIDTH = 320

    def __init__(self, master: tk.Tk, logic: AlignerLogic, floorplan_max_size: int = 4096) -> None:
        super().__init__(master)
        self.master = master
        self.logic = logic

        self._setup_window()
        self._initialize_state()
        self._build_gui()
        self._bind_events()

    def _setup_window(self) -> None:
        """Set the window title and geometry based on the floorplan image size."""
        self.master.title("UNav Aligner GUI")
        fp_w, fp_h = compute_floorplan_display_size(self.logic.floorplan_path)
        width = fp_w + self.LEFT_PANEL_WIDTH + self.INFO_PANEL_WIDTH
        height = max(600, fp_h + 80)
        self.master.geometry(f"{width}x{height}")

    def _initialize_state(self) -> None:
        """Initialize state variables and caches."""
        self.fp_image = Image.open(self.logic.floorplan_path)
        self.fp_w, self.fp_h = self.fp_image.size
        self.temporary_highlight_point3d = None
        self.persistent_highlight_point3d = None
        self._after_id = None
        self._last_overlay_data = {
            "points2D": None,
            "points3D": None,
            "matrix": None,
            "landmarks_3d": None,
            "cam_traj_3d": None,
            "curr_cam_pose": None,
            "curr_observed_3d": None
        }

    def _build_gui(self) -> None:
        """Create all panels and widgets, arranging with grid layout."""
        self.grid(row=0, column=0, sticky="nsew")
        self._create_left_panel()
        self._create_main_panel()
        self._create_info_panel()
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

    def _bind_events(self) -> None:
        """Bind window events and initialize view."""
        self.master.bind("<Configure>", self._on_root_resize)
        self.after(100, self._select_initial_keyframe)
        self.after(100, self.update_correspondence_table)

    # ---- Panel creation ----

    def _create_left_panel(self) -> None:
        """Left panel: keyframe selector and save button."""
        self.left = ttk.Frame(self, width=self.LEFT_PANEL_WIDTH)
        self.left.grid(row=0, column=0, sticky="ns")
        self.left.grid_propagate(False)
        self._populate_keyframe_listbox()
        self._create_save_matrix_button()

    def _populate_keyframe_listbox(self) -> None:
        panel = ttk.Frame(self.left)
        panel.grid(row=0, column=0, sticky="nsew")
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(1, weight=1)
        ttk.Label(panel, text="Keyframes:").grid(row=0, column=0, sticky="nw", pady=(0, 4))

        self.keyframe_listbox = tk.Listbox(panel, exportselection=False, width=18, height=12)
        self.keyframe_listbox.grid(row=1, column=0, sticky="nsew")
        sb = ttk.Scrollbar(panel, orient="vertical", command=self.keyframe_listbox.yview)
        sb.grid(row=1, column=1, sticky="ns")
        self.keyframe_listbox.config(yscrollcommand=sb.set)

        for name in sorted(self.logic.kf_data.keys(), key=lambda k: natural_sort_key(k.replace('.png', ''))):
            self.keyframe_listbox.insert(tk.END, name.replace('.png', ''))
        self.keyframe_listbox.bind('<<ListboxSelect>>', self._on_keyframe_select)

    def _create_save_matrix_button(self) -> None:
        btn_panel = ttk.Frame(self.left)
        btn_panel.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        btn_panel.columnconfigure(0, weight=1)
        ttk.Button(btn_panel, text="Save Matrix", command=self._on_save_matrix_clicked)\
            .grid(row=0, column=0, sticky="ew", pady=2)

    def _create_info_panel(self) -> None:
        """Right panel: correspondence table."""
        self.info = ttk.Frame(self, width=self.INFO_PANEL_WIDTH)
        self.info.grid(row=0, column=2, sticky="ns")
        self.info.grid_propagate(False)
        self.info.rowconfigure(0, weight=1)
        self.info.columnconfigure(0, weight=1)

        columns = ('idx', 'floor2d', 'proj3d', 'distance')
        self.info_table = ttk.Treeview(
            self.info, columns=columns, show='headings', selectmode='browse', height=22
        )
        for col, txt in zip(columns, ["Idx", "Floor2D (px)", "Proj3D (px)", "Error (meters)"]):
            self.info_table.heading(col, text=txt)
        self.info_table.column('idx', width=32, anchor='center')
        self.info_table.column('floor2d', width=90, anchor='w')
        self.info_table.column('proj3d', width=90, anchor='w')
        self.info_table.column('distance', width=60, anchor='center')
        self.info_table.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        vsb = ttk.Scrollbar(self.info, orient="vertical", command=self.info_table.yview)
        self.info_table.configure(yscrollcommand=vsb.set)
        vsb.grid(row=0, column=1, sticky="ns")
        self.info_table.bind('<ButtonRelease-1>', self._on_corr_select)
        self.info_table.bind('<Button-3>', self._on_corr_right_click)

    def _create_main_panel(self) -> None:
        """Central panel: image/floorplan viewers."""
        self.main_panel = ttk.Frame(self)
        self.main_panel.grid(row=0, column=1, sticky="nsew")
        self.main_panel.rowconfigure(0, weight=1)
        self.main_panel.columnconfigure(0, weight=1)

        vp = ttk.Frame(self.main_panel)
        vp.grid(row=0, column=0, sticky="nsew")
        vp.rowconfigure(0, weight=1)
        vp.rowconfigure(2, weight=2)
        vp.columnconfigure(0, weight=1)

        self.keyframe_viewer = KeyframeViewer(
            vp,
            keyframe_dir=self.logic.keyframe_dir,
            kf_data=self.logic.kf_data,
            correspondences=self.logic.correspondences
        )
        self.keyframe_viewer.grid(row=0, column=0, sticky="nsew")
        self.keyframe_viewer.bind(
            '<Double-Button-1>',
            lambda e: self.keyframe_viewer.open_feature_canvas(
                self.logic.correspondences,
                self._on_feature_selected,
                self._on_feature_highlight
            )
        )

        ttk.Separator(vp, orient="horizontal").grid(row=1, column=0, sticky="ew", pady=4)
        self.floorplan_viewer = FloorplanViewer(
            vp, floorplan_path=self.logic.floorplan_path
        )
        self.floorplan_viewer.grid(row=2, column=0, sticky="nsew")
        self.floorplan_viewer.render_floorplan()
        self.floorplan_viewer.bind('<Double-Button-1>', lambda e: self._open_floorplan_with_highlight())

    # ---- Window and resize handling ----

    def _on_root_resize(self, event: tk.Event) -> None:
        """Debounced handler for root window resize."""
        if event.widget is not self.master:
            return
        if self._after_id:
            self.after_cancel(self._after_id)
        self._after_id = self.after(100, lambda: self._do_resize(event.width, event.height))

    def _do_resize(self, win_w: int, win_h: int) -> None:
        """Re-layout all widgets based on new window size."""
        self._after_id = None
        w1, h1 = self.keyframe_viewer._orig_image.size
        w2, h2 = self.fp_w, self.fp_h
        r = (w1 * h2) / (h1 * w2)
        h1_scaled = win_h / (1 + r)
        h2_scaled = win_h - h1_scaled
        w_img = h1_scaled * (w1 / h1)

        w_remain = win_w - w_img
        w_left = w_remain / 4
        w_right = w_remain * 3 / 4

        self.left.config(width=int(w_left))
        self.info.config(width=int(w_right))
        self.keyframe_viewer.config(width=int(w_img), height=int(h1_scaled))
        self.floorplan_viewer.config(width=int(w_img), height=int(h2_scaled))

        self.keyframe_viewer.display_keyframe(self.keyframe_viewer.current_key, master_size=(int(w_img), int(h1_scaled)))
        self.floorplan_viewer.render_floorplan()

    # ---- Correspondence and interaction ----

    def _select_initial_keyframe(self) -> None:
        """Select last annotated or first keyframe on startup."""
        names = list(self.keyframe_listbox.get(0, tk.END))
        corr = self.logic.correspondences
        if corr:
            last_kf = corr[-1]["keyframe"].replace('.png', '')
            idx = names.index(last_kf) if last_kf in names else 0
        else:
            idx = 0
        self.keyframe_listbox.selection_clear(0, tk.END)
        self.keyframe_listbox.selection_set(idx)
        self.keyframe_listbox.see(idx)
        self._on_keyframe_select(None)

    def _on_keyframe_select(self, event) -> None:
        sel = self.keyframe_listbox.curselection()
        if not sel:
            return
        name = self.keyframe_listbox.get(sel[0])
        self.keyframe_viewer.display_keyframe(name)
        self.current_keyframe = name + '.png'
        self.try_draw_scene_overlay()

    def _on_feature_selected(self, correspondence) -> None:
        """Triggered when a keyframe feature is selected/confirmed."""
        self.logic.add_or_update_correspondence(correspondence)
        self._on_keyframe_select(None)
        self.update_correspondence_table()

    def _on_floorplan_confirm(self, correspondence) -> None:
        """Triggered when a floorplan point is confirmed."""
        self.logic.update_last_floorplan(correspondence)
        w, h = self.floorplan_viewer.winfo_width(), self.floorplan_viewer.winfo_height()
        if w < 10 or h < 10:
            self.after(100, self.try_draw_scene_overlay)
            return
        try:
            matrix = self.logic.compute_transform()
            self.logic.save_to_matrix(matrix=matrix)
        except Exception as e:
            print(f"Transform error: {str(e)}")
        self.update_correspondence_table()
        self.try_draw_scene_overlay()

    def _open_floorplan_with_highlight(self) -> None:
        """Open the floorplan selector with the currently highlighted 3D point."""
        correspondence = self.logic.correspondences[-1] if self.logic.correspondences else None
        point3d = self.persistent_highlight_point3d
        matrix = self._last_overlay_data.get("matrix")
        highlight2d = None
        if point3d is not None and matrix is not None:
            highlight2d = project_point3d_to_floor2d(point3d, matrix)
        self.floorplan_viewer.open_floorplan_point_selector(
            correspondence,
            self._on_floorplan_confirm,
            highlight2d=highlight2d
        )

    def _on_corr_select(self, event) -> None:
        """Highlight the selected correspondence in viewers."""
        sel = self.info_table.selection()
        if not sel:
            return
        idx = int(sel[0])
        corr = [
            c for c in self.logic.correspondences
            if c.get("floor2d") is not None and c.get("point3d") is not None
        ][idx]
        keyframe = corr['keyframe']
        keypoint_idx = corr['keypoint_idx']
        self.current_keyframe = keyframe
        self.keyframe_listbox.selection_clear(0, tk.END)
        kf_name = keyframe.replace('.png', '')
        all_names = list(self.keyframe_listbox.get(0, tk.END))
        if kf_name in all_names:
            kf_idx = all_names.index(kf_name)
            self.keyframe_listbox.selection_set(kf_idx)
            self.keyframe_listbox.see(kf_idx)
            self.keyframe_viewer.display_keyframe(kf_name)
            self.keyframe_viewer._highlight_feature_idx = keypoint_idx
        self.persistent_highlight_point3d = corr['point3d']
        self.try_draw_scene_overlay()

    def _on_corr_right_click(self, event) -> None:
        """Show right-click menu for correspondence deletion."""
        rowid = self.info_table.identify_row(event.y)
        if not rowid:
            return
        idx = int(rowid)
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="Delete", command=lambda: self.delete_correspondence(idx))
        menu.tk_popup(event.x_root, event.y_root)

    def delete_correspondence(self, idx: int) -> None:
        """Delete a correspondence and refresh the panels."""
        correspondences = [
            c for c in self.logic.correspondences
            if c.get("floor2d") is not None and c.get("point3d") is not None
        ]
        target = correspondences[idx]
        self.logic.correspondences.remove(target)
        self.logic.save_correspondences()
        self.update_correspondence_table()
        self.try_draw_scene_overlay()

    def update_correspondence_table(self) -> None:
        """Refresh the table showing floor2d <-> projected 3D and distances."""
        self.info_table.delete(*self.info_table.get_children())
        scale = getattr(self.logic, 'scale', 1.0)
        correspondences = [
            c for c in self.logic.correspondences
            if c.get("point3d") is not None
        ]
        if not (correspondences and self.logic.transform_matrix is not None):
            return

        T = self.logic.transform_matrix
        for idx, c in enumerate(correspondences):
            pt2d = c.get('floor2d')
            pt3d_proj = project_point3d_to_floor2d(c['point3d'], T)
            if pt2d is None:
                floor2d_str = "None"
                dist = ""
            else:
                floor2d_str = f"({pt2d[0]:.4f}, {pt2d[1]:.4f})"
                dist = np.linalg.norm(np.array(pt2d) - np.array(pt3d_proj)) * scale
                dist = f"{dist:.4f}"
            proj3d_str = f"({pt3d_proj[0]:.4f}, {pt3d_proj[1]:.4f})"
            self.info_table.insert(
                '', 'end', iid=idx,
                values=(idx, floor2d_str, proj3d_str, dist)
            )

    def _on_save_matrix_clicked(self) -> None:
        """Save the transformation matrix to disk."""
        if self.logic.transform_matrix is not None:
            try:
                self.logic.save_to_matrix(matrix=self.logic.transform_matrix)
                messagebox.showinfo("Success", "Transformation matrix saved.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save matrix:\n{e}")
        else:
            messagebox.showwarning("Not computed", "No transformation matrix computed yet.")

    def _on_feature_highlight(self, point3d, temporary=True) -> None:
        """Highlight a 3D point (temp or persistent)."""
        if temporary:
            self.temporary_highlight_point3d = point3d
        else:
            self.persistent_highlight_point3d = point3d
            self.temporary_highlight_point3d = None
        self.try_draw_scene_overlay()

    def _get_highlight2d_for_selector(self):
        """Return the 2D projected location of the current highlighted 3D point."""
        point3d = self.persistent_highlight_point3d
        T = self._last_overlay_data["matrix"]
        if point3d is None or T is None:
            return None
        return project_point3d_to_floor2d(point3d, T)

    def try_draw_scene_overlay(self) -> None:
        """
        Compute and update the overlay on the floorplan viewer.
        Caches last transformation, redraws only if inputs change.
        """
        current_key = getattr(self, "current_keyframe", None)
        points2D, points3D = self.logic.get_correspondence_points()
        landmarks_3d = self.logic.get_landmarks_3d()
        cam_traj_3d = self.logic.get_cam_trajectory_3d()
        curr_cam_pose = self.logic.get_current_camera_pose(current_key)
        curr_observed_3d = self.logic.get_current_observed_points(current_key)

        cache = self._last_overlay_data
        same_inputs = (
            lists_equal(cache["points2D"], points2D) and
            lists_equal(cache["points3D"], points3D) and
            np.array_equal(np.array(cache["landmarks_3d"]), np.array(landmarks_3d)) and
            np.array_equal(np.array(cache["cam_traj_3d"]), np.array(cam_traj_3d)) and
            pose_dicts_equal(cache["curr_cam_pose"], curr_cam_pose) and
            np.array_equal(np.array(cache["curr_observed_3d"]), np.array(curr_observed_3d))
        )
        if (
            same_inputs
            and cache["matrix"] is not None
            and self.temporary_highlight_point3d is None
            and self.persistent_highlight_point3d is None
        ):
            return  # No changes, skip redraw

        try:
            T = self.logic.compute_transform()
        except ValueError:
            return  # Not enough pairs

        self._last_overlay_data = {
            "points2D": list(points2D),
            "points3D": list(points3D),
            "matrix": np.array(T),
            "landmarks_3d": np.array(landmarks_3d),
            "cam_traj_3d": np.array(cam_traj_3d),
            "curr_cam_pose": curr_cam_pose.copy() if isinstance(curr_cam_pose, dict) else curr_cam_pose,
            "curr_observed_3d": np.array(curr_observed_3d)
        }

        self.floorplan_viewer.draw_scene_overlay(
            landmarks_3d=landmarks_3d,
            cam_traj_3d=cam_traj_3d,
            curr_cam_pose=curr_cam_pose,
            curr_observed_3d=curr_observed_3d,
            T_3d_to_2d=T,
            points2D=points2D,
            points3D=points3D,
            highlight_point3d=self.temporary_highlight_point3d,
            persistent_highlight_point3d=self.persistent_highlight_point3d 
        )
