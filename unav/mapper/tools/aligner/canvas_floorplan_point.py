import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from unav.mapper.tools.aligner.ui_helpers import CanvasImage
from unav.mapper.tools.aligner.aligner_utils import compute_arrow_length
import numpy as np

class FloorplanViewer(tk.Canvas):
    """
    Canvas widget for displaying a floorplan image and overlaying 2D projections of 3D scene elements.
    Supports resizing, overlays for scene geometry, and interactive user selection.
    """
    def __init__(self, parent, floorplan_path: str, max_size: int = 4096, **kwargs):
        """
        Args:
            parent: Parent widget.
            floorplan_path (str): Path to the floorplan image file.
            max_size (int): Maximum allowed display width.
            **kwargs: Additional arguments for tk.Canvas.
        """
        super().__init__(parent, bg='gray', **kwargs)
        self.floorplan_path = floorplan_path
        self._orig_floorplan = Image.open(self.floorplan_path)
        self._tk_floorplan = None
        self.last_overlay_args = None

        # Bind canvas resize and expose events to trigger redraw
        self.bind("<Configure>", self._on_resize)
        self.bind("<Expose>", self._on_resize)

    def _on_resize(self, event):
        """
        Handle canvas resizing: only render when valid window size is present.
        """
        w, h = self.winfo_width(), self.winfo_height()
        if w > 10 and h > 10:
            self.render_floorplan()
        else:
            # Try again shortly if canvas not ready
            self.after(50, lambda: self._on_resize(event))

    def render_floorplan(self) -> None:
        """
        Renders and displays the floorplan image on the canvas,
        scaling it to fit while preserving aspect ratio.
        """
        w_canvas, h_canvas = self.winfo_width(), self.winfo_height()
        MAX_W = 900

        # Limit width to MAX_W or canvas width, whichever is smaller
        target_w = min(w_canvas, MAX_W)
        orig_w, orig_h = self._orig_floorplan.size
        scale = target_w / orig_w if orig_w > 0 else 1.0
        target_h = int(orig_h * scale)

        if target_w < 10 or target_h < 10:
            self.after(50, self.render_floorplan)
            return

        # Resize and display floorplan image
        img = self._orig_floorplan.copy()
        img = img.resize((int(target_w), int(target_h)), Image.Resampling.LANCZOS)
        self._tk_floorplan = ImageTk.PhotoImage(img)

        # Clear canvas, center image
        self.delete("all")
        off_x = (w_canvas - int(target_w)) // 2
        off_y = (h_canvas - int(target_h)) // 2
        self.create_image(off_x, off_y, anchor='nw', image=self._tk_floorplan)
        if self.last_overlay_args is not None:
            self.draw_scene_overlay(**self.last_overlay_args)

    def draw_scene_overlay(
        self,
        landmarks_3d,
        cam_traj_3d,
        curr_cam_pose,
        curr_observed_3d,
        T_3d_to_2d,
        points2D=None,
        points3D=None,
        highlight_point3d=None,
        persistent_highlight_point3d=None,
        display_time_ms=1500,
        style=None
    ):
        """
        Draw overlays for 3D scene geometry, camera trajectory, and user correspondences.

        Args:
            landmarks_3d: Array-like, all 3D landmarks to be projected and drawn.
            cam_traj_3d: Array-like, camera trajectory points in 3D.
            curr_cam_pose: Dict, camera center and rotation matrix.
            curr_observed_3d: Array-like, currently observed 3D points.
            T_3d_to_2d: np.ndarray, 3D-to-2D projection matrix.
            points2D: Optional, list of hand-labeled 2D points.
            points3D: Optional, list of associated 3D points.
            highlight_point3d: Optional, temporary 3D point to highlight.
            persistent_highlight_point3d: Optional, persistent 3D point to highlight.
            display_time_ms: Optional, not used but for compatibility.
            style: Optional dict for overlay style parameters.
        """
        # ---- Style configuration ----
        style = style or {}
        R_LANDMARK = style.get("landmark_radius", 1)
        R_TRAJ = style.get("traj_radius", 1)
        R_OBS = style.get("obs_radius", 2)
        R_CAM = style.get("cam_radius", 8)
        LINE_ARROW = style.get("arrow_linewidth", 6)

        COLOR_LANDMARK = style.get("landmark_color", "gray")
        COLOR_TRAJ = style.get("traj_color", "blue")
        COLOR_OBS = style.get("obs_color", "lime")
        COLOR_CAM = style.get("cam_color", "red")
        COLOR_CAM_EDGE = style.get("cam_edge_color", "red")
        COLOR_ARROW = style.get("arrow_color", "red")
        COLOR_POINT2D = style.get("point2d_color", "#FFA500")   # Orange
        COLOR_POINT3D = style.get("point3d_color", "#00CED1")   # Cyan
        COLOR_PAIR_LINE = style.get("pair_line_color", "#FF1493") # DeepPink

        self.delete('overlay')
        self.last_overlay_args = {
            'landmarks_3d': landmarks_3d,
            'cam_traj_3d': cam_traj_3d,
            'curr_cam_pose': curr_cam_pose,
            'curr_observed_3d': curr_observed_3d,
            'T_3d_to_2d': T_3d_to_2d,
            'style': style
        }

        if self._tk_floorplan is None:
            self.render_floorplan()
            if self._tk_floorplan is None:
                print("Warning: _tk_floorplan is still None after render_floorplan.")
                return

        w_orig, h_orig = self._orig_floorplan.size
        w_disp = self._tk_floorplan.width()
        h_disp = self._tk_floorplan.height()
        w_canvas, h_canvas = self.winfo_width(), self.winfo_height()
        off_x = (w_canvas - w_disp) // 2
        off_y = (h_canvas - h_disp) // 2

        def project_and_scale(pts_3d):
            pts = np.asarray(pts_3d)
            if pts.size == 0:
                return np.zeros((0, 2), dtype=np.float32)
            if pts.shape[1] == 3:
                ones = np.ones((pts.shape[0], 1))
                pts_h = np.hstack([pts, ones])
            else:
                pts_h = pts
            pts_2d = (T_3d_to_2d @ pts_h.T).T
            pts_2d[:, 0] = pts_2d[:, 0] * w_disp / w_orig + off_x
            pts_2d[:, 1] = pts_2d[:, 1] * h_disp / h_orig + off_y
            return pts_2d

        def highlight_on_floor(point3d, color, outline, radius_out=10, radius_in=4, width=4):
            pt2d = project_and_scale([point3d])[0]
            x, y = pt2d
            self.create_oval(
                x-radius_out, y-radius_out, x+radius_out, y+radius_out,
                fill="", outline=outline, width=width, tags="overlay"
            )
            self.create_oval(
                x-radius_in, y-radius_in, x+radius_in, y+radius_in,
                fill=color, outline=outline, width=2, tags="overlay"
            )

        # 1. Draw all landmarks as small gray dots
        pts_2d = project_and_scale(landmarks_3d)
        for x, y in pts_2d:
            self.create_oval(
                x-R_LANDMARK, y-R_LANDMARK, x+R_LANDMARK, y+R_LANDMARK,
                fill=COLOR_LANDMARK, outline="", tags="overlay"
            )

        # 2. Draw camera trajectory (blue dots)
        cam_pts_2d = project_and_scale(cam_traj_3d)
        for x, y in cam_pts_2d:
            self.create_oval(
                x-R_TRAJ, y-R_TRAJ, x+R_TRAJ, y+R_TRAJ,
                fill=COLOR_TRAJ, outline="", tags="overlay"
            )

        # 3. Draw currently observed 3D points (lime)
        obs_2d = project_and_scale(curr_observed_3d)
        for x, y in obs_2d:
            self.create_oval(
                x-R_OBS, y-R_OBS, x+R_OBS, y+R_OBS,
                fill=COLOR_OBS, outline="", tags="overlay"
            )

        # 4. Hand-labeled 2D points and projected 3D points with dashed lines
        if points2D is not None and points3D is not None and len(points2D) == len(points3D):
            pts2d = np.asarray(points2D, dtype=np.float32)
            if pts2d.shape[0] > 0:
                pts2d_disp = np.empty_like(pts2d)
                pts2d_disp[:, 0] = pts2d[:, 0] * w_disp / w_orig + off_x
                pts2d_disp[:, 1] = pts2d[:, 1] * h_disp / h_orig + off_y
                pts3d_proj = project_and_scale(points3D)
                for i in range(len(pts2d_disp)):
                    x2d, y2d = pts2d_disp[i]
                    x3d, y3d = pts3d_proj[i]
                    self.create_oval(
                        x2d-4, y2d-4, x2d+4, y2d+4,
                        fill=COLOR_POINT2D, outline="black", width=1, tags="overlay"
                    )
                    self.create_oval(
                        x3d-4, y3d-4, x3d+4, y3d+4,
                        fill=COLOR_POINT3D, outline="black", width=1, tags="overlay"
                    )
                    self.create_line(
                        x2d, y2d, x3d, y3d,
                        fill=COLOR_PAIR_LINE, width=2, dash=(4,2), tags="overlay"
                    )

        # 5. Draw camera center (red dot)
        cam_center_2d = project_and_scale([curr_cam_pose['center']])[0]
        self.create_oval(
            cam_center_2d[0]-R_CAM, cam_center_2d[1]-R_CAM,
            cam_center_2d[0]+R_CAM, cam_center_2d[1]+R_CAM,
            fill=COLOR_CAM, outline=COLOR_CAM_EDGE, width=2, tags="overlay"
        )

        # 6. Draw camera heading arrow (red)
        if 'R' in curr_cam_pose:
            R = np.asarray(curr_cam_pose['R'])
            t = np.asarray(curr_cam_pose['center'])
            arrow_len = compute_arrow_length(cam_traj_3d)
            forward_vec = (R @ np.array([0, 0, arrow_len])) + t
            arrow_2d = project_and_scale([t, forward_vec])
            self.create_line(
                arrow_2d[0, 0], arrow_2d[0, 1], arrow_2d[1, 0], arrow_2d[1, 1],
                fill=COLOR_ARROW, width=LINE_ARROW, arrow=tk.LAST, tags="overlay"
            )

        # 7. Temporary highlight (yellow/purple)
        if highlight_point3d is not None:
            highlight_on_floor(
                highlight_point3d, color="yellow", outline="purple", radius_out=5, radius_in=3, width=1
            )

        # 8. Persistent highlight (yellow/red)
        if persistent_highlight_point3d is not None:
            highlight_on_floor(
                persistent_highlight_point3d, color="yellow", outline="red", radius_out=5, radius_in=3, width=1
            )

    def open_floorplan_point_selector(self, correspondence, on_confirm_callback, highlight2d=None):
        """
        Opens a popup window to select a 2D point on the floorplan image.
        Args:
            correspondence (dict): The correspondence dict to be updated.
            on_confirm_callback (callable): Called with updated correspondence on confirm.
            highlight2d: Optional, highlight an existing 2D point.
        """
        if not correspondence:
            messagebox.showinfo(
                "No correspondence",
                "Please select a feature point on the keyframe before choosing a location on the floorplan."
            )
            return

        # Step 1: Resize the floorplan for display
        MAX_W, MAX_H = 1024, 768
        img = Image.open(self.floorplan_path)
        img.thumbnail((MAX_W, MAX_H), Image.Resampling.LANCZOS)
        show_w, show_h = img.size

        # Step 2: Create popup window
        floorplan_window = tk.Toplevel(self)
        floorplan_window.title("Select Floorplan Point")
        floorplan_window.geometry(f"{show_w}x{show_h}")
        floorplan_window.minsize(300, 300)
        floorplan_window.rowconfigure(0, weight=1)
        floorplan_window.columnconfigure(0, weight=1)
        floorplan_window.resizable(True, True)

        # Step 3: Add interactive floorplan canvas
        canvas = CanvasFloorplanPoint(
            parent=floorplan_window,
            floorplan_path=self.floorplan_path,
            correspondence=correspondence,
            on_confirm=on_confirm_callback,
            highlight2d=highlight2d
        )
        canvas.grid(row=0, column=0, sticky="nsew")

        # Make window modal
        def set_grab():
            try:
                floorplan_window.grab_set()
            except tk.TclError:
                pass
        floorplan_window.after_idle(set_grab)

class CanvasFloorplanPoint(CanvasImage):
    """
    Interactive canvas for selecting a single 2D point on the floorplan image.
    - Left click: Select a point (only one at a time).
    - Right click: Confirm (choose) or cancel the selected point.
    - If 'floor2d' exists in the correspondence, initializes with that point selected.
    """

    def __init__(self, parent, floorplan_path, correspondence, on_confirm=None, highlight2d=None):
        """
        Args:
            parent: tk.Toplevel or tk.Frame containing this widget.
            floorplan_path (str): Path to the floorplan image.
            correspondence (dict): Dict to update with selected floor2d.
            on_confirm (callable): Callback when confirmed.
            highlight2d: Optional, highlight an initial 2D point.
        """
        super().__init__(parent, floorplan_path)
        self.parent = parent
        self.correspondence = correspondence
        self.on_confirm = on_confirm
        self.highlight2d = highlight2d

        self.radius = 10
        self.selected_point = None
        self.chosen_point = None

        if self.correspondence.get("floor2d") is not None:
            self.selected_point = list(self.correspondence["floor2d"])
            self.chosen_point = list(self.correspondence["floor2d"])

        self._draw_point()

        self.canvas.bind("<Button-1>", self._on_left_click, add="+")
        self.canvas.bind("<Button-3>", self._on_right_click, add="+")

    def _draw_point(self):
        """
        Draw the current selected and/or chosen point on the floorplan.
        """
        self.canvas.delete("floor_pt")
        box = self.canvas.coords(self.container)
        if not box or len(box) < 2:
            return
        x0, y0 = box[0], box[1]
        scale = self.imscale

        if self.selected_point:
            cx = self.selected_point[0] * scale + x0
            cy = self.selected_point[1] * scale + y0
            r = self.radius * scale
            color = "orange" if self.selected_point != self.chosen_point else "blue"
            outline = "white" if self.selected_point == self.chosen_point else "black"
            self.canvas.create_oval(
                cx - r, cy - r, cx + r, cy + r,
                fill=color, outline=outline, width=2, tags="floor_pt"
            )
        if self.highlight2d is not None:
            hx, hy = self.highlight2d
            cx, cy = hx * scale + x0, hy * scale + y0
            r_outer, r_inner = 10, 6
            self.canvas.create_oval(
                cx-r_outer, cy-r_outer, cx+r_outer, cy+r_outer,
                outline='red', width=3, tags="floor_pt"
            )
            self.canvas.create_oval(
                cx-r_inner, cy-r_inner, cx+r_inner, cy+r_inner,
                fill='yellow', outline='red', width=2, tags="floor_pt"
            )

    def _point_hit(self, px, py):
        """
        Determines if the image-space point (px, py) is within the area of the selected point.
        """
        if not self.selected_point:
            return False
        x, y = self.selected_point
        dist2 = (px - x) ** 2 + (py - y) ** 2
        return dist2 < (self.radius * 2) ** 2

    def _get_canvas_to_image(self, event):
        """
        Converts canvas event coordinates to image coordinates.
        """
        box = self.canvas.coords(self.container)
        if not box or len(box) < 2:
            return 0, 0
        x0, y0 = box[0], box[1]
        scale = self.imscale
        cx, cy = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        ix = (cx - x0) / scale
        iy = (cy - y0) / scale
        return ix, iy

    def _on_left_click(self, event):
        """
        Handle left mouse click for selecting/toggling points.
        - If chosen point exists and user clicks it: un-choose and select it.
        - If chosen point exists and click elsewhere: do nothing (canvas pan allowed).
        - If no chosen point:
            - Click on selected point: deselect it.
            - Click elsewhere: select new point.
        """
        ix, iy = self._get_canvas_to_image(event)
        if self.chosen_point is not None:
            if self._point_hit(ix, iy):
                self.selected_point = list(self.chosen_point)
                self.chosen_point = None
                self._draw_point()
        else:
            if self.selected_point is not None and self._point_hit(ix, iy):
                self.selected_point = None
                self._draw_point()
            else:
                self.selected_point = [ix, iy]
                self._draw_point()

    def _on_right_click(self, event):
        """
        Handle right click: confirm the selected point (set as chosen), or cancel it.
        """
        ix, iy = self._get_canvas_to_image(event)
        if self.selected_point and self._point_hit(ix, iy):
            if self.chosen_point and self.chosen_point == self.selected_point:
                self.chosen_point = None
                self._draw_point()
                self.correspondence["floor2d"] = None
            else:
                self.chosen_point = list(self.selected_point)
                self._draw_point()
                self.correspondence["floor2d"] = list(self.chosen_point)
                if self.on_confirm:
                    self.on_confirm(self.correspondence)
                self.parent.destroy()
