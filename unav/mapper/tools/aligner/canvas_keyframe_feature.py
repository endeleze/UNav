import os
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from unav.mapper.tools.aligner.ui_helpers import CanvasImage
from typing import List, Dict, Tuple, Any, Optional

class KeyframeViewer(tk.Canvas):
    """
    Canvas widget for visualizing keyframe images with annotated keypoints.
    Allows interactive feature selection and correspondence management.
    """
    def __init__(
        self,
        parent,
        keyframe_dir: str,
        kf_data: Dict[str, Any],
        correspondences: List[Dict[str, Any]],
        initial_size: Tuple[int, int] = (800, 240),
        **kwargs
    ):
        """
        Args:
            parent: Parent widget.
            keyframe_dir (str): Directory containing keyframe images.
            kf_data (dict): Metadata for each keyframe, including keypoints.
            correspondences (list): List of correspondence dicts.
            initial_size (tuple): Initial display size.
            **kwargs: Additional arguments for tk.Canvas.
        """
        super().__init__(parent, bg='gray', **kwargs)
        self.keyframe_dir = keyframe_dir
        self.kf_data = kf_data
        self.correspondences = correspondences
        self._initial_w, self._initial_h = initial_size

        # State for current frame and image
        self.current_key = None
        self._orig_image = None  # PIL.Image
        self._tk_image = None    # ImageTk.PhotoImage

        # Redraw when resized or exposed
        self.bind("<Configure>", self._on_resize)
        self.bind("<Expose>", self._on_resize)

    def _on_resize(self, event):
        """
        Redraws the current keyframe on canvas resize.
        """
        if self.current_key:
            self.display_keyframe(
                self.current_key,
                master_size=(event.width, event.height)
            )

    def display_keyframe(
        self,
        keyframe_name: str,
        master_size: Optional[Tuple[int, int]] = None
    ) -> None:
        """
        Display the keyframe image, scaling to fit, with all keypoints/correspondences drawn.
        Args:
            keyframe_name (str): Filename of the keyframe image (with or without .png).
            master_size (tuple): Optional display area override.
        """
        # Normalize filename
        name = keyframe_name if keyframe_name.endswith('.png') else keyframe_name + '.png'
        path = os.path.join(self.keyframe_dir, name)
        if not os.path.exists(path):
            return

        # Only reload image if switching frames
        if name != self.current_key or self._orig_image is None:
            self._orig_image = Image.open(path)
            self.current_key = name

        img_full = self._orig_image
        orig_w, orig_h = img_full.size

        MAX_W, MAX_H = 800, 450
        c_w, c_h = master_size if master_size else (
            min(self.winfo_width(), MAX_W),
            min(self.winfo_height(), MAX_H)
        )

        if c_w < 10 or c_h < 10:
            return

        scale = min(c_w / orig_w, c_h / orig_h, 1.0)
        disp_w = max(int(orig_w * scale), 1)
        disp_h = max(int(orig_h * scale), 1)
        if disp_w < 2 or disp_h < 2:
            return

        # Prepare resized image for Tkinter
        img_disp = img_full.resize((disp_w, disp_h), Image.Resampling.LANCZOS)
        self._tk_image = ImageTk.PhotoImage(img_disp)

        self.delete("all")
        off_x = (c_w - disp_w) // 2
        off_y = (c_h - disp_h) // 2
        self.create_image(off_x, off_y, anchor='nw', image=self._tk_image)

        # Draw all keypoints
        pts = self.kf_data[name]['keypoints']
        confirmed = {
            c['keypoint_idx']
            for c in self.correspondences
            if c['keyframe'] == name
        }
        pts_arr = np.asarray(pts)
        confirmed_indices = np.array(list(confirmed), dtype=int) if confirmed else np.array([], dtype=int)
        r = 3

        # Draw all keypoints as lime
        for idx, (x0, y0) in enumerate(pts_arr):
            x = off_x + x0 * scale
            y = off_y + y0 * scale
            self.create_oval(x - r, y - r, x + r, y + r, fill='lime', outline='')

        # Draw confirmed correspondences as red (drawn last for visibility)
        if len(confirmed_indices) > 0:
            for x0, y0 in pts_arr[confirmed_indices]:
                x = off_x + x0 * scale
                y = off_y + y0 * scale
                self.create_oval(x - r, y - r, x + r, y + r, fill='red', outline='')

    def open_feature_canvas(self, correspondences, on_confirm_callback, on_highlight_callback):
        """
        Opens a popup window for feature selection on the current keyframe.
        Only allowed if the last correspondence is finished (floor2d selected).
        Args:
            correspondences (list): Current list of correspondences.
            on_confirm_callback (callable): Called after feature selection.
            on_highlight_callback (callable): Called to trigger highlight.
        """
        if correspondences and correspondences[-1].get("floor2d") is None:
            messagebox.showerror(
                "Incomplete Correspondence",
                "Please finish selecting the corresponding point on the floorplan for the last feature before annotating a new keyframe feature."
            )
            return
        if self.current_key is None:
            return

        name = self.current_key
        path = os.path.join(self.keyframe_dir, name)
        if not os.path.exists(path):
            return

        keypoints = self.kf_data[name]['keypoints']
        matched_3d = self.kf_data[name]['matched_3d']
        data = {
            'frame': path,
            'gp': keypoints,
            'lm': matched_3d
        }

        top = tk.Toplevel(self.master)
        top.title(f'Annotate {name}')
        top.geometry('1024x768')
        top.transient(self.master)
        top.rowconfigure(0, weight=1)
        top.columnconfigure(0, weight=1)

        cif = CanvasImageFeature(top, data=data, correspondences=correspondences,
                                on_confirm=on_confirm_callback,
                                on_highlight=on_highlight_callback)
        cif.grid(row=0, column=0, sticky='nswe')

class CanvasImageFeature(CanvasImage):
    """
    Interactive canvas for feature selection on a keyframe image.
    Handles visualization, selection by mouse, highlight and confirmation.
    """
    def __init__(
        self,
        parent,
        data,
        correspondences=None,
        on_confirm=None,
        on_highlight=None
    ):
        """
        Args:
            parent: tk.Toplevel or tk.Frame.
            data: Dict with keys 'frame', 'gp' (keypoints), and 'lm' (matched 3D).
            correspondences: Existing correspondences (for visual feedback).
            on_confirm: Callback on confirmation.
            on_highlight: Callback for highlight (hover/selection).
        """
        super().__init__(parent, data['frame'])
        self.parent = parent
        self.data = data
        self.keypoints = data['gp']
        self.matched_3d = data['lm']
        self.on_confirm = on_confirm
        self.correspondences = correspondences or []
        self.on_highlight = on_highlight

        self.radius = 7
        self.hover_idx = None
        self.selected_idx = None
        self.chosen_idx = None
        self.existing_idx_set = self._extract_existing_indices()

        self._draw_points()

        self.canvas.bind("<Button-1>", self._on_left_click, add="+")
        self.canvas.bind("<Button-3>", self._on_right_click, add="+")
        self.canvas.bind("<Motion>", self._on_motion, add="+")

    def _extract_existing_indices(self):
        """
        Return a set of keypoint indices for this frame that already have correspondences.
        """
        frame_name = os.path.basename(self.data['frame'])
        idx_set = set()
        for corr in self.correspondences:
            if corr['keyframe'] == frame_name and corr['keypoint_idx'] is not None:
                idx_set.add(corr['keypoint_idx'])
        return idx_set

    def _get_canvas_to_image(self):
        """
        Returns the (scale, x0, y0) mapping canvas to image coordinates.
        """
        box = self.canvas.coords(self.container)
        if not box or len(box) < 2:
            return 1.0, 0.0, 0.0  # Fallback
        x0, y0 = box[0], box[1]
        scale = self.imscale
        return scale, x0, y0

    def _draw_points(self):
        """
        Draw all keypoints, using color and size to indicate their selection/highlight state.
        """
        self.canvas.delete("feature_pt")
        scale, x0, y0 = self._get_canvas_to_image()

        group_indices = {
            'chosen': [],
            'selected': [],
            'hover': [],
            'existing': [],
            'normal': []
        }
        n = len(self.keypoints)
        for i in range(n):
            if i == self.chosen_idx:
                group_indices['chosen'].append(i)
            elif i == self.selected_idx:
                group_indices['selected'].append(i)
            elif i == self.hover_idx:
                group_indices['hover'].append(i)
            elif i in self.existing_idx_set:
                group_indices['existing'].append(i)
            else:
                group_indices['normal'].append(i)

        for group, color, radius_scale in [
            ('normal',   "lime",   1.0),
            ('existing', "red",    1.2),
            ('hover',    "yellow", 1.2),
            ('selected', "orange", 1.4),
            ('chosen',   "blue",   1.6)
        ]:
            for i in group_indices[group]:
                x, y = self.keypoints[i]
                r = self.radius * radius_scale * scale
                cx = x * scale + x0
                cy = y * scale + y0
                self.canvas.create_oval(
                    cx - r, cy - r, cx + r, cy + r,
                    fill=color, outline="", tags="feature_pt"
                )

    def _on_motion(self, event):
        """
        Update the hovered keypoint for mouse-over highlight.
        """
        cx, cy = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        scale, x0, y0 = self._get_canvas_to_image()
        ix = (cx - x0) / scale
        iy = (cy - y0) / scale
        idx = None
        for i, (kx, ky) in enumerate(self.keypoints):
            if (ix - kx) ** 2 + (iy - ky) ** 2 < (self.radius * 2) ** 2:
                idx = i
                break
        if idx != self.hover_idx:
            self.hover_idx = idx
            self._draw_points()
            if self.on_highlight:
                pt3d = self.matched_3d[idx] if idx is not None else None
                self.on_highlight(pt3d, temporary=True)

    def _on_left_click(self, event):
        """
        Select a keypoint as the current candidate (not yet confirmed).
        """
        cx, cy = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        scale, x0, y0 = self._get_canvas_to_image()
        ix = (cx - x0) / scale
        iy = (cy - y0) / scale
        for i, (kx, ky) in enumerate(self.keypoints):
            if (ix - kx) ** 2 + (iy - ky) ** 2 < (self.radius * 2) ** 2:
                self.selected_idx = i
                self.chosen_idx = None
                self._draw_points()
                if self.on_highlight:
                    self.on_highlight(self.matched_3d[i], temporary=True)
                return

    def _on_right_click(self, event):
        """
        Confirm or cancel keypoint selection.
        If confirming, emits the correspondence and closes the popup.
        """
        cx, cy = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        scale, x0, y0 = self._get_canvas_to_image()
        ix = (cx - x0) / scale
        iy = (cy - y0) / scale
        for i, (kx, ky) in enumerate(self.keypoints):
            if (ix - kx) ** 2 + (iy - ky) ** 2 < (self.radius * 2) ** 2:
                if self.chosen_idx == i:
                    # Cancel selection
                    self.chosen_idx = None
                    self.selected_idx = None
                    self._draw_points()
                else:
                    # Confirm selection and emit correspondence
                    self.chosen_idx = i
                    self.selected_idx = None
                    self._draw_points()
                    if self.on_highlight:
                        self.on_highlight(self.matched_3d[self.chosen_idx], temporary=False)
                    if self.on_confirm:
                        frame_name = os.path.basename(self.data['frame'])
                        corr = {
                            "keyframe": frame_name,
                            "keypoint_idx": i,
                            "keypoint": [float(kx), float(ky)],
                            "point3d": self.matched_3d[i],
                            "floor2d": None
                        }
                        self.on_confirm(corr)
                    self.parent.destroy()
                return
