import os
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk, ImageDraw
from mapper.tools.aligner.ui_helpers import CanvasImage
from typing import List, Dict, Tuple, Any, Optional

class KeyframeViewer(tk.Canvas):
    """
    Canvas widget for visualizing keyframe images with keypoints.
    Allows for interactive feature annotation and correspondence management.
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
        self.keyframe_dir    = keyframe_dir
        self.kf_data         = kf_data
        self.correspondences = correspondences
        self._initial_w, self._initial_h = initial_size

        # Current frame cache
        self.current_key    = None
        self._orig_image    = None   # PIL.Image
        self._tk_image      = None   # ImageTk.PhotoImage

        # Redraw whenever the canvas is resized or exposed
        self.bind("<Configure>", self._on_resize)
        self.bind("<Expose>",    self._on_resize)

    def _on_resize(self, event):
        """
        Handles canvas resize events by redrawing the current keyframe.
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
        Display the keyframe image, scaling it to fit the available space.
        All keypoints and correspondences are rendered on top.
        Args:
            keyframe_name (str): Filename of the keyframe image (with or without .png).
            master_size (tuple): Optional size override for the display area.
        """
        # Normalize filename and check existence
        name = keyframe_name if keyframe_name.endswith('.png') else keyframe_name + '.png'
        path = os.path.join(self.keyframe_dir, name)
        if not os.path.exists(path):
            return

        # Cache image only when switching frames
        if name != self.current_key or self._orig_image is None:
            self._orig_image = Image.open(path)
            self.current_key = name

        img_full = self._orig_image
        orig_w, orig_h = img_full.size

        # Set maximum display size (limit for X11 rendering performance)
        MAX_W, MAX_H = 800, 450

        if master_size is not None:
            c_w, c_h = master_size
        else:
            c_w = min(self.winfo_width(), MAX_W)
            c_h = min(self.winfo_height(), MAX_H)

        # Skip if the canvas size is too small
        if c_w < 10 or c_h < 10:
            return

        # Compute scaling factor (do not upscale)
        scale = min(c_w / orig_w, c_h / orig_h, 1.0)
        disp_w = max(int(orig_w * scale), 1)
        disp_h = max(int(orig_h * scale), 1)
        if disp_w < 2 or disp_h < 2:
            return

        # Resize image and convert to Tkinter format
        img_disp = img_full.resize((disp_w, disp_h), Image.Resampling.LANCZOS)
        self._tk_image = ImageTk.PhotoImage(img_disp)

        # Clear canvas and center image
        self.delete("all")
        off_x = (c_w - disp_w) // 2
        off_y = (c_h - disp_h) // 2
        self.create_image(off_x, off_y, anchor='nw', image=self._tk_image)

        # Draw all keypoints and highlight confirmed correspondences
        pts = self.kf_data[name]['keypoints']
        confirmed = {
            c['keypoint_idx']
            for c in self.correspondences
            if c['keyframe'] == name
        }
        pts_arr = np.asarray(pts)
        confirmed_indices = np.array(list(confirmed), dtype=int)
        r = 3

        # Draw all keypoints in lime
        for idx, (x0, y0) in enumerate(pts_arr):
            x = off_x + x0 * scale
            y = off_y + y0 * scale
            self.create_oval(x - r, y - r, x + r, y + r, fill='lime', outline='')

        # Draw confirmed keypoints in red, on top
        for x0, y0 in pts_arr[confirmed_indices]:
            x = off_x + x0 * scale
            y = off_y + y0 * scale
            self.create_oval(x - r, y - r, x + r, y + r, fill='red', outline='')

    def open_feature_canvas(self, correspondences, on_confirm_callback, on_highlight_callback):
        """
        Opens a popup window for interactive feature selection on the current keyframe image.
        Only allows opening if the previous correspondence selection has been completed.
        Args:
            correspondences (list): List of current correspondences.
            on_confirm_callback (callable): Callback on feature selection confirmation.
            on_highlight_callback (callable): Callback for highlight actions.
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

        # Create the popup window for feature selection
        top = tk.Toplevel(self.master)
        top.title(f'Annotate {name}')
        top.geometry('1024x768')
        top.transient(self.master)
        top.rowconfigure(0, weight=1)
        top.columnconfigure(0, weight=1)

        # Optimized feature selection canvas
        cif = CanvasImageFeature(top, data=data, on_confirm=on_confirm_callback, on_highlight=on_highlight_callback)
        cif.grid(row=0, column=0, sticky='nswe')

class CanvasImageFeature(CanvasImage):
    """
    Interactive canvas for feature selection on a keyframe image.
    Handles visualization of keypoints, selection by mouse, hover effect,
    confirmation, and highlight of existing correspondences.
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
            parent (tk.Toplevel or tk.Frame): Parent widget.
            data (dict): Dictionary containing image path ('frame'), keypoints ('gp'), and 3D matches ('lm').
            correspondences (list): Existing correspondence dicts.
            on_confirm (callable): Called when selection is confirmed.
            on_highlight (callable): Called on hover/selection highlight.
        """
        super().__init__(parent, data['frame'])
        self.parent = parent
        self.data = data
        self.keypoints = data['gp']        # List of [x, y]
        self.matched_3d = data['lm']       # List of [X, Y, Z] or None
        self.on_confirm = on_confirm
        self.correspondences = correspondences or []
        self.on_highlight = on_highlight

        # Visualization state
        self.radius = 7
        self.hover_idx = None
        self.selected_idx = None
        self.chosen_idx = None
        self.existing_idx_set = self._extract_existing_indices()

        self._draw_points()

        # Bind mouse interaction events
        self.canvas.bind("<Button-1>", self._on_left_click, add="+")
        self.canvas.bind("<Button-3>", self._on_right_click, add="+")
        self.canvas.bind("<Motion>", self._on_motion, add="+")

    def _extract_existing_indices(self):
        """
        Returns the set of keypoint indices for this frame that already have correspondences.
        """
        frame_name = os.path.basename(self.data['frame'])
        idx_set = set()
        for corr in self.correspondences:
            if corr['keyframe'] == frame_name and corr['keypoint_idx'] is not None:
                idx_set.add(corr['keypoint_idx'])
        return idx_set

    def _get_canvas_to_image(self):
        """
        Returns the current (scale, x0, y0) so that: image_x = (canvas_x - x0) / scale
        """
        box = self.canvas.coords(self.container)
        if not box or len(box) < 2:
            return 1.0, 0.0, 0.0  # Fallback
        x0, y0 = box[0], box[1]
        scale = self.imscale
        return scale, x0, y0

    def _draw_points(self):
        """
        Draws all keypoints with color and size determined by selection/highlight state.
        """
        self.canvas.delete("feature_pt")
        scale, x0, y0 = self._get_canvas_to_image()

        # Collect indices by priority
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

        # Draw groups in increasing priority, so high-priority covers low-priority
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
        Update which keypoint is being hovered over for mouse-over highlight effect.
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
        Selects a keypoint as the current candidate (not yet confirmed).
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
        Confirms or cancels a keypoint selection.
        If confirming, emits the correspondence and closes the window.
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
                    # Confirm selection
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
